import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import copy
import os
import corner
from torch.utils.data import DataLoader
import argparse

matplotlib.use('agg')

from deconv.utils.make_2d_toy_data import data_gen
from deconv.utils.make_2d_toy_noise_covar import covar_gen
from deconv.utils.compute_2d_log_likelihood import compute_data_ll
from deconv.utils.misc import get_logger
from deconv.flow.svi_no_mdn import SVIFlowToy, SVIFlowToyNoise
from deconv.gmm.data import DeconvDataset

parser = argparse.ArgumentParser()
parser.add_argument('--infer', type=str, default='true_data',
                    choices=['noise', 'true_data'])
parser.add_argument('--data', type=str, default='mixture_final')
parser.add_argument('--covar', type=str, default='gmm')
parser.add_argument('--n_train_points', type=int, default=int(1e5))
parser.add_argument('--n_test_points', type=int, default=int(1e3))
parser.add_argument('--n_eval_points', type=int, default=int(1e3))
parser.add_argument('--n_kl_points', type=int, default=int(1e4))
parser.add_argument('--eval_based_scheduler', type=str, default='10,20,30')
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_batch_size', type=int, default=512)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--flow_steps_prior', type=int, default=5)
parser.add_argument('--flow_steps_posterior', type=int, default=5)
parser.add_argument('--posterior_context_size', type=int,
                    default=2)  # this is just dim when we use w
parser.add_argument('--n_epochs', type=int, default=int(1e2))
parser.add_argument('--objective', type=str, default='elbo',
                    choices=['elbo', 'iwae', 'iwae_sumo'])
parser.add_argument('--K', type=int, default=1,
                    help='# of samples for objective')
parser.add_argument('--viz_freq', type=int, default=10)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--maf_features', type=int, default=128)
parser.add_argument('--maf_hidden_blocks', type=int, default=2)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    torch.cuda.set_device(args.gpu)

else:
    torch.set_default_tensor_type('torch.DoubleTensor')

if args.dir is None:
    args.dir = 'toy_james/' + str(args.infer) + '/' + str(
        args.objective) + '/' + str(args.data) + '/' + str(args.covar) + '/'

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

if args.name is None:
    name = 'n_train_points_' + str(args.n_train_points) + \
        '_n_eval_points_' + str(args.n_eval_points) + \
        '_K_' + str(args.K) + \
        '_batch_size' + str(args.batch_size) + \
        '_fs_prior_' + str(args.flow_steps_prior) + \
        '_fs_posterior_' + str(args.flow_steps_posterior) + \
        '_seed_' + str(args.seed)


# if os.path.isfile(args.dir + 'logs/' + name + '.log'):
#   raise ValueError('This file already exists.')

if not os.path.exists(args.dir + 'logs/'):
    os.makedirs(args.dir + 'logs/')

if not os.path.exists(args.dir + 'out/'):
    os.makedirs(args.dir + 'out/')

if not os.path.exists(args.dir + 'models/'):
    os.makedirs(args.dir + 'models/')

logger = get_logger(logpath=(args.dir + 'logs/' + name +
                             '.log'), filepath=os.path.abspath(__file__))
logger.info(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def lr_scheduler(n_epochs_not_improved, optimzer, scheduler, logger):
    lr = args.lr

    for i in range(len(scheduler) - 1):
        if n_epochs_not_improved > scheduler[i]:
            lr *= 0.1

    for param_group in optimzer.param_groups:
        param_group['lr'] = lr

    message = 'New learning rate: %f' % lr
    logger.info(message)


def compute_eval_loss(model, eval_loader, device, n_points):
    loss = 0
    for _, data in enumerate(eval_loader):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)

        loss += -model.score(data).sum()

    return loss / n_points


def main():
    train_covar = covar_gen(args.covar, args.n_train_points).astype(np.float64)
    train_data_clean = data_gen(args.data, args.n_train_points)[
        0].astype(np.float64)

    corner.hist2d(train_data_clean[:, 0], train_data_clean[:, 1])
    plt.show()

    train_data = np.zeros_like(train_data_clean)
    for i in range(args.n_train_points):
        train_data[i] = train_data_clean[i] + \
            np.random.multivariate_normal(
                mean=np.zeros((2,)), cov=train_covar[i])

    corner.hist2d(train_data[:, 0], train_data[:, 1])
    plt.show()

    train_covar = torch.from_numpy(train_covar)
    train_data = torch.from_numpy(train_data.astype(np.float64))

    train_dataset = DeconvDataset(train_data, train_covar)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    test_data_clean = torch.from_numpy(
        data_gen(args.data, args.n_test_points)[0].astype(np.float64))

    eval_covar = covar_gen(args.covar, args.n_eval_points).astype(np.float64)
    eval_data_clean = data_gen(args.data, args.n_eval_points)[
        0].astype(np.float64)

    eval_data = np.zeros_like(eval_data_clean)
    for i in range(args.n_eval_points):
        eval_data[i] = eval_data_clean[i] + \
            np.random.multivariate_normal(
                mean=np.zeros((2,)), cov=eval_covar[i])

    eval_covar = torch.from_numpy(eval_covar)
    eval_data = torch.from_numpy(eval_data.astype(np.float64))

    eval_dataset = DeconvDataset(eval_data, eval_covar)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.test_batch_size, shuffle=False)

    if args.infer == 'true_data':
        model = SVIFlowToy(dimensions=2,
                           objective=args.objective,
                           posterior_context_size=args.posterior_context_size,
                           batch_size=args.batch_size,
                           device=device,
                           maf_steps_prior=args.flow_steps_prior,
                           maf_steps_posterior=args.flow_steps_posterior,
                           maf_features=args.maf_features,
                           maf_hidden_blocks=args.maf_hidden_blocks,
                           K=args.K)

    else:
        model = SVIFlowToyNoise(dimensions=2,
                                objective=args.objective,
                                flow_steps_prior=args.flow_steps_prior,
                                flow_steps_posterior=args.flow_steps_posterior,
                                n_posterior_flows=args.n_posterior_flows,
                                posterior_mdn=list(
                                    map(int, args.posterior_mdn.split(','))),
                                warmup_posterior_flow_diversity=args.warmup_posterior_flow_diversity,
                                warmup_kl=args.warmup_kl,
                                kl_init=args.kl_init,
                                posterior_context_size=args.posterior_context_size,
                                batch_size=args.batch_size,
                                device=device,
                                maf_features=args.maf_features,
                                maf_hidden_blocks=args.maf_hidden_blocks,
                                iwae_points=args.iwae_points)

    message = 'Total number of parameters: %s' % (
        sum(p.numel() for p in model.parameters()))
    logger.info(message)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # training
    scheduler = list(map(int, args.eval_based_scheduler.split(',')))
    epoch = 0

    model.eval()
    with torch.no_grad():
        best_eval_loss = compute_eval_loss(
            model, eval_loader, device, args.n_eval_points)

        best_model = copy.deepcopy(model.state_dict())

    n_epochs_not_improved = 0

    model.train()
    while n_epochs_not_improved < scheduler[-1] and epoch < args.n_epochs:
        for batch_idx, data in enumerate(train_loader):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)

            loss = -model.score(data).mean()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_loss = compute_eval_loss(
                model, eval_loader, device, args.n_eval_points)

            if eval_loss < best_eval_loss:
                best_model = copy.deepcopy(model.state_dict())
                best_eval_loss = eval_loss
                n_epochs_not_improved = 0

            else:
                n_epochs_not_improved += 1

            lr_scheduler(n_epochs_not_improved, optimizer, scheduler, logger)

            if (epoch + 1) % args.test_freq == 0:
                if args.infer == 'true_data':
                    test_loss_clean = - \
                        model.model._prior.log_prob(
                            test_data_clean.to(device)).mean()

                else:
                    test_loss_clean = - \
                        model.model._likelihood.log_prob(
                            test_data_clean.to(device)).mean()

                message = 'Epoch %s:' % (
                    epoch + 1), 'train loss = %.5f' % loss, 'eval loss = %.5f' % eval_loss, 'test loss (clean) = %.5f' % test_loss_clean
                logger.info(message)

            else:
                message = 'Epoch %s:' % (
                    epoch + 1), 'train loss = %.5f' % loss, 'eval loss = %.5f' % eval_loss
                logger.info(message)

            if (epoch + 1) % args.viz_freq == 0:
                if args.infer == 'true_data':
                    samples = model.model._prior.sample(
                        1000).detach().cpu().numpy()

                else:
                    samples = model.model._likelihood.sample(
                        1000).detach().cpu().numpy()

                corner.hist2d(samples[:, 0], samples[:, 1])
                fig_filename = args.dir + 'out/' + name + \
                    '_corner_fig_' + str(epoch + 1) + '.png'
                plt.savefig(fig_filename)
                plt.close()

                plt.scatter(samples[:, 0], samples[:, 1])
                fig_filename = args.dir + 'out/' + name + \
                    '_scatter_fig_' + str(epoch + 1) + '.png'
                plt.savefig(fig_filename)
                plt.close()

        model.train()
        epoch += 1

    model.load_state_dict(best_model)
    model.eval()

    if args.infer == 'true_data':
        test_loss_clean = - \
            model.model._prior.log_prob(test_data_clean.to(device)).mean()

    else:
        test_loss_clean = - \
            model.model._likelihood.log_prob(test_data_clean.to(device)).mean()

    message = 'Final test loss (clean) = %.5f' % test_loss_clean
    logger.info(message)

    torch.save(model.state_dict(), args.dir + 'models/' + name + '.model')
    logger.info('Training has finished.')

    if args.data.split('_')[0] == 'mixture' or args.data.split('_')[0] == 'gaussian':
        kl_points = data_gen(args.data, args.n_kl_points)[0].astype(np.float64)

        model_log_prob = model.model._prior.log_prob(
            torch.from_numpy(kl_points.astype(np.float64)).to(device)).mean()
        data_log_prob = compute_data_ll(args.data, kl_points).mean()

        approximate_KL = data_log_prob - model_log_prob
        message = 'KL div %.5f:' % approximate_KL
        logger.info(message)


if __name__ == '__main__':
    main()
