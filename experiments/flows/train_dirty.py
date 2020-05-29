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
from sklearn.utils import shuffle as util_shuffle
from sklearn.model_selection import KFold

# matplotlib.use('agg')

from deconv.utils.make_2d_toy_data import data_gen
from deconv.utils.make_2d_toy_noise_covar import covar_gen
from deconv.utils.compute_2d_log_likelihood import compute_data_ll
from deconv.utils.misc import get_logger
from deconv.flow.svi_no_mdn import SVIFlowToy, SVIFlowToyNoise
from deconv.gmm.data import DeconvDataset
from sklearn.datasets import load_boston

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='boston',
                    choices=['boston', 'white_wine', 'red_wine', 'ionosphere'])
parser.add_argument('--covar', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--flow_steps_prior', type=int, default=4)
parser.add_argument('--flow_steps_posterior', type=int, default=5)
parser.add_argument('--n_epochs', type=int, default=int(1e4))
parser.add_argument('--objective', type=str, default='iwae',
                    choices=['elbo', 'iwae', 'iwae_sumo'])
parser.add_argument('--K', type=int, default=50,
                    help='# of samples for objective')
parser.add_argument('--maf_features', type=int, default=128)
parser.add_argument('--maf_hidden_blocks', type=int, default=1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    torch.cuda.set_device(args.gpu)

else:
    torch.set_default_tensor_type('torch.DoubleTensor')


args.dir = 'results_flow/' + \
    str(args.objective) + '/' + str(args.covar) + '/'

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

if args.name is None:
    name = 'eval_' + str(args.data) + '_lr_' + str(args.lr) + '_fspr_' + str(args.flow_steps_prior) + '_fspo_' + \
        str(args.flow_steps_posterior) + '_mf_' + \
        str(args.maf_features) + '_mhb_' + \
        str(args.maf_hidden_blocks) + '_seed_' + str(args.seed)


if not os.path.exists(args.dir + 'logs/'):
    os.makedirs(args.dir + 'logs/')

logger = get_logger(logpath=(args.dir + 'logs/' + name +
                             '.log'), filepath=os.path.abspath(__file__))
logger.info(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def compute_eval_loss(model, eval_loader, device, n_points):
    loss = 0
    for _, data in enumerate(eval_loader):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)

        loss += -model.score(data).sum()

    return loss / n_points


def main():
    if args.data == 'boston':
        data = np.load('data_small/boston_no_discrete.npy')
    elif args.data == 'white_wine':
        data = np.load('data_small/white_no_discrete_no_corr_0.98.npy')
    elif args.data == 'red_wine':
        data = np.load('data_small/red_no_discrete_no_corr_0.98.npy')
    n_features = data.shape[1]
    n_train = int(data.shape[0] * 0.9)
    train_data_clean = util_shuffle(data[:n_train])
    test_data = data[:n_train]
    kf = KFold(n_splits=10)

    covar = np.diag(args.covar * np.ones((n_features,)))

    train_data = train_data_clean + \
        np.random.multivariate_normal(mean=np.zeros(
            (n_features,)), cov=covar, size=n_train)

    # train_covars = np.repeat(
    #     covar[np.newaxis, :, :], n_train, axis=0)

    # train_dataset = DeconvDataset(train_data, train_covars)
    for i, (train_index, eval_index) in enumerate(kf.split(train_data)):
        X_train, X_eval = train_data[train_index], train_data[eval_index]
        train_covars = np.repeat(
            covar[np.newaxis, :, :], X_train.shape[0], axis=0)
        eval_covars = np.repeat(
            covar[np.newaxis, :, :], X_eval.shape[0], axis=0)

        train_dataset = DeconvDataset(X_train, train_covars)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)

        eval_dataset = DeconvDataset(X_eval, eval_covars)
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False)

        model = SVIFlowToy(dimensions=n_features,
                           objective=args.objective,
                           posterior_context_size=n_features,
                           batch_size=args.batch_size,
                           device=device,
                           maf_steps_prior=args.flow_steps_prior,
                           maf_steps_posterior=args.flow_steps_posterior,
                           maf_features=args.maf_features,
                           maf_hidden_blocks=args.maf_hidden_blocks,
                           K=args.K)

        message = 'Total number of parameters: %s' % (
            sum(p.numel() for p in model.parameters()))
        logger.info(message)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        # training
        scheduler = [30]
        epoch = 0
        best_model = copy.deepcopy(model.state_dict())

        best_eval_loss = compute_eval_loss(
            model, eval_loader, device, len(eval_index))
        n_epochs_not_improved = 0

        model.train()
        while n_epochs_not_improved < scheduler[-1] and epoch < args.n_epochs:
            for batch_idx, data in enumerate(train_loader):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)

                # loss = -model.score(data).mean()
                loss = -model.model._prior.log_prob(data[0]).mean()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            model.eval()
            test_loss_clean = - \
                model.model._prior.log_prob(
                    torch.from_numpy(test_data).to(device)).mean()
            message = 'Test loss (clean) = %.5f' % test_loss_clean
            logger.info(message)
            eval_loss = compute_eval_loss(
                model, eval_loader, device, len(eval_index))

            if eval_loss < best_eval_loss:
                best_model = copy.deepcopy(model.state_dict())
                best_eval_loss = eval_loss
                n_epochs_not_improved = 0

            else:
                n_epochs_not_improved += 1

            model.train()
            epoch += 1
        break

        model = model.load_state_dict(best_model)
        test_loss_clean = - \
            model.model._prior.log_prob(
                torch.from_numpy(test_data).to(device)).mean()
        message = 'Final test loss (clean) = %.5f' % test_loss_clean
        logger.info(message)


if __name__ == '__main__':
    main()
