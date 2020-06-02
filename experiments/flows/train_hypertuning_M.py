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
from itertools import product

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
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=int(1e4))
parser.add_argument('--objective', type=str, default='iwae',
                    choices=['elbo', 'iwae', 'iwae_sumo'])
parser.add_argument('--K', type=int, default=50,
                    help='# of samples for objective')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    torch.cuda.set_device(args.gpu)

else:
    torch.set_default_tensor_type('torch.DoubleTensor')

args.dir = 'hypertuning/' + '/' + \
    str(args.objective) + '/' + str(args.covar) + '/'

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

name = args.data + '_K_' + str(args.K)

if not os.path.exists(args.dir + 'logs/'):
    os.makedirs(args.dir + 'logs/')

logger = get_logger(logpath=(args.dir + 'logs/' + name +
                             '.log'), filepath=os.path.abspath(__file__))

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
    elif args.data == 'ionosphere':
        data = np.load('data_small/ionosphere_no_discrete_no_corr_0.98.npy')

    n_features = data.shape[1]
    n_train = int(data.shape[0] * 0.9)
    train_data_clean = data[:n_train]

    covar = np.diag(args.covar * np.ones((n_features,)))

    train_data = train_data_clean + \
        np.random.multivariate_normal(mean=np.zeros(
            (n_features,)), cov=covar, size=n_train)

    kf = KFold(n_splits=5)

    # 54 combinations
    lr_list = [1e-3, 5e-4, 1e-4]
    flow_steps_prior_list = [3, 4, 5]
    flow_steps_posterior_list = [4, 5]
    maf_features_list = [128]
    maf_hidden_blocks_list = [1, 2]
    M_list = [5, 10, 20]

    n_combs = 0
    for lr, fspr, fspo, maf_f, maf_h in product(lr_list, flow_steps_posterior_list, flow_steps_posterior_list, maf_features_list, maf_hidden_blocks_list):
        print(n_combs, (lr, fspr, fspo, maf_f, maf_h))
        n_combs += 1

    best_eval = np.zeros((n_combs, 5))

    counter = 0
    for lr, fspr, fspo, maf_f, maf_h, M in product(lr_list, flow_steps_posterior_list, flow_steps_posterior_list, maf_features_list, maf_hidden_blocks_list, M_list):
        logger.info((lr, fspr, fspo, maf_f, maf_h, M))

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
                               maf_steps_prior=fspr,
                               maf_steps_posterior=fspo,
                               maf_features=maf_f,
                               maf_hidden_blocks=maf_h,
                               K=args.K)

            message = 'Total number of parameters: %s' % (
                sum(p.numel() for p in model.parameters()))
            logger.info(message)

            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

            # training
            scheduler = [30]  # stop after 30 epochs of no improvement
            epoch = 0

            model.eval()
            with torch.no_grad():
                best_eval_loss = compute_eval_loss(
                    model, eval_loader, device, X_eval.shape[0])

                best_model = copy.deepcopy(model.state_dict())

            n_epochs_not_improved = 0

            model.train()
            for prior_params in model.model._prior.parameters():
                prior_params.requires_grad = False

            while n_epochs_not_improved < scheduler[-1] and epoch < args.n_epochs:
                for batch_idx, data in enumerate(train_loader):
                    data[0] = data[0].to(device)
                    data[1] = data[1].to(device)

                    ap = copy.deepcopy(
                        model.model._approximate_posterior.state_dict())
                    for i in range(M):

                        loss = -model.score(data).mean()
                        message = 'Loss intermediate %s: %f' % (i, loss)
                        logger.info(message)
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()

                    for prior_params in model.model._prior.parameters():
                        prior_params.requires_grad = True

                    for post_params in model.model._approximate_posterior.parameters():
                        post_params.requires_grad = False

                    loss = -model.score(data).mean()
                    message = 'Loss prior %f' % loss
                    logger.info(message)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # replace optimized approximate posterior with old approximate posterior
                    model.model._approximate_posterior.load_state_dict(ap)

                    for prior_params in model.model._prior.parameters():
                        prior_params.requires_grad = False

                    for post_params in model.model._approximate_posterior.parameters():
                        post_params.requires_grad = True

                    loss = -model.score(data).mean()
                    message = 'Loss posterior %f' % loss
                    logger.info(message)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    eval_loss = compute_eval_loss(
                        model, eval_loader, device, X_eval.shape[0])

                    if eval_loss < best_eval_loss:
                        best_model = copy.deepcopy(model.state_dict())
                        best_eval_loss = eval_loss
                        n_epochs_not_improved = 0

                    else:
                        n_epochs_not_improved += 1

                    message = 'Epoch %s:' % (
                        epoch + 1), 'train loss = %.5f' % loss, 'eval loss = %.5f' % eval_loss
                    logger.info(message)

                model.train()
                epoch += 1

            best_eval[counter, i] = best_eval_loss
            np.save(args.data + '_hypertuning_results_tmp', best_eval)

        counter += 1

    np.save(args.data + '_hypertuning_results', best_eval)


if __name__ == '__main__':
    main()
