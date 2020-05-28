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

# matplotlib.use('agg')

from deconv.utils.make_2d_toy_data import data_gen
from deconv.utils.make_2d_toy_noise_covar import covar_gen
from deconv.utils.compute_2d_log_likelihood import compute_data_ll
from deconv.utils.misc import get_logger
from deconv.flow.svi_no_mdn import SVIFlowToy, SVIFlowToyNoise
from deconv.gmm.data import DeconvDataset
from sklearn.datasets import load_boston

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='boston')
parser.add_argument('--covar', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--flow_steps_prior', type=int, default=5)
parser.add_argument('--flow_steps_posterior', type=int, default=5)
parser.add_argument('--n_epochs', type=int, default=int(1e4))
parser.add_argument('--objective', type=str, default='iwae',
                    choices=['elbo', 'iwae', 'iwae_sumo'])
parser.add_argument('--K', type=int, default=50,
                    help='# of samples for objective')
parser.add_argument('--maf_features', type=int, default=128)
parser.add_argument('--maf_hidden_blocks', type=int, default=2)
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
    name = str(args.data) + '_lr_' + str(args.lr) + '_fspr_' + str(args.flow_steps_prior) + '_fspo_' + \
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


def main():
    data = np.load('data_small/boston_no_discrete.npy')
    n_features = data.shape[1]
    n_train = int(data.shape[0] * 0.9)
    train_data_clean = data[:n_train]

    covar = np.diag(args.covar * np.ones((11,)))

    train_data = train_data_clean + \
        np.random.multivariate_normal(mean=np.zeros(
            (11,)), cov=covar, size=n_train)

    train_covars = np.repeat(
        covar[np.newaxis, :, :], n_train, axis=0)

    train_dataset = DeconvDataset(train_data, train_covars)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

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

    # # training
    # epoch = 0
    # while epoch < args.n_epochs:
    #     train_loss = 0
    #     for batch_idx, data in enumerate(train_loader):
    #         data[0] = data[0].to(device)
    #         data[1] = data[1].to(device)

    #         log_prob = model.score(data)
    #         loss = -log_prob.mean()
    #         train_loss += -torch.sum(log_prob).item()
    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         optimizer.step()

    #     train_loss /= n_train
    #     message = 'Train loss %.5f' % train_loss
    #     logger.info(message)

    #     if train_loss < 9.02486:  # boston housing
    #         break

    test_loss_clean = - \
        model.model._prior.log_prob(
            torch.from_numpy(data[n_train:]).to(device)).mean()

    message = 'Test loss (clean) = %.5f' % test_loss_clean
    logger.info(message)


if __name__ == '__main__':
    main()
