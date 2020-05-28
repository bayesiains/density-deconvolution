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
from deconv.utils.make_2d_toy_data import data_gen
from deconv.utils.make_2d_toy_noise_covar import covar_gen
from deconv.gmm.sgd_deconv_gmm_tim import SGDDeconvGMM
from deconv.gmm.data import DeconvDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='boston')
parser.add_argument('--covar', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=int(1e4))
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    torch.cuda.set_device(args.gpu)

else:
    torch.set_default_tensor_type('torch.DoubleTensor')

args.dir = 'hypertuning_gmm/' + '/' + str(args.covar) + '/'

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

name = args.data

if not os.path.exists(args.dir + 'logs/'):
    os.makedirs(args.dir + 'logs/')

logger = get_logger(logpath=(args.dir + 'logs/' + name +
                             '.log'), filepath=os.path.abspath(__file__))

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

    kf = KFold(n_splits=5)

    # 54 combinations
    lr_list = [1e-2, 5e-3, 1e-3]
    K_list = [20, 50, 100, 200, 500]

    n_combs = 0
    for lr, K in product(lr_list, K_list):
        n_combs += 1

    best_eval = np.zeros((n_combs, 5))

    counter = 0
    for lr, K in product(lr_list, K_list):
        logger.info((lr, K))

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

            model = SGDDeconvGMM(
                K,
                n_features,
                device=device,
                batch_size=args.batch_size,
                epochs=args.n_epochs,
                lr=lr)

            message = 'Total number of parameters: %s' % (
                sum(p.numel() for p in model.module.parameters()))
            logger.info(message)

            best_eval_loss = model.fit(train_dataset, logger,
                                       val_data=eval_dataset, verbose=True)

            best_eval[counter, i] = best_eval_loss
            np.save(args.data + '_gmm_hypertuning_results_tmp', best_eval)

        counter += 1

    np.save(args.data + '_gmm_hypertuning_results', best_eval)


if __name__ == '__main__':
    main()
