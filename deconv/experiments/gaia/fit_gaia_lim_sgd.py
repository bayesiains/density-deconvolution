import argparse

import numpy as np
import torch

from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM, SGDDeconvDataset


def fit_gaia_lim_sgd(datafile, K, use_cuda):
    data = np.load(datafile)

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data = SGDDeconvDataset(
        torch.Tensor(data['X_train']),
        torch.Tensor(data['C_train'])
    )

    val_data = SGDDeconvDataset(
        torch.Tensor(data['X_val']),
        torch.Tensor(data['C_val'])
    )

    gmm = SGDDeconvGMM(
        K,
        7,
        device=device,
        batch_size=500,
        epochs=1000,
        restarts=1,
        lr=1e-1
    )
    gmm.fit(train_data, val_data=val_data, verbose=True)
    train_score = gmm.score_batch(train_data)
    val_score = gmm.score_batch(val_data)

    print('Training score: {}'.format(train_score))
    print('Val score: {}'.format(val_score))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--components', type=int)
    parser.add_argument('--use-cuda', action='store_true', help='Use GPU')
    parser.add_argument('datafile')

    args = parser.parse_args()

    fit_gaia_lim_sgd(args.datafile, args.components, args.use_cuda)
