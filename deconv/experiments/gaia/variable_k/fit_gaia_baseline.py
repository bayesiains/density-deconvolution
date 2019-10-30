import argparse
import json
import time

import numpy as np
import torch
import torch.utils.data as data_utils

from deconv.gmm.sgd_deconv_gmm import SGDDeconvDataset
from deconv.gmm.util import minibatch_k_means

from extreme_deconvolution import extreme_deconvolution


def fit_gaia_baseline(datafile, output_prefix, K, epochs, w_reg,
                      k_means_iters):
    data = np.load(datafile)

    train_data = SGDDeconvDataset(
        torch.Tensor(data['X_train']),
        torch.Tensor(data['C_train'])
    )

    loader = data_utils.DataLoader(
        train_data,
        batch_size=5000,
        num_workers=4,
        shuffle=True
    )

    start_time = time.time()
    counts, centroids = minibatch_k_means(loader, k=K, max_iters=10)

    weights = (counts / counts.sum()).numpy()
    means = centroids.numpy()
    covars = np.array(K * [np.eye(7)])

    ll = extreme_deconvolution(
        data['X_train'],
        data['C_train'],
        weights,
        means,
        covars,
        w=w_reg,
        logfile=str(output_prefix) + '_log',
        maxiter=epochs
    )

    end_time = time.time()

    train_score = ll * data['X_train'].shape[0]

    val_ll = extreme_deconvolution(
        data['X_val'],
        data['C_val'],
        weights,
        means,
        covars,
        w=w_reg,
        likeonly=True
    )
    val_score = val_ll * data['X_val'].shape[0]

    print('Training score: {}'.format(train_score))
    print('Val score: {}'.format(val_score))

    results = {
        'start_time': start_time,
        'end_time': end_time,
        'train_score': train_score,
        'val_score': val_score,
    }

    json.dump(results, open(str(output_prefix) + '_results.json', mode='w'))
    np.savez(
        output_prefix + '_params.npz',
        weights=weights,
        means=means,
        covar=covars
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--components', type=int)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-w', '--w_reg', type=float)
    parser.add_argument('-k', '--k-means-iters', type=int)
    parser.add_argument('datafile')
    parser.add_argument('output_prefix')

    args = parser.parse_args()

    fit_gaia_baseline(
        args.datafile,
        args.output_prefix,
        args.components,
        args.epochs,
        args.w_reg,
        args.k_means_iters,
    )
