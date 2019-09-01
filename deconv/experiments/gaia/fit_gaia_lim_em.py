import argparse
import json
import time

import numpy as np
import torch

from deconv.gmm.online_deconv_gmm import OnlineDeconvGMM
from deconv.gmm.sgd_deconv_gmm import SGDDeconvDataset


def fit_gaia_lim_em(datafile, output_prefix, K, batch_size, epochs, step_size, w_reg,
                    k_means_iters, use_cuda):
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

    gmm = OnlineDeconvGMM(
        K,
        7,
        device=device,
        batch_size=batch_size,
        restarts=1,
        w=w_reg,
        step_size=step_size,
        epochs=epochs,
        k_means_iters=k_means_iters
    )

    start_time = time.time()
    gmm.fit(train_data, val_data=val_data, verbose=True)
    end_time = time.time()

    train_score = gmm.score_batch(train_data)
    val_score = gmm.score_batch(val_data)

    print('Training score: {}'.format(train_score))
    print('Val score: {}'.format(val_score))

    results = {
        'start_time': start_time,
        'end_time': end_time,
        'train_score': train_score,
        'val_score': val_score,
        'train_curve': gmm.train_ll_curve,
        'val_curve': gmm.val_ll_curve
    }

    json.dump(results, open(str(output_prefix) + '_results.json', mode='w'))
    torch.save(
        (gmm.weights, gmm.means, gmm.covars),
        output_prefix + '_params.pkl'
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--components', type=int)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-s', '--step-size', type=float)
    parser.add_argument('-w', '--w_reg', type=float)
    parser.add_argument('-k', '--k-means-iters', type=int)
    parser.add_argument('--use-cuda', action='store_true', help='Use GPU')
    parser.add_argument('datafile')
    parser.add_argument('output_prefix')


    args = parser.parse_args()

    fit_gaia_lim_em(
        args.datafile,
        args.output_prefix,
        args.components,
        args.batch_size,
        args.epochs,
        args.step_size,
        args.w_reg,
        args.k_means_iters,
        args.use_cuda
    )
