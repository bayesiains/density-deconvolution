import argparse
import json
import time

import numpy as np
import torch

from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM, SGDDeconvDataset


def fit_gaia_lim_sgd(datafile, output_prefix, K, batch_size, epochs, lr,
                     w_reg, k_means_iters, lr_step, lr_gamma,
                     use_cuda):
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
        batch_size=batch_size,
        epochs=epochs,
        w=w_reg,
        restarts=1,
        k_means_iters=k_means_iters,
        lr=lr,
        lr_step=lr_step,
        lr_gamma=lr_gamma
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
        'train_curve': gmm.train_loss_curve,
        'val_curve': gmm.val_loss_curve
    }
    json.dump(results, open(str(output_prefix) + '_results.json', mode='w'))
    torch.save(gmm.module.state_dict(), output_prefix + '_params.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--components', type=int)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-l', '--learning-rate', type=float)
    parser.add_argument('-w', '--w_reg', type=float)
    parser.add_argument('-k', '--k-means-iters', type=int)
    parser.add_argument('--lr-step', type=int)
    parser.add_argument('--lr-gamma', type=float)
    parser.add_argument('--use-cuda', action='store_true', help='Use GPU')
    parser.add_argument('datafile')
    parser.add_argument('output_prefix')

    args = parser.parse_args()

    fit_gaia_lim_sgd(
        args.datafile,
        args.output_prefix,
        args.components,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.w_reg,
        args.k_means_iters,
        args.lr_step,
        args.lr_gamma,
        args.use_cuda
    )
