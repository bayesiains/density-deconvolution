"""Script to fit the SGD model."""
import argparse
import json
import time

import numpy as np
import torch

from deconv.flow.svi import SVIFLow
from deconv.gmm.data import DeconvDataset


def fit_gaia_lim_sgd(datafile, use_cuda=False):
    data = np.load(datafile)

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data = DeconvDataset(
        torch.Tensor(data['X_train']),
        torch.Tensor(data['L_train'])
    )

    val_data = DeconvDataset(
        torch.Tensor(data['X_val']),
        torch.Tensor(data['L_val'])
    )

    svi = SVIFLow(
        7,
        10,
        device=device,
        batch_size=512,
        epochs=40,
        lr=1e-3
    )
    svi.fit(train_data, val_data=val_data)

    val_log_prob = svi.score_batch(val_data, log_prob=True)

    print('Val log prob: {}'.format(val_log_prob / len(val_data)))
