import argparse

import numpy as np
import torch
import torch.utils.data as data_utils

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import corner

from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM
from deconv.gmm.data import DeconvDataset
from deconv.flow.svi import SVIFlow
from deconv.flow.svi_gmm import SVIGMMFlow, SVIGMMExact
from deconv.utils.data_gen import generate_mixture_data

parser = argparse.ArgumentParser(description='Train SVI model on toy GMM.')

parser.add_argument('-g', '--gmm', action='store_true')
parser.add_argument('-s', '--svi-gmm', action='store_true')
parser.add_argument('-x', '--svi-exact_gmm', action='store_true')
parser.add_argument('-f', '--freeze_gmm', action='store_true')
parser.add_argument('-k', '--samples', type=int)
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-l', '--learning-rate', type=float)
parser.add_argument('-i', '--use-iwae', action='store_true')
parser.add_argument('-c', '--grad_clip_norm', type=float)
parser.add_argument('-m', '--hidden-features', type=int)
parser.add_argument('output_prefix')

args = parser.parse_args()

K = 3
D = 2
N = 50000
N_val = int(0.25 * N)

ref_gmm, S, (z_train, x_train), (z_val, x_val), _ = generate_mixture_data()

if args.gmm:
    if args.svi_gmm:
        train_data = DeconvDataset(x_train.squeeze(), torch.cholesky(S.repeat(N, 1, 1)))
        val_data = DeconvDataset(x_val.squeeze(), torch.cholesky(S.repeat(N_val, 1, 1)))
        if args.svi_exact_gmm:
            svi_gmm = SVIGMMExact(
                2,
                5,
                device=torch.device('cuda'),
                batch_size=512,
                epochs=args.epochs,
                lr=args.learning_rate,
                n_samples=args.samples,
                use_iwae=args.use_iwae,
                context_size=64,
                hidden_features=args.hidden_features
            )
        else:
            svi_gmm = SVIGMMFlow(
                2,
                5,
                device=torch.device('cuda'),
                batch_size=512,
                epochs=args.epochs,
                lr=args.learning_rate,
                n_samples=args.samples,
                use_iwae=args.use_iwae,
                context_size=64,
                hidden_features=args.hidden_features
            )
            if args.freeze_gmm:
                svi_gmm.model._prior.load_state_dict(ref_gmm.module.state_dict())
                for param in svi_gmm.model._prior.parameters():
                    param.requires_grad = False

        svi_gmm.fit(train_data, val_data=val_data)
        torch.save(svi_gmm.model.state_dict(), args.output_prefix + '_params.pt')
    else:
        train_data = DeconvDataset(x_train.squeeze(), S.repeat(N, 1, 1))
        val_data = DeconvDataset(x_val.squeeze(), S.repeat(N_val, 1, 1))
        gmm = SGDDeconvGMM(
            K,
            D,
            batch_size=200,
            epochs=args.epochs,
            lr=args.learning_rate,
            device=torch.device('cuda')
        )
        gmm.fit(train_data, val_data=val_data, verbose=True)
        torch.save(gmm.module.state_dict(), args.output_prefix + '_params.pt')
else:
    train_data = DeconvDataset(x_train.squeeze(), torch.cholesky(S.repeat(N, 1, 1)))
    val_data = DeconvDataset(x_val.squeeze(), torch.cholesky(S.repeat(N_val, 1, 1)))
    svi = SVIFlow(
        2,
        5,
        device=torch.device('cuda'),
        batch_size=512,
        epochs=args.epochs,
        lr=args.learning_rate,
        n_samples=args.samples,
        use_iwae=args.use_iwae,
        grad_clip_norm=args.grad_clip_norm,
        context_size=64,
        hidden_features=args.hidden_features
    )
    svi.fit(train_data, val_data=val_data)

    torch.save(svi.model.state_dict(), args.output_prefix + '_params.pt')
