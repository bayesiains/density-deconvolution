import argparse
import collections
import os

import numpy as np
import torch
import torch.utils.data as data_utils

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('paper')

import corner

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import pandas as pd

from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM
from deconv.gmm.sgd_gmm import SGDGMM
from deconv.gmm.data import DeconvDataset
from deconv.gmm.plotting import plot_covariance
from deconv.flow.svi import SVIFlow
from deconv.flow.svi_gmm import SVIGMMFlow, SVIGMMExact
from deconv.utils.data_gen import generate_mixture_data

parser = argparse.ArgumentParser(description='Process SVI on GMM results.')

parser.add_argument('pretrained_results_dir')
parser.add_argument('posttrained_results_dir')
parser.add_argument('svi_gmm_results_dir')
parser.add_argument('svi_exact_gmm_results_dir')

args = parser.parse_args()

K = 3
D = 2
N = 10000

svi = SVIFlow(
    2,
    5,
    device=torch.device('cuda'),
    batch_size=4096,
    epochs=100,
    lr=1e-4,
    n_samples=50,
    use_iwae=False,
    context_size=64
)
svi_gmm = SVIGMMFlow(
    2,
    5,
    device=torch.device('cuda'),
    batch_size=512,
    epochs=100,
    lr=1e-4,
    n_samples=50,
    use_iwae=False,
    context_size=64,
    hidden_features=128
)
svi_exact_gmm = SVIGMMExact(
    2,
    5,
    device=torch.device('cuda'),
    batch_size=512,
    epochs=100,
    lr=1e-4,
    n_samples=50,
    use_iwae=False,
    context_size=64,
    hidden_features=128
)

test_gmm = SGDDeconvGMM(
    K,
    D,
    batch_size=200,
    device=torch.device('cuda'),
    w=0
)

torch.set_default_tensor_type(torch.cuda.FloatTensor)
params = []
params.append(os.path.join(
    args.pretrained_results_dir,
    os.listdir(args.pretrained_results_dir)[0]
))
params.append(os.path.join(
    args.posttrained_results_dir,
    os.listdir(args.posttrained_results_dir)[0]
))
params.append(os.path.join(
    args.svi_gmm_results_dir,
    os.listdir(args.svi_gmm_results_dir)[0]
))
params.append(os.path.join(
    args.svi_exact_gmm_results_dir,
    os.listdir(args.svi_exact_gmm_results_dir)[0]
))

mean = np.array([[3.0, 0.0], [0.0, 0.0]])
cov = np.array([
    [
        [0.1, 0],
        [0, 1]
    ],
    [
        [0.1, 0],
        [0, 1]
    ]
])
test_point = [
    torch.Tensor(mean).to(svi.device),
    torch.cholesky(torch.Tensor(cov)).to(svi.device)
]
test_point_gmm = [
    torch.Tensor(mean).to(svi.device),
    torch.Tensor(cov).to(svi.device)
]

for i, p in enumerate(params):
    if i == 2:
        svi_gmm.model.load_state_dict(torch.load(p))
        test_gmm.module.load_state_dict(svi_gmm.model._prior.state_dict())
        z_samples = test_gmm.sample_prior(10000)
        x_samples = svi_gmm.sample_posterior(test_point, 10000)
    elif i == 3:
        svi_exact_gmm.model.load_state_dict(torch.load(p))
        test_gmm.module.load_state_dict(svi_exact_gmm.model._prior.state_dict())
        z_samples = test_gmm.sample_prior(10000)
        x_samples = test_gmm.sample_posterior(test_point_gmm, 10000)
    else:
        svi.model.load_state_dict(torch.load(p))
        z_samples = svi.sample_prior(10000)
        x_samples = svi.sample_posterior(test_point, 10000)

    ax_lim = (-4, 4)

    fig, axes = plt.subplots(1, 2, figsize=(3, 1.5), sharex=True, sharey=True)

    corner.hist2d(z_samples[0, :N, 0].numpy(), z_samples[0, :N, 1].numpy(), ax=axes[0])
    corner.hist2d(x_samples[1, :N, 0].numpy(), x_samples[1, :N, 1].numpy(), ax=axes[1])
    
    plot_covariance(
        mean[1],
        cov[1],
        ax=axes[1],
        color='r'
    )
    
    axes[0].set_title(r'$p_{\theta}(\mathbf{v})$')
    axes[1].set_title(r'$q_{\phi}(\mathbf{v})$')

    axes[0].set_xticks([]) 
    axes[0].set_yticks([]) 
    axes[1].set_xticks([]) 
    axes[1].set_yticks([]) 

    axes[1].set_xlim(ax_lim)
    axes[1].set_ylim(ax_lim)

    fig.tight_layout()
    fig.savefig('additional_{}.pdf'.format(i))

