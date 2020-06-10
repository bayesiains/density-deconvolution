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
from deconv.gmm.data import DeconvDataset
from deconv.gmm.plotting import plot_covariance
from deconv.flow.svi import SVIFlow
from deconv.utils.data_gen import generate_mixture_data

parser = argparse.ArgumentParser(description='Process SVI on GMM results.')

parser.add_argument('gmm_results_dir')
parser.add_argument('elbo_results_dir')
parser.add_argument('iw_results_dir')



args = parser.parse_args()

K = 3
D = 2
N = 10000

torch.set_default_tensor_type(torch.FloatTensor)

ref_gmm, S, _, _, (z_test, x_test) = generate_mixture_data()

ax_lim = (-4, 4)

fig, axes = plt.subplots(1, 2, figsize=(3, 1.5), sharex=True, sharey=True)

corner.hist2d(z_test[0, :N, 0].numpy(), z_test[0, :N, 1].numpy(), ax=axes[0])
corner.hist2d(x_test[0, :N, 0].numpy(), x_test[0, :N, 1].numpy(), ax=axes[1])
axes[0].set_title(r'$p(\mathbf{v})$')
axes[1].set_title(r'$p(\mathbf{w})$')

axes[0].set_xticks([]) 
axes[0].set_yticks([]) 
axes[1].set_xticks([]) 
axes[1].set_yticks([]) 

axes[1].set_xlim(ax_lim)
axes[1].set_ylim(ax_lim)

fig.tight_layout()

fig.savefig('toy_data.pdf')

plt.show()

gmm_params = []

for f in os.listdir(args.gmm_results_dir):
    path = os.path.join(args.gmm_results_dir, f)
    gmm_params.append(path)

fig, axes = plt.subplots(2, 4, figsize=(8, 3), sharex=True, sharey=True)

ref_gmm.module.load_state_dict(torch.load(gmm_params[0]))
ref_samples = ref_gmm.sample_prior(10000)

corner.hist2d(ref_samples[0, :, 0].numpy(), ref_samples[0, :, 1].numpy(), ax=axes[0, 0])

axes[0, 0].set_title('GMM')

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
    torch.Tensor(mean).to(ref_gmm.device),
    torch.Tensor(cov).to(ref_gmm.device)
]

post_samples = ref_gmm.sample_posterior(test_point, 10000)
corner.hist2d(post_samples[1, :, 0].numpy(), post_samples[1, :, 1].numpy(), ax=axes[1, 0])

plot_covariance(
    mean[1],
    cov[1],
    ax=axes[1, 0],
    color='r'
)

axes[0, 0].set_xticks([]) 
axes[0, 0].set_yticks([]) 
axes[1, 0].set_xticks([]) 
axes[1, 0].set_yticks([]) 

axes[0, 0].set_ylabel(r'$p_{\theta}(\mathbf{v})$')
axes[1, 0].set_ylabel(r'$q_{\phi}(\mathbf{v})$')

elbo_params = collections.defaultdict(list)

for f in os.listdir(args.elbo_results_dir):
    if f.startswith('mixture'):
        path = os.path.join(args.elbo_results_dir, f)
        elbo_params[int(f[16:18])].append(path)
        
iw_params = collections.defaultdict(list)

for f in os.listdir(args.iw_results_dir):
    if f.startswith('mixture'):
        path = os.path.join(args.iw_results_dir, f)
        iw_params[int(f[16:18])].append(path)

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

test_point = [
    torch.Tensor(mean).to(svi.device),
    torch.cholesky(torch.Tensor(cov)).to(svi.device)
]

for i, k in enumerate(('1', '50_elbo', '50_iw')):
    if k == '1':
        p = elbo_params[1][0]
        title = r'Flow with $\mathcal{L}(1)$'
    elif k == '50_elbo':
        p = elbo_params[50][0]
        title = r'Flow with $\mathcal{L}(50)$'
    else:
        p = iw_params[50][0]
        title = r'Flow with $\mathcal{L}_{\rm{IW}}(50)$'
    svi.model.load_state_dict(torch.load(p))
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    prior_samples = svi.sample_prior(10000)
    
    axes[0, i + 1].set_title(title)
    corner.hist2d(prior_samples[0, :, 0].numpy(), prior_samples[0, :, 1].numpy(), ax=axes[0, i + 1])
    
    if k == '50_iw':
        post_samples = svi.resample_posterior(test_point, 10000)
    else:
        post_samples = svi.sample_posterior(test_point, 10000)
    corner.hist2d(post_samples[1, :, 0].numpy(), post_samples[1, :, 1].numpy(), ax=axes[1, i + 1])
    
    plot_covariance(
        mean[1],
        cov[1],
        ax=axes[1, i + 1],
        color='r'
    )
     
    axes[0, i + 1].get_xaxis().set_visible(False)
    axes[0, i + 1].get_yaxis().set_visible(False)
    axes[1, i + 1].get_xaxis().set_visible(False)
    axes[1, i + 1].get_yaxis().set_visible(False)

    axes[1, i + 1].set_xlim(ax_lim)
    axes[1, i + 1].set_ylim(ax_lim)

plt.tick_params()
        
fig.tight_layout()
fig.savefig('toy_model.pdf')
plt.show()