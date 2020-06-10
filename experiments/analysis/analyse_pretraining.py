import argparse
import collections
import os

import numpy as np
import torch
import torch.utils.data as data_utils

import pandas as pd

from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM
from deconv.gmm.sgd_gmm import SGDGMM
from deconv.gmm.data import DeconvDataset
from deconv.flow.svi import SVIFlow
from deconv.utils.data_gen import generate_mixture_data

_, S, _, _, (z_test, x_test) = generate_mixture_data()

parser = argparse.ArgumentParser(description='Process SVI on GMM results.')

parser.add_argument('pretrain_results_dir')
parser.add_argument('elbo_results_dir')
parser.add_argument('iw_results_dir')

args = parser.parse_args()

K = 3
D = 2
N = 50000

torch.set_default_tensor_type(torch.FloatTensor)

pretrained_params = []
for f in os.listdir(args.pretrain_results_dir):
    path = os.path.join(args.pretrain_results_dir, f)
    pretrained_params.append(path)

elbo_params = []
for f in os.listdir(args.elbo_results_dir):
    path = os.path.join(args.elbo_results_dir, f)
    elbo_params.append(path)

iw_params = []
for f in os.listdir(args.iw_results_dir):
    path = os.path.join(args.iw_results_dir, f)
    iw_params.append(path)

svi = SVIFlow(
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

results = []

test_data = DeconvDataset(x_test.squeeze(), torch.cholesky(S.repeat(N, 1, 1)))

torch.set_default_tensor_type(torch.cuda.FloatTensor)

for p in pretrained_params:
    svi.model.load_state_dict(torch.load(p))
    with torch.no_grad():
        logv = svi.model._prior.log_prob(z_test[0].to(torch.device('cuda'))).mean().item()
    elbo = svi.score_batch(test_data, num_samples=100) / N
    logp = svi.score_batch(test_data, num_samples=100, log_prob=True) / N

    results.append({
        'i': 50,
        'model': 'pretrained',
        'elbo': elbo,
        'log_p_v': logv,
        'log_p_w': logp,
        'kl': logp - elbo
    })

for p in elbo_params:
    svi.model.load_state_dict(torch.load(p))
    with torch.no_grad():
        logv = svi.model._prior.log_prob(z_test[0].to(torch.device('cuda'))).mean().item()
    elbo = svi.score_batch(test_data, num_samples=100) / N
    logp = svi.score_batch(test_data, num_samples=100, log_prob=True) / N

    results.append({
        'i': 50,
        'model': 'svi_elbo',
        'elbo': elbo,
        'log_p_v': logv,
        'log_p_w': logp,
        'kl': logp - elbo
    })

for p in iw_params:
    svi.model.load_state_dict(torch.load(p))
    with torch.no_grad():
        logv = svi.model._prior.log_prob(z_test[0].to(torch.device('cuda'))).mean().item()
    elbo = svi.score_batch(test_data, num_samples=100) / N
    logp = svi.score_batch(test_data, num_samples=100, log_prob=True) / N

    results.append({
        'i': 50,
        'model': 'svi_iw',
        'elbo': elbo,
        'log_p_v': logv,
        'log_p_w': logp,
        'kl': logp - elbo
    })

df = pd.DataFrame(results)
        
        
    
    