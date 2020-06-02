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

parser = argparse.ArgumentParser(description='Process SVI on GMM results.')

parser.add_argument('gmm_results_dir')
parser.add_argument('elbo_results_dir')
parser.add_argument('iw_results_dir')

args = parser.parse_args()

K = 4
D = 2
N = 50000

torch.set_default_tensor_type(torch.FloatTensor)

_, S, _, _, (z_test, x_test) = generate_mixture_data()
test_data = DeconvDataset(x_test.squeeze(), S.repeat(N, 1, 1))

gmm_params = []

for f in os.listdir(args.gmm_results_dir):
    path = os.path.join(args.gmm_results_dir, f)
    gmm_params.append(path)

elbo_params = collections.defaultdict(list)

for f in os.listdir(args.elbo_results_dir):
    path = os.path.join(args.elbo_results_dir, f)
    elbo_params[int(f[16:18])].append(path)

iw_params = collections.defaultdict(list)

for f in os.listdir(args.iw_results_dir):
    path = os.path.join(args.iw_results_dir, f)
    iw_params[int(f[16:18])].append(path)

gmm = SGDDeconvGMM(
    K,
    D,
    batch_size=200,
    device=torch.device('cuda')
)
test_gmm = SGDGMM(
    K,
    D,
    batch_size=200,
    device=torch.device('cuda')
)

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

for p in gmm_params:
    gmm.module.load_state_dict(torch.load(p))
    test_gmm.module.load_state_dict(torch.load(p))
    with torch.no_grad():
        logv = test_gmm.module([z_test[0].to(torch.device('cuda'))]).mean().item()
    logp = gmm.score_batch(test_data) / N
    results.append({
            'i': 0,
            'model': 'gmm',
            'elbo': None,
            'log_p_v': logv,
            'log_p_w': logp,
            'kl': None
    })

test_data = DeconvDataset(x_test.squeeze(), torch.cholesky(S.repeat(N, 1, 1)))

torch.set_default_tensor_type(torch.cuda.FloatTensor)

param_sets = {
    'svi_elbo': elbo_params,
    'svi_iw': iw_params
}

for k, params in param_sets.items():
    print('Processing {}'.format(k))
    for i in (1, 10, 25, 50):
        print('Processing K: {}'.format(i))
        for p in params[i]:
            svi.model.load_state_dict(torch.load(p))
            with torch.no_grad():
                logv = svi.model._prior.log_prob(z_test[0].to(torch.device('cuda'))).mean().item()
            elbo = svi.score_batch(test_data, num_samples=100) / N
            logp = svi.score_batch(test_data, num_samples=100, log_prob=True) / N

            results.append({
                'i': i,
                'model': k,
                'elbo': elbo,
                'log_p_v': logv,
                'log_p_w': logp,
                'kl': logp - elbo
            })

df = pd.DataFrame(results)
        
        
    
    