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

parser = argparse.ArgumentParser(description='Train SVI model on toy GMM.')

parser.add_argument('-k', '--samples', type=int)
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-l', '--learning-rate', type=float)
parser.add_argument('-i', '--use-iwae', action='store_true')
parser.add_argument('output_prefix')

args = parser.parse_args()

K = 4
D = 2
N = 50000
N_val = int(0.25 * N)

torch.set_default_tensor_type(torch.FloatTensor)

ref_gmm = SGDDeconvGMM(
    K,
    D,
    batch_size=512,
    device=torch.device('cpu')
)

ref_gmm.module.soft_weights.data = torch.zeros(K)
scale = 2

ref_gmm.module.means.data = torch.Tensor([
    [-scale, 0],
    [scale, 0],
    [0, -scale],
    [0, scale]
])

short_std = 0.3
long_std = 1

stds = torch.Tensor([
    [short_std, long_std],
    [short_std, long_std],
    [long_std, short_std],
    [long_std, short_std]
])

ref_gmm.module.l_diag.data = torch.log(stds)

state = torch.get_rng_state()
torch.manual_seed(432988)

z_train = ref_gmm.sample_prior(N)
z_val = ref_gmm.sample_prior(N_val)

noise_short = 0.1
noise_long = 1.0

S = torch.Tensor([
    [noise_short, 0],
    [0, noise_long]
])

noise_distribution = torch.distributions.MultivariateNormal(
    loc=torch.Tensor([0, 0]),
    covariance_matrix=S
)

x_train = z_train + noise_distribution.sample([N])
x_val = z_val + noise_distribution.sample([N_val])

torch.set_rng_state(state)

train_data = DeconvDataset(x_train.squeeze(), torch.cholesky(S.repeat(N, 1, 1)))
val_data = DeconvDataset(x_val.squeeze(), torch.cholesky(S.repeat(N, 1, 1)))

svi = SVIFlow(
    2,
    5,
    device=torch.device('cuda'),
    batch_size=512,
    epochs=args.epochs,
    lr=args.learning_rate,
    n_samples=args.samples,
    use_iwae=args.use_iwae,
    context_size=64
)
svi.fit(train_data, val_data=val_data)

torch.save(svi.model.state_dict(), args.output_prefix + '_params.pt')
