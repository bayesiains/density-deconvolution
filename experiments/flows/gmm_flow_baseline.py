import numpy as np

import torch

from deconv.flow.coupling import AffineCouplingFlow
from deconv.gmm.sgd_gmm import SGDGMM

D = 7
K = 128
N = 10000
N_train = 8000

weights, means, covars = torch.load(
    'results/variable_k/em_128_508798_params.pkl',
    map_location=torch.device('cpu')
)

X = torch.distributions.MultivariateNormal(
    loc=means, covariance_matrix=covars
).sample((N,)).numpy()

j = torch.distributions.Categorical(
    probs=weights.squeeze()
).sample((N,)).numpy()

X = X[np.arange(N), j, :]

X_train = [X[:N_train]]
X_val = [X[N_train:]]


# X_data = [torch.Tensor(X.reshape(-1, D).astype(np.float32))]
# X_val = [torch.Tensor(X_val.reshape(-1, D).astype(np.float32))]

#device = torch.device('cuda')

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.multiprocessing.set_start_method('spawn')

m = AffineCouplingFlow(
    D,
    10,
    100
)
m.fit(X_train, val_data=X_val)

gmm = SGDGMM(K, D, epochs=100, batch_size=256)
gmm.fit(X_train, val_data=X_val, verbose=True)
