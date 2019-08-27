import torch
import torch.distributions as dist
import torch.nn as nn
import torch.utils.data as data_utils

from .sgd_gmm import SGDGMMModule, BaseSGDGMM
from .util import minibatch_k_means

mvn = dist.multivariate_normal.MultivariateNormal


class SGDDeconvGMMModule(SGDGMMModule):

    def forward(self, data):
        x, noise_covars = data

        weights = self.soft_max(self.soft_weights)

        T = self.covars[None, :, :, :] + noise_covars[:, None, :, :]

        log_resp = mvn(loc=self.means, covariance_matrix=T).log_prob(
            x[:, None, :]
        )
        log_resp += torch.log(weights)

        log_prob = torch.logsumexp(log_resp, dim=1)

        return -1 * torch.sum(log_prob)


class SGDDeconvDataset(data_utils.Dataset):

    def __init__(self, X, noise_covars):
        self.X = X
        self.noise_covars = noise_covars

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (self.X[i, :], self.noise_covars[i, :, :])


class SGDDeconvGMM(BaseSGDGMM):

    def __init__(self, components, dimensions, epochs=10000, lr=1e-3,
                 batch_size=64, tol=1e-6, device=None):
        self.module = SGDDeconvGMMModule(components, dimensions, device)
        super().__init__(
            components, dimensions, epochs=epochs, lr=lr,
            batch_size=batch_size, tol=tol, device=device
        )

    def init_params(self, loader):
        counts, centroids = minibatch_k_means(loader, self.k, device=self.device)
        self.module.soft_weights.data = torch.log(counts / counts.sum())
        self.module.means.data = centroids
        self.module.l_diag.data = nn.Parameter(torch.zeros(self.k, self.d), device=self.device)
        self.module.l_lower.data = torch.zeros(self.k, self.d * (self.d - 1) // 2, device=self.device)
