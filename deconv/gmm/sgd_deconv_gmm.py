import torch
import torch.distributions as dist
import torch.utils.data as data_utils

from .sgd_gmm import SGDGMMModule, BaseSGDGMM

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
                 batch_size=64, tol=1e-6, w=1e-3,
                 k_means_factor=100, k_means_iters=10, lr_step=5,
                 lr_gamma=0.1, device=None):
        self.module = SGDDeconvGMMModule(components, dimensions, w, device)
        super().__init__(
            components, dimensions, epochs=epochs, lr=lr,
            batch_size=batch_size, w=w, tol=tol,
            k_means_factor=k_means_factor, k_means_iters=k_means_iters,
            lr_step=lr_step, lr_gamma=lr_gamma, device=device
        )
