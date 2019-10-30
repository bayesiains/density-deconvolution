import torch
import torch.distributions as dist

from .sgd_gmm import SGDGMMModule, BaseSGDGMM

mvn = dist.multivariate_normal.MultivariateNormal


class SGDDeconvGMMModule(SGDGMMModule):
    """Implementation of a deconvolving GMM as a PyTorch nn module."""

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


class SGDDeconvGMM(BaseSGDGMM):
    """Concrete implementation of class to fit a deconvolving GMM with SGD."""

    def __init__(self, components, dimensions, epochs=10000, lr=1e-3,
                 batch_size=64, tol=1e-6, w=1e-3, restarts=5,
                 k_means_factor=100, k_means_iters=10, lr_step=5,
                 lr_gamma=0.1, device=None):
        self.module = SGDDeconvGMMModule(components, dimensions, w, device)
        super().__init__(
            components, dimensions, epochs=epochs, lr=lr,
            batch_size=batch_size, w=w, tol=tol, restarts=restarts,
            k_means_factor=k_means_factor, k_means_iters=k_means_iters,
            lr_step=lr_step, lr_gamma=lr_gamma, device=device
        )
