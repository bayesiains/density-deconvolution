import math

import torch
import torch.distributions as dist

from .base import BaseGMM

mvn = dist.multivariate_normal.MultivariateNormal


class DeconvGMM(BaseGMM):

    def __init__(self, components, dimensions, epochs=1000,
                 w=1e-6, tol=1e-6, device=None):
        super().__init__(components, dimensions, epochs=epochs, tol=tol,
                         device=device)

        self.w = w * torch.eye(self.d, device=self.device)

    def _init_expectations(self, data):

        X = data[0]
        n = X.shape[0]

        resps = self._kmeans_init(X)
        log_resps = torch.log(resps)

        cond_means = X[:, None, :].repeat(1, self.k, 1)
        cond_covars = torch.eye(self.d, device=self.device).repeat(
            n, self.k, 1, 1
        )

        return (log_resps, cond_means, cond_covars)

    def predict(self, X):
        return torch.exp(self._e_step(X)[1][0])

    def _e_step(self, data):

        X, noise_covars = data

        T = self.covars[None, :, :, :] + noise_covars[:, None, :, :]
        try:
            T_chol = torch.cholesky(T)
        except RuntimeError:
            return torch.tensor(float('-inf')), None
        T_inv = torch.cholesky_solve(
            torch.eye(self.d, device=self.device), T_chol
        )

        diff = X[:, None, :] - self.means
        T_inv_diff = torch.matmul(T_inv, diff[:, :, :, None])
        log_resps = -0.5 * (
            torch.matmul(
                diff[:, :, None, :],
                T_inv_diff
            ) + self.d * math.log(2 * math.pi)
        ).squeeze()
        log_resps -= T_chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)

        log_resps += torch.log(self.weights[None, :, 0])

        cond_means = self.means + torch.matmul(  # n, j, d
            self.covars[None, :, :, :],     # 1, j, d, d
            T_inv_diff
        )[:, :, :, 0]

        cond_covars = self.covars - torch.matmul(   # n, j, d, d
            self.covars,    # j, d, d
            torch.matmul(   # n, j, d, d
                T_inv,          # n, j, d, d
                self.covars     # j, d, d
            )
        )

        log_prob = torch.logsumexp(log_resps, dim=1, keepdim=True)
        log_resps -= log_prob
        return torch.sum(log_prob), (log_resps, cond_means, cond_covars)

    def _m_step(self, data, expectations):
        log_resps, cond_means, cond_covars = expectations
        n = cond_means.shape[0]
        resps = torch.exp(log_resps)[:, :, None]    # n, j, 1
        weights = resps.sum(dim=0)  # j, 1

        self.means = (
            (resps * cond_means).sum(dim=0)
        ) / (weights)  # j, d

        for j in range(self.k):
            diffs = self.means - cond_means    # n, j, d
            outer_p = torch.matmul(     # n, d, d
                torch.transpose(diffs[:, j, None, :], 1, 2),    # n, d, 1
                diffs[:, j, None, :]    # n, 1, d
            )
            self.covars[j, :, :] = (
                torch.sum(   # d, d
                    resps[:, j, :, None] * (    # n, 1, 1
                        cond_covars[:, j, :, :] +   # n, d, d
                        outer_p     # n, d, d
                    ),
                    dim=0
                )
            )
            self.covars[j, :, :] += 2 * self.w
            self.covars[j, :, :] /= weights[j]

        self.weights = weights / n

    def score(self, data):
        log_prob, _ = self._e_step
        return log_prob
