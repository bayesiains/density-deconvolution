import torch
import torch.utils.data as data_utils

from .deconv_gmm import DeconvGMM


class OnlineDeconvGMM(DeconvGMM):

    def __init__(self, components, dimensions, max_iters=1000,
                 chol_factor=1e-6, tol=1e-9, restarts=5, device=None):
        super().__init__(components, dimensions, max_iters=max_iters,
                         chol_factor=chol_factor, tol=tol, restarts=restarts,
                         device=device)

    def _init_expectations(self, data):
        return super()._init_expectations(data)

    def fit(self, data):
        loader = data_utils.DataLoader(
            data,
            batch_size=100,
            num_workers=4
        )

        n = len(loader)

        d = next(iter(loader))

        n_inf = torch.tensor(float('-inf'), device=self.device)

        best_log_prob = torch.tensor(float('-inf'), device=self.device)

        for j in range(self.restarts):

            expectations = self._init_expectations(d)
            self.weights = torch.zeros(self.k, 1)
            self.means = torch.zeros(self.k, self.d)
            self.covars = torch.zeros(self.k, self.d, self.d)
            self._m_step(expectations, n, 0.1)

            prev_log_prob = torch.tensor(float('-inf'), device=self.device)

            for i in range(self.max_iters):
                running_log_prob = torch.zeros(1)
                for _, d in enumerate(loader):
                    log_prob, expectations = self._e_step(d)
                    if log_prob == n_inf:
                        break
                    running_log_prob += log_prob
                    self._m_step(expectations, n, 0.1)

                if torch.abs(running_log_prob - prev_log_prob) < self.tol:
                    break
                prev_log_prob = running_log_prob

            if running_log_prob > best_log_prob and running_log_prob != 0:
                best_params = (
                    self.weights.clone().detach(),
                    self.means.clone().detach(),
                    self.covars.clone().detach(),
                    self.chol_covars.clone().detach()
                )
                best_log_prob = running_log_prob

        if best_log_prob == n_inf:
            raise ValueError('Could not fit model. Try increasing chol_reg?')

        self.weights, self.means, self.covars, self.chol_covars = best_params

    def _m_step(self, expectations, n, gamma):
        log_resps, cond_means, cond_covars = expectations
        resps = torch.exp(log_resps)[:, :, None]    # n, j, 1
        weights = resps.sum(dim=0)  # j, 1

        self.means = ((1 - gamma) * self.means) + gamma * (
            (resps * cond_means).sum(dim=0) / weights  # j, d
        )

        for j in range(self.k):
            diffs = self.means - cond_means    # n, j, d
            outer_p = torch.matmul(     # n, d, d
                torch.transpose(diffs[:, j, None, :], 1, 2),    # n, d, 1
                diffs[:, j, None, :]    # n, 1, d
            )
            self.covars[j, :, :] = ((1 - gamma) * self.covars[j, :, :]) + (
                gamma * torch.sum(   # d, d
                    resps[:, j, :, None] * (    # n, 1, 1
                        cond_covars[:, j, :, :] +   # n, d, d
                        outer_p     # n, d, d
                    ),
                    dim=0
                ) / weights[j, :]
            )
        self.weights = self.weights * (1 - gamma) + gamma * (weights / n)
