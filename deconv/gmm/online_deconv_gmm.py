import torch
import torch.utils.data as data_utils

from .deconv_gmm import DeconvGMM


class OnlineDeconvGMM(DeconvGMM):

    def __init__(self, components, dimensions, max_iters=1000,
                 chol_factor=1e-6, tol=1e-6, step_size=0.1, batch_size=100,
                 restarts=5, max_no_improvement=20, device=None):
        super().__init__(components, dimensions, max_iters=max_iters,
                         chol_factor=chol_factor, tol=tol, restarts=restarts,
                         device=device)
        self.batch_size = batch_size
        self.step_size = step_size
        self.max_no_improvement = max_no_improvement

    def _init_expectations(self, data):
        return super()._init_expectations(data)

    def fit(self, data):
        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True
        )

        n = len(loader)

        n_inf = torch.tensor(float('-inf'), device=self.device)

        best_log_prob = torch.tensor(float('-inf'), device=self.device)

        for j in range(self.restarts):

            d = [a.to(self.device) for a in next(iter(loader))]

            expectations = self._init_expectations(d)
            self.weights = torch.zeros(self.k, 1, device=self.device)
            self.means = torch.zeros(self.k, self.d, device=self.device)
            self.covars = torch.zeros(
                self.k, self.d, self.d, device=self.device
            )
            self._m_step(expectations, n, self.step_size)

            prev_log_prob = torch.tensor(float('-inf'), device=self.device)
            max_log_prob = torch.tensor(float('-inf'), device=self.device)
            no_improvements = 0

            for i in range(self.max_iters):
                running_log_prob = torch.zeros(1, device=self.device)
                for _, d in enumerate(loader):
                    d = [a.to(self.device) for a in d]
                    log_prob, expectations = self._e_step(d)
                    if log_prob == n_inf:
                        break
                    running_log_prob += log_prob
                    self._m_step(expectations, n, self.step_size)

                if torch.abs(running_log_prob - prev_log_prob) < self.tol:
                    break
                if running_log_prob > max_log_prob:
                    no_improvements = 0
                    max_log_prob = running_log_prob
                else:
                    no_improvements += 1

                if no_improvements > self.max_no_improvement:
                    break

                prev_log_prob = running_log_prob

            if running_log_prob > best_log_prob and running_log_prob != 0.0:
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
