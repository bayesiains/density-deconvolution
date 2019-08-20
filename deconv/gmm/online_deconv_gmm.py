import torch
import torch.utils.data as data_utils

from .deconv_gmm import DeconvGMM
from .util import minibatch_k_means


class OnlineDeconvGMM(DeconvGMM):

    def __init__(self, components, dimensions, max_iters=10000, gamma=1,
                 omega=None, eta=0, w=1e-6, tol=1e-6, step_size=0.1,
                 batch_size=100, restarts=5, max_no_improvement=20,
                 device=None):
        super().__init__(components, dimensions, max_iters=max_iters,
                         gamma=gamma, omega=omega, eta=eta, w=w, tol=tol,
                         restarts=restarts, device=device)
        self.batch_size = batch_size
        self.step_size = step_size
        self.max_no_improvement = max_no_improvement

    def _init_sum_stats(self, loader, n):

        counts, centroids = minibatch_k_means(loader, self.k)

        self.weights = counts[:, None] / counts.sum()
        self.means = centroids
        self.covars = torch.eye(
            self.d, device=self.device
        ).repeat(self.k, 1, 1)

        self.sum_resps = torch.zeros(self.k, 1)

        self.sum_cond_means = torch.zeros(self.k, self.d)

        self.sum_dev_ps = torch.zeros(
            self.k, self.d, self.d, device=self.device
        )

    def fit(self, data, verbose=False):
        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True
        )

        n = len(data)

        n_inf = torch.tensor(float('-inf'), device=self.device)

        best_log_prob = torch.tensor(float('-inf'), device=self.device)

        for j in range(self.restarts):

            self._init_sum_stats(loader, n)

            prev_log_prob = torch.tensor(float('-inf'), device=self.device)
            max_log_prob = torch.tensor(float('-inf'), device=self.device)
            no_improvements = 0

            d = [a.to(self.device) for a in next(iter(loader))]
            _, expectations = self._e_step(d)
            self._m_step(expectations, n, 1)

            for i in range(self.max_iters):
                running_log_prob = torch.zeros(1, device=self.device)
                for _, d in enumerate(loader):
                    d = [a.to(self.device) for a in d]
                    log_prob, expectations = self._e_step(d)
                    if log_prob == n_inf:
                        break
                    running_log_prob += log_prob
                    self._m_step(expectations, n, self.step_size)

                if verbose and i % 10 == 0:
                    print('Epoch {}, Running Log Prob: {}'.format(
                        i, running_log_prob)
                    )

                if running_log_prob == 0.0:
                    print('Log prob 0, crashed.')
                    break

                if torch.abs(running_log_prob - prev_log_prob) < self.tol:
                    print('Converged within tolerance')
                    break
                if running_log_prob > max_log_prob:
                    no_improvements = 0
                    max_log_prob = running_log_prob
                else:
                    no_improvements += 1

                if no_improvements > self.max_no_improvement:
                    print('Reached max steps with no improvement.')
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

    def _m_step(self, expectations, n, step_size):
        log_resps, cond_means, cond_covars = expectations
        resps = torch.exp(log_resps)[:, :, None]    # n, j, 1

        self.sum_resps += step_size * (resps.sum(dim=0) - self.sum_resps)
        self.weights = (self.sum_resps + self.gamma - 1) / (
            self.batch_size + self.gamma.sum() - self.k
        )

        self.sum_cond_means += step_size * (
            (resps * cond_means).sum(dim=0) - self.sum_cond_means
        )

        self.means = (self.sum_cond_means + self.eta * self.m_hat) / (
            self.sum_resps + self.eta
        )

        diffs = self.means - cond_means    # n, j, d
        outer_p = diffs[:, :, :, None] * diffs[:, :, None, :]
        sum_dev_ps = torch.sum(
            resps[:, :, :, None] * (cond_covars + outer_p),
            dim=0
        )
        self.sum_dev_ps += step_size * (sum_dev_ps - self.sum_dev_ps)

        self.covars = self.sum_dev_ps / self.sum_resps[:, :, None]

        self.covars += self.w
