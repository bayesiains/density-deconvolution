import torch
import torch.utils.data as data_utils

from .deconv_gmm import DeconvGMM
from .util import minibatch_k_means


class OnlineDeconvGMM(DeconvGMM):

    def __init__(self, components, dimensions, epochs=1000, w=1e-6,
                 tol=1e-6, step_size=0.1, batch_size=100,
                 max_no_improvement=20, k_means_factor=100,
                 k_means_iters=10, lr_step=10, lr_gamma=0.1,
                 device=None):
        super().__init__(components, dimensions, epochs=epochs, w=w, tol=tol,
                         device=device)
        self.batch_size = batch_size
        self.step_size = step_size
        self.max_no_improvement = max_no_improvement
        self.k_means_factor = k_means_factor
        self.k_means_iters = k_means_iters
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma

    def _init_sum_stats(self, loader, n):

        counts, centroids = minibatch_k_means(
            loader, self.k, max_iters=self.k_means_iters, device=self.device
        )

        self.weights = counts[:, None] / counts.sum()
        self.means = centroids
        self.covars = torch.eye(
            self.d, device=self.device
        ).repeat(self.k, 1, 1)

        self.sum_resps = self.weights * self.batch_size

        self.sum_cond_means = self.means * self.sum_resps

    def fit(self, data, val_data=None, verbose=False, interval=1):
        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        init_loader = data_utils.DataLoader(
            data,
            batch_size=self.k_means_factor * self.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True
        )

        n = len(data)

        n_inf = float('-inf')

        self.train_ll_curve = []
        if val_data:
            self.val_ll_curve = []

        self._init_sum_stats(init_loader, n)

        prev_ll = float('-inf')
        max_val_ll = float('-inf')
        no_improvements = 0

        for i in range(self.epochs):
            train_ll = 0.0
            for _, d in enumerate(loader):
                d = [a.to(self.device) for a in d]
                log_prob, expectations = self._e_step(d)
                if log_prob == n_inf:
                    break
                train_ll += log_prob.item()
                self._m_step(expectations, n, self.step_size)

            if train_ll == 0.0:
                print('Log prob 0, crashed.')
                train_ll = 0.0
                val_ll = 0.0
                break

            train_ll = self.score_batch(data)
            self.train_ll_curve.append(train_ll)

            if val_data:
                val_ll = self.score_batch(val_data)
                self.val_ll_curve.append(val_ll)

            if (i + 1) % self.lr_step == 0:
                self.step_size *= self.lr_gamma

            if verbose and i % interval == 0:
                if val_data:
                    print('Epoch {}, Train LL: {}, Val LL: {}'.format(
                        i,
                        train_ll,
                        val_ll
                    ))
                else:
                    print('Epoch {}, Train LL: {}'.format(
                        i, train_ll
                    ))

            if abs(train_ll - prev_ll) < self.tol:
                print('Train LL converged within tolerance at {}'.format(
                    train_ll
                ))
                break

            if val_data:
                if val_ll > max_val_ll:
                    no_improvements = 0
                    max_val_ll = val_ll
                else:
                    no_improvements += 1

                if no_improvements > self.max_no_improvement:
                    print('No improvement in val LL for {} epochs. Early Stopping at {}'.format(
                        self.max_no_improvement,
                        val_ll
                    ))
                    break

            prev_ll = train_ll

    def _adjust(self, covar, scale, b, c):
        result = scale[:, :, None] * covar
        scale_sqrt = torch.sqrt(scale)

        diffs = (scale_sqrt * b - c)
        sums = (scale_sqrt * b + c)

        result += 0.5 * diffs[:, :, None] * sums[:, None, :]
        result += 0.5 * sums[:, :, None] * diffs[:, None, :]

        return result

    def _m_step(self, expectations, n, step_size):
        log_resps, cond_means, cond_covars = expectations
        resps = torch.exp(log_resps)[:, :, None]    # n, j, 1

        resps += 10 * torch.finfo(resps.dtype).eps

        sum_resps = resps.sum(dim=0)

        sum_resps += 10 * torch.finfo(sum_resps.dtype).eps

        sum_cond_means = (resps * cond_means).sum(dim=0)
        batch_means = sum_cond_means / sum_resps

        diffs = (cond_means - batch_means)
        outer_p = diffs[:, :, :, None] * diffs[:, :, None, :]
        outer_p += cond_covars

        batch_covars = torch.sum(
            resps[:, :, :, None] * outer_p,
            dim=0
        ) / sum_resps[:, :, None]

        m_old = self.means.clone()
        sum_resps_old = self.sum_resps.clone()

        self.sum_resps = (1 - step_size) * self.sum_resps + step_size * sum_resps

        self.sum_cond_means = (1 - step_size) * self.sum_cond_means + step_size * sum_cond_means
        self.means = self.sum_cond_means / self.sum_resps

        self.covars = (1 - step_size) * self._adjust(
            self.covars - self.w,
            sum_resps_old / self.sum_resps,
            m_old,
            self.means
        ) + step_size * self._adjust(
            batch_covars,
            sum_resps / self.sum_resps,
            batch_means,
            self.means
        ) + self.w

        self.weights = self.sum_resps / self.batch_size

    def score_batch(self, dataset):
        log_prob = 0.0

        loader = data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

        for _, d in enumerate(loader):
            d = [a.to(self.device) for a in d]
            lp, _ = self._e_step(d)
            log_prob += lp.item()

        return log_prob
