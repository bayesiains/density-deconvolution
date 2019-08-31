import torch
import torch.utils.data as data_utils

from .deconv_gmm import DeconvGMM
from .util import minibatch_k_means


class OnlineDeconvGMM(DeconvGMM):

    def __init__(self, components, dimensions, epochs=1000, w=1e-6,
                 tol=1e-6, step_size=0.1, batch_size=100, restarts=5,
                 max_no_improvement=20, k_means_factor=100,
                 k_means_iters=10, device=None):
        super().__init__(components, dimensions, epochs=epochs, w=w, tol=tol,
                         restarts=restarts, device=device)
        self.batch_size = batch_size
        self.step_size = step_size
        self.max_no_improvement = max_no_improvement
        self.k_means_factor = k_means_factor
        self.k_means_iters = k_means_iters

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

        self.sum_dev_ps = self.covars * self.sum_resps[:, :, None]

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

        best_ll = float('-inf')

        for j in range(self.restarts):

            train_ll_curve = []
            if val_data:
                val_ll_curve = []

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
                train_ll_curve.append(train_ll)

                if val_data:
                    val_ll = self.score_batch(val_data)
                    val_ll_curve.append(val_ll)

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

            if val_data:
                score = val_ll
            else:
                score = train_ll

            if score > best_ll and score != 0.0:
                best_params = (
                    self.weights.clone().detach(),
                    self.means.clone().detach(),
                    self.covars.clone().detach(),
                    self.chol_covars.clone().detach()
                )
                best_ll = score

                best_train_ll_curve = train_ll_curve
                if val_data:
                    best_val_ll_curve = val_ll_curve

        if best_ll == n_inf:
            raise ValueError('Could not fit model. Try increasing w?')

        self.weights, self.means, self.covars, self.chol_covars = best_params
        self.train_ll_curve = best_train_ll_curve
        if val_data:
            self.val_ll_curve = best_val_ll_curve

    def _m_step(self, expectations, n, step_size):
        log_resps, cond_means, cond_covars = expectations
        resps = torch.exp(log_resps)[:, :, None]    # n, j, 1

        sum_resps = resps.sum(dim=0)

        sum_cond_means = (resps * cond_means).sum(dim=0)

        diffs = cond_means - self.means

        outer_p = diffs[:, :, :, None] * diffs[:, :, None, :]
        outer_p += cond_covars

        sum_dev_ps = torch.sum(
            resps[:, :, :, None] * outer_p,
            dim=0
        )

        self.sum_resps = (1 - step_size) * self.sum_resps + step_size * sum_resps
        self.sum_cond_means = (1 - step_size) * self.sum_cond_means + step_size * sum_cond_means

        self.means = self.sum_cond_means / self.sum_resps

        self.sum_dev_ps = (1 - step_size) * self.sum_dev_ps + step_size * sum_dev_ps
        self.covars = self.sum_dev_ps / self.sum_resps[:, :, None] + self.w

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
