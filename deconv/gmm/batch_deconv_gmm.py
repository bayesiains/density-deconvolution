import torch
import torch.utils.data as data_utils

from .online_deconv_gmm import OnlineDeconvGMM


class BatchDeconvGMM(OnlineDeconvGMM):


    def fit(self, data, val_data=None, verbose=False, interval=1):
        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True
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
            train_ll = 0

            sum_resps = torch.zeros(self.k, 1)
            sum_cond_means = torch.zeros(self.k, self.d)
            sum_cond_covars = torch.zeros(self.k, self.d, self.d)


            # First pass for weights and means
            for d in loader:
                d = [a.to(self.device) for a in d]
                log_prob, (sum_r, sum_m) = self._step_1(d)

                train_ll += log_prob.item()

                sum_resps += sum_r
                sum_cond_means += sum_m

            self.train_ll_curve.append(train_ll)

            if val_data:
                val_ll = self.score_batch(val_data)
                self.val_ll_curve.append(val_ll)

            new_weights = sum_resps / n
            new_means = sum_cond_means / sum_resps

            # Second pass for covariances
            for d in loader:
                d = [a.to(self.device) for a in d]
                sum_cond_covars += self._step_2(d, new_means)

            self.weights = new_weights
            self.means = new_means
            self.covars = (sum_cond_covars + 2 * self.w) / sum_resps[:, :, None]

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

    def _step_1(self, data):
        log_prob, expectations = self._e_step(data)
        log_resps, cond_means, _ = expectations

        resps = torch.exp(log_resps)[:, :, None]

        sum_r = resps.sum(dim=0)
        sum_m = (resps * cond_means).sum(dim=0)

        return log_prob, (sum_r, sum_m)

    def _step_2(self, data, new_means):
        _, expectations = self._e_step(data)
        log_resps, cond_means, cond_covars = expectations

        resps = torch.exp(log_resps)[:, :, None]

        diffs = cond_means - new_means

        outer_p = diffs[:, :, :, None] * diffs[:, :, None, :]
        outer_p += cond_covars

        return torch.sum(
            resps[:, :, :, None] * outer_p,
            dim=0
        )
