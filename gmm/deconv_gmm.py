import torch
import torch.distributions as dist

mvn = dist.multivariate_normal.MultivariateNormal


class DeconvGMM:

    def __init__(self, components, dimensions, max_iters=100,
                 chol_factor=1e-6, tol=1e-9):
        self.k = components
        self.d = dimensions
        self.max_iters = max_iters
        self.chol_factor = chol_factor
        self.tol = tol

        self.weights = torch.empty(self.k)
        self.means = torch.empty(self.k, self.d)
        self.covars = torch.empty(self.k, self.d, self.d)
        self.chol_covars = torch.empty(self.k, self.d, self.d)

    def _kmeans_init(self, X, max_iters=50, tol=1e-9):

        n = X.shape[0]

        x_min = torch.min(X, dim=0)[0]
        x_max = torch.max(X, dim=0)[0]

        resp = torch.zeros(n, self.k, dtype=torch.bool)
        idx = torch.arange(n)

        centroids = torch.rand(self.k, self.d) * (x_max - x_min) + x_min

        prev_distance = torch.tensor(float('inf'))

        for i in range(max_iters):
            distances = (X[:, None, :] - centroids[None, :, :]).norm(dim=2)
            labels = distances.min(dim=1)[1]
            for j in range(self.k):
                centroids[j, :] = X[labels == j, :].mean(dim=0)
            resp[:] = False
            resp[idx, labels] = True
            total_distance = distances[resp].sum()

            if torch.abs(total_distance - prev_distance) < tol:
                break
            prev_distance = total_distance

        return resp.float()

    def fit(self, X, noise_covars):

        n = X.shape[0]

        resps = self._kmeans_init(X)
        log_resps = torch.log(resps)

        cond_means = X[:, None, :].repeat(1, self.k, 1)
        cond_covars = torch.eye(self.d).repeat(n, self.k, 1, 1)

        self._m_step(cond_means, cond_covars, log_resps)

        prev_log_prob = torch.tensor(float('-inf'))

        for i in range(self.max_iters):
            log_prob, log_resps, cond_means, cond_covars = self._e_step(
                X, noise_covars
            )
            self._m_step(cond_means, cond_covars, log_resps)

            if torch.abs(log_prob - prev_log_prob) < self.tol:
                break

            prev_log_prob = log_prob

            print(i)

    def predict(self, X):
        return torch.exp(self._e_step(X)[1])

    def _e_step(self, X, noise_covars):

        n = X.shape[0]
        log_resps = torch.empty(n, self.k)

        T = self.covars[None, :, :, :] + noise_covars[:, None, :, :]
        T_chol = torch.cholesky(T)
        T_inv = torch.cholesky_solve(torch.eye(self.d), T_chol)

        for j in range(self.k):
            log_resps[:, j] = mvn(
                loc=self.means[None, j, :],
                scale_tril=T_chol[:, j, :, :]
            ).log_prob(X)
            log_resps[:, j] += torch.log(self.weights[j])

        diff = X[:, None, :] - self.means

        cond_means = self.means + torch.matmul(  # n, j, d
            self.covars[None, :, :, :],     # 1, j, d, d
            torch.matmul(                   # n, j, d, 1
                T_inv,                      # n, j, d, d
                diff[:, :, :, None]         # n, j, d, 1
            )
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
        return torch.sum(log_prob), log_resps, cond_means, cond_covars

    def _m_step(self, cond_means, cond_covars, log_resps):
        n = cond_means.shape[0]
        resps = torch.exp(log_resps)[:, :, None]    # n, j, 1
        weights = resps.sum(dim=0)  # j, 1
        self.means = (resps * cond_means).sum(dim=0) / weights  # j, d

        for j in range(self.k):
            diffs = self.means - cond_means    # n, j, d
            outer_p = torch.matmul(     # n, d, d
                torch.transpose(diffs[:, j, None, :], 1, 2),    # n, d, 1
                diffs[:, j, None, :]    # n, 1, d
            )
            self.covars[j, :, :] = torch.sum(   # d, d
                resps[:, j, :, None] * (    # n, 1, 1
                    cond_covars[:, j, :, :] +   # n, d, d
                    outer_p     # n, d, d
                ),
                dim=0
            ) / weights[j, :]

        self.weights = weights / n
