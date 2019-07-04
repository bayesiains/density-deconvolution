import torch
import torch.distributions as dist

mvn = dist.multivariate_normal.MultivariateNormal


class BasicGMM:

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

    def fit(self, X):

        resps = self._kmeans_init(X)
        log_resps = torch.log(resps)
        self._m_step(X, log_resps)

        prev_log_prob = torch.tensor(float('-inf'))

        for i in range(self.max_iters):
            log_prob, log_resps = self._e_step(X)
            self._m_step(X, log_resps)

            if torch.abs(log_prob - prev_log_prob) < self.tol:
                break

            prev_log_prob = log_prob

    def predict(self, X):
        return torch.exp(self._e_step(X)[1])

    def _e_step(self, X):

        n = X.shape[0]
        log_resps = torch.empty(n, self.k)

        for j in range(self.k):
            log_resps[:, j] = mvn(
                loc=self.means[j, :],
                scale_tril=self.chol_covars[j, :, :]
            ).log_prob(X)
            log_resps += torch.log(self.weights[:, j])

        log_prob = torch.logsumexp(log_resps, dim=1, keepdim=True)
        log_resps -= log_prob

        return torch.sum(log_prob), log_resps

    def _m_step(self, X, log_resps):
        n = X.shape[0]
        resps = torch.exp(log_resps)
        weights = torch.sum(resps, dim=0, keepdim=True)
        self.means = torch.mm(torch.t(resps), X) / torch.t(weights)

        for j in range(self.k):
            diff = X - self.means[j, :]
            self.covars[j, :, :] = torch.mm(
                resps[:, j] * torch.t(diff),
                diff
            ) / weights[:, j]
            self.covars[j, :, :] += torch.diag(
                self.chol_factor * torch.ones(self.d)
            )
            self.chol_covars[j] = torch.cholesky(self.covars[j])

        self.weights = weights / n
