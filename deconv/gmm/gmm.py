import torch
import torch.distributions as dist

from .base import BaseGMM

mvn = dist.multivariate_normal.MultivariateNormal


class GMM(BaseGMM):

    def _init_expectations(self, data):
        X = data[0]
        resps = self._kmeans_init(X)
        return (torch.log(resps),)

    def predict(self, X):
        return torch.exp(self._e_step(X)[1][0])

    def _e_step(self, data):

        try:
            self.chol_covars = torch.cholesky(self.covars)
        except RuntimeError:
            return torch.tensor(float('-inf')), None

        X = data[0]

        n = X.shape[0]
        log_resps = torch.empty(n, self.k, device=self.device)

        for j in range(self.k):
            log_resps[:, j] = mvn(
                loc=self.means[j, :],
                scale_tril=self.chol_covars[j, :, :]
            ).log_prob(X)
            log_resps += torch.log(self.weights[:, j])

        log_prob = torch.logsumexp(log_resps, dim=1, keepdim=True)
        log_resps -= log_prob

        return torch.sum(log_prob), (log_resps,)

    def _m_step(self, data, expectations):
        X = data[0]
        n = X.shape[0]
        resps = torch.exp(expectations[0])
        weights = torch.sum(resps, dim=0, keepdim=True)
        self.means = torch.mm(torch.t(resps), X) / torch.t(weights)

        for j in range(self.k):
            diff = X - self.means[j, :]
            self.covars[j, :, :] = torch.mm(
                resps[:, j] * torch.t(diff),
                diff
            ) / weights[:, j]
            self.covars[j, :, :] += torch.diag(
                self.w * torch.ones(self.d, device=self.device)
            )

        self.weights = weights / n
