from abc import ABC, abstractmethod

import torch
import torch.distributions as dist

mvn = dist.multivariate_normal.MultivariateNormal


class BaseGMM(ABC):

    def __init__(self, components, dimensions, max_iters=100,
                 chol_factor=1e-6, tol=1e-9, restarts=5):
        self.k = components
        self.d = dimensions
        self.max_iters = max_iters
        self.chol_factor = chol_factor
        self.tol = tol
        self.restarts = restarts

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

    def fit(self, data):

        best_log_prob = torch.tensor(float('-inf'))

        for j in range(self.restarts):
            expectations = self._init_expectations(data)
            self._m_step(data, expectations)

            prev_log_prob = torch.tensor(float('-inf'))

            for i in range(self.max_iters):
                log_prob, expectations = self._e_step(data)
                if log_prob == torch.tensor(float('-inf')):
                    break
                self._m_step(data, expectations)

                if torch.abs(log_prob - prev_log_prob) < self.tol:
                    break

                prev_log_prob = log_prob

            print(log_prob)

            if log_prob > best_log_prob:
                best_params = (
                    self.weights.clone().detach(),
                    self.means.clone().detach(),
                    self.covars.clone().detach(),
                    self.chol_covars.clone().detach()
                )
                best_log_prob = log_prob

        if best_log_prob == torch.tensor(float('-inf')):
            raise ValueError('Could not fit model. Try increasing chol_reg?')
        self.weights, self.means, self.covars, self.chol_covars = best_params
        print(best_log_prob)

    def predict(self, X):
        return torch.exp(self._e_step(X)[1])

    @abstractmethod
    def _init_expectations(self, data):
        pass

    @abstractmethod
    def _e_step(self, data):
        pass

    @abstractmethod
    def _m_step(self, data, expectations):
        pass
