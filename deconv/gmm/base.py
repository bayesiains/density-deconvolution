from abc import ABC, abstractmethod

import torch
import torch.distributions as dist

from .util import k_means

mvn = dist.multivariate_normal.MultivariateNormal


class BaseGMM(ABC):

    def __init__(self, components, dimensions, epochs=100,
                 w=1e-6, tol=1e-9, restarts=5, device=None):
        self.k = components
        self.d = dimensions
        self.epochs = epochs
        self.w = w
        self.tol = tol
        self.restarts = restarts

        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.weights = torch.empty(self.k, device=self.device)
        self.means = torch.empty(self.k, self.d, device=self.device)
        self.covars = torch.empty(self.k, self.d, self.d, device=self.device)
        self.chol_covars = torch.empty(
            self.k, self.d, self.d, device=self.device
        )

    def _kmeans_init(self, X, max_iters=50, tol=1e-9):

        return k_means(X, self.k, max_iters, tol, self.device)[0]

    def fit(self, data):

        n_inf = torch.tensor(float('-inf'), device=self.device)

        best_log_prob = torch.tensor(float('-inf'), device=self.device)

        for j in range(self.restarts):
            expectations = self._init_expectations(data)
            self._m_step(data, expectations)

            prev_log_prob = torch.tensor(float('-inf'), device=self.device)

            for i in range(self.epochs):
                log_prob, expectations = self._e_step(data)
                if log_prob == n_inf:
                    break
                self._m_step(data, expectations)

                if torch.abs(log_prob - prev_log_prob) < self.tol:
                    break

                prev_log_prob = log_prob

            if log_prob > best_log_prob:
                best_params = (
                    self.weights.clone().detach(),
                    self.means.clone().detach(),
                    self.covars.clone().detach(),
                    self.chol_covars.clone().detach()
                )
                best_log_prob = log_prob

        if best_log_prob == n_inf:
            raise ValueError('Could not fit model. Try increasing w?')
        self.weights, self.means, self.covars, self.chol_covars = best_params
        print(self.weights.device)

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
