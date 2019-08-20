from abc import ABC

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.utils.data as data_utils

from .util import minibatch_k_means

mvn = dist.multivariate_normal.MultivariateNormal


class SGDGMMModule(nn.Module):

    def __init__(self, components, dimensions, device=None):
        super().__init__()

        self.k = components
        self.d = dimensions
        self.device = device

        self.soft_weights = nn.Parameter(torch.zeros(self.k))
        self.soft_max = torch.nn.Softmax(dim=0)

        self.means = nn.Parameter(torch.rand(self.k, self.d))
        self.l_diag = nn.Parameter(torch.zeros(self.k, self.d))

        self.l_lower = nn.Parameter(
            torch.zeros(self.k, self.d * (self.d - 1) // 2)
        )

        self.d_idx = torch.eye(self.d, device=self.device).to(torch.bool)
        self.l_idx = torch.tril_indices(self.d, self.d, -1, device=self.device)

    @property
    def L(self):
        L = torch.zeros(self.k, self.d, self.d, device=self.device)
        L[:, self.d_idx] = torch.exp(self.l_diag)
        L[:, self.l_idx[0], self.l_idx[1]] = self.l_lower
        return L

    @property
    def covars(self):
        return torch.matmul(self.L, torch.transpose(self.L, -2, -1))

    def forward(self, data):

        x = data[0]

        weights = self.soft_max(self.soft_weights)

        log_resp = mvn(loc=self.means, scale_tril=self.L).log_prob(
            x[:, None, :]
        )
        log_resp += torch.log(weights)

        log_prob = torch.logsumexp(log_resp, dim=1)

        return -1 * torch.sum(log_prob)


class BaseSGDGMM(ABC):

    def __init__(self, components, dimensions, epochs=10000, lr=1e-3,
                 batch_size=64, tol=1e-6, device=None):
        self.k = components
        self.d = dimensions
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = 1e-6
        self.lr = lr

        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.module.to(device)

        self.optimiser = torch.optim.Adam(
            params=self.module.parameters(),
            lr=self.lr
        )

    @property
    def means(self):
        return self.module.means.detach()

    @property
    def covars(self):
        return self.module.covars.detach()

    def fit(self, data, verbose=False):

        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True
        )

        self.init_params(loader)

        prev_loss = torch.tensor(float('inf'))

        for i in range(self.epochs):
            running_loss = torch.zeros(1)
            for j, d in enumerate(loader):

                d = [a.to(self.device) for a in d]

                self.optimiser.zero_grad()

                loss = self.module(d)
                loss.backward()
                self.optimiser.step()

                running_loss += loss

            if verbose and i % 10 == 0:
                print('Epoch {}, Loss: {}'.format(i, running_loss.item()))
            if torch.abs(running_loss - prev_loss) < self.tol:
                break
            prev_loss = running_loss


class SGDGMM(BaseSGDGMM):

    def __init__(self, components, dimensions, epochs=10000, lr=1e-3,
                 batch_size=64, tol=1e-6, device=None):
        self.module = SGDGMMModule(components, dimensions, device)
        super().__init__(
            components, dimensions, epochs=epochs, lr=lr,
            batch_size=batch_size, tol=tol, device=device
        )

    def init_params(self, loader):
        _, centroids = minibatch_k_means(loader, self.k, device=self.device)
        self.module.means.data = centroids
