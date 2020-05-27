import torch
import torch.distributions as dist
import torch.utils.data as data_utils

from .sgd_gmm_tim import SGDGMMModule, BaseSGDGMM

from ..utils.sampling import minibatch_sample

mvn = dist.multivariate_normal.MultivariateNormal


class SGDDeconvGMMModule(SGDGMMModule):

    def forward(self, data):
        x, noise_covars = data

        weights = self.soft_max(self.soft_weights)

        T = self.covars[None, :, :, :] + noise_covars[:, None, :, :]

        log_resp = mvn(loc=self.means, covariance_matrix=T).log_prob(
            x[:, None, :]
        )
        log_resp += torch.log(weights)

        log_prob = torch.logsumexp(log_resp, dim=1)

        return log_prob


class SGDDeconvDataset(data_utils.Dataset):

    def __init__(self, X, noise_covars):
        self.X = X
        self.noise_covars = noise_covars

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (self.X[i, :], self.noise_covars[i, :, :])


class SGDDeconvGMM(BaseSGDGMM):

    def __init__(self, components, dimensions, epochs=10000, lr=1e-3,
                 batch_size=64, tol=1e-6, w=1e-3,
                 k_means_factor=100, k_means_iters=10, lr_step=5,
                 lr_gamma=0.1, device=None):
        self.module = SGDDeconvGMMModule(components, dimensions, w, device)
        super().__init__(
            components, dimensions, epochs=epochs, lr=lr,
            batch_size=batch_size, w=w, tol=tol,
            k_means_factor=k_means_factor, k_means_iters=k_means_iters,
            lr_step=lr_step, lr_gamma=lr_gamma, device=device
        )

    def _sample_prior(self, num_samples, context=None):

        weights = self.module.soft_max(self.module.soft_weights)
        idx = dist.Categorical(probs=weights).sample([num_samples])
        X = dist.MultivariateNormal(
            loc=self.module.means, scale_tril=self.module.L).sample([num_samples])

        return X[
            torch.arange(num_samples, device=self.device),
            idx,
            :
        ]

    def sample_prior(self, num_samples, device=torch.device('cpu')):
        with torch.no_grad():
            return minibatch_sample(
                self._sample_prior,
                num_samples,
                self.d,
                self.batch_size,
                device
            )

    def log_prob_prior(self, x, device=torch.device('cpu')):
        with torch.no_grad():
            weights = self.module.soft_max(self.module.soft_weights)
            means = self.module.means
            scale_tril = self.module.L
            print(weights.shape)
            print(means.shape)
            print(scale_tril.shape)
            print(x.shape)

            logpdfs = torch.zeros((x.shape[0], self.k))
            for i in range(self.k):
                disti = mvn(loc=means[i], scale_tril=scale_tril[i])
                logpdfs[:, i] = disti.log_prob(x) + torch.log(weights[i])

            return torch.logsumexp(logpdfs, dim=1)

    def _sample_posterior(self, x, num_samples, context=None):
        log_weights = torch.log(self.module.soft_max(self.module.soft_weights))
        T = self.module.covars[None, :, :, :] + x[1][:, None, :, :]

        p_weights = log_weights + dist.MultivariateNormal(
            loc=self.module.means, covariance_matrix=T
        ).log_prob(x[0][:, None, :])
        p_weights -= torch.logsumexp(p_weights, axis=1)[:, None]

        L_t = torch.cholesky(T)
        T_inv = torch.cholesky_solve(
            torch.eye(self.d, device=self.device), L_t)

        diff = x[0][:, None, :] - self.module.means
        T_prod = torch.matmul(T_inv, diff[:, :, :, None])
        p_means = self.module.means + torch.matmul(
            self.module.covars,
            T_prod
        ).squeeze()

        p_covars = self.module.covars - torch.matmul(
            self.module.covars,
            torch.matmul(T_inv, self.module.covars)
        )

        idx = dist.Categorical(logits=p_weights).sample([num_samples])
        samples = dist.MultivariateNormal(
            loc=p_means, covariance_matrix=p_covars).sample([num_samples])

        return samples.transpose(0, 1)[
            torch.arange(len(x), device=self.device)[:, None, None, None],
            torch.arange(num_samples, device=self.device)[None, :, None, None],
            idx.T[:, :, None, None],
            torch.arange(self.d, device=self.device)[None, None, None, :]
        ].squeeze()

    def sample_posterior(self, x, num_samples, device=torch.device('cpu')):
        with torch.no_grad():
            return minibatch_sample(
                self._sample_posterior,
                num_samples,
                self.d,
                self.batch_size,
                device,
                x=x
            )
