from torch.distributions import MultivariateNormal

from nflows import distributions

class DeconvGaussianToy(distributions.Distribution):

    def log_prob(self, inputs, context):

        X, noise = inputs

        return MultivariateNormal(loc=context, covariance_matrix=noise).log_prob(X)


class DeconvGaussian(distributions.Distribution):

    def log_prob(self, inputs, context):

        X, noise_l = inputs

        return MultivariateNormal(loc=context, scale_tril=noise_l).log_prob(X)


