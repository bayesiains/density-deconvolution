from torch.distributions import MultivariateNormal

from nsflow.nde import distributions


class DeconvGaussian(distributions.Distribution):

    def log_prob(self, inputs, context):

        X, noise_l = inputs

        return MultivariateNormal(loc=context, scale_tril=noise_l).log_prob(X)
