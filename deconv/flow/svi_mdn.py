from torch import nn

from .mdn import MultivariateGaussianMDN
from .svi import SVIFlow

class SVIMDNFlow(SVIFlow):
    
    def _create_approximate_posterior(self):
        return MultivariateGaussianMDN(
            features=self.dimensions,
            context_features=self.context_size,
            hidden_features=self.context_size,
            hidden_net=nn.Linear(self.context_size, self.context_size),
            num_components=10,
            custom_initialization=False
        )
        