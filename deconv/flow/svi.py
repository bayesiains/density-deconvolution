import torch
import torch.nn as nn 
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_

from nflows import flows, transforms
from nflows.distributions import ConditionalDiagonalNormal, StandardNormal

from .distributions import DeconvGaussian
from .maf import MAFlow
from .nn import DeconvInputEncoder
from .vae import VariationalAutoencoder
from .mdn import MultivariateGaussianDiagonalMDN


from ..utils.sampling import minibatch_sample

class SVIFlowToy(nn.Module): 
    def __init__(self, 
                 dimensions, 
                 objective,
                 flow_steps_prior,
                 flow_steps_posterior,
                 n_posterior_flows,
                 posterior_mdn,
                 warmup_posterior_flow_diversity,
                 warmup_kl,
                 kl_init,
                 posterior_context_size,
                 batch_size,
                 device,
                 act_fun=nn.functional.relu):
    
        super(SVIFlowToy, self).__init__()

        self.dimensions = dimensions
        self.objective = objective
        self.flow_steps_prior = flow_steps_prior
        self.flow_steps_posterior = flow_steps_posterior
        self.n_posterior_flows = n_posterior_flows
        self.posterior_mdn = posterior_mdn
        self.warmup_posterior_flow_diversity = warmup_posterior_flow_diversity
        self.warmup_kl = warmup_kl
        self.kl_init = kl_init
        self.posterior_context_size = posterior_context_size 
        self.batch_size = batch_size
        self.device = device
        self.act_fun = act_fun

        self.model = VariationalAutoencoder(prior=self._create_prior(),
                                            approximate_posterior=self._create_approximate_posterior(),
                                            likelihood=self._create_likelihood(),
                                            inputs_encoder=self._create_input_encoder()).to(device)

    def _create_approximate_posterior(self):
        idx = torch.tril_indices(self.dimensions, self.dimensions)
        self.idx = (idx[0], idx[1])

        input_size = int(self.dimensions + self.dimensions * (self.dimensions + 1) / 2)

        mdn_hidden = nn.ModuleList()
        mdn_hidden.append(nn.Linear(input_size, self.posterior_mdn[0]))

        for h0, h1 in zip(self.posterior_mdn[:-1], self.posterior_mdn[1:]):
            mdn_hidden.append(nn.Linear(h0, h1))

        print(mdn_hidden)

        self.diagonal_mdn = MultivariateGaussianDiagonalMDN(self.dimensions,
                                                            input_size,
                                                            self.posterior_mdn[-1],
                                                            mdn_hidden,
                                                            self.n_posterior_flows,
                                                            self.act_fun)

        posterior_transform = self._create_transform(self.flow_steps_posterior, self.posterior_context_size)

        return flows.Flow(transforms.InverseTransform(posterior_transform),
                          self.diagonal_mdn)

    def _create_prior(self):
        self.transform = self._create_transform(self.flow_steps_prior)
        distribution = StandardNormal((self.dimensions,))

        return flows.Flow(self.transform,
                          distribution)

    def _create_likelihood(self):
        return DeconvGaussian()

    def _create_input_encoder(self):
        def input_encoder(data):
            w, noise_covar = data
            x = torch.cat((w, noise_covar[:, self.idx[0], self.idx[1]]), dim=1)
            return self.diagonal_mdn.get_context(x)

        return input_encoder

    def _create_linear_transform(self):
        return transforms.CompositeTransform([transforms.RandomPermutation(features=self.dimensions),
                                              transforms.LULinear(self.dimensions, identity_init=True)])

    def _create_transform(self, flow_steps, context_features=None):
        return transforms.CompositeTransform([transforms.CompositeTransform([self._create_linear_transform(),
                                                                             self._create_maf_transform(context_features)]) for i in range(flow_steps)] + 
                                             [self._create_linear_transform()])

    def _create_maf_transform(self, context_features=None): #fix this with parameters
        return transforms.MaskedAffineAutoregressiveTransform(features=self.dimensions,
                                                              hidden_features=256,
                                                              context_features=context_features,
                                                              num_blocks=2,
                                                              use_residual_blocks=True,
                                                              random_mask=False,
                                                              activation=torch.nn.functional.relu,
                                                              dropout_probability=0.0,
                                                              use_batch_norm=False)

    def score(self, data):

        self.model.eval()
        if self.objective == 'iwae':
            return self.model.log_prob_lower_bound(data, num_samples=10)

        elif self.objective == 'elbo':
            return self.model.stochastic_elbo(data)
        


class SVIFlow(MAFlow):

    def __init__(self, dimensions, flow_steps, lr, epochs, context_size=64,
                 batch_size=256, kl_warmup=0.2, kl_init_factor=0.5,
                 device=None):
        super().__init__(
            dimensions, flow_steps, lr, epochs, batch_size, device
        )
        self.context_size = context_size
        self.kl_warmup = kl_warmup
        self.kl_init_factor = kl_init_factor

        self.model = VariationalAutoencoder(
            prior=self._create_prior(),
            approximate_posterior=self._create_approximate_posterior(),
            likelihood=self._create_likelihood(),
            inputs_encoder=self._create_input_encoder()
        )

        self.model.to(self.device)

    def _create_prior(self):
        self.transform = self._create_transform()
        distribution = StandardNormal((self.dimensions,))
        return flows.Flow(
            self.transform,
            distribution
        )

    def _create_likelihood(self):
        return DeconvGaussian()

    def _create_input_encoder(self):
        return DeconvInputEncoder(self.dimensions, self.context_size)

    def _create_approximate_posterior(self):

        # context_encoder = torch.nn.Linear(
        #     self.context_size,
        #     2 * self.dimensions
        # )

        # distribution = ConditionalDiagonalNormal(
        #     shape=(self.dimensions,),
        #     context_encoder=context_encoder
        # )

        distribution = StandardNormal((self.dimensions,))

        posterior_transform = self._create_transform(self.context_size)

        return flows.Flow(
            transforms.InverseTransform(
                posterior_transform
            ),
            distribution
        )

    def _kl_factor(self, step, max_steps):

        f = min(
            1.0,
            self.kl_init_factor + (
                (1 - self.kl_init_factor) * step / (
                    self.kl_warmup * max_steps
                )
            )
        )
        return f


    def fit(self, data, val_data=None):

        optimiser = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr
        )

        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        batches = len(loader)

        max_steps = self.epochs * batches

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimiser,
        #    max_steps
        # )
        scheduler=None

        for i in range(self.epochs):

            self.model.train()

            train_loss = 0.0

            for j, d in enumerate(loader):

                d = [a.to(self.device) for a in d]

                optimiser.zero_grad()

                step = i * batches + j

                # d[1] = torch.cholesky(d[1])

                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                elbo = self.model.log_prob_lower_bound(
                    d,
                    num_samples=50
                )
                torch.set_default_tensor_type(torch.FloatTensor)

                train_loss += torch.sum(elbo).item()
                loss = -1 * torch.mean(elbo)
                loss.backward()
                optimiser.step()
                
                if scheduler:
                    scheduler.step()

            train_loss /= len(data)

            if val_data:
                val_loss = self.score_batch(val_data) / len(val_data)
                print('Epoch {}, Train Loss: {}, Val Loss: {}'.format(
                    i,
                    train_loss,
                    val_loss
                ))
            else:
                print('Epoch {}, Train Loss: {}'.format(i, train_loss))


    def score(self, data, log_prob=False):
        with torch.no_grad():
            self.model.eval()
            # data[1] = torch.cholesky(data[1])
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            if log_prob:
                return self.model.log_prob_lower_bound(data, num_samples=100)
            else:
                return self.model.stochastic_elbo(data)
            torch.set_default_tensor_type(torch.FloatTensor)

    def score_batch(self, dataset, log_prob=False):
        loader = data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
        score = 0.0

        for j, d in enumerate(loader):
            d = [a.to(self.device) for a in d]
            score += torch.sum(self.score(d, log_prob)).item()

        return score

    def sample_prior(self, num_samples, device=torch.device('cpu')):
        return minibatch_sample(
            self.model._prior.sample,
            num_samples,
            self.batch_size,
            device
        )
        
       
            
    
