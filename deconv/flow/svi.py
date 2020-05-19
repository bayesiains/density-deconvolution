import torch
import torch.nn as nn 
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_

from nflows import flows, transforms, utils
from nflows.distributions import ConditionalDiagonalNormal, StandardNormal

from .distributions import DeconvGaussian, DeconvGaussianToy, DeconvGaussianToyNoise
from .maf import MAFlow
from .nn import DeconvInputEncoder
from .vae import VariationalAutoencoder, VariationalAutoencoderToyNoise
from .mdn import MultivariateGaussianDiagonalMDN


from ..utils.sampling import minibatch_sample

class SVIFlowToyNoise(nn.Module):
    def __init__(self, 
                 dimensions, 
                 objective,
                 posterior_context_size,
                 batch_size,
                 device,
                 maf_steps_prior,
                 maf_steps_posterior,
                 maf_features,
                 maf_hidden_blocks,
                 K=1,
                 posterior_mdn_net=None,
                 posterior_mdn_components=None,
                 act_fun=nn.functional.relu):
    
        super(SVIFlowToyNoise, self).__init__()

        self.dimensions = dimensions
        self.objective = objective
        self.posterior_context_size = posterior_context_size 
        self.batch_size = batch_size
        self.device = device
        self.maf_steps_prior = maf_steps_prior
        self.maf_steps_posterior = maf_steps_posterior
        self.maf_features = maf_features
        self.maf_hidden_blocks = maf_hidden_blocks
        self.K = K
        self.posterior_mdn_net = posterior_mdn_net
        self.posterior_mdn_components = self.psoterior_mdn_components
        self.act_fun = act_fun

        self.model = VariationalAutoencoderToyNoise(prior=self._create_prior(),
                                                    approximate_posterior=self._create_approximate_posterior(),
                                                    likelihood=self._create_likelihood(),
                                                    inputs_encoder=self._create_input_encoder()).to(device)

    def _create_approximate_posterior(self):
        if self.posterior_mdn_net is not None:
            mdn_hidden = nn.ModuleList()
            mdn_hidden.append(nn.Linear(self.dimensions, self.posterior_mdn_net[0]))

            for h0, h1 in zip(self.posterior_mdn_net[:-1], self.posterior_mdn_net[1:]):
                mdn_hidden.append(nn.Linear(h0, h1))

            distribution = MultivariateGaussianDiagonalMDN(self.dimensions,
                                                           self.dimensions,
                                                           self.posterior_mdn_net[-1],
                                                           mdn_hidden,
                                                           self.psoterior_mdn_components,
                                                           self.act_fun)
            self.diagonal_mdn = distribution

        else:
            distribution = StandardNormal((self.dimensions,))
            self.diagonal_mdn = None

        posterior_transform = self._create_transform(self.maf_steps_posterior, self.posterior_context_size)

        return flows.Flow(transforms.InverseTransform(posterior_transform),
                          distribution)

    def _create_prior(self):
        return DeconvGaussianToyNoise()

    def _create_likelihood(self):
        self.transform = self._create_transform(self.maf_steps_prior)
        distribution = StandardNormal((self.dimensions,))

        return flows.Flow(self.transform,
                          distribution)

    def _create_input_encoder(self):
        def input_encoder(data):
            w, _ = data

            if self.diagonal_mdn is not None:
                return self.diagonal_mdn.get_context(w)

            else:
                return w

        return input_encoder

    def _create_linear_transform(self):
        return transforms.CompositeTransform([transforms.RandomPermutation(features=self.dimensions),
                                              transforms.LULinear(self.dimensions, identity_init=True)])

    def _create_transform(self, flow_steps, context_features=None):
        return transforms.CompositeTransform([transforms.CompositeTransform([self._create_linear_transform(),
                                                                             self._create_maf_transform(context_features)]) for i in range(flow_steps)] + 
                                             [self._create_linear_transform()])

    def _create_maf_transform(self, context_features=None): 
        return transforms.MaskedAffineAutoregressiveTransform(features=self.dimensions,
                                                              hidden_features=self.maf_features,
                                                              context_features=context_features,
                                                              num_blocks=self.maf_hidden_blocks,
                                                              use_residual_blocks=False,
                                                              random_mask=False,
                                                              activation=torch.nn.functional.relu,
                                                              dropout_probability=0.0,
                                                              use_batch_norm=False)

    def score(self, data):
        self.model.eval()
        if self.objective == 'iwae':
            return self.model.log_prob_lower_bound(data, num_samples=self.K)

        elif self.objective == 'elbo':
            return self.model.stochastic_elbo(data, num_samples=self.K)

class SVIFlowToy(nn.Module): 
    def __init__(self, 
                 dimensions, 
                 objective,
                 posterior_context_size,
                 batch_size,
                 device,
                 maf_steps_prior,
                 maf_steps_posterior,
                 maf_features,
                 maf_hidden_blocks,
                 K=1,
                 posterior_mdn_net=None,
                 posterior_mdn_components=None,
                 act_fun=nn.functional.relu):
    
        super(SVIFlowToy, self).__init__()

        self.dimensions = dimensions
        self.objective = objective
        self.posterior_context_size = posterior_context_size 
        self.batch_size = batch_size
        self.device = device
        self.maf_steps_prior = maf_steps_prior
        self.maf_steps_posterior = maf_steps_posterior
        self.maf_features = maf_features
        self.maf_hidden_blocks = maf_hidden_blocks
        self.K = K
        self.posterior_mdn_net = posterior_mdn_net
        self.posterior_mdn_components = posterior_mdn_components
        self.act_fun = act_fun

        self.model = VariationalAutoencoder(prior=self._create_prior(),
                                                approximate_posterior=self._create_approximate_posterior(),
                                                    likelihood=self._create_likelihood(),
                                                    inputs_encoder=self._create_input_encoder()).to(device)

    def _create_approximate_posterior(self):
        if self.posterior_mdn_net is not None:
            mdn_hidden = nn.ModuleList()
            mdn_hidden.append(nn.Linear(self.dimensions, self.posterior_mdn_net[0]))

            for h0, h1 in zip(self.posterior_mdn_net[:-1], self.posterior_mdn_net[1:]):
                mdn_hidden.append(nn.Linear(h0, h1))

            distribution = MultivariateGaussianDiagonalMDN(self.dimensions,
                                                           self.dimensions,
                                                           self.posterior_mdn_net[-1],
                                                           mdn_hidden,
                                                           self.posterior_mdn_components,
                                                           self.act_fun)
            self.diagonal_mdn = distribution

        else:
            distribution = StandardNormal((self.dimensions,))
            self.diagonal_mdn = None

        posterior_transform = self._create_transform(self.maf_steps_posterior, self.posterior_context_size)

        return flows.Flow(transforms.InverseTransform(posterior_transform),
                          distribution)

    def _create_prior(self):
        self.transform = self._create_transform(self.maf_steps_prior)
        distribution = StandardNormal((self.dimensions,))

        return flows.Flow(self.transform,
                          distribution)

    def _create_likelihood(self):
        return DeconvGaussianToy()

    def _create_input_encoder(self):
        def input_encoder(data):
            w, _ = data

            if self.diagonal_mdn is not None:
                return self.diagonal_mdn.get_context(w)

            else:
                return w

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
                                                              hidden_features=self.maf_features,
                                                              context_features=context_features,
                                                              num_blocks=self.maf_hidden_blocks,
                                                              use_residual_blocks=False,
                                                              random_mask=False,
                                                              activation=torch.nn.functional.relu,
                                                              dropout_probability=0.0,
                                                              use_batch_norm=False)

    def score(self, data):
        self.model.eval()
        if self.objective == 'iwae':
            return self.model.log_prob_lower_bound(data, num_samples=self.K)

        elif self.objective == 'elbo':
            return self.model.stochastic_elbo(data, num_samples=self.K)
        


class SVIFlow(MAFlow):

    def __init__(self, dimensions, flow_steps, lr, epochs, context_size=64,
                 batch_size=256, kl_warmup=0.2, kl_init_factor=0.5,
                 n_samples=50, use_iwae=False, device=None):
        super().__init__(
            dimensions, flow_steps, lr, epochs, batch_size, device
        )
        self.context_size = context_size
        self.kl_warmup = kl_warmup
        self.kl_init_factor = kl_init_factor
        
        self.n_samples = n_samples
        self.use_iwae = use_iwae

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
                
                if self.use_iwae:
                    objective = self.model.log_prob_lower_bound(
                        d,
                        num_samples=self.n_samples
                    )
                else:
                    objective = self.model.stochastic_elbo(
                        d,
                        num_samples=self.n_samples,
                        keepdim=True
                    )
                torch.set_default_tensor_type(torch.FloatTensor)

                train_loss += torch.sum(objective).item()
                loss = -1 * torch.mean(objective)
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
        with torch.no_grad():
            self.model.eval()
            return minibatch_sample(
                self.model._prior.sample,
                num_samples,
                self.dimensions,
                self.batch_size,
                device
            )
        
    def sample_posterior(self, x, num_samples, device=torch.device('cpu')):
        with torch.no_grad():
            self.model.eval()
            context = self.model._inputs_encoder(x)
            return minibatch_sample(
                self.model._approximate_posterior.sample,
                num_samples,
                self.dimensions,
                self.batch_size,
                device,
                context=context
            )
        
    def _resample_posterior(self, x, num_samples, context):
        
        samples, log_q_z = self.model._approximate_posterior.sample_and_log_prob(
            num_samples,
            context=context
        )
        samples = utils.merge_leading_dims(samples, num_dims=2)
        log_q_z = utils.merge_leading_dims(log_q_z, num_dims=2)

        # Compute log prob of latents under the prior.
        log_p_z = self.model._prior.log_prob(samples)

        # Compute log prob of inputs under the decoder,
        x = utils.repeat_rows(x, num_reps=num_samples)
        log_p_x = self.model._likelihood.log_prob(x, context=samples)

        # Compute ELBO.
        log_w = log_p_x + log_p_z - log_q_z
        log_w = utils.split_leading_dim(log_w, [-1, num_samples])
        log_w -= torch.logsumexp(log_w, dim=-1)[:, None]
        
        samples = utils.split_leading_dim(samples, [-1, num_samples])
        idx = torch.distributions.Categorical(logits=log_w).sample([num_samples])
        
        return samples[
            torch.arange(len(x), device=self.device)[:, None, None],
            idx.T[:, :, None],
            torch.arange(self.dimensions, device=self.device)[None, None, :]
        ]
    
    def resample_posterior(self, x, num_samples, device=torch.device('cpu')):
        with torch.no_grad():
            self.model.eval()
            context = self.model._inputs_encoder(x)
            
            return minibatch_sample(
                self._resample_posterior,
                num_samples,
                self.dimensions,
                self.batch_size,
                device,
                context=context,
                x=x
            )
        
        
        
       
            
    