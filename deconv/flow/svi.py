import torch
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_

from nflows import flows, transforms, utils
from nflows.distributions import ConditionalDiagonalNormal, StandardNormal

from .distributions import DeconvGaussian
from .maf import MAFlow
from .nn import DeconvInputEncoder
from .vae import VariationalAutoencoder


from ..utils.sampling import minibatch_sample

class SVIFlow(MAFlow):

    def __init__(self, dimensions, flow_steps, lr, epochs, context_size=64, hidden_features=128,
                 batch_size=256, kl_warmup=0.2, kl_init_factor=0.5,
                 n_samples=50, use_iwae=False, device=None):
        super().__init__(
            dimensions, flow_steps, lr, epochs, batch_size, device
        )
        self.context_size = context_size
        self.hidden_features = hidden_features
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
        self.transform = self._create_transform(context_features=None, hidden_features=self.hidden_features)
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

        distribution = StandardNormal((self.dimensions,))

        posterior_transform = self._create_transform(self.context_size, hidden_features=self.hidden_features)

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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode='max',
            factor=0.8,
            patience=20,
            verbose=True,
            threshold=1e-6
        )
        
        for i in range(self.epochs):

            self.model.train()

            train_loss = 0.0

            for j, d in enumerate(loader):

                d = [a.to(self.device) for a in d]

                optimiser.zero_grad()

                torch.set_default_tensor_type(torch.cuda.FloatTensor)
                
                if self.use_iwae and (i >= 5):
                    objective = self.model.log_prob_lower_bound(
                        d,
                        num_samples=self.n_samples
                    )
                else:
                    objective = self.model.stochastic_elbo(
                        d,
                        num_samples=self.n_samples
                    )
                torch.set_default_tensor_type(torch.FloatTensor)

                train_loss += torch.sum(objective).item()
                loss = -1 * torch.mean(objective)
                loss.backward()
                optimiser.step()
                
            train_loss /= len(data)
            
            if val_data:
                val_loss = self.score_batch(
                    val_data,
                    log_prob=self.use_iwae,
                    num_samples=self.n_samples
                ) / len(val_data)
                print('Epoch {}, Train Loss: {}, Val Loss: {}'.format(
                    i,
                    train_loss,
                    val_loss
                ))
                scheduler.step(val_loss)
            else:
                print('Epoch {}, Train Loss: {}'.format(i, train_loss))
                scheduler.step(train_loss)

    def score(self, data, log_prob=False, num_samples=None):
        
        if not num_samples:
            num_samples = self.n_samples

        with torch.no_grad():
            self.model.eval()

            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            if log_prob:
                return self.model.log_prob_lower_bound(data, num_samples=num_samples)
            else:
                return self.model.stochastic_elbo(data, num_samples=num_samples)
            torch.set_default_tensor_type(torch.FloatTensor)

    def score_batch(self, dataset, log_prob=False, num_samples=None):
        loader = data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
        score = 0.0

        for j, d in enumerate(loader):
            d = [a.to(self.device) for a in d]
            score += torch.sum(self.score(d, log_prob, num_samples)).item()

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
        
        
        
       
            
    