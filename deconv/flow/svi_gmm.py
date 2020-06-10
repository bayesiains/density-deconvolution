import copy

import torch

from deconv.gmm.sgd_gmm import SGDGMMModule
from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM

from .svi import SVIFlow


class SVIGMMFlow(SVIFlow):
    
    def _create_prior(self):
        return SGDGMMModule(
            3,
            self.dimensions,
            w=0,
            device=self.device
        )
    
class GMMPosterior():
    
    def __init__(self, gmm):
        self.gmm = gmm
        self.module = gmm.module
        
    def sample_and_log_prob(self, num_samples, context):
        x, L = context
        cov = torch.matmul(L, L.transpose(-1, -2))
        samples = self.gmm._sample_posterior((x, cov), num_samples)
        log_prob = self.gmm.posterior_log_prob(samples, (x, cov))
        return samples, log_prob
        
    
class SVIGMMExact(SVIFlow):
    
    def _create_prior(self):
        self.gmm = SGDDeconvGMM(
            3,
            self.dimensions,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device
        )
        self.gmm.module = SGDGMMModule(
            3,
            self.dimensions,
            w=0,
            device=self.device
        )
        return self.gmm.module
        
    def _create_input_encoder(self):
        return None
    
    def _create_approximate_posterior(self):
        return GMMPosterior(self.gmm)
        
        
    
    