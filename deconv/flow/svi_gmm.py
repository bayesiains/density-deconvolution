from deconv.gmm.sgd_gmm import SGDGMMModule

from .svi import SVIFlow


class SVIGMMFlow(SVIFlow):
    
    def _create_prior(self):
        return SGDGMMModule(
            4,
            self.dimensions,
            w=0,
            device=self.device
        )