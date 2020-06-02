import torch
from nflows import transforms

from .base import BaseFlow


class MAFlow(BaseFlow):

    def _create_maf_transform(self, context_features=None, hidden_features=128):
        return transforms.MaskedAffineAutoregressiveTransform(
            features=self.dimensions,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=2,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.nn.functional.relu,
            dropout_probability=0.2,
            use_batch_norm=False
        )

    def _create_transform(self, context_features=None, hidden_features=128):

        return transforms.CompositeTransform([

            transforms.CompositeTransform([
                self._create_linear_transform(),
                self._create_maf_transform(context_features, hidden_features)
            ]) for i in range(self.flow_steps)
        ] + [
            self._create_linear_transform()
        ])

