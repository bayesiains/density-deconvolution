import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.nn.nets import ResidualNet


class DeconvInputEncoder(nn.Module):

    def __init__(self, d, context_size):

        super().__init__()

        idx = torch.tril_indices(d, d)
        self.idx = (idx[0], idx[1])

        input_size = int(d + d * (d + 1) / 2)

        self.resnet = ResidualNet(
            in_features=input_size,
            out_features=context_size,
            hidden_features=256,
            context_features=None,
            num_blocks=3,
            activation=F.relu,
            dropout_probability=0.2,
            use_batch_norm=False
        )

    def forward(self, inputs):
        x, noise_l = inputs

        x = torch.cat((x, noise_l[:, self.idx[0], self.idx[1]]), dim=1)

        return self.resnet(x)

