import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvInputEncoder(nn.Module):

    def __init__(self, d, context_size):

        super().__init__()

        idx = torch.tril_indices(d, d)
        self.idx = (idx[0], idx[1])

        input_size = int(d + d * (d + 1) / 2)

        self.fc1 = nn.Linear(input_size, context_size)
        self.fc2 = nn.Linear(context_size, context_size)
        self.fc3 = nn.Linear(context_size, context_size)

    def forward(self, inputs):
        x, noise_l = inputs

        x = torch.cat((x, noise_l[:, self.idx[0], self.idx[1]]), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

