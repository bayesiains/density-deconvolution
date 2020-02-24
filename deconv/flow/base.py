from abc import ABC, abstractmethod

import torch
import torch.utils.data as data_utils
from nsflow import nde
from nsflow.nde.flows import Flow

class BaseFlow(ABC):
    """ABC for flow-type density estimation."""

    def __init__(self, dimensions, flow_steps, lr, epochs, batch_size=256, device=None):

        self.dimensions = dimensions
        self.flow_steps = flow_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.lr = lr   
        transform = self._create_transform()

        self.flow = nde.flows.Flow(
            transform,
            nde.distributions.StandardNormal((self.dimensions,))
        )

        self.flow.to(self.device)

    def _create_linear_transform(self):
        return nde.transforms.CompositeTransform([
            nde.transforms.RandomPermutation(features=self.dimensions),
            nde.transforms.LULinear(self.dimensions, identity_init=True)
        ])

    @abstractmethod
    def _create_transform(context_features=None):
        pass

    def fit(self, data, val_data=None):

        optimiser = torch.optim.Adam(
            params=self.flow.parameters(),
            lr=self.lr
        )

        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            # num_workers=8,
            shuffle=True,
            # pin_memory=True
        )

        for i in range(self.epochs):

            self.flow.train()

            train_loss = 0.0

            for j, d in enumerate(loader):
                d = [a.to(self.device) for a in d]
                optimiser.zero_grad()
                log_prob = self.flow.log_prob(d[0])
                loss = -1 * torch.mean(log_prob)
                train_loss += loss.item()
                loss.backward()
                optimiser.step()

            train_loss /= len(loader)

            if val_data:
                val_loss = self.score_batch(val_data)

                print('Epoch {}, Train Loss: {}, Val Loss :{}'.format(
                    i,
                    train_loss,
                    val_loss
                ))

    def score(self, data):
        with torch.no_grad():
            self.flow.eval()
            return self.flow.log_prob(data)

    def score_batch(self, dataset):
        loader = data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            # num_workers=4,
            # pin_memory=True
        )
        log_prob = 0.0

        for j, d in enumerate(loader):
            d = [a.to(self.device) for a in d]
            log_prob += torch.mean(self.score(d[0])).item()

        return log_prob / len(loader)
