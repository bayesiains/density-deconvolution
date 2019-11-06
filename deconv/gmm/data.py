import torch
import torch.utils.data as data_utils

import h5py


class DeconvDataset(data_utils.Dataset):

    def __init__(self, X, noise_covars):
        self.X = X
        self.noise_covars = noise_covars

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (self.X[i, :], self.noise_covars[i, :, :])


class H5DeconvDataset(data_utils.Dataset):

    def __init__(self, filepath, key, limit=None, batch_size=512):

        self.filepath = filepath
        self.key = key
        self.limit = limit
        self.group = None
        self.batch_size = batch_size

    def __len__(self):
        if self.limit:
            return self.limit // self.batch_size
        else:
            store = h5py.File(self.filepath, 'r')
            return store[self.key]['X'].shape[0] // self.batch_size

    def __getitem__(self, i):
        if self.group is None:
            store = h5py.File(self.filepath, 'r')
            self.group = store[self.key]

        start = self.batch_size * i
        stop = self.batch_size * (i + 1)
        return (
            self.group['X'][start:stop, :],
            self.group['C'][start:stop, :, :]
        )
