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

    def __init__(self, filepath, key, limit=None):

        self.filepath = filepath
        self.key = key
        self.limit = limit
        self.group = None

    def __len__(self):
        if self.limit:
            return self.limit
        else:
            store = h5py.File(self.filepath, 'r')
            return store[self.key]['X'].shape[0]

    def __getitem__(self, i):
        if self.group is None:
            store = h5py.File(self.filepath, 'r')
            self.group = store[self.key]
        return (
            self.group['X'][i, :],
            self.group['C'][i, :, :]
        )
