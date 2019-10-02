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

    def __init__(self, filepath, key):

        self.store = h5py.File(filepath, 'r')
        self.group = self.store[key]

    def __len__(self):
        return self.group['X'].shape[0]

    def __getitem__(self, i):
        return (
            self.group['X'][i, :],
            self.group['C'][i, :, :]
        )
