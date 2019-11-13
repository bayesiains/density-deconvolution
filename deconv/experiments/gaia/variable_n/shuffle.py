"""
Script to shuffle the hdf5 dataset.

Does everything in memory.
"""
import argparse

import numpy as np
import h5py

np.random.seed(9246130)


def shuffle_store(hdf5_in, hdf5_out):
    """Create a shuffled copy of dataset."""
    store_in = h5py.File(hdf5_in, 'r')
    store_out = h5py.File(hdf5_out)

    for d in ('train', 'val', 'test'):

        print('Shuffling {}'.format(d))

        group = store_out.create_group(d)
        X = store_in[d]['X'][()]
        C = store_in[d]['C'][()]

        print('X shape: {}'.format(X.shape))
        print('C shape: {}'.format(C.shape))

        idx = np.random.permutation(X.shape[0])

        group.create_dataset(
            'X',
            X.shape,
            maxshape=X.shape,
            dtype=np.float32,
            chunks=(512, 7),
            compression='lzf'
        )
        group['X'][:, :] = X[idx, :]

        group.create_dataset(
            'C',
            C.shape,
            maxshape=C.shape,
            dtype=np.float32,
            chunks=(512, 7, 7),
            compression='lzf'
        )
        group['C'][:, :, :] = C[idx, :, :]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_in')
    parser.add_argument('hdf5_out')

    args = parser.parse_args()
    shuffle_store(args.hdf5_in, args.hdf5_out)
