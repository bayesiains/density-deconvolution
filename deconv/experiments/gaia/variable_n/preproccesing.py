import argparse
import os

import h5py
import numpy as np
import pandas as pd

from astropy.table import Table
from sklearn.model_selection import train_test_split
import tqdm

np.random.seed(90115)

columns = [
    'ra',
    'dec',
    'parallax',
    'pmra',
    'pmdec',
    'bp_rp',
    'phot_g_mean_mag'
]

error_columns = [
    'ra_error',
    'dec_error',
    'parallax_error',
    'pmra_error',
    'pmdec_error',
]

corr_map = {
    'ra_dec_corr': [0, 1],
    'ra_parallax_corr': [0, 2],
    'ra_pmra_corr': [0, 3],
    'ra_pmdec_corr': [0, 4],
    'dec_parallax_corr': [1, 2],
    'dec_pmra_corr': [1, 3],
    'dec_pmdec_corr': [1, 4],
    'parallax_pmra_corr': [2, 3],
    'parallax_pmdec_corr': [2, 4],
    'pmra_pmdec_corr': [3, 4]
}


def get_covar(row):
    return np.diag(row[error_columns].fillna(1e12).to_numpy(dtype=np.float32))


def vot_to_pandas(vot_file):

    tb = Table.read(vot_file)

    df = tb[columns + error_columns + list(corr_map.keys())].to_pandas()
    return df


def pandas_to_numpy(df):
    df.insert(12, column='phot_g_mean_mag_error', value=0.01)
    df.insert(12, column='bp_rp_error', value=0.01)

    ec = error_columns.copy()
    ec.insert(5, 'phot_g_mean_mag_error')
    ec.insert(5, 'bp_rp_error')

    X = df[columns].fillna(0.0).to_numpy(dtype=np.float32)
    C = np.zeros((len(df), 7, 7), dtype=np.float32)
    diag = np.arange(7)
    C[:, diag, diag] = df[ec].fillna(1e6).to_numpy(
        dtype=np.float32
    )

    for column, (i, j) in corr_map.items():
        C[:, i, j] = df[column].fillna(0).to_numpy(dtype=np.float32)
        C[:, i, j] *= (C[:, i, i] * C[:, j, j])
        C[:, j, i] = C[:, i, j]

    C[:, diag, diag] = C[:, diag, diag]**2

    X_train, X_test, C_train, C_test = train_test_split(
        X, C, test_size=0.2, random_state=90115
    )
    X_val, X_test, C_val, C_test = train_test_split(
        X_test, C_test, test_size=0.5, random_state=84253
    )

    return (
        (X_train, C_train),
        (X_val, C_val),
        (X_test, C_test)
    )


def numpy_to_file(data, output_file):
    (X_train, C_train), (X_val, C_val), (X_test, C_test) = data
    np.savez(
        output_file,
        X_train=X_train,
        C_train=C_train,
        X_val=X_val,
        C_val=C_val,
        X_test=X_test,
        C_test=C_test
    )


def process_csv(csv_dir, output_file):
    file_list = os.listdir(csv_dir)

    store = h5py.File(output_file, 'w')

    train_data = store.create_group('train')
    val_data = store.create_group('val')
    test_data = store.create_group('test')

    groups = (train_data, val_data, test_data)

    for g in groups:
        g.create_dataset(
            'X', (0, 7),
            maxshape=(None, 7),
            dtype=np.float32,
            chunks=(512, 7),
            compression='gzip'
        )
        g.create_dataset(
            'C', (0, 7, 7),
            maxshape=(None, 7, 7),
            dtype=np.float32,
            chunks=(512, 7, 7),
            compression='gzip'
        )

    for f in tqdm.tqdm(file_list):
        df = pd.read_csv(csv_dir + f)
        data = pandas_to_numpy(df)

        for d, g in zip(data, groups):
            X, C = d
            g['X'].resize((g['X'].shape[0] + X.shape[0], 7))
            g['X'][-X.shape[0]:, :] = X

            g['C'].resize((g['C'].shape[0] + C.shape[0], 7, 7))
            g['C'][-C.shape[0]:, :, :] = C


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('csv_dir')
    parser.add_argument('output_file')

    args = parser.parse_args()

    process_csv(args.csv_dir, args.output_file)
