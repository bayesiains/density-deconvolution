import argparse

import numpy as np
import pandas as pd

from astropy.table import Table
from sklearn.model_selection import train_test_split

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


def pandas_to_numpy(df, output_file):
    df.insert(12, column='phot_g_mean_mag_error', value=0.01)
    df.insert(12, column='bp_rp_error', value=0.01)
    error_columns.insert(5, 'phot_g_mean_mag_error')
    error_columns.insert(5, 'bp_rp_error')

    X = df[columns].fillna(0.0).to_numpy(dtype=np.float32)
    C = np.zeros((len(df), 7, 7), dtype=np.float32)
    diag = np.arange(7)
    C[:, diag, diag] = df[error_columns].fillna(1e6).to_numpy(
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

    np.savez(
        output_file,
        X_train=X_train,
        C_train=C_train,
        X_val=X_val,
        C_val=C_val,
        X_test=X_test,
        C_test=C_test
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')

    args = parser.parse_args()

    df = vot_to_pandas(args.input_file)
    pandas_to_numpy(df, args.output_file)
