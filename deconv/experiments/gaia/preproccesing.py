
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
    'phot_g_mean_flux'
]

error_columns = [
    'ra_error',
    'dec_error',
    'parallax_error',
    'pmra_error',
    'pmdec_error',
    'phot_g_mean_flux_error'
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


def vot_to_pandas(vot_file, output):

    tb = Table.read(vot_file)

    df = tb[columns + error_columns + corr_map.keys()].to_pandas()
    df.to_hdf(output, 'table')


def pandas_to_numpy(pandas_hdf5_file, output_file):
    df = pd.read_hdf(pandas_hdf5_file, 'table')
    df.insert(12, column='bp_rp_error', value=0.01)
    error_columns.insert(5, 'bp_rp_error')
    print(error_columns)

    X = df[columns].fillna(0.0).to_numpy(dtype=np.float32)
    C = np.zeros((len(df), 7, 7))
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
