import numpy as np
from sklearn.utils import shuffle as util_shuffle


def covar_gen(covar, n_samples, rng=np.random):
    if covar == 'gmm':
        covar = np.zeros((n_samples, 2, 2))
        covar[:] = np.array([[0.1, 0.0],
                             [0.0, 1.0]])

        return covar

    elif covar == 'fixed_diagonal_covar1':
        covar = np.zeros((n_samples, 2, 2))
        covar[:] = np.array([[0.1, 0.0],
                             [0.0, 3.0]])

        return covar

    elif covar == 'fixed_diagonal_covar2':
        covar = np.zeros((n_samples, 2, 2))
        covar[:] = np.array([[0.3, 0.0],
                             [0.0, 0.3]])

        return covar

    elif covar == 'fixed_diagonal_covar3':
        covar = np.zeros((n_samples, 2, 2))
        covar[:] = np.array([[0.2, 0.0],
                             [0.0, 0.2]])

        return covar

    elif covar == 'fixed_diagonal_covar4':
        covar = np.zeros((n_samples, 2, 2))
        covar[:] = np.array([[0.05, 0.0],
                             [0.0, 1.0]])

        return covar

    elif covar == 'fixed_diagonal_covar5':
        covar = np.zeros((n_samples, 2, 2))
        covar[:] = np.array([[0.05, 0.0],
                             [0.0, 0.8]])

        return covar

    elif covar == 'fixed_diagonal_covar6':
        covar = np.zeros((n_samples, 2, 2))
        covar[:] = np.array([[0.5, 0.0],
                             [0.0, 0.01]])

        return covar

    elif covar == 'fixed_diagonal_covar9':
        covar = np.zeros((n_samples, 2, 2))
        covar[:] = np.array([[2.0, 0.0],
                             [0.0, 0.01]])

        return covar

    elif covar == 'fixed_diagonal_covar8':
        covar = np.zeros((n_samples, 2, 2))
        covar[:] = np.array([[0.1, 0.0],
                             [0.0, 1.0]])

        return covar

    elif covar == 'random_diagonal_covar1':
        sigma_x = rng.normal(0.0, 0.1, n_samples)**2
        sigma_y = rng.normal(0.0, 1.0, n_samples)**2

        covar = np.zeros((n_samples, 2, 2))
        for i in range(n_samples):
            covar[i] = np.array([[sigma_x[i], 0.0],
                                 [0.0, sigma_y[i]]])

        return covar

    else:
        raise ValueError('Choose one of the available covariance options.')
