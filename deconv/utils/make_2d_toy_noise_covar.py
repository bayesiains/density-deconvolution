import numpy as np
from sklearn.utils import shuffle as util_shuffle

def covar_gen(covar, n_samples, rng=np.random):
	if covar == 'fixed_diagonal_covar1':
		covar = np.zeros((n_samples, 2, 2))
		covar[:] = np.array([[0.1, 0.0], 
							 [0.0, 3.0]])

		return covar

	elif covar == 'fixed_diagonal_covar2':
		covar = np.zeros((n_samples, 2, 2))
		covar[:] = np.array([[0.3, 0.0], 
							 [0.0, 0.3]])

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
