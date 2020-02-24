import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from deconv.gmm.data import DeconvDataset
from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM
from deconv.flow.svi import SVIFLow

sns.set()


def check_svi(N, plot=False, device=None):

    K = 2
    D = 2

    if not device:
        device = torch.device('cpu')

    m_1 = np.array([-0, 0])
    C_1 = np.array([[10, 0], [0, 10]])
    N_1 = np.array([[2, 0], [0, 2]])

    m_2 = np.array([0, 0])
    C_2 = np.array([[1, 0], [0, 1]])
    N_2 = np.array([[2, 0], [0, 2]])

    X_train = np.zeros((2 * N, 2))
    nc_train = np.zeros((2 * N, 2, 2))

    X_train[:N] = np.random.multivariate_normal(m_1, C_1, size=N)
    X_train[:N] += np.random.multivariate_normal([0, 0], N_1, size=N)
    nc_train[:N, :, :] = np.linalg.cholesky(N_1)

    X_train[N:] = np.random.multivariate_normal(m_2, C_2, size=N)
    X_train[N:] += np.random.multivariate_normal([0, 0], N_2, size=N)
    nc_train[N:, :, :] = np.linalg.cholesky(N_2)

    X_test = np.zeros((2 * N, 2))
    nc_test = np.zeros((2 * N, 2, 2))

    X_test[:N] = np.random.multivariate_normal(m_1, C_1, size=N)
    X_test[:N] += np.random.multivariate_normal([0, 0], N_1, size=N)
    nc_test[:N, :, :] = np.linalg.cholesky(N_1)

    X_test[N:] = np.random.multivariate_normal(m_2, C_2, size=N)
    X_test[N:] += np.random.multivariate_normal([0, 0], N_2, size=N)
    nc_test[N:, :, :] = np.linalg.cholesky(N_2)

    train_data = DeconvDataset(
        torch.Tensor(X_train.reshape(-1, D).astype(np.float32)),
        torch.Tensor(
            nc_train.reshape(-1, D, D).astype(np.float32)
        )
    )

    test_data = DeconvDataset(
        torch.Tensor(X_test.reshape(-1, D).astype(np.float32)),
        torch.Tensor(
            nc_test.reshape(-1, D, D).astype(np.float32)
        )
    )

    svi = SVIFLow(
        D,
        5,
        device=device,
        batch_size=512,
        epochs=50,
        lr=1e-3
    )
    svi.fit(train_data, val_data=None)

    test_log_prob = svi.score_batch(test_data, log_prob=True)

    print('Test log prob: {}'.format(test_log_prob / len(test_data)))

    gmm = SGDDeconvGMM(
        K,
        D,
        device=device,
        batch_size=256,
        epochs=50,
        lr=1e-1
    )
    gmm.fit(train_data, val_data=test_data, verbose=True)
    test_log_prob = gmm.score_batch(test_data)
    print('Test log prob: {}'.format(test_log_prob / len(test_data)))

    if plot:
        prior_samples = svi.model._prior.sample(1000).detach().numpy()

        fig, ax = plt.subplots()

        ax.scatter(X_train[:, 0], X_train[:, 1], alpha=0.5)

        ax.scatter(prior_samples[:, 0], prior_samples[:, 1])
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        plt.show()


if __name__ == '__main__':
    N = 5000
    check_svi(N, plot=True)
