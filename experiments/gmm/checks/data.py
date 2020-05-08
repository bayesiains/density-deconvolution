import numpy as np


def generate_data(D, K, N):
    means = (np.random.rand(K, D) * 20) - 10
    q = (2 * np.random.randn(K, D, D))
    covars = np.matmul(q.swapaxes(1, 2), q)

    qn = (0.5 * np.random.randn(2 * N, K, D, D))
    noise_covars = np.matmul(qn.swapaxes(2, 3), qn) + 1e-3 * np.eye(D)

    X = np.empty((2 * N, K, D))

    for i in range(K):
        X[:, i, :] = np.random.multivariate_normal(
            mean=means[i, :],
            cov=covars[i, :, :],
            size=2 * N
        )
        for j in range(2 * N):
            X[j, i, :] += np.random.multivariate_normal(
                mean=np.zeros(D),
                cov=noise_covars[j, i, :, :]
            )

    p = np.random.permutation(2 * N)

    X_train = X[p, :][:N]
    X_test = X[p, :][N:]

    nc_train = noise_covars[p, :, :][:N]
    nc_test = noise_covars[p, :, :][N:]

    data = (X_train, nc_train, X_test, nc_test)
    params = (means, covars)

    return data, params
