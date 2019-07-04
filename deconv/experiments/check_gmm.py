import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from deconv.gmm.plotting import plot_covariance
from deconv.gmm.basic_gmm import BasicGMM

D = 2
K = 3

N = 200

means = (np.random.rand(K, D) * 20) - 10
q = (2 * np.random.randn(K, D, D))
covars = np.matmul(q.swapaxes(1, 2), q)

X = np.empty((N, K, D))

fig, ax = plt.subplots()

for i in range(K):
    X[:, i, :] = np.random.multivariate_normal(
        mean=means[i, :],
        cov=covars[i, :, :],
        size=N
    )
    sc = ax.scatter(X[:, i, 0], X[:, i, 1], alpha=0.2, marker='x')
    plot_covariance(
        means[i, :],
        covars[i, :, :],
        ax,
        color=sc.get_facecolor()[0]
    )

X_data = torch.from_numpy(X.reshape(-1, D).astype(np.float32))

gmm = BasicGMM(K, D, max_iters=1000)
gmm.fit(X_data)

sc = ax.scatter(gmm.means[:, 0], gmm.means[:, 1], marker='+')


for i in range(K):
    plot_covariance(
        gmm.means[i, :],
        gmm.covars[i, :, :],
        ax,
        color=sc.get_facecolor()[0]
    )

plt.show()
