import numpy as np
import torch
import torch.utils.data as data_utils


import matplotlib.pyplot as plt
import seaborn as sns

from deconv.gmm.sgd_deconv_gmm import SGDDeconvDataset
from deconv.gmm.util import minibatch_k_means

sns.set()


def check_minibatch_k_means(plot=True):

    x = np.random.randn(200, 2)
    x[:100, :] += np.array([-5, 0])
    x[100:, :] += np.array([5, 0])

    noise_covars = np.zeros((200, 2, 2))

    data = SGDDeconvDataset(
        torch.Tensor(x.astype(np.float32)),
        torch.Tensor(
            noise_covars.astype(np.float32)
        )
    )

    loader = data_utils.DataLoader(
        data,
        batch_size=20,
        num_workers=4,
        shuffle=True
    )

    counts, centroids = minibatch_k_means(loader, 2)

    print(centroids)
    print(counts)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1])

        plt.show()


if __name__ == '__main__':
    check_minibatch_k_means(plot=True)
