import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from deconv.gmm.data import DeconvDataset
from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM
from deconv.gmm.plotting import plot_covariance
from deconv.flow.svi import SVIFlow

from deconv.experiments.checks.data import generate_data

sns.set()


K = 3
D = 2
N = 10000

plot = True
device = None

if not device:
    device = torch.device('cpu')

m_1 = np.array([0, 0])
C_1 = np.array([
    [0.01, 0],
    [0, 1.5]
])

m_2 = np.array([3, 4.5])
C_2 = np.array([
    [1, 0],
    [0, 0.01]
])

m_3 = np.array([3, -4.5])
C_3 = np.array([
    [1, 0],
    [0, 0.01]
])

# m_4 = np.array([6, 0])
# C_4 = np.array([
#     [0.01, 0],
#     [0, 1]
# ])

X_1 = np.random.multivariate_normal(m_1, C_1, N)
X_2 = np.random.multivariate_normal(m_2, C_2, N)
X_3 = np.random.multivariate_normal(m_3, C_3, N)
# X_4 = np.random.multivariate_normal(m_4, C_4, N)

X = np.concatenate((X_1, X_2, X_3), axis=0)

S = np.array([
    [0.1, 0],
    [0, 2]
])

idx = np.random.permutation(3 * N)

X = X[idx, :]

X_noisy = X + np.random.multivariate_normal([0, 0], S, 3 * N)

S = np.repeat([S], 3 * N, axis=0) 

fig, ax = plt.subplots()
ax.scatter(X_noisy[:, 0], X_noisy[:, 1])
ax.scatter(X[:, 0], X[:, 1])
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
plt.show()

X_train = X_noisy[:(2 * N), :]
X_test = X_noisy[(2 * N):, :]

nc_train = S[:(2 * N), :, :]
nc_test = S[(2 * N):, :, :]

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

svi = SVIFlow(
    D,
    5,
    device=device,
    batch_size=512,
    epochs=50,
    lr=1e-4
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

    x_width = 200
    y_width = 200

    x = np.linspace(-5, 10, num=x_width, dtype=np.float32)
    y = np.linspace(-15, 15, num=y_width, dtype=np.float32)

    xx, yy = np.meshgrid(x, y)

    d = torch.tensor(
        np.concatenate((xx[:, :, None], yy[:, :, None]), axis=-1)
    )

    z = np.zeros((y_width, x_width))

    with torch.no_grad():
        for i in range(x_width):
            z[i, :] = svi.model._prior.log_prob(d[:, i, :]).detach().numpy()

    fig, ax = plt.subplots()

    ax.imshow(np.exp(z), extent=[-5, 10, -15, 15], origin='lower')

    target = (
        torch.Tensor([[4.0, 0.0]]),
        torch.cholesky(torch.Tensor([[
            [0.1, 0],
            [0, 2]
        ]]))
    )
    ctx = svi.model._inputs_encoder(target)
    posterior_samples = svi.model.encode(
        ctx, num_samples=1000
    ).detach().numpy()

    plt.show()



