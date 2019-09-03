import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import corner

sns.set()
sns.set_context('paper')

N = 100000

weights, means, covars = torch.load(
    '../results/em_256_507993_params.pkl',
    map_location=torch.device('cpu')
)

X = torch.distributions.MultivariateNormal(
    loc=means, covariance_matrix=covars
).sample((N,)).numpy()

j = torch.distributions.Categorical(
    probs=weights.squeeze()
).sample((N,)).numpy()

X = X[np.arange(N), j, :]

fig, ax = plt.subplots(figsize=(3, 2))

figure = corner.hist2d(X[:, 0], X[:, 1], bins=100)

ax.set_xlim(0, 360)
ax.set_xticks(np.arange(0, 361, 60))

ax.set_ylim(-90, 90)
ax.set_yticks(np.arange(-90, 91, 30))

ax.set_xlabel(r'Right Ascension ($\circ$)')
ax.set_ylabel(r'Declination ($\circ$)')

fig.tight_layout()

fig.savefig('density.pdf')

plt.show()
