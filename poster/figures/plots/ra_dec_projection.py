import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set()
sns.set_context('poster')

weights, means, covars = torch.load(
    '../results/em_256_508798_params.pkl',
    map_location=torch.device('cpu')
)

x_width = 6000
y_width = 3000

x = np.linspace(0, 360, num=x_width, dtype=np.float32)
y = np.linspace(-90, 90, num=y_width, dtype=np.float32)

xx, yy = np.meshgrid(x, y)

d = torch.tensor(np.concatenate((xx[:, :, None], yy[:, :, None]), axis=-1))

z = np.zeros((y_width, x_width))

mvn = torch.distributions.MultivariateNormal(
    loc=means[:, :2], covariance_matrix=covars[:, :2, :2]
)

for i in range(x_width):
    z[:, i] = torch.logsumexp(
        mvn.log_prob(d[:, i, None, :]) + torch.log(weights[:, 0]),
        axis=-1
    )


fig, ax = plt.subplots(figsize=(24, 12))

cmap = sns.cubehelix_palette(8, as_cmap=True)

im = ax.imshow(z, extent=[0, 360, -90, 90], origin='lower')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

ax.set_xlim(0, 360)
ax.set_xticks(np.arange(0, 361, 60))

ax.set_ylim(-90, 90)
ax.set_yticks(np.arange(-90, 91, 30))

ax.grid(False)
fig.colorbar(im, cax=cax)

ax.set_xlabel(r'Right Ascension ($\circ$)')
ax.set_ylabel(r'Declination ($\circ$)')

fig.tight_layout()

fig.savefig('figures/density.pdf')
