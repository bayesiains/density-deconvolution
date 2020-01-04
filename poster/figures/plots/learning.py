import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns

from results_table import get_table

sns.set()
sns.set_context('poster')
sns.set_palette('colorblind')

fig, ax = plt.subplots(1, 1, figsize=(12, 6)) 

table = get_table('../results/')

ax.set_xlabel('Time-scaled Epoch')
ax.set_ylabel('Avg. Log-likelihood')

labels = ('Existing EM', 'Minibatch EM', 'SGD')

result_files = os.listdir('../results/')
baseline_curves = np.array([
    np.loadtxt('../results/' + f) for f in result_files if f.startswith(
        'baseline_256'
    ) and f.endswith('loglike.log')
])

linestyles = ('-', '--', '-.')

for l, ls, result in zip(labels, linestyles, table.values()):
    if l != 'Existing EM':
        x = np.arange(1, 21) * (result[256][1].mean() / 20)
        y = result[256][2].mean(axis=0)
        y_err = 1.96 * result[256][2].std(axis=0)
    else:
        x = np.arange(1, 21) * (result[256][1].mean() / 20)
        y = baseline_curves.mean(axis=0)
        y_err = 1.96 * baseline_curves.std(axis=0)

    ax.plot(x, y, linestyle=ls, label=l)
    ax.fill_between(x, y + y_err, y - y_err, alpha=0.5)

ax.set_xscale('log')
ax.set_ylim(-28, -25)
ax.legend()

fig.tight_layout()

fig.savefig('figures/plots/epochs.pdf')

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

k = np.array([64, 128, 256, 512])

ax.set_xlabel(r'Mixture Components $K$')
ax.set_ylabel(r'Time (minutes)')
ax.set_xticks(k)

for l, ls, result in zip(labels, linestyles, table.values()):
    t = np.array([[r[1].mean(), r[1].std()] for r in result.values()])
    print(t.shape)
    ax.errorbar(k, t[:, 0], 1.96 * t[:, 1], label=l, linestyle=ls)

ax.legend(loc='upper left')
ax.set_ylim(-1, 430)

fig.tight_layout()

fig.savefig('figures/plots/learning.pdf')
