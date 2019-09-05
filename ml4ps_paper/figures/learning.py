import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns

from results_table import get_table

sns.set()
sns.set_context('paper')

fig, axes = plt.subplots(1, 2, figsize=(6, 2)) 

table = get_table('../results/')

axes[0].set_xlabel('Time-scaled Epoch')
axes[0].set_ylabel('Avg. Log-likelihood')

labels = ('Existing EM', 'Minibatch EM', 'SGD')

result_files = os.listdir('../results/')
baseline_curves = np.array([
    np.loadtxt('../results/' + f) for f in result_files if f.startswith(
        'baseline_256'
    ) and f.endswith('loglike.log')
])

for l, result in zip(labels, table.values()):
    if l != 'Existing EM':
        x = np.arange(1, 21) * (result[256][1].mean() / 20)
        y = result[256][2].mean(axis=0)
        y_err = 1.96 * result[256][2].std(axis=0)
    else:
        x = np.arange(1, 21) * (result[256][1].mean() / 20)
        y = baseline_curves.mean(axis=0)
        y_err = 1.96 * baseline_curves.std(axis=0)

    axes[0].plot(x, y)
    axes[0].fill_between(x, y + y_err, y - y_err, alpha=0.5)

axes[0].set_xscale('log')
axes[0].set_ylim(-28, -25)


k = np.array([64, 128, 256, 512])

axes[1].set_xlabel(r'Mixture Components $K$')
axes[1].set_ylabel(r'Time (minutes)')
axes[1].set_xticks(k)

for l, result in zip(labels, table.values()):
    t = np.array([[r[1].mean(), r[1].std()] for r in result.values()])
    print(t.shape)
    axes[1].errorbar(k, t[:, 0], 1.96 * t[:, 1], label=l)

axes[1].legend(loc='lower left', bbox_to_anchor=(1.01, 0))

fig.tight_layout()

fig.savefig('figures/learning.pdf')

plt.show()
