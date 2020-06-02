import numpy as np

import torch

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import corner

from deconv.flow.maf import MAFlow
from deconv.gmm.sgd_gmm import SGDGMM

torch.set_default_tensor_type(torch.FloatTensor)

N = 10000
N_train = 8000

gmm = SGDGMM(
    2,
    2,
    batch_size=512,
    device=torch.device('cuda')
)

gmm.module.means.data = torch.Tensor(
    [
        [-10.0, 0.0],
        [10.0, 0.0]
    ]
).to(torch.device('cuda'))
gmm.module.soft_weights.data = torch.Tensor([0.0, 0.0]).to(torch.device('cuda'))

torch.set_default_tensor_type(torch.cuda.FloatTensor)

flow_kl = MAFlow(
    2,
    5,
    lr=1e-3,
    epochs=500,
    batch_size=512,
    device=torch.device('cuda')
)

optimiser = torch.optim.Adam(
    params=flow_kl.flow.parameters(),
    lr=1e-3
)

K = 512

for i in range(5000):
    
    flow_kl.flow.train()
    
    optimiser.zero_grad()
    
    samples, log_q = flow_kl.flow.sample_and_log_prob(K)
    log_p = gmm.module([samples])
    kl = (log_q - log_p).mean()
    
    kl.backward()
    optimiser.step()
    if i % 100 == 0:
        print('Epoch {}:, KL: {}'.format(i, kl.item()))
    
with torch.no_grad():
    X_kl = flow_kl.flow.sample(10000).cpu().squeeze()
    
fig, ax = plt.subplots()
corner.hist2d(X_kl[:, 0].numpy(), X_kl[:, 1].numpy(), ax=ax)
ax.set_xlim(-15, 15)
ax.set_ylim(-4, 4)
plt.show()