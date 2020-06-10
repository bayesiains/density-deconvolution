import argparse

import numpy as np
import torch
import torch.utils.data as data_utils

from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM
from deconv.gmm.data import DeconvDataset
from deconv.flow.svi import SVIFlow
from deconv.utils.data_gen import generate_mixture_data

parser = argparse.ArgumentParser(description='Train SVI model on toy GMM with pretraining.')
parser.add_argument('-f', '--freeze-prior', action='store_true')
parser.add_argument('-k', '--samples', type=int)
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-l', '--learning-rate', type=float)
parser.add_argument('-m', '--hidden-features', type=int)
parser.add_argument('output_prefix')
args = parser.parse_args()


K = 4
D = 2
N = 50000
N_val = int(0.25 * N)

ref_gmm, S, (z_train, x_train), (z_val, x_val), _ = generate_mixture_data()

train_data = DeconvDataset(x_train.squeeze(), torch.cholesky(S.repeat(N, 1, 1)))
val_data = DeconvDataset(x_val.squeeze(), torch.cholesky(S.repeat(N, 1, 1)))

svi = SVIFlow(
    2,
    5,
    device=torch.device('cuda'),
    batch_size=512,
    epochs=args.epochs,
    lr=args.learning_rate,
    n_samples=args.samples,
    use_iwae=False,
    context_size=64,
    hidden_features=args.hidden_features
)

optimiser_prior = torch.optim.Adam(
    params=svi.model._prior.parameters(),
    lr=1e-3
)

scheduler_prior = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser_prior,
    mode='max',
    factor=0.5,
    patience=20,
    verbose=True,
    threshold=1e-6
)

loader_prior = data_utils.DataLoader(
    z_train.squeeze(),
    batch_size=svi.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

print('Pretraining prior')

for i in range(svi.epochs):
    
    train_loss = torch.tensor(0.0, device=svi.device)
    
    for j, d in enumerate(loader_prior):
        
        optimiser_prior.zero_grad()
        d = d.to(svi.device)
                
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        log_p_z = svi.model._prior.log_prob(d)
        torch.set_default_tensor_type(torch.FloatTensor)
        loss = -1 * log_p_z.mean()
        
        train_loss += log_p_z.sum()
        
        loss.backward()
        optimiser_prior.step()
        
    tl = train_loss.item() / len(z_train.squeeze())
    print('Epoch {}, Train Loss: {}'.format(i, tl))
    
    scheduler_prior.step(tl)



if args.freeze_prior:
    for param in svi.model._prior.parameters():
        param.requires_grad = False
else:
    optimiser_post = torch.optim.Adam(
        [
            {'params': svi.model._inputs_encoder.parameters()},
            {'params': svi.model._approximate_posterior.parameters()}
        ],
        lr=1e-3
    )

    scheduler_post = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser_post,
        mode='max',
        factor=0.5,
        patience=20,
        verbose=True,
        threshold=1e-6
    )

    loader_post = data_utils.DataLoader(
        train_data,
        batch_size=svi.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print('Pretraining posterior')

    for i in range(svi.epochs):

        train_loss = torch.tensor(0.0, device=svi.device)

        for j, d in enumerate(loader_post):

            optimiser_post.zero_grad()

            d_g = [a.to(svi.device) for a in d]
            d[1] = torch.matmul(d[1], d[1].transpose(-2, -1))
            t = ref_gmm._sample_posterior(d, 1).to(svi.device)

            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            ctx = svi.model._inputs_encoder(d_g)
            log_q_z = svi.model._approximate_posterior.log_prob(t, context=ctx)
            torch.set_default_tensor_type(torch.FloatTensor)

            loss = -1 * log_q_z.mean()

            train_loss += log_q_z.sum()

            loss.backward()
            optimiser_post.step()

        tl = train_loss.item() / len(train_data)

        print('Epoch {}, Train Loss: {}'.format(i, tl))

        scheduler_post.step(tl)
        
    pretrained_score = svi.score_batch(val_data, log_prob=True, num_samples=100)
    print('Val log-likelihood after pretraining: {}'.format(pretrained_score / len(val_data)))

pretrained_params = svi.model.state_dict()
torch.save(pretrained_params, args.output_prefix + '_pretrained_params.pt')    

svi.fit(train_data, val_data=val_data)
torch.save(svi.model.state_dict(), args.output_prefix + '_elbo_posttrained_params.pt')

trained_score = svi.score_batch(val_data, log_prob=True, num_samples=100)
print('Val log-likelihood after elbo training: {}'.format(trained_score / len(val_data)))

svi.use_iwae = True
svi.model.load_state_dict(pretrained_params)
svi.fit(train_data, val_data=val_data)
torch.save(svi.model.state_dict(), args.output_prefix + '_iw_posttrained_params.pt')

trained_score = svi.score_batch(val_data, log_prob=True, num_samples=100)
print('Val log-likelihood after iw training: {}'.format(trained_score / len(val_data)))