import torch 
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
import logging
import copy
import os
import corner
from torch.utils.data import DataLoader

matplotlib.use('agg')

from deconv.lib.make_2d_toy_data import data_gen
from deconv.lib.make_2d_toy_noise_covar import covar_gen
from deconv.flow.svi import SVIFlow
from deconv.gmm.data import DeconvDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='mixture1')
parser.add_argument('--covar', type=str, default='fixed_diagonal_covar1')
parser.add_argument('--n_train_points', type=int, default=int(1e5))
parser.add_argument('--n_test_points', type=int, default=int(1e3))
parser.add_argument('--n_eval_points', type=int, default=int(1e3))
parser.add_argument('--eval_based_scheduler', type=str, default='10,20,30')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ds_size', type=int, required=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dir', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--flow_steps_prior', type=int, default=1)
parser.add_argument('--flow_steps_posterior', type=int, default=1)
parser.add_argument('--n_posterior_flows', type=int, default=1, help='approximate posterior is a mixture of normalizing flows')
parser.add_argument('--warmup_posterior_flow_diversity', type=int, default=20)
parser.add_argument('--warmup_kl', type=int, default=20)
parser.add_argument('--kl_init', type=float, default=0.5)
parser.add_argument('--posterior_context_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=int(1e4))
parser.add_argument('--objective', type=str, default='elbo', choices=['elbo', 'iwae', 'iwae_sumo'])
parser.add_argument('--K', type=int, default=1, help='# of samples for objective')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=100)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(args.gpu)

if args.dir is None:
	args.dir = 'toy/' + str(args.objective) + '/' + str(args.data) + '/' + str(args.covar) + '/'

if args.name is None:
	name = 'logs/seed_' + str(args.seed)

if os.path.isfile(args.dir + 'logs/' + name + '.log'):
  raise ValueError('This file already exists.')

logger = utils_misc.get_logger(logpath=(args.dir + 'logs/' + name + '.log'), filepath=os.path.abspath(__file__))
logger.info(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

def lr_scheduler(n_epochs_not_improved, optimzer, scheduler, logger):
	for i in range(len(scheduler) - 1):
		if n_epochs_not_improved < scheduler[i]:
			lr = args.lr * 0.1

	for param_group in optimzer.param_groups:
		param_group['lr'] = lr

	message = 'New learning rate: %f' % lr
	logger.info(message)

def main():
	train_data = torch.from_numpy(data_gen(args.data, args.n_train_points).astype(np.float32))
	train_covar = torch.from_numpy(covar_gen(args.covar, args.n_train_points).astype(np.float32))
	train_dataset = DeconvDataset(train_data, train_covar)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	test_data_clean = torch.from_numpy(data_gen(args.data, args.n_test_points).astype(np.float32)) 

	eval_data = torch.from_numpy(data_gen(args.data, args.n_eval_points).astype(np.float32))
	eval_covar = torch.from_numpy(covar_gen(args.covar, args.n_eval_points).astype(np.float32))

	model = SVIFlow(dimensions=2,
					objective=args.objective,
					flow_steps_prior=args.flow_steps_prior,
					flow_steps_posterior=args.flow_steps_posterior,
					n_posterior_flows=args.n_posterior_flows,
					warmup_posterior_flow_diversity=args.warmup_posterior_flow_diversity,
					warmup_kl=args.warmup_kl,
					kl_init=args.kl_init,
					posterior_context_size=args.posterior_context_size,
					batch_size=args.batch_size,
					device=device)


	#training
	scheduler =  list(map(int, args.eval_based_scheduler.split(',')))
	epoch = 0
	best_model = copy.deepcopy(model.state_dict())
	best_eval_loss = -model.score(eval_data.to(device), log_prob=False)
	n_epochs_not_improved = 0

	model.train()
	while n_epochs_not_improved < scheduler[-1] and epoch < args.n_epochs:
		for batch_idx, data in enumerate(train_loader):
		    loss = -model.log_prob(data.to(device)).mean()
		    optimizer.zero_grad()
		    loss.backward(retain_graph=True)
		    optimizer.step()

		model.eval()
		eval_loss = -model.log_prob(eval_data.to(device)).mean()

		if eval_loss < best_eval_loss:
		    best_model = copy.deepcopy(model.state_dict())
		    best_eval_loss = eval_loss
		    n_epochs_not_improved = 0

		else:
		    n_epochs_not_improved += 1

		lr_scheduler(n_epochs_not_improved, optimizer, scheduler, logger)

		if (epoch + 1) % args.test_freq == 0:
			test_loss_clean = -model.prior.log_prob(test_data_clean.to(device)).mean()

			message = 'Epoch %s:' % (epoch + 1), 'train loss = %.5f' % loss, 'eval loss = %.5f' % eval_loss, 'train loss (clean) = %.5f' % test_loss_clean
			logger.info(message)

		else:
			message = 'Epoch %s:' % (epoch + 1), 'train loss = %.5f' % loss, 'eval loss = %.5f' % eval_loss
			logger.info(message)

		if (epoch + 1) % args.viz_freq == 0:
			samples = model.prior.sample(10000).detach().cpu().numpy()
			corner.hist2d(samples[:, 0], samples[:, 1])

			fig_filename = args.dir + 'out/' + name + '_fig_' + str(epoch + 1) + '.png'
			plt.savefig(fig_filename)
			plt.close()

		model.train()

	torch.save(model.state_dict(), args.dir + 'models/' + name + '.model')
	logger.info('Training has finished.')

if __name__ == '__main__':
	main()

