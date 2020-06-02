import torch
from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM



def generate_mixture_data():
    K = 4
    D = 2
    N = 50000
    N_val = int(0.25 * N)

    torch.set_default_tensor_type(torch.FloatTensor)

    ref_gmm = SGDDeconvGMM(
        K,
        D,
        batch_size=512,
        device=torch.device('cpu')
    )

    ref_gmm.module.soft_weights.data = torch.zeros(K)
    scale = 2

    ref_gmm.module.means.data = torch.Tensor([
        [-scale, 0],
        [scale, 0],
        [0, -scale],
        [0, scale]
    ])

    short_std = 0.3
    long_std = 1

    stds = torch.Tensor([
        [short_std, long_std],
        [short_std, long_std],
        [long_std, short_std],
        [long_std, short_std]
    ])

    ref_gmm.module.l_diag.data = torch.log(stds)

    state = torch.get_rng_state()
    torch.manual_seed(432988)

    z_train = ref_gmm.sample_prior(N)
    z_val = ref_gmm.sample_prior(N_val)

    noise_short = 0.1
    noise_long = 1.0

    S = torch.Tensor([
        [noise_short, 0],
        [0, noise_long]
    ])

    noise_distribution = torch.distributions.MultivariateNormal(
        loc=torch.Tensor([0, 0]),
        covariance_matrix=S
    )

    x_train = z_train + noise_distribution.sample([N])
    x_val = z_val + noise_distribution.sample([N_val])
    
    torch.manual_seed(263568)
    
    z_test = ref_gmm.sample_prior(N)
    x_test = z_test + noise_distribution.sample([N])

    torch.set_rng_state(state)
    
    return (
        ref_gmm,
        S,
        (z_train, x_train),
        (z_val, x_val),
        (z_test, x_test)
    )