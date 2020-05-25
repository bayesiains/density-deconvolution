import torch


    
def minibatch_sample(sample_f, num_samples, dimensions, batch_size, device=torch.device('cpu'), context=None, x=None):
    
    if x is not None:
        ld = len(x)
    elif context is not None:
        ld = context.shape[0]
    else:
        ld = 1
    
    samples = torch.zeros((ld, num_samples, dimensions), device=device)

    for i in range(-(-num_samples // batch_size)):
        start = i * batch_size
        stop = (i + 1) * batch_size
        n = min(batch_size, num_samples - start)
        if x is None:
            samples[:, start:stop, :] = sample_f(n, context=context).to(device)
        else:
            samples[:, start:stop, :] = sample_f(x, n, context=context).to(device)
        
    return samples