import torch


    
def minibatch_sample(sample_f, num_samples, batch_size, device=torch.device('cpu'), 
                     **kwargs):

    samples = torch.zeros((num_samples, self.dimensions), device=device)

    for i in range(-(-num_samples // batch_size)):
        start = i * batch_size
        stop = (i + 1) * batch_size
        n = min(batch_size, num_samples - start)
        samples[start:stop] = sample_f(n, **kwargs).to(device)