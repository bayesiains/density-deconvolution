import numpy as np 
from scipy.stats import multivariate_normal as mn 

def compute_data_ll(data, samples):
    if data == 'gaussian_1':
        mean = [1.0, 1.0]
        covar = [[0.09, 0.0], [0.0, 0.09]]

        return mn.logpdf(samples, mean, covar)

    elif data == 'gaussian_2':
        mean = [1.0, 1.0]
        covar = [[0.25, 0.0], [0.0, 0.25]]
        print(mean)

        return mn.logpdf(samples, mean, covar)

    elif data == 'gaussian_3':
        mean = [1.0, 1.0]
        covar = [[1.0, 0.0], [0.0, 1.0]]

        return mn.logpdf(samples, mean, covar)

    elif data == 'mixture_1':
        means = [[0.0, 0.0], [2.0, 3.0], [2.0, -3.0]]
        covars = [[[0.1, 0.0], [0.0, 1.5]], 
                  [[1.0, 0.0], [0.0, 0.1]],
                  [[1.0, 0.0], [0.0, 0.1]]]

        pdf1 = mn.pdf(samples, means[0], covars[0]) / 3
        pdf2 = mn.pdf(samples, means[1], covars[1]) / 3
        pdf3 = mn.pdf(samples, means[2], covars[2]) / 3

        return np.log(pdf1 + pdf2 + pdf3)

    elif data == 'mixture_2':
        means = [[-3.0, -3.0], [3.0, 3.0]]
        covars = [[[0.09, 0.0], [0.0, 0.09]], 
                  [[0.09, 0.0], [0.0, 0.09]]]

        pdf1 = mn.pdf(samples, means[0], covars[0]) / 2
        pdf2 = mn.pdf(samples, means[1], covars[1]) / 2

        return np.log(pdf1 + pdf2)

    elif data == 'mixture_3':
        means = [[-1.0, -1.0], [1.0, 1.0]]
        covars = [[[0.25, 0.0], [0.0, 0.25]], 
                  [[0.09, 0.0], [0.0, 0.09]]]

        pdf1 = mn.pdf(samples, means[0], covars[0]) / 2
        pdf2 = mn.pdf(samples, means[1], covars[1]) / 2

        return np.log(pdf1 + pdf2)
