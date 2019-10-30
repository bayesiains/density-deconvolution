import argparse
import json
import os

import numpy as np
import torch

from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM
from decon.gmm.data import DeconvDataset


def score_sgd(datafile, results_dir, output_file):
    data = np.load(datafile)

    test_data = SGDDeconvDataset(
        torch.Tensor(data['X_test']),
        torch.Tensor(data['C_test'])
    )

    rf = os.listdir(results_dir)

    param_files = [
        f for f in rf if f.startswith('sgd_512') and f.endswith('.pkl')
    ]

    gmm = SGDDeconvGMM(
        512,
        7,
        batch_size=500
    )

    scores = []

    for p in param_files:
        state_dict = torch.load(
            results_dir + p,
            map_location=torch.device('cpu')
        )
        gmm.module.load_state_dict(state_dict)

        test_score = gmm.score_batch(test_data)
        print(test_score)

        scores.append(test_score)

    print('Test Score: {} +- {}'.format(
        np.mean(scores),
        np.std(scores)
    ))

    json.dump(scores, open(output_file, 'w'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('datafile')
    parser.add_argument('results_dir')
    parser.add_argument('output_file')

    args = parser.parse_args()

    score_sgd(args.datafile, args.results_dir, args.output_file)
