import argparse
import json
import os

import numpy as np

from extreme_deconvolution import extreme_deconvolution


def score_baseline(datafile, results_dir, output_file):
    data = np.load(datafile)

    rf = os.listdir(results_dir)

    param_files = [
        f for f in rf if f.startswith('baseline_512') and f.endswith('.npz')
    ]

    scores = []

    for p in param_files:
        params = np.load(results_dir + p)
        weights = params['weights']
        means = params['means']
        covars = params['covar']

        test_score = extreme_deconvolution(
            data['X_test'],
            data['C_test'],
            weights,
            means,
            covars,
            likeonly=True
        )
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

    score_baseline(args.datafile, args.results_dir, args.output_file)
