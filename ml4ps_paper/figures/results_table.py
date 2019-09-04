import argparse
import json
import os

import numpy as np

MODELS = [
    'baseline',
    'em',
    'sgd'
]

SIZES = [
    64,
    128,
    256
]

def get_results(files, model, size, file_dir):
    return [
        json.load(open(str(file_dir) + f)) for f in files if f.startswith(
            '{}_{}'.format(model, size)
        ) and f.endswith('.json')
    ]


def produce_table(results_dir):

    results_files = os.listdir(str(results_dir))

    table = {}

    for model in MODELS:
        table[model] = {}
        for size in SIZES:
            results = get_results(results_files, model, size, results_dir)
            scores = np.array([r['val_score'] for r in results]) / 200000
            times = np.array(
                [r['end_time'] - r['start_time'] for r in results]
            ) / 60
            if model != 'baseline':
                curve = np.array([r['train_curve'] for r in results]) / 1600000
            else:
                curve = None

            if model == 'sgd':
                scores = scores * -1
                curve = curve * -1

            table[model][size] = (scores, times, curve)

            print('{} {}: {:.2f} +- {:.2f}, {:.1f} +- {:.2f}'.format(
                model,
                size,
                scores.mean(),
                scores.std(),
                times.mean(),
                times.std()
            ))

    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    args = parser.parse_args()

    print(args)

    produce_table(args.results_dir)
