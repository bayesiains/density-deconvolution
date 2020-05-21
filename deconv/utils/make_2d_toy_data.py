import numpy as np
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


def data_gen(data, n_samples, noise=None, rng=np.random):
    if data == 'swissroll':
        if noise is None:
            noise = 1.0

        data = sklearn.datasets.make_swiss_roll(
            n_samples=n_samples, noise=noise)[0]
        data = data.astype(np.float32)[:, [0, 2]]
        data /= 5

        return data, noise

    elif data == 'gaussian_1':
        return np.random.multivariate_normal([1.0, 1.0], [[0.09, 0.0], [0.0, 0.09]], n_samples), None

    elif data == 'gaussian_2':
        return np.random.multivariate_normal([1.0, 1.0], [[0.25, 0.0], [0.0, 0.25]], n_samples), None

    elif data == 'gaussian_3':
        return np.random.multivariate_normal([1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], n_samples), None

    elif data == 'mixture_1':
        coins = np.random.choice(3, n_samples, p=[1. / 3, 1. / 3, 1. / 3])
        bincounts = np.bincount(coins)

        means = [[0.0, 0.0], [2.0, 3.0], [2.0, -3.0]]
        covars = [[[0.1, 0.0], [0.0, 1.5]],
                  [[1.0, 0.0], [0.0, 0.1]],
                  [[1.0, 0.0], [0.0, 0.1]]]

        samples = np.zeros((n_samples, 2))

        offset = 0
        for i in range(3):
            samples[offset:(offset + bincounts[i])
                    ] = np.random.multivariate_normal(means[i], covars[i], bincounts[i])

            offset += bincounts[i]

        return util_shuffle(samples), None

    elif data == 'mixture_2':
        coins = np.random.choice(2, n_samples, p=[1. / 2, 1. / 2])
        bincounts = np.bincount(coins)

        means = [[-3.0, -3.0], [3.0, 3.0]]
        covars = [[[0.09, 0.0], [0.0, 0.09]],
                  [[0.09, 0.0], [0.0, 0.09]]]

        samples = np.zeros((n_samples, 2))

        offset = 0
        for i in range(2):
            samples[offset:(offset + bincounts[i])
                    ] = np.random.multivariate_normal(means[i], covars[i], bincounts[i])

            offset += bincounts[i]

        return util_shuffle(samples), None

    elif data == 'mixture_3':
        coins = np.random.choice(2, n_samples, p=[1. / 2, 1. / 2])
        bincounts = np.bincount(coins)

        means = [[-1.0, -1.0], [1.0, 1.0]]
        covars = [[[0.25, 0.0], [0.0, 0.25]],
                  [[0.09, 0.0], [0.0, 0.09]]]

        samples = np.zeros((n_samples, 2))

        offset = 0
        for i in range(2):
            samples[offset:(offset + bincounts[i])
                    ] = np.random.multivariate_normal(means[i], covars[i], bincounts[i])

            offset += bincounts[i]

        return util_shuffle(samples), None

    elif data == 'circles':
        if noise is None:
            noise = 0.08

        data = sklearn.datasets.make_circles(
            n_samples=n_samples, factor=.5, noise=noise)[0]
        data = data.astype(np.float32)
        # data *= 3
        data *= 5

        return data, noise

    elif data == 'circles_easy':
        if noise is None:
            noise = 0.02

        data = sklearn.datasets.make_circles(
            n_samples=n_samples, factor=0.4, noise=noise)[0]
        data = data.astype(np.float32)
        data *= 3
        # data *= 5

        return data, noise

    elif data == 'rings':
        if noise is None:
            noise = 0.08

        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=noise, size=X.shape)

        return X.astype(np.float32), noise

    elif data == 'moons':
        if noise is None:
            noise = 0.03

        data = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise)[0]
        data = data.astype(np.float32)
        data = data * 3
        # data = data * 2 + np.array([-1, -0.2])

        return data, noise

    elif data == '8gaussians':
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(n_samples):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype=np.float32)
        dataset /= 1.414

        return dataset, None

    elif data == '4gaussians':
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(n_samples):
            point = rng.randn(2) * 0.5
            idx = rng.randint(4)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype=np.float32)
        dataset /= 1.414

        return dataset, None

    elif data == 'pinwheel':
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes * num_per_class, 2) * \
            np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum('ti', 'tij->tj', features, rotations)), None

    elif data == '2spirals':
        if noise is None:
            noise = 0.1

        n = np.sqrt(np.random.rand(n_samples // 2, 1)) * \
            540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_samples // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(n_samples // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * noise

        return x, noise

    elif data == 'checkerboard':
        x1 = np.random.rand(n_samples) * 4 - 2
        x2_ = np.random.rand(n_samples) - \
            np.random.randint(0, 2, n_samples) * 2
        x2 = x2_ + (np.floor(x1) % 2)

        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2, None

    elif data == 'line':
        x = rng.rand(n_samples) * 5 - 2.5
        y = x

        return np.stack((x, y), 1), None

    elif data == 'cos':
        if noise is None:
            noise = 0.1

        x = rng.rand(n_samples) * 5 - 2.5 + noise * rng.randn(n_samples)
        y = np.sin(x) * 2.5 + noise * rng.randn(n_samples)

        return np.stack((x, y), 1), noise

    else:
        raise ValueError('Choose one of the available data options.')
