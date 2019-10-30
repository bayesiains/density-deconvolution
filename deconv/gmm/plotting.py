import numpy as np
import matplotlib as mpl


def plot_covariance(mean, cov, ax, alpha=0.5, color=None):
    """
    Plot a Gaussian convariance on a Matplotlib axis.

    Adapted from https://scikit-learn.org/stable/auto_examples/
    mixture/plot_gmm_covariances.html
    """
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0]) * 180 / np.pi
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(
        mean,
        v[0],
        v[1],
        180 + angle,
        color=color
    )
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(alpha)
    ax.add_artist(ell)
