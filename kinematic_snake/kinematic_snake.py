__doc__ = """ Core concepts """

import numpy as np
from scipy.integrate import trapz, cumtrapz


def zero_mean_integral(sampled_func, samples):
    """ Zero mean integral from 0 to 1 always.
    Can be scaled up and down as a post-processing step.

    Parameters
    ----------
    sampled_func : A (dim, n_samples) array

    Returns
    -------

    """

    int_func = cumtrapz(sampled_func, samples)
    # Add zeros at the start
    zeros = np.zeros((int_func.shape[0], 1))
    int_func = np.hstack((zeros, int_func))
    return int_func - trapz(int_func, samples).reshape(-1, 1)
