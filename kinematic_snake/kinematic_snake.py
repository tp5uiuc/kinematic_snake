__doc__ = """ Core concepts """

import numpy as np
from functools import partial
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
    # Add zeros at the start and make a bigger array
    int_func_with_initial_values = np.zeros(
        int_func.shape[:-1] + (int_func.shape[-1] + 1,)
    )
    int_func_with_initial_values[..., 1:] = int_func
    return int_func_with_initial_values - trapz(
        int_func_with_initial_values, samples
    ).reshape(-1, 1)
