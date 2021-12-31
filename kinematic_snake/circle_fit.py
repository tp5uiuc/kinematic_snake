__doc__ = """ Fits a set of 2D data points (x,y) to a circle

See https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
From https://dtcenter.org/met/users/docs/write_ups/circle_fit.pdf
For more complicated shapes, see https://arxiv.org/pdf/cs/0301001.pdf
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import linregress


def fit_circle_to_data(in_data, verbose=False):
    """
    Wrapper around _fit_circle_impl that takes care of unwrapping shapes and so on...
    Returns
    -------

    """
    slope, intercept, r_value, p_value, std_err = linregress(in_data[0], in_data[1])

    if verbose:
        print("R-square of linear fit : {}".format(r_value ** 2))
        print("std-error of linear fit : {}".format(std_err))

    if r_value ** 2 < 0.98 and std_err > 1e-3:
        xc, yc, r, opt = circle_fit_impl(in_data)

        if verbose:
            print("Number of function evaluations : {}".format(opt.nfev))
            print("Number of Jacobian evaluations : {}".format(opt.njev))
            print("Stddev. residuals at optimal : {}".format(np.std(opt.fun)))

        return xc, yc, r
    else:
        # It's almost a straight line, so return all infinitirys
        return np.inf, np.inf, np.inf


def circle_fit_impl(x):
    """(2,n) array as input"""

    def distance_from_center(xc, yc):
        """calculate the distance of each 2D points from the center (xc, yc)"""
        return np.sqrt((x[0] - xc) ** 2 + (x[1] - yc) ** 2)

    def objective(c):
        """calculates the objective : algebraic distance between the data points
        and the mean circle centered at c=(xc, yc)

        Maps from R^2 -> R^n where n is the number of points to map
        """
        Ri = distance_from_center(*c)
        return Ri - Ri.mean()

    def jacobian_objective(c):
        """Jacobian of above objective function
        By definition from scipy J_{ij} = \\partial f_{i}/ \\partial x_{j}
        Hence J is a (n,2) matrix below

        Parameters
        ----------
        c

        Returns
        -------

        """
        xc, yc = c
        df2b_dc = np.empty((x.shape[-1], c.shape[0]))

        r_i = distance_from_center(xc, yc)
        df2b_dc[..., 0] = (xc - x[0]) / r_i  # dR/dxc
        df2b_dc[..., 1] = (yc - x[1]) / r_i  # dR/dyc
        df2b_dc -= np.mean(df2b_dc, axis=0)

        return df2b_dc

    # estimate center via mean of data
    center_estimate = np.mean(x, axis=-1)
    optimum_results = least_squares(
        objective, center_estimate, jac=jacobian_objective, method="lm"
    )

    center_optimized = optimum_results.x
    radius_samples = distance_from_center(*center_optimized)
    radius = np.mean(radius_samples)
    return (*center_optimized, radius, optimum_results)
