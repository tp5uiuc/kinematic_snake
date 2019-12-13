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


def project(unit_vec, proj_axis):
    mag_projection = np.einsum("ij, ij -> j", unit_vec, proj_axis)
    projection = mag_projection * proj_axis
    return mag_projection, projection


class KinematicSnake:
    """
    States are COM (x, y, theta, xdot, ydot, thetadot)
    """

    def __init__(
        self, *, froude_number: float, friction_coefficients: dict, samples=200
    ):
        self.froude = froude_number if froude_number > 0.0 else -froude_number
        self.curvature_activation = None
        self.forward_mu = friction_coefficients.get("mu_f", 1.0)
        self.backward_mu = friction_coefficients.pop("mu_b")
        self.lateral_mu = friction_coefficients.pop("mu_lat")
        self.samples = samples
        self.centerline = np.linspace(0.0, 1.0, samples).reshape(1, -1)
        self.zmi_along_centerline = partial(zero_mean_integral, samples=self.centerline)

        # Set via
        dofs = 3
        self.state = np.zeros((dofs * 2,))

    def set_activation(self, func):
        def broadcast(fun):
            """ Required because sp.diff of a constant function
            returns by default a constant value (float) and not
            an array of floats as one would expect.

            Obtained from:
            https://github.com/sympy/sympy/issues/5642#issuecomment-516419510

            Parameters
            ----------
            fun

            Returns
            -------

            """
            return lambda *x: np.broadcast_arrays(fun(*x), *x)[0]

        if func.__call__ is not None:
            import sympy as sp

            t, s = sp.symbols("t, s")
            kappa = func(t, s)
            kappa_rate = sp.diff(kappa, t)
            kappa_acc = sp.diff(kappa_rate, t)
            self.curvature_activation = broadcast(sp.lambdify([t, s], kappa, "numpy"))
            self.dcurvature_activation_dt = broadcast(
                sp.lambdify([t, s], kappa_rate, "numpy")
            )
            self.d2curvature_activation_dt2 = broadcast(
                sp.lambdify([t, s], kappa_acc, "numpy")
            )
        else:
            raise RuntimeError("Activation function not callable")

    def _construct(self, time):

        # First construct theta(s,t) and \frac{\partial X}{\partial s} from eq (2b)
        theta_com = self.state[2]
        self.theta_s = theta_com + self.zmi_along_centerline(
            self.curvature_activation(time, self.centerline)
        )
        self.dx_ds = np.vstack((np.cos(self.theta_s), np.sin(self.theta_s)))

        # The cross operator \perp of \frac{\partial X}{\partial s}
        # Can do cross with [0., 0., 1], but why bother?
        dx_ds_perp = 0.0 * self.dx_ds
        dx_ds_perp[0, ...] = -self.dx_ds[1, ...]
        dx_ds_perp[1, ...] = self.dx_ds[0, ...]

        # Reconstruct x(s,t) from (2a)
        x_com = self.state[:2].reshape(-1, 1).copy()
        self.x_s = x_com + self.zmi_along_centerline(self.dx_ds)

        # Get \frac{\partial \theta}{\partial t} from (3b)
        theta_dot_at_com = self.state[5]
        self.dtheta_dt = theta_dot_at_com + self.zmi_along_centerline(
            self.dcurvature_activation_dt(time, self.centerline)
        )

        # Get \frac{\partial X}{\partial t} from (3a)
        x_dot_at_com = self.state[3:5].reshape(-1, 1).copy()
        self.dx_dt = x_dot_at_com + self.zmi_along_centerline(
            dx_ds_perp * self.dtheta_dt
        )

        return dx_ds_perp

    def non_dim_friction_force(self, time):
        dx_ds_perp = self._construct(time)

        mag_dx_dt = np.sqrt(np.einsum("ij,ij->j", self.dx_dt, self.dx_dt))
        normalized_dx_dt = self.dx_dt / mag_dx_dt

        mag_proj_along_normal, proj_along_normal = project(normalized_dx_dt, dx_ds_perp)
        mag_proj_along_tangent, proj_along_tangent = project(
            normalized_dx_dt, self.dx_ds
        )

        is_forward_friction_active = np.heaviside(mag_proj_along_tangent, 0.5)

        # friction according to eq.(8)
        friction_force = (
            self.lateral_mu * proj_along_normal
            + (
                self.forward_mu * is_forward_friction_active
                + self.backward_mu * (1.0 - is_forward_friction_active)
            )
            * proj_along_tangent
        )
        return -friction_force

    def __call__(self, time, state, *args, **kwargs):
        pass
