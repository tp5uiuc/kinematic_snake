__doc__ = """ Core snake """

import numpy as np
from functools import partial
from scipy.integrate import trapz, cumtrapz


def zero_mean_integral(sampled_func, samples):
    """Zero mean integral from 0 to 1 always.
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
        self,
        *,
        froude_number: float,
        friction_coefficients: dict,
        samples=300,
        **kwargs
    ):
        self.froude = froude_number if froude_number > 0.0 else -froude_number
        self.curvature_activation = None
        self.forward_mu = friction_coefficients.get("mu_f", None)
        self.backward_mu = friction_coefficients.get("mu_b", None)
        self.lateral_mu = friction_coefficients.get("mu_lat", None)
        self.samples = samples
        self.centerline = np.linspace(0.0, 1.0, samples).reshape(1, -1)
        self.zmi_along_centerline = partial(zero_mean_integral, samples=self.centerline)
        print(
            "Setup {snake_type} with Fr {1}, mu_f {0}, mu_b {2}, mu_lat {3} with dimensions {4}".format(
                self.forward_mu,
                self.froude,
                self.backward_mu,
                self.lateral_mu,
                samples,
                snake_type=self.__class__.__name__,
            )
        )

        # Set via
        dofs = 3
        self.state = np.zeros((dofs * 2, 1))
        # Setting initial x-velocity to be a very high number
        # self.state[3, ...] = 1.0

        try:
            self.set_activation(kwargs.pop("activation"))
        except KeyError:
            pass

    def set_activation(self, func):
        def broadcast(fun):
            """Required because sp.diff of a constant function
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
            kappa = func(s, t)
            kappa_rate = sp.diff(kappa, t)
            kappa_acc = sp.diff(kappa_rate, t)
            self.curvature_activation = partial(
                broadcast(sp.lambdify([s, t], kappa, "numpy")), self.centerline
            )
            self.dcurvature_activation_dt = partial(
                broadcast(sp.lambdify([s, t], kappa_rate, "numpy")), self.centerline
            )
            self.d2curvature_activation_dt2 = partial(
                broadcast(sp.lambdify([s, t], kappa_acc, "numpy")), self.centerline
            )
        else:
            raise RuntimeError("Activation function not callable")

        # Once activated, construct all quantities at the initial time
        self._construct(0.0)

    def _construct(self, time):
        # First construct theta(s,t) and \frac{\partial X}{\partial s} from eq (2b)
        theta_com = self.state[2, ...]
        self.theta_s = theta_com + self.zmi_along_centerline(
            self.curvature_activation(time)
        )
        self.dx_ds = np.vstack((np.cos(self.theta_s), np.sin(self.theta_s)))

        # The cross operator \perp of \frac{\partial X}{\partial s}
        # Can do cross with [0., 0., 1], but why bother?
        self.dx_ds_perp = np.empty_like(self.dx_ds)
        self.dx_ds_perp[0, ...] = -self.dx_ds[1, ...]
        self.dx_ds_perp[1, ...] = self.dx_ds[0, ...]

        # Reconstruct x(s,t) from (2a)
        x_com = self.state[:2, ...]
        self.x_s = x_com + self.zmi_along_centerline(self.dx_ds)

        # Get \frac{\partial \theta}{\partial t} from (3b)
        theta_dot_com = self.state[5, ...]
        self.dtheta_dt = theta_dot_com + self.zmi_along_centerline(
            self.dcurvature_activation_dt(time)
        )

        # Get \frac{\partial X}{\partial t} from (3a)
        x_dot_com = self.state[3:5, ...]
        self.dx_dt = x_dot_com + self.zmi_along_centerline(
            self.dx_ds_perp * self.dtheta_dt
        )

    def __calculate_vector_in_velocity_direction(self):
        dx_dt_com = self.state[3:5, ...].reshape(
            -1,
        )
        mag_dx_dt_com = np.linalg.norm(dx_dt_com)
        return dx_dt_com, mag_dx_dt_com

    def __calculate_pose_impl(self, dx_dt_com, mag_dx_dt_com):
        theta_com = self.state[2, ...]
        unit_vector_in_average_orientation_direction = np.array(
            [np.cos(theta_com), np.sin(theta_com)]
        ).reshape(
            -1,
        )
        with np.errstate(invalid="ignore"):
            # Similar to doing an arctan in relative coordinates
            signbit = np.sign(
                np.cross(unit_vector_in_average_orientation_direction, dx_dt_com)
            )
            dot_product = (
                np.inner(unit_vector_in_average_orientation_direction, dx_dt_com)
                / mag_dx_dt_com
            )
            # normalized_ = np.linalg.norm(dx_dt_com/mag_dx_dt_com)
            # if np.isfinite(normalized_) : assert (np.allclose(normalized_, 1.0))
            return signbit * np.arccos(dot_product)

    def calculate_instantaneous_pose_angle(self, time=0.0):
        """
        Angle between the instantaneous velocity vector and
        the average angle (theta_com) of the snake

        Returns
        -------

        """
        dx_dt_com, mag_dx_dt_com = self.__calculate_vector_in_velocity_direction()
        return self.__calculate_pose_impl(dx_dt_com, mag_dx_dt_com)

    def calculate_instantaneous_pose_rate(self, time):
        """Checked with a finite-difference version, seems to match-up

        Parameters
        ----------
        time

        Returns
        -------

        """
        dx_dt_com, mag_dx_dt_com = self.__calculate_vector_in_velocity_direction()

        pose_angle = self.__calculate_pose_impl(dx_dt_com, mag_dx_dt_com)

        # Linear accelerations, a (2,) array
        external_force_distribution = self.external_force_distribution(time)
        linear_acceleration = (
            trapz(external_force_distribution, self.centerline) / self.froude
        ).reshape(
            -1,
        )

        theta_com = self.state[2, ...]
        unit_vector_in_average_orientation_direction = np.array(
            [np.cos(theta_com), np.sin(theta_com)]
        ).reshape(
            -1,
        )
        first_rhs = np.inner(
            linear_acceleration, unit_vector_in_average_orientation_direction
        )

        unit_vector_perp_to_average_orientation = np.array(
            [-np.sin(theta_com), np.cos(theta_com)]
        ).reshape(
            -1,
        )
        second_rhs = (
            np.inner(dx_dt_com, unit_vector_perp_to_average_orientation)
            * self.state[5, 0]
        )  # last term is theta_dot_com as state is a (6,1) array

        with np.errstate(invalid="ignore"):
            mod_dx_dt_dot = np.inner(dx_dt_com, linear_acceleration) / mag_dx_dt_com
        term_from_lhs = np.cos(pose_angle) * mod_dx_dt_dot

        return (
            -(first_rhs + second_rhs - term_from_lhs)
            / mag_dx_dt_com
            / np.sin(pose_angle)
        )

    def calculate_instantaneous_steering_angle(self, time=0.0):
        return (
            self.state[2, 0] + self.calculate_instantaneous_pose_angle() - 0.5 * np.pi
        )

    def calculate_instantaneous_steering_rate(self, time):
        return self.state[5, 0] + self.calculate_instantaneous_pose_rate(time)

    def calculate_moment_of_inertia(self):
        # distribution of moments first, (x-x_com)^T (x - x_com)
        x_minus_xcom = self.x_s - self.state[:2, ...]
        moment_of_inertia_distribution = np.einsum(
            "ij,ij->j", x_minus_xcom, x_minus_xcom
        )
        return trapz(moment_of_inertia_distribution, self.centerline)

    def external_force_distribution(self, time):
        mag_dx_dt = np.sqrt(np.einsum("ij,ij->j", self.dx_dt, self.dx_dt))
        normalized_dx_dt = self.dx_dt / (mag_dx_dt)

        _, proj_along_normal = project(normalized_dx_dt, self.dx_ds_perp)
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

        # ## Purely meant for testing purposes, ignore
        # friction_force_lateral = ( self.lateral_mu * proj_along_normal)
        # friction_force_fwd =  self.forward_mu * np.heaviside(mag_proj_along_tangent, 0.5) * proj_along_tangent
        # friction_force_bwd =  self.backward_mu * np.heaviside(-mag_proj_along_tangent, 0.5) * proj_along_tangent
        # friction_force = friction_force_lateral + friction_force_fwd + friction_force_bwd
        return -friction_force

    def external_torque_distribution(self, ext_force):
        x_minus_xcom = self.x_s - self.state[:2, ...]
        external_torque_arm = np.empty_like(x_minus_xcom)
        external_torque_arm[0, ...] = -x_minus_xcom[1, ...]
        external_torque_arm[1, ...] = x_minus_xcom[0, ...]
        return np.einsum("ij,ij->j", external_torque_arm, ext_force)

    def internal_torque_distribution(self, time):
        # Common to both terms on the RHS
        zmi_dx_ds_perp = self.zmi_along_centerline(self.dx_ds_perp)

        # Last term of first part in RHS
        zmi_dx_ds_times_dtheta_dt_squared = self.zmi_along_centerline(
            self.dx_ds * (self.dtheta_dt ** 2)
        )

        # Last term of second part in RHS
        zmi_d2_curvature_dt2 = self.zmi_along_centerline(
            self.d2curvature_activation_dt2(time)
        )
        zmi_dx_ds_perp_times_zmi_acc_curvature = self.zmi_along_centerline(
            self.dx_ds_perp * zmi_d2_curvature_dt2
        )

        return np.einsum(
            "ij,ij->j",
            zmi_dx_ds_perp,
            zmi_dx_ds_times_dtheta_dt_squared - zmi_dx_ds_perp_times_zmi_acc_curvature,
        )

    def __call__(self, time, state, *args, **kwargs):
        self.state = state.copy().reshape(-1, 1)

        # Construct snake properties from state at current time t
        self._construct(time)

        # Linear accelerations, a (2,) array
        external_force_distribution = self.external_force_distribution(time)
        linear_acceleration = (
            trapz(external_force_distribution, self.centerline) / self.froude
        )

        # angular accelerations
        inv_moment_of_inertia = 1.0 / self.calculate_moment_of_inertia()
        external_torque_distribution = self.external_torque_distribution(
            external_force_distribution
        )

        angular_acceleration = inv_moment_of_inertia * (
            trapz(
                external_torque_distribution / self.froude
                + self.internal_torque_distribution(time),
                self.centerline,
            )
        )

        dstate_dt = np.empty_like(state)

        # Copy velocities at the front
        dstate_dt[:3] = state[3:]

        # accelerations at the back
        dstate_dt[3:5] = linear_acceleration
        dstate_dt[5] = angular_acceleration

        return dstate_dt


class LiftingKinematicSnake(KinematicSnake):
    def __init__(
        self,
        *,
        froude_number: float,
        friction_coefficients: dict,
        samples=300,
        **kwargs
    ):
        super().__init__(
            froude_number=froude_number,
            friction_coefficients=friction_coefficients,
            samples=samples,
            **kwargs
        )
        self.lifting_activation = None

    def set_lifting_activation(self, func):
        # lifting activation is only a function of s and t
        self.lifting_activation = partial(func, self.centerline)

    def external_force_distribution(self, time):
        friction_forces = super().external_force_distribution(time)
        return self.lifting_activation(time) * friction_forces
