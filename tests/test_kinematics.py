import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import trapz
import pytest

from kinematic_snake.kinematic_snake import zero_mean_integral, project, KinematicSnake


class TestZeroMeanIntegral:
    @pytest.fixture(scope="class", params=[101, np.random.randint(100, 200)])
    def load_sample_points(self, request):
        bs = request.param
        sample_points = np.linspace(0.0, 1.0, bs)
        sample_points = sample_points.reshape(1, -1)
        return sample_points

    @pytest.mark.parametrize("dim", [1, 2])
    def test_integrity(self, load_sample_points, dim):
        sample_points = load_sample_points
        sampled_func = 1.0 + sample_points ** 2

        if dim == 2:
            sampled_func = np.vstack((sampled_func, sampled_func))

        zmi = zero_mean_integral(sampled_func, sample_points)
        assert zmi.shape == (dim, sample_points.shape[1])
        assert_allclose(trapz(zmi, sample_points), np.zeros((dim)), atol=1e-12)

    @pytest.mark.parametrize(
        "func_pairs",
        [
            (lambda x: np.sin(np.pi * x), lambda z: -np.cos(np.pi * z) / np.pi),
            (lambda x: 1.0 + x ** 3, lambda z: z + z ** 4 / 4.0 - 11.0 / 20.0),
        ],
    )
    def test_accuracy(self, load_sample_points, func_pairs):
        test_func = func_pairs[0]
        correct_func = func_pairs[1]

        sample_points = load_sample_points
        sampled_test_func = test_func(sample_points)
        zmi_of_test_func = zero_mean_integral(sampled_test_func, sample_points)
        sampled_correct_func = correct_func(sample_points)

        assert_allclose(zmi_of_test_func, sampled_correct_func, rtol=1e-4, atol=1e-4)


# TODO Add more tests
class TestProjection:
    @pytest.mark.parametrize("proj_axis", ["x", "y"])
    def test_accuracy_on_canonical_axes(self, proj_axis):
        vec = np.random.randn(2, 100)

        dir = [0.0, 0.0]
        if proj_axis == "x":
            ax_idx = 0
        elif proj_axis == "y":
            ax_idx = 1

        # Set unit vector on ax_idx
        dir[ax_idx] = 1.0

        ax = np.array(dir).reshape(-1, 1) + 0.0 * vec

        mag, p = project(vec, ax)
        assert_allclose(mag, vec[ax_idx, :])
        assert_allclose(np.flipud(p)[ax_idx, :], 0.0)

    def test_integrity_on_perpendicular_axes(self):
        vec = np.random.randn(2, 100)

        theta = np.random.uniform(0.0, 2.0 * np.pi)
        ax = [np.cos(theta), np.sin(theta)]
        perp_ax = [-np.sin(theta), np.cos(theta)]

        axis = np.array(ax).reshape(-1, 1) + 0.0 * vec
        perp_axis = np.array(perp_ax).reshape(-1, 1) + 0.0 * vec

        mag_par, p_par = project(vec, axis)
        mag_perp, p_perp = project(vec, perp_axis)

        # magnitues
        mag_vec_sq = np.einsum("ij,ij->j", vec, vec)
        assert_allclose(mag_par ** 2 + mag_perp ** 2, mag_vec_sq)

        # directions
        mag_pr, _ = project(p_par, p_perp)
        assert_allclose(mag_pr, 0.0 * mag_pr, rtol=1e-8, atol=1e-12)


class TestKinematicSnake:
    @pytest.fixture(scope="class")
    def load_snake(self):
        froude_number = 0.1
        friction_coefficients = {"mu_f": 0.1, "mu_b": 0.2, "mu_lat": 0.3}

        def activation(s, time):
            """Flat snake"""
            return 0.0 * (s + time)

        snake = KinematicSnake(
            froude_number=froude_number,
            friction_coefficients=friction_coefficients,
            samples=50,
        )
        snake.set_activation(activation)
        snake._construct(0.0)
        return snake, friction_coefficients

    def test_forward_friction(self, load_snake):
        snake, fc = load_snake

        # Moves forward while snake faces forward
        # Activates only forward friction forces
        snake.dx_dt[0, ...] = 1.0
        snake.dx_dt[1, ...] = 0.0

        friction_force = snake.external_force_distribution(time=0.0)
        expected_force = 0.0 * snake.dx_dt
        expected_force[0, :] = -fc["mu_f"]
        assert_allclose(friction_force, expected_force)

        friction_torque = snake.external_torque_distribution(friction_force)
        expected_torque = 0.0 * friction_torque
        assert_allclose(friction_torque, expected_torque)

    def test_backward_friction(self, load_snake):
        snake, fc = load_snake

        # Moves backward while snake faces forward
        # Activates only forward friction forces
        snake.dx_dt[0, ...] = -1.0
        snake.dx_dt[1, ...] = 0.0

        friction_force = snake.external_force_distribution(time=0.0)
        expected_force = 0.0 * snake.dx_dt
        expected_force[0, :] = fc["mu_b"]
        assert_allclose(friction_force, expected_force)

        friction_torque = snake.external_torque_distribution(friction_force)
        expected_torque = 0.0 * friction_torque
        assert_allclose(friction_torque, expected_torque)

    def test_lateral_friction(self, load_snake):
        snake, fc = load_snake

        # Faces y axes, rolling frinction only!
        snake.dx_dt[0, ...] = 0.0
        snake.dx_dt[1, ...] = 1.0

        friction_force = snake.external_force_distribution(time=0.0)
        expected_force = 0.0 * snake.dx_dt
        expected_force[1, :] = -fc["mu_lat"]
        assert_allclose(friction_force, expected_force)

        friction_torque = snake.external_torque_distribution(friction_force)
        expected_torque = 0.0
        assert_allclose(np.sum(friction_torque), expected_torque, atol=1e-6)

    def test_moment_of_inertia_accuracy(self, load_snake):
        """For a straight rod, compare against analytical solution

        Parameters
        ----------
        load_snake

        Returns
        -------

        """

        snake, fc = load_snake

        moment_of_inertia = snake.calculate_moment_of_inertia()
        # 2 * (l/2)^3 / 3.0
        expected_moment_of_inertia = 2.0 * (1.0 / 2.0) ** 3 / 3.0

        assert_allclose(
            moment_of_inertia, expected_moment_of_inertia, rtol=1e-4, atol=1e-4
        )

    def test_internal_torque(self, load_snake):
        snake, fc = load_snake

        internal_torque = snake.internal_torque_distribution(time=0.0)
        expected_torque = 0.0 * internal_torque

        assert_allclose(internal_torque, expected_torque, atol=1e-6)


def generate_unevenly_spaced_time_series(start=0.0, stop=2.0 * np.pi, size=(35,)):
    assert start < stop
    local_start = start
    local_stop = 0.5 * (start + stop)
    t = np.random.uniform(low=local_start, high=local_stop, size=size)

    def mirror(x):
        return 2.0 * local_stop - x

    t = np.hstack((np.array(start), t, mirror(t[::-1]), np.array(stop)))
    t = np.sort(t)
    return t
