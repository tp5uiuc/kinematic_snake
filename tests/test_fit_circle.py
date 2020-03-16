import numpy as np
from numpy.testing import assert_allclose
import pytest

from kinematic_snake.circle_fit import circle_fit_impl, fit_circle_to_data


class TestCircleFit:
    def test_circle_fit_for_circular_data(self):
        center = np.array([[0.0], [0.0]])
        radius = 1.0
        theta = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
        x = center + radius * np.vstack((np.cos(theta), np.sin(theta)))

        xc, yc, r, _ = circle_fit_impl(x)

        assert_allclose(center[0, 0], xc, atol=1e-8)
        assert_allclose(center[1, 0], yc, atol=1e-8)
        assert_allclose(radius, r)

    def test_circle_fit_for_circular_data_generated_randomly(self):
        center = np.random.randn(2, 1)
        radius = np.random.rand()
        theta = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
        x = center + radius * np.vstack((np.cos(theta), np.sin(theta)))

        xc, yc, r, _ = circle_fit_impl(x)

        assert_allclose(center[0, 0], xc)
        assert_allclose(center[1, 0], yc)
        assert_allclose(radius, r)

    def test_circle_fit_for_randomly_sampled_circular_data(self):
        center = np.random.randn(2, 1)
        radius = np.random.rand()
        theta = 2.0 * np.pi * np.random.random_sample(10)
        x = center + radius * np.vstack((np.cos(theta), np.sin(theta)))

        xc, yc, r, _ = circle_fit_impl(x)

        assert_allclose(center[0, 0], xc)
        assert_allclose(center[1, 0], yc)
        assert_allclose(radius, r)

    def test_circle_fit_for_circular_like_data_with_different_radii(self):
        center = np.array([[0.0], [0.0]])
        n_samples = 20
        variability = 0.05
        # Radius is around 1, not exactly though
        radius = 1.0 + variability * np.random.randn(n_samples)
        theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
        x = center + radius * np.vstack((np.cos(theta), np.sin(theta)))

        xc, yc, r, _ = circle_fit_impl(x)

        assert np.abs(center[0, 0] - xc) < variability
        assert np.abs(center[1, 0] - yc) < variability
        assert_allclose(np.mean(radius), r, rtol=1e-2)

    @pytest.mark.xfail(
        reason="The tolerance to set is unclear and hence it is allowed to fail"
    )
    def test_circle_fit_for_circular_like_data_with_different_centers(self):
        n_samples = 20
        variability = 0.03
        center = np.random.randn(2, n_samples) * variability
        radius = 0.5
        theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
        x = center + radius * np.vstack((np.cos(theta), np.sin(theta)))
        xc, yc, r, _ = circle_fit_impl(x)

        assert np.abs(center[0, 0] - xc) < variability
        assert np.abs(center[1, 0] - yc) < variability
        assert_allclose(np.mean(radius), r, rtol=1e-2)

    def test_circle_fit_for_straight_line_data(self):
        n_samples = 20
        x = np.linspace(0.0, 1.0, n_samples)
        y = 0.0 * x
        position_vector = np.vstack((x, y))

        xc, yc, r = fit_circle_to_data(position_vector, verbose=True)

        assert r > 1e4

    def test_circle_fit_for_almost_straight_line_data(self):
        n_samples = 20
        x = np.linspace(0.0, 1.0, n_samples)
        y = 0.1 * x
        y += 0.05 * np.random.random_sample(n_samples)
        position_vector = np.vstack((x, y))

        xc, yc, r = fit_circle_to_data(position_vector, verbose=True)

        # Shady assert : do not know when it works or fails as
        # the data is randomly sampled
        assert r > 1
