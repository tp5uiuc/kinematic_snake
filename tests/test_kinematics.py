import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import trapz
import pytest

from kinematic_snake.kinematic_snake import zero_mean_integral


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