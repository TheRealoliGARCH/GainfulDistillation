import math

import pytest

from gds_core.sinc_kernel import normalized_sinc, sinc_covariance, sinc_covariance_matrix


def test_normalized_sinc_at_zero_and_nonzero():
    assert normalized_sinc(0.0) == 1.0
    assert normalized_sinc(1.0) == pytest.approx(0.0, abs=1e-15)
    assert normalized_sinc(0.5) == pytest.approx(2.0 / math.pi)


def test_sinc_covariance_variance_and_stationarity():
    assert sinc_covariance(3.0, 3.0, bandwidth=2.0, variance=4.0) == 4.0
    assert sinc_covariance(1.0, 2.0, bandwidth=1.0) == sinc_covariance(
        11.0, 12.0, bandwidth=1.0
    )


def test_frequency_modulation_and_covariance_matrix():
    value = sinc_covariance(0.0, 1.0, bandwidth=0.5, center_frequency=0.25)
    assert value == pytest.approx(0.0, abs=1e-15)
    matrix = sinc_covariance_matrix(
        [0.0, 1.0, 2.0], bandwidth=0.5, center_frequency=0.25, variance=2.0
    )
    assert len(matrix) == 3
    assert all(matrix[i][j] == pytest.approx(matrix[j][i]) for i in range(3) for j in range(3))
    assert all(matrix[i][i] == 2.0 for i in range(3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bandwidth": 0.0},
        {"bandwidth": -1.0},
        {"bandwidth": 1.0, "center_frequency": -0.1},
        {"bandwidth": 1.0, "variance": -1.0},
        {"bandwidth": float("nan")},
    ],
)
def test_invalid_kernel_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        sinc_covariance(0.0, 1.0, **kwargs)


def test_covariance_matrix_requires_finite_nonempty_times():
    with pytest.raises(ValueError):
        sinc_covariance_matrix([], bandwidth=1.0)
    with pytest.raises(ValueError):
        sinc_covariance_matrix([0.0, float("inf")], bandwidth=1.0)
