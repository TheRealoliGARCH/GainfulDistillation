import pytest

from gds_core.sandwich import sandwich_covariance


def test_diagonal_sandwich_and_sample_scaling():
    result = sandwich_covariance(
        [[2.0, 0.0], [0.0, 4.0]],
        [[8.0, 0.0], [0.0, 16.0]],
        sample_size=10,
    )
    assert result.asymptotic[0][0] == pytest.approx(2.0)
    assert result.asymptotic[0][1] == pytest.approx(0.0)
    assert result.asymptotic[1][0] == pytest.approx(0.0)
    assert result.asymptotic[1][1] == pytest.approx(1.0)
    assert result.estimator_covariance[0][0] == pytest.approx(0.2)
    assert result.estimator_covariance[1][1] == pytest.approx(0.1)
    assert result.sample_size == 10


def test_non_diagonal_sandwich_is_symmetric():
    result = sandwich_covariance(
        [[2.0, 0.0], [0.0, 1.0]],
        [[4.0, 2.0], [2.0, 9.0]],
    )
    assert result.asymptotic[0][1] == pytest.approx(result.asymptotic[1][0])
    assert result.estimator_covariance is None


@pytest.mark.parametrize(
    "a,b",
    [
        ([[1.0, 2.0]], [[1.0]]),
        ([[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]),
        ([[1.0]], [[1.0, 2.0], [2.0, 1.0]]),
        ([[1.0]], [[1.0, float("nan")], [float("nan"), 1.0]]),
        ([[1.0]], [[1.0, 0.5], [0.0, 1.0]]),
    ],
)
def test_invalid_matrices_are_rejected(a, b):
    with pytest.raises(ValueError):
        sandwich_covariance(a, b)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_sample_size_is_rejected(n):
    with pytest.raises(ValueError):
        sandwich_covariance([[1.0]], [[1.0]], sample_size=n)
