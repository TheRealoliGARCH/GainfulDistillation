"""Sinc covariance kernels for band-limited Gaussian-process models.

The Sinc function is used here as a covariance kernel, not as a probability
density. The spectral construction follows the band-limited Gaussian-process
formulation of Tobar (2019). The covariance is

    K(t, t') = variance * sinc(bandwidth * (t - t'))
               * cos(2*pi*center_frequency * (t - t')).

A Sinc function by itself changes sign and is not a valid density. The
covariance construction has a non-negative rectangular spectral density and
is positive semidefinite for the parameter restrictions enforced below.
"""
from math import cos, isfinite, pi, sin
from typing import Sequence


def normalized_sinc(value: float) -> float:
    """Return sin(pi*x)/(pi*x), with its continuous value at zero."""
    x = float(value)
    if not isfinite(x):
        raise ValueError("value must be finite")
    if x == 0.0:
        return 1.0
    return sin(pi * x) / (pi * x)


def sinc_covariance(
    left: float,
    right: float,
    *,
    bandwidth: float,
    center_frequency: float = 0.0,
    variance: float = 1.0,
) -> float:
    """Evaluate a stationary band-limited Sinc covariance kernel."""
    bandwidth = float(bandwidth)
    center_frequency = float(center_frequency)
    variance = float(variance)
    left = float(left)
    right = float(right)
    if not all(isfinite(x) for x in (bandwidth, center_frequency, variance, left, right)):
        raise ValueError("all kernel arguments must be finite")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be strictly positive")
    if center_frequency < 0.0:
        raise ValueError("center_frequency must be non-negative")
    if variance < 0.0:
        raise ValueError("variance must be non-negative")
    lag = left - right
    return variance * normalized_sinc(bandwidth * lag) * cos(2.0 * pi * center_frequency * lag)


def sinc_covariance_matrix(
    times: Sequence[float],
    *,
    bandwidth: float,
    center_frequency: float = 0.0,
    variance: float = 1.0,
) -> tuple[tuple[float, ...], ...]:
    """Return the Gram/covariance matrix for observation times."""
    values = tuple(float(t) for t in times)
    if not values:
        raise ValueError("times must contain at least one value")
    if any(not isfinite(t) for t in values):
        raise ValueError("times must all be finite")
    return tuple(
        tuple(
            sinc_covariance(
                left, right,
                bandwidth=bandwidth,
                center_frequency=center_frequency,
                variance=variance,
            )
            for right in values
        )
        for left in values
    )
