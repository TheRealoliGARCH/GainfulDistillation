"""Scalar Bayesian balanced-loss variance correction.

This implements the one-dimensional specialization of Proposition 3.1 in
Li and Rice, "A Bayesian 'Sandwich' for Variance Estimation" (Statistical
Science, 2024).

For posterior variance V and scalar empirical Fisher information I_n(theta),
the posterior correction is

    Omega(theta) = (1/n) * sum_i score_i(theta)**2 / I_n(theta)
    Omega_hat = E_posterior[Omega(theta)]
    Sigma_hat = V * Omega_hat.

The caller supplies per-observation score values and the empirical average
negative Hessian through a callback. This module does not estimate those
model-specific quantities from raw data by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Callable, Sequence

from .posterior import Posterior1D


@dataclass(frozen=True)
class BalancedLossVariance1D:
    """Posterior point estimate and scalar robust variance correction."""

    point_estimate: float
    posterior_variance: float
    omega: float
    robust_variance: float
    robust_standard_error: float
    sample_size: int
    posterior_points: int


def balanced_loss_variance_1d(
    posterior: Posterior1D,
    score_information: Callable[[float], tuple[Sequence[float], float]],
) -> BalancedLossVariance1D:
    """Apply the scalar Bayesian balanced-inference-loss variance correction.

    For each theta on the posterior quadrature grid, score_information(theta)
    must return:
      * one finite score value per independent observation, and
      * the positive scalar empirical Fisher information I_n(theta), defined
        on the average-per-observation scale.

    The correction factor is computed at each grid point as the mean squared
    score divided by I_n(theta), then averaged using posterior cell masses.

    This is the scalar form of the published balanced-loss rule. It assumes
    independent observations, as in the cited development. It is not a
    multivariate implementation and should not be applied to clustered or
    dependent data without a corresponding derivation.
    """
    if not isinstance(posterior, Posterior1D):
        raise TypeError("posterior must be a Posterior1D result")
    if not callable(score_information):
        raise TypeError("score_information must be callable")

    omega_values: list[float] = []
    sample_size: int | None = None

    for theta in posterior.grid:
        try:
            result = score_information(theta)
        except Exception as exc:
            raise ValueError("score_information failed at a posterior grid point") from exc

        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError(
                "score_information must return (observation_scores, empirical_information)"
            )
        raw_scores, raw_information = result
        if isinstance(raw_scores, (str, bytes)):
            raise ValueError("observation_scores must be a non-empty numeric sequence")
        try:
            scores = tuple(float(value) for value in raw_scores)
            information = float(raw_information)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("scores and information must be real numbers") from exc

        if not scores:
            raise ValueError("observation_scores must contain at least one value")
        if any(not isfinite(value) for value in scores):
            raise ValueError("observation scores must all be finite")
        if not isfinite(information) or information <= 0.0:
            raise ValueError("empirical_information must be finite and strictly positive")
        if sample_size is None:
            sample_size = len(scores)
        elif len(scores) != sample_size:
            raise ValueError("score vector length must be constant across the posterior grid")

        mean_squared_score = fsum(value * value for value in scores) / len(scores)
        omega_theta = mean_squared_score / information
        if not isfinite(omega_theta) or omega_theta < 0.0:
            raise ValueError("score-information correction must be finite and non-negative")
        omega_values.append(omega_theta)

    assert sample_size is not None
    omega = fsum(
        mass * value
        for mass, value in zip(posterior.probability_masses, omega_values)
    )
    if not isfinite(omega) or omega <= 0.0:
        raise ValueError(
            "posterior expected score-information correction must be strictly positive"
        )

    robust_variance = posterior.variance * omega
    if not isfinite(robust_variance) or robust_variance < 0.0:
        raise ValueError("robust variance is not finite and non-negative")

    return BalancedLossVariance1D(
        point_estimate=posterior.mean,
        posterior_variance=posterior.variance,
        omega=omega,
        robust_variance=robust_variance,
        robust_standard_error=sqrt(robust_variance),
        sample_size=sample_size,
        posterior_points=posterior.points,
    )
