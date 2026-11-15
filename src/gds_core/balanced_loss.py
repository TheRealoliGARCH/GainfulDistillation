"""Scalar Bayesian balanced-loss variance correction.

This implements the one-dimensional specialization of Proposition 3.1 in
Li and Rice, "A Bayesian 'Sandwich' for Variance Estimation" (Statistical
Science, 2024).

For posterior variance V and scalar empirical Fisher information I_n(theta),
the posterior correction is

    Omega(theta) = (1/n) * sum_i score_i(theta)**2 / I_n(theta)
    Omega_hat = E_posterior[Omega(theta)]
    Sigma_hat = V * Omega_hat.

The caller supplies model-specific score and empirical-information values.
For efficiency, it may provide either the individual scores or their exact
mean squared value and sample size.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Callable, Sequence

from .posterior import Posterior1D


@dataclass(frozen=True)
class ScalarScoreSummary:
    """Sufficient scalar score summary for the balanced-loss correction.

    mean_squared_score must equal n**-1 * sum_i score_i(theta)**2.
    This summary is exact for the scalar loss; it does not approximate the
    score calculation. It avoids materialising all individual scores when a
    model can compute the mean square from sufficient statistics.
    """

    sample_size: int
    mean_squared_score: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
            or self.sample_size < 1
        ):
            raise ValueError("sample_size must be a positive integer")
        if isinstance(self.mean_squared_score, bool):
            raise ValueError("mean_squared_score must be finite and non-negative")
        try:
            value = float(self.mean_squared_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("mean_squared_score must be finite and non-negative") from exc
        if not isfinite(value) or value < 0.0:
            raise ValueError("mean_squared_score must be finite and non-negative")
        object.__setattr__(self, "mean_squared_score", value)


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
    score_information: Callable[
        [float], tuple[Sequence[float] | ScalarScoreSummary, float]
    ],
) -> BalancedLossVariance1D:
    """Apply the scalar Bayesian balanced-inference-loss variance correction.

    For each theta on the posterior quadrature grid, score_information(theta)
    must return:
      * either one finite score per independent observation, or a
        ScalarScoreSummary containing their exact mean square and sample size;
      * the positive scalar empirical Fisher information I_n(theta), defined
        on the average-per-observation scale.

    The correction at each grid point is the mean squared score divided by
    I_n(theta), then averaged using posterior cell masses.

    This scalar form assumes independent observations, as in the cited
    development. It is not a multivariate implementation and should not be
    applied to clustered or dependent data without a corresponding derivation.
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
                "score_information must return (observation_scores_or_summary, empirical_information)"
            )
        raw_scores, raw_information = result
        try:
            information = float(raw_information)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("empirical information must be a real number") from exc

        if not isfinite(information) or information <= 0.0:
            raise ValueError("empirical_information must be finite and strictly positive")

        if isinstance(raw_scores, ScalarScoreSummary):
            current_n = raw_scores.sample_size
            mean_squared_score = raw_scores.mean_squared_score
        else:
            if isinstance(raw_scores, (str, bytes)):
                raise ValueError("observation_scores must be a non-empty numeric sequence")
            try:
                scores = tuple(float(value) for value in raw_scores)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("scores must be real numbers") from exc
            if not scores:
                raise ValueError("observation_scores must contain at least one value")
            if any(not isfinite(value) for value in scores):
                raise ValueError("observation scores must all be finite")
            current_n = len(scores)
            mean_squared_score = fsum(value * value for value in scores) / current_n

        if sample_size is None:
            sample_size = current_n
        elif current_n != sample_size:
            raise ValueError("score vector length must be constant across the posterior grid")

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
