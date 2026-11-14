"""Deterministic one-dimensional Bayesian posterior by midpoint quadrature.

The parameter domain is an explicit finite interval. The prior is conditioned
on that interval before the likelihood is applied. The returned marginal
likelihood is therefore conditional on the declared parameter domain.

This is a transparent low-dimensional reference implementation for testing
priors and likelihoods. It is not a general multivariate posterior sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite, log
from typing import Callable

from .sinc_prior import SincModulatedPrior, SincPriorMixture


@dataclass(frozen=True)
class Posterior1D:
    """Posterior approximation on equally spaced midpoint cells.

    probability_masses are cell probabilities and sum to one.
    density contains the corresponding piecewise-constant cell density
    values at the cell midpoints. grid contains those midpoints.
    """

    bounds: tuple[float, float]
    grid: tuple[float, ...]
    density: tuple[float, ...]
    probability_masses: tuple[float, ...]
    mean: float
    variance: float
    standard_deviation: float
    credible_level: float
    credible_interval: tuple[float, float]
    log_marginal_likelihood: float
    points: int
    cell_width: float

    @property
    def integrated_density(self) -> float:
        """Return the midpoint-quadrature integral of the posterior density."""
        return fsum(self.density) * self.cell_width


def _logsumexp(values: tuple[float, ...]) -> float:
    finite_values = tuple(value for value in values if value != float("-inf"))
    if not finite_values:
        return float("-inf")
    maximum = max(finite_values)
    return maximum + log(fsum(exp(value - maximum) for value in finite_values))


def _quantile_from_cell_masses(
    masses: tuple[float, ...],
    *,
    lower: float,
    cell_width: float,
    probability: float,
) -> float:
    """Interpolate a quantile assuming constant density within each cell."""
    if probability <= 0.0:
        return lower
    if probability >= 1.0:
        return lower + len(masses) * cell_width

    cumulative = 0.0
    for index, mass in enumerate(masses):
        next_cumulative = cumulative + mass
        if probability <= next_cumulative or index == len(masses) - 1:
            if mass <= 0.0:
                fraction = 0.0
            else:
                fraction = (probability - cumulative) / mass
            fraction = min(1.0, max(0.0, fraction))
            return lower + (index + fraction) * cell_width
        cumulative = next_cumulative
    return lower + len(masses) * cell_width


def compute_posterior_1d(
    prior: SincModulatedPrior,
    log_likelihood: Callable[[float], float],
    *,
    bounds: tuple[float, float],
    points: int = 4096,
    credible_level: float = 0.95,
) -> Posterior1D:
    """Compute a one-dimensional posterior using midpoint quadrature.

    The finite interval bounds is part of the model specification. The
    prior is renormalised over this interval, so the returned posterior is
    conditional on the parameter lying within these bounds.

    log_likelihood(theta) must return a finite real number or negative infinity.
    Returning a probability-scale likelihood is discouraged because it can
    underflow; provide its logarithm instead.

    The log marginal likelihood is approximated under the prior conditioned
    on bounds. points is the number of equal-width midpoint cells.
    """
    if not isinstance(prior, (SincModulatedPrior, SincPriorMixture)):
        raise TypeError("prior must be a SincModulatedPrior or SincPriorMixture")
    if not callable(log_likelihood):
        raise TypeError("log_likelihood must be callable")
    if (
        not isinstance(bounds, (tuple, list))
        or len(bounds) != 2
        or any(isinstance(value, bool) for value in bounds)
    ):
        raise ValueError("bounds must contain two finite real numbers")
    try:
        lower, upper = (float(bounds[0]), float(bounds[1]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bounds must contain two finite real numbers") from exc
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError("bounds must be finite and strictly increasing")
    if isinstance(points, bool) or not isinstance(points, int) or points < 16:
        raise ValueError("points must be an integer of at least 16")
    if isinstance(credible_level, bool):
        raise ValueError("credible_level must lie strictly between zero and one")
    try:
        level = float(credible_level)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("credible_level must be a real number") from exc
    if not isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("credible_level must lie strictly between zero and one")

    cell_width = (upper - lower) / points
    grid = tuple(lower + (index + 0.5) * cell_width for index in range(points))

    log_prior_values = tuple(prior.log_density(value) for value in grid)
    log_likelihood_values: list[float] = []
    for value in grid:
        try:
            result = float(log_likelihood(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("log_likelihood must return finite values or -inf") from exc
        if result != float("-inf") and not isfinite(result):
            raise ValueError("log_likelihood must not return NaN or +inf")
        log_likelihood_values.append(result)

    log_prior_integrand = tuple(
        value + log(cell_width) if value != float("-inf") else value
        for value in log_prior_values
    )
    log_joint_integrand = tuple(
        (log_prior + log_like + log(cell_width))
        if log_prior != float("-inf") and log_like != float("-inf")
        else float("-inf")
        for log_prior, log_like in zip(log_prior_values, log_likelihood_values)
    )

    log_prior_mass = _logsumexp(log_prior_integrand)
    log_joint_mass = _logsumexp(log_joint_integrand)
    if log_prior_mass == float("-inf"):
        raise ValueError("prior has zero numerical mass on the declared bounds")
    if log_joint_mass == float("-inf"):
        raise ValueError("likelihood has zero numerical support under the prior")

    masses = tuple(
        0.0 if value == float("-inf") else exp(value - log_joint_mass)
        for value in log_joint_integrand
    )
    mass_total = fsum(masses)
    masses = tuple(value / mass_total for value in masses)
    density = tuple(value / cell_width for value in masses)

    mean = fsum(mass * value for mass, value in zip(masses, grid))
    variance = fsum(
        mass * (value - mean) ** 2 for mass, value in zip(masses, grid)
    )
    variance = max(0.0, variance)
    tail_probability = (1.0 - level) / 2.0
    credible_interval = (
        _quantile_from_cell_masses(
            masses, lower=lower, cell_width=cell_width,
            probability=tail_probability,
        ),
        _quantile_from_cell_masses(
            masses, lower=lower, cell_width=cell_width,
            probability=1.0 - tail_probability,
        ),
    )

    return Posterior1D(
        bounds=(lower, upper),
        grid=grid,
        density=density,
        probability_masses=masses,
        mean=mean,
        variance=variance,
        standard_deviation=variance**0.5,
        credible_level=level,
        credible_interval=credible_interval,
        log_marginal_likelihood=log_joint_mass - log_prior_mass,
        points=points,
        cell_width=cell_width,
    )
