"""Prior-predictive and posterior-shrinkage diagnostics for scalar Sinc priors.

These routines are diagnostic: they do not automatically alter, temper,
recenter, or reject a prior. Numerical summaries are conditional on an
explicit finite interval, and report how much original prior mass is retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite, log, sqrt
import random
from typing import Callable

from .posterior import Posterior1D
from .sinc_prior import SincModulatedPrior


def _validate_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2 or any(
        isinstance(value, bool) for value in bounds
    ):
        raise ValueError("bounds must contain two finite real numbers")
    try:
        lower, upper = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bounds must contain two finite real numbers") from exc
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError("bounds must be finite and strictly increasing")
    return lower, upper


def _validate_points(points: int) -> int:
    if isinstance(points, bool) or not isinstance(points, int) or points < 64:
        raise ValueError("points must be an integer of at least 64")
    return points


def _logsumexp(values: tuple[float, ...]) -> float:
    finite_values = tuple(value for value in values if value != float("-inf"))
    if not finite_values:
        return float("-inf")
    maximum = max(finite_values)
    return maximum + log(fsum(exp(value - maximum) for value in finite_values))


def _cell_quantile(
    masses: tuple[float, ...], *, lower: float, cell_width: float, probability: float
) -> float:
    if probability <= 0.0:
        return lower
    if probability >= 1.0:
        return lower + len(masses) * cell_width
    cumulative = 0.0
    for index, mass in enumerate(masses):
        next_cumulative = cumulative + mass
        if probability <= next_cumulative or index == len(masses) - 1:
            fraction = 0.0 if mass <= 0.0 else (
                (probability - cumulative) / mass
            )
            return lower + (index + min(1.0, max(0.0, fraction))) * cell_width
        cumulative = next_cumulative
    return lower + len(masses) * cell_width


def _sample_quantile(values: tuple[float, ...], probability: float) -> float:
    if probability <= 0.0:
        return values[0]
    if probability >= 1.0:
        return values[-1]
    position = probability * (len(values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    fraction = position - lower_index
    return values[lower_index] * (1.0 - fraction) + values[upper_index] * fraction


@dataclass(frozen=True)
class PriorSummary1D:
    bounds: tuple[float, float]
    retained_prior_mass: float
    conditional_mean: float
    conditional_variance: float
    conditional_standard_deviation: float
    credible_level: float
    central_interval: tuple[float, float]
    interval_width: float
    boundary_cell_mass: float
    points: int
    cell_width: float


def summarize_prior_1d(
    prior: SincModulatedPrior,
    *,
    bounds: tuple[float, float],
    points: int = 8192,
    credible_level: float = 0.95,
) -> PriorSummary1D:
    """Summarise a proper Sinc prior conditional on a finite interval.

    retained_prior_mass is the prior probability inside bounds before
    conditioning. Moments and quantiles are conditional on those bounds.
    """
    if not isinstance(prior, SincModulatedPrior):
        raise TypeError("prior must be a SincModulatedPrior")
    lower, upper = _validate_bounds(bounds)
    points = _validate_points(points)
    if isinstance(credible_level, bool):
        raise ValueError("credible_level must lie strictly between zero and one")
    try:
        level = float(credible_level)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("credible_level must be a real number") from exc
    if not isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("credible_level must lie strictly between zero and one")

    width = (upper - lower) / points
    grid = tuple(lower + (i + 0.5) * width for i in range(points))
    log_terms = tuple(
        prior.log_density(x) + log(width)
        if prior.log_density(x) != float("-inf") else float("-inf")
        for x in grid
    )
    log_mass = _logsumexp(log_terms)
    if log_mass == float("-inf"):
        raise ValueError("prior has zero numerical mass inside bounds")
    retained = min(1.0, max(0.0, exp(log_mass)))
    raw_masses = tuple(
        0.0 if term == float("-inf") else exp(term - log_mass)
        for term in log_terms
    )
    total = fsum(raw_masses)
    masses = tuple(value / total for value in raw_masses)
    mean_value = fsum(mass * x for mass, x in zip(masses, grid))
    variance = fsum(
        mass * (x - mean_value) ** 2 for mass, x in zip(masses, grid)
    )
    tail = (1.0 - level) / 2.0
    interval = (
        _cell_quantile(masses, lower=lower, cell_width=width, probability=tail),
        _cell_quantile(masses, lower=lower, cell_width=width, probability=1.0 - tail),
    )
    return PriorSummary1D(
        bounds=(lower, upper),
        retained_prior_mass=retained,
        conditional_mean=mean_value,
        conditional_variance=max(0.0, variance),
        conditional_standard_deviation=sqrt(max(0.0, variance)),
        credible_level=level,
        central_interval=interval,
        interval_width=interval[1] - interval[0],
        boundary_cell_mass=masses[0] + masses[-1],
        points=points,
        cell_width=width,
    )


@dataclass(frozen=True)
class PriorPredictiveResult1D:
    prior_summary: PriorSummary1D
    draws: int
    predictive_mean: float
    predictive_variance: float
    predictive_standard_deviation: float
    predictive_interval: tuple[float, float]
    observed_statistic: float | None
    lower_tail_probability: float | None
    upper_tail_probability: float | None
    two_sided_tail_probability: float | None
    warnings: tuple[str, ...]
    seed: int


def prior_predictive_check_1d(
    prior: SincModulatedPrior,
    simulate_statistic: Callable[[float, random.Random], float],
    *,
    bounds: tuple[float, float],
    draws: int = 10000,
    points: int = 8192,
    seed: int = 20261114,
    observed_statistic: float | None = None,
    credible_level: float = 0.95,
    surprise_threshold: float = 0.01,
    minimum_retained_prior_mass: float = 0.99,
) -> PriorPredictiveResult1D:
    """Simulate a scalar statistic under the prior predictive distribution.

    simulate_statistic(theta, rng) must return one scalar statistic drawn
    from the sampling model conditional on theta. Parameter values are
    sampled from the prior conditioned on bounds using midpoint-cell masses.
    """
    if not callable(simulate_statistic):
        raise TypeError("simulate_statistic must be callable")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(surprise_threshold, bool) or isinstance(
        minimum_retained_prior_mass, bool
    ):
        raise ValueError("diagnostic thresholds must be real numbers")
    try:
        surprise = float(surprise_threshold)
        minimum_mass = float(minimum_retained_prior_mass)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("diagnostic thresholds must be real numbers") from exc
    if not isfinite(surprise) or not 0.0 < surprise < 1.0:
        raise ValueError("surprise_threshold must lie strictly between zero and one")
    if not isfinite(minimum_mass) or not 0.0 < minimum_mass <= 1.0:
        raise ValueError("minimum_retained_prior_mass must lie in (0, 1]")

    summary = summarize_prior_1d(
        prior, bounds=bounds, points=points, credible_level=credible_level
    )
    lower, _ = summary.bounds
    grid = tuple(
        lower + (i + 0.5) * summary.cell_width for i in range(summary.points)
    )
    log_terms = tuple(
        prior.log_density(x) + log(summary.cell_width)
        if prior.log_density(x) != float("-inf") else float("-inf")
        for x in grid
    )
    log_mass = _logsumexp(log_terms)
    masses = tuple(
        0.0 if term == float("-inf") else exp(term - log_mass)
        for term in log_terms
    )
    cumulative: list[float] = []
    total = 0.0
    for mass in masses:
        total += mass
        cumulative.append(total)
    cumulative[-1] = 1.0

    rng = random.Random(seed)
    predictive: list[float] = []
    for _ in range(draws):
        u = rng.random()
        lo, hi = 0, len(cumulative) - 1
        while lo < hi:
            middle = (lo + hi) // 2
            if u <= cumulative[middle]:
                hi = middle
            else:
                lo = middle + 1
        cell_left = lower + lo * summary.cell_width
        theta = cell_left + rng.random() * summary.cell_width
        try:
            value = float(simulate_statistic(theta, rng))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("simulate_statistic must return finite real values") from exc
        if not isfinite(value):
            raise ValueError("simulate_statistic must return finite real values")
        predictive.append(value)

    ordered = tuple(sorted(predictive))
    pred_mean = fsum(predictive) / draws
    pred_variance = fsum((x - pred_mean) ** 2 for x in predictive) / draws
    tail = (1.0 - summary.credible_level) / 2.0
    pred_interval = (
        _sample_quantile(ordered, tail),
        _sample_quantile(ordered, 1.0 - tail),
    )

    observed = None
    lower_probability = upper_probability = two_sided_probability = None
    warnings: list[str] = []
    if observed_statistic is not None:
        if isinstance(observed_statistic, bool):
            raise ValueError("observed_statistic must be finite")
        try:
            observed = float(observed_statistic)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("observed_statistic must be finite") from exc
        if not isfinite(observed):
            raise ValueError("observed_statistic must be finite")
        lower_probability = (
            1 + sum(x <= observed for x in predictive)
        ) / (draws + 1)
        upper_probability = (
            1 + sum(x >= observed for x in predictive)
        ) / (draws + 1)
        two_sided_probability = min(
            1.0, 2.0 * min(lower_probability, upper_probability)
        )
        if two_sided_probability <= surprise:
            warnings.append(
                "Observed statistic is in a low-probability prior-predictive tail; "
                "review prior-data compatibility and the sampling model."
            )
    if summary.retained_prior_mass < minimum_mass:
        warnings.append(
            "The declared finite parameter interval excludes appreciable mass "
            "from the original prior; widen or justify the interval."
        )

    return PriorPredictiveResult1D(
        prior_summary=summary,
        draws=draws,
        predictive_mean=pred_mean,
        predictive_variance=max(0.0, pred_variance),
        predictive_standard_deviation=sqrt(max(0.0, pred_variance)),
        predictive_interval=pred_interval,
        observed_statistic=observed,
        lower_tail_probability=lower_probability,
        upper_tail_probability=upper_probability,
        two_sided_tail_probability=two_sided_probability,
        warnings=tuple(warnings),
        seed=seed,
    )


@dataclass(frozen=True)
class PriorBiasThresholds:
    """Configurable warning thresholds, not universal statistical cutoffs."""

    max_prior_reference_gap_z: float = 2.0
    max_posterior_pull_fraction: float = 0.5
    max_boundary_cell_mass: float = 0.01
    minimum_retained_prior_mass: float = 0.99
    minimum_prior_to_reference_sd_ratio: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.max_prior_reference_gap_z,
            self.max_posterior_pull_fraction,
            self.max_boundary_cell_mass,
            self.minimum_retained_prior_mass,
            self.minimum_prior_to_reference_sd_ratio,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in values
        ):
            raise ValueError("diagnostic thresholds must be finite real numbers")
        if (
            self.max_prior_reference_gap_z < 0.0
            or self.max_posterior_pull_fraction < 0.0
            or not 0.0 <= self.max_boundary_cell_mass <= 1.0
            or not 0.0 < self.minimum_retained_prior_mass <= 1.0
            or self.minimum_prior_to_reference_sd_ratio < 0.0
        ):
            raise ValueError("diagnostic thresholds are outside their allowed ranges")


@dataclass(frozen=True)
class PosteriorBiasDiagnostic1D:
    prior_mean: float
    likelihood_reference: float
    posterior_mean: float
    prior_reference_gap: float
    posterior_reference_gap: float
    prior_reference_gap_z: float | None
    posterior_reference_gap_z: float | None
    posterior_pull_fraction: float | None
    prior_to_reference_sd_ratio: float | None
    boundary_cell_mass: float
    retained_prior_mass: float
    warnings: tuple[str, ...]


def assess_posterior_bias_1d(
    prior_summary: PriorSummary1D,
    posterior: Posterior1D,
    *,
    likelihood_reference: float,
    reference_standard_error: float | None = None,
    thresholds: PriorBiasThresholds = PriorBiasThresholds(),
) -> PosteriorBiasDiagnostic1D:
    """Report prior displacement and posterior pull against a likelihood-only reference.

    This is a single-dataset shrinkage diagnostic, not an estimate of
    frequentist bias and not an automatic posterior correction.
    """
    if not isinstance(prior_summary, PriorSummary1D):
        raise TypeError("prior_summary must be a PriorSummary1D")
    if not isinstance(posterior, Posterior1D):
        raise TypeError("posterior must be a Posterior1D")
    if not isinstance(thresholds, PriorBiasThresholds):
        raise TypeError("thresholds must be PriorBiasThresholds")
    if posterior.bounds != prior_summary.bounds:
        raise ValueError("prior summary and posterior must use identical bounds")
    if isinstance(likelihood_reference, bool):
        raise ValueError("likelihood_reference must be finite")
    try:
        reference = float(likelihood_reference)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("likelihood_reference must be finite") from exc
    if not isfinite(reference):
        raise ValueError("likelihood_reference must be finite")

    if reference_standard_error is None:
        reference_se = None
    else:
        if isinstance(reference_standard_error, bool):
            raise ValueError("reference_standard_error must be positive and finite")
        try:
            reference_se = float(reference_standard_error)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reference_standard_error must be positive and finite") from exc
        if not isfinite(reference_se) or reference_se <= 0.0:
            raise ValueError("reference_standard_error must be positive and finite")


    prior_gap = prior_summary.conditional_mean - reference
    posterior_gap = posterior.mean - reference
    gap_z = None if reference_se is None else prior_gap / sqrt(
        prior_summary.conditional_variance + reference_se**2
    )
    posterior_z = None if reference_se is None else posterior_gap / reference_se
    prior_to_reference_sd_ratio = (
        None if reference_se is None else
        prior_summary.conditional_standard_deviation / reference_se
    )
    pull_fraction = (
        None if abs(prior_gap) <= 1e-12 else posterior_gap / prior_gap
    )

    warnings: list[str] = []
    if gap_z is not None and abs(gap_z) > thresholds.max_prior_reference_gap_z:
        warnings.append(
            "Prior mean is materially separated from the likelihood-only reference "
            "after accounting for prior and reference uncertainty."
        )
    if (
        prior_to_reference_sd_ratio is not None
        and prior_to_reference_sd_ratio < thresholds.minimum_prior_to_reference_sd_ratio
    ):
        warnings.append(
            "Prior spread is concentrated relative to the likelihood-reference "
            "standard error under the configured threshold."
        )
    if (
        pull_fraction is not None
        and abs(prior_gap) > 1e-12
        and pull_fraction > thresholds.max_posterior_pull_fraction
    ):
        warnings.append(
            "Posterior mean remains materially displaced toward the prior "
            "relative to the likelihood-only reference."
        )
    if prior_summary.boundary_cell_mass > thresholds.max_boundary_cell_mass:
        warnings.append(
            "Prior mass is concentrated in the first or last numerical grid cell; "
            "inspect the finite parameter bounds and grid resolution."
        )
    if prior_summary.retained_prior_mass < thresholds.minimum_retained_prior_mass:
        warnings.append(
            "The finite parameter interval retains less than the configured "
            "minimum mass of the original prior."
        )

    return PosteriorBiasDiagnostic1D(
        prior_mean=prior_summary.conditional_mean,
        likelihood_reference=reference,
        posterior_mean=posterior.mean,
        prior_reference_gap=prior_gap,
        posterior_reference_gap=posterior_gap,
        prior_reference_gap_z=gap_z,
        posterior_reference_gap_z=posterior_z,
        posterior_pull_fraction=pull_fraction,
        prior_to_reference_sd_ratio=prior_to_reference_sd_ratio,
        boundary_cell_mass=prior_summary.boundary_cell_mass,
        retained_prior_mass=prior_summary.retained_prior_mass,
        warnings=tuple(warnings),
    )
