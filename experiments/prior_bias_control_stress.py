"""Stress-test prior-location and diffuse-contamination choices.

A paired scalar Monte Carlo design crosses four independent-observation
data-generating processes with sample sizes, prior-location offsets,
contamination weights, and diffuse-scale multipliers. Each prior variant
sees exactly the same dataset within a scenario/sample-size replication.

The working likelihood remains independent Normal(theta, 1), so Normal
variance misspecification, heavy tails and heteroskedasticity are deliberately
represented as likelihood misspecification. All mixture components retain
the nominal prior centre: mixtures cannot correct a wrong location.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from math import exp, fsum, isfinite, log, sqrt
from statistics import mean
from typing import Any

from gds_core.balanced_loss import ScalarScoreSummary, balanced_loss_variance_1d
from gds_core.posterior import Posterior1D
from gds_core.prior_diagnostics import (
    PriorBiasThresholds,
    assess_posterior_bias_1d,
    summarize_prior_1d,
)
from gds_core.sinc_prior import (
    SincModulatedPrior,
    SincPriorMixture,
    contamination_robust_prior,
)

Z_975 = 1.959963984540054
LOG_2PI = 1.8378770664093453
PARAMETER_BOUNDS = (-16.0, 16.0)



@dataclass(frozen=True)
class PriorGridCache:
    bounds: tuple[float, float]
    grid: tuple[float, ...]
    log_prior_values: tuple[float, ...]
    log_prior_mass: float
    points: int
    cell_width: float


def _logsumexp(values: tuple[float, ...]) -> float:
    finite_values = tuple(value for value in values if value != float("-inf"))
    if not finite_values:
        return float("-inf")
    maximum = max(finite_values)
    return maximum + log(fsum(exp(value - maximum) for value in finite_values))


def _quantile_from_masses(
    masses: tuple[float, ...],
    *,
    lower: float,
    cell_width: float,
    probability: float,
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
            fraction = min(1.0, max(0.0, fraction))
            return lower + (index + fraction) * cell_width
        cumulative = next_cumulative
    return lower + len(masses) * cell_width


def _build_prior_grid_cache(
    prior: SincModulatedPrior | SincPriorMixture,
    *,
    bounds: tuple[float, float],
    points: int,
) -> PriorGridCache:
    lower, upper = bounds
    cell_width = (upper - lower) / points
    grid = tuple(lower + (index + 0.5) * cell_width for index in range(points))
    log_prior_values = tuple(prior.log_density(value) for value in grid)
    log_prior_mass = _logsumexp(tuple(
        value + log(cell_width) if value != float("-inf") else value
        for value in log_prior_values
    ))
    if log_prior_mass == float("-inf"):
        raise ValueError("prior has zero numerical mass inside parameter bounds")
    return PriorGridCache(
        bounds=bounds,
        grid=grid,
        log_prior_values=log_prior_values,
        log_prior_mass=log_prior_mass,
        points=points,
        cell_width=cell_width,
    )


def _compute_posterior_from_cache(
    cache: PriorGridCache,
    log_likelihood,
) -> Posterior1D:
    """Build a posterior while reusing the prior's fixed quadrature grid."""
    log_joint = []
    log_width = log(cache.cell_width)
    for theta, log_prior in zip(cache.grid, cache.log_prior_values):
        if log_prior == float("-inf"):
            log_joint.append(float("-inf"))
            continue
        likelihood_value = float(log_likelihood(theta))
        if likelihood_value != float("-inf") and not isfinite(likelihood_value):
            raise ValueError("log_likelihood must return finite values or -inf")
        log_joint.append(log_prior + likelihood_value + log_width)

    log_joint_mass = _logsumexp(tuple(log_joint))
    if log_joint_mass == float("-inf"):
        raise ValueError("likelihood has zero numerical support under the prior")
    masses = tuple(
        0.0 if value == float("-inf") else exp(value - log_joint_mass)
        for value in log_joint
    )
    total = fsum(masses)
    masses = tuple(value / total for value in masses)
    density = tuple(value / cache.cell_width for value in masses)
    posterior_mean = fsum(
        mass * theta for mass, theta in zip(masses, cache.grid)
    )
    variance = max(0.0, fsum(
        mass * (theta - posterior_mean) ** 2
        for mass, theta in zip(masses, cache.grid)
    ))
    tail = (1.0 - 0.95) / 2.0
    interval = (
        _quantile_from_masses(
            masses, lower=cache.bounds[0],
            cell_width=cache.cell_width, probability=tail,
        ),
        _quantile_from_masses(
            masses, lower=cache.bounds[0],
            cell_width=cache.cell_width, probability=1.0 - tail,
        ),
    )
    return Posterior1D(
        bounds=cache.bounds,
        grid=cache.grid,
        density=density,
        probability_masses=masses,
        mean=posterior_mean,
        variance=variance,
        standard_deviation=sqrt(variance),
        credible_level=0.95,
        credible_interval=interval,
        log_marginal_likelihood=log_joint_mass - cache.log_prior_mass,
        points=cache.points,
        cell_width=cache.cell_width,
    )


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str


SCENARIOS = (
    Scenario("normal_sd_1", "Normal observations with SD 1; working model correctly specified."),
    Scenario("normal_sd_2", "Normal observations with SD 2; working model fixes SD at 1."),
    Scenario("student_t5_var4", "Student-t(5) observations rescaled to variance 4."),
    Scenario(
        "heteroskedastic_normal_var4",
        "Independent Normal observations with deterministic row-specific scales, average variance 4.",
    ),
)


@dataclass(frozen=True)
class PriorCase:
    name: str
    location_offset: float
    contamination_weight: float
    scale_multiplier: float
    prior: SincModulatedPrior | SincPriorMixture


def _draw_sample(
    rng: random.Random,
    *,
    scenario: str,
    theta_true: float,
    sample_size: int,
) -> tuple[tuple[float, ...], float]:
    """Return observations and oracle variance of their arithmetic mean."""
    if scenario == "normal_sd_1":
        observations = tuple(
            rng.gauss(theta_true, 1.0) for _ in range(sample_size)
        )
        oracle_variance = 1.0 / sample_size
    elif scenario == "normal_sd_2":
        observations = tuple(
            rng.gauss(theta_true, 2.0) for _ in range(sample_size)
        )
        oracle_variance = 4.0 / sample_size
    elif scenario == "student_t5_var4":
        scale = sqrt(4.0 * 3.0 / 5.0)
        values = []
        for _ in range(sample_size):
            normal = rng.gauss(0.0, 1.0)
            chi_square = rng.gammavariate(2.5, 2.0)
            values.append(theta_true + scale * normal / sqrt(chi_square / 5.0))
        observations = tuple(values)
        oracle_variance = 4.0 / sample_size
    elif scenario == "heteroskedastic_normal_var4":
        raw_scales = tuple(
            0.5 + 1.5 * (index + 0.5) / sample_size
            for index in range(sample_size)
        )
        raw_mean_square = fsum(value * value for value in raw_scales) / sample_size
        normalizer = 2.0 / sqrt(raw_mean_square)
        scales = tuple(value * normalizer for value in raw_scales)
        observations = tuple(
            theta_true + rng.gauss(0.0, scale) for scale in scales
        )
        oracle_variance = fsum(scale * scale for scale in scales) / (sample_size**2)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return observations, oracle_variance


def _build_prior_cases(
    *,
    theta_true: float,
    location_offsets: tuple[float, ...],
    contamination_weights: tuple[float, ...],
    scale_multipliers: tuple[float, ...],
) -> tuple[PriorCase, ...]:
    cases: list[PriorCase] = []
    for offset in location_offsets:
        location = theta_true + offset
        nominal = SincModulatedPrior(
            location=location, bandwidth=1.0, scale=2.0,
            center_frequency=0.0, modulation_depth=0.0,
        )
        cases.append(PriorCase(
            name="nominal",
            location_offset=offset,
            contamination_weight=0.0,
            scale_multiplier=1.0,
            prior=nominal,
        ))
        for multiplier in scale_multipliers:
            for weight in contamination_weights:
                cases.append(PriorCase(
                    name="diffuse_contamination",
                    location_offset=offset,
                    contamination_weight=weight,
                    scale_multiplier=multiplier,
                    prior=contamination_robust_prior(
                        nominal,
                        contamination_weight=weight,
                        scale_multiplier=multiplier,
                    ),
                ))

    reference = SincModulatedPrior(
        location=theta_true, bandwidth=1.0, scale=2.0,
        center_frequency=0.0, modulation_depth=0.0,
    )
    cases.append(PriorCase(
        name="centred_diffuse_reference",
        location_offset=0.0,
        contamination_weight=0.0,
        scale_multiplier=1.0,
        prior=reference,
    ))
    return tuple(cases)


def _case_key(case: PriorCase) -> str:
    return (
        case.name + "|" + str(case.location_offset) + "|"
        + str(case.contamination_weight) + "|" + str(case.scale_multiplier)
    )


def _fit_case(
    observations: tuple[float, ...],
    oracle_variance: float,
    *,
    theta_true: float,
    prior_case: PriorCase,
    prior_summary: Any,
    prior_cache: PriorGridCache,
) -> dict[str, float]:
    n = len(observations)
    sample_mean = fsum(observations) / n
    centred_sum_squares = fsum(
        (value - sample_mean) ** 2 for value in observations
    )

    def log_likelihood(theta: float) -> float:
        squared_error = centred_sum_squares + n * (sample_mean - theta) ** 2
        return -0.5 * (n * LOG_2PI + squared_error)

    posterior = _compute_posterior_from_cache(prior_cache, log_likelihood)

    def score_information(theta: float) -> tuple[ScalarScoreSummary, float]:
        mean_squared_score = (
            centred_sum_squares / n + (sample_mean - theta) ** 2
        )
        return ScalarScoreSummary(n, mean_squared_score), 1.0

    robust = balanced_loss_variance_1d(posterior, score_information)
    diagnostic = assess_posterior_bias_1d(
        prior_summary,
        posterior,
        likelihood_reference=sample_mean,
        reference_standard_error=sqrt(oracle_variance),
        thresholds=PriorBiasThresholds(),
    )
    credible_lower, credible_upper = posterior.credible_interval
    robust_lower = posterior.mean - Z_975 * robust.robust_standard_error
    robust_upper = posterior.mean + Z_975 * robust.robust_standard_error
    error = posterior.mean - theta_true
    warning_text = diagnostic.warnings
    return {
        "posterior_error": error,
        "squared_error": error * error,
        "posterior_sd": posterior.standard_deviation,
        "robust_se": robust.robust_standard_error,
        "robust_variance": robust.robust_variance,
        "oracle_variance": oracle_variance,
        "posterior_covered": float(
            credible_lower <= theta_true <= credible_upper
        ),
        "robust_wald_covered": float(
            robust_lower <= theta_true <= robust_upper
        ),
        "prior_retained_mass": prior_summary.retained_prior_mass,
        "posterior_pull": (
            diagnostic.posterior_pull_fraction
            if diagnostic.posterior_pull_fraction is not None else 0.0
        ),
        "prior_conflict_flag": float(any(
            "Prior mean is materially separated" in warning
            for warning in warning_text
        )),
        "concentration_flag": float(any(
            "Prior spread is concentrated" in warning
            for warning in warning_text
        )),
        "posterior_displacement_flag": float(any(
            "materially displaced" in warning for warning in warning_text
        )),
        "warning_count": float(len(warning_text)),
    }


def _aggregate(
    records: list[dict[str, float]],
    *,
    scenario: str,
    prior_case: PriorCase,
    sample_size: int,
    repetitions: int,
    theta_true: float,
) -> dict[str, Any]:
    errors = [record["posterior_error"] for record in records]
    bias = mean(errors)
    posterior_coverage = mean(record["posterior_covered"] for record in records)
    robust_coverage = mean(record["robust_wald_covered"] for record in records)
    mean_robust_variance = mean(record["robust_variance"] for record in records)
    mean_oracle_variance = mean(record["oracle_variance"] for record in records)
    return {
        "scenario": scenario,
        "prior_variant": prior_case.name,
        "sample_size": sample_size,
        "repetitions": repetitions,
        "true_parameter": theta_true,
        "prior_location_offset": prior_case.location_offset,
        "prior_location": theta_true + prior_case.location_offset,
        "contamination_weight": prior_case.contamination_weight,
        "diffuse_scale_multiplier": prior_case.scale_multiplier,
        "bias": bias,
        "absolute_bias": abs(bias),
        "bias_mcse": sqrt(
            fsum((error - bias) ** 2 for error in errors)
            / (repetitions * (repetitions - 1))
        ),
        "rmse": sqrt(mean(record["squared_error"] for record in records)),
        "posterior_credible_interval_coverage": posterior_coverage,
        "posterior_coverage_mcse": sqrt(
            posterior_coverage * (1.0 - posterior_coverage) / repetitions
        ),
        "robust_wald_interval_coverage": robust_coverage,
        "robust_coverage_mcse": sqrt(
            robust_coverage * (1.0 - robust_coverage) / repetitions
        ),
        "mean_posterior_sd": mean(record["posterior_sd"] for record in records),
        "mean_balanced_loss_robust_se": mean(
            record["robust_se"] for record in records
        ),
        "mean_balanced_loss_variance": mean_robust_variance,
        "mean_oracle_variance": mean_oracle_variance,
        "robust_to_oracle_variance_ratio": (
            mean_robust_variance / mean_oracle_variance
            if mean_oracle_variance > 0.0 else None
        ),
        "mean_signed_posterior_pull": mean(
            record["posterior_pull"] for record in records
        ),
        "mean_absolute_posterior_pull": mean(
            abs(record["posterior_pull"]) for record in records
        ),
        "prior_conflict_warning_rate": mean(
            record["prior_conflict_flag"] for record in records
        ),
        "prior_concentration_warning_rate": mean(
            record["concentration_flag"] for record in records
        ),
        "posterior_displacement_warning_rate": mean(
            record["posterior_displacement_flag"] for record in records
        ),
        "mean_diagnostic_warning_count": mean(
            record["warning_count"] for record in records
        ),
        "mean_prior_retained_mass": mean(
            record["prior_retained_mass"] for record in records
        ),
        "minimum_prior_retained_mass": min(
            record["prior_retained_mass"] for record in records
        ),
    }


def run_stress_grid(
    *,
    sample_sizes: tuple[int, ...] = (40, 80, 160),
    repetitions: int = 100,
    points: int = 1024,
    seed: int = 2026111402,
    theta_true: float = 0.5,
    scenarios: tuple[str, ...] = tuple(item.name for item in SCENARIOS),
    location_offsets: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0),
    contamination_weights: tuple[float, ...] = (0.05, 0.10, 0.20),
    scale_multipliers: tuple[float, ...] = (3.0, 5.0, 10.0),
) -> dict[str, Any]:
    """Run the paired location × weight × diffuse-scale × DGP stress grid."""
    if not sample_sizes or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in sample_sizes
    ):
        raise ValueError("sample_sizes must contain integers of at least two")
    if len(set(sample_sizes)) != len(sample_sizes):
        raise ValueError("sample_sizes must be unique")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("repetitions must be an integer of at least two")
    if isinstance(points, bool) or not isinstance(points, int) or points < 64:
        raise ValueError("points must be an integer of at least 64")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(theta_true, bool) or not isinstance(theta_true, (int, float)):
        raise ValueError("theta_true must be finite")
    if not isfinite(float(theta_true)) or not -8.0 < float(theta_true) < 8.0:
        raise ValueError("theta_true must lie strictly inside the parameter bounds")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be non-empty and unique")
    valid_scenarios = {item.name for item in SCENARIOS}
    if any(name not in valid_scenarios for name in scenarios):
        raise ValueError("unknown data-generating scenario")

    for name, values in (
        ("location_offsets", location_offsets),
        ("contamination_weights", contamination_weights),
        ("scale_multipliers", scale_multipliers),
    ):
        if not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(float(value)) for value in values
        ):
            raise ValueError(f"{name} must contain finite numeric values")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique values")
    if any(not 0.0 < weight < 1.0 for weight in contamination_weights):
        raise ValueError("contamination_weights must lie strictly between zero and one")
    if any(multiplier <= 1.0 for multiplier in scale_multipliers):
        raise ValueError("scale_multipliers must be greater than one")
    if any(not -16.0 < offset < 16.0 for offset in location_offsets):
        raise ValueError("location_offsets are outside the supported finite bounds")
    if any(not -8.0 < float(theta_true) + float(offset) < 8.0 for offset in location_offsets):
        raise ValueError("prior locations must lie strictly inside the finite bounds")

    cases = _build_prior_cases(
        theta_true=float(theta_true),
        location_offsets=tuple(float(value) for value in location_offsets),
        contamination_weights=tuple(float(value) for value in contamination_weights),
        scale_multipliers=tuple(float(value) for value in scale_multipliers),
    )
    keys = {_case_key(case): case for case in cases}
    prior_caches = {
        key: _build_prior_grid_cache(
            case.prior, bounds=PARAMETER_BOUNDS, points=points,
        )
        for key, case in keys.items()
    }
    summaries = {
        key: summarize_prior_1d(
            case.prior,
            bounds=PARAMETER_BOUNDS,
            points=max(4096, points * 2),
        )
        for key, case in keys.items()
    }

    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    scenario_lookup = {item.name: item.description for item in SCENARIOS}

    for scenario in scenarios:
        for sample_size in sample_sizes:
            buckets: dict[str, list[dict[str, float]]] = {
                key: [] for key in keys
            }
            for _ in range(repetitions):
                observations, oracle_variance = _draw_sample(
                    rng,
                    scenario=scenario,
                    theta_true=float(theta_true),
                    sample_size=sample_size,
                )
                for key, case in keys.items():
                    buckets[key].append(
                        _fit_case(
                            observations,
                            oracle_variance,
                            theta_true=float(theta_true),
                            prior_case=case,
                            prior_summary=summaries[key],
                            prior_cache=prior_caches[key],
                        )
                    )
            for key, case in keys.items():
                results.append(
                    _aggregate(
                        buckets[key],
                        scenario=scenario,
                        prior_case=case,
                        sample_size=sample_size,
                        repetitions=repetitions,
                        theta_true=float(theta_true),
                    )
                )

    return {
        "experiment": "GDS prior-location and diffuse-contamination robustness stress grid",
        "design_note": (
            "Paired common-random-number Monte Carlo study. Each dataset is reused "
            "across all prior variants within each DGP and sample-size cell. The "
            "working likelihood is independent Normal(theta, 1). The heteroskedastic "
            "case is independent but non-identically distributed; no serial or "
            "cluster dependence is simulated. All contamination mixtures share "
            "their nominal component's location, so they reduce concentration but "
            "do not correct prior-centre misspecification. This is a finite-simulation "
            "GDS study, not an exact replication of Li and Rice (2024)."
        ),
        "settings": {
            "sample_sizes": list(sample_sizes),
            "repetitions_per_cell": repetitions,
            "quadrature_points": points,
            "seed": seed,
            "true_parameter": float(theta_true),
            "working_likelihood": "independent Normal(theta, 1)",
            "parameter_bounds": list(PARAMETER_BOUNDS),
            "scenarios": list(scenarios),
            "scenario_descriptions": {
                name: scenario_lookup[name] for name in scenarios
            },
            "location_offsets_from_truth": list(location_offsets),
            "contamination_weights": list(contamination_weights),
            "diffuse_scale_multipliers": list(scale_multipliers),
            "common_random_numbers_across_prior_variants": True,
            "prior_case_count": len(cases),
            "cell_count": len(results),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=(40, 80, 160))
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026111402)
    parser.add_argument("--theta-true", type=float, default=0.5)
    parser.add_argument("--scenarios", nargs="+", default=tuple(item.name for item in SCENARIOS))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = run_stress_grid(
        sample_sizes=tuple(args.sample_sizes),
        repetitions=args.repetitions,
        points=args.points,
        seed=args.seed,
        theta_true=args.theta_true,
        scenarios=tuple(args.scenarios),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
