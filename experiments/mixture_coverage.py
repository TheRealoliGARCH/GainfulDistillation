"""Area covered by all convex mixtures of the six source PDFs.

Mixture weights satisfy w_i >= 0 and sum_i w_i = 1. At each x, the
attainable values form [min_i f_i(x), max_i f_i(x)]. Pairwise crossings
are located with SciPy brentq and the envelope width is integrated with
scipy.integrate.quad. No NumPy API is imported or called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import scipy
from scipy.integrate import quad
from scipy.optimize import brentq

from experiments.six_density_verification import _kernel, _normalising_constants

DENSITY_NAMES = (
    "Ghosh", "Mahaloy", "Madras", "Temple bell", "Rocket cone", "Cup",
)
MAHALOY_CUSPS = (
    math.sqrt(math.pi / 8.0),
    math.sqrt(math.pi / 4.0),
)


def _build_densities() -> dict[str, Any]:
    constants = _normalising_constants()
    kernels = {name: _kernel(name) for name in DENSITY_NAMES}
    return {
        name: (lambda x, n=name: kernels[n](x) / constants[n])
        for name in DENSITY_NAMES
    }


def _find_pairwise_crossings(densities: dict[str, Any], scan_intervals: int) -> list[float]:
    """Bracket every sign-changing pairwise crossing on [0,1], then refine."""
    grid = [i / scan_intervals for i in range(scan_intervals + 1)]
    sampled = [{name: densities[name](x) for name in DENSITY_NAMES} for x in grid]
    roots = [0.0, 1.0]
    for i, first in enumerate(DENSITY_NAMES):
        for second in DENSITY_NAMES[i + 1:]:
            previous = sampled[0][first] - sampled[0][second]
            for j in range(1, len(grid)):
                current = sampled[j][first] - sampled[j][second]
                if previous == 0.0 and 0.0 < grid[j - 1] < 1.0:
                    roots.append(grid[j - 1])
                if current == 0.0 and 0.0 < grid[j] < 1.0:
                    roots.append(grid[j])
                elif previous * current < 0.0:
                    roots.append(float(brentq(
                        lambda x, a=first, b=second:
                            densities[a](x) - densities[b](x),
                        grid[j - 1], grid[j], xtol=1.0e-14,
                    )))
                previous = current
    roots.sort()
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 1.0e-10:
            unique.append(root)
    return unique


def _envelope_segments(densities: dict[str, Any], roots: list[float]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for start, end in zip(roots[:-1], roots[1:]):
        midpoint = (start + end) / 2.0
        values = {name: densities[name](midpoint) for name in DENSITY_NAMES}
        upper = max(DENSITY_NAMES, key=lambda name: values[name])
        lower = min(DENSITY_NAMES, key=lambda name: values[name])
        if segments and segments[-1]["upper"] == upper and segments[-1]["lower"] == lower:
            segments[-1]["end"] = end
        else:
            segments.append({"start": start, "end": end, "upper": upper, "lower": lower})
    return segments


def run_mixture_coverage(
    *, scan_intervals: int = 16384,
    epsabs: float = 1.0e-13,
    epsrel: float = 1.0e-13,
) -> dict[str, Any]:
    if isinstance(scan_intervals, bool) or not isinstance(scan_intervals, int) or scan_intervals < 1024:
        raise ValueError("scan_intervals must be an integer of at least 1024")
    if not math.isfinite(epsabs) or epsabs <= 0:
        raise ValueError("epsabs must be finite and positive")
    if not math.isfinite(epsrel) or epsrel <= 0:
        raise ValueError("epsrel must be finite and positive")

    densities = _build_densities()
    roots = _find_pairwise_crossings(densities, scan_intervals)
    segments = _envelope_segments(densities, roots)
    half_integral = 0.0
    total_error = 0.0

    for segment in segments:
        start, end = segment["start"], segment["end"]
        cuts = [start] + [z for z in MAHALOY_CUSPS if start < z < end] + [end]
        integral = error = 0.0
        upper, lower = segment["upper"], segment["lower"]
        for left, right in zip(cuts[:-1], cuts[1:]):
            value, estimate = quad(
                lambda x: densities[upper](x) - densities[lower](x),
                left, right, epsabs=epsabs, epsrel=epsrel, limit=300,
            )
            integral += float(value)
            error += float(estimate)
        segment["span_integral_on_positive_half"] = integral
        segment["quad_error_estimate"] = error
        segment["integration_split_points"] = cuts
        half_integral += integral
        total_error += error

    rectangle_area = 2.0
    covered_area = 2.0 * half_integral
    uncovered_area = rectangle_area - covered_area
    return {
        "method": "SciPy brentq pairwise crossings + scipy.integrate.quad (QUADPACK)",
        "scipy_version": scipy.__version__,
        "direct_numpy_api_used": False,
        "mixture_definition": "w_i >= 0 and sum_i(w_i) = 1",
        "geometry": "union of graphs of all six-PDF convex mixtures",
        "x_domain": [-1.0, 1.0],
        "y_domain": [0.0, 1.0],
        "rectangle_area": rectangle_area,
        "scan_intervals_per_pair": scan_intervals,
        "pairwise_crossings_found_excluding_endpoints": len(roots) - 2,
        "mahaloy_cusps_integration_splits": list(MAHALOY_CUSPS),
        "envelope_segments_on_positive_half": segments,
        "positive_half_envelope_span_integral": half_integral,
        "covered_area": covered_area,
        "coverage_fraction": covered_area / rectangle_area,
        "coverage_percentage": 100.0 * covered_area / rectangle_area,
        "uncovered_area": uncovered_area,
        "uncovered_percentage": 100.0 * uncovered_area / rectangle_area,
        "positive_half_quadrature_error_estimate": total_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-intervals", type=int, default=16384)
    parser.add_argument("--output", type=Path, default=Path("results/six_density_mixture_coverage.json"))
    args = parser.parse_args()
    result = run_mixture_coverage(scan_intervals=args.scan_intervals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "coverage_percentage": result["coverage_percentage"],
        "covered_area": result["covered_area"],
        "uncovered_percentage": result["uncovered_percentage"],
        "envelope_segments": len(result["envelope_segments_on_positive_half"]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
