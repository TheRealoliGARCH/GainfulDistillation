"""SciPy verification of the six densities in Six Probability Density Functions.

No NumPy API is used directly. SciPy's QUADPACK integrator and special
functions are used to evaluate normalisers, integrals, moments, and peaks.

The Mahaloy paper writes sinc(8*x**2) but does not define the convention.
The displayed normaliser and plot zeros agree with sinc(u)=sin(u)/u, which
is the convention explicitly implemented here.
"""
from __future__ import annotations

import json
import math
from typing import Callable

import scipy
from scipy.integrate import quad
from scipy.special import erf, erfi, fresnel, gamma

EPSABS = 1.0e-12
EPSREL = 1.0e-12
LIMIT = 500


def fresnel_c(x: float) -> float:
    """C(x) = integral_0^x cos(pi*t**2/2) dt."""
    _, c_value = fresnel(x)
    return float(c_value)


def fresnel_s(x: float) -> float:
    """S(x) = integral_0^x sin(pi*t**2/2) dt."""
    s_value, _ = fresnel(x)
    return float(s_value)


def sinc_unnormalised(u: float) -> float:
    """Unnormalised sinc: sin(u)/u, continuously extended at zero."""
    if u == 0.0:
        return 1.0
    return math.sin(u) / u


def _normalising_constants() -> dict[str, float]:
    sqrt_pi = math.sqrt(math.pi)
    return {
        "Ghosh": 2.0 * (1.0 + fresnel_c(math.sqrt(6.0)) / math.sqrt(6.0)),
        "Mahaloy": 0.25 * (
            -math.sin(8.0)
            - 8.0 * sqrt_pi * fresnel_c(2.0)
            + 8.0 * sqrt_pi * fresnel_c(math.sqrt(2.0))
            + 4.0 * sqrt_pi * fresnel_c(4.0 / sqrt_pi)
        ),
        "Madras": math.sqrt(2.0) * fresnel_s(math.sqrt(2.0)),
        "Temple bell": 2.0 * float(gamma(5.0 / 4.0)),
        "Rocket cone": sqrt_pi * float(erf(1.0)),
        "Cup": sqrt_pi * float(erfi(1.0)),
    }


def _kernel(name: str) -> Callable[[float], float]:
    if name == "Ghosh":
        # cos(3*pi*x**2) + cos(2*pi) = cos(3*pi*x**2) + 1.
        return lambda x: 1.0 + math.cos(3.0 * math.pi * x * x)
    if name == "Mahaloy":
        return lambda x: abs(sinc_unnormalised(8.0 * x * x))
    if name == "Madras":
        return lambda x: abs(math.sin(math.pi * x * x))
    if name == "Temple bell":
        return lambda x: math.exp(-(x**4))
    if name == "Rocket cone":
        return lambda x: math.exp(-(x * x))
    if name == "Cup":
        return lambda x: math.exp(x * x)
    raise KeyError(name)


def _quad_half(
    name: str,
    integrand: Callable[[float], float],
) -> tuple[float, float]:
    """Integrate over the nonnegative half-domain, returning value and QUADPACK error."""
    if name == "Temple bell":
        value, error = quad(
            integrand, 0.0, math.inf,
            epsabs=EPSABS, epsrel=EPSREL, limit=LIMIT,
        )
    elif name == "Mahaloy":
        # The absolute sinc kernel changes formula at its two interior zeros.
        split_points = (math.sqrt(math.pi / 8.0), math.sqrt(math.pi / 4.0))
        value, error = quad(
            integrand, 0.0, 1.0, points=split_points,
            epsabs=EPSABS, epsrel=EPSREL, limit=LIMIT,
        )
    else:
        value, error = quad(
            integrand, 0.0, 1.0,
            epsabs=EPSABS, epsrel=EPSREL, limit=LIMIT,
        )
    return float(value), float(error)


def verify_six_densities() -> dict[str, object]:
    constants = _normalising_constants()
    peaks = {
        "Ghosh": 2.0 / constants["Ghosh"],
        "Mahaloy": 1.0 / constants["Mahaloy"],
        "Madras": 1.0 / constants["Madras"],
        "Temple bell": 1.0 / constants["Temple bell"],
        "Rocket cone": 1.0 / constants["Rocket cone"],
        "Cup": math.e / constants["Cup"],
    }

    results: dict[str, dict[str, float | bool]] = {}
    for name, normaliser in constants.items():
        kernel = _kernel(name)
        raw_half, raw_error_half = _quad_half(name, kernel)
        raw_integral = 2.0 * raw_half
        raw_error = 2.0 * raw_error_half

        second_half, second_error = _quad_half(
            name, lambda x, k=kernel, d=normaliser: x**2 * k(x) / d
        )
        fourth_half, fourth_error = _quad_half(
            name, lambda x, k=kernel, d=normaliser: x**4 * k(x) / d
        )
        variance = 2.0 * second_half
        fourth_moment = 2.0 * fourth_half
        results[name] = {
            "normalising_constant": normaliser,
            "raw_kernel_integral": raw_integral,
            "raw_kernel_integral_quad_error": raw_error,
            "normalised_density_integral": raw_integral / normaliser,
            "variance": variance,
            "variance_quad_error": 2.0 * second_error,
            "fourth_moment": fourth_moment,
            "fourth_moment_quad_error": 2.0 * fourth_error,
            "kurtosis": fourth_moment / (variance * variance),
            "mean": 0.0,
            "peak_density": peaks[name],
            "peak_strictly_below_one": peaks[name] < 1.0,
        }

    mahaloy_formula_error = (
        results["Mahaloy"]["raw_kernel_integral"]
        - results["Mahaloy"]["normalising_constant"]
    )
    return {
        "verification_engine": "SciPy scipy.integrate.quad (QUADPACK)",
        "scipy_version": scipy.__version__,
        "direct_numpy_api_used": False,
        "quadrature_tolerances": {"epsabs": EPSABS, "epsrel": EPSREL, "limit": LIMIT},
        "sinc_convention": "unnormalised sinc(u)=sin(u)/u, sinc(0)=1",
        "mahaloy_normaliser_minus_direct_integral": mahaloy_formula_error,
        "all_normalised_integrals_within_1e-10_of_one": all(
            abs(float(row["normalised_density_integral"]) - 1.0) < 1.0e-10
            for row in results.values()
        ),
        "all_peak_values_strictly_below_one": all(
            bool(row["peak_strictly_below_one"]) for row in results.values()
        ),
        "cup_peak_strictly_below_one": bool(results["Cup"]["peak_strictly_below_one"]),
        "cup_normaliser_minus_e": constants["Cup"] - math.e,
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(verify_six_densities(), indent=2, sort_keys=True))
