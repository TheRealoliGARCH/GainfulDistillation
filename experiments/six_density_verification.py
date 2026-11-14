"""Reproducible numerical audit of the six densities in the supplied PDF.

The Mahaloy expression uses the unnormalised sinc convention sin(u)/u,
which is consistent with its printed normaliser and plotted zero locations.
Only the Python standard library is required.
"""
import json
import math

TOL = 2.0e-11


def _adaptive_simpson(f, a, b, tolerance=TOL, depth=22):
    if a == b:
        return 0.0
    m = (a + b) / 2.0
    fa, fm, fb = f(a), f(m), f(b)
    whole = (b - a) * (fa + 4.0 * fm + fb) / 6.0

    def recurse(left, right, fl, fc, fr, old, tol, remaining):
        mid = (left + right) / 2.0
        q1, q3 = (left + mid) / 2.0, (mid + right) / 2.0
        fq1, fq3 = f(q1), f(q3)
        first = (mid - left) * (fl + 4.0 * fq1 + fc) / 6.0
        second = (right - mid) * (fc + 4.0 * fq3 + fr) / 6.0
        refined = first + second
        delta = refined - old
        if remaining <= 0 or abs(delta) <= 15.0 * tol:
            return refined + delta / 15.0
        return recurse(left, mid, fl, fq1, fc, first, tol / 2, remaining - 1) + recurse(
            mid, right, fc, fq3, fr, second, tol / 2, remaining - 1
        )

    return recurse(a, b, fa, fm, fb, whole, tolerance, depth)


def _integrate(f, points):
    points = tuple(points)
    tol = TOL / (len(points) - 1)
    return sum(
        _adaptive_simpson(f, a, b, tolerance=tol)
        for a, b in zip(points, points[1:])
    )


def fresnel_c(x):
    sign = -1.0 if x < 0 else 1.0
    return sign * _adaptive_simpson(
        lambda t: math.cos(math.pi * t * t / 2), 0.0, abs(x)
    )


def fresnel_s(x):
    sign = -1.0 if x < 0 else 1.0
    return sign * _adaptive_simpson(
        lambda t: math.sin(math.pi * t * t / 2), 0.0, abs(x)
    )


def erfi_one():
    # erfi(1) = 2/sqrt(pi) * sum_{n>=0} 1/(n!*(2n+1)).
    total, term = 0.0, 1.0
    for n in range(100):
        addend = term / (2 * n + 1)
        total += addend
        if abs(addend) < 2e-17:
            break
        term /= n + 1
    return 2.0 * total / math.sqrt(math.pi)


def verify_six_densities():
    root_pi = math.sqrt(math.pi)
    normalisers = {
        "Ghosh": 2.0 * (1.0 + fresnel_c(math.sqrt(6)) / math.sqrt(6)),
        "Mahaloy": 0.25 * (
            -math.sin(8)
            - 8 * root_pi * fresnel_c(2)
            + 8 * root_pi * fresnel_c(math.sqrt(2))
            + 4 * root_pi * fresnel_c(4 / root_pi)
        ),
        "Madras": math.sqrt(2) * fresnel_s(math.sqrt(2)),
        "Temple bell": 2 * math.gamma(5 / 4),
        "Rocket cone": root_pi * math.erf(1),
        "Cup": root_pi * erfi_one(),
    }

    def compact(kernel, normaliser):
        return lambda x: kernel(x) / normaliser if abs(x) <= 1 else 0.0

    sinc = lambda u: 1.0 if u == 0 else math.sin(u) / u
    pdfs = {
        "Ghosh": compact(lambda x: 1 + math.cos(3 * math.pi * x * x),
                        normalisers["Ghosh"]),
        "Mahaloy": compact(lambda x: abs(sinc(8 * x * x)),
                           normalisers["Mahaloy"]),
        "Madras": compact(lambda x: abs(math.sin(math.pi * x * x)),
                          normalisers["Madras"]),
        "Temple bell": lambda x: math.exp(-(x ** 4)) / normalisers["Temple bell"],
        "Rocket cone": compact(lambda x: math.exp(-(x * x)),
                               normalisers["Rocket cone"]),
        "Cup": compact(lambda x: math.exp(x * x), normalisers["Cup"]),
    }
    roots = (0.0, math.sqrt(math.pi / 8), math.sqrt(math.pi / 4), 1.0)
    grids = {
        "Ghosh": (0.0, 1.0),
        "Mahaloy": roots,
        "Madras": (0.0, 1.0),
        "Temple bell": (0.0, 1.0, 2.0, 4.0, 8.0, 12.0),
        "Rocket cone": (0.0, 1.0),
        "Cup": (0.0, 1.0),
    }
    kernel_peaks = {
        "Ghosh": 2.0, "Mahaloy": 1.0, "Madras": 1.0,
        "Temple bell": 1.0, "Rocket cone": 1.0, "Cup": math.e,
    }
    results = {}
    for name, pdf in pdfs.items():
        grid = grids[name]

        def moment(power):
            return 2.0 * _integrate(lambda x: x ** power * pdf(x), grid)

        variance, fourth = moment(2), moment(4)
        peak = kernel_peaks[name] / normalisers[name]
        results[name] = {
            "normalising_constant": normalisers[name],
            "integral_numeric": moment(0),
            "peak_density": peak,
            "mean": 0.0,
            "variance": variance,
            "fourth_moment": fourth,
            "kurtosis": fourth / variance ** 2,
            "strictly_below_one": peak < 1.0,
        }

    return {
        "method": "adaptive Simpson quadrature; symmetry; split at Mahaloy zeros",
        "sinc_convention": "sinc(u) = sin(u)/u (unnormalised)",
        "results": results,
        "all_integrals_within_1e-8_of_one": all(
            abs(row["integral_numeric"] - 1.0) < 1e-8
            for row in results.values()
        ),
        "all_peaks_strictly_below_one": all(
            row["strictly_below_one"] for row in results.values()
        ),
        "cup_peak_is_strictly_below_one": results["Cup"]["peak_density"] < 1.0,
        "cup_normaliser_minus_e": normalisers["Cup"] - math.e,
    }


if __name__ == "__main__":
    print(json.dumps(verify_six_densities(), indent=2, sort_keys=True))
