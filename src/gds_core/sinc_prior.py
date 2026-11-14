"""Proper Sinc-based priors with compact Fourier support.

The manuscript's raw Sinc-times-cosine expression is not generally
non-negative and is not automatically normalised. This module uses the
fourth-power Sinc envelope and a bounded cosine modulation:

    p(theta) = sinc(Delta * (theta - mu) / sigma)**4
               * (1 + rho*cos(2*pi*xi0*(theta-mu))) / Z.

Here sinc(x) = sin(pi*x)/(pi*x), Delta > 0, sigma > 0,
xi0 >= 0, and 0 <= rho <= 1.

The fourth power is non-negative, integrable, and has a finite second
moment. Its Fourier transform is the compactly supported convolution of
four rectangular spectra. Multiplication by the cosine shifts that spectrum
to the two sidebands. The normalising constant is evaluated analytically.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, log, pi, sin
from typing import Sequence

from .sinc_kernel import normalized_sinc


def sinc4_spectral_overlap(frequency: float) -> float:
    """Return the Fourier transform of normalized_sinc(x)**4.

    The Fourier convention is exp(-2*pi*i*f*x). The result is the
    convolution of four unit-width rectangular spectra, equivalently the
    convolution of two triangular spectra. It is non-negative and supported
    on [-2, 2].
    """
    value = float(frequency)
    if not isfinite(value):
        raise ValueError("frequency must be finite")
    value = abs(value)
    if value <= 1.0:
        return 2.0 / 3.0 - value * value + 0.5 * value**3
    if value < 2.0:
        return (2.0 - value) ** 3 / 6.0
    return 0.0


@dataclass(frozen=True)
class SincModulatedPrior:
    """A proper univariate Sinc-based prior density.

    Parameters
    ----------
    location:
        Location parameter mu.
    bandwidth:
        Positive dimensionless bandwidth multiplier Delta.
    scale:
        Positive scale sigma.
    center_frequency:
        Non-negative modulation frequency xi0, in cycles per parameter unit.
    modulation_depth:
        Cosine modulation rho, restricted to [0, 1] so the modulation
        factor is non-negative everywhere.
    """

    location: float = 0.0
    bandwidth: float = 1.0
    scale: float = 1.0
    center_frequency: float = 0.0
    modulation_depth: float = 0.5

    def __post_init__(self) -> None:
        fields = (
            ("location", self.location),
            ("bandwidth", self.bandwidth),
            ("scale", self.scale),
            ("center_frequency", self.center_frequency),
            ("modulation_depth", self.modulation_depth),
        )
        for name, raw_value in fields:
            if isinstance(raw_value, bool):
                raise ValueError(f"{name} must be a finite real number")
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite real number") from exc
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

        if self.bandwidth <= 0.0:
            raise ValueError("bandwidth must be strictly positive")
        if self.scale <= 0.0:
            raise ValueError("scale must be strictly positive")
        if self.center_frequency < 0.0:
            raise ValueError("center_frequency must be non-negative")
        if not 0.0 <= self.modulation_depth <= 1.0:
            raise ValueError("modulation_depth must lie in [0, 1]")

    @property
    def scaled_frequency(self) -> float:
        """Return xi0*sigma/Delta, the frequency in envelope units."""
        return self.center_frequency * self.scale / self.bandwidth

    @property
    def normalization_constant(self) -> float:
        """Return the exact integral of the unnormalised density."""
        return (self.scale / self.bandwidth) * (
            2.0 / 3.0
            + self.modulation_depth * sinc4_spectral_overlap(self.scaled_frequency)
        )

    def _centred_argument(self, value: float) -> tuple[float, float]:
        if isinstance(value, bool):
            raise ValueError("value must be a finite real number")
        try:
            x = float(value) - self.location
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("value must be a finite real number") from exc
        if not isfinite(x):
            raise ValueError("value must be finite")
        argument = self.bandwidth * x / self.scale
        if not isfinite(argument):
            raise ValueError("scaled value is outside the finite numerical range")
        return x, argument

    def density(self, value: float) -> float:
        """Evaluate the normalised prior density at value."""
        x, argument = self._centred_argument(value)
        sinc_value = normalized_sinc(argument)
        modulation = 1.0 + self.modulation_depth * cos(
            2.0 * pi * self.center_frequency * x
        )
        # Round-off can produce a tiny negative value at a theoretical zero.
        modulation = max(0.0, modulation)
        return (sinc_value**4) * modulation / self.normalization_constant

    def log_density(self, value: float) -> float:
        """Evaluate log density, returning -inf at exact density zeros."""
        x, argument = self._centred_argument(value)
        modulation = 1.0 + self.modulation_depth * cos(
            2.0 * pi * self.center_frequency * x
        )
        if modulation <= 0.0:
            return float("-inf")
        if argument == 0.0:
            log_sinc = 0.0
        else:
            numerator = abs(sin(pi * argument))
            if numerator == 0.0:
                return float("-inf")
            log_sinc = log(numerator) - log(pi * abs(argument))
        return (
            4.0 * log_sinc
            + log(modulation)
            - log(self.normalization_constant)
        )


@dataclass(frozen=True)
class IndependentSincPrior:
    """A product prior with independent SincModulatedPrior marginals."""

    marginals: tuple[SincModulatedPrior, ...]

    def __post_init__(self) -> None:
        values = tuple(self.marginals)
        if not values:
            raise ValueError("marginals must contain at least one prior")
        if any(not isinstance(value, SincModulatedPrior) for value in values):
            raise ValueError("every marginal must be a SincModulatedPrior")
        object.__setattr__(self, "marginals", values)

    def _values(self, values: Sequence[float]) -> tuple[float, ...]:
        try:
            result = tuple(values)
        except TypeError as exc:
            raise ValueError("values must be a sequence") from exc
        if len(result) != len(self.marginals):
            raise ValueError("values length must match the number of marginals")
        return result

    def log_density(self, values: Sequence[float]) -> float:
        """Evaluate the joint log density under coordinate independence."""
        total = 0.0
        for prior, value in zip(self.marginals, self._values(values)):
            term = prior.log_density(value)
            if term == float("-inf"):
                return term
            total += term
        return total

    def density(self, values: Sequence[float]) -> float:
        """Evaluate the joint product density under coordinate independence."""
        log_value = self.log_density(values)
        if log_value == float("-inf"):
            return 0.0
        # Avoid overflow from exponentiating a mathematically valid but
        # exceptionally concentrated/high-dimensional density.
        try:
            from math import exp
            return exp(log_value)
        except OverflowError:
            return float("inf")
