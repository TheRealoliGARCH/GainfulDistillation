"""Online Student Learning Dynamics Estimator (SLDE).

One profile is maintained per student × domain × method. Learning response
is estimated from performance change and instructional dose using scalar
recursive least squares (RLS). Optional bounded metrics are tracked
separately; missing values are never imputed.
"""
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Optional

_SCORE_METRICS = ("retention", "transfer", "breadth", "depth")


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _score(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


@dataclass(frozen=True)
class LearningObservation:
    """One measured instructional episode with normalised performance scores."""

    student_id: str
    domain: str
    method: str
    instructional_dose: float
    performance_before: float
    performance_after: float
    retention: Optional[float] = None
    transfer: Optional[float] = None
    breadth: Optional[float] = None
    depth: Optional[float] = None

    def __post_init__(self) -> None:
        _identifier(self.student_id, "student_id")
        _identifier(self.domain, "domain")
        _identifier(self.method, "method")
        dose = _finite(self.instructional_dose, "instructional_dose")
        if dose <= 0.0:
            raise ValueError("instructional_dose must be strictly positive")
        before = _score(self.performance_before, "performance_before")
        after = _score(self.performance_after, "performance_after")
        object.__setattr__(self, "instructional_dose", dose)
        object.__setattr__(self, "performance_before", before)
        object.__setattr__(self, "performance_after", after)
        for metric in _SCORE_METRICS:
            object.__setattr__(self, metric, _score(getattr(self, metric), metric))

    @property
    def performance_change(self) -> float:
        return self.performance_after - self.performance_before


@dataclass(frozen=True)
class MetricEstimate:
    """Running descriptive estimate for one bounded observed metric."""

    count: int
    mean: float
    sample_variance: Optional[float]
    standard_error: Optional[float]


@dataclass(frozen=True)
class SLDEEstimate:
    """Immutable snapshot of one student-domain-method learning profile."""

    student_id: str
    domain: str
    method: str
    observations: int
    learning_rate: float
    rls_covariance: float
    residual_variance: Optional[float]
    learning_rate_standard_error: Optional[float]
    retention: Optional[MetricEstimate]
    transfer: Optional[MetricEstimate]
    breadth: Optional[MetricEstimate]
    depth: Optional[MetricEstimate]


class _RunningMoments:
    """Numerically stable online mean and sample variance (Welford update)."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    def estimate(self) -> Optional[MetricEstimate]:
        if self.count == 0:
            return None
        variance = self._m2 / (self.count - 1) if self.count > 1 else None
        standard_error = sqrt(variance / self.count) if variance is not None else None
        return MetricEstimate(self.count, self.mean, variance, standard_error)


class _Profile:
    def __init__(self, prior_mean: float, prior_covariance: float, forgetting_factor: float):
        self.learning_rate = prior_mean
        self.covariance = prior_covariance
        self.forgetting_factor = forgetting_factor
        self.observations = 0
        self.residuals = _RunningMoments()
        self.metrics = {name: _RunningMoments() for name in _SCORE_METRICS}

    def update(self, observation: LearningObservation) -> None:
        x = observation.instructional_dose
        y = observation.performance_change
        # Scalar RLS gain and covariance recursion.
        denominator = self.forgetting_factor + x * x * self.covariance
        gain = self.covariance * x / denominator
        innovation = y - x * self.learning_rate
        self.learning_rate += gain * innovation
        self.covariance = (self.covariance - gain * x * self.covariance) / self.forgetting_factor
        self.residuals.update(innovation)
        self.observations += 1
        for name in _SCORE_METRICS:
            value = getattr(observation, name)
            if value is not None:
                self.metrics[name].update(value)

    def snapshot(self, student_id: str, domain: str, method: str) -> SLDEEstimate:
        residual = self.residuals.estimate()
        residual_variance = residual.sample_variance if residual is not None else None
        rate_se = sqrt(max(0.0, residual_variance * self.covariance)) if residual_variance is not None else None
        return SLDEEstimate(
            student_id, domain, method, self.observations, self.learning_rate,
            self.covariance, residual_variance, rate_se,
            self.metrics["retention"].estimate(), self.metrics["transfer"].estimate(),
            self.metrics["breadth"].estimate(), self.metrics["depth"].estimate(),
        )


class StudentLearningDynamicsEstimator:
    """Maintain online RLS profiles for student/domain/method keys.

    A forgetting factor of one gives ordinary RLS. Values below one discount
    older observations and are a modelling choice that should be reported.
    """

    def __init__(self, *, prior_mean: float = 0.0, prior_covariance: float = 1000.0, forgetting_factor: float = 1.0):
        prior_mean = _finite(prior_mean, "prior_mean")
        prior_covariance = _finite(prior_covariance, "prior_covariance")
        forgetting_factor = _finite(forgetting_factor, "forgetting_factor")
        if prior_covariance <= 0.0:
            raise ValueError("prior_covariance must be strictly positive")
        if not 0.0 < forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must lie in (0, 1]")
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.forgetting_factor = forgetting_factor
        self._profiles: dict[tuple[str, str, str], _Profile] = {}

    def update(self, observation: LearningObservation) -> SLDEEstimate:
        """Update one profile and return its post-update estimate."""
        key = (observation.student_id, observation.domain, observation.method)
        profile = self._profiles.get(key)
        if profile is None:
            profile = _Profile(self.prior_mean, self.prior_covariance, self.forgetting_factor)
            self._profiles[key] = profile
        profile.update(observation)
        return profile.snapshot(*key)

    def estimate(self, student_id: str, domain: str, method: str) -> Optional[SLDEEstimate]:
        """Return a snapshot, or None if this profile has no observations."""
        key = (_identifier(student_id, "student_id"), _identifier(domain, "domain"), _identifier(method, "method"))
        profile = self._profiles.get(key)
        return None if profile is None else profile.snapshot(*key)

    def profile_count(self) -> int:
        """Return the number of distinct student-domain-method profiles."""
        return len(self._profiles)
