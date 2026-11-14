"""Probabilistic student task routing for Gainful Distillation.

Scores combine estimated learning response, task-relevant Breadth/Depth fit,
uncertainty as an exploration signal, and execution cost. Probabilities are
softmax-normalised with an explicit per-candidate exploration floor.
"""
from dataclasses import dataclass
from math import exp, isfinite
from typing import Optional, Sequence

from .slde import SLDEEstimate


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _unit_interval(value: float, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


@dataclass(frozen=True)
class TaskRequest:
    """Task-level requirements, expressed on a normalised [0, 1] scale."""

    task_id: str
    domain: str
    breadth_requirement: float = 0.0
    depth_requirement: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("domain must be a non-empty string")
        object.__setattr__(self, "task_id", self.task_id.strip())
        object.__setattr__(self, "domain", self.domain.strip())
        object.__setattr__(
            self, "breadth_requirement",
            _unit_interval(self.breadth_requirement, "breadth_requirement"),
        )
        object.__setattr__(
            self, "depth_requirement",
            _unit_interval(self.depth_requirement, "depth_requirement"),
        )


@dataclass(frozen=True)
class TaskCandidate:
    """Candidate student profile for one task.

    expected_gain is the estimated task-relevant gain, uncertainty is a
    non-negative uncertainty measure, and optional fit metrics lie in [0, 1].
    Missing fit metrics remain missing and are handled using the explicit
    neutral_fit prior configured in RoutingWeights.
    """

    student_id: str
    expected_gain: float
    uncertainty: float = 0.0
    breadth_fit: Optional[float] = None
    depth_fit: Optional[float] = None
    execution_cost: float = 0.0
    domain: Optional[str] = None
    method: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.student_id, str) or not self.student_id.strip():
            raise ValueError("student_id must be a non-empty string")
        object.__setattr__(self, "student_id", self.student_id.strip())
        for name in ("domain", "method"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be None or a non-empty string")
                object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "expected_gain", _finite(self.expected_gain, "expected_gain"))
        uncertainty = _finite(self.uncertainty, "uncertainty")
        cost = _finite(self.execution_cost, "execution_cost")
        if uncertainty < 0.0:
            raise ValueError("uncertainty must be non-negative")
        if cost < 0.0:
            raise ValueError("execution_cost must be non-negative")
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "execution_cost", cost)
        for name in ("breadth_fit", "depth_fit"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit_interval(value, name))


@dataclass(frozen=True)
class RoutingWeights:
    """Non-negative score weights and a neutral prior for missing fit metrics."""

    breadth: float = 1.0
    depth: float = 1.0
    uncertainty: float = 0.1
    cost: float = 0.1
    neutral_fit: float = 0.5

    def __post_init__(self) -> None:
        for name in ("breadth", "depth", "uncertainty", "cost"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} weight must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "neutral_fit", _unit_interval(self.neutral_fit, "neutral_fit"))


@dataclass(frozen=True)
class RoutingDecision:
    """Auditable routing scores, probabilities, and deterministic shortlist."""

    task_id: str
    domain: str
    scores: tuple[tuple[str, float], ...]
    probabilities: tuple[tuple[str, float], ...]
    selected_student_ids: tuple[str, ...]
    temperature: float
    exploration_floor: float


def candidate_from_slde(
    estimate: SLDEEstimate,
    *,
    execution_cost: float = 0.0,
) -> TaskCandidate:
    """Convert one SLDE profile into a routing candidate.

    The RLS learning-rate estimate is used as expected gain. Its reported
    standard error supplies the exploration uncertainty signal. Optional
    Breadth and Depth means remain None when unobserved.
    """
    return TaskCandidate(
        student_id=estimate.student_id,
        expected_gain=estimate.learning_rate,
        uncertainty=estimate.learning_rate_standard_error or 0.0,
        breadth_fit=None if estimate.breadth is None else estimate.breadth.mean,
        depth_fit=None if estimate.depth is None else estimate.depth.mean,
        execution_cost=execution_cost,
        domain=estimate.domain,
        method=estimate.method,
    )


def route_task(
    task: TaskRequest,
    candidates: Sequence[TaskCandidate],
    *,
    weights: RoutingWeights = RoutingWeights(),
    temperature: float = 1.0,
    exploration_floor: float = 0.02,
    coalition_size: int = 1,
) -> RoutingDecision:
    """Return softmax probabilities and a top-k shortlist for a task.

    The floor is a probability lower bound for every candidate. It must
    satisfy n * floor < 1. The shortlist is deterministic (highest
    probability first); probabilities are exposed separately for stochastic
    sampling by a downstream scheduler.
    """
    if not candidates:
        raise ValueError("at least one candidate is required")
    ids = [candidate.student_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate student_id values must be unique")
    if not isinstance(coalition_size, int) or isinstance(coalition_size, bool):
        raise ValueError("coalition_size must be an integer")
    if not 1 <= coalition_size <= len(candidates):
        raise ValueError("coalition_size must be between one and candidate count")

    temperature = _finite(temperature, "temperature")
    floor = _finite(exploration_floor, "exploration_floor")
    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive")
    if floor < 0.0 or len(candidates) * floor >= 1.0:
        raise ValueError("exploration_floor must satisfy 0 <= n * floor < 1")

    scored: list[tuple[str, float]] = []
    for candidate in candidates:
        if candidate.domain is not None and candidate.domain != task.domain:
            raise ValueError(
                f"candidate {candidate.student_id} has domain {candidate.domain!r}, "
                f"but task domain is {task.domain!r}"
            )
        breadth_fit = (
            weights.neutral_fit if candidate.breadth_fit is None else candidate.breadth_fit
        )
        depth_fit = (
            weights.neutral_fit if candidate.depth_fit is None else candidate.depth_fit
        )
        score = (
            candidate.expected_gain
            + weights.breadth * task.breadth_requirement * breadth_fit
            + weights.depth * task.depth_requirement * depth_fit
            + weights.uncertainty * candidate.uncertainty
            - weights.cost * candidate.execution_cost
        )
        scored.append((candidate.student_id, score))

    maximum = max(score for _, score in scored)
    exponentials = [
        (student_id, exp((score - maximum) / temperature))
        for student_id, score in scored
    ]
    denominator = sum(value for _, value in exponentials)
    residual_mass = 1.0 - len(candidates) * floor
    probabilities = [
        (student_id, floor + residual_mass * value / denominator)
        for student_id, value in exponentials
    ]
    probabilities.sort(key=lambda item: (-item[1], item[0]))
    ordered_scores = tuple(sorted(scored, key=lambda item: item[0]))
    return RoutingDecision(
        task_id=task.task_id,
        domain=task.domain,
        scores=ordered_scores,
        probabilities=tuple(probabilities),
        selected_student_ids=tuple(
            student_id for student_id, _ in probabilities[:coalition_size]
        ),
        temperature=temperature,
        exploration_floor=floor,
    )
