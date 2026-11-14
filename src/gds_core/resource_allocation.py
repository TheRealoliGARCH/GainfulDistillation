"""Constrained quadratic resource allocation for Gainful Distillation.

Maximises a concave quadratic objective over a capped simplex using projected
gradient ascent. Returns convergence diagnostics; no numerical dependency.
"""
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(values: Sequence[float], n: int, name: str) -> tuple[float, ...]:
    if len(values) != n:
        raise ValueError(f"{name} must contain exactly {n} values")
    return tuple(_finite(value, f"{name}[{i}]") for i, value in enumerate(values))


def _validate_covariance(values: Sequence[Sequence[float]], n: int) -> tuple[tuple[float, ...], ...]:
    if len(values) != n or any(len(row) != n for row in values):
        raise ValueError(f"covariance must be an {n} by {n} matrix")
    matrix = [[_finite(values[i][j], f"covariance[{i}][{j}]") for j in range(n)] for i in range(n)]
    scale = max(1.0, max(abs(value) for row in matrix for value in row))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-10 * scale:
                raise ValueError("covariance matrix must be symmetric")
            average = 0.5 * (matrix[i][j] + matrix[j][i])
            matrix[i][j] = matrix[j][i] = average

    # Jacobi rotations give a dependency-free positive-semidefinite check.
    diagonalised = [row[:] for row in matrix]
    for _ in range(max(1, 50 * n * n)):
        p, q, largest = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                magnitude = abs(diagonalised[i][j])
                if magnitude > largest:
                    p, q, largest = i, j, magnitude
        if largest <= 1e-12 * scale:
            break
        app, aqq, apq = diagonalised[p][p], diagonalised[q][q], diagonalised[p][q]
        tau = (aqq - app) / (2.0 * apq)
        sign = 1.0 if tau >= 0.0 else -1.0
        tangent = sign / (abs(tau) + sqrt(1.0 + tau * tau))
        cosine = 1.0 / sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        for k in range(n):
            if k in (p, q):
                continue
            akp, akq = diagonalised[k][p], diagonalised[k][q]
            diagonalised[k][p] = diagonalised[p][k] = cosine * akp - sine * akq
            diagonalised[k][q] = diagonalised[q][k] = sine * akp + cosine * akq
        diagonalised[p][p] = app - tangent * apq
        diagonalised[q][q] = aqq + tangent * apq
        diagonalised[p][q] = diagonalised[q][p] = 0.0
    if min(diagonalised[i][i] for i in range(n)) < -1e-10 * scale:
        raise ValueError("covariance matrix must be positive semidefinite")
    return tuple(tuple(row) for row in matrix)


def _project_capped_simplex(values, lower, upper, budget):
    """Project onto lower <= x <= upper and sum(x) <= budget."""
    clipped = [min(upper[i], max(lower[i], values[i])) for i in range(len(values))]
    if sum(clipped) <= budget:
        return tuple(clipped)
    low = min(values[i] - upper[i] for i in range(len(values))) - 1.0
    high = max(values[i] - lower[i] for i in range(len(values))) + 1.0
    for _ in range(120):
        threshold = 0.5 * (low + high)
        total = sum(min(upper[i], max(lower[i], values[i] - threshold)) for i in range(len(values)))
        if total > budget:
            low = threshold
        else:
            high = threshold
    threshold = 0.5 * (low + high)
    return tuple(min(upper[i], max(lower[i], values[i] - threshold)) for i in range(len(values)))


@dataclass(frozen=True)
class ResourceAllocationResult:
    """Allocation and audit diagnostics for one optimisation run."""
    student_ids: tuple[str, ...]
    allocations: tuple[tuple[str, float], ...]
    objective_value: float
    expected_gain: float
    covariance_penalty: float
    regularization_penalty: float
    total_budget: float
    budget_used: float
    budget_unused: float
    iterations: int
    converged: bool
    projected_gradient_norm: float


def allocate_resources(
    student_ids: Sequence[str],
    expected_gains: Sequence[float],
    covariance: Sequence[Sequence[float]],
    total_budget: float,
    *,
    risk_aversion: float = 1.0,
    regularization: float = 0.1,
    min_share: float = 0.0,
    max_share: float = 1.0,
    tolerance: float = 1e-9,
    max_iterations: int = 20_000,
) -> ResourceAllocationResult:
    """Solve the constrained quadratic resource-allocation problem.

    Objective:
      mu.T @ r - 0.5 * risk_aversion * r.T @ covariance @ r
      - 0.5 * regularization * ||r||^2

    Constraints:
      sum(r) <= total_budget
      min_share * total_budget <= r_i <= max_share * total_budget

    A false converged flag means the iteration limit was reached.
    """
    ids = tuple(student_ids)
    n = len(ids)
    if n == 0:
        raise ValueError("at least one student is required")
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("student_ids must be non-empty strings")
    ids = tuple(item.strip() for item in ids)
    if len(set(ids)) != n:
        raise ValueError("student_ids must be unique")

    gains = _vector(expected_gains, n, "expected_gains")
    sigma = _validate_covariance(covariance, n)
    budget = _finite(total_budget, "total_budget")
    risk = _finite(risk_aversion, "risk_aversion")
    ridge = _finite(regularization, "regularization")
    minimum_share = _finite(min_share, "min_share")
    maximum_share = _finite(max_share, "max_share")
    tolerance = _finite(tolerance, "tolerance")
    if budget <= 0.0:
        raise ValueError("total_budget must be strictly positive")
    if risk < 0.0 or ridge < 0.0:
        raise ValueError("risk_aversion and regularization must be non-negative")
    if not 0.0 <= minimum_share <= maximum_share <= 1.0:
        raise ValueError("shares must satisfy 0 <= min_share <= max_share <= 1")
    if n * minimum_share > 1.0 + 1e-12:
        raise ValueError("minimum allocations exceed the total budget")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be strictly positive")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    lower = tuple(minimum_share * budget for _ in range(n))
    upper = tuple(maximum_share * budget for _ in range(n))
    hessian = tuple(
        tuple(risk * sigma[i][j] + (ridge if i == j else 0.0) for j in range(n))
        for i in range(n)
    )
    lipschitz = max(sum(abs(hessian[i][j]) for j in range(n)) for i in range(n))

    def gradient(point):
        return tuple(gains[i] - sum(hessian[i][j] * point[j] for j in range(n)) for i in range(n))

    def components(point):
        gross = sum(gains[i] * point[i] for i in range(n))
        variance = sum(point[i] * sigma[i][j] * point[j] for i in range(n) for j in range(n))
        covariance_penalty = 0.5 * risk * variance
        regularization_penalty = 0.5 * ridge * sum(value * value for value in point)
        return gross - covariance_penalty - regularization_penalty, gross, covariance_penalty, regularization_penalty

    # If the quadratic curvature is zero, solve the linear programme exactly.
    if lipschitz == 0.0:
        point = list(lower)
        remaining = max(0.0, budget - sum(point))
        for index in sorted(range(n), key=lambda i: (-gains[i], ids[i])):
            if gains[index] <= 0.0:
                break
            increment = min(upper[index] - point[index], remaining)
            point[index] += increment
            remaining -= increment
            if remaining <= 1e-14 * max(1.0, budget):
                break
        converged, iterations, projected_norm = True, 1, 0.0
    else:
        point = list(_project_capped_simplex([0.0] * n, lower, upper, budget))
        step = 1.0 / lipschitz
        converged, projected_norm, iterations = False, float("inf"), 0
        for iteration in range(1, max_iterations + 1):
            grad = gradient(point)
            trial = [point[i] + step * grad[i] for i in range(n)]
            next_point = _project_capped_simplex(trial, lower, upper, budget)
            difference = max(abs(next_point[i] - point[i]) for i in range(n))
            point = list(next_point)
            iterations = iteration
            projected_norm = difference / step
            scale = max(1.0, max(abs(value) for value in point))
            if difference <= tolerance * scale:
                converged = True
                break

    value, gross, covariance_penalty, regularization_penalty = components(point)
    used = sum(point)
    return ResourceAllocationResult(
        student_ids=ids,
        allocations=tuple((ids[i], point[i]) for i in range(n)),
        objective_value=value,
        expected_gain=gross,
        covariance_penalty=covariance_penalty,
        regularization_penalty=regularization_penalty,
        total_budget=budget,
        budget_used=used,
        budget_unused=max(0.0, budget - used),
        iterations=iterations,
        converged=converged,
        projected_gradient_norm=projected_norm,
    )
