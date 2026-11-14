"""Dependency-free asymptotic sandwich covariance calculations.

For an estimating-equation estimator with sensitivity A and score/estimating-
function variability B, this module computes A^{-1} B A^{-T}. This is the
asymptotic covariance of sqrt(n) times the estimation error; dividing by n
gives the first-order covariance approximation for the estimator itself.

This is a frequentist sandwich building block. It is not, by itself, the
Bayesian balanced-loss procedure of Li and Rice (2022/2024), nor does it
manufacture a robust covariance without model-specific A and B inputs.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class SandwichCovariance:
    """Asymptotic covariance and optional finite-sample approximation."""
    asymptotic: Matrix
    estimator_covariance: Matrix | None
    sample_size: int | None


def _matrix(values: Sequence[Sequence[float]], name: str) -> Matrix:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    rows = tuple(tuple(float(value) for value in row) for row in values)
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular and non-empty")
    if any(not isfinite(value) for row in rows for value in row):
        raise ValueError(f"{name} must contain only finite values")
    return rows


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0])))


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    if len(left[0]) != len(right):
        raise ValueError("incompatible matrix dimensions")
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(len(right))) for j in range(len(right[0])))
        for i in range(len(left))
    )


def _inverse(matrix: Matrix) -> Matrix:
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("sensitivity matrix must be square")
    work = [list(matrix[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = max(abs(value) for row in matrix for value in row)
    if scale == 0.0:
        raise ValueError("sensitivity matrix is singular")
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= 1e-14 * scale:
            raise ValueError("sensitivity matrix is singular or numerically singular")
        work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        work[column] = [value / divisor for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            factor = work[row][column]
            work[row] = [work[row][j] - factor * work[column][j] for j in range(2 * n)]
    return tuple(tuple(work[i][n:]) for i in range(n))


def sandwich_covariance(
    sensitivity: Sequence[Sequence[float]],
    variability: Sequence[Sequence[float]],
    *,
    sample_size: int | None = None,
) -> SandwichCovariance:
    """Compute A^-1 B A^-T, optionally also returning the /n estimate.

    Sensitivity is the empirical average negative Jacobian (bread inverse
    before inversion); variability is the empirical covariance/outer-product
    meat on the same scaling convention. The caller must choose an estimating
    equation and consistent scaling for both matrices. Symmetry of the meat
    is checked; positive semidefiniteness remains an input/model condition.
    """
    a = _matrix(sensitivity, "sensitivity")
    b = _matrix(variability, "variability")
    if len(a) != len(a[0]):
        raise ValueError("sensitivity matrix must be square")
    p = len(a)
    if len(b) != p or len(b[0]) != p:
        raise ValueError("variability matrix must match sensitivity dimensions")
    for i in range(p):
        for j in range(p):
            if abs(b[i][j] - b[j][i]) > 1e-10:
                raise ValueError("variability matrix must be symmetric")
    if sample_size is not None and (isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0):
        raise ValueError("sample_size must be a positive integer")
    a_inv = _inverse(a)
    asymptotic = _multiply(_multiply(a_inv, b), _transpose(a_inv))
    asymptotic = tuple(
        tuple(0.5 * (asymptotic[i][j] + asymptotic[j][i]) for j in range(p))
        for i in range(p)
    )
    estimator = None if sample_size is None else tuple(
        tuple(value / sample_size for value in row) for row in asymptotic
    )
    return SandwichCovariance(asymptotic, estimator, sample_size)
