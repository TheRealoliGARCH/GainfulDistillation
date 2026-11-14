"""Directed, adaptive transmission strengths on the fixed GDS topology."""
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from .graph import ADJACENCY_MATRIX, NODE_LABELS


def _validate_nonnegative_finite(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    if len(values) != 5:
        raise ValueError(f"{name} must contain exactly five values")
    return tuple(_validate_nonnegative_finite(v, f"{name}[{i}]") for i, v in enumerate(values))


@dataclass(frozen=True)
class TransmissionWeights:
    """Directed channels, distinct from structural adjacency.

    Matrix convention: matrix[source][target] is the strength from source
    to target. Peer multipliers follow the same source-to-target convention
    and are multiplied by the structural edge multiplicity.
    """

    teacher_to_student: tuple[float, ...]
    student_to_teacher: tuple[float, ...]
    peer_to_student: tuple[tuple[float, ...], ...]

    @classmethod
    def from_channels(
        cls,
        teacher_to_student: Sequence[float],
        student_to_teacher: Sequence[float],
        peer_to_student: Sequence[Sequence[float]],
    ) -> "TransmissionWeights":
        tau = _vector(teacher_to_student, "teacher_to_student")
        nu = _vector(student_to_teacher, "student_to_teacher")
        if len(peer_to_student) != 5 or any(len(row) != 5 for row in peer_to_student):
            raise ValueError("peer_to_student must be a 5 by 5 matrix")
        peer = tuple(
            tuple(
                0.0 if source == target else _validate_nonnegative_finite(
                    peer_to_student[source][target],
                    f"peer_to_student[{source}][{target}]",
                )
                for target in range(5)
            )
            for source in range(5)
        )
        return cls(tau, nu, peer)

    def as_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return the 6x6 directed matrix W, indexed [source][target]."""
        matrix = [[0.0 for _ in NODE_LABELS] for _ in NODE_LABELS]
        for target, strength in enumerate(self.teacher_to_student):
            matrix[0][target + 1] = strength
        for source, strength in enumerate(self.student_to_teacher):
            matrix[source + 1][0] = strength
        for source in range(5):
            for target in range(5):
                if source == target:
                    continue
                structural_weight = ADJACENCY_MATRIX[source + 1][target + 1]
                matrix[source + 1][target + 1] = (
                    structural_weight * self.peer_to_student[source][target]
                )
        return tuple(tuple(row) for row in matrix)

    def outgoing_strength(self, source_index: int) -> float:
        if not 0 <= source_index < len(NODE_LABELS):
            raise IndexError("node index must be between 0 and 5")
        return sum(self.as_matrix()[source_index])
