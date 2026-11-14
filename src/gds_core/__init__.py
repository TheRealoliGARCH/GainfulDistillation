"""Core graph and transmission primitives for Gainful Distillation."""
from .graph import (
    ADJACENCY_MATRIX,
    NODE_LABELS,
    STUDENT_INDICES,
    build_adjacency_matrix,
    student_edge_multiplicity,
    validate_topology,
    weighted_degree,
)
from .transmission import TransmissionWeights

__all__ = [
    "ADJACENCY_MATRIX",
    "NODE_LABELS",
    "STUDENT_INDICES",
    "TransmissionWeights",
    "build_adjacency_matrix",
    "student_edge_multiplicity",
    "validate_topology",
    "weighted_degree",
]
