"""Core graph, transmission, learning-estimation, and routing primitives."""

from .graph import (
    ADJACENCY_MATRIX,
    NODE_LABELS,
    STUDENT_INDICES,
    build_adjacency_matrix,
    student_edge_multiplicity,
    validate_topology,
    weighted_degree,
)
from .resource_allocation import ResourceAllocationResult, allocate_resources
from .routing import (
    RoutingDecision,
    RoutingWeights,
    TaskCandidate,
    TaskRequest,
    candidate_from_slde,
    route_task,
)
from .slde import (
    LearningObservation,
    MetricEstimate,
    SLDEEstimate,
    StudentLearningDynamicsEstimator,
)
from .transmission import TransmissionWeights

__all__ = [
    "ADJACENCY_MATRIX",
    "NODE_LABELS",
    "STUDENT_INDICES",
    "LearningObservation",
    "MetricEstimate",
    "ResourceAllocationResult",
    "RoutingDecision",
    "RoutingWeights",
    "SLDEEstimate",
    "StudentLearningDynamicsEstimator",
    "TaskCandidate",
    "TaskRequest",
    "TransmissionWeights",
    "allocate_resources",
    "build_adjacency_matrix",
    "candidate_from_slde",
    "route_task",
    "student_edge_multiplicity",
    "validate_topology",
    "weighted_degree",
]
