"""Core graph, learning, routing, allocation, and validation primitives."""

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
from .sandwich import SandwichCovariance, sandwich_covariance
from .sinc_kernel import normalized_sinc, sinc_covariance, sinc_covariance_matrix
from .slde import (
    LearningObservation,
    MetricEstimate,
    SLDEEstimate,
    StudentLearningDynamicsEstimator,
)
from .transmission import TransmissionWeights
from .verification import (
    Finding,
    ReviewVerdict,
    ValidatorReview,
    VerificationDecision,
    VerificationStatus,
    adjudicate,
)

__all__ = [
    "ADJACENCY_MATRIX",
    "NODE_LABELS",
    "STUDENT_INDICES",
    "Finding",
    "LearningObservation",
    "MetricEstimate",
    "ResourceAllocationResult",
    "ReviewVerdict",
    "RoutingDecision",
    "RoutingWeights",
    "SLDEEstimate",
    "SandwichCovariance",
    "StudentLearningDynamicsEstimator",
    "TaskCandidate",
    "TaskRequest",
    "TransmissionWeights",
    "ValidatorReview",
    "VerificationDecision",
    "VerificationStatus",
    "adjudicate",
    "allocate_resources",
    "build_adjacency_matrix",
    "candidate_from_slde",
    "normalized_sinc",
    "route_task",
    "sandwich_covariance",
    "sinc_covariance",
    "sinc_covariance_matrix",
    "student_edge_multiplicity",
    "validate_topology",
    "weighted_degree",
]
