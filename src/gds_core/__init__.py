"""Core graph, learning, routing, allocation, and validation primitives."""

from .balanced_loss import (
    BalancedLossVariance1D,
    ScalarScoreSummary,
    balanced_loss_variance_1d,
)

from .graph import (
    ADJACENCY_MATRIX,
    NODE_LABELS,
    STUDENT_INDICES,
    build_adjacency_matrix,
    student_edge_multiplicity,
    validate_topology,
    weighted_degree,
)
from .posterior import Posterior1D, compute_posterior_1d
from .prior_diagnostics import (
    PosteriorBiasDiagnostic1D,
    PriorBiasThresholds,
    PriorPredictiveResult1D,
    PriorSummary1D,
    assess_posterior_bias_1d,
    prior_predictive_check_1d,
    summarize_prior_1d,
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
from .sinc_prior import (
    IndependentSincPrior,
    SincPriorMixture,
    contamination_robust_prior,
    SincModulatedPrior,
    sinc4_spectral_overlap,
)
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
    "BalancedLossVariance1D",
    "NODE_LABELS",
    "Posterior1D",
    "PosteriorBiasDiagnostic1D",
    "PriorBiasThresholds",
    "PriorPredictiveResult1D",
    "PriorSummary1D",
    "STUDENT_INDICES",
    "Finding",
    "IndependentSincPrior",
    "SincPriorMixture",
    "LearningObservation",
    "MetricEstimate",
    "ResourceAllocationResult",
    "ReviewVerdict",
    "RoutingDecision",
    "RoutingWeights",
    "SLDEEstimate",
    "SandwichCovariance",
    "ScalarScoreSummary",
    "SincModulatedPrior",
    "StudentLearningDynamicsEstimator",
    "TaskCandidate",
    "TaskRequest",
    "TransmissionWeights",
    "ValidatorReview",
    "VerificationDecision",
    "VerificationStatus",
    "adjudicate",
    "allocate_resources",
    "assess_posterior_bias_1d",
    "balanced_loss_variance_1d",
    "build_adjacency_matrix",
    "candidate_from_slde",
    "compute_posterior_1d",
    "contamination_robust_prior",
    "normalized_sinc",
    "prior_predictive_check_1d",
    "route_task",
    "sandwich_covariance",
    "sinc_covariance",
    "sinc_covariance_matrix",
    "sinc4_spectral_overlap",
    "student_edge_multiplicity",
    "summarize_prior_1d",
    "validate_topology",
    "weighted_degree",
]
