import math

import pytest

from gds_core.routing import (
    RoutingWeights,
    TaskCandidate,
    TaskRequest,
    candidate_from_slde,
    route_task,
)
from gds_core.slde import LearningObservation, StudentLearningDynamicsEstimator


def test_probabilities_sum_to_one_and_respect_exploration_floor():
    task = TaskRequest("q1", "algebra")
    candidates = [
        TaskCandidate("S1", expected_gain=0.8),
        TaskCandidate("S2", expected_gain=0.2),
        TaskCandidate("S3", expected_gain=-0.1),
    ]
    decision = route_task(task, candidates, exploration_floor=0.03)
    probabilities = dict(decision.probabilities)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(value >= 0.03 for value in probabilities.values())
    assert decision.selected_student_ids == ("S1",)


def test_task_breadth_and_depth_requirements_change_ranking():
    candidates = [
        TaskCandidate("broad", expected_gain=0.0, breadth_fit=0.9, depth_fit=0.1),
        TaskCandidate("deep", expected_gain=0.0, breadth_fit=0.1, depth_fit=0.9),
    ]
    broad_task = TaskRequest("q-b", "research", breadth_requirement=1.0)
    deep_task = TaskRequest("q-d", "research", depth_requirement=1.0)
    assert route_task(broad_task, candidates).selected_student_ids == ("broad",)
    assert route_task(deep_task, candidates).selected_student_ids == ("deep",)


def test_uncertainty_can_reward_exploration():
    task = TaskRequest("q", "physics")
    candidates = [
        TaskCandidate("certain", expected_gain=0.5, uncertainty=0.0),
        TaskCandidate("uncertain", expected_gain=0.5, uncertainty=2.0),
    ]
    weights = RoutingWeights(breadth=0, depth=0, uncertainty=1.0, cost=0)
    result = route_task(task, candidates, weights=weights)
    assert dict(result.probabilities)["uncertain"] > dict(result.probabilities)["certain"]


def test_execution_cost_reduces_candidate_score():
    task = TaskRequest("q", "economics")
    candidates = [
        TaskCandidate("cheap", expected_gain=0.5, execution_cost=0.0),
        TaskCandidate("expensive", expected_gain=0.5, execution_cost=2.0),
    ]
    weights = RoutingWeights(breadth=0, depth=0, uncertainty=0, cost=1.0)
    result = route_task(task, candidates, weights=weights)
    assert result.selected_student_ids == ("cheap",)


def test_missing_fit_metrics_use_explicit_neutral_prior():
    task = TaskRequest("q", "history", breadth_requirement=1.0)
    candidates = [
        TaskCandidate("missing", expected_gain=0.0),
        TaskCandidate("neutral", expected_gain=0.0, breadth_fit=0.5),
    ]
    result = route_task(task, candidates)
    assert dict(result.scores)["missing"] == pytest.approx(dict(result.scores)["neutral"])


def test_coalition_shortlist_contains_top_k_candidates():
    task = TaskRequest("q", "mathematics")
    candidates = [
        TaskCandidate("S1", expected_gain=0.2),
        TaskCandidate("S2", expected_gain=0.9),
        TaskCandidate("S3", expected_gain=0.5),
    ]
    result = route_task(task, candidates, coalition_size=2)
    assert result.selected_student_ids == ("S2", "S3")


def test_candidate_can_be_constructed_from_slde_profile():
    estimator = StudentLearningDynamicsEstimator()
    estimator.update(LearningObservation("S1", "algebra", "retrieval", 1, 0.2, 0.4, breadth=0.8, depth=0.6))
    estimator.update(LearningObservation("S1", "algebra", "retrieval", 1, 0.2, 0.5, breadth=0.6, depth=0.8))
    estimate = estimator.estimate("S1", "algebra", "retrieval")
    candidate = candidate_from_slde(estimate, execution_cost=0.3)
    assert candidate.student_id == "S1"
    assert candidate.expected_gain == pytest.approx(estimate.learning_rate)
    assert candidate.uncertainty == pytest.approx(estimate.learning_rate_standard_error)
    assert candidate.breadth_fit == pytest.approx(0.7)
    assert candidate.depth_fit == pytest.approx(0.7)
    assert candidate.execution_cost == pytest.approx(0.3)


def test_softmax_is_numerically_stable_for_large_scores():
    task = TaskRequest("q", "logic")
    candidates = [
        TaskCandidate("S1", expected_gain=1e6),
        TaskCandidate("S2", expected_gain=1e6 - 1),
    ]
    result = route_task(task, candidates, temperature=0.1)
    assert all(math.isfinite(p) for _, p in result.probabilities)
    assert sum(p for _, p in result.probabilities) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"temperature": -1},
        {"exploration_floor": -0.1},
        {"exploration_floor": 0.5},
        {"coalition_size": 0},
        {"coalition_size": 3},
    ],
)
def test_invalid_routing_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        route_task(
            TaskRequest("q", "domain"),
            [TaskCandidate("S1", 0.1), TaskCandidate("S2", 0.2)],
            **kwargs,
        )


def test_duplicate_candidate_ids_are_rejected():
    with pytest.raises(ValueError):
        route_task(
            TaskRequest("q", "domain"),
            [TaskCandidate("S1", 0.1), TaskCandidate("S1", 0.2)],
        )


def test_candidate_metrics_and_cost_are_validated():
    with pytest.raises(ValueError):
        TaskCandidate("S1", 0.1, breadth_fit=1.1)
    with pytest.raises(ValueError):
        TaskCandidate("S1", 0.1, uncertainty=-0.1)
    with pytest.raises(ValueError):
        TaskCandidate("S1", 0.1, execution_cost=-0.1)


def test_task_requirements_are_validated():
    with pytest.raises(ValueError):
        TaskRequest("q", "domain", breadth_requirement=1.1)
