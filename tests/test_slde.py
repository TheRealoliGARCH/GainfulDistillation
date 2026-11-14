import math
import pytest
from gds_core.slde import LearningObservation, StudentLearningDynamicsEstimator

def observation(*, student="S1", domain="algebra", method="worked_examples", dose=1.0, before=0.4, after=0.6, retention=None, transfer=None, breadth=None, depth=None):
    return LearningObservation(student, domain, method, dose, before, after, retention, transfer, breadth, depth)

def test_first_rls_update_moves_estimate_toward_observed_response():
    estimator = StudentLearningDynamicsEstimator(prior_mean=0.0, prior_covariance=1000.0)
    result = estimator.update(observation(dose=2.0, before=0.2, after=0.6))
    assert result.observations == 1
    assert result.learning_rate == pytest.approx(0.2, rel=1e-3)
    assert result.rls_covariance > 0.0

def test_repeated_observations_recover_synthetic_learning_response():
    estimator = StudentLearningDynamicsEstimator(prior_mean=0.0, prior_covariance=100.0)
    for dose in (0.5, 1.0, 1.5, 2.0, 0.75, 1.25):
        estimator.update(observation(dose=dose, before=0.3, after=0.3 + 0.12 * dose))
    result = estimator.estimate("S1", "algebra", "worked_examples")
    assert result.learning_rate == pytest.approx(0.12, abs=2e-4)
    assert result.observations == 6

def test_profiles_are_separated_by_student_domain_and_method():
    estimator = StudentLearningDynamicsEstimator()
    estimator.update(observation(student="S1"))
    estimator.update(observation(student="S2"))
    estimator.update(observation(method="retrieval_practice"))
    assert estimator.profile_count() == 3
    assert estimator.estimate("S1", "algebra", "worked_examples").observations == 1
    assert estimator.estimate("S2", "algebra", "worked_examples").observations == 1
    assert estimator.estimate("S1", "algebra", "retrieval_practice").observations == 1

def test_optional_metrics_are_tracked_independently():
    estimator = StudentLearningDynamicsEstimator()
    estimator.update(observation(retention=0.8, transfer=0.7, breadth=0.6, depth=0.9))
    estimator.update(observation(retention=0.6, breadth=0.8))
    result = estimator.estimate("S1", "algebra", "worked_examples")
    assert result.retention.count == 2
    assert result.retention.mean == pytest.approx(0.7)
    assert result.transfer.count == 1
    assert result.transfer.mean == pytest.approx(0.7)
    assert result.breadth.count == 2
    assert result.depth.count == 1

def test_missing_metrics_are_not_imputed_as_zero():
    result = StudentLearningDynamicsEstimator().update(observation())
    assert result.retention is None and result.transfer is None
    assert result.breadth is None and result.depth is None

def test_metric_sample_variance_and_standard_error():
    estimator = StudentLearningDynamicsEstimator()
    estimator.update(observation(retention=0.2))
    result = estimator.update(observation(retention=0.8))
    assert result.retention.sample_variance == pytest.approx(0.18)
    assert result.retention.standard_error == pytest.approx(0.3)

def test_uncertainty_not_reported_with_only_one_residual():
    result = StudentLearningDynamicsEstimator().update(observation())
    assert result.residual_variance is None
    assert result.learning_rate_standard_error is None

def test_learning_rate_uncertainty_reported_after_multiple_residuals():
    estimator = StudentLearningDynamicsEstimator()
    estimator.update(observation(before=0.2, after=0.3))
    result = estimator.update(observation(before=0.2, after=0.4))
    assert result.residual_variance is not None and result.residual_variance >= 0.0
    assert result.learning_rate_standard_error is not None
    assert math.isfinite(result.learning_rate_standard_error)

def test_unknown_profile_returns_none():
    assert StudentLearningDynamicsEstimator().estimate("S5", "physics", "project") is None

@pytest.mark.parametrize("dose", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_instructional_dose_is_rejected(dose):
    with pytest.raises(ValueError):
        observation(dose=dose)

@pytest.mark.parametrize("score", [-0.01, 1.01, float("inf"), float("nan")])
def test_invalid_performance_scores_are_rejected(score):
    with pytest.raises(ValueError):
        observation(after=score)

@pytest.mark.parametrize("metric", ["retention", "transfer", "breadth", "depth"])
def test_optional_metrics_must_be_bounded_scores(metric):
    with pytest.raises(ValueError):
        observation(**{metric: 1.2})

def test_invalid_estimator_hyperparameters_are_rejected():
    with pytest.raises(ValueError):
        StudentLearningDynamicsEstimator(prior_covariance=0)
    with pytest.raises(ValueError):
        StudentLearningDynamicsEstimator(forgetting_factor=0)
    with pytest.raises(ValueError):
        StudentLearningDynamicsEstimator(forgetting_factor=1.01)

def test_forgetting_factor_discounts_information_and_inflates_covariance():
    ordinary = StudentLearningDynamicsEstimator(forgetting_factor=1.0)
    adaptive = StudentLearningDynamicsEstimator(forgetting_factor=0.95)
    for _ in range(3):
        ordinary.update(observation())
        adaptive.update(observation())
    a = ordinary.estimate("S1", "algebra", "worked_examples")
    b = adaptive.estimate("S1", "algebra", "worked_examples")
    assert b.rls_covariance > a.rls_covariance

def test_observation_identifiers_must_be_nonempty():
    with pytest.raises(ValueError):
        observation(student="   ")
