import math

import pytest

from gds_core.resource_allocation import allocate_resources


def test_quadratic_allocator_finds_known_interior_solution():
    result = allocate_resources(
        ["S1", "S2"],
        [1.0, 1.0],
        [[1.0, 0.0], [0.0, 4.0]],
        1.0,
        risk_aversion=1.0,
        regularization=0.5,
        tolerance=1e-11,
    )
    allocation = dict(result.allocations)
    assert result.converged
    assert allocation["S1"] == pytest.approx(2.0 / 3.0, abs=1e-7)
    assert allocation["S2"] == pytest.approx(2.0 / 9.0, abs=1e-7)
    assert result.budget_used == pytest.approx(8.0 / 9.0, abs=1e-7)
    assert result.budget_unused == pytest.approx(1.0 / 9.0, abs=1e-7)


def test_budget_and_share_constraints_are_enforced():
    result = allocate_resources(
        ["S1", "S2", "S3"],
        [10.0, 2.0, 1.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        10.0,
        risk_aversion=0.2,
        regularization=0.1,
        min_share=0.05,
        max_share=0.4,
    )
    values = dict(result.allocations)
    assert sum(values.values()) <= 10.0 + 1e-8
    assert all(0.5 - 1e-8 <= value <= 4.0 + 1e-8 for value in values.values())
    assert result.converged


def test_correlated_covariance_changes_risk_penalty():
    diagonal = allocate_resources(
        ["S1", "S2"], [1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], 1.0,
        risk_aversion=2.0, regularization=0.1,
    )
    correlated = allocate_resources(
        ["S1", "S2"], [1.0, 1.0], [[1.0, 0.8], [0.8, 1.0]], 1.0,
        risk_aversion=2.0, regularization=0.1,
    )
    assert correlated.covariance_penalty > diagonal.covariance_penalty
    assert correlated.objective_value < diagonal.objective_value


def test_linear_program_case_allocates_to_highest_positive_gain():
    result = allocate_resources(
        ["S1", "S2", "S3"],
        [1.0, 3.0, -1.0],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        5.0,
        risk_aversion=0.0,
        regularization=0.0,
        max_share=0.6,
    )
    values = dict(result.allocations)
    assert values == pytest.approx({"S1": 2.0, "S2": 3.0, "S3": 0.0})
    assert result.converged


def test_allocation_is_deterministic_for_tied_linear_objective():
    result = allocate_resources(
        ["S2", "S1"], [1.0, 1.0], [[0.0, 0.0], [0.0, 0.0]], 1.0,
        risk_aversion=0.0, regularization=0.0,
    )
    assert dict(result.allocations) == pytest.approx({"S2": 0.0, "S1": 1.0})


def test_non_psd_covariance_is_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        allocate_resources(["S1", "S2"], [1, 1], [[1, 2], [2, 1]], 1.0)


def test_asymmetric_covariance_is_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        allocate_resources(["S1", "S2"], [1, 1], [[1, 0.1], [0.2, 1]], 1.0)


def test_covariance_must_have_correct_dimensions():
    with pytest.raises(ValueError, match="matrix"):
        allocate_resources(["S1", "S2"], [1, 1], [[1.0]], 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_budget": 0.0},
        {"risk_aversion": -1.0},
        {"regularization": -1.0},
        {"min_share": 0.6, "max_share": 1.0},
        {"min_share": 0.0, "max_share": -0.1},
        {"tolerance": 0.0},
        {"max_iterations": 0},
    ],
)
def test_invalid_problem_parameters_are_rejected(kwargs):
    parameters = {"total_budget": 1.0}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        allocate_resources(
            ["S1", "S2"], [1.0, 0.5], [[1.0, 0.0], [0.0, 1.0]],
            **parameters,
        )


def test_minimum_shares_cannot_exceed_budget():
    with pytest.raises(ValueError, match="exceed"):
        allocate_resources(
            ["S1", "S2", "S3"], [1, 1, 1],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            1.0, min_share=0.34,
        )


def test_identifiers_and_vector_lengths_are_validated():
    with pytest.raises(ValueError, match="unique"):
        allocate_resources(["S1", "S1"], [1, 1], [[1, 0], [0, 1]], 1.0)
    with pytest.raises(ValueError, match="exactly"):
        allocate_resources(["S1", "S2"], [1], [[1, 0], [0, 1]], 1.0)


def test_result_reports_objective_components_and_diagnostics():
    result = allocate_resources(
        ["S1", "S2"], [2.0, 1.0], [[1.0, 0.2], [0.2, 1.0]], 3.0,
        risk_aversion=0.5, regularization=0.1,
    )
    assert result.objective_value == pytest.approx(
        result.expected_gain - result.covariance_penalty - result.regularization_penalty
    )
    assert result.budget_used + result.budget_unused == pytest.approx(3.0)
    assert result.iterations >= 1
    assert result.projected_gradient_norm >= 0.0
    assert all(math.isfinite(value) for _, value in result.allocations)


def test_iteration_limit_is_reported_not_silently_claimed_as_convergence():
    result = allocate_resources(
        ["S1", "S2"], [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 1.0,
        risk_aversion=1.0, regularization=0.1, max_iterations=1,
        tolerance=1e-15,
    )
    assert result.iterations == 1
    assert not result.converged
