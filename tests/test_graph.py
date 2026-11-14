from itertools import combinations

import pytest

from gds_core.graph import (
    ADJACENCY_MATRIX,
    NODE_LABELS,
    STUDENT_INDICES,
    student_edge_multiplicity,
    validate_topology,
    weighted_degree,
)


def test_six_agent_topology_and_node_order():
    assert NODE_LABELS == ("T", "S1", "S2", "S3", "S4", "S5")
    assert len(ADJACENCY_MATRIX) == 6
    assert all(len(row) == 6 for row in ADJACENCY_MATRIX)
    assert ADJACENCY_MATRIX == (
        (0, 1, 1, 1, 1, 1),
        (1, 0, 2, 1, 1, 2),
        (1, 2, 0, 2, 1, 1),
        (1, 1, 2, 0, 2, 1),
        (1, 1, 1, 2, 0, 2),
        (1, 2, 1, 1, 2, 0),
    )
    validate_topology()


def test_teacher_has_five_edges_and_students_form_k5():
    assert sum(ADJACENCY_MATRIX[0]) == 5
    assert sum(1 for i, j in combinations(STUDENT_INDICES, 2)
               if ADJACENCY_MATRIX[i][j] > 0) == 10


def test_overlay_doubles_exactly_five_perimeter_pairs():
    doubled = [(i, j) for i, j in combinations(STUDENT_INDICES, 2)
               if ADJACENCY_MATRIX[i][j] == 2]
    single = [(i, j) for i, j in combinations(STUDENT_INDICES, 2)
              if ADJACENCY_MATRIX[i][j] == 1]
    assert len(doubled) == 5
    assert len(single) == 5
    assert student_edge_multiplicity(0, 4) == 2
    assert student_edge_multiplicity(0, 2) == 1


def test_weighted_degrees_and_total_edge_multiplicity():
    assert weighted_degree(0) == 5
    assert [weighted_degree(i) for i in STUDENT_INDICES] == [7] * 5
    assert sum(sum(row) for row in ADJACENCY_MATRIX) // 2 == 20


def test_student_edge_multiplicity_rejects_invalid_indices():
    with pytest.raises(IndexError):
        student_edge_multiplicity(0, 5)


def test_student_self_pair_has_zero_multiplicity():
    assert student_edge_multiplicity(3, 3) == 0
