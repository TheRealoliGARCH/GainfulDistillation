import pytest

from gds_core import ADJACENCY_MATRIX, TransmissionWeights


def sample_weights():
    peer = [[0.0 if i == j else 1.0 for j in range(5)] for i in range(5)]
    return TransmissionWeights.from_channels(
        teacher_to_student=[0.1, 0.2, 0.3, 0.4, 0.5],
        student_to_teacher=[0.5, 0.4, 0.3, 0.2, 0.1],
        peer_to_student=peer,
    )


def test_directed_matrix_has_expected_shape_and_orientation():
    matrix = sample_weights().as_matrix()
    assert len(matrix) == 6
    assert all(len(row) == 6 for row in matrix)
    assert matrix[0][1] == pytest.approx(0.1)
    assert matrix[1][0] == pytest.approx(0.5)
    assert matrix[1][2] == pytest.approx(2.0)
    assert matrix[1][3] == pytest.approx(1.0)


def test_structural_and_directed_matrices_are_distinct():
    directed = sample_weights().as_matrix()
    assert ADJACENCY_MATRIX[0][1] == 1
    assert directed[0][1] == pytest.approx(0.1)
    assert directed[1][0] == pytest.approx(0.5)
    assert directed[0][1] != directed[1][0]


def test_peer_channels_can_be_asymmetric():
    peer = [[0.0 if i == j else 1.0 for j in range(5)] for i in range(5)]
    peer[0][1] = 0.25
    peer[1][0] = 0.75
    weights = TransmissionWeights.from_channels([1] * 5, [1] * 5, peer)
    matrix = weights.as_matrix()
    assert matrix[1][2] == pytest.approx(0.5)
    assert matrix[2][1] == pytest.approx(1.5)


@pytest.mark.parametrize("bad", [-0.1, float("inf"), float("nan")])
def test_invalid_transmission_strengths_are_rejected(bad):
    peer = [[0 if i == j else 1 for j in range(5)] for i in range(5)]
    with pytest.raises(ValueError):
        TransmissionWeights.from_channels([bad, 1, 1, 1, 1], [1] * 5, peer)


def test_peer_matrix_must_be_five_by_five():
    with pytest.raises(ValueError):
        TransmissionWeights.from_channels([1] * 5, [1] * 5, [[0] * 5] * 4)


def test_self_transmission_is_forced_to_zero():
    peer = [[9 if i == j else 1 for j in range(5)] for i in range(5)]
    weights = TransmissionWeights.from_channels([1] * 5, [1] * 5, peer)
    assert all(weights.as_matrix()[i + 1][i + 1] == 0 for i in range(5))
