"""Structural topology for the six-agent Gainful Distillation System.

Node order is (T, S1, S2, S3, S4, S5). The student graph is K5 plus a
separate circumferential overlay. Perimeter student pairs therefore have
multiplicity two; student diagonals have multiplicity one.
"""
from itertools import combinations

NODE_LABELS = ("T", "S1", "S2", "S3", "S4", "S5")
TEACHER_INDEX = 0
STUDENT_INDICES = (1, 2, 3, 4, 5)
STUDENT_LOCAL_INDICES = (0, 1, 2, 3, 4)


def _is_perimeter_pair(i: int, j: int) -> bool:
    distance = (j - i) % 5
    return distance in (1, 4)


def build_adjacency_matrix() -> tuple[tuple[int, ...], ...]:
    rows = [[0 for _ in NODE_LABELS] for _ in NODE_LABELS]
    for student in STUDENT_INDICES:
        rows[0][student] = 1
        rows[student][0] = 1
    for i, j in combinations(STUDENT_LOCAL_INDICES, 2):
        multiplicity = 1 + int(_is_perimeter_pair(i, j))
        source, target = STUDENT_INDICES[i], STUDENT_INDICES[j]
        rows[source][target] = multiplicity
        rows[target][source] = multiplicity
    return tuple(tuple(row) for row in rows)


ADJACENCY_MATRIX = build_adjacency_matrix()


def student_edge_multiplicity(i: int, j: int) -> int:
    if i not in STUDENT_LOCAL_INDICES or j not in STUDENT_LOCAL_INDICES:
        raise IndexError("student indices must be integers from 0 through 4")
    if i == j:
        return 0
    return 1 + int(_is_perimeter_pair(i, j))


def weighted_degree(node_index: int) -> int:
    if not 0 <= node_index < len(NODE_LABELS):
        raise IndexError("node index must be between 0 and 5")
    return sum(ADJACENCY_MATRIX[node_index])


def validate_topology() -> None:
    matrix = ADJACENCY_MATRIX
    assert len(matrix) == 6
    assert all(len(row) == 6 for row in matrix)
    assert all(matrix[i][j] == matrix[j][i] for i in range(6) for j in range(6))
    assert all(matrix[i][i] == 0 for i in range(6))
    assert sum(matrix[0]) == 5
    assert all(weighted_degree(i) == 7 for i in STUDENT_INDICES)
    assert sum(sum(row) for row in matrix) == 40
    assert sum(1 for i, j in combinations(range(6), 2) if matrix[i][j] > 0) == 15
    assert sum(1 for i, j in combinations(STUDENT_INDICES, 2) if matrix[i][j] == 2) == 5
    assert sum(1 for i, j in combinations(STUDENT_INDICES, 2) if matrix[i][j] == 1) == 5


validate_topology()
