"""Tests for graph-adjacency utilities (docs/tests-math.md §5b.1 MM12).

`common_neighbors_ratio` was rewritten from a Python double-loop into a
vectorised boolean-matmul (audit §3D). MM12 pins parity with the
previous loop implementation on random binary inputs.
"""
import numpy as np
import pytest

from shrec.utils import common_neighbors_ratio


def _common_neighbors_ratio_loop(adj_matrix):
    """The original O(T² · k) Python implementation, kept here as the
    reference oracle for MM12 only."""
    weighted = np.zeros_like(adj_matrix, dtype=float)
    n = adj_matrix.shape[0]
    neighbour_lists = [np.where(adj_matrix[i] > 0)[0] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            overlap = np.intersect1d(neighbour_lists[i], neighbour_lists[j])
            union = np.union1d(neighbour_lists[i], neighbour_lists[j])
            if len(union) > 0:
                weighted[i, j] = 1 - len(overlap) / len(union)
    weighted = weighted + weighted.T
    np.fill_diagonal(weighted, 0)
    return weighted


# --- §5b.1 MM12 — vectorised ↔ loop parity ---------------------------------

class TestCommonNeighborsRatioParity:

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_loop_on_random_binary(self, seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(5, 20)
        density = rng.uniform(0.2, 0.6)
        A = (rng.random((n, n)) < density).astype(float)
        A = ((A + A.T) > 0).astype(float)  # symmetric binary
        np.fill_diagonal(A, 0)

        loop_out = _common_neighbors_ratio_loop(A)
        vec_out = common_neighbors_ratio(A)
        np.testing.assert_allclose(vec_out, loop_out, atol=1e-12)

    def test_complete_graph_with_self_loops_returns_zero(self):
        # K_n with self-loops: every node's neighbour set is the full
        # vertex set, so every pair has Jaccard = 1 → output = 0
        # everywhere (including off-diagonal). Without self-loops, K_n
        # gives 1/(n-1) off-diagonal — the closed form is correct but
        # less informative as an oracle.
        n = 6
        A = np.ones((n, n))
        out = common_neighbors_ratio(A)
        np.testing.assert_allclose(out, np.zeros((n, n)), atol=1e-12)

    def test_disjoint_cliques_have_unit_inter_cluster_value(self):
        # Two disjoint cliques share zero neighbours; output = 1 across
        # the inter-cluster block.
        a, b = 4, 5
        A = np.zeros((a + b, a + b))
        A[:a, :a] = 1 - np.eye(a)
        A[a:, a:] = 1 - np.eye(b)
        out = common_neighbors_ratio(A)
        # Inter-cluster block should be all ones.
        np.testing.assert_allclose(out[:a, a:], np.ones((a, b)), atol=1e-12)
        np.testing.assert_allclose(out[a:, :a], np.ones((b, a)), atol=1e-12)
        # Diagonal still zero.
        assert np.all(np.diag(out) == 0)
