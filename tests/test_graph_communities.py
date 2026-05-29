"""Regression tests for the Leiden community-detection adapter (`_leiden`).

Tied to docs/history/2026-05-refactor-assessment.md §3A bugs A4 (deprecated NetworkX API) and A8
(missing else branch on the method dispatch).
"""
import numpy as np
import pytest
import scipy.sparse as sp

from sklearn.metrics import adjusted_rand_score

from shrec.models.models import _leiden


def _barbell_adjacency(n=5):
    """Two K_n cliques joined by a single bridge edge.
    Output is a symmetric (2n, 2n) numpy array of 0/1.
    """
    N = 2 * n
    A = np.zeros((N, N), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = 1
                A[n + i, n + j] = 1
    # bridge
    A[n - 1, n] = 1
    A[n, n - 1] = 1
    return A


# --- §3A.A8 — _leiden missing else branch -----------------------------------

class TestLeidenUnknownMethod:
    """Pre-fix: an unknown `method=` silently fell through with `indices` /
    `labels` unbound, raising an obscure UnboundLocalError downstream. The
    fix raises a clear ValueError up front.
    """

    def test_unknown_method_raises_value_error(self):
        A = _barbell_adjacency(n=5)
        with pytest.raises(ValueError, match="Unknown community-detection method"):
            _leiden(A, method="not-a-real-method")

    def test_known_methods_dispatch_without_error(self):
        """At least one supported backend (graspologic) is a hard dependency
        per pyproject.toml, so it must work."""
        A = _barbell_adjacency(n=5)
        # graspologic requires random_seed to be a *positive* integer (or None)
        indices, labels = _leiden(A, method="graspologic", random_state=1)
        assert len(indices) == A.shape[0]
        assert len(labels) == A.shape[0]


# --- §3A.A4 — deprecated NetworkX API in the cdlib branch -------------------

class TestNetworkXModernAPI:
    """Pre-fix: `_leiden(... method='cdlib')` called
    `nx.convert_matrix.from_scipy_sparse_matrix` / `from_numpy_matrix`,
    which were removed in NetworkX 3.x and would raise AttributeError on
    any modern install. The fix uses the supported `from_*_array` shims.
    """

    def test_cdlib_dense_path_does_not_raise(self):
        cdlib = pytest.importorskip("cdlib")
        A = _barbell_adjacency(n=5).astype(float)
        # The pre-fix code raised AttributeError at the from_numpy_matrix call.
        indices, labels = _leiden(A, method="cdlib")
        assert len(indices) == A.shape[0]

    def test_cdlib_sparse_path_does_not_raise(self):
        cdlib = pytest.importorskip("cdlib")
        A = sp.csr_matrix(_barbell_adjacency(n=5).astype(float))
        indices, labels = _leiden(A, method="cdlib")
        assert len(indices) == A.shape[0]


# --- §5b.1 MM10 — Leiden backend agreement ----------------------------------

def _barbell_ground_truth(n=5):
    """The two K_n cliques are the two ground-truth communities."""
    return np.array([0] * n + [1] * n)


class TestLeidenBackendAgreement:
    """MM10 (must) — on a clean barbell (two cliques + one bridge) every
    Leiden backend must recover the same 2-community partition (ARI=1). The
    barbell has an unambiguous community structure, so backend-specific
    defaults (RNG, tie-breaking) should not change the answer.

    graspologic is a hard dependency so its leg always runs; igraph /
    leidenalg / cdlib are gated by `importorskip` and only run under the
    optional extra.
    """

    def test_graspologic_recovers_two_cliques(self):
        A = _barbell_adjacency(n=6)
        _, labels = _leiden(A, method="graspologic", random_state=1)
        assert adjusted_rand_score(_barbell_ground_truth(6), labels) == 1.0

    @pytest.mark.parametrize("method", ["igraph", "leidenalg", "cdlib"])
    def test_optional_backends_agree_with_graspologic(self, method):
        pytest.importorskip(method if method != "cdlib" else "cdlib")
        A = _barbell_adjacency(n=6)
        _, ref = _leiden(A, method="graspologic", random_state=1)
        _, other = _leiden(A.astype(float), method=method)
        assert adjusted_rand_score(ref, other) == 1.0


class TestIgraphResolutionForwarded:
    """Regression for the igraph branch hardcoding `resolution_parameter=1.0`
    and dropping the caller's `resolution`. With the bug present, sweeping
    `resolution` has no effect and the partition is identical at every value.
    After the fix, a high CPM resolution shatters the graph into more
    communities than a low one.
    """

    def test_resolution_changes_community_count(self):
        pytest.importorskip("igraph")
        A = _barbell_adjacency(n=6).astype(float)
        _, coarse = _leiden(A, method="igraph", objective="cpm", resolution=0.01)
        _, fine = _leiden(A, method="igraph", objective="cpm", resolution=0.99)
        assert len(np.unique(fine)) > len(np.unique(coarse))
