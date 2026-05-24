"""Math-correctness tests for `RecurrenceManifold` (continuous driver).

Tied to docs/tests-math.md §5b.3 — **MM22** (block-diagonal Fiedler oracle)
is one of the four "must" tests driving the canonical-algorithm PR
(§6 step 2). It is expected to FAIL against the current implementation,
which uses `TruncatedSVD(A)` instead of the Fiedler eigenvector of
`L = D − A`. Locking in the fix flips this test green.
"""
import numpy as np
import pytest

from shrec.models.models import RecurrenceManifold


def _two_block_affinity(p1=10, p2=30, bridge=1e-3):
    """K=2 weakly-connected blocks of **unequal size**, with unnormalised
    all-ones blocks (rows in block_i sum to ≈ p_i, so the graph is
    *irregular*). A single bridge edge of weight `bridge` connects the
    two blocks.

    The irregularity matters: for a regular graph A and L = D − A share
    eigenvectors, so SVD and Fiedler agree. By using row sums that differ
    by block, the top-2 right singular vectors of A become the per-block
    indicators (pure block 1 / pure block 2), while the Fiedler vector of
    L is the *block-difference* (zero-mean) vector spanning both blocks.
    The two answers measurably diverge — `|corr| ≈ 0.866` vs `1.0` — so
    the test fails sharply under the current `TruncatedSVD` path and
    passes after the §6 step-2 fix.
    """
    N = p1 + p2
    A = np.zeros((N, N))
    A[:p1, :p1] = np.ones((p1, p1))
    A[p1:, p1:] = np.ones((p2, p2))
    A[p1 - 1, p1] = bridge
    A[p1, p1 - 1] = bridge
    return A


def _analytical_fiedler(p1=10, p2=30):
    """For two unequal blocks, the Fiedler vector is constant on each
    block with opposite signs, weighted so the vector is zero-mean and
    unit-norm. The small bridge perturbation does not change this to
    leading order.
    """
    N = p1 + p2
    v = np.empty(N)
    # a, b chosen so p1*a + p2*b = 0 (zero mean) and ‖v‖₂ = 1
    b = -1.0
    a = -p2 * b / p1
    v[:p1] = a
    v[p1:] = b
    return v / np.linalg.norm(v)


class TestFiedlerEigenvectorOracle:
    """§5b.3 MM22 — `RecurrenceManifold` must extract the Fiedler eigenvector
    of `L = D − A`, not the second right singular vector of `A`. On a
    block-structured affinity matrix the two answers diverge to ~|corr| ≈
    0.7 vs 1.0, so this test fails sharply under the current code and is
    the canonical regression guard once the fix lands.
    """

    def test_two_block_affinity_recovers_block_indicator(self, monkeypatch):
        p1, p2 = 10, 30
        A = _two_block_affinity(p1=p1, p2=p2, bridge=1e-3)
        fiedler = _analytical_fiedler(p1=p1, p2=p2)

        # Bypass the time-series preprocessing — feed the precomputed
        # affinity straight into the spectral step. After the §6 step-4
        # split, `RecurrenceManifold` imports `data_to_connectivity2`
        # from `shrec.recurrence` directly, so patch the binding in the
        # model's own module rather than the legacy `shrec.models.models`
        # shim.
        import shrec.models.recurrence_manifold as RM
        monkeypatch.setattr(RM, "data_to_connectivity2", lambda X, **kw: A)

        # `_make_embedding` is still called on the input, so pass an X with
        # the right shape (n_timepoints, n_responses); contents are irrelevant.
        dummy_X = np.zeros((A.shape[0], 1))

        model = RecurrenceManifold(n_components=1, standardize=False)
        model.fit(dummy_X)
        pred = np.asarray(model.labels_).squeeze()
        pred = pred / (np.linalg.norm(pred) + 1e-12)

        # Cosine similarity, not Pearson correlation: do NOT subtract the mean.
        # The fixed (Fiedler) output is zero-mean by construction; the current
        # (SVD) output is a one-sided block indicator with non-zero mean.
        # Subtracting the mean would collapse any block-piecewise-constant
        # vector onto the analytical Fiedler direction, masking the bug.
        cos = abs(float(np.dot(pred, fiedler)))
        # Expected today: |cos| ≈ 0.866 (TruncatedSVD's 2nd singular vector
        # is the indicator of the smaller block, not the block-difference
        # Fiedler vector). After the §6 step-2 fix this should be ≈ 1.
        assert cos > 0.99, (
            f"|cos(pred, analytical Fiedler)| = {cos:.4f}; "
            "RecurrenceManifold is not returning the Laplacian Fiedler vector."
        )
