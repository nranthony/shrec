"""Regression tests for the exp-of-distance recurrence kernel.

Covers the `distance_to_connectivity` bracket fragility (same failure class as
the `fit_rho_sigma` σ-solve stall fixed earlier): `root_scalar` assumed a sign
change on a fixed bracket `[1e-16, dscale]` that does not exist for every
distance distribution.
"""
import warnings

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from shrec.recurrence.kernel import distance_to_connectivity


class TestDistanceToConnectivityBracket:
    """Pre-fix: small distance matrices raised
    `ValueError: f(a) and f(b) must have different signs` at the default
    sparsity, because the achievable mean-affinity floor (the diagonal alone
    forces mean ≥ 1/N) sits above the requested target, so no root is bracketed.
    """

    @pytest.mark.parametrize("n", [6, 8, 15, 40])
    def test_small_matrix_does_not_raise(self, n):
        rng = np.random.default_rng(n)
        # cdist(X, X) has a zero diagonal -> mean-affinity floor 1/N, which is
        # above the default target for these sizes. Pre-fix this raised
        # ValueError from the unbracketed root_scalar.
        X = rng.standard_normal((n, 2))
        dmat = cdist(X, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = distance_to_connectivity(dmat)
        assert out.shape == dmat.shape
        assert np.all(np.isfinite(out))

    def test_infeasible_sparsity_warns(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 2))
        dmat = cdist(X, X)  # zero diagonal -> floor 1/8 > 1 - 0.99
        with pytest.warns(UserWarning, match="below the achievable floor"):
            distance_to_connectivity(dmat, sparsity=0.99)

    def test_feasible_sparsity_hits_target_mean_affinity(self):
        # Large enough that the 1/N floor is well below the target; the solver
        # must land on the requested mean mass (1 - sparsity).
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 2))
        dmat = cdist(X, X)
        sparsity = 0.9
        out = distance_to_connectivity(dmat, sparsity=sparsity)
        assert out.mean() == pytest.approx(1 - sparsity, abs=1e-3)

    def test_dscale_path_is_unchanged(self):
        dmat = cdist(np.arange(10)[:, None], np.arange(10)[:, None]).astype(float)
        np.testing.assert_allclose(
            distance_to_connectivity(dmat, dscale=2.0), np.exp(-dmat / 2.0),
        )
