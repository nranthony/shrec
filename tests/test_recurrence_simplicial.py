"""Math-correctness tests for the fuzzy simplicial complex.

Tied to docs/tests-math.md §5b.1 — currently only **MM5** (parity between the
two in-repo simplicial implementations) is in scope; this is one of the
four "must" tests called out for the canonical-algorithm PR (§6 step 2).
"""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from shrec.models.models import dataset_to_simplex, relu


# --- §5b.1 MM5 — cross-implementation parity --------------------------------

class TestSimplicialParity:
    """The repo carries two fuzzy-simplicial implementations:

      * `models.dataset_to_simplex` — fsolve on each row, applied densely
        (paper Appendix B, eq. for σ_i).
      * `models.data_to_connectivity2` — delegates the ρ/σ root-solve to
        `umap.umap_.fuzzy_simplicial_set`, then applies those parameters
        densely (so the only difference *should* be the solver).

    Per §3 of the assessment, only one of these can be canonical. If they
    disagree on the same input, one is wrong (or differs only on
    parameter-choice conventions worth documenting).
    """

    @pytest.mark.xfail(
        reason=(
            "Confirmed divergence: the two simplicial implementations "
            "carry different conventions for the σ root-solve (likely "
            "the k-vs-(k+1) neighbour count and/or the ρ-as-min-distance "
            "rule). The §6 step-2 PR chose `dataset_to_simplex` (the "
            "self-contained, paper-faithful version) and dropped the "
            "umap-learn dependency from the canonical pipeline. This "
            "test remains as a historical record of the comparison; "
            "lifting xfail requires reconciling the two conventions."
        ),
        strict=True,
    )
    def test_dataset_to_simplex_matches_umap_sigmas_and_rhos(self, rng):
        """Apply the umap-derived (ρ, σ) the same way `data_to_connectivity2`
        does (dense `exp(-ReLU(d − ρ)/σ)` + fuzzy union), and compare to
        the in-repo `dataset_to_simplex`. If both implement Appendix B
        faithfully, they should agree to ~1e-5 on a random input.
        """
        umap_mod = pytest.importorskip("umap.umap_")
        fuzzy_simplicial_set = umap_mod.fuzzy_simplicial_set

        X = rng.standard_normal((50, 3))
        k = 10

        ours = dataset_to_simplex(X, k=k)

        # Mirror data_to_connectivity2:271-279 exactly.
        _, sigmas, rhos, _ = fuzzy_simplicial_set(
            X, k, 0, "euclidean", return_dists=True,
        )
        sigmas = np.asarray(sigmas)
        rhos = np.asarray(rhos)
        dmat = cdist(X, X)
        umap_aff = np.exp(-relu(dmat - rhos[None, :]) / sigmas[None, :])
        umap_aff = umap_aff + umap_aff.T - umap_aff * umap_aff.T

        # Expected to FAIL today on at least one of: σ-solver convention
        # (log₂ k vs ln k), the k-vs-(k-1) neighbour count, or ρ choice
        # (smallest vs smallest non-zero). Whatever the diff is, naming
        # it concretely is the goal of this test.
        np.testing.assert_allclose(ours, umap_aff, atol=1e-5)
