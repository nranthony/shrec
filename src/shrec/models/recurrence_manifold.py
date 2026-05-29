"""Continuous-driver SHREC model — Fiedler eigenvector of the consensus
recurrence Laplacian (paper Appendix B step 5)."""
import warnings
from datetime import datetime

import numpy as np
import scipy.linalg

from shrec.models.base import RecurrenceModel
from shrec.recurrence import data_to_connectivity2
from shrec.utils import nan_fill


class RecurrenceManifold(RecurrenceModel):
    """Reconstruct a continuous driver signal by spectral embedding of the
    consensus recurrence graph.

    Parameters
    ----------
    n_components : int
        Number of non-trivial Laplacian eigenvectors to return (the driver
        embedding dimension). The Fiedler vector is `n_components=1`.
    normalize_laplacian : bool
        If False (default), use the unnormalised Laplacian `L = D − A`
        (RatioCut), matching the paper. If True, use the NCut / random-walk
        normalisation (generalised eigenproblem `L v = λ D v`), which corrects
        for degree heterogeneity across responses ("response bias").
    """

    def __init__(self, n_components=1, normalize_laplacian=False, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.normalize_laplacian = normalize_laplacian

    def fit(self, X, y=None):
        """
        Args:
            X (array-like): shape (n_timepoints, n_channels).
            y: ignored, present for sklearn API parity.
        """
        X = self._preprocess(X)
        X = self._make_embedding(X)

        if self.verbose:
            print("Computing distance matrix... ", flush=True, end='')
        t1 = datetime.now()

        # Per paper Appendix B step 3+4: per-response fuzzy simplicial
        # complex with mean consensus aggregation.
        neighbor_matrix = data_to_connectivity2(X, time_exclude=self.time_exclude)

        if self.verbose:
            elapsed = datetime.now() - t1
            print(f"Done in {elapsed.total_seconds():.2f} seconds", flush=True)

        if self.verbose:
            print("Computing diffusion components... ", flush=True, end='')
        t1 = datetime.now()
        if self.store_adjacency_matrix:
            self.adjacency_matrix = neighbor_matrix

        # Continuous driver = Fiedler eigenvector of the graph Laplacian
        # L = D − A, per paper Appendix B step 5. The previous TruncatedSVD
        # path returned A's second right singular vector, which only
        # coincides with the Fiedler vector when A is regular (constant
        # row sums) — see MM22 oracle.
        affinity = np.asarray(neighbor_matrix)
        degree = affinity.sum(axis=1)
        laplacian = np.diag(degree) - affinity

        if self.normalize_laplacian:
            # NCut / random-walk normalisation via the generalised eigenproblem
            # L v = λ D v (Shi–Malik). Balancing by volume rather than node
            # count down-weights high-degree responses, so the Fiedler vector
            # tracks the shared driver instead of localising on the densest
            # channel — the "preconditioning the graph Laplacian" remedy the
            # paper names for response bias. Off by default to preserve the
            # paper-faithful unnormalised (RatioCut) behaviour. Generalised
            # eigenvalues live in [0, 2], so the connectivity scale is O(1).
            eigvals, eigvecs = scipy.linalg.eigh(
                laplacian, np.diag(degree), subset_by_index=[1, self.n_components],
            )
            connectivity_scale = 1.0
        else:
            eigvals, eigvecs = scipy.linalg.eigh(
                laplacian, subset_by_index=[1, self.n_components],
            )
            connectivity_scale = degree.sum()

        # Algebraic connectivity (λ₂) > 0 iff the graph is connected; a 0 has
        # multiplicity = number of connected components. When λ₂ ≈ 0 the
        # returned eigenvector lies in a degenerate near-null space and is a
        # component indicator, not a smooth driver coordinate. Threshold is
        # scaled to the spectrum (trace = Σ degree for L; O(1) for the
        # normalised generalised problem).
        if eigvals[0] <= 1e-10 * connectivity_scale:
            warnings.warn(
                "Consensus recurrence graph is (nearly) disconnected "
                f"(algebraic connectivity λ₂={eigvals[0]:.3e}); the Fiedler "
                "eigenvector may be a connected-component indicator rather "
                "than a smooth driver. Consider lowering `tolerance`/"
                "`time_exclude` or increasing the response ensemble size."
            )

        pt_vals = eigvecs.squeeze()

        if self.verbose:
            elapsed = datetime.now() - t1
            print(f"Done in {elapsed.total_seconds():.2f} seconds", flush=True)

        self.indices = np.arange(len(pt_vals))
        self.labels_ = nan_fill(pt_vals)
        return self
