"""Continuous-driver SHREC model — Fiedler eigenvector of the consensus
recurrence Laplacian (paper Appendix B step 5)."""
from datetime import datetime

import numpy as np
import scipy.linalg

from shrec.models.base import RecurrenceModel
from shrec.recurrence import data_to_connectivity2
from shrec.utils import nan_fill


class RecurrenceManifold(RecurrenceModel):
    """Reconstruct a continuous driver signal by spectral embedding of the
    consensus recurrence graph."""

    def __init__(self, n_components=1, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

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
        _, eigvecs = scipy.linalg.eigh(
            laplacian, subset_by_index=[1, self.n_components],
        )
        pt_vals = eigvecs.squeeze()

        if self.verbose:
            elapsed = datetime.now() - t1
            print(f"Done in {elapsed.total_seconds():.2f} seconds", flush=True)

        self.indices = np.arange(len(pt_vals))
        self.labels_ = nan_fill(pt_vals)
        return self
