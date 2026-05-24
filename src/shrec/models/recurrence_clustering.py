"""Discrete-driver SHREC model — Leiden community detection on the
consensus fuzzy simplicial complex (paper Appendix B steps 3–5)."""
import numpy as np

from shrec.graph import _leiden
from shrec.models.base import RecurrenceModel
from shrec.recurrence import data_to_connectivity2


class RecurrenceClustering(RecurrenceModel):
    """Assign a discrete cluster label to each timepoint based on community
    structure in the consensus recurrence graph. Best suited to discrete-
    time driver signals.
    """

    def __init__(
        self,
        resolution=1.0,
        objective="modularity",
        method="graspologic",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resolution = resolution
        self.objective = objective
        self.method = method

    def fit(self, X, y=None):
        """
        Args:
            X (array-like): shape (n_timepoints, n_channels).
            y: ignored, present for sklearn API parity.
        """
        X = self._preprocess(X)
        X = self._make_embedding(X)

        # Per paper Appendix B step 3: per-response fuzzy simplicial
        # complex + step 4 consensus aggregation. The simplicial output
        # is already the canonical affinity — no further `sparsify` step,
        # since the row-adaptive σ_i provides the locality.
        neighbor_matrix = data_to_connectivity2(X, time_exclude=self.time_exclude)

        if self.store_adjacency_matrix:
            self.adjacency_matrix = neighbor_matrix

        indices, labels = _leiden(
            neighbor_matrix,
            resolution=self.resolution,
            objective=self.objective,
            method=self.method,
            random_state=self.random_state,
        )
        sort_inds = np.argsort(indices)
        indices, labels = indices[sort_inds], labels[sort_inds]
        reference_indices = np.arange(neighbor_matrix.shape[0])

        self.indices = np.copy(reference_indices)
        self.labels_ = -np.ones_like(self.indices)
        self.labels_[indices] = labels

        self.has_unclassified = np.any(self.labels_ < 0)
        self.n_clusters = len(np.unique(self.labels_)) - self.has_unclassified
        return self
