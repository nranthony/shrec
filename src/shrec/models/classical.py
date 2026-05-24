"""Classical Sauer-style recurrence clustering baseline (PRL 2004)."""
import numpy as np

from shrec.graph import solve_union_find
from shrec.models.base import RecurrenceModel
from shrec.recurrence import data_to_connectivity
from shrec.utils import allclose_len, sparsify


class ClassicalRecurrenceClustering(RecurrenceModel):
    """Cluster a time series using Sauer (PRL 2004) union-find equivalence
    classes on a binarised recurrence matrix.

    This is the noise-free Sauer-limit baseline. For the canonical SHREC
    pipeline (adaptive simplicial complex + Leiden), use
    `RecurrenceClustering` / `RecurrenceManifold` instead.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        """Cluster the input time series and populate ``self.labels_``."""
        X = self._preprocess(X)
        X = self._make_embedding(X)

        dist_mat_bin = data_to_connectivity(
            X, time_exclude=0, ord=500., scale=self.scale,
        )
        dist_mat_bin = sparsify(
            dist_mat_bin, (1 - self.tolerance),
            weighted=self.weighted_connectivity,
        )

        if self.store_adjacency_matrix:
            self.adjacency_matrix = dist_mat_bin

        all_merged_inds = [np.sort(np.where(row)[0]) for row in dist_mat_bin]

        merged_inds = solve_union_find([list(item) for item in all_merged_inds])
        merged_inds = [np.sort(np.array(item)) for item in merged_inds]

        known_items = []
        item_labels = []
        for item in merged_inds:
            add_flag = False
            for j, known_item in enumerate(known_items):
                if allclose_len(item, known_item):
                    item_labels.append(j)
                    add_flag = True
            if not add_flag:
                known_items.append(item)
                item_labels.append(1)
        item_labels = np.array(item_labels)

        reference_indices = np.arange(dist_mat_bin.shape[0])
        self.indices = np.copy(reference_indices)
        self.labels_ = item_labels
        return self
