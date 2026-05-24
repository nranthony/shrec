"""Hirata–Nomura recurrence-manifold baseline.

Recurrence-manifold reconstruction (Hirata et al. 2008) using the
common-neighbour-ratio consensus similarity matrix of Nomura et al.
(2022), embedded with Isomap.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import Isomap

from shrec.models.base import RecurrenceModel
from shrec.utils import common_neighbors_ratio, nan_fill


class HirataNomuraIsomap(RecurrenceModel):
    """HN-Isomap baseline for continuous driver reconstruction."""

    def __init__(self, n_components=2, percentile=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.percentile = percentile

    def fit(self, X, y=None):
        X = self._preprocess(X)
        X = self._make_embedding(X)

        amat = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            dmat = cdist(X[i], X[i])
            thresh = np.percentile(dmat, self.percentile)
            amat += (dmat <= thresh).astype(int)
        amat[amat > 0] = 1

        wmat = common_neighbors_ratio(amat)
        if self.store_adjacency_matrix:
            self.adjacency_matrix = wmat

        iso = Isomap(n_components=self.n_components, metric='precomputed')
        pt_vals = iso.fit_transform(wmat)

        self.indices = np.arange(len(pt_vals))
        self.labels_ = nan_fill(pt_vals)
        return self

    def transform(self, X):
        X = self._preprocess(X)
        X = self._make_embedding(X)
        return common_neighbors_ratio(X)
