"""Recurrence-graph construction primitives (paper Appendix B steps 2–4).

`simplicial.dataset_to_simplex` is the canonical adaptive-σ fuzzy simplicial
complex; `kernel.data_to_connectivity` is the simpler fixed-scale exp-of-
distance variant retained for the Sauer baseline; `consensus.data_to_connectivity2`
aggregates per-response simplicial complexes across an ensemble.
"""
from shrec.recurrence.simplicial import dataset_to_simplex, relu
from shrec.recurrence.kernel import data_to_connectivity, distance_to_connectivity
from shrec.recurrence.consensus import data_to_connectivity2

__all__ = [
    "dataset_to_simplex",
    "relu",
    "data_to_connectivity",
    "distance_to_connectivity",
    "data_to_connectivity2",
]
