"""Back-compat shim: the original 900-line monolith is now split across
`base.py`, `classical.py`, `recurrence_clustering.py`, `recurrence_manifold.py`,
and `hirata_nomura.py`. Imports against `shrec.models.models` keep working
via the re-exports below.
"""
from shrec.embeddings import embed_ts  # noqa: F401
from shrec.graph import DisjointSet, _leiden, solve_union_find  # noqa: F401
from shrec.models.base import RecurrenceModel  # noqa: F401
from shrec.models.classical import ClassicalRecurrenceClustering  # noqa: F401
from shrec.models.hirata_nomura import HirataNomuraIsomap  # noqa: F401
from shrec.models.recurrence_clustering import RecurrenceClustering  # noqa: F401
from shrec.models.recurrence_manifold import RecurrenceManifold  # noqa: F401
from shrec.recurrence import (  # noqa: F401
    data_to_connectivity,
    data_to_connectivity2,
    dataset_to_simplex,
    distance_to_connectivity,
    relu,
)
