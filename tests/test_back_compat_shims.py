"""Back-compat import surface and preset signatures (decision 0011 in DECISIONS.md).

The in-tree notebooks and 0.1.x callers import through `shrec.models.models` and
`shrec.utils`, and construct the presets by scalar keyword. Dropping a name or
renaming a keyword here is a release-boundary change, not cleanup.
"""
import importlib
import inspect

import pytest

from shrec import (
    ClassicalRecurrenceClustering,
    HirataNomuraIsomap,
    RecurrenceClustering,
    RecurrenceManifold,
)

MODELS_SHIM = (
    "embed_ts", "DisjointSet", "_leiden", "solve_union_find", "RecurrenceModel",
    "ClassicalRecurrenceClustering", "HirataNomuraIsomap", "RecurrenceClustering",
    "RecurrenceManifold", "data_to_connectivity", "data_to_connectivity2",
    "dataset_to_simplex", "distance_to_connectivity", "relu",
)

UTILS_SHIM = (
    "get_time", "curr_time", "load_pickle_file",
    "nan_pca", "matrix_lowrank", "zero_topk", "nan_fill", "transform_stationary",
    "detrend_ts", "unroll_phase", "lift_ts", "standardize_ts", "minmax_ts",
    "embed_ts", "hankel_matrix", "_hankel_matrix", "allclose_len",
    "array2d_to_list", "dict_to_vals", "spherize_ts", "whiten_zca",
    "discretize_signal", "discretize_ts", "find_psd", "group_consecutives",
    "find_characteristic_timescale",
    "common_neighbors_ratio", "community_list_to_labels", "neighbors_to_mutual",
    "arg_find", "hollow_matrix", "largest_connected_component",
    "susceptibility_smallcomponents", "susceptibility_subleading", "sort_graph",
    "multigraph_to_weighted", "graph_from_associations", "adjmat_from_associations",
    "graph_threshold", "adjmat", "merge_labels", "compress_adjacency",
    "make_surrogate", "array_select",
    "sparsity", "otsu_threshold", "sparsify", "outlier_detection_pca",
    "evaluate_clustering",
    "RigidTransform",
)

PRESET_KNOBS = {
    RecurrenceClustering: {"resolution": 1.0, "objective": "modularity",
                           "method": "graspologic", "k": 10, "tol": 1e-5,
                           "aggregation": "mean"},
    RecurrenceManifold: {"n_components": 1, "normalize_laplacian": False, "k": 10,
                         "tol": 1e-5, "aggregation": "mean"},
    ClassicalRecurrenceClustering: {"scale": 1.0, "order": 500.0},
    HirataNomuraIsomap: {"n_components": 2, "percentile": 0.1},
}


@pytest.mark.parametrize(
    "module, names", [("shrec.models.models", MODELS_SHIM), ("shrec.utils", UTILS_SHIM)]
)
def test_shim_still_exports_every_name(module, names):
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert missing == [], f"{module} no longer exports {missing}"


@pytest.mark.parametrize("cls", list(PRESET_KNOBS), ids=lambda c: c.__name__)
def test_preset_keeps_its_scalar_keywords_and_defaults(cls):
    params = inspect.signature(cls.__init__).parameters
    for name, default in PRESET_KNOBS[cls].items():
        assert name in params, f"{cls.__name__} lost keyword {name!r}"
        assert params[name].default == default, f"{cls.__name__}.{name} default changed"
