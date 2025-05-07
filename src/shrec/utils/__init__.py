# shrec/utils/__init__.py

from .io import (
    get_time,
    curr_time,
    load_pickle_file,
)

from .preprocessing import (
    nan_pca,
    matrix_lowrank,
    zero_topk,
    nan_fill,
    transform_stationary,
    detrend_ts,
    unroll_phase,
    lift_ts,
    standardize_ts,
    minmax_ts,
    embed_ts,
    hankel_matrix,
    _hankel_matrix,
    allclose_len,
    array2d_to_list,
    dict_to_vals,
    spherize_ts,
    whiten_zca,
)

from .signal import (
    discretize_signal,
    discretize_ts,
    find_psd,
    group_consecutives,
    find_characteristic_timescale,
)

from .graph_tools import (
    common_neighbors_ratio,
    community_list_to_labels,
    neighbors_to_mutual,
    arg_find,
    hollow_matrix,
    largest_connected_component,
    susceptibility_smallcomponents,
    susceptibility_subleading,
    sort_graph,
    multigraph_to_weighted,
    graph_from_associations,
    adjmat_from_associations,
    graph_threshold,
    adjmat,
    merge_labels,
    compress_adjacency,
)

from .surrogates import (
    make_surrogate,
    array_select,
)

from .metrics import (
    sparsity,
    otsu_threshold,
    sparsify,
    outlier_detection_pca,
    evaluate_clustering,
)

from .transforms import RigidTransform



# from .utils import (
#     # Time and file utilities
#     get_time,
#     curr_time,
#     load_pickle_file,

#     # Array/matrix preprocessing
#     nan_pca,
#     matrix_lowrank,
#     zero_topk,
#     nan_fill,
#     transform_stationary,
#     detrend_ts,
#     unroll_phase,
#     lift_ts,
#     standardize_ts,
#     minmax_ts,
#     embed_ts,
#     hankel_matrix,
#     _hankel_matrix,
#     hollow_matrix,
#     allclose_len,
#     array2d_to_list,
#     dict_to_vals,
#     spherize_ts,
#     whiten_zca,

#     # Discretization & signal processing
#     discretize_signal,
#     discretize_ts,
#     find_psd,
#     group_consecutives,
#     find_characteristic_timescale,

#     # Distance and graph utilities
#     common_neighbors_ratio,
#     neighbors_to_mutual,
#     arg_find,
#     distance_to_connectivity,

#     # Graph utilities
#     largest_connected_component,
#     susceptibility_smallcomponents,
#     susceptibility_subleading,
#     sort_graph,
#     multigraph_to_weighted,
#     graph_from_associations,
#     adjmat_from_associations,
#     graph_threshold,
#     adjmat,
#     merge_labels,
#     compress_adjacency,

#     # Surrogate & selection
#     make_surrogate,
#     array_select,

#     # Clustering & metrics
#     sparsity,
#     otsu_threshold,
#     sparsify,
#     outlier_detection_pca,
#     evaluate_clustering,

#     # Transformation
#     RigidTransform,
# )
