"""Leiden community-detection adapter — multi-backend (paper Appendix B step 5).

Dispatches to graspologic-native (default; hard dep), igraph, leidenalg, or
cdlib if installed. Backends are imported lazily so the core library is
usable without optional packages.

The default backend is the Rust core that `graspologic.partition.leiden`
wraps, called directly (docs/decisions/0010). The method keeps the name
``"graspologic"`` so callers and saved configs need no change; the pure-Python
`graspologic` package is no longer required or imported.
"""
import warnings

import numpy as np
import scipy.sparse as sp

from shrec.utils.graph_tools import community_list_to_labels

# graspologic 3.4 compared `abs(A - A.T).max()` against this before handing an
# adjacency matrix to the native core; kept so the same inputs are accepted.
_SYMMETRY_ATOL = 1e-15

_DIRECTED_MSG = (
    "leiden only supports undirected graphs and the adjacency matrix provided "
    "was found to be directed"
)


def _adjacency_to_edge_list(g):
    """Edge list for graspologic-native from a symmetric adjacency matrix.

    Mirrors graspologic 3.4's `_adjacency_matrix_to_edge_list` so the native
    core sees the same edges in the same order as before and the partitions
    match: every non-zero entry on or above the diagonal becomes
    ``(str(i), str(j), float(weight))``, row-major; the lower triangle is
    implied. Any scipy sparse format is accepted (graspologic took only
    `csr_array`).

    Returns:
        (n_nodes, edges): the matrix order and the edge list.
    """
    if sp.issparse(g):
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError("adjacency matrix must be square (n x n)")
        coo = sp.coo_matrix(g)
        coo.sum_duplicates()
        asym = abs(coo - coo.T)
        if asym.nnz and asym.max() > _SYMMETRY_ATOL:
            raise ValueError(_DIRECTED_MSG)
        keep = (coo.row <= coo.col) & (coo.data != 0)
        rows, cols, data = coo.row[keep], coo.col[keep], coo.data[keep]
        order = np.lexsort((cols, rows))
        edges = [
            (str(int(i)), str(int(j)), float(w))
            for i, j, w in zip(rows[order], cols[order], data[order], strict=True)
        ]
        return g.shape[0], edges

    a = np.asarray(g)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency matrix must be square (n x n)")
    if a.size and np.abs(a - a.T).max() > _SYMMETRY_ATOL:
        raise ValueError(_DIRECTED_MSG)
    iu, ju = np.triu_indices(a.shape[0])
    w = a[iu, ju]
    keep = w != 0
    edges = [
        (str(int(i)), str(int(j)), float(x))
        for i, j, x in zip(iu[keep], ju[keep], w[keep], strict=True)
    ]
    return a.shape[0], edges


def _leiden_native(g, use_modularity, resolution, random_state):
    """Run Leiden through `graspologic_native.leiden` on an adjacency matrix.

    The call graspologic 3.4 made (one iteration, one trial, randomness
    0.001, no starting communities), minus its beartype layer and its
    weighted/unweighted scan, which only ever rewrote a weight of 1 to 1.0.
    Isolate nodes are not returned, as before; callers that need a label for
    every node fill the gap themselves (`reconstruct.LeidenReconstructor`).
    """
    import graspologic_native as gn

    if resolution <= 0:
        raise ValueError("resolution must be a positive float")
    if random_state is not None:
        if int(random_state) != random_state or random_state <= 0:
            raise ValueError(
                "random_state must be a positive integer or None (graspologic-native "
                "seeds an unsigned 64-bit PRNG)"
            )
        random_state = int(random_state)

    n_nodes, edges = _adjacency_to_edge_list(g)
    _quality, partition = gn.leiden(
        edges=edges,
        starting_communities=None,
        resolution=float(resolution),
        randomness=0.001,
        iterations=1,
        use_modularity=bool(use_modularity),
        seed=random_state,
        trials=1,
    )
    if len(partition) < n_nodes:
        warnings.warn(
            "Leiden partitions do not contain all nodes from the input graph because "
            "input graph contained isolate nodes.",
            stacklevel=3,
        )
    indices = np.fromiter((int(k) for k in partition), dtype=int, count=len(partition))
    labels = np.fromiter((int(v) for v in partition.values()), dtype=int, count=len(partition))
    return indices, labels


def _leiden(g, method="graspologic", objective="modularity", resolution=1.0, random_state=None):
    """
    Compute the Leiden clustering of a graph (numpy or sparse matrix).

    Args:
        g (ndarray or sparse_csr_matrix): graph as an adjacency matrix.
        method ("graspologic" | "leidenalg" | "igraph" | "cdlib"): backend;
            "graspologic" is graspologic-native called directly.
        objective ("modularity" | "cpm"): quality function.
        resolution (float): resolution parameter.
        random_state (int or None): random seed for backends that accept it.

    Returns:
        (indices, labels): sorted node indices and their cluster labels.
    """
    if objective not in ["modularity", "cpm"]:
        warnings.warn("Objective function not recognized; falling back to modularity")
        objective = "modularity"

    valid_methods = {"graspologic", "igraph", "leidenalg", "cdlib"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown community-detection method {method!r}; "
            f"expected one of {sorted(valid_methods)}"
        )

    if method in ["igraph", "leidenalg"]:
        import igraph as ig

        g_ig = ig.Graph.Adjacency(g).as_undirected()

    if method == "cdlib":
        import networkx as nx

        if sp.issparse(g):
            g_nx = nx.from_scipy_sparse_array(g)
        else:
            g_nx = nx.from_numpy_array(np.asarray(g))

    if method == "graspologic":
        indices, labels = _leiden_native(
            g,
            use_modularity=objective == "modularity",
            resolution=resolution,
            random_state=random_state,
        )

    elif method == "igraph":
        # igraph names its objectives "modularity" / "CPM" and takes the
        # resolution through `resolution_parameter`. The previous call
        # hardcoded `resolution_parameter=1.0`, silently dropping the caller's
        # `resolution` (the graspologic branch forwards it correctly).
        objective_function = "CPM" if objective == "cpm" else "modularity"
        cluster_obj = g_ig.community_leiden(
            objective_function=objective_function, resolution_parameter=resolution
        )
        labels = cluster_obj.membership
        indices = np.arange(len(labels))

    elif method == "cdlib":
        import cdlib.algorithms as cdlib_algorithms

        if objective == "modularity":
            coms = cdlib_algorithms.leiden(g_nx)
        else:  # objective == "cpm"
            coms = cdlib_algorithms.cpm(g_nx)
        indices, labels = np.array(community_list_to_labels(coms.communities)).T

    elif method == "leidenalg":
        import leidenalg as la

        objective_obj = (
            la.ModularityVertexPartition
            if objective == "modularity"
            else la.CPMVertexPartition
        )
        cluster_membership = la.find_partition(g_ig, objective_obj)
        labels = cluster_membership._membership
        indices = np.arange(len(labels))

    sort_inds = np.argsort(indices)
    indices, labels = indices[sort_inds], labels[sort_inds]

    return indices, labels
