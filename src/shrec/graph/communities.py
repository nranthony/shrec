"""Leiden community-detection adapter — multi-backend (paper Appendix B step 5).

Dispatches to graspologic (default; hard dep), igraph, leidenalg, or
cdlib if installed. Backends are imported lazily so the core library is
usable without optional packages.
"""
import warnings

import numpy as np
import scipy.sparse as sp

from shrec.utils.graph_tools import community_list_to_labels


def _leiden(g, method="graspologic", objective="modularity", resolution=1.0, random_state=None):
    """
    Compute the Leiden clustering of a graph (numpy or sparse matrix).

    Args:
        g (ndarray or sparse_csr_matrix): graph as an adjacency matrix.
        method ("graspologic" | "leidenalg" | "igraph" | "cdlib"): backend.
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
        import graspologic

        mod_flag = objective == "modularity"
        partition = graspologic.partition.leiden(
            g, use_modularity=mod_flag, resolution=resolution, random_seed=random_state,
        )
        indices, labels = np.array([(key, partition[key]) for key in partition]).T

    elif method == "igraph":
        cluster_obj = g_ig.community_leiden(
            objective_function=objective, resolution_parameter=1.0
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
