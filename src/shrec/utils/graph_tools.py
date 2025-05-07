
import itertools
from collections import Counter
import numpy as np
import networkx as nx
import scipy.sparse as sp


def hollow_matrix(arr):
    """
    Set the diagonal of a matrix to zero
    """
    return arr * (1 - np.eye(arr.shape[0]))


def common_neighbors_ratio(adj_matrix):
    """
    Given a binary adjacency matrix, compute the common neighbors ratio matrix. Each
    entry in the matrix is one minus the ratio of the number of common neighbors of two 
    nodes to the total unique neighbors of the two nodes.

    Args:
        adj_matrix (np.ndarray): A binary adjacency matrix.

    Returns:
        np.ndarray: A common neighbor-weighted adjacency matrix.
    """
    weighted_matrix = np.zeros_like(adj_matrix)
    n = adj_matrix.shape[0]
    all_neighbor_lists = []
    for i in range(n):
        neighbor_indices = np.where(adj_matrix[i] > 0)[0]
        all_neighbor_lists.append(neighbor_indices)

    for i in range(n):
        for j in range(i + 1, n):
            overlap = np.intersect1d(all_neighbor_lists[i], all_neighbor_lists[j])
            union = np.union1d(all_neighbor_lists[i], all_neighbor_lists[j])
            weighted_matrix[i, j] = 1 - len(overlap) / len(union)

    weighted_matrix = weighted_matrix + weighted_matrix.T
    np.fill_diagonal(weighted_matrix, 0)
    return weighted_matrix


def neighbors_to_mutual(a):
    """
    Given either a binary neighbor matrix or a connectivity matrix with all elements
    between zero and one, return a mutual neighbor matrix, which is a symmetric 
    matrix comprising a subgraph of mutual neighbors
    """
    return (a * a.T) / (0.5 * (a + a.T))


def arg_find(search_vals, target_vals):
    """
    Return the indices in search_vals pointing to the values in target_vals
    The output is unordered
    """
    search_vals, target_vals = np.asarray(search_vals), np.asarray(target_vals)
    return np.where(
        np.prod(search_vals[:, None] - target_vals[None, :], axis=1) == 0
    )[0]


def community_list_to_labels(community_list):
    """
    Given a list of community members, create a list of labels
    """
    all_labels = list()
    for ind, com in enumerate(community_list):
        for member in com:
            all_labels.append((member, ind))

    all_labels = sorted(all_labels, key=lambda x: x[0])
    return all_labels

# def get_all_pairs(ind_list):
#     """
#     Return all unique pairs from the list
#     """
#     all_pairs = list()
#     for ind in ind_list:
#         all_pairs += [(ind, ind2) for ind2 in ind_list if ind2 > ind]
#     return all_pairs


# NetworkX utilities


def largest_connected_component(g):
    """Return the scaled largest connected component of a graph."""
    n = g.number_of_nodes()
    giant = max(nx.connected_components(g), key=len)
    lcc = len(giant) / n
    return lcc


def susceptibility_smallcomponents(g):
    """Return the susceptibility of a graph based on the size of small components."""
    n = g.number_of_nodes()
    all_components = np.array([len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)])
    return np.sum(all_components[1:]**2) / n


def susceptibility_subleading(g):
    """Return the susceptibility of a graph based on the size of largest subleading 
    component."""
    n = g.number_of_nodes()
    all_components = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    if len(all_components) > 1:
        return all_components[1] / n
    else:
        return 0


def sort_graph(g):
    """
    Return a graph with sorted nodes
    """
    h = nx.Graph()
    h.add_nodes_from(sorted(g.nodes(data=True)))
    h.add_edges_from(g.edges(data=True))
    return h


def multigraph_to_weighted(g):
    """
    Convert a MultiGraph to a weighted graph
    """
    c = Counter(g.edges())
    for u, v, d in g.edges(data=True):
        d['weight'] = c[u, v]
    adj = nx.linalg.graphmatrix.adjacency_matrix(g).todense()
    adj = np.sqrt(adj)
    out_g = nx.Graph(adj)
    return out_g


def graph_from_associations(mat, weighted=False):
    """
    Given an association matrix, create a graph of all non-zero elements occurring 
    within a row are assumed to be densely connected to eachother, forming a clique

    Example:
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1]

    Corresponds to the edges
    1 -- 2
    0 -- 1
    1 -- 2
    """
    if weighted:
        g = nx.MultiGraph()
    else:
        g = nx.Graph()
    for ind, row in enumerate(mat):
        inds = np.where(row)[0]
        g.add_edges_from(itertools.combinations(inds, 2))

    if weighted:
        g = multigraph_to_weighted(g)
    return g


def adjmat_from_associations(mat, weighted=False, use_sparse=False):
    """
    Given an association matrix, create an adjacency matrix 
    representing a graph of cliques

    All non-zero elements occurring within a row  of the input
    matrix are assumed to be densely connected to eachother, 
    forming a clique in the output matrix

    Example:
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1]
    --->
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]

    Starting from the edges
    0 -- 1
    0 -- 2
    2 -- 2
    We add the additional edges
    0 -- 0
    1 -- 2
    """

    if use_sparse:
        row_inds = list()
        col_inds = list()
        vals = list()
        for row in mat:
            inds = np.where(row)[0]
            for ind_pair in itertools.combinations(inds, 2):
                row_inds += [ind_pair[0], ind_pair[1]]
                col_inds += [ind_pair[1], ind_pair[0]]
                vals += [1, 1]
        g_adj = sp.csr_matrix((vals, (row_inds, col_inds)), shape=mat.shape)

    # NOT IMPLEMENTED: Weighted
    #     g_adj = g_adj.todense()
    #     if not weighted:
    #         g_adj[g_adj > 0] = 1
    else:

        g_adj = np.zeros(mat.shape)
        for row in mat:
            inds = np.where(row)[0]
            for ind_pair in itertools.combinations(inds, 2):
                g_adj[ind_pair[0], ind_pair[1]] += 1
                g_adj[ind_pair[1], ind_pair[0]] += 1

        # Is this step even needed?
        if not weighted:
            g_adj[g_adj > 0] = 1

    return g_adj


def graph_threshold(g, threshold=1.0):
    """
    Given a weighted graph, return an unweighted graph defined by 
    thresholding the edges
    """
    h = nx.Graph()
    for u, v, d in g.edges(data=True):
        if d["weight"] > threshold:
            h.add_edge(u, v)
    return h


def adjmat(g):
    """
    Wrapper for networkx adjacency matrix
    """
    return nx.linalg.graphmatrix.adjacency_matrix(g)


def multigraph_to_weighted(g):
    """
    Convert a MultiGraph to a weighted graph
    """
    c = Counter(g.edges())
    for u, v, d in g.edges(data=True):
        d['weight'] = c[u, v]
    adj = nx.linalg.graphmatrix.adjacency_matrix(g).todense()
    adj = np.sqrt(adj)
    out_g = nx.Graph(adj)
    return out_g


def merge_labels(labels_list, labels, label_merge="majority"):
    """
    Given a jagged list of labels, merge them into a single list of labels.

    Args:
        labels_list (list): A list of lists of labels.
        labels (list): A list of labels.
        label_merge (str): How to merge labels. Options are "majority" and "average".

    Returns:
        labels_consolidated (list): A list of labels for the merged network
    """

    labels_consolidated = list()
    for item in labels_list:
        if np.isscalar(item):
            labels_consolidated.append(labels[item])
        else:
            votes = list()
            for item2 in item:
                # votes.append(np.argmax(np.bincount(item2)))
                votes.append(labels[item2])

            if label_merge == "majority":
                consensus = np.argmax(np.bincount(votes))
                labels_consolidated.append(votes[consensus])
            elif label_merge == "average":
                labels_consolidated.append(np.mean(votes))
            else:
                consensus = np.argmax(np.bincount(votes))
                labels_consolidated.append(votes[consensus])

    return labels_consolidated


def compress_adjacency(amat0, n_target, return_labels=False, label_merge="majority"):
    """
    Consolidate a network by merging correlated nodes.

    Args:
        amat0 (np.ndarray): Adjacency matrix of the network to be compressed.
        n_target (int): Number of nodes desired for the output network
        labels (np.ndarray | None): If labels are passed, the labels are matched to the 
            compressed network by taking the most common label of the nodes that were 
            merged.

    Returns:
        amat (np.ndarray): Compressed adjacency matrix
        labels (np.ndarray): Labels of the compressed network

    """

    amat = np.copy(amat0)
    all_labels = np.arange(amat.shape[0])

    # drop unconnected components
    where_unconnected = np.isclose(np.sum(np.abs(amat), axis=0), 0)
    amat = amat[np.logical_not(where_unconnected)]
    amat = amat[:, np.logical_not(where_unconnected)]
    amat = hollow_matrix(amat)

    all_labels = all_labels[np.logical_not(where_unconnected)].tolist()

    n_steps = max(amat.shape[0] - n_target, 0)

    for step_ind in range(n_steps):

        if step_ind % (n_steps // 20) == 0:
            print(100 * step_ind / n_steps, end=" ")

        amat_m = amat - np.mean(amat, axis=0, keepdims=True)
        corr_top = np.dot(amat_m, amat_m .T)

        scales = np.linalg.norm(amat_m,  axis=0)**2
        scales_matrix = scales[:, None] * scales[None, :]

        pearson_matrix = corr_top / np.sqrt(scales_matrix)
        pearson_matrix = hollow_matrix(pearson_matrix)

        merge_inds = np.unravel_index(np.argmax(np.ravel(pearson_matrix)), pearson_matrix.shape)

        amat[merge_inds[0]] += amat[merge_inds[1]]
        amat[:, merge_inds[0]] += amat[:, merge_inds[1]]

        all_labels[merge_inds[0]] = np.hstack([
            np.squeeze(np.array(all_labels[merge_inds[0]])),
            np.squeeze(np.array(all_labels[merge_inds[1]]))
        ]).tolist()

        all_labels = np.delete(all_labels, merge_inds[1])
        amat = np.delete(amat, merge_inds[1], axis=0)
        amat = np.delete(amat, merge_inds[1], axis=1)

    if return_labels:
        return amat, all_labels
    else:
        return amat
