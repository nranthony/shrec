"""Fixed-scale exp-of-distance recurrence kernel (Sauer-style baseline).

Used by `ClassicalRecurrenceClustering`; the canonical SHREC pipeline
uses the adaptive simplicial complex in `simplicial`/`consensus` instead.
"""
import warnings

import numpy as np
from scipy.optimize import root_scalar
from scipy.spatial.distance import cdist


def data_to_connectivity(X, time_exclude=0, scale=1.0, ord=1.0, metric="euclidean"):
    """
    Given a stack of M time series, each of shape N x D, compute a single
    consolidated N x N connectivity matrix via the exp-of-distance kernel.

    Args:
        X (array-like): coordinates of shape (M, N, D) — M response
            channels, N timepoints, D embedding dimensions.
        metric ("euclidean" | "dtw"): distance metric. Only "euclidean"
            is wired through `scipy.spatial.distance.cdist`.
        time_exclude (int): zero-out band around the diagonal to suppress
            trivial trajectory recurrences.
        scale (float): rescaling of the distance matrix (1 = unchanged).
        ord (float): p-norm aggregation order across the ensemble. 1 is
            elementwise mean; ord → ∞ approaches the min-over-channels.

    Returns:
        bd (np.ndarray): connectivity matrix of shape (N, N).
    """
    sel_inds = np.all(np.isclose(X, X[:, :1, :], 1e-12), axis=(1, 2))
    if np.sum(sel_inds) > 0:
        warnings.warn(
            f"{np.sum(sel_inds)} Constant time series detected. "
            "Skipping these datasets."
        )
    X = X[np.logical_not(sel_inds)]
    nb, nt, _ = X.shape

    thresh = np.median(
        np.linalg.norm(X - np.median(X, axis=1, keepdims=True), axis=-1)
    )
    thresh = 1 / (scale * thresh)

    bd = np.zeros((nt, nt))
    for i in range(nb):
        dmat = cdist(X[i], X[i])
        surprise = dmat / np.std(dmat)
        bd += (1 / nb) * np.exp(-surprise * ord / thresh)

    if time_exclude > 0:
        mask_mat = 1 - (
            np.triu(np.ones_like(bd), k=-time_exclude)
            * np.tril(np.ones_like(bd), k=time_exclude)
        )
        bd *= mask_mat.astype(float)

    return bd ** (1 / ord)


def distance_to_connectivity(dmat, dscale=None, sparsity=None):
    """
    Convert a distance matrix to a connectivity matrix via an exponential
    transform. If no crossover scale is specified, one is auto-computed
    from a target sparsity.

    Args:
        dmat (array-like): distance matrix of shape (N, N) or a stack
            (B, N, N).
        dscale (float): crossover scale. Mutually exclusive with
            ``sparsity``.
        sparsity (float): target fraction of zero entries in the output.
            Mutually exclusive with ``dscale``.

    Returns:
        cmat (array-like): connectivity matrix of the same shape as
            ``dmat``.
    """
    if dscale is None and sparsity is None:
        sparsity = 0.99

    if dscale is not None and sparsity is not None:
        warnings.warn(
            "Both a distance scale and a sparsity have been specified; "
            "only the distance scale will be used"
        )

    if dscale is not None:
        return np.exp(-dmat / dscale)

    # sparsity is not None
    dscale = -np.median(dmat) / np.log(1 - sparsity)
    denom = np.sum(np.ones_like(dmat))

    def optfun(x):
        return np.sum(np.exp(-dmat / x)) / denom - (1 - sparsity)

    scale_factor = root_scalar(optfun, bracket=[1e-16, dscale]).root
    return np.exp(-dmat / scale_factor)
