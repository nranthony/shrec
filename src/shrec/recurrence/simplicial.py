"""Per-response fuzzy simplicial complex (paper Appendix B step 3).

For each point, find its k nearest neighbours, set `ρ_i` to the closest-
neighbour distance, then root-solve for `σ_i` such that
`Σ_m exp(-ReLU(d_im − ρ_i)/σ_i) = log₂(k)`. The resulting per-row
affinity is symmetrised via the fuzzy union `A + Aᵀ − A∘Aᵀ`.
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.distance import cdist


def relu(x):
    return np.maximum(0, x)


def dataset_to_simplex(X, k=20, tol=1e-6, precomputed=False):
    """
    Fuzzy simplicial complex over a single point cloud.

    Args:
        X (np.ndarray): dataset of shape (n_samples, n_features) or, if
            ``precomputed=True``, a precomputed (n, n) distance matrix.
        k (int): number of nearest neighbours used in the σ root-solve.
        tol (float): tolerance passed to `scipy.optimize.fsolve`.
        precomputed (bool): treat ``X`` as a distance matrix.

    Returns:
        wmat (np.ndarray): affinity matrix of shape (n_samples, n_samples).

    References:
        McInnes, Healy, Melville. "UMAP: Uniform Manifold Approximation
        and Projection for Dimension Reduction." arXiv:1802.03426 (2018).
    """
    dmat = X.copy() if precomputed else cdist(X, X)

    n = dmat.shape[0]
    dmat_zerofilled = dmat.copy()
    dmat_zerofilled[dmat_zerofilled < 1e-10] = np.inf
    dists = np.partition(dmat_zerofilled, k + 1, axis=1)
    dists = np.sort(dists, axis=1)[:, 1:k + 1]
    rho = dists[:, 0]
    for i in range(n):
        def func(sig): return sum(np.exp(-relu(dists[i] - rho[i]) / sig)) - np.log2(k)
        def jac(sig): return sum(np.exp(-relu(dists[i] - rho[i]) / sig) * relu(dists[i] - rho[i])) / sig**2
        sigma_i = fsolve(func, rho[i], fprime=jac, xtol=tol)[0]
        dmat[i] = np.exp(-relu(dmat[i] - rho[i]) / sigma_i)
    return dmat + dmat.T - dmat * dmat.T
