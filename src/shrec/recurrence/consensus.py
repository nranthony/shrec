"""Consensus aggregation of per-response affinity matrices (paper Appendix B step 4)."""
import warnings

import numpy as np

from shrec.recurrence.simplicial import dataset_to_simplex


def data_to_connectivity2(X, k=10, tol=1e-5, time_exclude=0, verbose=False):
    """
    Consensus fuzzy simplicial complex across an ensemble of responses.

    For each response in the ensemble, build the per-response adaptive
    recurrence affinity A^(k) via `dataset_to_simplex` (paper Appendix B
    step 3), then mean-aggregate across responses (step 4). This is the
    canonical SHREC connectivity for the discrete/continuous driver
    models — the alternative `data_to_connectivity` is a simpler fixed-
    scale exp-of-distance kernel kept for the Sauer baseline.

    Args:
        X (np.ndarray): dataset of shape (n_responses, n_times, n_dims).
        k (int): number of nearest neighbours used per row in the
            simplicial-complex root-solve.
        tol (float): tolerance for the σ root-solver.
        time_exclude (int): if >0, zero entries within ±time_exclude of
            the diagonal to suppress trivial trajectory recurrences.
        verbose (bool): whether to print progress.

    Returns:
        wmat (np.ndarray): consensus adjacency matrix of shape
            (n_times, n_times).
    """
    nb, nt, _ = X.shape

    if not verbose:
        warnings.filterwarnings('ignore')

    wmat = np.zeros((nt, nt))
    for ind, X0 in enumerate(X):
        if verbose and nb >= 10 and ind % (nb // 10) == 0:
            print(ind, "/", len(X), flush=True)
        wmat += dataset_to_simplex(X0, k=k, tol=tol) / nb

    if time_exclude > 0:
        mask = 1 - (
            np.triu(np.ones_like(wmat), k=-time_exclude)
            * np.tril(np.ones_like(wmat), k=time_exclude)
        )
        wmat *= mask

    return wmat
