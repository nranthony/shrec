"""Time-delay embedding (Takens / method-of-delays).

Extracted from `utils/preprocessing.py` so the embedding step is testable
in isolation and consumable by `models.base` without dragging in scalers
and detrenders.
"""
import numpy as np
from scipy.linalg import hankel


def embed_ts(X, m, padding=None):
    """
    Create a time delay embedding of a time series or a set of time series.

    Args:
        X (array-like): A matrix of shape (n_timepoints, n_dims) or
            of shape (n_timepoints,) or (n_series, n_timepoints, n_dims).
        m (int): The embedding dimension.
        padding: numpy.pad mode for left-padding the series so the output
            has the same length as the input (e.g. "symmetric", "edge",
            "constant"). If falsy, the output is shorter by ``m`` samples.

    Returns:
        Xp (ndarray): the delay embedding.
    """
    if padding:
        if len(X.shape) == 1:
            X = np.pad(X, [m, 0], padding)
        if len(X.shape) == 2:
            X = np.pad(X, [[m, 0], [0, 0]], padding)
        if len(X.shape) == 3:
            X = np.pad(X, [[0, 0], [m, 0], [0, 0]], padding)
    Xp = hankel_matrix(X, m)
    Xp = np.moveaxis(Xp, (0, 1, 2), (1, 2, 0))
    return Xp


def hankel_matrix(data, q, p=None):
    """
    Find the Hankel matrix dimensionwise for multiple multidimensional
    time series.

    Args:
        data (ndarray): An array of shape (N, T, 1) or (N, T, D) corresponding
            to a collection of N time series of length T and dimensionality D.
        q (int): The width of the matrix (number of features).
        p (int): The height of the matrix (number of samples).
    """
    if len(data.shape) == 3:
        return np.stack([_hankel_matrix(item, q, p) for item in data])

    if len(data.shape) == 1:
        data = data[:, None]
    return _hankel_matrix(data, q, p)


def _hankel_matrix(data, q, p=None):
    """Hankel matrix of a multivariate time series of shape (T, D)."""
    if len(data.shape) == 1:
        data = data[:, None]

    if not p:
        p = len(data) - q
    all_hmats = []
    for row in data.T:
        first, last = row[-(p + q): -p], row[-p - 1:]
        out = hankel(first, last)
        all_hmats.append(out)
    out = np.dstack(all_hmats)
    return np.transpose(out, (1, 0, 2))[:-1]


def make_embedding(X, d_embed, padding="symmetric", noise=0.0, rng=None):
    """Time-delay-embed `X`, broadcasting over the multivariate channel axis.

    Extracted from `RecurrenceModel._make_embedding`. Side-effect free —
    pass a `numpy.random.Generator` for reproducible noise injection
    rather than relying on global RNG state.

    Args:
        X (ndarray): shape (T, D) or (B, T, D).
        d_embed (int): embedding dimension per channel.
        padding: numpy.pad mode (see `embed_ts`).
        noise (float): standard deviation of i.i.d. Gaussian regularisation
            added to the embedded output. Zero (default) means no noise.
        rng (numpy.random.Generator | None): generator used for the noise.
            Required when ``noise > 0``.

    Returns:
        X_embed (ndarray): shape (D, T, d_embed) for 2-D input, or
            (D, T, d_embed * channels) for 3-D input.
    """
    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], -1))
        X_embed = embed_ts(X, d_embed, padding=padding)
    elif X.ndim == 3:
        slices = []
        for i in range(X.shape[-1]):
            slice_2d = np.reshape(X[..., i], (X.shape[0], -1))
            slices.append(embed_ts(slice_2d, d_embed, padding=padding))
        X_embed = np.hstack(slices)
    else:
        raise ValueError(
            f"Input must be 2-D or 3-D, got shape {X.shape}"
        )

    if noise > 0.0:
        if rng is None:
            raise ValueError("rng must be provided when noise > 0")
        X_embed = X_embed + noise * rng.standard_normal(X_embed.shape)

    return X_embed
