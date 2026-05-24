import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.stats import iqr
from sklearn.decomposition import PCA
from scipy.linalg import hankel


def nan_fill(a):
    """
    Backfill nan values in a numpy array along the first axis
    """
    out = np.copy(a)
    for i in range(out.shape[0]):
        if np.any(np.isnan(out[i])):
            out[i] = out[i - 1]
    return out


def nan_pca(X, weights=None):
    """
    Perform pca on a data matrix with missing values

    Args:
        X (array): A data matrix with shape (N, D)
        weights (array): A weight matrix with shape N,

    Returns:
        X_transformed (array): The data after projection onto the eigenvectors of the 
            covariance matrix
    """
    if weights is None:
        weights = np.ones(X.shape[1])
    w_sum = np.nansum(weights)
    w_mat = np.diag(weights)
    X_mean = (1 / X.shape[0]) * np.nansum(weights * X, axis=0, keepdims=True)
    X = X - X_mean
    cov = (1 / w_sum) * np.nansum(X[..., None] * w_mat.dot(X.T).T[:, None, :], axis=0)
    eigs, vecs = np.linalg.eigh(cov)
    vecs = vecs.T
    eigs, vecs = eigs[::-1], vecs[::-1]

    X_transformed = X.dot(vecs.T)
    return X_transformed

    # X = X - np.mean(X, axis=0, keepdims=True)
    # cov = np.dot(X.T, X)
    # #cov = np.cov(X.T)
    # print(cov)
    # eigs, vecs = np.linalg.eigh(cov)
    # vecs = vecs.T
    # eigs, vecs = eigs[::-1], vecs[::-1]
    # return vecs

    # X -= np.mean(X, axis = 0)
    # cov = np.cov(X, rowvar = False)
    # eigs, vecs = np.linalg.eigh(cov)
    # idx = np.argsort(eigs)[::-1]
    # vecs = vecs[:,idx]
    # eigs = eigs[idx]
    # return vecs.T


def matrix_lowrank(a, k=-1):
    """Returns the low-rank approximation of a matrix"""
    U, s, V = np.linalg.svd(a)
    return U[:, :k] @ np.diag(s[:k]) @ V[:k, :]


def zero_topk(a, k=1, magnitude=False):
    """
    Return a copy of a vector with all components except the top k set equal to zero

    Args:
        a (array): A vector with shape (D,)
        k (int): The number of components to zero
        magnitude (bool): If True, zero everything but the top k components

    Returns:
        a2 (array): A copy of a with the top k components set equal to zero
    """
    a2 = np.zeros_like(a)
    if magnitude:
        topk_inds = np.argsort(np.abs(a))[::-1][:k]
    else:
        topk_inds = np.argsort(a)[::-1][:k]
    a2[topk_inds] = a[topk_inds]
    return a2


def detrend_ts(a, method="global_linear"):
    """
    Detrend a time series along its first axis using a variety of methods

    Arguments
        a (array): A time series of shape (T, D)
        method (str): "global_linear" - subtracts the best fit straight line from the data
                      "naive" - subtracts the line bridging the first and final values

    Development
        Fully vectorize line fitting by estimating inverse of Vandermonde matrix
    """
    if len(a.shape) < 2:
        a = a[:, None]

    if method == "naive":
        trend = a[0] + (a[-1] - a[0])[None, :] * np.arange(a.shape[0])[:, None]
        return a - np.squeeze(trend)
    elif method == "global_linear":
        all_trends = list()
        for row in a.T:
            m, b = np.polyfit(np.arange(a.shape[0]), row, 1)
            trend = (m * np.arange(a.shape[0]) + b)
            all_trends.append(trend)
        all_trends = np.array(all_trends).T
        return a - all_trends
    elif method == "global_exponential":
        all_trends = list()
        for row in a.T:
            m, b = np.polyfit(np.arange(a.shape[0]), np.log(row), 1)
            trend = np.exp(m * np.arange(a.shape[0])) * np.exp(b)
            all_trends.append(trend)
        all_trends = np.array(all_trends).T
        return a - all_trends
    else:
        trend = (a[-1] - a[0]) * np.arange(a.shape[0])
        return np.squeeze(a - trend)


def transform_stationary(ts, pthresh=0.05):
    """ 
    Transform a time series to be stationary using the Augmented Dickey-Fuller test and
    the Kwiatkowski-Phillips-Schmidt-Shin test. Depending on which combination of tests
    the time series passes, the time series is transformed using differencing or
    detrending.

    Args:
        ts (np.ndarray): Time series to be transformed.
        pthresh (float): Threshold for p-value of statistical tests.

    Returns:
        out (np.ndarray): Transformed time series.
    """

    ts = np.squeeze(ts).copy()
    ad_fuller = adfuller(ts)
    kpss_test = kpss(ts)

    if kpss_test[1] < pthresh:
        ts = np.diff(ts)

    if ad_fuller[1] > pthresh:
        ts = detrend_ts(ts)

    return ts.squeeze()


def unroll_phase(theta0, wrapval=2*np.pi):
    """
    Given a list of phases, unroll them in order to prevent wraparound discountinuities
    """
    theta = np.copy(theta0)
    sel_inds = np.abs(np.diff(theta)) > wrapval * 0.9
    inds = np.sort(np.where(sel_inds)[0])
    for ind in inds:
        diffval = theta[ind + 1] - theta[ind]
        theta[ind+1:] -= np.sign(diffval) * wrapval
    return theta


def lift_ts(a, target=2):
    """
    If needed, pad the dimensionality of a univariate time series
    """
    deficit = target - len(a.shape)
    if deficit == 1:
        return a[:, None]
    else:
        return a


def standardize_ts(a, scale=1.0, median=False):
    """Standardize an array along dimension -2
    For dimensions with zero variance, divide by one instead of zero

    Args:
        a (ndarray): a matrix containing a time series or batch of time series
            with shape (T, D) or (B, T, D)
        scale (float): the number of standard deviations by which to scale
        median (bool): whether to use median/IQR to normalize

    Returns:
        ts_scaled (ndarray): A standardized time series with the same shape as 
            the input
    """
    a = lift_ts(a)

    if median:
        center = np.median(a, axis=-2, keepdims=True)
        stds = iqr(a, axis=-2, keepdims=True)
    else:
        center = np.mean(a, axis=-2, keepdims=True)
        stds = np.std(a, axis=-2, keepdims=True)
    stds[stds == 0] = 1
    ts_scaled = (a - center) / (scale * stds)
    return np.squeeze(ts_scaled)


def minmax_ts(a, clipping=None):
    """MinMax scale an array along dimension -2
    For dimensions with zero variance, divide by one instead of zero

    Args:
        a (ndarray): a matrix containing a time series or batch of time series
            with shape (T, D) or (B, T, D)
        clipping (float): A number between 0 and 1, the range of values 
            to use for rescaling

    Returns:
        ts_scaled (ndarray): A minmax scaled time series with the same shape as 
            the input
    """
    a = lift_ts(a)

    if clipping:
        minval = np.percentile(a, clipping * 100, axis=-2, keepdims=True)
        maxval = np.percentile(a, (1 - clipping) * 100, axis=-2, keepdims=True)
    else:
        minval = np.min(a, axis=-2, keepdims=True)
        maxval = np.max(a, axis=-2, keepdims=True)
    spans = (maxval - minval)
    spans[spans == 0] = 1
    ts_scaled = (a - minval) / spans
    return np.squeeze(ts_scaled)


# Time-delay embedding lives in shrec.embeddings; re-exported here for
# backwards-compatible imports (`from shrec.utils import embed_ts`).
from shrec.embeddings import embed_ts, hankel_matrix, _hankel_matrix  # noqa: F401


def allclose_len(arr1, arr2):
    """Test whether all entries are close, and return False if different shapes.

    np.allclose broadcasts before comparing, so it returns True for inputs
    of mismatched-but-broadcastable shape (e.g. (3,) vs (2, 3)). Explicit
    shape comparison is required to honour the documented contract.
    """
    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)
    if arr1.shape != arr2.shape:
        return False
    return bool(np.allclose(arr1, arr2))


def array2d_to_list(arr):
    """Convert a 2D ndarray to a list of lists"""
    return [list(item) for item in arr]


def dict_to_vals(d):
    """
    Convert a dictionary to an array of values, in the order of the sorted keys.

    Args:
        d (dict): dictionary to convert

    Returns:
        d_arr (np.array): array of values
        key_names (list): list of keys
    """
    d_arr = list()
    key_names = sorted(d.keys())
    for key in key_names:
        d_arr.append(d[key])
    d_arr = np.array(d_arr)
    return d_arr, key_names


def spherize_ts(X):
    """Spherize a time series with PCA"""
    X = X - np.mean(X, axis=0)
    X = X.dot(PCA().fit(X).components_.T)
    return X


def whiten_zca(X):
    """
    Whiten a dataset with ZCA whitening (Mahalanobis whitening).
    Args:
        X: numpy array of shape (n_samples, n_features)
    """
    sigma = np.cov(X, rowvar=True)
    U, S, V = np.linalg.svd(sigma)
    zmat = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + 1e-6)), U.T))
    X_out = np.dot(zmat, X)
    return X_out
