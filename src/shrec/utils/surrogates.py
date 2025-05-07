
from sklearn.preprocessing import PowerTransformer
import numpy as np


def make_surrogate(X, ns=1, method="random_phase", gaussianize=True, random_state=None):
    """
    Create a surrogate time series from a given time series. If gaussianize is True,
    then the surrogate will be gaussianized beforehand. The default configuration 
    approximates the AAFT surrogate method.

    Args:
        X (ndarray): A one-dimensional time series
        method (str): "random_shuffle" or "random_phase"
        gaussianize (bool): If True, the surrogate will be gaussianized
        random_state (int): Random seed for reproducibility

    Returns:
        Xs (ndarray): A single random surrogate time series
    """
    if gaussianize:
        model = PowerTransformer(standardize=True)
        X = np.squeeze(model.fit_transform(X[:, None]))

    np.random.seed(random_state)
    if method == "random_phase":
        phases, radii = np.angle(np.fft.fft(X)), np.abs(np.fft.fft(X))
        random_phases = 2 * np.pi * (2 * (np.random.random(size=(phases.shape[0], ns)) - 0.5))
        Xs = np.real(
            np.fft.ifft(
                radii[:, None] * np.cos(random_phases) + 1j * radii[:, None] * np.sin(random_phases),
                axis=0
            )
        )
    else:
        Xs = np.random.permutation(X)

    if gaussianize:
        Xs = np.array([
            model.inverse_transform(item[:, None]) for item in Xs.T
        ]).T

    Xs = np.squeeze(Xs)
    return Xs


def array_select(arr, inds):
    """
    Selects a subset of an array, given a set of indices or boolean slices
    """
    arr_out = np.copy(arr)
    arr_out = arr_out[inds]
    arr_out = arr_out[:, inds]
    return arr_out
