import numpy as np
from scipy.signal import find_peaks
from scipy.signal import periodogram
from scipy.signal.windows import blackmanharris
from scipy.ndimage import gaussian_filter1d


def discretize_signal(signal, max_states=50):
    """
    Given a continuous signal, discretize into a finite number of states

    Args:
        signal (array): A signal with shape (T, D)
        max_states (int): The maximum number of states to use
    """
    n = len(signal)
    _, bins = np.histogram(signal, max_states)
    signal_d = np.digitize(signal, bins, right=True)  # returns which bin

    # prune bins with few elements
    vals, counts = np.unique(
        signal_d,
        return_counts=True
    )
    sel_vals = (counts / n) < 0.01
    for i, loc in enumerate(vals[sel_vals]):
        if i + 1 < len(vals):
            signal_d[signal_d == loc] = vals[i + 1]
        else:
            signal_d[signal_d == loc] = vals[i - 1]

    # convert to fewer labels
    keys, vals = np.unique(signal_d), np.arange(len(signal_d))
    trans_dict = dict(zip(keys, vals))
    signal_d = np.array([trans_dict[key] for key in signal_d])

    return signal_d


def discretize_ts(ts, **kwargs):
    """
    Given a univariate time series, return a discretization based on peak crossings

    keyword arguments are passed on to scipy.find_peaks
    """
    peak_inds = np.sort(np.hstack([find_peaks(-ts, **kwargs)[0],
                                   find_peaks(ts, **kwargs)[0]]))
    return ts[peak_inds]


def find_psd(y, window=True):
    """
    Find the power spectrum of a signal

    Args:
        y (array): A signal of shape (T,)

    Returns:
        fvals (array): The frequencies of the power spectrum with shape (T // 2,)
        psd (array): The power spectrum of shape (T // 2,)

    """
    if window:
        y = y * blackmanharris(len(y))
    halflen = int(len(y)/2)
    fvals, psd = periodogram(y, fs=1)
    return fvals[:halflen], psd[:halflen]


def group_consecutives(vals, step=1):
    """
    Return list of consecutive lists of numbers from vals (number list).

    References:
        Modified from the following
        https://stackoverflow.com/questions/7352684/
        how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy 
    """
    run = list()
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def find_characteristic_timescale(y, k=1, window=False):
    """
    Find the k leading characteristic timescales in a time series
    using the power spectrum..
    """
    y = gaussian_filter1d(y, 3)

    fvals, psd = find_psd(y, window=window)
    max_indices = np.argsort(psd)[::-1]

    # Merge adjacent peaks
    grouped_maxima = group_consecutives(max_indices)
    max_indices_grouped = np.array([np.mean(item) for item in grouped_maxima])
    max_indices_grouped = max_indices_grouped[max_indices_grouped != 1]

    return np.squeeze(1/(np.median(np.diff(fvals))*max_indices_grouped[:k]))
