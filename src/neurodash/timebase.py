"""Resampling one signal onto another's time base.

Pure numpy — no UI, no data loading. Exists so every exported channel lands on a
single time base (the spectrogram bins) instead of each stream keeping its own,
which is how the source notebooks ended up with two copies of every variable.
"""

import warnings

import numpy as np


def window_average(src_times, values, target_times, window_duration):
    """Average `values` over a window centred on each target time.

    For each `t` in target_times, returns the nanmean of every source sample
    falling in ``[t - window/2, t + window/2]``.

    The window is a parameter rather than the target spacing because the point
    is to match the *support* of whatever already lives on the target grid. The
    spectrogram's theta values each summarize a 1.5 s multitaper window, so the
    behavioral value sharing that row should summarize the same 1.5 s — not the
    0.1 s the bin spacing would imply. Passing ``window_duration`` equal to the
    target spacing degenerates to ordinary non-overlapping downsampling.

    Averaging (not decimation) follows the source notebook's `mean_downsample`.

    Parameters
    ----------
    src_times : np.ndarray — source sample times, ascending.
    values : np.ndarray — source values, same length as src_times.
    target_times : np.ndarray — times to average onto.
    window_duration : float — full window width in seconds.

    Returns
    -------
    np.ndarray — one value per target time; NaN where the window holds no
    samples, or only NaNs.
    """
    src_times = np.asarray(src_times, dtype=float)
    values = np.asarray(values, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    half = window_duration / 2.0
    starts = np.searchsorted(src_times, target_times - half, side="left")
    stops = np.searchsorted(src_times, target_times + half, side="right")

    out = np.full(len(target_times), np.nan)
    with warnings.catch_warnings():
        # An all-NaN window is a legitimate result (behavior dropout), not a bug.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, (lo, hi) in enumerate(zip(starts, stops)):
            if hi > lo:
                out[i] = np.nanmean(values[lo:hi])
    return out
