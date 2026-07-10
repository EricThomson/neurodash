"""Spectral analysis utilities for LFP signals.

Provides multitaper spectrogram computation via lspopt. Functions here are
pure signal processing — they take numpy arrays and return numpy arrays,
with no UI or I/O dependencies.
"""

import numpy as np
from lspopt import spectrogram_lspopt


def compute_multitaper_spectrogram(
    signal,
    sampling_rate,
    window_duration=2.0,
    step_duration=0.1,
    c_parameter=20,
    max_frequency=50.0,
):
    """Compute multitaper spectrogram using lspopt.

    Returns: frequencies, times, power_db
    """
    nperseg = int(window_duration * sampling_rate)
    noverlap = int((window_duration - step_duration) * sampling_rate)
    noverlap = max(0, min(noverlap, nperseg - 1))

    freqs, times, power = spectrogram_lspopt(
        signal,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        c_parameter=c_parameter,
    )

    mask = freqs <= max_frequency
    freqs = freqs[mask]
    power = power[mask, :]

    # Convert to dB
    power_db = 10 * np.log10(power + 1e-12)
    return freqs, times, power_db
