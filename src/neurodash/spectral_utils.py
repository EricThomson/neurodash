"""Spectral analysis utilities for LFP signals.

Provides multitaper spectrogram computation via lspopt. Functions here are
pure signal processing — they take numpy arrays and return numpy arrays,
with no UI or I/O dependencies.
"""

import numpy as np
import anaties as ana
from lspopt import spectrogram_lspopt
from scipy.interpolate import interp1d


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


# ---------------------------------------------------------------------------
# Theta analysis — peak frequency and band power over time
#
# Both operate on a spectrogram's *linear* power (n_freqs, n_times). The repo's
# spectrogram returns dB, so callers convert first (see ``db_to_linear``); argmax
# for the peak is unaffected by dB-vs-linear, but a band *mean* is not, so the
# mean is taken in linear power (the correct convention).
# ---------------------------------------------------------------------------

def db_to_linear(power_db):
    """Invert ``10*log10`` back to linear power (the spectrogram is stored in dB)."""
    return np.power(10.0, np.asarray(power_db, dtype=float) / 10.0)


def smooth_series(series, width):
    """Hann-smooth a 1-D series via anaties (`ana.smooth` — zero-phase filtfilt, the
    same call the analysis notebooks use). ``width`` is the filter width in bins;
    returned unchanged for width < 3 (a Hann of width 2 is degenerate).

    NaN-safe: `filtfilt` raises on non-finite input. Gaps are bridged by linear
    interpolation for the filter pass and restored afterwards, so a NaN never
    propagates outward across ``width`` bins the way it would if fed straight in.
    """
    series = np.asarray(series, dtype=float)
    if int(width) < 3:
        return series

    missing = ~np.isfinite(series)
    if not missing.any():
        return ana.smooth(series, window_type="hann",
                          filter_width=int(width), plot_on=False)[0]
    if missing.all():
        return series

    idx = np.arange(len(series))
    filled = series.copy()
    filled[missing] = np.interp(idx[missing], idx[~missing], series[~missing])
    smoothed = ana.smooth(filled, window_type="hann",
                          filter_width=int(width), plot_on=False)[0]
    smoothed[missing] = np.nan
    return smoothed


def smooth_time(power, width):
    """Hann-smooth a (n_freqs, n_times) array along time via anaties' ``smooth_rows``
    (smooths each frequency's time series). ``width`` is the filter width in bins."""
    power = np.asarray(power, dtype=float)
    if int(width) < 3:
        return power
    return ana.smooth_rows(power, window_type="hann", filter_width=int(width))


def theta_band_power(freqs, power, theta_band=(4.0, 12.0)):
    """Mean linear power across the theta band, per time bin.

    Parameters
    ----------
    freqs : (n_freqs,) array
    power : (n_freqs, n_times) linear power
    theta_band : (low_hz, high_hz)

    Returns
    -------
    (n_times,) array — mean band power (linear).
    """
    freqs = np.asarray(freqs, dtype=float)
    low, high = theta_band
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return np.full(power.shape[1], np.nan)
    return np.asarray(power)[mask, :].mean(axis=0)


def bandpass_lfp(signal, sampling_rate, low_hz, high_hz, order=8):
    """Butterworth bandpass via anaties (`ana.bandpass_filter`, zero-phase SOS).

    Used by the "bandpass" theta estimator: strip the contaminating energy from the
    LFP *before* the spectrogram, then a plain argmax works. Most of the benefit is
    de-leakage rather than in-band attenuation — the spectrogram's ~0.86 Hz kernel
    smears a 3 Hz delta peak into 4 Hz, and filtering deletes the source. (Scaling
    the unfiltered spectrum by |H|^2 reproduces only about a quarter of the effect.)

    The caveat is that this is **destructive and undetectable**. Whatever the
    filter removes is gone before any estimator sees it, so an edge set inside the
    real theta range mangles the answer with no downstream signature. Bandpassing
    to exactly 4-12 Hz drops a genuine 4.1 Hz peak to a reported 5.7-5.9 Hz;
    3.5-12.5 leaves it intact. Keep the edges outside the analysis band.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    low = max(float(low_hz), 0.01)
    high = min(float(high_hz), 0.99 * sampling_rate / 2.0)
    if high <= low:
        return signal
    return ana.bandpass_filter(signal, low, high, sampling_rate,
                               order=int(order), plot_on=0)[0]


def theta_peak_frequency(freqs, power, theta_band=(4.0, 12.0), interp_step=0.1):
    """Frequency of maximum power within the theta band, per time bin.

    Quadratic-interpolates each time slice onto an ``interp_step`` (Hz) grid so
    the peak isn't quantized to the spectrogram's coarse bin width, then takes
    the argmax frequency. The interpolation is applied as a single precomputed
    matrix (quadratic-spline interpolation is linear in the data), so all time
    bins are done in one matmul rather than a per-bin Python loop.

    Parameters
    ----------
    freqs : (n_freqs,) array
    power : (n_freqs, n_times) linear power
    theta_band : (low_hz, high_hz)
    interp_step : float — fine-grid spacing in Hz

    Returns
    -------
    (n_times,) array — peak frequency in Hz.
    """
    freqs = np.asarray(freqs, dtype=float)
    power = np.asarray(power, dtype=float)
    low, high = theta_band
    mask = (freqs >= low) & (freqs <= high)
    band_freqs = freqs[mask]
    band_power = power[mask, :]

    if band_freqs.size == 0:
        return np.full(power.shape[1], np.nan)
    if band_freqs.size < 3:
        # too few bins to interpolate — fall back to the raw argmax
        return band_freqs[np.argmax(band_power, axis=0)]

    n_steps = max(int((band_freqs[-1] - band_freqs[0]) / interp_step), 1)
    fine_freqs = np.linspace(band_freqs[0], band_freqs[-1], n_steps)
    # Interpolation matrix: column j is e_j interpolated onto fine_freqs.
    basis = np.eye(band_freqs.size)
    interp_matrix = interp1d(band_freqs, basis, kind="quadratic", axis=0)(fine_freqs)
    fine_power = interp_matrix @ band_power  # (n_fine, n_times)
    return fine_freqs[np.argmax(fine_power, axis=0)]
