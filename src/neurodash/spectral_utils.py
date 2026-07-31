"""Spectral analysis utilities for LFP signals.

Provides multitaper spectrogram computation via lspopt. Functions here are
pure signal processing — they take numpy arrays and return numpy arrays,
with no UI or I/O dependencies.
"""

import numpy as np
import anaties as ana
from lspopt import spectrogram_lspopt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


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

    NaN-safe: `filtfilt` raises on non-finite input, and the crest estimator can
    legitimately return NaN where the band holds no peak. Gaps are bridged by linear
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


def whiten_spectrum(freqs, power, exclude=((3.5, 13.5), (48.0, 52.0)),
                    fit_range=(1.5, 45.0)):
    """Remove the aperiodic 1/f background. Returns residual in dB over background.

    1/f is a straight line in log-log, so this is a plain least-squares fit per time
    bin (vectorised over all bins — ~60 ms for a whole recording). The theta band
    itself is excluded from the fit, along with line noise, so the background is
    interpolated *under* theta rather than being dragged up by it.

    Why this matters for peak finding: a bump sitting on a downward slope has its
    apex pulled downhill, so a raw peak reads systematically low. It also has to
    compete against the slope, which buries it. Removing the background fixes both.

    Parameters
    ----------
    freqs : (n_freqs,) array
    power : (n_freqs, n_times) linear power
    exclude : sequence of (low, high) Hz ranges kept out of the fit
    fit_range : (low, high) Hz over which the background is fitted

    Returns
    -------
    (n_freqs, n_times) array — dB above the fitted background.
    """
    freqs = np.asarray(freqs, dtype=float)
    log_power = np.log10(np.asarray(power, dtype=float) + 1e-30)

    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    for low, high in exclude:
        mask &= ~((freqs >= low) & (freqs <= high))
    if mask.sum() < 3:
        return 10.0 * (log_power - log_power.mean(axis=0, keepdims=True))

    # The spectrogram includes a 0 Hz bin; clamp so log10 doesn't warn. Anything
    # below fit_range is excluded from the fit and never read back out, so the
    # clamped value is only ever a placeholder.
    log_freqs = np.log10(np.maximum(freqs, 1e-6))
    design = np.column_stack([np.ones(mask.sum()), log_freqs[mask]])
    coef, *_ = np.linalg.lstsq(design, log_power[mask, :], rcond=None)
    background = coef[0][None, :] + coef[1][None, :] * log_freqs[:, None]
    return 10.0 * (log_power - background)


def crest_peak_frequency(freqs, whitened, power, theta_band=(4.0, 12.0),
                         search_band=(2.0, 16.0)):
    """Frequency of the strongest genuine spectral crest inside the theta band.

    Two decisions, deliberately made on different quantities:

    **What counts as a peak** is decided on the *whitened* spectrum. A crest is a
    local maximum — power falling away on both sides. An argmax over a bounded band
    must return something even when the spectrum is monotone across it, so delta's
    flank sloping up through 4-12 Hz pins the argmax to the 4 Hz edge and invents a
    peak. A crest can't be produced by a neighbouring hillside, whichever side it is
    on (verified symmetric: an injected 13 Hz tone pins argmax to 12 Hz on 100% of
    bins and moves the crest by 0.03 Hz).

    **Which peak wins** is decided on *raw* power. Whitening removes the 1/f slope,
    which otherwise guarantees low frequencies win — but that slope is also why a
    genuine 4 Hz rhythm reads as dominant. Picking by whitened height over-promotes
    high-frequency crests: at a real 4 Hz theta bout it returned 10 Hz because the
    upper crest stood higher above background, even though 4 Hz carried 4 dB more
    actual power.

    The peak search runs over ``search_band``, wider than ``theta_band``, because
    ``find_peaks`` cannot see a maximum sitting at the first or last sample of its
    input — a real peak exactly at 4 Hz is invisible if the array starts there. Only
    crests whose centre lands inside ``theta_band`` are accepted.

    The apex is refined by parabolic interpolation on the three surrounding points,
    which recovers sub-bin precision without the ~2 s cost of zero-padding the FFT.

    Parameters
    ----------
    freqs : (n_freqs,) array
    whitened : (n_freqs, n_times) dB over background, from ``whiten_spectrum``
    power : (n_freqs, n_times) linear power — used only to rank accepted crests
    theta_band : (low_hz, high_hz) — a crest's centre must land in here
    search_band : (low_hz, high_hz) — where crests are looked for

    Returns
    -------
    peak_hz : (n_times,) array — NaN where no crest lands inside the band.
    peak_height_db : (n_times,) array — dB above background; the confidence measure.
    """
    freqs = np.asarray(freqs, dtype=float)
    search = (freqs >= search_band[0]) & (freqs <= search_band[1])
    search_freqs = freqs[search]
    whitened_vals = np.asarray(whitened, dtype=float)[search, :]
    log_power = 10.0 * np.log10(np.asarray(power, dtype=float)[search, :] + 1e-30)

    n_times = whitened_vals.shape[1]
    peak_hz = np.full(n_times, np.nan)
    peak_height = np.full(n_times, np.nan)
    if search_freqs.size < 3:
        return peak_hz, peak_height

    low, high = theta_band
    in_band = (search_freqs >= low - 1e-9) & (search_freqs <= high + 1e-9)
    df = search_freqs[1] - search_freqs[0]

    for i in range(n_times):
        column = whitened_vals[:, i]
        idx, _ = find_peaks(column)
        if not idx.size:
            continue
        idx = idx[in_band[idx]]
        if not idx.size:
            continue
        j = idx[np.argmax(log_power[idx, i])]
        peak_height[i] = column[j]
        offset = 0.0
        if 0 < j < len(search_freqs) - 1:
            left, mid, right = column[j - 1], column[j], column[j + 1]
            curvature = left - 2.0 * mid + right
            if curvature < 0:
                candidate = 0.5 * (left - right) / curvature
                if abs(candidate) <= 0.5:
                    offset = candidate
        peak_hz[i] = search_freqs[j] + offset * df
    return peak_hz, peak_height


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
