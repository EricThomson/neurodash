"""Central session object and file-loading machinery for neurodash.

A Session holds all loaded data for one recording — neural, behavioral, or both.
Files are loaded directly from filesystem paths — no copying.

Caching uses functools.lru_cache (process-level). The Dash app is a standard
Python process so module-level lru_cache works fine.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from neurodash import config
from neurodash import spectral_utils
from neurodash.neural_io import load_pl2_block, list_analog_signal_summaries, get_analog_signal, extract_time_window
from neurodash.spectral_utils import compute_multitaper_spectrogram
from neurodash.behavior_io import load_behavior_file


class Session:
    """Container for one recording session.

    At least one of neural or behavioral data must be loaded.
    Check has_neural / has_behavior before accessing those attributes.
    """

    def __init__(self,
                 pl2_path=None, block=None, channel_names=None,
                 behavior_path=None, behavior_metadata=None, behavior_data=None):
        # Neural
        self.pl2_path = pl2_path
        self.block = block
        self.analog_signal_summaries = (
            list_analog_signal_summaries(block, channel_names) if block else []
        )
        self.rec_datetime = (
            block.annotations.get("m_CreatorDateTime", None) if block else None
        )

        # Behavioral
        self.behavior_path = behavior_path
        self.behavior_metadata = behavior_metadata
        self.behavior_data = behavior_data

    @property
    def has_neural(self):
        return self.block is not None

    @property
    def has_behavior(self):
        return self.behavior_data is not None


# ---------------------------------------------------------------------------
# Cached loaders — load directly from path, no copying
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _cached_load_block(pl2_path_str):
    block, channel_names = load_pl2_block(Path(pl2_path_str))
    return block, channel_names


@lru_cache(maxsize=4)
def _cached_load_behavior(behavior_path_str):
    metadata, data = load_behavior_file(Path(behavior_path_str))
    return metadata, data


@lru_cache(maxsize=8)
def compute_spectrogram(
    pl2_path_str,
    analog_signal_index,
    channel_index,
    start_time_sec,
    duration_sec,
    max_frequency,
    window_duration,
    step_duration,
    c_parameter,
    bandpass_low=None,
    bandpass_high=None,
    bandpass_order=8,
):
    """Compute and cache spectrogram by pl2_path + parameters.

    ``bandpass_low``/``bandpass_high`` optionally filter the LFP *before* the
    spectrogram — used by the "bandpass" theta estimator. They are part of the cache
    key, so filtered and unfiltered spectrograms coexist rather than evicting each
    other when you flip estimators.

    Keys on pl2_path_str for hashability (lru_cache requirement).
    """
    block, _ = _cached_load_block(pl2_path_str)
    sig = get_analog_signal(block, analog_signal_index)
    t, y, sr = extract_time_window(sig, channel_index, start_time_sec, duration_sec)
    if bandpass_low is not None and bandpass_high is not None:
        y = spectral_utils.bandpass_lfp(y, sr, bandpass_low, bandpass_high,
                                        order=bandpass_order)
    freqs, times, power_db = compute_multitaper_spectrogram(
        y,
        sampling_rate=sr,
        window_duration=window_duration,
        step_duration=step_duration,
        c_parameter=c_parameter,
        max_frequency=max_frequency,
    )
    return freqs, times, power_db


def _clamp_band(band, theta_band, label):
    """Keep a ratio sub-band inside the theta band, warning if it wasn't.

    Two reasons it has to be inside. The interpolated grid only spans the theta
    band, so a sub-band outside it has no data at all; and under the "bandpass"
    estimator everything beyond theta band +/- the margin has been filtered away,
    so a band on the rolloff reports attenuated power as if it were signal.
    """
    low = max(float(band[0]), float(theta_band[0]))
    high = min(float(band[1]), float(theta_band[1]))
    if (low, high) != (float(band[0]), float(band[1])):
        print(f"WARNING: theta ratio {label} band {tuple(band)} reaches outside the "
              f"theta band {tuple(theta_band)}; clamped to ({low}, {high}).")
    return (low, high) if high > low else (float(theta_band[0]), float(theta_band[1]))


@lru_cache(maxsize=8)
def compute_theta_channels(
    pl2_path_str,
    analog_signal_index,
    channel_index,
    duration_sec,
    max_frequency,
    window_duration,
    step_duration,
    c_parameter,
    theta_low,
    theta_high,
    interp_step,
    smooth_width,
    estimator="argmax",
    ratio_low_band=config.DEFAULT_THETA_RATIO_LOW_BAND,
    ratio_high_band=config.DEFAULT_THETA_RATIO_HIGH_BAND,
):
    """Peak-theta-frequency, theta-band-power and theta-ratio series for one channel.

    Returns ``(times, theta_peak_hz, theta_power_db, theta_ratio)``.

    Two estimators:

    ``argmax``  the original — largest power in the band. Fast, but an argmax over
                a bounded interval must return something, so when delta's flank
                slopes up through the band it pins to the 4 Hz edge and reports a
                peak that isn't there (2-15% of bins depending on channel).

    Keys on pl2_path_str (+ params) for lru_cache hashability, mirroring
    ``compute_spectrogram``.
    """
    # The "bandpass" estimator filters the LFP before the spectrogram, so the
    # contaminating energy is gone before any peak-finding happens and a plain
    # argmax suffices. Edges sit a margin OUTSIDE the analysis band, so changing
    # the band moves the filter with it and they can't be set inconsistently.
    bp_low = bp_high = None
    if estimator == "bandpass":
        margin = config.DEFAULT_THETA_BANDPASS_MARGIN_HZ
        bp_low, bp_high = theta_low - margin, theta_high + margin

    freqs, times, power_db = compute_spectrogram(
        pl2_path_str,
        analog_signal_index,
        channel_index,
        0.0,
        duration_sec,
        max_frequency,
        window_duration,
        step_duration,
        c_parameter,
        bp_low,
        bp_high,
        config.DEFAULT_THETA_BANDPASS_ORDER,
    )
    # The spectrogram is stored in dB; peak/power work on linear power. Smooth in
    # time first so the argmax peak doesn't hop between adjacent frequency bins.
    power = spectral_utils.db_to_linear(power_db)
    power = spectral_utils.smooth_time(power, config.THETA_SPECT_TIME_SMOOTH_WIDTH)

    band = (theta_low, theta_high)
    peak = spectral_utils.theta_peak_frequency(freqs, power, band, interp_step)

    band_power = spectral_utils.theta_band_power(freqs, power, band)
    power_db_series = 10.0 * np.log10(band_power + 1e-12)

    # Ratio takes the *interpolated* band — its sub-bands are narrower than the
    # spectrogram's frequency resolution, so on the raw grid the edges snap to
    # bins (see spectral_utils.band_power_ratio). Band power above keeps the raw
    # grid: an 8 Hz-wide band over ~12 bins has no such problem.
    fine_freqs, fine_power = spectral_utils.interpolate_band(
        freqs, power, band, interp_step)
    if fine_freqs is None:
        ratio = np.full(len(times), np.nan)
    else:
        ratio, _, _ = spectral_utils.band_power_ratio(
            fine_freqs, fine_power,
            _clamp_band(ratio_low_band, band, "low"),
            _clamp_band(ratio_high_band, band, "high"),
        )

    peak = spectral_utils.smooth_series(peak, smooth_width)
    power_db_series = spectral_utils.smooth_series(power_db_series, smooth_width)
    ratio = spectral_utils.smooth_series(ratio, smooth_width)
    return times, peak, power_db_series, ratio


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

def load_session_from_paths(neural_path_str, behavior_path_str):
    """Load a Session directly from filesystem paths. No file copying.

    Parameters
    ----------
    neural_path_str : str — path to .pl2 file, or empty string / None
    behavior_path_str : str — path to .xlsx file, or empty string / None

    Returns
    -------
    Session
    """
    block = channel_names = pl2_path = None
    if neural_path_str:
        pl2_path = Path(neural_path_str)
        block, channel_names = _cached_load_block(str(pl2_path))

    behavior_metadata = behavior_data = behavior_path = None
    if behavior_path_str:
        behavior_path = Path(behavior_path_str)
        behavior_metadata, behavior_data = _cached_load_behavior(str(behavior_path))

    return Session(
        pl2_path=pl2_path,
        block=block,
        channel_names=channel_names,
        behavior_path=behavior_path,
        behavior_metadata=behavior_metadata,
        behavior_data=behavior_data,
    )
