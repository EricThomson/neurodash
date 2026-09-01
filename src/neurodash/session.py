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
from neurodash import alignment
from neurodash.neural_io import (
    load_pl2_block, list_analog_signal_summaries, get_analog_signal,
    extract_time_window, select_lfp_signal_index, detect_channel_banks,
    load_events, segment_bounds,
)
from neurodash.spectral_utils import compute_multitaper_spectrogram
from neurodash.behavior_io import load_behavior_file
from neurodash.channel_io import load_identity, resolve_animal_id


class Session:
    """Container for one recording session.

    At least one of neural or behavioral data must be loaded.
    Check has_neural / has_behavior before accessing those attributes.
    """

    def __init__(self,
                 pl2_path=None, block=None, channel_names=None, stream_info=None,
                 behavior_path=None, behavior_metadata=None, behavior_data=None,
                 bank_index=None, events=None, segment=None):
        # Neural
        self.pl2_path = pl2_path
        self.block = block
        self.analog_signal_summaries = (
            list_analog_signal_summaries(block, channel_names, stream_info)
            if block else []
        )
        self.rec_datetime = (
            block.annotations.get("m_CreatorDateTime", None) if block else None
        )
        # Which analog stream is the LFP. A property of the file, not a UI choice,
        # so it is resolved once here rather than passed around as a control.
        self.lfp_signal_index = (
            select_lfp_signal_index(self.analog_signal_summaries)
            if self.analog_signal_summaries else 0
        )
        # Which subject's channels this session is about, when the file holds more
        # than one animal. None means "not chosen yet" and the caller should ask.
        self.bank_index = bank_index
        # TTL events and the segment's pl2-clock bounds, for aligning to behavior.
        self.events = events or {}
        self.segment = segment

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

    @property
    def lfp_info(self):
        """Summary dict for the LFP stream — what callers used to spell ``[0]``."""
        if not self.analog_signal_summaries:
            return None
        return self.analog_signal_summaries[self.lfp_signal_index]

    @property
    def channel_banks(self):
        """Per-subject channel groups in the LFP stream (one entry when single-animal)."""
        info = self.lfp_info
        return detect_channel_banks(info["channel_labels"]) if info else []

    @property
    def is_multi_animal(self):
        return len(self.channel_banks) > 1

    def channel_indices(self):
        """Channel indices this session may show or export.

        The whole stream for a single-animal file; only the selected bank's
        channels when the file holds two animals. Filtering here — rather than
        warning after the fact — is what keeps the other animal's ephys out of the
        channel pickers entirely, so it cannot be plotted or exported by mistake.

        Returns an empty list when a multi-animal file has no bank chosen yet, so
        the UI shows nothing rather than defaulting to somebody's data.
        """
        banks = self.channel_banks
        if not banks:
            return []
        if len(banks) == 1:
            return list(banks[0]["indices"])
        if self.bank_index is None or not (0 <= self.bank_index < len(banks)):
            return []
        return list(banks[self.bank_index]["indices"])

    @property
    def neural_time_offset(self):
        """Seconds to add to neural sample time to get behavior time.

        0 for open field, where the two recordings were started together, and 0
        for a .pl2 loaded on its own — there is nothing to align to yet, so the
        neural axis stays in sample time until a behavior file gives it an anchor.
        -33.047 on the acquisition test file.
        """
        return alignment.neural_time_offset(self.events, self.behavior_metadata,
                                            self.segment)

    @property
    def trial_events(self):
        """Tone/shock times on the behavior clock, plus the measured trace interval.

        Empty values for a session with no TTLs, which is every open-field file
        and any fear session whose events weren't recorded.
        """
        return alignment.trial_structure(
            self.events,
            alignment.behavior_start_in_pl2(self.events, self.behavior_metadata,
                                            self.segment))

    def check_alignment(self):
        """(ok, message) from re-deriving the offset off the animal's startle."""
        return alignment.check_alignment(self.events, self.behavior_metadata,
                                         self.behavior_data, self.segment)

    @property
    def animal_id(self):
        """This session's animal ID: the saved override, else what can be inferred.

        The single answer every panel and every export uses, which is the point —
        the Channel Viewer and the neural CSV used to take the *inferred* ID while
        the Session Info panel and the analysis CSV took the *override*, so one
        recording could export as `G20_3` in one file and `G20-3` in the other.
        That is exactly the split into two JMP groups that canonical_id exists to
        prevent, and it was invisible until the two were compared side by side.

        Blank when nothing can identify the animal yet: on a .pl2 holding two
        animals the filename names both, so it can't identify either until the
        behavior file says (see resolve_animal_id's filename_fallback).
        """
        if self.pl2_path:
            override = load_identity(self.pl2_path)["animal"]
            if override:
                return override
        return resolve_animal_id(self.pl2_path, self.behavior_metadata,
                                 filename_fallback=not self.is_multi_animal)

    def channel_options(self):
        """(index, label) pairs for the channels this session may show or export.

        The single place the UI asks "which channels exist?", so bank filtering
        happens once and every picker inherits it.
        """
        info = self.lfp_info
        if not info:
            return []
        labels = info["channel_labels"]
        return [(i, labels[i]) for i in self.channel_indices()]


# ---------------------------------------------------------------------------
# Cached loaders — load directly from path, no copying
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _cached_load_block(pl2_path_str):
    block, channel_names, stream_info = load_pl2_block(Path(pl2_path_str))
    return block, channel_names, stream_info


@lru_cache(maxsize=4)
def _cached_load_events(pl2_path_str):
    """TTL events and segment bounds — a cheap header read, cached like the block."""
    path = Path(pl2_path_str)
    return load_events(path), segment_bounds(path)


@lru_cache(maxsize=4)
def _cached_load_behavior(behavior_path_str):
    metadata, data = load_behavior_file(Path(behavior_path_str))
    return metadata, data


# maxsize covers a full 8-channel analysis export plus the channel on screen. At 8
# an export exactly filled the cache and evicted the displayed channel, so the
# Session Viewer figure recomputed from cold right after every export. ~2.3 MB each.
@lru_cache(maxsize=16)
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
    time_offset=0.0,
):
    """Compute and cache spectrogram by pl2_path + parameters.

    ``bandpass_low``/``bandpass_high`` optionally filter the LFP *before* the
    spectrogram — used by the "bandpass" theta estimator. They are part of the cache
    key, so filtered and unfiltered spectrograms coexist rather than evicting each
    other when you flip estimators.

    ``time_offset`` shifts the returned times onto the behavior clock and is part
    of the cache key, so a session viewed before and after its behavior file is
    loaded doesn't reuse the wrong axis. It is 0 for open field.

    Keys on pl2_path_str for hashability (lru_cache requirement).
    """
    block, _, _ = _cached_load_block(pl2_path_str)
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
    return freqs, times + time_offset, power_db


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


@lru_cache(maxsize=16)   # same reason as compute_spectrogram: an 8-channel export
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
    time_offset=0.0,
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
        time_offset,
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

def load_session_from_paths(neural_path_str, behavior_path_str, bank_index=None):
    """Load a Session directly from filesystem paths. No file copying.

    Parameters
    ----------
    neural_path_str : str — path to .pl2 file, or empty string / None
    behavior_path_str : str — path to .xlsx file, or empty string / None
    bank_index : int or None — which subject's channels, for a pl2 holding two
        animals. None on a single-animal file means "the only bank"; on a
        multi-animal file it means "not chosen yet" and no channels are offered.
        Left None, the saved choice beside the .pl2 is used — the bank is a fact
        about the recording, like its annotations, so every caller gets the right
        subject without having to pass it through.

    Returns
    -------
    Session
    """
    block = channel_names = stream_info = pl2_path = None
    events = segment = None
    if neural_path_str:
        pl2_path = Path(neural_path_str)
        block, channel_names, stream_info = _cached_load_block(str(pl2_path))
        events, segment = _cached_load_events(str(pl2_path))
        if bank_index is None:
            bank_index = load_identity(pl2_path)["bank"]

    behavior_metadata = behavior_data = behavior_path = None
    if behavior_path_str:
        behavior_path = Path(behavior_path_str)
        behavior_metadata, behavior_data = _cached_load_behavior(str(behavior_path))

    return Session(
        pl2_path=pl2_path,
        block=block,
        channel_names=channel_names,
        stream_info=stream_info,
        behavior_path=behavior_path,
        behavior_metadata=behavior_metadata,
        behavior_data=behavior_data,
        bank_index=bank_index,
        events=events,
        segment=segment,
    )
