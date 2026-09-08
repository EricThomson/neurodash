"""Building the exported analysis table.

One table, one time base. Neural-derived channels (theta peak/power) define the
grid — they come off the spectrogram bins — and the behavioral channels are
averaged onto it. Every variable is stored exactly once; nothing is written in
two time bases.
"""

import numpy as np
import pandas as pd

from neurodash import config
from neurodash import epochs, freezeframe_io
from neurodash.behavior_io import (
    behavior_time, extract_position, behavior_format, FREEZEFRAME)
from neurodash.session import compute_theta_channels
from neurodash.timebase import window_average


def analysis_time_base(session):
    """Return "spectral" or "behavior" — which grid the analysis table will use.

    The spectral grid wins whenever neural data is loaded; behavior-only sessions
    have no spectrogram to define bins, so they keep their native sample times.
    """
    return "spectral" if session.has_neural else "behavior"


def _behavior_columns(session, times, step):
    """Behavioral channels for the export, on ``times``.

    Averaged over the **bin spacing** (`step`, 0.1 s → ~3 EthoVision samples), not
    the spectrogram's 1.5 s window. Matching theta's 1.5 s support was tried and
    rejected: it drags behavior down to theta's information rate, losing ~45% of
    peak amplitude (r 0.86 vs raw, and single-frame events smear into 3 s plateaus)
    — see sandbox/probe_behavior_binning.py. At 0.1 s the behavior stays faithful
    (r 0.98, ~85% of peaks) while still landing on one grid.

    All four are continuous, so a plain mean is right for each; a binary state
    (freezing) would need a different operator.

    ``times`` None means "no spectral grid" — behavior keeps its own sample times.
    """
    if not session.has_behavior:
        return {}

    data = session.behavior_data
    if behavior_format(data) == FREEZEFRAME:
        return _freezeframe_columns(session, times, step)
    t_behav, x, y = extract_position(data, point="center")
    velocity = data["Velocity"].to_numpy(dtype=float)
    mobility = (data["Mobility"].to_numpy(dtype=float) if "Mobility" in data.columns
                else np.full(len(t_behav), np.nan))

    if times is None:
        return {"velocity": velocity, "mobility": mobility, "x": x, "y": y}
    return {
        "velocity": window_average(t_behav, velocity, times, step),
        "mobility": window_average(t_behav, mobility, times, step),
        "x": window_average(t_behav, x, times, step),
        "y": window_average(t_behav, y, times, step),
    }


def _freezeframe_columns(session, times, step):
    """Behavioral columns for a fear-conditioning session.

    `motion` averages onto the grid like velocity does. `freezing` must not:
    averaging a 0/1 state gives a fraction, not a state, and it blurs exactly the
    bout edges the rig's minimum-duration criterion exists to define. Bouts are
    >= 1 s against 0.1 s bins, so sampling the state at each bin loses only the
    sub-bin edge.

    `epoch` is a label, so it lands by interval containment. Blank is a real value
    meaning a bin no epoch covers — the guard bands are deliberate, and roughly a
    third of the session falls in them. `tone` and `shock` are the stimuli
    themselves as 0/1, which is what the notebooks exported alongside the epoch.
    """
    data = session.behavior_data
    t_behav = behavior_time(data)
    motion = data[freezeframe_io.MOTION_COLUMN].to_numpy(dtype=float)
    frozen = freezeframe_io.freezing_state(data, session.behavior_metadata)

    if times is None:
        columns = {"motion": motion, "freezing": frozen.astype(int)}
        grid = t_behav
    else:
        nearest = np.clip(np.searchsorted(t_behav, times), 0, len(t_behav) - 1)
        columns = {
            "motion": window_average(t_behav, motion, times, step),
            "freezing": frozen[nearest].astype(int),
        }
        grid = times

    trial = session.trial_events
    if trial["tones"] is not None:
        params = session.epoch_params
        built = epochs.build_epochs(trial["tones"], trial["shocks"], params,
                                    float(t_behav[-1]) if len(t_behav) else None)
        spans = session.event_spans(params)
        columns["epoch"] = epochs.epoch_labels(grid, built)
        columns["tone"] = epochs.event_mask(grid, spans, "tone_event")
        columns["shock"] = epochs.event_mask(grid, spans, "shock_event")
    return columns


def build_analysis_table(session, animal, channel_indices, band, spect_params,
                         session_name="", estimator=None,
                         ratio_low_band=None, ratio_high_band=None):
    """Merged per-time-bin table for CSV export — one row per bin, channels in columns.

    Columns: animal, session, time, then the behavior columns (velocity, mobility,
    x, y — or motion, freezing, epoch, tone, shock), then theta_peak_<CH> for each
    saved channel, then theta_power_<CH>, then theta_ratio_<CH>.

    Grouped by variable rather than by channel so "select these and overlay" is one
    contiguous block of columns. Units are in the header, not the names.

    Behavior sits to the LEFT of the spectral block because it is a fixed set of
    columns while the spectral block grows with the channel count. Put it on the
    right and `motion` lands in a different column in every export; on the left it
    is always in the same place and the spectral columns grow into empty space.

    **Wide, not long** (a `channel` column with one row per channel per bin): a
    channel is a separate signal, not a level of a factor — long asks you to think
    about the data in a shape neurophysiologists don't use. Wide is also the only
    shape that lets you compare channels against each other *within* a time bin
    without pivoting. The cost is that merging animals which ticked different
    channels leaves those columns NaN for the animals that didn't save them —
    honest, but ragged. It disappears if the same channels are saved across animals.

    The suffix is applied even for a single channel, so a 1-channel and a 4-channel
    export still stack: an unsuffixed ``theta_peak`` beside a suffixed one would be
    two columns meaning the same thing.

    Note ``theta_power`` is dB with an arbitrary offset (amplifier gain / units), so
    only differences are meaningful — ~3 dB is 2x. It is NOT comparable across
    channels as an absolute level; measured medians span 11.4 dB within one
    recording, mostly electrode impedance.

    Every channel lands on the same grid (same spectrogram params), so the
    behavioral columns are computed once rather than re-binned per channel.

    Parameters
    ----------
    session : Session
    animal : str — animal ID, same for every row so tables concat across animals.
    session_name : str — session label ('hab1'), same for every row. A column, not
        just a header field, because merging day 1 with day 2 needs it to tell the
        rows apart — and `time` is seconds *into a session*, so the same value
        recurs across days and grouping on time alone would silently pool them.
    channel_indices : sequence of int — channels to export theta for. A bare int is
        accepted for convenience and treated as a single-channel list.
    band : (float, float) — theta band low/high in Hz.
    spect_params : dict — window/step/c/max_freq; missing keys fall back to config.
    estimator : "argmax" | "bandpass" | None — which theta-peak estimator to export.
        Passed from the UI so the CSV matches what is on screen; these must not be
        allowed to diverge.
    ratio_low_band, ratio_high_band : (float, float) or None — theta-ratio sub-bands,
        likewise from the UI. Default to config's James bands.

    Returns
    -------
    pd.DataFrame
    """
    window = spect_params.get("window") or config.DEFAULT_SPECT_WINDOW_SEC
    step = spect_params.get("step") or config.DEFAULT_SPECT_STEP_SEC
    c_param = spect_params.get("c") or config.DEFAULT_SPECT_C_PARAM
    max_freq = spect_params.get("max_freq") or config.DEFAULT_SPECT_MAX_FREQ

    if not session.has_neural:
        # No spectrogram, so no spectral grid and no theta columns at all.
        times = behavior_time(session.behavior_data)
        return pd.DataFrame({"animal": animal, "session": session_name,
                             "time": times,
                             **_behavior_columns(session, None, step)})

    if isinstance(channel_indices, (int, np.integer)):
        channel_indices = [int(channel_indices)]
    channel_indices = list(channel_indices) or [0]

    sig_info = session.lfp_info
    labels = sig_info["channel_labels"]

    # No units in column names — they'd only compete with the channel suffix for
    # room. The units are recorded once in the CSV's `column_units` header line,
    # which matters most for theta_power: it's dB with an arbitrary offset, so only
    # differences between values mean anything.
    times = behavior = None
    per_channel = {"theta_peak": {}, "theta_power": {}, "theta_ratio": {}}
    for channel_index in channel_indices:
        times, peak, power, ratio = compute_theta_channels(
            str(session.pl2_path), session.lfp_signal_index, channel_index,
            sig_info["duration_sec"],
            max_freq, window, step, c_param,
            band[0], band[1],
            config.DEFAULT_THETA_INTERP_STEP_HZ, config.DEFAULT_THETA_SMOOTH_WIDTH,
            estimator or config.DEFAULT_THETA_ESTIMATOR,
            tuple(ratio_low_band or config.DEFAULT_THETA_RATIO_LOW_BAND),
            tuple(ratio_high_band or config.DEFAULT_THETA_RATIO_HIGH_BAND),
            time_offset=session.neural_time_offset,
        )
        if behavior is None:
            behavior = _behavior_columns(session, times, step)
        label = labels[channel_index]
        per_channel["theta_peak"][label] = peak
        per_channel["theta_power"][label] = power
        per_channel["theta_ratio"][label] = ratio

    # Behavior BEFORE the spectral block, deliberately. The behavior columns are
    # a fixed set, so putting them first pins them to the same position in every
    # export; the spectral block is the part that grows with channel count, and
    # it grows rightward into empty space instead of shoving motion/freezing
    # further right in every wider file.
    columns = {"animal": animal, "session": session_name, "time": times}
    columns.update(behavior)
    for variable, by_label in per_channel.items():
        for label, values in by_label.items():
            columns[f"{variable}_{label}"] = values
    return pd.DataFrame(columns)
