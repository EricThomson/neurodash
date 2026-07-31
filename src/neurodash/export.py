"""Building the exported analysis table.

One table, one time base. Neural-derived channels (theta peak/power) define the
grid — they come off the spectrogram bins — and the behavioral channels are
averaged onto it. Every variable is stored exactly once; nothing is written in
two time bases.
"""

import numpy as np
import pandas as pd

from neurodash import config
from neurodash.behavior_io import extract_position
from neurodash.session import compute_theta_channels
from neurodash.timebase import window_average


def analysis_time_base(session):
    """Return "spectral" or "behavior" — which grid the analysis table will use.

    The spectral grid wins whenever neural data is loaded; behavior-only sessions
    have no spectrogram to define bins, so they keep their native sample times.
    """
    return "spectral" if session.has_neural else "behavior"


def build_analysis_table(session, animal, channel_index, band, spect_params):
    """Merged per-time-bin table for CSV export.

    Columns: animal, time, theta_peak_hz, theta_power_db, velocity, mobility, x, y
    — whichever of those the session has data for. One row per spectral bin.

    Behavioral channels are averaged over the **bin spacing** (`step`, 0.1 s → ~3
    EthoVision samples), not the spectrogram's 1.5 s window. Matching theta's 1.5 s
    support was tried and rejected: it drags behavior down to theta's information
    rate, losing ~45% of peak amplitude (r 0.86 vs raw, and single-frame events
    smear into 3 s plateaus) — see sandbox/probe_behavior_binning.py. At 0.1 s the
    behavior stays faithful (r 0.98, ~85% of peaks) while still landing on one grid.

    All four behavioral channels are continuous, so a plain mean is right for each;
    a binary state (freezing) would need a different operator.

    Parameters
    ----------
    session : Session
    animal : str — animal ID, same for every row so tables concat across animals.
    channel_index : int — analysis channel the theta channels come from.
    band : (float, float) — theta band low/high in Hz.
    spect_params : dict — window/step/c/max_freq; missing keys fall back to config.

    Returns
    -------
    pd.DataFrame
    """
    window = spect_params.get("window") or config.DEFAULT_SPECT_WINDOW_SEC
    step = spect_params.get("step") or config.DEFAULT_SPECT_STEP_SEC
    c_param = spect_params.get("c") or config.DEFAULT_SPECT_C_PARAM
    max_freq = spect_params.get("max_freq") or config.DEFAULT_SPECT_MAX_FREQ

    columns = {}

    if session.has_neural:
        sig_info = session.analog_signal_summaries[0]
        times, peak, power, _height = compute_theta_channels(
            str(session.pl2_path), 0, channel_index, sig_info["duration_sec"],
            max_freq, window, step, c_param,
            band[0], band[1],
            config.DEFAULT_THETA_INTERP_STEP_HZ, config.DEFAULT_THETA_SMOOTH_WIDTH,
            config.DEFAULT_THETA_ESTIMATOR, config.DEFAULT_THETA_WINDOW_SEC,
        )
        columns["theta_peak_hz"] = peak
        columns["theta_power_db"] = power
    else:
        # No spectrogram, so no spectral grid — behavior keeps its own sample times
        # and nothing needs resampling.
        times = session.behavior_data["Recording time"].to_numpy(dtype=float)

    if session.has_behavior:
        data = session.behavior_data
        t_behav, x, y = extract_position(data, point="center")
        velocity = data["Velocity"].to_numpy(dtype=float)
        if "Mobility" in data.columns:
            mobility = data["Mobility"].to_numpy(dtype=float)
        else:
            mobility = np.full(len(t_behav), np.nan)

        if session.has_neural:
            columns["velocity"] = window_average(t_behav, velocity, times, step)
            columns["mobility"] = window_average(t_behav, mobility, times, step)
            columns["x"] = window_average(t_behav, x, times, step)
            columns["y"] = window_average(t_behav, y, times, step)
        else:
            columns["velocity"] = velocity
            columns["mobility"] = mobility
            columns["x"] = x
            columns["y"] = y

    return pd.DataFrame({"animal": animal, "time": times, **columns})
