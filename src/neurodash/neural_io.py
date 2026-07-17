"""Low-level I/O for Plexon .pl2 files via Neo.

Handles reading raw neural data from disk and extracting signal arrays.
The main entry points are:
- ``load_pl2_block``: read a .pl2 file using neo
- ``list_analog_signal_summaries``: inspect what analog signals are available
- ``extract_time_window``: pull a (t, y) slice from a single channel
"""

from pathlib import Path

import numpy as np
import pandas as pd
import neo


def load_pl2_block(pl2_path):
    """Load a Plexon .pl2 file into a Neo Block.

    Returns (block, channel_names) where channel_names is a list of lists:
    channel_names[i] is the list of Plexon channel name strings for analogsignals[i].
    Both come from the same reader instantiation so no extra file reads are needed.
    """
    reader = neo.io.Plexon2IO(filename=str(pl2_path))
    block = reader.read_block(lazy=False)

    # Build channel name lists per analog signal from the reader header.
    # signal_streams[i] corresponds to analogsignals[i].
    # signal_channels rows carry the per-channel names and their stream_id.
    signal_streams = reader.header["signal_streams"]
    signal_channels = reader.header["signal_channels"]

    channel_names = []
    for stream in signal_streams:
        stream_id = stream["id"]
        names = [
            str(sc["name"])
            for sc in signal_channels
            if sc["stream_id"] == stream_id
        ]
        channel_names.append(names)

    return block, channel_names


def list_analog_signal_summaries(block, channel_names=None):
    """Return a list of analog signal summaries for the UI.

    Each summary is a dict with:
    - label: human readable label
    - n_channels, n_samples
    - sampling_rate_hz
    - duration_sec
    - channel_labels: list[str] for UI selection
    - channel_indices: list[int]

    channel_names, if provided, should be a list of lists as returned by
    load_pl2_block — the Plexon channel name strings (e.g. "AI17") for each
    analog signal. Falls back to 1-based indices when absent.
    """
    summaries = []
    for i, sig in enumerate(block.segments[0].analogsignals):
        # Neo AnalogSignal: shape (n_samples, n_channels)
        n_samples, n_channels = sig.shape[0], sig.shape[1]
        sr_hz = float(sig.sampling_rate.rescale("Hz").magnitude)
        duration_sec = float(n_samples / sr_hz)

        # Prefer Plexon channel names from the reader header (e.g. "AI17").
        # Fall back to 1-based indices if not available.
        if channel_names and i < len(channel_names) and channel_names[i]:
            channel_labels = channel_names[i][:n_channels]
        else:
            channel_labels = [f"Ch {ch_idx + 1}" for ch_idx in range(n_channels)]

        summaries.append(
            {
                "analog_signal_index": i,
                "label": f"Analog signal {i}",
                "n_channels": int(n_channels),
                "n_samples": int(n_samples),
                "sampling_rate_hz": sr_hz,
                "duration_sec": duration_sec,
                "channel_labels": channel_labels,
                "channel_indices": list(range(n_channels)),
            }
        )
    return summaries


def get_analog_signal(block, analog_signal_index):
    return block.segments[0].analogsignals[analog_signal_index]


def extract_time_window(sig, channel_index, start_time_sec, duration_sec):
    """Extract (t, y) for one channel from a Neo AnalogSignal."""
    sr = float(sig.sampling_rate.rescale("Hz").magnitude)
    n_samples = sig.shape[0]
    start_idx = int(round(start_time_sec * sr))
    end_idx = int(round((start_time_sec + duration_sec) * sr))
    start_idx = max(0, min(start_idx, n_samples))
    end_idx = max(start_idx, min(end_idx, n_samples))

    y = np.asarray(sig[start_idx:end_idx, channel_index]).squeeze()
    t = np.arange(start_idx, end_idx) / sr
    return t, y, sr


def build_neural_table(session, channel_indices, animal):
    """Wide LFP table for CSV export: animal, time, one voltage column per channel.

    Full-resolution signal for each requested channel over the whole recording —
    the neural parallel to the behavioral table. `animal` is the same for every
    row. `channel_indices` should already be filtered to the channels the user
    chose to export (and is used in the given order for column order).

    Parameters
    ----------
    session : Session — must have neural data loaded.
    channel_indices : list[int] — 0-based channel indices to include.
    animal : str — animal ID.

    Returns
    -------
    pd.DataFrame
    """
    sig = get_analog_signal(session.block, 0)
    sig_info = session.analog_signal_summaries[0]
    labels = sig_info["channel_labels"]
    full_duration = sig_info["duration_sec"]

    t = None
    columns = {}
    for ch in channel_indices:
        tt, y, _ = extract_time_window(sig, ch, 0, full_duration)
        if t is None:
            t = tt
        columns[labels[ch]] = y

    return pd.DataFrame({"animal": animal, "time": t, **columns})
