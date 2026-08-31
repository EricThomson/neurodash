"""Low-level I/O for Plexon .pl2 files via Neo.

Handles reading raw neural data from disk and extracting signal arrays.
The main entry points are:
- ``load_pl2_block``: read a .pl2 file using neo
- ``list_analog_signal_summaries``: inspect what analog signals are available
- ``select_lfp_signal_index``: which of those streams actually holds the LFP
- ``detect_channel_banks``: split a stream's channels into per-subject groups
- ``extract_time_window``: pull a (t, y) slice from a single channel
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import neo

from neurodash import config


def load_pl2_block(pl2_path):
    """Load a Plexon .pl2 file into a Neo Block.

    Returns (block, channel_names, stream_info), all indexed by analog signal:
    channel_names[i] is the list of Plexon channel name strings for
    analogsignals[i], and stream_info[i] is {"id", "name"} for its stream
    (e.g. {"id": "FP", "name": "FPl-Low Pass Filtered"}). All come from the same
    reader instantiation so no extra file reads are needed.

    The stream identity matters because a pl2 can carry several analog streams
    and only one of them is the LFP — see select_lfp_signal_index.
    """
    reader = neo.io.Plexon2IO(filename=str(pl2_path))
    block = reader.read_block(lazy=False)

    # Build channel name lists per analog signal from the reader header.
    # signal_streams[i] corresponds to analogsignals[i].
    # signal_channels rows carry the per-channel names and their stream_id.
    signal_streams = reader.header["signal_streams"]
    signal_channels = reader.header["signal_channels"]

    channel_names = []
    stream_info = []
    for stream in signal_streams:
        stream_id = stream["id"]
        names = [
            str(sc["name"])
            for sc in signal_channels
            if sc["stream_id"] == stream_id
        ]
        channel_names.append(names)
        stream_info.append({"id": str(stream_id), "name": str(stream["name"])})

    return block, channel_names, stream_info


def select_lfp_signal_index(summaries):
    """Index of the analog signal holding the LFP.

    A pl2 may carry more than one analog stream, and which one is the LFP depends
    on the recording rig — the lab's own schema records this as
    ``setup: original or digiamp``. The open-field files have a single
    ``AI`` (auxiliary input) stream and the LFP is in it. The acquisition files
    have both ``AI`` (32 ch) and ``FP`` (10 ch), and there the LFP is in ``FP``:
    "FPl-Low Pass Filtered" is Plexon's field-potential stream, and its channels
    measure ~10x the amplitude of the AI ones at the proper amplifier gain
    (3.05e-06 vs 1.53e-04 V/bit).

    This is not cosmetic. Every caller used to hardcode analog signal 0, which on
    an acquisition file is AI — so the app would have plotted 32 channels of
    auxiliary-input noise as LFP, computed theta on it and exported it, all
    silently and all looking like plausible noisy data.

    Preference order is config.LFP_STREAM_PREFERENCE; an unrecognized set of
    streams falls back to the first, which is the old behavior.
    """
    for preferred in config.LFP_STREAM_PREFERENCE:
        for summary in summaries:
            if summary.get("stream_id") == preferred:
                return summary["analog_signal_index"]
    return summaries[0]["analog_signal_index"] if summaries else 0


def _channel_number(label):
    """Trailing integer in a Plexon channel name ('FP17' -> 17), or None."""
    match = re.search(r"(\d+)\s*$", str(label))
    return int(match.group(1)) if match else None


def detect_channel_banks(channel_labels):
    """Split channel labels into contiguously-numbered banks.

    A pl2 can hold two animals recorded at once on separate headstages, which
    shows up as two runs of channel numbers with a gap between them — the
    acquisition test file is FP01-FP05 and FP17-FP21. Each bank is one subject,
    so the app can offer one animal's channels and keep the other's out of reach
    rather than trusting the user to tick the right five.

    The split is verifiable rather than conventional: on that file the mean
    correlation *across* banks is 0.001 (max |r| 0.008) while within-bank pairs
    run 0.28-0.99 — two electrically independent brains. See
    ``banks_look_independent`` for the load-time check.

    A single run (open field's AI17-AI21) returns one bank, so nothing changes
    for single-animal files. Labels with no trailing number can't be grouped and
    are returned as one bank.

    Returns a list of dicts: {"label", "labels", "indices"}, in input order.
    """
    numbers = [_channel_number(label) for label in channel_labels]
    if not channel_labels or any(n is None for n in numbers):
        return [{"label": "all channels",
                 "labels": list(channel_labels),
                 "indices": list(range(len(channel_labels)))}]

    banks = []
    for index, (label, number) in enumerate(zip(channel_labels, numbers)):
        if banks and number == banks[-1]["_last"] + 1:
            banks[-1]["labels"].append(label)
            banks[-1]["indices"].append(index)
            banks[-1]["_last"] = number
        else:
            banks.append({"labels": [label], "indices": [index], "_last": number})

    for bank in banks:
        bank.pop("_last")
        first, last = bank["labels"][0], bank["labels"][-1]
        bank["label"] = first if first == last else f"{first}-{last}"
    return banks


def banks_look_independent(signal_array, banks,
                           threshold=config.CHANNEL_BANK_INDEPENDENCE_THRESHOLD):
    """True when every pair of banks is uncorrelated, i.e. really separate subjects.

    Channel numbering is a convention and conventions get broken; correlation is
    physics. Two animals on separate headstages share no reference and no volume
    conduction, so across-bank correlation sits at the noise floor (measured 0.001
    mean, 0.008 max on the acquisition file). If a gap in the numbering does *not*
    correspond to a real subject boundary, the banks will be correlated and the
    caller should say so rather than silently splitting one animal in two.

    Returns True for fewer than two banks — nothing to check.
    """
    if len(banks) < 2:
        return True
    corr = np.corrcoef(np.asarray(signal_array).T)
    for i, bank_a in enumerate(banks):
        for bank_b in banks[i + 1:]:
            block = corr[np.ix_(bank_a["indices"], bank_b["indices"])]
            if np.nanmax(np.abs(block)) > threshold:
                return False
    return True


def list_analog_signal_summaries(block, channel_names=None, stream_info=None):
    """Return a list of analog signal summaries for the UI.

    Each summary is a dict with:
    - label: human readable label
    - n_channels, n_samples
    - sampling_rate_hz
    - duration_sec
    - channel_labels: list[str] for UI selection
    - channel_indices: list[int]
    - stream_id, stream_name: which Plexon stream this is ("FP", "AI")

    channel_names, if provided, should be a list of lists as returned by
    load_pl2_block — the Plexon channel name strings (e.g. "AI17") for each
    analog signal. Falls back to 1-based indices when absent.

    stream_info, likewise from load_pl2_block, identifies each stream so callers
    can pick the one holding the LFP rather than assuming signal 0.
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

        stream = (stream_info[i] if stream_info and i < len(stream_info)
                  else {"id": "", "name": ""})

        summaries.append(
            {
                "analog_signal_index": i,
                "stream_id": stream["id"],
                "stream_name": stream["name"],
                "label": stream["name"] or f"Analog signal {i}",
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
    sig = get_analog_signal(session.block, session.lfp_signal_index)
    sig_info = session.lfp_info
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
