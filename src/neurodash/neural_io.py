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


def load_events(pl2_path):
    """TTL event times from a .pl2, in pl2 clock seconds.

    Returns ``{channel_name: np.ndarray}`` for channels that actually carry
    events; the empty ones (a pl2 declares 40-odd whether used or not) are left
    out, so the result doubles as "what event structure does this file have".

    That structure is how the app tells acquisition/tone/context apart — tones
    plus shocks, tones alone, or neither — rather than trusting a label. On the
    acquisition file: EVT01 = 5 tone onsets, EVT02 = 5 shocks (each 39.994 s
    after its tone), EVT03 = a single stop marker.

    Read with the raw reader rather than through the Block: event channels are
    cheap to read on their own, and the Block path costs a full signal load.
    """
    reader = neo.io.Plexon2IO(filename=str(pl2_path))
    reader.parse_header()
    events = {}
    for i, channel in enumerate(reader.header["event_channels"]):
        timestamps, _durations, _labels = reader.get_event_timestamps(
            block_index=0, seg_index=0, event_channel_index=i)
        if timestamps is None or len(timestamps) == 0:
            continue
        events[str(channel["name"])] = np.asarray(
            reader.rescale_event_timestamp(timestamps, dtype="float64"),
            dtype=float)
    return events


def segment_bounds(pl2_path):
    """(t_start, t_stop) of the recording segment, in pl2 clock seconds.

    ``t_start`` is where the analog samples begin — non-zero in every real file
    (4343.83 for the open-field test file, 5545.32 for acquisition), because the
    Plexon clock runs from when the software was launched, not when recording
    started.
    """
    reader = neo.io.Plexon2IO(filename=str(pl2_path))
    reader.parse_header()
    return (float(reader.segment_t_start(0, 0)), float(reader.segment_t_stop(0, 0)))


def analog_fragments(pl2_path, channel_name):
    """Contiguous runs of samples in one analog channel, in acquisition order.

    A .pl2 analog channel is not necessarily one continuous recording — OmniPlex
    writes it as *fragments*, and pausing acquisition leaves a gap between them.
    neo concatenates the fragments into a single array, so sample index maps to
    time only within a fragment; after a gap, every sample is reported that much
    too early.

    This is not hypothetical. The acquisition test file has two fragments —
    113 samples, then a **33.906 s gap**, then 1,290,004 samples — so neo places
    99.99% of the recording 33.9 s before it actually happened, which is what put
    every LFP trace a third of a minute away from its own shock artifacts.

    Returns a list of {"start_s", "n_samples"}, where start_s is seconds from the
    start of the recording (add segment t_start for pl2 clock). Empty when the
    fragment table can't be read, in which case callers should assume contiguity.
    """
    try:
        reader = neo.io.Plexon2IO(filename=str(pl2_path))
        reader.parse_header()
        pl2 = reader.pl2reader
        timestamps, counts, _values = pl2.pl2_get_analog_channel_data_by_name(
            channel_name)
        frequency = float(pl2.pl2_file_info.m_TimestampFrequency)
    except Exception as e:
        print(f"WARNING: could not read analog fragments for {channel_name}: {e}")
        return []
    return [{"start_s": int(ts) / frequency, "n_samples": int(n)}
            for ts, n in zip(timestamps, counts) if int(n) > 0]


def fragment_gap_warning(fragments, tolerance_s=1.0):
    """Message when a channel's fragments can't be placed by one shared offset.

    A single offset is exact only when all the real data sits in one fragment.
    Two substantial fragments either side of a gap need different offsets, and
    nothing downstream could detect the resulting misalignment, so it is said out
    loud rather than silently mis-drawn.
    """
    substantial = [f for f in fragments if f["n_samples"] / 1000.0 > tolerance_s]
    if len(substantial) < 2:
        return ""
    gaps = [b["start_s"] - (a["start_s"] + a["n_samples"] / 1000.0)
            for a, b in zip(substantial, substantial[1:])]
    return (f"This channel has {len(substantial)} separate recording fragments "
            f"(gaps of {', '.join(f'{g:.1f}' for g in gaps)} s). Times after the "
            f"first gap cannot be trusted — one offset cannot place them all.")


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


def extract_time_window(sig, channel_index, start_time_sec, duration_sec,
                        time_offset=0.0):
    """Extract (t, y) for one channel from a Neo AnalogSignal.

    ``t`` is seconds from the first sample, plus ``time_offset``. Note it is
    deliberately NOT the pl2 clock: ``sig.t_start`` is non-zero in every real file
    (4343.83 for the open-field test file) because the Plexon clock runs from when
    the software launched, and sample-relative time is what every panel wants.
    ``time_offset`` is what puts it on the behavior clock — see
    alignment.neural_time_offset, which is 0 whenever the two already agree.
    """
    sr = float(sig.sampling_rate.rescale("Hz").magnitude)
    n_samples = sig.shape[0]
    start_idx = int(round(start_time_sec * sr))
    end_idx = int(round((start_time_sec + duration_sec) * sr))
    start_idx = max(0, min(start_idx, n_samples))
    end_idx = max(start_idx, min(end_idx, n_samples))

    y = np.asarray(sig[start_idx:end_idx, channel_index]).squeeze()
    t = np.arange(start_idx, end_idx) / sr + time_offset
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
        tt, y, _ = extract_time_window(sig, ch, 0, full_duration,
                                       session.neural_time_offset)
        if t is None:
            t = tt
        columns[labels[ch]] = y

    return pd.DataFrame({"animal": animal, "time": t, **columns})
