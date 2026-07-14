"""Per-channel Channel QC annotations — sidecar persistence and CSV export.

QC annotations (a quality rating and comment per LFP channel, plus one exemplar
channel per recording) live in a JSON sidecar written next to the .pl2 file as
``<stem>.qc.json``. They reload on the next open and feed the CSV export.

Uses the same json.dump / json.load idiom as the pyqtdash handoff in callbacks.py.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1


def qc_sidecar_path(pl2_path):
    """Return the QC sidecar path for a .pl2 file: <stem>.qc.json beside it."""
    pl2_path = Path(pl2_path)
    return pl2_path.parent / (pl2_path.stem + ".qc.json")


def _empty_channel_entries(session):
    """Build {index_str: {label, quality, comment}} for every channel."""
    sig_info = session.analog_signal_summaries[0]
    return {
        str(idx): {"label": lbl, "quality": "", "comment": ""}
        for idx, lbl in zip(sig_info["channel_indices"], sig_info["channel_labels"])
    }


def default_qc(session):
    """Return a fresh QC dict — all channels unrated, no exemplar."""
    return {
        "schema_version": SCHEMA_VERSION,
        "pl2_filename": Path(session.pl2_path).name if session.pl2_path else "",
        "exemplar_channel_index": None,
        "channels": _empty_channel_entries(session),
        "updated_at": None,
    }


def load_qc(pl2_path, session=None):
    """Load the QC sidecar for a recording, or a default if none exists.

    When ``session`` is given, saved values are merged onto a fresh per-channel
    skeleton (missing channels filled in, labels refreshed from the session) so
    callers never special-case a missing or stale file.
    """
    path = qc_sidecar_path(pl2_path)
    data = None
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = None

    if session is None:
        # No session to reconcile against — return the file as-is, or a stub.
        return data if data is not None else {
            "schema_version": SCHEMA_VERSION,
            "pl2_filename": Path(pl2_path).name,
            "exemplar_channel_index": None,
            "channels": {},
            "updated_at": None,
        }

    base = default_qc(session)
    if data is None:
        return base

    base["exemplar_channel_index"] = data.get("exemplar_channel_index")
    base["updated_at"] = data.get("updated_at")
    saved_channels = data.get("channels", {})
    for key, entry in base["channels"].items():
        saved = saved_channels.get(key)
        if saved:
            entry["quality"] = saved.get("quality", "")
            entry["comment"] = saved.get("comment", "")
    return base


def save_qc(pl2_path, qc_data):
    """Write the QC dict to the sidecar next to the .pl2 (whole file)."""
    qc_data = dict(qc_data)
    qc_data["schema_version"] = SCHEMA_VERSION
    qc_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(qc_sidecar_path(pl2_path), "w") as f:
        json.dump(qc_data, f, indent=2)


def qc_to_dataframe(qc_data):
    """Flatten QC data to one row per channel for CSV export."""
    exemplar = qc_data.get("exemplar_channel_index")
    rows = [
        {
            "channel_index": int(key),
            "channel_label": entry.get("label", ""),
            "quality": entry.get("quality", ""),
            "comment": entry.get("comment", ""),
            "is_exemplar": int(key) == exemplar,
        }
        for key, entry in sorted(qc_data.get("channels", {}).items(),
                                 key=lambda kv: int(kv[0]))
    ]
    return pd.DataFrame(
        rows,
        columns=["channel_index", "channel_label", "quality", "comment", "is_exemplar"],
    )
