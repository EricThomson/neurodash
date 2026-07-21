"""Per-channel Channel Viewer annotations — sidecar persistence and CSV export.

Channel annotations (a quality rating and comment per LFP channel, plus one
exemplar channel per recording) live in a JSON sidecar written next to the .pl2
file as ``<stem>.channels.json``. They reload on the next open and feed the CSV
export.

Uses the same json.dump / json.load idiom as the pyqtdash handoff in callbacks.py.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 2


def channel_sidecar_path(pl2_path):
    """Return the sidecar path for a .pl2 file: <stem>.channels.json beside it."""
    pl2_path = Path(pl2_path)
    return pl2_path.parent / (pl2_path.stem + ".channels.json")


def parse_animal_id(pl2_path):
    """Best-effort animal ID from a pl2 filename.

    Takes the last '_'-delimited token of the stem, e.g.
    '170505_open_field_theta_FC33-4' -> 'FC33-4'. Single-animal only; returns
    '' when no path is given. The value is editable + persisted in the sidecar,
    so a bad guess is correctable in the Channel Viewer.
    """
    if not pl2_path:
        return ""
    stem = Path(pl2_path).stem
    return stem.split("_")[-1] if stem else ""


def _empty_channel_entries(session):
    """Build {index_str: {label, quality, comment, include}} for every channel."""
    sig_info = session.analog_signal_summaries[0]
    return {
        str(idx): {"label": lbl, "quality": "", "comment": "", "include": True}
        for idx, lbl in zip(sig_info["channel_indices"], sig_info["channel_labels"])
    }


def default_channels(session):
    """Return a fresh channel record — all channels unrated, no exemplar."""
    return {
        "schema_version": SCHEMA_VERSION,
        "pl2_filename": Path(session.pl2_path).name if session.pl2_path else "",
        "animal": parse_animal_id(session.pl2_path),
        "comment": "",  # free-text note on the whole channel review
        "exemplar_channel_index": None,
        "channels": _empty_channel_entries(session),
        "updated_at": None,
    }


def load_channels(pl2_path, session=None):
    """Load the channel sidecar for a recording, or a default if none exists.

    When ``session`` is given, saved values are merged onto a fresh per-channel
    skeleton (missing channels filled in, labels refreshed from the session) so
    callers never special-case a missing or stale file.
    """
    path = channel_sidecar_path(pl2_path)
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
            "animal": parse_animal_id(pl2_path),
            "exemplar_channel_index": None,
            "channels": {},
            "updated_at": None,
        }

    base = default_channels(session)
    if data is None:
        return base

    # animal is read-only and always re-derived from the filename (base already
    # holds the fresh parse); a stale saved value is intentionally ignored.
    base["comment"] = data.get("comment", "")
    base["exemplar_channel_index"] = data.get("exemplar_channel_index")
    base["updated_at"] = data.get("updated_at")
    saved_channels = data.get("channels", {})
    for key, entry in base["channels"].items():
        saved = saved_channels.get(key)
        if saved:
            entry["quality"] = saved.get("quality", "")
            entry["comment"] = saved.get("comment", "")
            entry["include"] = saved.get("include", True)
    return base


def save_channels(pl2_path, channel_data):
    """Write the channel record to the sidecar next to the .pl2 (whole file)."""
    channel_data = dict(channel_data)
    channel_data["schema_version"] = SCHEMA_VERSION
    channel_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(channel_sidecar_path(pl2_path), "w") as f:
        json.dump(channel_data, f, indent=2)


def channels_to_dataframe(channel_data):
    """Flatten channel annotations to a tidy per-channel table.

    One row per exported channel: animal, channel, quality, comment, exemplar.
    Not currently wired into an export (the single-animal neural CSV carries this
    metadata in its `#` header instead) — retained for the planned multi-animal
    combine step, where a cross-animal QC table (plain concat on `animal`) is the
    natural JMP artifact.
    """
    animal = channel_data.get("animal", "")
    exemplar = channel_data.get("exemplar_channel_index")
    rows = [
        {
            "animal": animal,
            "channel": entry.get("label", ""),
            "quality": entry.get("quality", ""),
            "comment": entry.get("comment", ""),
            "exemplar": int(key) == exemplar,
        }
        for key, entry in sorted(channel_data.get("channels", {}).items(),
                                 key=lambda kv: int(kv[0]))
        if entry.get("include", True)  # only channels the user kept checked
    ]
    return pd.DataFrame(
        rows,
        columns=["animal", "channel", "quality", "comment", "exemplar"],
    )
