"""Per-channel Channel Viewer annotations — sidecar persistence and CSV export.

Channel annotations (a quality rating and comment per LFP channel, plus one
exemplar channel per recording) live in a JSON sidecar written next to the .pl2
file as ``<stem>.channels.json``. They reload on the next open and feed the CSV
export.

Uses the same json.dump / json.load idiom as the pyqtdash handoff in callbacks.py.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from neurodash.behavior_io import get_mouse_id

SCHEMA_VERSION = 2


def channel_sidecar_path(pl2_path):
    """Return the sidecar path for a .pl2 file: <stem>.channels.json beside it."""
    pl2_path = Path(pl2_path)
    return pl2_path.parent / (pl2_path.stem + ".channels.json")


def parse_animal_id(pl2_path):
    """Best-effort animal ID from a pl2 filename: the last token, splitting on
    whitespace *or* underscores. Handles both conventions seen in the wild —
    '170505_open_field_theta_FC33-4' and '170505 open field theta FC33-4' both
    -> 'FC33-4'. Single-animal only; returns '' when no path is given.

    Read-only in the UI (always re-derived from the filename on load), so if a
    filename doesn't end in the animal token the fix is to rename the file — see
    the filename-handling note in CLAUDE.md.
    """
    if not pl2_path:
        return ""
    stem = Path(pl2_path).stem.strip()
    return re.split(r"[\s_]+", stem)[-1] if stem else ""


def resolve_animal_id(pl2_path, behavior_metadata=None):
    """The animal ID, from the best source available.

    The EthoVision header wins when a behavior file is loaded; the pl2 filename is
    the fallback. One resolver so every panel and every export agrees — the two
    disagree in practice (`multisession_ofd` has `C43-1` in the header and `c43-1`
    in the filename), and a case mix would split one animal into two groups
    downstream. The header is also the only source that works at all when the pl2
    isn't named after the animal.

    Still read-only in the UI and re-derived on every load, so a stale value in a
    sidecar is ignored.
    """
    if behavior_metadata:
        mouse_id = get_mouse_id(behavior_metadata)
        if mouse_id is not None and str(mouse_id).strip():
            return str(mouse_id).strip()
    return parse_animal_id(pl2_path)


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
        "animal": resolve_animal_id(session.pl2_path, session.behavior_metadata),
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

    # animal is read-only and always re-derived (base already holds the fresh
    # resolve); a stale saved value is intentionally ignored.
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
