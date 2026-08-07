"""Channel Viewer annotations and session identity — saved to file, and CSV export.

Also owns *session identity* (animal and session label): inference from the data,
canonicalization, and the user's overrides. That isn't per-channel, but it shares
this module's annotations file and every export leads with it, so it lives here rather than
in a module of its own.

Channel annotations (a quality rating and comment per LFP channel, plus one
exemplar channel per recording) live in a companion JSON file written next to the
.pl2 as ``<stem>.channels.json``. They reload on the next open and feed the CSV
export.

Uses the same json.dump / json.load idiom as the pyqtdash handoff in callbacks.py.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from neurodash.behavior_io import get_mouse_id, get_session_name

SCHEMA_VERSION = 2


def channel_notes_path(pl2_path):
    """Where a recording's annotations are kept: <stem>.channels.json, beside the .pl2."""
    pl2_path = Path(pl2_path)
    return pl2_path.parent / (pl2_path.stem + ".channels.json")


def parse_animal_id(pl2_path):
    """Best-effort animal ID from a pl2 filename: the last token, splitting on
    whitespace *or* underscores. Handles both conventions seen in the wild —
    '170505_open_field_theta_FC33-4' and '170505 open field theta FC33-4' both
    -> 'FC33-4'. Single-animal only; returns '' when no path is given.

    The weakest source, and only a starting guess: the sidebar's Animal field can
    be corrected when a filename doesn't end in the animal token. Renaming the file
    used to be the only fix — see the filename-handling note in CLAUDE.md.
    """
    if not pl2_path:
        return ""
    stem = Path(pl2_path).stem.strip()
    return re.split(r"[\s_]+", stem)[-1] if stem else ""


def canonical_id(value, lowercase=False):
    """A whitespace-free identifier, so one label can't split into several groups.

    `hab 1` in the xlsx and `hab1` in the pl2 filename would be two levels of a
    grouping variable in JMP — the same failure `C43-1` vs `c43-1` would have caused
    for the animal. Sessions are lowercased as well; animals keep their case,
    because `C43-1` is how the ID is actually written.

    Only whitespace and case are touched. `hab_1` vs `hab-1` is a naming decision a
    human has to make, which is what the Edit button in the sidebar is for.
    """
    text = re.sub(r"\s+", "", str(value if value is not None else "").strip())
    return text.lower() if lowercase else text


def resolve_animal_id(pl2_path, behavior_metadata=None):
    """The inferred animal ID, from the best source available.

    The EthoVision header wins when a behavior file is loaded; the pl2 filename is
    the fallback. One resolver so every panel and every export agrees — the two
    disagree in practice (`multisession_ofd` has `C43-1` in the header and `c43-1`
    in the filename), and a case mix would split one animal into two groups
    downstream. The header is also the only source that works at all when the pl2
    isn't named after the animal.

    Inference only. The user can override it in the sidebar — this used to be
    strictly read-only with "rename the file" as the fix, which was reasonable when
    a filename was the only source and is not, now that the authoritative field is
    free text somebody typed. `load_identity` holds the override.
    """
    if behavior_metadata:
        mouse_id = get_mouse_id(behavior_metadata)
        if mouse_id is not None and str(mouse_id).strip():
            return canonical_id(mouse_id)
    return canonical_id(parse_animal_id(pl2_path))


def resolve_session_name(behavior_metadata=None):
    """The inferred session label, canonicalized: 'hab 1' -> 'hab1'.

    Blank when the recording has no Session field at all (FC33-4 doesn't) — there
    is nothing to guess from, so the sidebar shows it empty for a human to fill.
    """
    if not behavior_metadata:
        return ""
    return canonical_id(get_session_name(behavior_metadata), lowercase=True)


def load_identity(pl2_path):
    """Saved animal/session overrides for a recording, or blanks.

    Kept in the annotations file beside the .pl2 — it already carried `animal`, and
    the pl2 is the anchor for exports. A behavior-only session therefore can't
    persist an override; it still edits fine for the current browser session.
    """
    if not pl2_path:
        return {"animal": "", "session": ""}
    path = channel_notes_path(pl2_path)
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return {"animal": data.get("animal_override", ""),
                    "session": data.get("session", "")}
        except (json.JSONDecodeError, OSError):
            pass
    return {"animal": "", "session": ""}


def save_identity(pl2_path, animal="", session=""):
    """Persist animal/session overrides without disturbing the channel annotations.

    Read-modify-write rather than rebuilding the record, so this can't clobber
    quality ratings or the exemplar. `animal_override` is stored separately from
    the derived `animal` so a blank override still falls back to inference.
    """
    if not pl2_path:
        return
    path = channel_notes_path(pl2_path)
    data = {}
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data["animal_override"] = canonical_id(animal)
    data["session"] = canonical_id(session, lowercase=True)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


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
    """Load a recording's channel annotations, or a default if none are saved yet.

    When ``session`` is given, saved values are merged onto a fresh per-channel
    skeleton (missing channels filled in, labels refreshed from the session) so
    callers never special-case a missing or stale file.
    """
    path = channel_notes_path(pl2_path)
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
    """Write the channel record to its companion file next to the .pl2.

    Identity overrides are carried across rather than dropped: this rewrites the
    whole file from a record that doesn't contain them, so rating a channel would
    otherwise silently erase the animal/session the user typed in the sidebar.
    """
    existing = load_identity(pl2_path)
    channel_data = dict(channel_data)
    channel_data["schema_version"] = SCHEMA_VERSION
    channel_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    channel_data["animal_override"] = existing["animal"]
    channel_data["session"] = existing["session"]
    with open(channel_notes_path(pl2_path), "w") as f:
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
