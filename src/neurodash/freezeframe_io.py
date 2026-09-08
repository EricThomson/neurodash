"""FreezeFrame / Video Freeze behavioral data loading.

The fear-conditioning rig's behavior software, used for acquisition/tone/context
sessions. Nothing like EthoVision: a flat CSV with a key/value preamble, the
animal's name on its own row, then a single table at the video frame rate.

    Date :,8/5/2026 9:58
    Run Time :,1290.933,(seconds)
    ...
    Min Freeze Duration:,30,(frames),1,(seconds)
    Number of boxes:,1

    ,,G20_3
    Time,,Motion Index,Freeze Cnt,Time Freeze,C. Pct,I. Pct,Pct Export Time

    0.033,,324,0,0,0,0,0

The camera is side-on, so there is **no x/y position at all** — no position panel,
no arena calibration, no dot in the viewer. The movement variable is
``Motion Index``, not velocity.

One clock, unlike EthoVision's Trial time / Recording time pair: ``Time`` starts
at 0.033 (frame 1) at 30 fps and matches the video 1:1 with no offset. That also
means it provides no bridge to the neural clock — the pl2's TTL events carry the
alignment instead.
"""

import csv
import re

import numpy as np
import pandas as pd


# Read by name like the EthoVision loader, so extra columns and column order are
# both fine — only absence matters. 'Time' and 'Motion Index' are the plotted
# signals; the two cumulative columns are what freezing is reconstructed from.
REQUIRED_COLUMNS = ("Time", "Motion Index", "Freeze Cnt", "Time Freeze")

TIME_COLUMN = "Time"
MOTION_COLUMN = "Motion Index"

# Default when the preamble doesn't say. The rig's own setting is read per file.
DEFAULT_MIN_FREEZE_FRAMES = 30


def looks_like_freezeframe(path):
    """True when this file is a FreezeFrame export, by its first two keys.

    Cheap enough to run on every behavior file before choosing a reader: it reads
    two cells rather than the whole 38k-row table.
    """
    try:
        head = pd.read_csv(path, header=None, nrows=2, dtype=str,
                           usecols=[0], encoding="utf-8-sig")
    except (ValueError, OSError, UnicodeDecodeError, pd.errors.ParserError):
        return False
    keys = [str(v).strip().rstrip(":").strip().lower()
            for v in head[0].tolist() if v is not None]
    return "date" in keys and "run time" in keys


def load_freezeframe_file(path):
    """Load a FreezeFrame CSV.

    Returns ``(metadata, data)`` in the same shape as the EthoVision loader, so
    callers that only read columns by name don't care which rig produced the file.

    metadata keys are the preamble's, with the trailing colon stripped ('Run
    Time', 'Trial', 'Sample Rate (fps)', ...), plus:
      - 'Animal'   the name on its own row above the column headers
      - 'Min Freeze Duration (frames)' / '(seconds)' — the bout criterion, which
        the preamble states twice in different units on one row.
    """
    raw = _read_rows(path)

    header_row = _find_header_row(raw)
    metadata = _parse_preamble(raw, header_row)
    subject = _parse_subject(raw, header_row)
    # Some exports name the animal on that row, others name the BOX. A box is
    # not an animal, and guessing would be worse than blank: the behavior file
    # wins in resolve_animal_id, so "Box: Box 1" would become the exported animal
    # id for a real recording. Left None instead, for a human to fill in — the
    # same rule a multi-animal .pl2 and a missing Session already follow.
    metadata["Animal"] = None if _is_box_label(subject) else subject
    metadata["Box"] = subject if _is_box_label(subject) else None

    names = [str(v).strip() if v is not None and str(v) != "nan" else ""
             for v in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1:].copy()
    data.columns = names

    # The export puts a blank spacer column between 'Time' and 'Motion Index'.
    # Dropping unnamed columns keeps that out of the table without disturbing the
    # named ones, which are matched by name anyway.
    data = data.loc[:, [n != "" for n in names]]
    data = data.reset_index(drop=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    # A blank row sits between the header and the first sample.
    data = data.dropna(subset=[TIME_COLUMN]).reset_index(drop=True)

    data.attrs["units"] = {TIME_COLUMN: "s", MOTION_COLUMN: "au",
                           "Time Freeze": "s"}
    return metadata, data


def _find_header_row(raw):
    """Index of the column-name row — the one whose first cell is 'Time'.

    Found rather than hardcoded: the preamble's length is not guaranteed, and a
    wrong row number would silently shift every column.
    """
    for i in range(len(raw)):
        if str(raw.iloc[i, 0]).strip().lower() == TIME_COLUMN.lower():
            return i
    raise ValueError(
        f"No '{TIME_COLUMN}' header row found — this does not look like a "
        f"FreezeFrame export.")


def _parse_preamble(raw, header_row):
    """Key/value pairs above the table, e.g. 'Run Time :' -> 1290.933."""
    metadata = {}
    for i in range(header_row):
        key = str(raw.iloc[i, 0]).strip()
        if not key or key == "nan":
            continue
        key = key.rstrip(":").strip()
        value = raw.iloc[i, 1] if raw.shape[1] > 1 else None
        metadata[key] = None if _blank(value) else str(value).strip()

    # 'Min Freeze Duration:,30,(frames),1,(seconds)' — the bout criterion, stated
    # twice on one row. Both units are worth keeping: frames is what the
    # reconstruction needs, seconds is what a person recognizes.
    for i in range(header_row):
        if str(raw.iloc[i, 0]).strip().rstrip(":").strip().lower() != "min freeze duration":
            continue
        cells = [str(c).strip() for c in raw.iloc[i].tolist() if not _blank(c)]
        for value, unit in zip(cells[1:], cells[2:]):
            unit = unit.strip("()").lower()
            if unit in ("frames", "seconds"):
                metadata[f"Min Freeze Duration ({unit})"] = value
    return metadata


def _read_rows(path):
    """The file as a padded table of strings.

    Not `pd.read_csv`: two FreezeFrame exporters are in the wild and only one
    pads its preamble to the table's width. The other writes ragged rows (2, 3
    and 5 fields against the data's 8) and quotes its values, and pandas infers
    the column count from the FIRST row, so it raises a tokenizing error on line
    2 before reaching any data. Reading with the csv module and padding to the
    widest row accepts both.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty — not a FreezeFrame export.")
    width = max(len(row) for row in rows)
    return pd.DataFrame([row + [None] * (width - len(row)) for row in rows],
                        dtype=object)


def _parse_subject(raw, header_row):
    """Whatever sits alone on the row above the column headers.

    An animal name in one exporter ("G20_3"), a box label in another
    ("Box: Box 1"). Returned as written; the caller decides which it is.
    """
    for i in range(header_row - 1, -1, -1):
        row = raw.iloc[i].tolist()
        if not _blank(row[0]):
            break  # back into the preamble — no subject row
        values = [str(v).strip() for v in row if not _blank(v)]
        if values:
            return values[0]
    return None


def _is_box_label(value):
    """True for 'Box: Box 1' and friends — a chamber, not an animal."""
    return bool(value) and re.match(r"^\s*box\b", str(value), re.IGNORECASE)


def box_number(metadata):
    """Which chamber this recording came from, as an int, or None.

    Only some exports state it, on the row where others name the animal
    ("Box: Box 1"). Worth reading because the box is a real experimental fact:
    per the lab's session notes Box 1 is the no-shock control and Box 2 is
    shocked, and the rig wiring is Box N -> channel bank N.
    """
    if not metadata:
        return None
    match = re.search(r"(\d+)", str(metadata.get("Box") or ""))
    return int(match.group(1)) if match else None


def _blank(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() in ("", "nan")


def min_freeze_frames(metadata):
    """The rig's minimum-bout criterion in frames, per this file's preamble."""
    value = (metadata or {}).get("Min Freeze Duration (frames)")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return DEFAULT_MIN_FREEZE_FRAMES


def freezing_state(data, metadata=None):
    """Per-frame boolean freezing, reconstructed from the cumulative columns.

    **The obvious decode is wrong and looks fine.** ``Freeze Cnt`` and ``Time
    Freeze`` are cumulative, and simply diffing ``Time Freeze`` gives 10.8% frozen
    on the acquisition test file against the file's own ``I. Pct`` of 18.4% — a
    41% undercount that still produces a plausible-looking trace.

    The cause: FreezeFrame credits a bout only once it has lasted the minimum
    duration, so at each onset ``Time Freeze`` jumps by exactly 1.000 s (30 frames
    at 30 fps) and then ticks at the frame interval. Those 30 frames belong to the
    bout and are already over by the time they are counted, so the state has to be
    backfilled by ``min_freeze_frames - 1`` frames before every ``Freeze Cnt``
    increment.

    That reproduces the file exactly: 237.633 s vs its stated 237.633, 18.41% vs
    18.4%, 101 bouts, shortest bout exactly 1.000 s, and peak motion while frozen
    18 against the preamble's threshold of 19.

    Unlike open field there is no threshold to choose here — the rig already
    applied the bout criterion, so this decodes its answer rather than deriving
    one. See the freezing note in CLAUDE.md.
    """
    time_freeze = data["Time Freeze"].to_numpy(dtype=float)
    freeze_count = data["Freeze Cnt"].to_numpy(dtype=float)
    if len(time_freeze) == 0:
        return np.zeros(0, dtype=bool)

    frozen = np.diff(time_freeze, prepend=time_freeze[0]) > 1e-9
    backfill = max(0, min_freeze_frames(metadata) - 1)
    for onset in np.flatnonzero(np.diff(freeze_count, prepend=freeze_count[0]) > 0):
        frozen[max(0, onset - backfill):onset] = True
    return frozen


def freezing_percentage(data, metadata=None):
    """Percent of frames frozen — compare against the file's own 'I. Pct'."""
    frozen = freezing_state(data, metadata)
    return float(100.0 * frozen.mean()) if len(frozen) else 0.0


def stated_freezing_percentage(data):
    """The percentage FreezeFrame itself reports, from the last 'I. Pct' row.

    The independent check on `freezing_percentage`: the two must agree, and they
    are computed by completely different routes (ours from the cumulative columns,
    theirs written by the rig).
    """
    if "I. Pct" not in data.columns or len(data) == 0:
        return None
    value = data["I. Pct"].to_numpy(dtype=float)[-1]
    return None if np.isnan(value) else float(value)


def run_time_seconds(metadata):
    """'Run Time' from the preamble, in seconds.

    This is the behavior recording's duration, and it anchors the neural
    alignment: with no start pulse anywhere, behavior t=0 is found by subtracting
    this from the pl2's stop event.
    """
    raw = (metadata or {}).get("Run Time")
    if raw is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", str(raw))
    return float(match.group()) if match else None
