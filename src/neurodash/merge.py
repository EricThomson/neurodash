"""Combine per-animal `_analysis.csv` exports into one JMP-ready table.

neurodash exports one analysis table per animal per session. Analysis in JMP wants
them stacked into a single table — `session`, `time`, `animal`, then the same derived
columns. Because every export already carries those and shares a time base, that is a
plain concat; the work here is deciding *whether* a set of files may be concatenated,
and reading the `#` provenance header back in (nothing else in the codebase does).

The rule that matters: a merged column has to mean the same thing in every row. The
analysis parameters (window/step/c/max_freq, theta band, ratio bands, estimator) are
therefore required to match across files, while the per-recording facts (animal,
session, source, channels, times, comments) are expected to differ — that is the
point of merging. **The same animal across several days is the main case**, which is
why the uniqueness key is animal *and* session rather than animal alone.

Enforcing the first group is also what keeps the merged header small and, more
importantly, a FIXED height. If parameters could vary they would have to be recorded
per animal, and at 40 animals the header becomes a 40-line preamble of a different
shape on every file. Shared parameters mean one small parameter block that applies to
every row; per-animal provenance stays in the individual exports next door.

No UI and no Dash here — callbacks own the dialogs and the writing.
"""

from io import StringIO
from pathlib import Path

import pandas as pd


# Header fields that define what a column *means*. A mismatch in any of these makes
# the merged table incoherent — either the rows don't align in time (window/step) or
# a column holds two different variables (bands/estimator) — so it blocks the merge.
BLOCKING_FIELDS = (
    "time_base",            # carries window, step and c together, as written
    "theta_band_hz",
    "theta_ratio_bands_hz",
    "theta_estimator",
)

# Plain-language names for the refusal message; the raw keys are jargon.
FIELD_LABELS = {
    "time_base": "time grid (spectrogram window/step)",
    "theta_band_hz": "theta band",
    "theta_ratio_bands_hz": "theta ratio bands",
    "theta_estimator": "theta peak estimator",
}

ANALYSIS_GLOB = "*_analysis.csv"

# Theta columns as written before channels moved into column names: bare, with the
# channel named only in the header. Their presence identifies a superseded export.
_UNSUFFIXED_THETA = ("theta_peak_hz", "theta_power_db", "theta_ratio")

# Theta columns briefly carried their units *and* the suffix ("theta_peak_hz_AI17").
# Prefix-matched so those files are caught too — they'd otherwise concat alongside
# today's "theta_peak_AI17" as a second, half-empty column for the same variable.
_LEGACY_PREFIXES = ("theta_peak_hz_", "theta_power_db_")


def _is_superseded_layout(df):
    """True if this export predates the current theta column names."""
    return (any(c in df.columns for c in _UNSUFFIXED_THETA)
            or any(str(c).startswith(_LEGACY_PREFIXES) for c in df.columns))


def _compare_value(field, value):
    """The part of a header value that has to match, ignoring per-animal detail.

    ``time_base`` reads "spectral bins, step 0.1s (window 1.5s, c=10)". Older exports
    append " on AI17" — one analysis channel per file, expected to differ per animal —
    so anything from " on " onward is dropped before comparing. Channels now live in
    the table's own ``channel`` column instead, and the suffix is no longer written.
    """
    value = (value or "").strip()
    if field == "time_base":
        return value.split(" on ")[0].strip()
    return value


def read_analysis_csv(path):
    """Read one neurodash analysis export back in.

    Returns ``(meta, df)`` — the `#` header as a dict, and the table.

    Deliberately not ``pd.read_csv(comment="#")``: the header carries free-text
    comment fields, and a `#` typed inside one would truncate that line. Splitting
    on the first non-`#` line is exact, and it is how `_write_export` builds the
    file (all comment lines first, then the CSV body).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    meta = {}
    body_lines = []
    in_header = True
    for line in text.splitlines(keepends=True):
        if in_header and line.startswith("#"):
            key, sep, value = line[1:].partition(":")
            if sep:
                meta[key.strip()] = value.strip()
            continue
        in_header = False
        body_lines.append(line)
    df = pd.read_csv(StringIO("".join(body_lines)))
    return meta, df


def scan_folder(folder):
    """Find analysis exports in ``folder``.

    Returns ``(entries, problems)``. Each entry is a dict with ``path``, ``name``,
    ``animal``, ``meta`` and ``df``. ``problems`` holds one message per file that
    couldn't be read — a single corrupt file in a folder of twenty shouldn't take
    the whole scan down, it should be named.
    """
    entries, problems = [], []
    for path in sorted(Path(folder).glob(ANALYSIS_GLOB)):
        try:
            meta, df = read_analysis_csv(path)
        except Exception as e:
            problems.append(f"{path.name}: could not be read ({e})")
            continue
        if "animal" not in df.columns or "time" not in df.columns:
            problems.append(f"{path.name}: not a neurodash analysis export "
                            f"(no animal/time columns)")
            continue
        if "session" not in df.columns:
            # Without it there is no way to tell day 1 from day 2, and `time` is
            # seconds *into* a session, so the rows would look like duplicates of
            # each other. Refuse rather than merge something ungroupable.
            problems.append(f"{path.name}: exported before the session column — "
                            f"re-export it from the Session Viewer")
            continue
        if _is_superseded_layout(df):
            # Concatenating one of these with a current export would leave two
            # columns meaning the same thing, each half NaN, and the older one not
            # even saying which channel it came from. Name it instead.
            problems.append(f"{path.name}: exported before the current theta column "
                            f"names (expected theta_peak_<channel>) — "
                            f"re-export it from the Session Viewer")
            continue
        entries.append({
            "path": path,
            "name": path.name,
            "animal": meta.get("animal") or str(df["animal"].iloc[0]),
            "session": meta.get("session") or str(df["session"].iloc[0]),
            "meta": meta,
            "df": df,
        })
    entries.sort(key=lambda e: (e["animal"], e["session"]))
    return entries, problems


def _group_by_value(entries, field):
    """Map each distinct comparable value of ``field`` to the files carrying it."""
    groups = {}
    for entry in entries:
        value = _compare_value(field, entry["meta"].get(field, ""))
        groups.setdefault(value, []).append(entry["name"])
    return groups


def _display_value(field, value):
    """Trim a compared value down to what's worth showing in a message."""
    if field == "time_base":
        return value.replace("spectral bins, ", "")
    return value or "(blank)"


def check_compatible(entries):
    """Reasons these files can't be merged. Empty list means go ahead.

    Each reason names the field, every distinct value with a file count, the
    minority files by name, *and its own fix* — a bare "files are incompatible"
    leaves the user hunting through twenty exports, and a generic closing line
    would give the wrong advice for whichever problem it doesn't fit.
    """
    reasons = []
    if len(entries) < 2:
        return reasons

    for field in BLOCKING_FIELDS:
        groups = _group_by_value(entries, field)
        if len(groups) < 2:
            continue
        label = FIELD_LABELS.get(field, field)
        # Largest group is the presumed-correct one; the rest are what to re-export.
        ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        summary = ", ".join(
            f"{len(files)} file{'s' if len(files) != 1 else ''} "
            f"{'use' if len(files) != 1 else 'uses'} {_display_value(field, value)}"
            for value, files in ordered
        )
        odd = [name for _value, files in ordered[1:] for name in files]
        reasons.append(
            f"The {label} differs — {summary}.\n"
            f"    Odd ones out: {', '.join(odd)}\n"
            f"    Fix: re-export those with the same settings as the rest."
        )

    # One row per animal per SESSION per time bin. The same animal across several
    # days is the point of merging, so only a repeated animal+session pair is a real
    # duplicate — that would be two exports of one recording, and a single export
    # already carries every channel that was ticked to save.
    duplicates = {}
    for entry in entries:
        duplicates.setdefault((entry["animal"], entry["session"]), []).append(entry["name"])
    for (animal, session), files in sorted(duplicates.items()):
        if len(files) > 1:
            reasons.append(
                f"Animal {animal} appears {len(files)} times in session "
                f"'{session}': {', '.join(files)}.\n"
                f"    An animal may appear once per session — different sessions of\n"
                f"    the same animal are fine, and are the point of merging.\n"
                f"    Fix: untick all but one, re-exporting with all the channels you\n"
                f"    want if they're split across files."
            )
    return reasons


def refusal_message(reasons):
    """The text shown when a merge is refused.

    Says what's wrong, which files, and what to do about it. A block with no path
    forward is the kind that makes people give up on the tool.
    """
    body = "\n".join(f"  - {r}" for r in reasons)
    return ("These files can't be combined as-is — their columns wouldn't mean the "
            f"same thing in every row.\n{body}")


def merge_tables(entries):
    """Stack the entries into one long-format table.

    Column order and sort are both ``session, time, animal`` — the leftmost columns
    read in sort order, so the file explains its own layout. **Session is the OUTER
    key**: the whole hab1 block first, then hab2, each internally in time order with
    a time bin's animals together. Sorting by time first would interleave the days,
    which is not how anyone reads these.

    Sessions order alphabetically, which is what you want for hab1/hab2/hab3 (and
    would put hab10 before hab2, if it ever came to that).

    Animals that saved different channels contribute different theta columns, so
    the concat leaves NaN where an animal didn't save that channel. That is the
    known cost of channels-as-columns; it vanishes when the same channels are
    saved across animals.
    """
    frames = [entry["df"] for entry in entries]
    merged = pd.concat(frames, ignore_index=True)

    # Grids are identical when the parameters match (enforced upstream), but rounding
    # before the sort costs nothing and stops float drift from splitting a time bin.
    merged["time"] = merged["time"].round(6)

    lead = [c for c in ("session", "time", "animal") if c in merged.columns]
    rest = [c for c in merged.columns if c not in lead]
    merged = merged[lead + rest]
    return merged.sort_values(lead, kind="stable").reset_index(drop=True)


def merged_header(entries):
    """Parameter block for the merged file — a FIXED number of rows, always.

    Deliberately does not carry per-animal provenance. A line per animal would
    make the header height vary with the animal count, which at 40 animals is a
    40-line preamble on every file and a different shape each time. The
    per-animal detail (source recording, analysis channel, comments) already
    lives in the individual ``_analysis.csv`` exports, which sit alongside this
    file — ``source_folder`` points at them.

    What is here is exactly what applies to *every* row: the analysis parameters,
    which check_compatible has already guaranteed are shared. Adding a field means
    every merged file grows by one row, so append rather than insert.
    """
    first = entries[0]["meta"]
    lines = ["merged: analysis tables combined by neurodash"]
    for field in BLOCKING_FIELDS:
        lines.append(f"{field}: {_compare_value(field, first.get(field, ''))}")
    lines.append(f"n_animals: {len(entries)}")
    lines.append(f"source_folder: {entries[0]['path'].parent}")
    return "\n".join(lines)
