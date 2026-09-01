"""Reading FreezeFrame exports, and decoding freezing out of them.

The freezing decode is the part worth pinning. FreezeFrame already applied the
bout criterion, so unlike open field there is no threshold to choose — but its
columns are cumulative and the obvious diff undercounts freezing by 41% while
still producing a plausible trace. The file states its own answer in `I. Pct`,
which makes it self-checking, so that is what these assert against.

Synthetic CSVs rather than the 1.3 MB real export, built to the same layout.
"""

import numpy as np
import pytest

from neurodash import freezeframe_io as ff
from neurodash.behavior_io import (
    detect_behavior_format, load_behavior_file, behavior_time, behavior_format,
    ETHOVISION, FREEZEFRAME,
)

FPS = 30


def write_csv(tmp_path, rows, run_time=None, animal="G20_3", trial="Acquisition",
              min_freeze=30, name="session.csv"):
    """A FreezeFrame export with the real preamble layout."""
    if run_time is None:
        run_time = len(rows) / FPS
    lines = [
        f"Date :,8/5/2026 9:58,,,,,,",
        f"Run Time :,{run_time},(seconds),,,,,",
        "Protocol:,,,,,,,",
        "Experiment:,VF021126_104816,,,,,,",
        f"Trial:,{trial},,,,,,",
        "Video Saved:,Yes,,,,,,",
        f"Sample Rate (fps):,{FPS},,,,,,",
        "Motion Threshold (au):,19,,,,,,",
        "Detection Method:,Linear,,,,,,",
        f"Min Freeze Duration:,{min_freeze},(frames),{min_freeze / FPS:g},(seconds),,,",
        "Number of boxes:,1,,,,,,",
        ",,,,,,,",
        f",,{animal},,,,,",
        "Time,,Motion Index,Freeze Cnt,Time Freeze,C. Pct,I. Pct,Pct Export Time",
        ",,,,,,,",
    ]
    total = sum(1 for r in rows if r[2]) / FPS
    pct = 100.0 * sum(1 for r in rows if r[2]) / len(rows) if rows else 0.0
    for t, motion, _frozen, count, time_freeze in rows:
        lines.append(f"{t:g},,{motion},{count},{time_freeze:g},{pct:g},{pct:g},{pct:g}")
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, total, pct


def bouts_to_rows(n_frames, bouts, min_freeze=30):
    """Build rows the way FreezeFrame writes them.

    The subtlety being modelled, and the whole reason the naive decode fails:
    `Time Freeze` stays **flat** for the first `min_freeze` frames of a bout,
    because the rig doesn't know it is a bout yet. At frame `start + min_freeze
    - 1` it credits the whole minimum duration at once — a 1.000 s jump — and
    only then ticks per frame. So the frames it pays for are already in the past
    and nothing in the columns marks them.

    Getting this wrong makes the test worthless rather than failing: an earlier
    version ticked during those frames too, which quietly leaked the answer and
    let a decode with no backfill still pass.
    """
    for _start, length in bouts:
        assert length >= min_freeze, "a bout shorter than the criterion is not a bout"

    frozen = np.zeros(n_frames, dtype=bool)
    credited = np.zeros(n_frames, dtype=bool)   # the frame that books the bout
    ticking = np.zeros(n_frames, dtype=bool)    # frames counted one at a time
    for start, length in bouts:
        frozen[start:start + length] = True
        credited[start + min_freeze - 1] = True
        ticking[start + min_freeze:start + length] = True

    rows, count, time_freeze = [], 0, 0.0
    for i in range(n_frames):
        if credited[i]:
            count += 1
            time_freeze += min_freeze / FPS
        elif ticking[i]:
            time_freeze += 1 / FPS
        rows.append(((i + 1) / FPS, 20 if frozen[i] else 400, frozen[i],
                     count, round(time_freeze, 3)))
    return rows, frozen


# --- format sniffing ------------------------------------------------------

def test_a_freezeframe_export_is_recognized(tmp_path):
    path, _, _ = write_csv(tmp_path, bouts_to_rows(300, [(60, 90)])[0])
    assert ff.looks_like_freezeframe(path)
    assert detect_behavior_format(path) == FREEZEFRAME


def test_something_else_is_not_mistaken_for_one(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert not ff.looks_like_freezeframe(path)
    assert detect_behavior_format(path) == ETHOVISION


def test_a_missing_file_does_not_raise(tmp_path):
    assert not ff.looks_like_freezeframe(tmp_path / "nope.csv")


# --- reading --------------------------------------------------------------

def test_preamble_and_columns(tmp_path):
    rows, _ = bouts_to_rows(300, [(60, 90)])
    path, _, _ = write_csv(tmp_path, rows, run_time=10.0)
    metadata, data = load_behavior_file(path)

    assert metadata["Trial"] == "Acquisition"
    assert metadata["Animal"] == "G20_3"
    assert ff.run_time_seconds(metadata) == 10.0
    assert ff.min_freeze_frames(metadata) == 30
    # The blank spacer column between Time and Motion Index is dropped, and the
    # blank row between the header and the data doesn't become a NaN sample.
    assert list(data.columns)[:3] == ["Time", "Motion Index", "Freeze Cnt"]
    assert len(data) == len(rows)
    assert behavior_format(data) == FREEZEFRAME
    assert behavior_time(data)[0] == pytest.approx(1 / FPS)


def test_min_freeze_duration_is_read_in_both_units(tmp_path):
    rows, _ = bouts_to_rows(300, [(60, 90)], min_freeze=15)
    path, _, _ = write_csv(tmp_path, rows, min_freeze=15)
    metadata, _ = load_behavior_file(path)
    assert metadata["Min Freeze Duration (frames)"] == "15"
    assert metadata["Min Freeze Duration (seconds)"] == "0.5"


def test_a_file_with_no_time_column_is_refused(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date :,x\nRun Time :,1\nnothing,here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not look like a FreezeFrame"):
        ff.load_freezeframe_file(path)


# --- freezing decode ------------------------------------------------------

@pytest.mark.parametrize("bouts", [
    [(60, 90)],
    [(60, 90), (300, 45)],
    [(0, 30)],                       # a bout starting on the first frame
    [(60, 30), (120, 30), (200, 60)],
])
def test_freezing_matches_the_frames_that_were_really_frozen(tmp_path, bouts):
    rows, frozen = bouts_to_rows(600, bouts)
    path, _, _ = write_csv(tmp_path, rows)
    metadata, data = load_behavior_file(path)
    assert np.array_equal(ff.freezing_state(data, metadata), frozen)


def test_freezing_percentage_matches_the_files_own_figure(tmp_path):
    """The independent check: our decode vs the number the rig wrote itself."""
    rows, _ = bouts_to_rows(1200, [(100, 120), (500, 90), (900, 60)])
    path, _, pct = write_csv(tmp_path, rows)
    _metadata, data = load_behavior_file(path)
    assert ff.freezing_percentage(data) == pytest.approx(
        ff.stated_freezing_percentage(data), abs=0.05)
    assert ff.freezing_percentage(data) == pytest.approx(pct, abs=0.05)


def test_the_naive_diff_undercounts_and_is_not_what_we_do(tmp_path):
    """Guards the actual bug: diffing Time Freeze misses each bout's first second.

    On the real file that reads 10.8% against a stated 18.4%.
    """
    rows, frozen = bouts_to_rows(1200, [(100, 60), (400, 60), (800, 60)])
    path, _, _ = write_csv(tmp_path, rows)
    metadata, data = load_behavior_file(path)

    time_freeze = data["Time Freeze"].to_numpy(float)
    naive = np.diff(time_freeze, prepend=time_freeze[0]) > 1e-9
    assert naive.sum() < frozen.sum()
    assert ff.freezing_state(data, metadata).sum() == frozen.sum()


def test_no_freezing_at_all(tmp_path):
    rows, frozen = bouts_to_rows(300, [])
    path, _, _ = write_csv(tmp_path, rows)
    metadata, data = load_behavior_file(path)
    assert not ff.freezing_state(data, metadata).any()
    assert ff.freezing_percentage(data, metadata) == 0.0
