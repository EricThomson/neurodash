"""Merge rules: what may be combined, what must be refused, and in what order.

The requirement is that a superseded export is refused *by name* rather than
silently concatenated into half-NaN columns — the layout changed three times in a
day, so old files are out there.
"""

import pandas as pd

from neurodash import merge


# --- what counts as a usable export --------------------------------------

def test_load_entry_reads_a_good_export(tmp_path, write_export):
    path = write_export(tmp_path, "a_analysis.csv", animal="AAA", session="hab1")
    entry, problem = merge.load_entry(path)
    assert problem is None
    assert entry["animal"] == "AAA" and entry["session"] == "hab1"
    assert entry["pairs"] == {("AAA", "hab1")}


def test_export_without_session_column_is_refused(tmp_path, write_export):
    """`time` is seconds *into* a session, so unlabelled rows can't be grouped."""
    frame = pd.DataFrame({"animal": ["AAA"], "time": [0.0], "theta_peak_AI17": [1.0]})
    path = write_export(tmp_path, "old_analysis.csv", frame=frame)
    entry, problem = merge.load_entry(path)
    assert entry is None
    assert "session column" in problem and "re-export" in problem


def test_pre_per_channel_export_is_refused(tmp_path, write_export):
    """Bare `theta_peak_hz` predates channels-as-columns."""
    frame = pd.DataFrame({"animal": ["AAA"], "session": ["hab1"], "time": [0.0],
                          "theta_peak_hz": [1.0]})
    path = write_export(tmp_path, "old_analysis.csv", frame=frame)
    entry, problem = merge.load_entry(path)
    assert entry is None and "theta column" in problem


def test_unit_carrying_theta_names_are_refused(tmp_path, write_export):
    """The short-lived `theta_peak_hz_<CH>` spelling, caught by prefix."""
    frame = pd.DataFrame({"animal": ["AAA"], "session": ["hab1"], "time": [0.0],
                          "theta_peak_hz_AI17": [1.0]})
    path = write_export(tmp_path, "interim_analysis.csv", frame=frame)
    entry, problem = merge.load_entry(path)
    assert entry is None and "theta column" in problem


def test_scan_folder_names_the_bad_file_and_keeps_the_rest(tmp_path, write_export):
    write_export(tmp_path, "good_analysis.csv")
    write_export(tmp_path, "bad_analysis.csv",
                 frame=pd.DataFrame({"animal": ["AAA"], "time": [0.0]}))
    entries, problems = merge.scan_folder(tmp_path)
    assert [e["name"] for e in entries] == ["good_analysis.csv"]
    assert len(problems) == 1 and "bad_analysis.csv" in problems[0]


# --- the uniqueness rule --------------------------------------------------

def test_same_animal_different_sessions_merges(tmp_path, write_export):
    """The whole point: one animal run across several days."""
    write_export(tmp_path, "a1_analysis.csv", animal="AAA", session="hab1")
    write_export(tmp_path, "a2_analysis.csv", animal="AAA", session="hab2")
    entries, problems = merge.scan_folder(tmp_path)
    assert not problems
    assert merge.check_compatible(entries) == []


def test_same_animal_and_session_twice_is_refused(tmp_path, write_export):
    write_export(tmp_path, "a_analysis.csv", animal="AAA", session="hab1")
    write_export(tmp_path, "a_copy_analysis.csv", animal="AAA", session="hab1")
    entries, _ = merge.scan_folder(tmp_path)
    reasons = merge.check_compatible(entries)
    assert reasons and "hab1" in reasons[0] and "AAA" in reasons[0]


def test_mismatched_analysis_parameters_are_refused(tmp_path, write_export):
    write_export(tmp_path, "a_analysis.csv", animal="AAA")
    write_export(tmp_path, "b_analysis.csv", animal="BBB", theta_band_hz="5.0-11.0")
    entries, _ = merge.scan_folder(tmp_path)
    reasons = merge.check_compatible(entries)
    assert reasons and "theta band" in reasons[0]


# --- output shape ---------------------------------------------------------

def test_merged_is_sorted_by_session_then_time(tmp_path, write_export):
    """Session is the OUTER key: the whole hab1 block, then hab2."""
    write_export(tmp_path, "a2_analysis.csv", animal="AAA", session="hab2")
    write_export(tmp_path, "a1_analysis.csv", animal="AAA", session="hab1")
    entries, _ = merge.scan_folder(tmp_path)
    merged = merge.merge_tables(entries)

    assert list(merged.columns)[:3] == ["session", "time", "animal"]
    order = merged["session"].tolist()
    blocks = [k for i, k in enumerate(order) if i == 0 or k != order[i - 1]]
    assert blocks == ["hab1", "hab2"]
    for session in blocks:
        block = merged[merged["session"] == session]
        assert block["time"].is_monotonic_increasing


def test_different_channel_sets_leave_nan_not_an_error(tmp_path, write_export):
    """Ragged but honest — the known cost of channels-as-columns."""
    write_export(tmp_path, "a_analysis.csv", animal="AAA", channels=("AI17",))
    write_export(tmp_path, "b_analysis.csv", animal="BBB", channels=("AI19",))
    entries, problems = merge.scan_folder(tmp_path)
    assert not problems and not merge.check_compatible(entries)
    merged = merge.merge_tables(entries)
    assert merged.loc[merged["animal"] == "AAA", "theta_peak_AI19"].isna().all()
    assert merged.loc[merged["animal"] == "AAA", "theta_peak_AI17"].notna().all()


def test_header_counts_what_is_really_there(tmp_path, write_export):
    for animal in ("AAA", "BBB"):
        for session in ("hab1", "hab2", "hab3"):
            write_export(tmp_path, f"{animal}_{session}_analysis.csv",
                         animal=animal, session=session)
    entries, _ = merge.scan_folder(tmp_path)
    counts = dict(line.split(": ") for line in merge.merged_header(entries).splitlines()
                  if line.startswith("n_"))
    assert counts == {"n_files": "6", "n_animals": "2", "n_sessions": "3"}


# --- merging merged tables ------------------------------------------------

def test_merge_of_merges(tmp_path, write_export):
    """Merge each animal's sessions, then merge those — a reasonable thing to do."""
    stage2 = tmp_path / "stage2"
    for animal in ("AAA", "BBB"):
        folder = tmp_path / animal
        for session in ("hab1", "hab2"):
            write_export(folder, f"{session}_analysis.csv", animal=animal, session=session)
        entries, _ = merge.scan_folder(folder)
        write_export(stage2, f"{animal}_all_analysis.csv",
                     frame=merge.merge_tables(entries), animal=animal)

    entries, problems = merge.scan_folder(stage2)
    assert not problems and not merge.check_compatible(entries)
    # each input must describe itself by what it CONTAINS, not by its first row
    for entry in entries:
        assert entry["sessions"] == ["hab1", "hab2"], entry["sessions"]

    final = merge.merge_tables(entries)
    assert final.duplicated(subset=["session", "time", "animal"]).sum() == 0
    assert sorted(final["animal"].unique()) == ["AAA", "BBB"]
    assert sorted(final["session"].unique()) == ["hab1", "hab2"]


def test_overlap_with_a_merged_file_is_caught(tmp_path, write_export):
    """Used to slip through: no first-row collision, so rows silently doubled."""
    source = tmp_path / "src"
    for session in ("hab1", "hab2"):
        write_export(source, f"{session}_analysis.csv", animal="AAA", session=session)
    entries, _ = merge.scan_folder(source)

    trap = tmp_path / "trap"
    write_export(trap, "AAA_all_analysis.csv", frame=merge.merge_tables(entries))
    write_export(trap, "AAA_hab2_analysis.csv", animal="AAA", session="hab2")

    entries, _ = merge.scan_folder(trap)
    reasons = merge.check_compatible(entries)
    assert reasons and "hab2" in reasons[0]
