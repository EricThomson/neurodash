"""Animal and session: canonical form, where the value comes from, and persistence.

One resolver feeds every panel and every export. A case or whitespace mix would
split one animal into two groups downstream, which is the whole reason these are
canonicalized rather than passed through.
"""

import json

import pytest

from neurodash import channel_io
from neurodash.channel_io import (
    canonical_id, channel_notes_path, load_identity, parse_animal_id,
    resolve_animal_id, resolve_session_name, save_channels, save_identity,
)


# --- canonical form -------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("C43-1", "C43-1"),
    ("  C43-1 ", "C43-1"),
    ("C43 -1", "C43-1"),        # whitespace goes, case stays
    ("", ""),
    (None, ""),
])
def test_canonical_animal(raw, expected):
    assert canonical_id(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("hab 1", "hab1"),
    ("Hab  1", "hab1"),
    ("HAB1", "hab1"),
    ("hab_1", "hab_1"),         # a human's naming decision, not ours to rewrite
])
def test_canonical_session(raw, expected):
    assert canonical_id(raw, lowercase=True) == expected


# --- where the value comes from -------------------------------------------

def test_metadata_beats_the_filename():
    """They disagree in practice: header 'C43-1' vs filename 'c43-1'."""
    pl2 = "/data/250818 dual hab1 c43-1.pl2"
    assert parse_animal_id(pl2) == "c43-1"
    assert resolve_animal_id(pl2, {"Mouse ID": "C43-1"}) == "C43-1"


def test_filename_is_the_fallback():
    assert resolve_animal_id("/data/170505_open_field_theta_FC33-4.pl2") == "FC33-4"
    assert resolve_animal_id("/data/x.pl2", {}) == "x"
    assert resolve_animal_id("", None) == ""


@pytest.mark.parametrize("key", ["Mouse ID", "mouse ID", "mouse id", "Animal ID"])
def test_animal_key_is_case_insensitive(key):
    """It's a user-defined EthoVision variable, spelled however it was typed."""
    assert resolve_animal_id("/data/x.pl2", {key: "C43-1"}) == "C43-1"


def test_useless_ethovision_serials_are_not_aliased():
    """'Subject ID'/'Arena ID' are real fields carrying 0 — they must not shadow."""
    assert resolve_animal_id("/data/x.pl2", {"Subject ID": 0, "Arena ID": 0}) == "x"


def test_session_is_canonicalized_and_optional():
    assert resolve_session_name({"Session": "hab 1"}) == "hab1"
    assert resolve_session_name({}) == ""       # FC33-4 has no Session field
    assert resolve_session_name(None) == ""


# --- persistence ----------------------------------------------------------

def test_identity_round_trip(tmp_path):
    pl2 = tmp_path / "rec.pl2"
    assert load_identity(pl2) == {"animal": "", "session": "", "bank": None}

    save_identity(pl2, "  C43-1", "Hab 2")
    assert load_identity(pl2) == {"animal": "C43-1", "session": "hab2", "bank": None}


def test_clearing_the_override_falls_back_to_inference(tmp_path):
    pl2 = tmp_path / "rec.pl2"
    save_identity(pl2, "C43-1", "hab2")
    save_identity(pl2, "", "")
    assert load_identity(pl2) == {"animal": "", "session": "", "bank": None}


def test_saving_channel_annotations_keeps_the_identity(tmp_path):
    """save_channels rewrites the whole file — it must carry these across."""
    pl2 = tmp_path / "rec.pl2"
    save_identity(pl2, "C43-1", "hab2")
    save_channels(pl2, {"pl2_filename": "rec.pl2", "comment": "a note",
                        "exemplar_channel_index": 3, "channels": {}})

    assert load_identity(pl2) == {"animal": "C43-1", "session": "hab2", "bank": None}
    saved = json.loads(channel_notes_path(pl2).read_text(encoding="utf-8"))
    assert saved["exemplar_channel_index"] == 3 and saved["comment"] == "a note"


def test_unreadable_notes_file_does_not_raise(tmp_path):
    pl2 = tmp_path / "rec.pl2"
    channel_notes_path(pl2).write_text("{not json", encoding="utf-8")
    assert load_identity(pl2) == {"animal": "", "session": "", "bank": None}


def test_identity_without_a_pl2_is_a_no_op():
    """A behavior-only session has nowhere to write; it must not raise."""
    save_identity(None, "AAA", "hab1")
    assert load_identity(None) == {"animal": "", "session": "", "bank": None}


# --- subject bank ---------------------------------------------------------
# Which of two animals in one .pl2 this session is about. It lives with
# animal/session because it is part of the same answer — whose recording is
# this — and because both other writers rewrite the whole file.

def test_bank_round_trips(tmp_path):
    """On a multi-animal file the animal is asked for BY BANK.

    Reading it without one returns blank rather than a guess: a single scalar
    beside a two-animal .pl2 is ambiguous, and answering anyway is how G16-1's
    ephys came back labelled G20-3.
    """
    pl2 = tmp_path / "rec.pl2"
    save_identity(pl2, "G20-3", "acquisition", bank=1)
    assert load_identity(pl2, bank=1) == {"animal": "G20-3",
                                          "session": "acquisition", "bank": 1}
    assert load_identity(pl2)["session"] == "acquisition"   # session is shared
    assert load_identity(pl2)["bank"] == 1


def test_bank_zero_survives(tmp_path):
    """Bank 0 is a real answer; only None means 'not chosen yet'."""
    pl2 = tmp_path / "rec.pl2"
    save_identity(pl2, "G16-1", "acquisition", bank=0)
    assert load_identity(pl2)["bank"] == 0


def test_saving_channel_annotations_keeps_the_bank(tmp_path):
    """Rating a channel must not un-assign the subject."""
    pl2 = tmp_path / "rec.pl2"
    save_identity(pl2, "G20-3", "acquisition", bank=1)
    save_channels(pl2, {"pl2_filename": "rec.pl2", "comment": "",
                        "exemplar_channel_index": None, "channels": {}})
    assert load_identity(pl2)["bank"] == 1


# --- one .pl2, two animals: the override must not leak between them --------
# Found the first time the second animal of the acquisition file was ever
# loaded. The sidecar held one `animal_override` for the whole file, so
# selecting G16-1's bank returned "G20-3" — the name typed for its neighbour —
# and its ephys would have exported under the wrong animal. Same class of
# mix-up the bank filtering prevents for the data itself.

def test_each_bank_keeps_its_own_animal(tmp_path):
    pl2 = tmp_path / "acquisition G16-1 and G20-3.pl2"
    save_identity(pl2, "G20-3", "acquisition", bank=1)
    save_identity(pl2, "G16-1", "acquisition", bank=0)

    assert load_identity(pl2, bank=0)["animal"] == "G16-1"
    assert load_identity(pl2, bank=1)["animal"] == "G20-3"


def test_an_unnamed_bank_is_blank_not_its_neighbours_name(tmp_path):
    """The failure that started this: only bank 1 was ever named."""
    pl2 = tmp_path / "acquisition G16-1 and G20-3.pl2"
    save_identity(pl2, "G20-3", "acquisition", bank=1)

    assert load_identity(pl2, bank=1)["animal"] == "G20-3"
    assert load_identity(pl2, bank=0)["animal"] == "", (
        "bank 0 inherited bank 1's animal name")


def test_a_legacy_scalar_migrates_to_the_bank_it_was_typed_under(tmp_path):
    """Sidecars written before this hold a scalar plus the active bank.

    That is enough to place it, so an already-entered name survives instead of
    being discarded, and it lands on one animal rather than both.
    """
    pl2 = tmp_path / "acquisition G16-1 and G20-3.pl2"
    path = channel_notes_path(pl2)
    path.write_text(json.dumps({"animal_override": "G20-3",
                                "session": "acquisition", "bank": 1}))

    assert load_identity(pl2, bank=1)["animal"] == "G20-3"
    assert load_identity(pl2, bank=0)["animal"] == ""

    # naming the other animal keeps the migrated one
    save_identity(pl2, "G16-1", "acquisition", bank=0)
    assert load_identity(pl2, bank=0)["animal"] == "G16-1"
    assert load_identity(pl2, bank=1)["animal"] == "G20-3"


def test_rating_a_channel_keeps_both_animals(tmp_path):
    """save_channels rewrites the whole file; it must not flatten the dict."""
    pl2 = tmp_path / "acquisition G16-1 and G20-3.pl2"
    save_identity(pl2, "G20-3", "acquisition", bank=1)
    save_identity(pl2, "G16-1", "acquisition", bank=0)
    save_channels(pl2, {"pl2_filename": pl2.name, "comment": "",
                        "exemplar_channel_index": None, "channels": {}})

    assert load_identity(pl2, bank=0)["animal"] == "G16-1"
    assert load_identity(pl2, bank=1)["animal"] == "G20-3"


# --- Save-As reopens where the last export went ---------------------------

def test_export_dir_defaults_to_export_dir_then_follows_you(tmp_path, isolated_state):
    """Exports come in sets, so the second dialog should start where the first ended."""
    from neurodash import app_state, config

    assert app_state.last_export_dir() == str(config.EXPORT_DIR)

    somewhere = tmp_path / "acquisition" / "save_testing"
    somewhere.mkdir(parents=True)
    app_state.remember_export_dir(somewhere / "G20-3_analysis.csv")
    assert app_state.last_export_dir() == str(somewhere)


def test_a_deleted_export_dir_falls_back(tmp_path, isolated_state):
    """A folder that has moved must not leave the dialog pointing at nothing."""
    from neurodash import app_state, config

    gone = tmp_path / "gone"
    gone.mkdir()
    app_state.remember_export_dir(gone)
    gone.rmdir()
    assert app_state.last_export_dir() == str(config.EXPORT_DIR)


def test_export_dir_is_separate_from_the_browse_dir(tmp_path, isolated_state):
    """You browse to raw recordings and save analysis CSVs elsewhere."""
    from neurodash import app_state

    raw = tmp_path / "raw"; raw.mkdir()
    out = tmp_path / "out"; out.mkdir()
    app_state.remember_browse_dir(raw / "rec.pl2")
    app_state.remember_export_dir(out / "rec_analysis.csv")
    assert app_state.last_browse_dir() == str(raw)
    assert app_state.last_export_dir() == str(out)
