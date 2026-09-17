"""What a recording's filename says about it.

These rules used to live in two modules - `channel_io` had the animal parsers,
`Session` had its own stem logic - and they had already drifted: one folded `_`
against `-` before comparing two animal names and the other did not. That is the
failure this module exists to prevent, so the folding rule gets its own tests.

Session type matters more than it looks. A tone session's .pl2 is structurally
identical to an acquisition one - same EVT01 x5, same EVT02 forty seconds later,
same 222 s ITI, same duration - because a tone test runs the same program in a
different chamber. The filename is the only thing that distinguishes them.
"""

import pytest

from neurodash.filename_metadata import (
    animal_for_bank, filename_names_animal, names_another_animal,
    parse_animal_id, parse_animal_ids, parse_session_type, same_animal,
)


# --- session type ----------------------------------------------------------

@pytest.mark.parametrize("stem, expected", [
    ("acquisition G16-1 and G20-3", "acquisition"),
    ("tone G16-1 and G20-3", "tone"),
    ("context G16-1 and G20-3", "context"),
    # the fear rig's own exports are spelled this way; a parser that insists on
    # the correct spelling silently fails on real files
    ("G16-1 Raw Aquisition", "acquisition"),
    ("acq G16-1", "acquisition"),
    # order is not fixed: the type may be any token
    ("G16-1_context", "context"),
    # open field states no session type, and must not be forced into one
    ("170505 open field theta FC33-4", ""),
    ("250818 dual hab1 c43-1", ""),
    ("", ""),
])
def test_parse_session_type(tmp_path, stem, expected):
    path = str(tmp_path / f"{stem}.pl2") if stem else ""
    assert parse_session_type(path) == expected


def test_session_type_is_the_only_thing_separating_tone_from_acquisition(tmp_path):
    """Their TTLs are identical, so nothing inside either file can decide this."""
    acq = str(tmp_path / "acquisition G16-1 and G20-3.pl2")
    tone = str(tmp_path / "tone G16-1 and G20-3.pl2")
    assert parse_animal_ids(acq) == parse_animal_ids(tone)
    assert parse_session_type(acq) != parse_session_type(tone)


# --- animals ---------------------------------------------------------------

@pytest.mark.parametrize("stem, expected", [
    ("170505 open field theta FC33-4", "FC33-4"),
    ("170505_open_field_theta_FC33-4", "FC33-4"),
    ("", ""),
])
def test_parse_animal_id(tmp_path, stem, expected):
    path = str(tmp_path / f"{stem}.pl2") if stem else ""
    assert parse_animal_id(path) == expected


@pytest.mark.parametrize("stem, expected", [
    ("acquisition G16-1 and G20-3", ["G16-1", "G20-3"]),
    ("acquisition G16-1 AND G20-3", ["G16-1", "G20-3"]),
    ("170505 open field theta FC33-4", ["FC33-4"]),
])
def test_parse_animal_ids(tmp_path, stem, expected):
    assert parse_animal_ids(str(tmp_path / f"{stem}.pl2")) == expected


# --- the folding rule that had drifted -------------------------------------

def test_punctuation_folds_when_comparing_names():
    """The raw CSV writes `G20_3` where the .pl2 and the 1-s xlsx write `G20-3`."""
    assert same_animal("G20_3", "G20-3")
    assert same_animal("g20-3", "G20-3")
    assert not same_animal("G16-1", "G20-3")
    assert not same_animal("", "")          # two blanks are not an animal


def test_a_behavior_filename_naming_the_animal_with_an_underscore_still_matches(tmp_path):
    """The bug the split hid: this comparison used a plain `.lower()` and missed it."""
    pl2 = str(tmp_path / "acquisition G16-1 and G20-3.pl2")
    assert animal_for_bank(pl2, str(tmp_path / "G20_3 Raw Aquisition.csv"), 1) == "G20-3"


def test_filename_names_animal():
    assert filename_names_animal("/d/G16-1 Raw Aquisition.csv", "G16-1")
    assert not filename_names_animal("/d/G16-1 Raw Aquisition.csv", "G20-3")
    assert not filename_names_animal("", "G16-1")
    assert not filename_names_animal("/d/x.csv", "")


# --- the veto: a behavior file paired with the wrong channel group ----------

def test_the_filename_names_the_animal_on_its_bank(tmp_path):
    pl2 = str(tmp_path / "acquisition G16-1 and G20-3.pl2")
    assert animal_for_bank(pl2, str(tmp_path / "G16-1 Raw Aquisition.csv"), 0) == "G16-1"
    assert animal_for_bank(pl2, str(tmp_path / "G20-3 Raw Aquisition.csv"), 1) == "G20-3"


def test_a_behavior_file_for_the_other_animal_names_nobody(tmp_path):
    """Worse than no answer: it would label one animal's ephys with its neighbour's."""
    pl2 = str(tmp_path / "acquisition G16-1 and G20-3.pl2")
    assert animal_for_bank(pl2, str(tmp_path / "G20-3 Raw Aquisition.csv"), 0) == ""
    assert animal_for_bank(pl2, str(tmp_path / "G16-1 Raw Aquisition.csv"), 1) == ""


def test_an_uncorroborated_name_is_not_guessed(tmp_path):
    """The .pl2 filename alone is a naming habit, not evidence."""
    pl2 = str(tmp_path / "acquisition G16-1 and G20-3.pl2")
    assert animal_for_bank(pl2, str(tmp_path / "Box 1 Raw Aquisition.csv"), 0) == ""
    assert animal_for_bank(pl2, "", 0) == ""
    assert animal_for_bank("", "/d/G16-1.csv", 0) == ""
    assert animal_for_bank(pl2, "/d/G16-1.csv", None) == ""


def test_names_another_animal(tmp_path):
    pl2 = str(tmp_path / "acquisition G16-1 and G20-3.pl2")
    assert names_another_animal(pl2, 0, "G20-3")
    assert not names_another_animal(pl2, 0, "G16-1")
    assert names_another_animal(pl2, 0, "G20_3"), "punctuation must fold here too"
    assert not names_another_animal(pl2, None, "G20-3")
    assert not names_another_animal(pl2, 5, "G20-3")
