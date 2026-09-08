"""Which analog stream is the LFP, and which channels belong to this animal.

Both are load-bearing for correctness rather than convenience:

- A .pl2 can carry several analog streams and only one is the LFP. The
  acquisition files have the real signal in `FP` and auxiliary-input noise in
  `AI`, and `AI` comes first — so assuming signal 0 plots noise as LFP, computes
  theta on it and exports it, all silently.
- A .pl2 can hold two animals on separate headstages, which shows up as two runs
  of channel numbers. Offering both animals' channels in one picker is how one
  animal's ephys ends up exported under the other's name.

No real Plexon data here (the files are ~300 MB); both are pure logic over the
channel labels and stream ids that load_pl2_block returns.
"""

import numpy as np
import pytest

from neurodash.neural_io import (
    select_lfp_signal_index, detect_channel_banks, banks_look_independent,
)


def summary(index, stream_id, n_channels=5):
    return {"analog_signal_index": index, "stream_id": stream_id,
            "stream_name": f"{stream_id}-stream", "n_channels": n_channels}


# ---------------------------------------------------------------------------
# Stream selection
# ---------------------------------------------------------------------------

def test_fp_wins_over_ai_even_though_ai_comes_first():
    """The acquisition case: AI is signal 0, but FP holds the LFP."""
    summaries = [summary(0, "AI", 32), summary(1, "FP", 10)]
    assert select_lfp_signal_index(summaries) == 1


def test_ai_is_used_when_there_is_no_fp_stream():
    """The open-field case: one AI stream, and the LFP is in it."""
    assert select_lfp_signal_index([summary(0, "AI", 5)]) == 0


def test_unrecognized_streams_fall_back_to_the_first():
    assert select_lfp_signal_index([summary(0, "WB"), summary(1, "XYZ")]) == 0


def test_no_streams_at_all_does_not_raise():
    assert select_lfp_signal_index([]) == 0


# ---------------------------------------------------------------------------
# Bank detection
# ---------------------------------------------------------------------------

def test_single_run_is_one_bank():
    """Open field's AI17-AI21 — nothing about single-animal files changes."""
    banks = detect_channel_banks(["AI17", "AI18", "AI19", "AI20", "AI21"])
    assert len(banks) == 1
    assert banks[0]["indices"] == [0, 1, 2, 3, 4]
    assert banks[0]["label"] == "AI17-AI21"


def test_gap_in_numbering_splits_into_two_banks():
    """The acquisition case: FP01-FP05 and FP17-FP21 are two animals."""
    labels = ["FP01", "FP02", "FP03", "FP04", "FP05",
              "FP17", "FP18", "FP19", "FP20", "FP21"]
    banks = detect_channel_banks(labels)
    assert [b["label"] for b in banks] == ["FP01-FP05", "FP17-FP21"]
    assert [b["indices"] for b in banks] == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_a_single_channel_bank_is_labelled_by_itself():
    banks = detect_channel_banks(["FP01", "FP07"])
    assert [b["label"] for b in banks] == ["FP01", "FP07"]


def test_labels_without_numbers_stay_one_bank():
    """The 'Ch 1'-style fallback can't be grouped, and must not be split wrongly."""
    banks = detect_channel_banks(["alpha", "beta"])
    assert len(banks) == 1
    assert banks[0]["indices"] == [0, 1]


def test_no_channels_at_all_does_not_raise():
    assert len(detect_channel_banks([])) == 1


# ---------------------------------------------------------------------------
# Independence check — numbering is convention, correlation is physics
# ---------------------------------------------------------------------------

def _banks_of_five():
    return [{"indices": [0, 1, 2, 3, 4]}, {"indices": [5, 6, 7, 8, 9]}]


def test_independent_banks_pass():
    """Two animals share no reference: across-bank correlation is the noise floor."""
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(5000, 10))
    assert banks_look_independent(signal, _banks_of_five())


def test_banks_sharing_a_signal_are_flagged():
    """A numbering gap that isn't a subject boundary must not silently split one animal."""
    rng = np.random.default_rng(1)
    shared = rng.normal(size=(5000, 1))
    signal = shared + rng.normal(scale=0.1, size=(5000, 10))
    assert not banks_look_independent(signal, _banks_of_five())


def test_a_single_bank_has_nothing_to_check():
    assert banks_look_independent(np.zeros((10, 5)), [{"indices": [0, 1, 2, 3, 4]}])


# ---------------------------------------------------------------------------
# What the UI is allowed to offer. This is the guarantee of the whole feature:
# on a two-animal .pl2 nothing is offered until a subject is chosen, so the other
# animal's ephys cannot be plotted, starred or exported under this animal's name.
#
# Built by hand rather than from a .pl2 — these are ~300 MB and the logic under
# test is pure. Session.__new__ skips the loader; the methods are the real ones.
# ---------------------------------------------------------------------------

from neurodash.session import Session

TWO_ANIMALS = ["FP01", "FP02", "FP03", "FP04", "FP05",
               "FP17", "FP18", "FP19", "FP20", "FP21"]
ONE_ANIMAL = ["AI17", "AI18", "AI19", "AI20", "AI21"]


def fake_session(labels, bank_index=None):
    s = Session.__new__(Session)
    s.analog_signal_summaries = [{
        "analog_signal_index": 0, "stream_id": "FP", "stream_name": "FPl-Low Pass Filtered",
        "channel_labels": labels, "channel_indices": list(range(len(labels))),
        "n_channels": len(labels),
    }]
    s.lfp_signal_index = 0
    s.bank_index = bank_index
    # A real Session always has these; the double skips __init__, so set the
    # "nothing loaded" values explicitly rather than leaving attributes missing.
    s.pl2_path = None
    s.behavior_path = None
    s.behavior_metadata = None
    return s


def test_single_animal_offers_every_channel_without_being_asked():
    """Open field must be untouched by any of this."""
    s = fake_session(ONE_ANIMAL)
    assert not s.is_multi_animal
    assert s.channel_indices() == [0, 1, 2, 3, 4]


def test_two_animals_offer_nothing_until_a_subject_is_chosen():
    s = fake_session(TWO_ANIMALS, bank_index=None)
    assert s.is_multi_animal
    assert s.channel_indices() == []
    assert s.channel_options() == []


@pytest.mark.parametrize("bank, expected", [
    (0, ["FP01", "FP02", "FP03", "FP04", "FP05"]),
    (1, ["FP17", "FP18", "FP19", "FP20", "FP21"]),
])
def test_a_chosen_subject_offers_only_its_own_channels(bank, expected):
    s = fake_session(TWO_ANIMALS, bank_index=bank)
    assert [label for _, label in s.channel_options()] == expected


def test_out_of_range_bank_offers_nothing_rather_than_guessing():
    """A stale saved bank (file changed under us) must not silently pick animal 0."""
    assert fake_session(TWO_ANIMALS, bank_index=7).channel_indices() == []


def test_channel_options_carry_absolute_stream_indices():
    """Indices address the whole stream, so they stay valid in extract_time_window."""
    s = fake_session(TWO_ANIMALS, bank_index=1)
    assert [i for i, _ in s.channel_options()] == [5, 6, 7, 8, 9]


# --- animal ID on a multi-animal file -------------------------------------
# The .pl2 filename names BOTH animals ("acquisition G16-1 and G20-3"), so
# parse_animal_id's last token is an arbitrary pick between them. Blank until
# the behavior file says — the same "leave it for a human" pattern open field
# already uses for a missing Session field.

def test_filename_is_not_trusted_when_the_pl2_holds_two_animals():
    s = fake_session(TWO_ANIMALS, bank_index=1)
    s.pl2_path = "/data/acquisition G16-1 and G20-3.pl2"
    s.behavior_metadata = None
    assert s.animal_id == ""


def test_the_behavior_file_still_names_the_animal():
    s = fake_session(TWO_ANIMALS, bank_index=1)
    s.pl2_path = "/data/acquisition G16-1 and G20-3.pl2"
    s.behavior_metadata = {"Mouse ID": "G20-3"}
    assert s.animal_id == "G20-3"


def test_single_animal_files_still_use_the_filename():
    """Open field must keep inferring FC33-4 with no behavior file loaded."""
    s = fake_session(ONE_ANIMAL)
    s.pl2_path = "/data/170505_open_field_theta_FC33-4.pl2"
    s.behavior_metadata = None
    assert s.animal_id == "FC33-4"


def test_a_saved_override_wins_over_inference():
    """One answer for every panel and every export.

    The Channel Viewer and the neural CSV took the inferred ID while the Session
    Info panel and the analysis CSV took the override, so one recording could
    export as `G20_3` in one file and `G20-3` in the other — the same split into
    two JMP groups that canonical_id exists to prevent.
    """
    from neurodash.channel_io import save_identity
    import neurodash.channel_io as channel_io

    s = fake_session(TWO_ANIMALS, bank_index=1)
    s.pl2_path = "/data/acquisition G16-1 and G20-3.pl2"
    s.behavior_metadata = {"Animal": "G20_3"}
    assert s.animal_id == "G20_3"          # inferred from the behavior file

    saved = {"animal": "G20-3", "session": "acquisition", "bank": 1}
    original = channel_io.load_identity
    channel_io.load_identity = lambda _path, _bank=None: saved
    try:
        import neurodash.session as session_module
        session_module.load_identity = lambda _path, _bank=None: saved
        assert s.animal_id == "G20-3"      # the correction the user typed
    finally:
        channel_io.load_identity = original
        session_module.load_identity = original


def test_a_blank_override_falls_back_to_inference():
    import neurodash.session as session_module
    s = fake_session(ONE_ANIMAL)
    s.pl2_path = "/data/170505_open_field_theta_FC33-4.pl2"
    s.behavior_metadata = None
    original = session_module.load_identity
    session_module.load_identity = lambda _p, _b=None: {"animal": "", "session": "", "bank": None}
    try:
        assert s.animal_id == "FC33-4"
    finally:
        session_module.load_identity = original


# --- naming the channel groups --------------------------------------------
# Filename order is ASSUMED to match headstage-bank order, pending confirmation
# from the lab. It labels the picker and nothing else, so a wrong assumption is a
# visible mislabel rather than data exported under the wrong animal.

@pytest.mark.parametrize("stem, expected", [
    ("acquisition G16-1 and G20-3", ["G16-1", "G20-3"]),
    ("170505_open_field_theta_FC33-4", ["FC33-4"]),
    ("phase animalA and animalB", ["animalA", "animalB"]),
    ("rec G1 AND G2", ["G1", "G2"]),          # case-insensitive separator
    ("", []),
])
def test_parse_animal_ids(stem, expected):
    from neurodash.channel_io import parse_animal_ids
    path = f"/data/{stem}.pl2" if stem else ""
    assert parse_animal_ids(path) == expected


def test_naming_does_not_leak_into_the_authoritative_animal_id():
    """The label is a guess; the exported ID must still come from a real source."""
    s = fake_session(TWO_ANIMALS, bank_index=1)
    s.pl2_path = "/data/acquisition G16-1 and G20-3.pl2"
    s.behavior_metadata = None
    import neurodash.session as session_module
    original = session_module.load_identity
    session_module.load_identity = lambda _p, _b=None: {"animal": "", "session": "", "bank": 1}
    try:
        assert s.animal_id == ""
    finally:
        session_module.load_identity = original


# --- naming one animal of a two-animal recording --------------------------
# Some FreezeFrame exports name the BOX where others name the animal, so the
# behavior file's contents can identify nothing. The recording still does, twice
# over: the .pl2 filename lists its animals in bank order, and the behavior
# FILENAME usually contains the animal outright. Neither is trusted alone.

def two_animal_session(bank_index, behavior_name=None, box=None, animal=None):
    s = fake_session(TWO_ANIMALS, bank_index=bank_index)
    s.pl2_path = "/data/acquisition G16-1 and G20-3.pl2"
    s.behavior_path = f"/data/{behavior_name}" if behavior_name else None
    s.behavior_metadata = {}
    if box is not None:
        s.behavior_metadata["Box"] = f"Box: Box {box}"
    if animal is not None:
        s.behavior_metadata["Animal"] = animal
    return s


def test_behavior_filename_names_the_animal():
    s = two_animal_session(0, behavior_name="G16-1 Raw Aquisition.csv")
    assert s.animal_id == "G16-1"


def test_stated_box_names_the_animal():
    """Box N -> bank N is confirmed rig wiring, so the box is enough on its own."""
    s = two_animal_session(0, behavior_name="raw acquisition.csv", box=1)
    assert s.animal_id == "G16-1"


def test_nothing_corroborating_leaves_it_blank():
    """The .pl2 filename alone is a convention, not evidence."""
    s = two_animal_session(0, behavior_name="raw acquisition.csv")
    assert s.animal_id == ""


def test_behavior_file_for_the_other_animal_is_refused():
    """Wrong behavior file for the selected channels: blank, never a name."""
    assert two_animal_session(1, behavior_name="G16-1 Raw Aquisition.csv").animal_id == ""
    assert two_animal_session(1, behavior_name="raw.csv", box=1).animal_id == ""


def test_a_header_animal_from_the_other_bank_is_refused():
    """The header normally wins, but not against the recording itself.

    G20-3's file paired with G16-1's channels would otherwise label G16-1's
    ephys `G20_3` — silently, and in every export.
    """
    s = two_animal_session(0, behavior_name="G20-3 Raw Aquisition.csv",
                           animal="G20_3")
    assert s.animal_id == ""


def test_punctuation_does_not_defeat_the_check():
    """The raw CSV writes G20_3 where the .pl2 writes G20-3."""
    s = two_animal_session(0, animal="G20_3")
    assert s.animal_id == ""
