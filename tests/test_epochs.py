"""Carving a trace-conditioning session into epochs.

The window arithmetic is small but it has already gone wrong once in a way that
reached published-ish results — the notebooks' ISI windows came out 90 s where
120 s was intended, because a trailing buffer was subtracted from a length that
already excluded it. See sandbox/acquisition/isi_window_bug.md. That case has a
regression test here.

Times mirror the acquisition test file: tones every 222.007 s from 160.906 on the
behavior clock, shocks 39.994 s later.
"""

import numpy as np
import pytest

from neurodash import config, epochs

TONES = 160.906 + 222.007 * np.arange(5)
SHOCKS = TONES + 39.994
SESSION_END = 1290.933


def built(**overrides):
    return epochs.build_epochs(TONES, SHOCKS, overrides or None)


def by_label(built_epochs):
    return {e["label"]: e for e in built_epochs}


# --- the windows ----------------------------------------------------------

def test_every_epoch_is_present_and_in_order():
    e = built()
    assert [x["label"] for x in e][:4] == ["baseline", "tone1", "trace1", "isi1"]
    assert len(e) == 1 + 5 + 5 + 5
    starts = [x["start"] for x in e]
    assert starts == sorted(starts)


def test_baseline_skips_the_start_and_stops_before_the_first_tone():
    b = by_label(built())["baseline"]
    assert b["start"] == pytest.approx(10.0)
    assert b["end"] == pytest.approx(TONES[0] - 1.0)


def test_tone_and_trace_are_inset_by_their_pads():
    e = by_label(built())
    assert e["tone1"]["end"] - e["tone1"]["start"] == pytest.approx(20 - 0.5)
    assert e["trace1"]["start"] == pytest.approx(TONES[0] + 20 + 0.25)
    assert e["trace1"]["end"] == pytest.approx(SHOCKS[0] - 0.25)


def test_isi_is_delay_plus_duration():
    """Regression: the notebooks' formula gave 90 s where 120 s was intended.

    Expressed as delay + duration there is no second subtraction to get wrong, so
    the window is exactly as long as the parameter says whatever the other
    buffers are set to.
    """
    e = by_label(built())["isi1"]
    assert e["start"] == pytest.approx(SHOCKS[0] + 30.0)
    assert e["end"] - e["start"] == pytest.approx(120.0)


@pytest.mark.parametrize("duration", [60.0, 90.0, 120.0, 150.0])
def test_isi_length_always_equals_the_parameter(duration):
    e = by_label(built(isi_duration=duration))["isi1"]
    assert e["end"] - e["start"] == pytest.approx(duration)


# --- the two properties that are easy to get wrong -------------------------

def test_epochs_are_not_a_partition():
    """Guard bands are deliberate; a contiguous tiling would mean they were lost."""
    e = built()
    gaps = [b["start"] - a["end"] for a, b in zip(e, e[1:])]
    assert all(g > 0 for g in gaps), "epochs should never touch or overlap"


def test_the_shock_falls_in_no_epoch():
    """It sits between trace (ends before it) and isi (starts well after it)."""
    for shock in SHOCKS:
        inside = [e["label"] for e in built()
                  if e["start"] <= shock <= e["end"]]
        assert inside == []


# --- degenerate sessions ---------------------------------------------------

def test_a_session_with_no_tones_gets_no_epochs():
    """Context: baseline is defined off the first tone, so there is nothing to anchor."""
    assert epochs.build_epochs(None, None) == []
    assert epochs.build_epochs([], []) == []


def test_a_session_with_tones_but_no_shocks_gets_no_trace_or_isi():
    """A tone session: the trace is bounded by the shock, which never happens."""
    kinds = {e["kind"] for e in epochs.build_epochs(TONES, None)}
    assert kinds == {"baseline", "tone"}


def test_windows_are_clipped_to_the_recording():
    e = epochs.build_epochs(TONES, SHOCKS, None, session_end=1200.0)
    assert max(x["end"] for x in e) <= 1200.0


def test_a_negative_width_window_is_dropped_rather_than_drawn():
    """An absurd pad shouldn't produce an inside-out band."""
    labels = {e["label"] for e in built(tone_pad=15.0)}
    assert not any(l.startswith("tone") for l in labels)


# --- warnings --------------------------------------------------------------

def test_an_isi_running_into_the_next_tone_is_reported():
    """The quiet failure the notebooks had no check for."""
    e = built(isi_duration=200.0)
    warnings = epochs.overlap_warnings(e, TONES)
    assert warnings and "past the next tone" in warnings[0]


def test_sensible_defaults_produce_no_warnings():
    assert epochs.overlap_warnings(built(), TONES) == []


# --- labelling -------------------------------------------------------------

def test_labels_are_blank_outside_every_epoch():
    """Blank is a real value: a bin the analysis deliberately excludes."""
    times = np.array([0.0, 5.0, 100.0, TONES[0] + 10, SHOCKS[0] + 1])
    labels = epochs.epoch_labels(times, built())
    assert list(labels) == ["", "", "baseline", "tone1", ""]


def test_labels_never_average_across_a_boundary():
    """Epochs are categorical — containment only, no interpolation."""
    e = by_label(built())["tone1"]
    just_outside = np.array([e["start"] - 0.01, e["end"] + 0.01])
    assert list(epochs.epoch_labels(just_outside, built())) == ["", ""]


def test_event_masks_are_binary_and_cover_the_stimulus():
    spans = epochs.event_spans(TONES, SHOCKS)
    times = np.array([TONES[0] + 5, SHOCKS[0] + 1, 5.0])
    assert list(epochs.event_mask(times, spans, "tone_event")) == [1, 0, 0]
    assert list(epochs.event_mask(times, spans, "shock_event")) == [0, 1, 0]


def test_the_tone_event_is_wider_than_the_tone_epoch():
    """The event is what happened; the epoch is inset by tone_pad."""
    event = [s for s in epochs.event_spans(TONES, SHOCKS) if s["kind"] == "tone_event"][0]
    epoch = by_label(built())["tone1"]
    assert event["start"] < epoch["start"] and event["end"] > epoch["end"]


def test_params_fall_back_to_config_defaults():
    assert epochs.epoch_params(None) == dict(config.ACQUISITION_EPOCH_PARAMS)
    assert epochs.epoch_params({"isi_duration": None})["isi_duration"] == \
        config.ACQUISITION_EPOCH_PARAMS["isi_duration"]
