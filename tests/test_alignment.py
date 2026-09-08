"""Tying the neural clock to the behavior clock, and proving it held.

The alignment is anchored on the END of both recordings, because there is no
behavior-onset pulse anywhere in these files. That works, but its failure mode is
specific and silent: anything that changes the behavior file's *duration* shifts
everything, and the result still plots. So the self-check matters as much as the
arithmetic, and both are tested here.

Synthetic events and motion rather than real files — the arithmetic is pure, and
the .pl2s are ~300 MB. Numbers mirror the acquisition test file.
"""

import numpy as np
import pandas as pd
import pytest

from neurodash import alignment, config


# The real file: tones every 222.007 s from 5739.277, shocks 39.994 s later,
# a lone stop marker at 6869.304, and a 1290.933 s behavior recording whose
# t=0 therefore sits at pl2 5578.371.
TONES = np.array([5739.277, 5961.284, 6183.289, 6405.296, 6627.302])
SHOCKS = TONES + 39.994
STOP = 6869.304
RUN_TIME = 1290.933
ANCHOR = STOP - RUN_TIME
SEGMENT = (5545.324, 6869.349)

EVENTS = {"EVT01": TONES, "EVT02": SHOCKS, "EVT03": np.array([STOP])}
METADATA = {"Run Time": str(RUN_TIME)}


def behavior_frame(shift_s=0.0, spike=8000.0, baseline=80.0):
    """30 fps Motion Index with a startle at each shock, optionally mis-shifted."""
    times = np.arange(1, int(RUN_TIME * 30) + 1) / 30.0
    rng = np.random.default_rng(0)
    motion = rng.gamma(2.0, baseline / 2.0, size=times.size)
    for shock in SHOCKS - ANCHOR + shift_s:
        motion[np.abs(times - shock) < 0.2] = spike
    frame = pd.DataFrame({"Time": times, "Motion Index": motion})
    frame.attrs["format"] = "freezeframe"
    return frame


# The real acquisition file's fragment table: 113 samples, a 33.906 s gap, then
# the recording proper. Behavior t=0 is the start of that main fragment.
FRAGMENTS = [{"start_s": 0.00055, "n_samples": 113},
             {"start_s": 34.01955, "n_samples": 1290004}]
ANALOG_START = SEGMENT[0] + FRAGMENTS[1]["start_s"]     # 5579.344


# --- the anchor: the recordings START together ----------------------------
# The lab's wrangling notes: "time zero of the behavioral measure is the start
# time TTL". Anchoring on the end instead put behavior t=0 0.97 s early, which
# drew every shock TTL almost a second after the animal's response to it.

def test_anchor_is_the_start_of_the_analog_data():
    assert alignment.behavior_start_in_pl2(EVENTS, METADATA, SEGMENT, FRAGMENTS) ==         pytest.approx(5579.344, abs=1e-3)


def test_anchor_ignores_the_stop_marker_when_fragments_are_known():
    """A wildly wrong stop marker must not move a fragment-anchored session."""
    events = dict(EVENTS, EVT03=np.array([STOP + 500.0]))
    assert alignment.behavior_start_in_pl2(events, METADATA, SEGMENT, FRAGMENTS) ==         alignment.behavior_start_in_pl2(EVENTS, METADATA, SEGMENT, FRAGMENTS)


def test_end_anchoring_survives_only_as_a_fallback():
    """Without a fragment table there is nothing better, but it is ~1 s out."""
    fallback = alignment.behavior_start_in_pl2(EVENTS, METADATA, SEGMENT, None)
    real = alignment.behavior_start_in_pl2(EVENTS, METADATA, SEGMENT, FRAGMENTS)
    assert fallback == pytest.approx(STOP - RUN_TIME, abs=1e-6)
    assert real - fallback == pytest.approx(0.973, abs=1e-2)


def test_offset_only_undoes_neos_fragment_concatenation():
    """Inside the main fragment, sample time already IS behavior time.

    All the offset does is cancel the 113 orphan samples neo prepends.
    """
    assert alignment.neural_time_offset(EVENTS, METADATA, SEGMENT, NEURAL_DURATION,
                                        FRAGMENTS, 1000.0) ==         pytest.approx(-0.113, abs=1e-6)


def test_shock_times_land_on_the_protocol_baseline():
    """Cross-check against the lab's own ingested cohort, which has tone1 at
    159.936 s. End-anchoring gave 160.906 — a full second out."""
    anchor = alignment.behavior_start_in_pl2(EVENTS, METADATA, SEGMENT, FRAGMENTS)
    trial = alignment.trial_structure(EVENTS, anchor)
    assert trial["tones"][0] == pytest.approx(159.933, abs=0.02)
    assert trial["shocks"][0] == pytest.approx(199.927, abs=0.02)


# --- the offset -----------------------------------------------------------
# NOTE: the tests below pass fragments=None, so they exercise the FALLBACK.
# The fragment-aware path — what actually runs — is covered above.

def test_anchor_is_the_stop_marker_minus_the_behavior_duration():
    assert alignment.behavior_start_in_pl2(EVENTS, METADATA, SEGMENT) == \
        pytest.approx(5578.371, abs=1e-3)


NEURAL_DURATION = 1290.117


def test_offset_comes_from_the_two_durations_not_from_t_start():
    """Both recordings stop together, so the shorter one started later.

    Regression: the first implementation used neo's `segment.t_start` (5545.324)
    as the time of neural sample 0, and was wrong by 33 s — it put the shock
    artifacts a third of a minute away from the shock TTLs. The analog samples
    simply do not begin at `t_start`.
    """
    offset = alignment.neural_time_offset(EVENTS, METADATA, SEGMENT,
                                          NEURAL_DURATION)
    assert offset == pytest.approx(RUN_TIME - NEURAL_DURATION, abs=1e-6)
    assert offset == pytest.approx(0.816, abs=1e-3)


def test_t_start_is_not_used_for_the_offset():
    """A wildly different t_start must not change the answer."""
    weird = (SEGMENT[0] - 5000.0, SEGMENT[1])
    assert alignment.neural_time_offset(EVENTS, METADATA, weird, NEURAL_DURATION) == \
        alignment.neural_time_offset(EVENTS, METADATA, SEGMENT, NEURAL_DURATION)


def test_equal_durations_mean_no_offset():
    assert alignment.neural_time_offset(EVENTS, METADATA, SEGMENT, RUN_TIME) == \
        pytest.approx(0.0)


def test_no_neural_duration_means_no_offset():
    """A .pl2 with no analog stream has nothing to place."""
    assert alignment.neural_time_offset(EVENTS, METADATA, SEGMENT, None) == 0.0


def test_segment_stop_agrees_with_the_stop_marker():
    """The free cross-check: two independent routes to the same anchor."""
    from_marker = alignment.behavior_start_in_pl2(EVENTS, METADATA, SEGMENT)
    from_segment = alignment.behavior_start_in_pl2({}, METADATA, SEGMENT)
    assert abs(from_marker - from_segment) < 0.1


def test_no_events_means_no_offset():
    """Open field: the two recordings were started together, so sample 0 is t=0.

    This is why the app was always right while ignoring a non-zero t_start —
    sample-relative time is the correct primitive.
    """
    assert alignment.neural_time_offset({}, None, SEGMENT, NEURAL_DURATION) == 0.0


def test_offset_is_zero_until_a_behavior_file_supplies_a_duration():
    """A .pl2 on its own has nothing to anchor to and stays in sample time."""
    assert alignment.neural_time_offset(EVENTS, None, SEGMENT, NEURAL_DURATION) == 0.0


# --- trial structure ------------------------------------------------------

def test_tones_and_shocks_land_on_the_behavior_clock():
    trial = alignment.trial_structure(EVENTS, ANCHOR)
    assert trial["tones"][0] == pytest.approx(160.906, abs=1e-2)
    assert trial["shocks"][0] == pytest.approx(200.900, abs=1e-2)


def test_trace_interval_is_measured_not_assumed():
    """Shock onset minus tone offset — the one timing the file actually states."""
    assert alignment.trial_structure(EVENTS, ANCHOR)["trace_s"] == \
        pytest.approx(20.0, abs=0.05)


def test_a_session_without_shocks_still_reports_its_tones():
    """A tone session: tones but no shocks, so no trace interval either."""
    trial = alignment.trial_structure({"EVT01": TONES}, ANCHOR)
    assert len(trial["tones"]) == 5
    assert trial["shocks"] is None and trial["trace_s"] is None


# --- the self-check -------------------------------------------------------
# Only the duration test remains. The startle check (does Motion Index jump at
# each shock TTL?) was removed: it can only speak for shocked animals, and the
# shock TTL fires rig-wide while only Box 2 delivers it, so every no-shock
# control reported "0 of 5 startles found" — correct, and pure noise on half the
# sessions. See check_alignment's docstring for the two designs that failed
# before it, so they are not retried.

def test_a_correct_alignment_passes_quietly():
    ok, message = alignment.check_alignment(EVENTS, METADATA, behavior_frame(), SEGMENT)
    assert ok and message == ""


def test_a_truncated_behavior_export_is_caught():
    """The failure the anchor is exposed to: a changed duration moves everything."""
    truncated = {"Run Time": str(RUN_TIME - 60)}
    ok, message = alignment.check_alignment(EVENTS, truncated, behavior_frame(), SEGMENT)
    assert not ok and "truncated" in message


def test_a_no_shock_control_says_nothing():
    """The whole point of the removal: controls must not be flagged."""
    no_startle = behavior_frame(spike=80.0)     # nothing above baseline motion
    ok, message = alignment.check_alignment(EVENTS, METADATA, no_startle, SEGMENT)
    assert ok and message == ""


def test_a_session_with_no_shocks_at_all_says_nothing():
    ok, message = alignment.check_alignment({"EVT01": TONES}, METADATA,
                                            behavior_frame(), SEGMENT)
    assert ok and message == ""


def test_missing_motion_column_is_not_a_failure():
    """The duration check does not read Motion Index at all."""
    frame = behavior_frame().drop(columns=["Motion Index"])
    assert alignment.check_alignment(EVENTS, METADATA, frame, SEGMENT)[0]


def test_no_behavior_data_is_not_a_failure():
    assert alignment.check_alignment(EVENTS, METADATA, None, SEGMENT) == (True, "")
