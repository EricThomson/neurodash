"""The navigator strip: extent, and what gets drawn on it.

The strip is positioned entirely in percent of `session_extent`, so that number
is the calibration for everything on it. Get it wrong and every segment is drawn
at a plausible but wrong place, which is the kind of failure that looks like a
working feature.

Deliberately data-driven with no session-type branch, so the open-field case is
tested here as a first-class outcome rather than an edge case.
"""

import numpy as np
import pandas as pd
import pytest

from neurodash import config, session_navigator
from neurodash.session import Session

# Mirrors the acquisition test file (see test_event_overlay).
TONES = np.array([5739.277, 5961.284, 6183.289, 6405.296, 6627.302])
SHOCKS = TONES + 39.994
SEGMENT = (5545.324, 6869.349)
FRAGMENTS = [{"start_s": 0.00055, "n_samples": 113},
             {"start_s": 34.01955, "n_samples": 1290004}]
EVENTS = {"EVT01": TONES, "EVT02": SHOCKS, "EVT03": np.array([6869.304])}
METADATA = {"Run Time": "1290.933"}

BEHAVIOR_END = 1290.9
NEURAL_DURATION = 1290.117


def behavior_frame(end=BEHAVIOR_END):
    # Ends exactly at `end`: np.arange stops short of its endpoint, and the
    # extent is read off the LAST sample, so an approximate tail would make
    # these tests agree with themselves rather than with the session.
    times = np.append(np.arange(0.0, end, 1.0), end)
    frame = pd.DataFrame({"Time": times})
    frame.attrs["format"] = "freezeframe"
    return frame


class FakeSession(Session):
    """A Session with events and a fragment table but no .pl2 behind it."""

    def __init__(self, events=None, behavior=None, neural_duration=None):
        self.events = events if events is not None else {}
        self.segment = SEGMENT
        self.behavior_metadata = METADATA
        self.behavior_data = behavior
        self.block = object() if neural_duration else None
        self.pl2_path = None
        self.lfp_signal_index = 0
        self.bank_index = None
        self.epoch_params = None
        self.no_shock = False
        self.analog_signal_summaries = (
            [{"duration_sec": neural_duration, "channel_labels": [],
              "sampling_rate_hz": 1000.0}]
            if neural_duration else [])

    @property
    def analog_fragments(self):
        return FRAGMENTS


def acquisition_session():
    return FakeSession(events=EVENTS, behavior=behavior_frame(),
                       neural_duration=NEURAL_DURATION)


def open_field_session():
    """No events at all, which is every open-field file."""
    return FakeSession(events={}, behavior=behavior_frame(480.0),
                       neural_duration=480.0)


def descendants(component):
    """Flatten a Dash component tree into a list."""
    out = [component]
    children = getattr(component, "children", None)
    if children is None:
        return out
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "className"):
            out.extend(descendants(child))
    return out


def by_class(strip, name):
    return [c for c in descendants(strip) if getattr(c, "className", None) == name]


def find_strip(wrapper):
    return next(c for c in descendants(wrapper)
                if getattr(c, "id", None) == "nav-strip")


# --- extent: the calibration everything else rests on ----------------------

def test_extent_is_the_union_not_either_file_alone():
    """Plexon stopped ~34 s before EthoVision, so neural alone cuts the strip short."""
    session = acquisition_session()
    assert session_navigator.session_extent(session) == pytest.approx(BEHAVIOR_END)


def test_extent_from_neural_when_behavior_is_absent():
    session = FakeSession(neural_duration=NEURAL_DURATION)
    extent = session_navigator.session_extent(session)
    assert extent == pytest.approx(NEURAL_DURATION + session.neural_time_offset)


def test_extent_from_behavior_when_neural_is_absent():
    session = FakeSession(behavior=behavior_frame())
    assert session_navigator.session_extent(session) == pytest.approx(BEHAVIOR_END)


def test_no_extent_means_no_strip():
    assert session_navigator.session_extent(FakeSession()) is None
    assert session_navigator.build_strip(FakeSession()) is None


# --- what gets drawn -------------------------------------------------------

def test_acquisition_strip_has_a_segment_per_epoch_and_a_tick_per_shock():
    from neurodash import epochs

    session = acquisition_session()
    strip = session_navigator.build_strip(session)
    expected = len(epochs.build_epochs(TONES - session.behavior_anchor,
                                       SHOCKS - session.behavior_anchor,
                                       None, BEHAVIOR_END))

    assert len(by_class(strip, "nav-segment")) == expected
    assert len(by_class(strip, "nav-shock-tick")) == len(SHOCKS)


def test_open_field_gets_a_strip_with_no_segments():
    """The decision that open field is served too, pinned.

    Not an edge case being tolerated: jobs 2 and 3 (navigate, and see what
    fraction is on screen) need only a time axis. A regression here would show
    up as the strip quietly vanishing for the session type most people run.
    """
    strip = session_navigator.build_strip(open_field_session())

    assert strip is not None
    assert by_class(strip, "nav-segment") == []
    assert by_class(strip, "nav-shock-tick") == []
    assert any(getattr(c, "id", None) == "nav-viewport" for c in descendants(strip))


def test_segments_stay_inside_the_strip():
    """Percent positions must not run past 100, or a band spills off the end."""
    strip = session_navigator.build_strip(acquisition_session())
    for segment in by_class(strip, "nav-segment"):
        left = float(segment.style["left"].rstrip("%"))
        width = float(segment.style["width"].rstrip("%"))
        assert 0.0 <= left and left + width <= 100.0 + 1e-6


def test_narrow_segments_get_no_label():
    """Below the threshold the text truncates to debris ("ton", "trac")."""
    strip = session_navigator.build_strip(acquisition_session())
    extent = session_navigator.session_extent(acquisition_session())
    for segment in by_class(strip, "nav-segment"):
        width_fraction = float(segment.style["width"].rstrip("%")) / 100.0
        labelled = segment.children is not None
        assert labelled == (width_fraction >= session_navigator.LABEL_MIN_FRACTION)


def test_strip_carries_the_values_the_js_reads():
    """Python is the single source for both; the JS only reads them back."""
    strip = find_strip(session_navigator.build_strip(acquisition_session()))
    assert float(strip.__dict__["data-tend"]) == pytest.approx(BEHAVIOR_END)
    assert int(strip.__dict__["data-throttle-ms"]) == config.NAVIGATOR_LIVE_THROTTLE_MS


def test_epoch_params_reach_the_segments():
    """The strip must show the windows the sidebar is set to, like the figure does."""
    session = acquisition_session()
    default = session_navigator.build_strip(session)
    widened = session_navigator.build_strip(
        session, {"isi_duration": config.ACQUISITION_EPOCH_PARAMS["isi_duration"] / 2})

    def isi_widths(strip):
        return [s.style["width"] for s in by_class(strip, "nav-segment")
                if s.title.startswith("isi")]

    assert isi_widths(default) != isi_widths(widened)


def test_strip_padding_matches_the_figure_margins():
    """If these drift the strip stops indexing the axis below it."""
    wrapper = session_navigator.build_strip(acquisition_session())
    assert wrapper.style["paddingLeft"] == f"{config.PLOT_MARGIN_LEFT_PX}px"
    assert wrapper.style["paddingRight"] == f"{config.PLOT_MARGIN_RIGHT_PX}px"
