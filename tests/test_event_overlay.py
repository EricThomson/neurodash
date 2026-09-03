"""Everything that draws trial structure must agree on where it goes.

This exists because it didn't. The behavior anchor moved from end-anchored to
start-anchored, `Session.trial_events` was updated, and the TTL overlay — which
derived the anchor itself — was not. Its lines sat ~0.97 s from the epoch bands
built from the very same events, and it shipped: the whole suite stayed green
because nothing exercised the overlay.

The failure mode is what makes it worth a test. Omitting `analog_fragments` from
`behavior_start_in_pl2` doesn't raise; it silently returns the old end-anchored
answer, and the result plots perfectly plausibly.
"""

import ast
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from neurodash import alignment, plot_utils
from neurodash.session import Session

# Mirrors the acquisition test file.
TONES = np.array([5739.277, 5961.284, 6183.289, 6405.296, 6627.302])
SHOCKS = TONES + 39.994
STOP = 6869.304
SEGMENT = (5545.324, 6869.349)
FRAGMENTS = [{"start_s": 0.00055, "n_samples": 113},
             {"start_s": 34.01955, "n_samples": 1290004}]
EVENTS = {"EVT01": TONES, "EVT02": SHOCKS, "EVT03": np.array([STOP])}
METADATA = {"Run Time": "1290.933"}


def panel():
    """A one-panel figure with a trace in it.

    The trace matters: `add_vline(row="all")` defaults to
    `exclude_empty_subplots=True`, so it silently draws nothing on a subplot that
    has no data. Real panels always have some, but an empty test figure would
    pass this file for the wrong reason.
    """
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=[0.0, 1300.0], y=[0.0, 0.0]), row=1, col=1)
    return fig


class FakeSession(Session):
    """A Session with events and a fragment table but no .pl2 behind it."""

    def __init__(self):
        self.events = EVENTS
        self.segment = SEGMENT
        self.behavior_metadata = METADATA
        self.behavior_data = None
        self.block = None
        self.pl2_path = None
        self.analog_signal_summaries = []
        self.lfp_signal_index = 0
        self.bank_index = None
        self.epoch_params = None

    @property
    def analog_fragments(self):
        return FRAGMENTS


# --- the invariant that broke ---------------------------------------------

def test_ttl_overlay_lands_on_the_same_times_as_trial_events():
    """The lines and the bands come from one set of events; they must agree.

    They differed by 0.97 s in the shipped app, which is what this pins.
    """
    session = FakeSession()
    fig = panel()
    plot_utils._add_ttl_pulses(fig, session, {"show_ttl_pulses": True})

    drawn = sorted({round(shape.x0, 4) for shape in fig.layout.shapes})
    trial = session.trial_events
    expected = sorted({round(float(x), 4) for x in
                       list(trial["tones"]) + list(trial["shocks"]) +
                       [STOP - session.behavior_anchor]})
    assert drawn == expected


def test_the_anchor_is_start_anchored_not_end_anchored():
    """Guards the specific regression: a fallback answer ~1 s out."""
    session = FakeSession()
    end_anchored = alignment.behavior_start_in_pl2(EVENTS, METADATA, SEGMENT, None)
    assert session.behavior_anchor == pytest.approx(5579.344, abs=1e-3)
    assert session.behavior_anchor - end_anchored == pytest.approx(0.973, abs=1e-2)


def test_tones_land_on_the_protocol_baseline():
    """Cross-check against the lab's ingested cohort, whose tone1 is 159.936 s."""
    assert FakeSession().trial_events["tones"][0] == pytest.approx(159.933, abs=0.02)


def test_no_overlay_without_the_toggle():
    fig = panel()
    plot_utils._add_ttl_pulses(fig, FakeSession(), {"show_ttl_pulses": False})
    assert not fig.layout.shapes


def test_a_session_with_no_events_draws_nothing():
    """Open field: no TTLs, so no overlay and no crash."""
    session = FakeSession()
    session.events = {}
    fig = panel()
    plot_utils._add_ttl_pulses(fig, session, {"show_ttl_pulses": True})
    assert not fig.layout.shapes


# --- structural: only one place may decide the anchor ----------------------

def test_only_session_derives_the_behavior_anchor():
    """No module may call behavior_start_in_pl2 itself.

    Getting its arguments wrong is silent — leave off `analog_fragments` and it
    returns the end-anchored answer rather than raising. `Session.behavior_anchor`
    is the single place that assembles them, so anything else calling the resolver
    directly is the bug this file exists for, reappearing.
    """
    allowed = {"alignment.py", "session.py"}
    offenders = []
    for path in Path("src/neurodash").rglob("*.py"):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", None))
            if name == "behavior_start_in_pl2":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"{offenders} calls behavior_start_in_pl2 directly — use "
        f"session.behavior_anchor, or the fragments argument will eventually be "
        f"forgotten and the answer will be ~1 s out without failing")
