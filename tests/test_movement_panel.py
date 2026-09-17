"""The open-field movement panel: three variables, one normalized axis.

Mobility (%), Velocity (cm/s) and Activity (%) are three different measurements
on three unrelated scales, and two of them share a unit while differing by ~13x
in range. Putting them on one panel only works because each is divided by its own
high percentile, and that has two consequences worth pinning: the panel gives up
its right-hand axis, and the drawn y is no longer a readable value.

The second is the one that would rot quietly. Hover reads the real number out of
`customdata`, so a refactor that drops customdata leaves a panel that still looks
perfect and reports "Mobility: 0.19" on mouseover.
"""

import numpy as np
import pandas as pd
import pytest

from neurodash import config
from neurodash.plot_utils import movement_scale, plot_session_view
from neurodash.session import Session

N = 300


def behavior_frame(with_activity=True):
    """EthoVision-shaped, with the three variables at their real relative scales.

    The magnitudes matter: on the test session Mobility tops out near 30 and
    Activity near 1.5, which is what makes a shared raw axis unusable.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "Recording time": np.arange(N) / 30.0,
        "X center": rng.uniform(-20, 20, N),
        "Y center": rng.uniform(-20, 20, N),
        "Velocity": rng.uniform(0, 14, N),
        "Mobility": rng.uniform(0, 30, N),
    })
    if with_activity:
        frame["Activity"] = rng.uniform(0, 1.5, N)
    frame.attrs["format"] = "ethovision"
    return frame


class BehaviorOnlySession(Session):
    """A session with a behavior file and no .pl2 behind it."""

    def __init__(self, frame):
        self.events = {}
        self.segment = None
        self.behavior_metadata = {}
        self.behavior_data = frame
        self.behavior_path = "trial.xlsx"
        self.block = None
        self.pl2_path = None
        self.analog_signal_summaries = []
        self.lfp_signal_index = 0
        self.bank_index = None
        self.epoch_params = None
        self.no_shock = False


def figure(with_activity=True):
    fig, _px = plot_session_view(BehaviorOnlySession(behavior_frame(with_activity)), {})
    return fig


def movement_traces(fig):
    return [t for t in fig.data if t.legendgroup]


# --- the panel gave up its second axis ------------------------------------

def test_the_movement_panel_consumes_one_y_axis_slot():
    """It used to need two (% on the left, cm/s on the right).

    `_axis_refs` exists because a secondary-y panel offsets every panel below it,
    so a panel that stops needing one is worth pinning: the figure here is
    movement + position, which is two panels and must now be two y-axes.
    """
    fig = figure()
    y_axes = [k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")]
    assert len(y_axes) == 2, f"expected one y-axis per panel, got {y_axes}"


def test_the_axis_says_what_the_numbers_are():
    """Not what the panel is - that is the title's job.

    "0-1 because each variable was divided by its own percentile" is the one
    thing a reader could get wrong here, so the axis has to say "Normalized".
    """
    fig = figure()
    assert fig.layout.yaxis.title.text == "Movement (Normalized)"
    low, high = fig.layout.yaxis.range
    assert low == pytest.approx(0, abs=0.05) and high == pytest.approx(1, abs=0.05)


def test_the_panel_is_titled_above_itself():
    fig = figure()
    titles = [a for a in fig.layout.annotations if a.text == "Movement Variables"]
    assert len(titles) == 1, "expected exactly one panel title"
    assert titles[0].yanchor == "bottom" and titles[0].y == 1.0, (
        "the title must sit above the panel, not inside it")


# --- normalization actually separates the scales ---------------------------

def test_each_variable_fills_the_axis_rather_than_the_floor():
    """The whole point: unnormalized, Activity's range is ~9% of Mobility's.

    Every variable must reach the top of the panel, or one of them is unreadable
    and the shared axis has bought nothing.
    """
    for trace in movement_traces(figure()):
        peak = np.nanmax(trace.y)
        assert 0.8 <= peak <= 1.2, f"{trace.name} peaks at {peak:.2f} of the axis"


def test_the_divisor_is_the_configured_percentile():
    values = np.arange(1000, dtype=float)
    assert movement_scale(values) == pytest.approx(
        np.nanpercentile(values, config.BEHAVIOR_YMAX_PERCENTILE))


def test_a_degenerate_column_does_not_blow_up_the_panel():
    """All-zero or all-NaN must draw flat, not divide by zero."""
    assert movement_scale(np.zeros(10)) == 1.0
    assert movement_scale(np.full(10, np.nan)) == 1.0


# --- hover still reads real units ------------------------------------------

def test_hover_reports_the_real_value_not_the_normalized_one():
    """customdata carries the unscaled series; the template reads from it."""
    frame = behavior_frame()
    fig = plot_session_view(BehaviorOnlySession(frame), {})[0]
    for trace in movement_traces(fig):
        label = trace.legendgroup
        assert "customdata" in trace.hovertemplate, (
            f"{label} hover reads y, which is normalized")
        assert np.allclose(np.asarray(trace.customdata, dtype=float),
                           frame[label].to_numpy(dtype=float))


def test_hover_names_the_real_unit():
    units = {"Mobility": "%", "Velocity": " cm/s", "Activity": "%"}
    for trace in movement_traces(figure()):
        assert units[trace.legendgroup] in trace.hovertemplate


# --- the legend ------------------------------------------------------------

def test_one_legend_entry_per_variable():
    """Not one per trace: raw and binned share a group so they toggle together."""
    fig = figure()
    assert [t.name for t in fig.data if t.showlegend] == [
        "Mobility", "Velocity", "Activity"]


def test_the_legend_sits_inside_its_own_panel():
    """It labels that panel, so it has to follow it - `_plot_position` puts its X
    and Y labels inside the plot area for the same reason.

    Positioned off the subplot's measured domain rather than a constant, which is
    what makes it move when panels are added or removed above it.
    """
    fig = figure()
    movement = fig.get_subplot(1, 1)   # movement is the top panel here
    x0, x1 = movement.xaxis.domain
    y0, y1 = movement.yaxis.domain
    legend = fig.layout.legend
    assert x0 <= legend.x < x1, "legend is outside the panel horizontally"
    assert y0 < legend.y <= y1, "legend is outside the panel vertically"
    assert legend.xanchor == "left" and legend.yanchor == "top"
    # Inset from the measured domain, not stated as a constant. Every figure in
    # this file happens to put the movement panel on top, so a hardcoded 1.0
    # would pass the bounds check above; a real session with neural panels above
    # it would then strand the legend over the LFP.
    assert legend.x == pytest.approx(x0 + 0.004)
    assert legend.y == pytest.approx(y1 - 0.008)


def test_no_other_panel_puts_anything_in_the_legend():
    """The legend is figure-level, so every other trace has to opt out."""
    fig = figure()
    assert not [t for t in fig.data if t.showlegend and not t.legendgroup]


def test_a_figure_without_the_panel_shows_no_legend():
    """Otherwise plotly draws an empty legend box in the corner."""
    frame = behavior_frame()
    frame = frame.drop(columns=["Velocity"])   # the panel is gated on Velocity
    frame.attrs["format"] = "ethovision"
    fig = plot_session_view(BehaviorOnlySession(frame), {})[0]
    assert fig.layout.showlegend is False


# --- Activity is optional --------------------------------------------------

def test_activity_is_optional():
    """Every open-field file exported before Sep 2026 lacks the column."""
    fig = figure(with_activity=False)
    assert [t.name for t in fig.data if t.showlegend] == ["Mobility", "Velocity"]
    assert fig.layout.yaxis.title.text == "Movement (Normalized)"
