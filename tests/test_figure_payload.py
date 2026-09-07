"""Guards on what a control change costs to send.

Changing one epoch buffer used to rebuild and re-send the entire figure: 119.6 MB
and about ten seconds on a real five-channel acquisition session, for an edit
that moves coloured rectangles. Three separate causes, each pinned below.

None of this needs a .pl2 — every fact here is about how the figure is
constructed and wired, not about the data in it.
"""

import ast
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from neurodash import callbacks, plot_utils

CALLBACKS_SRC = Path(callbacks.__file__)


def decorator_args(function_name):
    """The @callback(...) arguments for one callback, as source strings."""
    tree = ast.parse(CALLBACKS_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "callback":
                    # ast.unparse emits single quotes; normalize so the
                    # assertions can be written the way the source reads.
                    return [ast.unparse(a).replace("'", '"') for a in dec.args]
    raise AssertionError(f"{function_name} carries no @callback")


# --- 1. epoch edits must not trigger a full figure rebuild -----------------

def test_epoch_params_is_a_state_on_update_figure():
    """As an Input it rebuilt and re-sent the whole figure on every edit.

    This is the regression that costs ~10 s and is invisible in any functional
    test: the figure would still be correct, just slow.
    """
    args = decorator_args("update_figure")
    epoch = [a for a in args if "store-epoch-params" in a]
    assert epoch, "update_figure no longer reads the epoch params at all"
    assert all(a.startswith("State(") for a in epoch), (
        f"store-epoch-params must be a State on update_figure, got {epoch}. "
        "As an Input, every buffer edit re-sends the entire figure.")


def test_epoch_overlay_callback_owns_the_redraw():
    args = decorator_args("update_epoch_overlay")
    assert any(a.startswith('Input("store-epoch-params"') for a in args)
    assert any("main-plot" in a and "figure" in a for a in args)


# --- 2. the LFP traces must not each carry their own copy of the x array ---

def test_lfp_traces_use_x0_dx_not_an_x_array():
    """Five channels sharing one time base serialized it five times: 1.29M
    timestamps per trace, most of a 119.6 MB payload."""
    fig = make_subplots(rows=1, cols=1)
    session = _StubNeuralSession(n_channels=3, n_samples=5000)
    plot_utils._plot_lfp(fig, 1, session, {"raw_channel_indices": [0, 1, 2]})

    assert len(fig.data) == 3
    for trace in fig.data:
        assert trace.x is None, "LFP trace carries an explicit x array again"
        assert trace.x0 is not None and trace.dx is not None
        assert trace.y.dtype == np.float32, "float64 y doubles the payload"


def test_normalize_returns_float32():
    out, _ = plot_utils.normalize_lfp_traces([np.arange(10, dtype=np.float64)])
    assert out[0].dtype == np.float32


# --- 3. overlays must be assigned in bulk, not one shape at a time ---------

def test_overlays_do_not_use_row_all_helpers():
    """add_vrect/add_vline with row="all" re-validate the whole shapes list on
    every call. Measured on a real figure: 108 shapes took 2257 ms added one at
    a time against 13 ms assigned in bulk."""
    src = Path(plot_utils.__file__).read_text(encoding="utf-8")
    for helper in ("add_vrect", "add_vline"):
        assert f"fig.{helper}(" not in src, (
            f"{helper} is back in plot_utils; it is quadratic in shape count. "
            "Build the shape dicts and pass them to _extend_overlay instead.")


def test_extend_overlay_preserves_existing_annotations():
    """The LFP channel labels are annotations too — assigning wholesale ate them."""
    fig = go.Figure()
    fig.add_annotation(text="AI17", x=0, y=0)
    plot_utils._extend_overlay(fig, [], [dict(text="baseline", x=1, y=1)])
    assert [a.text for a in fig.layout.annotations] == ["AI17", "baseline"]


def test_overlay_axis_refs_skips_panels_with_no_data():
    """Reproduces exclude_empty_subplots: no bands over an empty LFP panel."""
    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=3, col=1)
    refs = plot_utils._overlay_axis_refs(fig)
    assert len(refs) == 2, f"row 2 has no traces and must be skipped: {refs}"
    assert refs == [("x", "y"), ("x3", "y3")]


class _StubNeuralSession:
    """Just enough Session for _plot_lfp: a signal, labels, and an offset."""

    def __init__(self, n_channels, n_samples):
        self._n = n_samples
        self._c = n_channels
        self.lfp_signal_index = 0
        self.block = None
        self.neural_time_offset = 0.0

    @property
    def lfp_info(self):
        return {"channel_labels": [f"FP{i:02d}" for i in range(self._c)],
                "duration_sec": self._n / 1000.0,
                "sampling_rate_hz": 1000.0}


@pytest.fixture(autouse=True)
def _stub_signal_access(monkeypatch):
    """_plot_lfp reaches into neural_io; give it arrays instead of a .pl2."""
    n = 5000

    def fake_get_analog_signal(block, index):
        return object()

    def fake_extract(sig, channel_index, start, duration, offset=0.0):
        t = np.arange(n) / 1000.0 + offset
        y = np.sin(t * (channel_index + 1))
        return t, y, 1000.0

    monkeypatch.setattr(plot_utils, "get_analog_signal", fake_get_analog_signal)
    monkeypatch.setattr(plot_utils, "extract_time_window", fake_extract)
