"""Plotting functions for neurodash3 session views.

Produces figures for the main panel. Takes a Session and controls
dict — no knowledge of UI state or data loading.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neurodash3.neural_io import get_analog_signal, extract_time_window
from neurodash3.behavior_io import extract_position
from neurodash3.session import compute_spectrogram


def plot_behavior_view(session):
    """Render position + velocity subplots for the behavior panel.

    Returns None if no behavior data.
    """
    if not session.has_behavior:
        return None

    t = session.behavior_data["Recording time"].to_numpy(dtype=float)
    x = session.behavior_data["X center"].to_numpy(dtype=float)
    y = session.behavior_data["Y center"].to_numpy(dtype=float)
    v = session.behavior_data["Velocity"].to_numpy(dtype=float)

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                        row_heights=[0.5, 0.5])

    fig.add_trace(go.Scatter(x=t, y=x, mode="lines",
                             line=dict(color="cyan", width=0.8), name="X"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines",
                             line=dict(color="magenta", width=0.8), name="Y"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=v, mode="lines",
                             line=dict(color="seagreen", width=0.8), name="Vel"),
                  row=2, col=1)

    fig.update_yaxes(title_text="Position (cm)", fixedrange=True, row=1, col=1)
    fig.update_yaxes(title_text="Velocity (cm/s)",
                     range=list(velocity_ylim(v)), fixedrange=True, row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(autorange=False)

    fig.update_layout(
        margin=dict(l=60, r=10, t=20, b=40),
        showlegend=False,
        hovermode=False,
        dragmode="pan",
    )
    return fig


def plot_session_view(session, controls):
    """Render the main panel figure for the current session and controls.

    Handles neural-only, behavior-only, and combined sessions.
    Returns None if there is nothing to plot.
    """
    panels = []  # list of (label, plot_fn) to stack vertically

    if session.has_neural:
        panels.append(("neural", _plot_lfp))
        if controls.get("show_spectrogram"):
            panels.append(("spectrogram", _plot_spectrogram))

    if session.has_behavior:
        if controls.get("show_velocity", True):
            panels.append(("velocity", _plot_velocity))
        if controls.get("show_position", True):
            panels.append(("position", _plot_position))

    if not panels:
        return None

    n = len(panels)
    # Spectrogram is most important. LFP is 2/3 of spectrogram. Behavioral panels smallest.
    def _height(label):
        if label == "spectrogram": return 3
        if label == "neural": return 2
        return 1
    height_ratios = [_height(label) for label, _ in panels]
    total = sum(height_ratios)
    row_heights = [h / total for h in height_ratios]

    fig = make_subplots(
        rows=n, cols=1,
        row_heights=row_heights,
        vertical_spacing=0.05,
    )

    for i, (label, plot_fn) in enumerate(panels, start=1):
        plot_fn(fig, i, session, controls)

    # Link all x-axes to row 1 for shared panning/zooming.
    # Using matches rather than shared_xaxes=True avoids a plotly rendering
    # issue where go.Heatmap doesn't display on non-anchor shared x-axes.
    for i in range(2, n + 1):
        fig.update_xaxes(matches='x', row=i, col=1)
    # Hide tick labels on all but the bottom row.
    for i in range(1, n):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    fig.update_xaxes(title_text="Time (s)", row=n, col=1)
    fig.update_xaxes(autorange=False)
    fig.update_layout(
        height=250 * n,
        margin=dict(l=60, r=20, t=20, b=40),
        dragmode="pan",
        showlegend=False,
        hovermode=False,
    )

    return fig


# ---------------------------------------------------------------------------
# Shared display helpers
# ---------------------------------------------------------------------------

def velocity_ylim(v):
    """Return (y_min, y_max) for velocity display.

    Parameters
    ----------
    v : np.ndarray

    Returns
    -------
    (float, float)
    """
    return (-1.0, float(np.nanpercentile(v, 99.5)))


def normalize_lfp_traces(ys):
    """Normalize LFP traces for stacked waterfall display.

    Mean-subtracts and std-normalizes each trace, then applies a fixed
    per-channel offset. Returns the normalized+offset arrays and the
    y-axis range needed to contain them.

    Parameters
    ----------
    ys : list of np.ndarray

    Returns
    -------
    normalized : list of np.ndarray
    y_range : (float, float)
    """
    normalized = []
    for i, y in enumerate(ys):
        y_norm = (y - np.mean(y)) / (np.nanstd(y) + 1e-12) * 0.7 + i * 5.0
        normalized.append(y_norm)
    n = len(ys)
    return normalized, (-6.0, (n - 1) * 5.0 + 6.0)


# ---------------------------------------------------------------------------
# Panel plot functions
# ---------------------------------------------------------------------------

def _plot_lfp(fig, row, session, controls):
    raw_channel_indices = controls.get("raw_channel_indices") or [0]
    sig = get_analog_signal(session.block, controls.get("analog_signal_index", 0))
    sig_info = session.analog_signal_summaries[0]
    channel_labels = sig_info["channel_labels"]
    full_duration = sig_info["duration_sec"]

    if not raw_channel_indices:
        axis_suffix = "" if row == 1 else str(row)
        fig.add_annotation(
            text="No channels selected.",
            xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix} domain",
            x=0.5, y=0.5, showarrow=False,
        )
        fig.update_yaxes(title_text="LFP", row=row, col=1)
        return

    ts = None
    ys = []
    for ch in raw_channel_indices:
        t, y, sr = extract_time_window(sig, ch, 0, full_duration)
        if ts is None:
            ts = t
        ys.append((ch, y))

    normalized, y_range = normalize_lfp_traces([y for _, y in ys])
    axis_suffix = "" if row == 1 else str(row)
    for i, (ch, y_norm) in enumerate(zip([ch for ch, _ in ys], normalized)):
        offset = i * 5.0
        fig.add_trace(
            go.Scattergl(
                x=ts, y=y_norm,
                mode="lines",
                line=dict(width=0.8),
            ),
            row=row, col=1,
        )
        fig.add_annotation(
            x=0.01, y=float(offset),
            xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix}",
            text=channel_labels[ch],
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(size=11, color="white"),
            bgcolor="rgba(0,0,0,0.45)",
            borderpad=2,
        )

    fig.update_yaxes(
        title_text="LFP", showticklabels=False,
        range=list(y_range), fixedrange=True,
        row=row, col=1,
    )


def _plot_spectrogram(fig, row, session, controls):
    sig_info = session.analog_signal_summaries[0]
    channel_labels = sig_info["channel_labels"]
    full_duration = sig_info["duration_sec"]
    ch = controls.get("spectrogram_channel_index", 0)

    try:
        freqs, times, power_db = compute_spectrogram(
            str(session.pl2_path),
            controls.get("analog_signal_index", 0),
            ch,
            0,
            full_duration,
            controls.get("spect_max_freq", 90.0),
            controls.get("spect_window_sec", 2.0),
            controls.get("spect_step_sec", 0.1),
            controls.get("spect_c_param", 20),
        )
    except Exception as e:
        print(f"ERROR in _plot_spectrogram: {e}")
        import traceback
        traceback.print_exc()
        axis_suffix = "" if row == 1 else str(row)
        fig.add_annotation(
            text=f"Spectrogram error: {e}",
            xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix} domain",
            x=0.5, y=0.5, showarrow=False,
        )
        return

    fig.add_trace(
        go.Heatmap(
            x=times, y=freqs, z=power_db,
            colorscale="Inferno",
            showscale=False,
            name=f"Spectrogram ({channel_labels[ch]})",
        ),
        row=row, col=1,
    )
    fig.update_yaxes(
        title_text=f"Freq (Hz) — {channel_labels[ch]}",
        range=[0, controls.get("spect_max_freq", 90.0)], fixedrange=True,
        row=row, col=1,
    )


def _plot_position(fig, row, session, controls):
    t, x, y = extract_position(session.behavior_data, point="center")
    axis_suffix = "" if row == 1 else str(row)

    fig.add_trace(
        go.Scatter(x=t, y=x, mode="lines", line=dict(color="steelblue", width=0.8)),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=y, mode="lines", line=dict(color="coral", width=0.8)),
        row=row, col=1,
    )
    fig.add_annotation(
        x=0.01, y=0.95, text="X",
        xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix} domain",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=11, color="steelblue"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2,
    )
    fig.add_annotation(
        x=0.01, y=0.70, text="Y",
        xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix} domain",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=11, color="coral"),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2,
    )
    fig.update_yaxes(title_text="Position (cm)", fixedrange=True, row=row, col=1)


def _plot_velocity(fig, row, session, controls):
    t = session.behavior_data["Recording time"].to_numpy(dtype=float)
    v = session.behavior_data["Velocity"].to_numpy(dtype=float)

    fig.add_trace(
        go.Scatter(
            x=t, y=v,
            mode="lines",
            name="Velocity",
            line=dict(color="seagreen", width=0.8),
        ),
        row=row, col=1,
    )
    fig.update_yaxes(
        title_text="Velocity (cm/s)",
        range=list(velocity_ylim(v)), fixedrange=True,
        row=row, col=1,
    )
