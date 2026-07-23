"""Plotting functions for neurodash session views.

Produces figures for the main panel. Takes a Session and controls
dict — no knowledge of UI state or data loading.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neurodash import config
from neurodash.neural_io import get_analog_signal, extract_time_window
from neurodash.behavior_io import extract_position
from neurodash.session import compute_spectrogram, compute_theta_channels


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

    Handles neural-only, behavior-only, and combined sessions. Returns
    ``(figure, content_px)`` where content_px is the figure's natural height
    (panel weight x SESSION_PANEL_UNIT_PX) so the caller can size it to content
    and cap it at the viewport; ``(None, 0)`` when there is nothing to plot.
    """
    panels = []  # list of (label, plot_fn) to stack vertically

    if session.has_neural:
        panels.append(("neural", _plot_lfp))
        if controls.get("show_spectrogram"):
            panels.append(("spectrogram", _plot_spectrogram))
        # Theta peak overlays on the spectrogram when it's shown (see
        # _plot_spectrogram); it only needs its own panel when the heatmap is off.
        if controls.get("show_theta_peak") and not controls.get("show_spectrogram"):
            panels.append(("theta_peak", _plot_theta_peak))
        if controls.get("show_theta_power"):
            panels.append(("theta_power", _plot_theta_power))

    if session.has_behavior:
        if controls.get("show_velocity", True):
            panels.append(("velocity", _plot_velocity))
        if controls.get("show_position", True):
            panels.append(("position", _plot_position))
        # Mobility is a continuous % channel; skip gracefully if the file lacks it.
        if controls.get("show_mobility", True) and "Mobility" in session.behavior_data.columns:
            panels.append(("mobility", _plot_mobility))

    if not panels:
        return None, 0

    n = len(panels)
    # Spectrogram is most important. LFP is 2/3 of spectrogram. Behavioral panels smallest.
    def _height(label):
        if label == "spectrogram": return 3
        if label == "neural": return 2
        return 1
    height_ratios = [_height(label) for label, _ in panels]
    total = sum(height_ratios)
    row_heights = [h / total for h in height_ratios]
    content_px = total * config.SESSION_PANEL_UNIT_PX

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
    # Scrub crosshair: a vertical spike line at the cursor's time spanning every
    # panel (spikemode="across"), following the mouse (spikesnap="cursor"). With
    # hovermode="x" each panel shows its value at that time as you move over it.
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikethickness=1, spikedash="dot", spikecolor="#888",
    )
    # No fixed height — the graph is responsive and fills its container (the
    # viewer pane sizes it to the viewport), so all panels fit without scrolling.
    fig.update_layout(
        autosize=True,
        margin=dict(l=60, r=20, t=20, b=40),
        dragmode="pan",
        showlegend=False,
        hovermode="x",
        spikedistance=-1,
    )

    return fig, content_px


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
    return (-1.0, float(np.nanpercentile(v, 99.9)))


def mobility_ylim(m):
    """Return (y_min, y_max) for mobility (%) display.

    Most of the action sits in the low percentages, so cap the top at a high
    percentile rather than the full 0–100 range. Floored at 0 (a percentage
    can't be negative).

    Parameters
    ----------
    m : np.ndarray

    Returns
    -------
    (float, float)
    """
    return (0.0, float(np.nanpercentile(m, 99.9)))


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
                hoverinfo="skip",  # LFP is normalized for display; its y isn't µV
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
            hovertemplate="%{y:.1f} Hz, %{z:.1f} dB<extra></extra>",
        ),
        row=row, col=1,
    )

    # Overlay theta peak (a frequency) as a black line+dots when requested — it
    # rides on the spectrogram's own frequency axis, most visible where theta is
    # strong (matches the marker='.' overlay from the notebook). Very light dashed
    # guides mark the theta band edges (the window the peak is searched in).
    if controls.get("show_theta_peak"):
        low, high = controls.get(
            "theta_band", (config.DEFAULT_THETA_LOW_HZ, config.DEFAULT_THETA_HIGH_HZ))
        for edge in (low, high):
            fig.add_hline(
                y=edge, row=row, col=1,
                line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"),
            )
        try:
            t_theta, peak, _power = _theta_channels(session, controls)
            fig.add_trace(_theta_peak_trace(t_theta, peak, controls), row=row, col=1)
        except Exception as e:
            print(f"ERROR overlaying theta peak: {e}")

    fig.update_yaxes(
        title_text=f"Freq (Hz) — {channel_labels[ch]}",
        range=[0, controls.get("spect_max_freq", 90.0)], fixedrange=True,
        row=row, col=1,
    )


# --- Theta channels (peak frequency + band power) — derived from the
# spectrogram on the shared analysis channel. Both call the cached
# compute_theta_channels, so the two panels share one computation. ---

def _theta_channels(session, controls):
    """Fetch (times, peak_hz, power_db) for the analysis channel from cache."""
    sig_info = session.analog_signal_summaries[0]
    ch = controls.get("spectrogram_channel_index", 0)
    band = controls.get("theta_band",
                        (config.DEFAULT_THETA_LOW_HZ, config.DEFAULT_THETA_HIGH_HZ))
    return compute_theta_channels(
        str(session.pl2_path),
        controls.get("analog_signal_index", 0),
        ch,
        sig_info["duration_sec"],
        controls.get("spect_max_freq", config.DEFAULT_SPECT_MAX_FREQ),
        controls.get("spect_window_sec", config.DEFAULT_SPECT_WINDOW_SEC),
        controls.get("spect_step_sec", config.DEFAULT_SPECT_STEP_SEC),
        controls.get("spect_c_param", config.DEFAULT_SPECT_C_PARAM),
        band[0], band[1],
        config.DEFAULT_THETA_INTERP_STEP_HZ,
        config.DEFAULT_THETA_SMOOTH_WIDTH,
    )


def _theta_error(fig, row, message):
    axis_suffix = "" if row == 1 else str(row)
    fig.add_annotation(
        text=message,
        xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix} domain",
        x=0.5, y=0.5, showarrow=False,
    )


def _theta_peak_trace(x, y, controls):
    """The theta-peak line, styled by the right-sidebar controls (color + dots).

    Shared by the spectrogram overlay and the standalone panel so they match.
    """
    color = controls.get("theta_peak_color", "black")
    mode = "lines+markers" if controls.get("theta_peak_markers", True) else "lines"
    size = controls.get("theta_peak_dot_size", config.DEFAULT_THETA_DOT_SIZE)
    return go.Scattergl(
        x=x, y=y, mode=mode,
        line=dict(color=color, width=1.0),
        marker=dict(color=color, size=size),
        name="Theta peak",
        hovertemplate="θ peak: %{y:.2f} Hz<extra></extra>",
    )


def _plot_theta_peak(fig, row, session, controls):
    band = controls.get("theta_band",
                        (config.DEFAULT_THETA_LOW_HZ, config.DEFAULT_THETA_HIGH_HZ))
    try:
        times, peak, _power = _theta_channels(session, controls)
    except Exception as e:
        print(f"ERROR in _plot_theta_peak: {e}")
        _theta_error(fig, row, f"Theta peak error: {e}")
        return
    fig.add_trace(_theta_peak_trace(times, peak, controls), row=row, col=1)
    # Channel is shown on the spectrogram/LFP; keep this title short (theta glyph)
    # so it doesn't crowd its neighbors on a thin panel.
    fig.update_yaxes(
        title_text="θ Peak (Hz)",
        range=[band[0], band[1]], fixedrange=True,
        row=row, col=1,
    )


def _plot_theta_power(fig, row, session, controls):
    try:
        times, _peak, power = _theta_channels(session, controls)
    except Exception as e:
        print(f"ERROR in _plot_theta_power: {e}")
        _theta_error(fig, row, f"Theta power error: {e}")
        return
    # Dots + size track the theta-peak controls (same time bins, so they line up);
    # colour stays teal to match this panel's line.
    mode = "lines+markers" if controls.get("theta_peak_markers", True) else "lines"
    size = controls.get("theta_peak_dot_size", config.DEFAULT_THETA_DOT_SIZE)
    fig.add_trace(
        go.Scattergl(x=times, y=power, mode=mode,
                     line=dict(color="teal", width=1.0),
                     marker=dict(color="teal", size=size),
                     hovertemplate="θ power: %{y:.1f} dB<extra></extra>"),
        row=row, col=1,
    )
    fig.update_yaxes(
        title_text="θ Power (dB)",
        range=list(_theta_power_ylim(power)), fixedrange=True,
        row=row, col=1,
    )


def _theta_power_ylim(power):
    """Robust y-range for the theta-power panel, padded off the 1/99 pcts."""
    lo = float(np.nanpercentile(power, 1))
    hi = float(np.nanpercentile(power, 99))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return (lo - pad, hi + pad)


def _plot_position(fig, row, session, controls):
    t, x, y = extract_position(session.behavior_data, point="center")
    axis_suffix = "" if row == 1 else str(row)

    fig.add_trace(
        go.Scatter(x=t, y=x, mode="lines", line=dict(color="steelblue", width=0.8),
                   hovertemplate="X: %{y:.1f} cm<extra></extra>"),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=y, mode="lines", line=dict(color="coral", width=0.8),
                   hovertemplate="Y: %{y:.1f} cm<extra></extra>"),
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
            hovertemplate="Velocity: %{y:.1f} cm/s<extra></extra>",
        ),
        row=row, col=1,
    )
    fig.update_yaxes(
        title_text="Velocity (cm/s)",
        range=list(velocity_ylim(v)), fixedrange=True,
        row=row, col=1,
    )


def _plot_mobility(fig, row, session, controls):
    t = session.behavior_data["Recording time"].to_numpy(dtype=float)
    m = session.behavior_data["Mobility"].to_numpy(dtype=float)

    fig.add_trace(
        go.Scatter(
            x=t, y=m,
            mode="lines",
            name="Mobility",
            line=dict(color="darkorange", width=0.8),
            hovertemplate="Mobility: %{y:.1f}%<extra></extra>",
        ),
        row=row, col=1,
    )
    fig.update_yaxes(
        title_text="Mobility (%)",
        range=list(mobility_ylim(m)), fixedrange=True,
        row=row, col=1,
    )


# ---------------------------------------------------------------------------
# Channel Viewer — one combined figure for all channels (own lazy-rendered view)
# ---------------------------------------------------------------------------

def plot_channel_figure(session, view_range=None, spect_params=None):
    """Combined channel figure: raw LFP (col 1) + spectrogram (col 2) per channel.

    All x-axes are linked (matches='x') so panning/zooming one panel moves every
    panel together — the single-figure sync the Session Viewer uses. One figure
    is one WebGL context regardless of channel count, so full-resolution traces
    stay crisp at any zoom without a per-channel WebGL-context limit.

    view_range, if given, sets the initial [t0, t1] window (else shows the full
    recording). spect_params (window/step/c/max_freq) tunes the spectrograms;
    missing keys fall back to config.DEFAULT_SPECT_*.
    """
    sig_info = session.analog_signal_summaries[0]
    channel_indices = sig_info["channel_indices"]
    channel_labels = sig_info["channel_labels"]
    full_duration = sig_info["duration_sec"]
    n = len(channel_indices)
    sig = get_analog_signal(session.block, 0)

    sp = spect_params or {}
    window = sp.get("window") or config.DEFAULT_SPECT_WINDOW_SEC
    step = sp.get("step") or config.DEFAULT_SPECT_STEP_SEC
    c_param = sp.get("c") or config.DEFAULT_SPECT_C_PARAM
    max_freq = sp.get("max_freq") or config.DEFAULT_SPECT_MAX_FREQ

    fig = make_subplots(
        rows=n, cols=2,
        column_widths=[0.5, 0.5],
        vertical_spacing=0.0,   # contiguous rows so the controls gutter aligns
        horizontal_spacing=0.06,
    )

    for i, ch in enumerate(channel_indices):
        row = i + 1
        # --- raw LFP (col 1): full-resolution, normalized like the main viewer
        t, y, _ = extract_time_window(sig, ch, 0, full_duration)
        normalized, y_range = normalize_lfp_traces([y])
        fig.add_trace(
            go.Scattergl(x=t, y=normalized[0], mode="lines",
                         line=dict(width=0.7, color="#1f77b4")),
            row=row, col=1,
        )
        fig.update_yaxes(title_text=channel_labels[ch], showticklabels=False,
                         range=list(y_range), fixedrange=True, row=row, col=1)

        # --- spectrogram (col 2): full recording, cached
        try:
            freqs, times, power_db = compute_spectrogram(
                str(session.pl2_path), 0, ch, 0, full_duration,
                max_freq, window, step, c_param,
            )
            fig.add_trace(
                go.Heatmap(x=times, y=freqs, z=power_db,
                           colorscale="Inferno", showscale=False),
                row=row, col=2,
            )
        except Exception as e:
            print(f"Channel spectrogram error ch{ch}: {e}")
        fig.update_yaxes(title_text="Hz", range=[0, max_freq],
                         fixedrange=True, row=row, col=2)

    # Link every x-axis to the anchor so all panels pan/zoom together.
    for r in range(1, n + 1):
        for c in (1, 2):
            if r == 1 and c == 1:
                continue
            fig.update_xaxes(matches='x', row=r, col=c)
    # Only the bottom row shows time ticks.
    for r in range(1, n):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
        fig.update_xaxes(showticklabels=False, row=r, col=2)
    fig.update_xaxes(title_text="Time (s)", row=n, col=1)
    fig.update_xaxes(title_text="Time (s)", row=n, col=2)

    if view_range:
        fig.update_xaxes(range=list(view_range), autorange=False)

    fig.update_layout(
        height=config.CHANNEL_ROW_HEIGHT * n + 50,  # + top/bottom margins below
        margin=dict(l=50, r=10, t=10, b=40),
        dragmode="pan",
        showlegend=False,
        hovermode=False,
    )
    return fig
