"""Dash callbacks for neurodash."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from dash import (
    Input, Output, State, ALL, callback, clientside_callback, ctx, dcc, html, no_update,
)

from neurodash.config import (
    DEFAULT_VIEW_DURATION, DEFAULT_SPECT_MAX_FREQ,
    DEFAULT_SPECT_WINDOW_SEC, DEFAULT_SPECT_STEP_SEC, DEFAULT_SPECT_C_PARAM,
    DEFAULT_THETA_LOW_HZ, DEFAULT_THETA_HIGH_HZ, DEFAULT_THETA_INTERP_STEP_HZ,
    DEFAULT_THETA_SMOOTH_WIDTH, DEFAULT_THETA_DOT_SIZE,
    DEFAULT_THETA_ESTIMATOR,
    DEFAULT_THETA_RATIO_LOW_BAND, DEFAULT_THETA_RATIO_HIGH_BAND,
    AUTOLOAD_ON_STARTUP, AUTOLOAD_NEURAL_PATH, AUTOLOAD_BEHAVIOR_PATH,
    EXEMPLAR_SEEDS_DEFAULT_CHANNEL, EXPORT_DIR,
)
from neurodash.app_state import last_browse_dir, remember_browse_dir
from neurodash.file_picker import pick_file, pick_save_path, pick_directory
from neurodash import merge
from neurodash.behavior_io import (
    get_display_metadata, load_behavior_notes, save_behavior_notes,
    missing_columns, missing_columns_message,
)
from neurodash.export import build_analysis_table
from neurodash.neural_io import get_analog_signal, extract_time_window, build_neural_table
from neurodash.plot_utils import normalize_lfp_traces, plot_session_view, plot_channel_figure
from neurodash.session import (
    load_session_from_paths, compute_spectrogram, compute_theta_channels,
)
from neurodash.channel_io import load_channels, save_channels
from neurodash.layout import build_channel_view, exemplar_glyph, exemplar_button_style


# ---------------------------------------------------------------------------
# File browse callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("store-neural-path", "data"),
    Output("div-neural-filename", "children"),
    Output("div-neural-metadata", "children"),
    Output("dropdown-channels", "options"),
    Output("dropdown-channels", "value"),
    Output("dropdown-spectrogram-channel", "options"),
    Output("dropdown-spectrogram-channel", "value"),
    Input("btn-browse-neural", "n_clicks"),
)
def browse_neural(n_clicks):
    if not n_clicks:
        # Initial page load — autoload the default file (temporary dev convenience).
        if not AUTOLOAD_ON_STARTUP or not Path(AUTOLOAD_NEURAL_PATH).exists():
            return (no_update,) * 7
        path = AUTOLOAD_NEURAL_PATH
    else:
        path = pick_file("Select .pl2 file", "Plexon (*.pl2)", last_browse_dir())
    if not path:
        return (no_update,) * 7
    remember_browse_dir(path)

    session = load_session_from_paths(path, "")
    sig_info = session.analog_signal_summaries[0]
    options = [
        {"label": lbl, "value": idx}
        for idx, lbl in zip(sig_info["channel_indices"], sig_info["channel_labels"])
    ]

    # Channel sidecar — source of the (filename-inferred) animal ID and exemplar.
    channel_data = load_channels(path, session)

    # Build metadata display
    dur = sig_info["duration_sec"]
    minutes = int(dur // 60)
    seconds = dur % 60
    rec_dt = session.rec_datetime
    dt_str = rec_dt.strftime("%Y-%m-%d %H:%M") if rec_dt else "Unknown"

    metadata = html.Div([
        html.Div(f"Mouse: {channel_data.get('animal') or 'Unknown'}"),
        html.Div(f"Recorded: {dt_str}"),
        html.Div(f"Duration: {minutes}m {seconds:.1f}s"),
        html.Div(f"Sampling rate: {sig_info['sampling_rate_hz']:.0f} Hz"),
        html.Div(f"Channels: {sig_info['n_channels']}"),
    ])

    # Seed the default selected channel from a saved exemplar (if any).
    default_channel = 0
    if EXEMPLAR_SEEDS_DEFAULT_CHANNEL:
        ex = channel_data.get("exemplar_channel_index")
        if ex is not None and 0 <= ex < sig_info["n_channels"]:
            default_channel = ex

    return (path, Path(path).name, metadata, options,
            [default_channel], options, default_channel)


def _load_error(message):
    """A load failure shown in place of the metadata block — red, and wrapping,
    since these name specific columns and run long."""
    return html.Div(message, style={"color": "#b00", "fontSize": "0.85em",
                                    "whiteSpace": "normal"})


@callback(
    Output("store-behavior-path", "data"),
    Output("div-behavior-filename", "children"),
    Output("div-behavior-metadata", "children"),
    Input("btn-browse-behavior", "n_clicks"),
)
def browse_behavior(n_clicks):
    if not n_clicks:
        # Initial page load — autoload the default file (temporary dev convenience).
        if not AUTOLOAD_ON_STARTUP or not Path(AUTOLOAD_BEHAVIOR_PATH).exists():
            return no_update, no_update, no_update
        path = AUTOLOAD_BEHAVIOR_PATH
    else:
        path = pick_file("Select behavior file", "Excel (*.xlsx)", last_browse_dir())
    if not path:
        return no_update, no_update, no_update
    remember_browse_dir(path)

    try:
        session = load_session_from_paths("", path)
    except Exception as e:
        return no_update, Path(path).name, _load_error(f"Could not read this file: {e}")

    # Refuse a file we can't actually plot, and say which columns are missing.
    # Storing the path anyway would just move the failure into the figure callback,
    # where a bare KeyError surfaces as nothing happening at all.
    missing = missing_columns(session.behavior_data)
    if missing:
        return (no_update, Path(path).name,
                _load_error(missing_columns_message(path, missing)))

    info = get_display_metadata(session.behavior_metadata)

    # Format start_time as date string
    start = info["start_time"]
    date_str = start.strftime("%Y-%m-%d %H:%M") if hasattr(start, "strftime") else str(start or "Unknown")

    # Data quality line
    quality_parts = []
    for label, key in [("missed", "missed_samples_pct"),
                       ("not found", "subject_not_found_pct"),
                       ("interpolated", "interpolated_pct")]:
        val = info[key]
        if val is not None:
            quality_parts.append(f"{label} {val:.1f}%")
    quality_str = ", ".join(quality_parts) if quality_parts else "N/A"

    metadata = html.Div([
        html.Div(f"Mouse: {info['mouse_id'] or 'Unknown'}"),
        html.Div(f"Experiment: {info['experiment'] or 'Unknown'}"),
        html.Div(f"Trial: {info['trial_name'] or 'Unknown'}"),
        html.Div(f"Date: {date_str}"),
        html.Div(f"Duration: {info['recording_duration'] or 'Unknown'}"),
        html.Div(f"Arena name: {info['arena_name'] or 'Unknown'}"),
        html.Div(f"Arena ID: {info['arena_id'] or 'Unknown'}"),
        html.Div(f"Video: {info['video_filename'] or 'None'}"),
        html.Div(f"Quality: {quality_str}"),
    ])

    return path, Path(path).name, metadata


# ---------------------------------------------------------------------------
# Viewport tracking — clientside for zero latency
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(relayoutData, currentRange) {
        if (!relayoutData) return window.dash_clientside.no_update;
        var x0 = relayoutData["xaxis.range[0]"];
        var x1 = relayoutData["xaxis.range[1]"];
        if (x0 != null && x1 != null) return [x0, x1];
        return window.dash_clientside.no_update;
    }
    """,
    Output("store-view-range", "data"),
    Input("main-plot", "relayoutData"),
    State("store-view-range", "data"),
    prevent_initial_call=True,
)


# Panning/zooming the combined channel figure updates the same shared window.
clientside_callback(
    """
    function(relayoutData) {
        if (!relayoutData) return window.dash_clientside.no_update;
        var x0 = relayoutData["xaxis.range[0]"];
        var x1 = relayoutData["xaxis.range[1]"];
        if (x0 != null && x1 != null) return [x0, x1];
        return window.dash_clientside.no_update;
    }
    """,
    Output("store-view-range", "data", allow_duplicate=True),
    Input("channel-plot", "relayoutData"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Sync view start back to time input
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(viewRange) {
        if (!viewRange) return window.dash_clientside.no_update;
        var rounded = Math.round(viewRange[0]);
        // Only update if different to avoid unnecessary triggers
        var current = window._neurodash_last_synced_time;
        if (current === rounded) return window.dash_clientside.no_update;
        window._neurodash_last_synced_time = rounded;
        return rounded;
    }
    """,
    Output("input-jump-to-time", "value"),
    Input("store-view-range", "data"),
    prevent_initial_call=True,
)


# Scroll/zoom changes the view width — mirror it back into the View duration
# input so the RHS controls stay in sync. Guarded on the last-synced value so it
# doesn't re-trigger the view-controls callback in a loop.
clientside_callback(
    """
    function(viewRange) {
        if (!viewRange) return window.dash_clientside.no_update;
        var rounded = Math.round(viewRange[1] - viewRange[0]);
        if (rounded <= 0) return window.dash_clientside.no_update;
        var current = window._neurodash_last_synced_dur;
        if (current === rounded) return window.dash_clientside.no_update;
        window._neurodash_last_synced_dur = rounded;
        return rounded;
    }
    """,
    Output("input-window-duration", "value"),
    Input("store-view-range", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Spectrogram / theta controls visibility — show/hide with toggles
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(spectValue, thetaValue, tabValue) {
        var spectOn = spectValue && spectValue.indexOf("on") !== -1;
        var thetaOn = thetaValue && thetaValue.length > 0;
        var onChannels = tabValue === "channels";
        /* Analysis channel: any single-channel analysis, Session Viewer only.
           Spectrogram params: spectrogram OR theta on (theta rides on them),
           and always on the Channel Viewer. Theta band: theta on, Session
           Viewer only. */
        var showChannel = (spectOn || thetaOn) && !onChannels;
        var showSpectParams = spectOn || thetaOn || onChannels;
        var showTheta = thetaOn && !onChannels;
        return [
            {"display": showChannel ? "block" : "none"},
            {"display": showSpectParams ? "block" : "none"},
            {"display": showTheta ? "block" : "none"}
        ];
    }
    """,
    Output("div-analysis-channel", "style"),
    Output("div-spectrogram-controls", "style"),
    Output("div-theta-controls", "style"),
    Input("toggle-spectrogram", "value"),
    Input("toggle-theta", "value"),
    Input("tabs-main", "value"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# View controls — window duration and jump-to-time (clientside for speed)
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(windowDuration, jumpTo, currentRange) {
        var NU = window.dash_clientside.no_update;
        if (windowDuration == null || windowDuration <= 0) return NU;

        var ctx = window.dash_clientside.callback_context;
        var triggered = ctx.triggered.length ? ctx.triggered[0].prop_id : "";

        var x0 = currentRange ? currentRange[0] : 0;
        var x1 = currentRange ? currentRange[1] : windowDuration;

        if (triggered.indexOf("input-jump-to-time") !== -1) {
            // Only jump if user entered a value different from current position
            var currentStart = Math.round(currentRange ? currentRange[0] : -1);
            if (jumpTo === currentStart) return NU;
            x0 = (jumpTo != null && jumpTo >= 0) ? jumpTo : x0;
            x1 = x0 + windowDuration;
        } else {
            // Window duration changed. If it already matches the current view
            // width (e.g. it was just synced from a scroll/zoom), leave the view
            // alone — otherwise we'd re-center and undo the cursor-centered zoom.
            if (currentRange && Math.abs((x1 - x0) - windowDuration) < 0.5) return NU;
            var center = (x0 + x1) / 2;
            x0 = center - windowDuration / 2;
            x1 = center + windowDuration / 2;
        }

        // Relayout the primary x-axis; matches='x' propagates to others
        var graphDiv = document.querySelector("#main-plot .js-plotly-plot");
        if (graphDiv) {
            Plotly.relayout(graphDiv, {"xaxis.range": [x0, x1], "xaxis.autorange": false});
        }
        var channelDiv = document.querySelector("#channel-plot .js-plotly-plot");
        if (channelDiv) {
            Plotly.relayout(channelDiv, {"xaxis.range": [x0, x1], "xaxis.autorange": false});
        }

        return [x0, x1];
    }
    """,
    Output("store-view-range", "data", allow_duplicate=True),
    Input("input-window-duration", "value"),
    Input("input-jump-to-time", "value"),
    State("store-view-range", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Figure callback — single combined figure
# ---------------------------------------------------------------------------

def _ratio_band(low, high, default):
    """A theta-ratio sub-band from two sidebar inputs, falling back per-edge.

    Tuple (not list) because it is passed to the lru_cached compute_theta_channels
    and so has to be hashable. A blank or inverted pair falls back to the default
    rather than producing an empty band that would come back all-NaN.
    """
    low = default[0] if low is None else float(low)
    high = default[1] if high is None else float(high)
    return (low, high) if high > low else tuple(default)

@callback(
    Output("main-plot", "figure"),
    Output("main-plot", "style"),
    Output("div-plot-placeholder", "style"),
    Input("store-neural-path", "data"),
    Input("store-behavior-path", "data"),
    Input("dropdown-channels", "value"),
    Input("toggle-spectrogram", "value"),
    Input("dropdown-spectrogram-channel", "value"),
    Input("input-spect-window", "value"),
    Input("input-spect-step", "value"),
    Input("input-spect-c-param", "value"),
    Input("input-spect-max-freq", "value"),
    Input("toggle-theta", "value"),
    Input("input-theta-low", "value"),
    Input("input-theta-high", "value"),
    Input("radio-theta-peak-color", "value"),
    Input("toggle-theta-peak-markers", "value"),
    Input("input-theta-dot-size", "value"),
    Input("radio-theta-estimator", "value"),
    Input("input-theta-ratio-low-lo", "value"),
    Input("input-theta-ratio-low-hi", "value"),
    Input("input-theta-ratio-high-lo", "value"),
    Input("input-theta-ratio-high-hi", "value"),
    State("store-view-range", "data"),
)
def update_figure(neural_path, behavior_path, selected_channels, spect_toggle,
                  spect_channel, spect_window, spect_step, spect_c, spect_max_freq,
                  theta_toggle, theta_low, theta_high, peak_color, peak_markers,
                  peak_dot_size, theta_estimator,
                  ratio_low_lo, ratio_low_hi, ratio_high_lo, ratio_high_hi,
                  view_range):
    if not neural_path and not behavior_path:
        return no_update, no_update, no_update

    session = load_session_from_paths(neural_path or "", behavior_path or "")
    channels = selected_channels or [0]
    theta_toggle = theta_toggle or []
    controls = {
        "raw_channel_indices": channels,
        "show_spectrogram": "on" in (spect_toggle or []),
        "spectrogram_channel_index": spect_channel if spect_channel is not None else 0,
        "spect_window_sec": spect_window or DEFAULT_SPECT_WINDOW_SEC,
        "spect_step_sec": spect_step or DEFAULT_SPECT_STEP_SEC,
        "spect_c_param": spect_c or DEFAULT_SPECT_C_PARAM,
        "spect_max_freq": spect_max_freq or DEFAULT_SPECT_MAX_FREQ,
        "show_theta_band": "band" in theta_toggle,
        "show_theta_peak": "peak" in theta_toggle,
        "show_theta_power": "power" in theta_toggle,
        "show_theta_ratio": "ratio" in theta_toggle,
        "theta_band": (theta_low or DEFAULT_THETA_LOW_HZ,
                       theta_high or DEFAULT_THETA_HIGH_HZ),
        "theta_ratio_low_band": _ratio_band(ratio_low_lo, ratio_low_hi,
                                            DEFAULT_THETA_RATIO_LOW_BAND),
        "theta_ratio_high_band": _ratio_band(ratio_high_lo, ratio_high_hi,
                                             DEFAULT_THETA_RATIO_HIGH_BAND),
        "theta_peak_color": peak_color or "black",
        "theta_peak_markers": "dots" in (peak_markers or []),
        "theta_peak_dot_size": peak_dot_size or DEFAULT_THETA_DOT_SIZE,
        "theta_estimator": theta_estimator or DEFAULT_THETA_ESTIMATOR,
    }
    fig, content_px = plot_session_view(session, controls)
    if fig is None:
        return no_update, no_update, no_update

    x0, x1 = view_range or [0, DEFAULT_VIEW_DURATION]
    fig.update_xaxes(range=[x0, x1], autorange=False)
    # Size to content but never exceed the viewport: a lone LFP panel is a top
    # strip; a full stack fills the screen without scrolling.
    graph_height = f"min({content_px}px, calc(100vh - 120px))"
    return fig, {"display": "block", "height": graph_height}, {"display": "none"}


# ---------------------------------------------------------------------------
# Enable launch button only when both files are loaded
# ---------------------------------------------------------------------------

@callback(
    Output("btn-launch-viewer", "disabled"),
    Output("div-viewer-hint", "style"),
    Input("store-neural-path", "data"),
    Input("store-behavior-path", "data"),
)
def toggle_launch_button(neural_path, behavior_path):
    ready = bool(behavior_path)
    hint_style = {"display": "none"} if ready else {
        "marginTop": "4px", "fontSize": "0.8em", "color": "#999", "fontStyle": "italic",
    }
    return not ready, hint_style


# ---------------------------------------------------------------------------
# Launch pyqtdash viewer — single instance
# ---------------------------------------------------------------------------

_viewer_process = None
_last_handoff_dir = None


@callback(
    Output("div-viewer-status", "children"),
    Output("store-video-path", "data"),
    Output("div-video-filename", "children"),
    Output("btn-launch-viewer", "children"),
    Input("btn-launch-viewer", "n_clicks"),
    State("store-neural-path", "data"),
    State("store-behavior-path", "data"),
    State("store-video-path", "data"),
    State("dropdown-channels", "value"),
    State("toggle-spectrogram", "value"),
    State("dropdown-spectrogram-channel", "value"),
    State("input-spect-window", "value"),
    State("input-spect-step", "value"),
    State("input-spect-c-param", "value"),
    State("input-spect-max-freq", "value"),
    State("toggle-theta", "value"),
    State("input-theta-low", "value"),
    State("input-theta-high", "value"),
    State("radio-theta-peak-color", "value"),
    State("radio-theta-estimator", "value"),
    State("store-view-range", "data"),
    prevent_initial_call=True,
)
def launch_viewer(n_clicks, neural_path, behavior_path, existing_video_path,
                  selected_channels, spect_toggle, spect_channel,
                  spect_window, spect_step, spect_c, spect_max_freq,
                  theta_toggle, theta_low, theta_high, theta_peak_color,
                  theta_estimator,
                  view_range):
    global _viewer_process, _last_handoff_dir

    if not behavior_path:
        return no_update, no_update, no_update, no_update

    # If viewer is still running, don't launch another
    if _viewer_process is not None and _viewer_process.poll() is None:
        return "Video viewer is already open", no_update, no_update, no_update

    # Resolve video: reuse stored path, or find from behavior metadata, or ask
    video_path = existing_video_path or None
    if not video_path:
        session = load_session_from_paths("", behavior_path)
        info = get_display_metadata(session.behavior_metadata)
        video_filename = info.get("video_filename")
        behavior_dir = Path(behavior_path).parent

        if video_filename:
            candidate = behavior_dir / video_filename
            if candidate.exists():
                video_path = str(candidate)

    if not video_path:
        video_path = pick_file(
            "Select video file",
            "Video (*.avi *.mp4)",
            str(Path(behavior_path).parent),
        )
    if not video_path:
        return "No video selected.", no_update, no_update, no_update

    # Clean up previous handoff directory
    if _last_handoff_dir and Path(_last_handoff_dir).exists():
        shutil.rmtree(_last_handoff_dir, ignore_errors=True)

    # Create handoff directory
    handoff_dir = tempfile.mkdtemp(prefix="neurodash_handoff_")
    _last_handoff_dir = handoff_dir

    x0, x1 = view_range or [0, DEFAULT_VIEW_DURATION]
    duration = x1 - x0
    t_center = x0 + duration / 2
    channels = selected_channels or [0]
    show_spect = "on" in (spect_toggle or [])
    # Theta peak rides on the spectrogram in the viewer, so it only ships when the
    # heatmap does (matching the "overlay only" placement there).
    show_theta_peak = "peak" in (theta_toggle or []) and show_spect
    theta_band = (theta_low or DEFAULT_THETA_LOW_HZ, theta_high or DEFAULT_THETA_HIGH_HZ)

    # Serialize LFP arrays if neural data is loaded
    has_neural = bool(neural_path)
    if has_neural:
        session = load_session_from_paths(neural_path, "")
        sig_info = session.analog_signal_summaries[0]
        sig = get_analog_signal(session.block, 0)
        full_duration = sig_info["duration_sec"]
        channel_labels = sig_info["channel_labels"]

        # Extract selected channel traces
        ys = []
        t_neural = None
        for ch in channels:
            t, y, sr = extract_time_window(sig, ch, 0, full_duration)
            if t_neural is None:
                t_neural = t
            ys.append(y)

        normalized, y_range = normalize_lfp_traces(ys)
        selected_labels = [channel_labels[ch] for ch in channels]

        np.savez(
            Path(handoff_dir) / "lfp.npz",
            t=t_neural,
            traces=np.array(normalized),
            y_range=np.array(y_range),
            channel_names=np.array(selected_labels),
        )

        # Serialize spectrogram if enabled
        if show_spect:
            spect_ch = spect_channel if spect_channel is not None else 0
            freqs, times, power_db = compute_spectrogram(
                neural_path, 0, spect_ch, 0, full_duration,
                spect_max_freq or DEFAULT_SPECT_MAX_FREQ,
                spect_window or DEFAULT_SPECT_WINDOW_SEC,
                spect_step or DEFAULT_SPECT_STEP_SEC,
                spect_c or DEFAULT_SPECT_C_PARAM,
            )
            spect_ch_name = channel_labels[spect_ch]
            np.savez(
                Path(handoff_dir) / "spectrogram.npz",
                freqs=freqs, times=times, power_db=power_db,
            )

            # Theta peak: computed here (same cache the figure uses, so this is
            # usually free) and handed over as a plain array — the viewer only draws it.
            if show_theta_peak:
                theta_times, theta_peak, _theta_power, _theta_ratio = compute_theta_channels(
                    neural_path, 0, spect_ch, full_duration,
                    spect_max_freq or DEFAULT_SPECT_MAX_FREQ,
                    spect_window or DEFAULT_SPECT_WINDOW_SEC,
                    spect_step or DEFAULT_SPECT_STEP_SEC,
                    spect_c or DEFAULT_SPECT_C_PARAM,
                    theta_band[0], theta_band[1],
                    DEFAULT_THETA_INTERP_STEP_HZ,
                    DEFAULT_THETA_SMOOTH_WIDTH,
                    theta_estimator,
                )
                np.savez(
                    Path(handoff_dir) / "theta.npz",
                    times=theta_times, peak_hz=theta_peak,
                )

    # Write handoff JSON
    handoff = {
        "behavior_path": behavior_path,
        "video_path": video_path,
        "has_neural": has_neural,
        "show_spectrogram": show_spect and has_neural,
        "spectrogram_channel_name": spect_ch_name if (show_spect and has_neural) else None,
        "show_theta_peak": show_theta_peak and has_neural,
        "theta_band": list(theta_band),
        "theta_peak_color": theta_peak_color or "black",
        "t_start": t_center,
        "window_duration": duration,
    }
    with open(Path(handoff_dir) / "handoff.json", "w") as f:
        json.dump(handoff, f)

    _viewer_process = subprocess.Popen(
        [sys.executable, "-m", "neurodash.pyqtdash.launch", "--handoff", handoff_dir],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    return "", video_path, Path(video_path).name, "Relaunch viewer"


# ---------------------------------------------------------------------------
# Channel Viewer — tab switching, lazy grid render, annotation persistence
# ---------------------------------------------------------------------------

# Show the active tab's content pane, hide the other (both stay mounted so the
# main-plot figure + viewport survive a tab switch).
clientside_callback(
    """
    function(tabValue) {
        var showChannels = tabValue === "channels";
        return [
            {"display": showChannels ? "none" : "block"},
            {"display": showChannels ? "block" : "none"}
        ];
    }
    """,
    Output("div-viewer-content", "style"),
    Output("div-channel-wrap", "style"),
    Input("tabs-main", "value"),
)


def _spect_params(window, step, c, max_freq):
    return {"window": window, "step": step, "c": c, "max_freq": max_freq}


@callback(
    Output("channel-content", "children"),
    Output("store-channel-exemplar", "data"),
    Output("div-channel-animal", "children"),
    Output("input-channel-comment", "value"),
    Input("tabs-main", "value"),
    Input("store-neural-path", "data"),
    State("store-view-range", "data"),
    State("input-spect-window", "value"),
    State("input-spect-step", "value"),
    State("input-spect-c-param", "value"),
    State("input-spect-max-freq", "value"),
    prevent_initial_call=True,
)
def render_channel_tab(tab_value, neural_path, view_range,
                       spect_window, spect_step, spect_c, spect_max_freq):
    """Build the Channel Viewer when its tab is opened *or* a new file is loaded,
    at the current window. Refreshing on file load (not just tab switch) keeps the
    comment/annotations tied to the loaded recording — no stale bleed from a
    previously-loaded file."""
    if tab_value != "channels":
        return no_update, no_update, no_update, no_update
    if not neural_path:
        return (html.Div("Load neural data to review channels.",
                         style={"color": "#999", "padding": "20px"}),
                no_update, no_update, no_update)

    session = load_session_from_paths(neural_path, "")
    channel_data = load_channels(session.pl2_path, session)
    params = _spect_params(spect_window, spect_step, spect_c, spect_max_freq)
    return (build_channel_view(session, channel_data, view_range, params),
            channel_data.get("exemplar_channel_index"),
            f"Animal: {channel_data.get('animal') or 'Unknown'}",
            channel_data.get("comment", ""))


@callback(
    Output("channel-plot", "figure"),
    Input("input-spect-window", "value"),
    Input("input-spect-step", "value"),
    Input("input-spect-c-param", "value"),
    Input("input-spect-max-freq", "value"),
    State("tabs-main", "value"),
    State("store-neural-path", "data"),
    State("store-view-range", "data"),
    prevent_initial_call=True,
)
def update_channel_spectrograms(spect_window, spect_step, spect_c, spect_max_freq,
                                tab_value, neural_path, view_range):
    """Rebuild the channel figure when a spectrogram control changes (Channel Viewer only).

    Only the figure is rebuilt — the controls gutter and its state are untouched.
    The current window is preserved.
    """
    if tab_value != "channels" or not neural_path:
        return no_update
    session = load_session_from_paths(neural_path, "")
    params = _spect_params(spect_window, spect_step, spect_c, spect_max_freq)
    return plot_channel_figure(session, view_range, params)


@callback(
    Output("store-channel-exemplar", "data", allow_duplicate=True),
    Input({"type": "channel-exemplar", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_exemplar(_n_clicks):
    """A star click sets the single exemplar channel (index in the store)."""
    triggered = ctx.triggered_id
    if not triggered:
        return no_update
    # Ignore the n_clicks=0 population that fires when the grid first mounts.
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return no_update
    return triggered["index"]


@callback(
    Output({"type": "channel-exemplar", "index": ALL}, "children"),
    Output({"type": "channel-exemplar", "index": ALL}, "style"),
    Input("store-channel-exemplar", "data"),
    State({"type": "channel-exemplar", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def refresh_exemplar_stars(exemplar_idx, ids):
    """Keep the ★/☆ glyphs in sync with the current exemplar."""
    glyphs, styles = [], []
    for id_ in ids:
        is_ex = id_["index"] == exemplar_idx
        glyphs.append(exemplar_glyph(is_ex))
        styles.append(exemplar_button_style(is_ex))
    return glyphs, styles


@callback(
    Output("store-channel-saved", "data"),
    Input({"type": "channel-quality", "index": ALL}, "value"),
    Input({"type": "channel-comment", "index": ALL}, "value"),
    Input({"type": "channel-include", "index": ALL}, "value"),
    Input("store-channel-exemplar", "data"),
    Input("input-channel-comment", "value"),
    State({"type": "channel-quality", "index": ALL}, "id"),
    State({"type": "channel-comment", "index": ALL}, "id"),
    State({"type": "channel-include", "index": ALL}, "id"),
    State("store-neural-path", "data"),
    prevent_initial_call=True,
)
def save_channel_edits(qualities, comments, includes, exemplar_idx, general_comment,
                       quality_ids, comment_ids, include_ids, neural_path):
    """Silently persist the channel record whenever a rating/comment/include/exemplar/note changes.

    Rebuilds from the UI (which holds the complete state) and writes only when
    something actually differs from the sidecar — so re-opening the Channel Viewer,
    which re-fires these inputs, does not trigger a spurious write. No user-facing
    confirmation (the sink store just satisfies the callback's Output requirement).
    The animal ID is filename-derived and read-only, so it is not edited here.
    """
    if not neural_path or not quality_ids:
        return no_update

    session = load_session_from_paths(neural_path, "")
    channel_data = load_channels(session.pl2_path, session)
    quality_by_idx = {qid["index"]: (q or "") for qid, q in zip(quality_ids, qualities)}
    comment_by_idx = {cid["index"]: (c or "") for cid, c in zip(comment_ids, comments)}
    include_by_idx = {iid["index"]: bool(v) for iid, v in zip(include_ids, includes)}

    changed = channel_data.get("exemplar_channel_index") != exemplar_idx
    new_note = general_comment or ""
    if new_note != channel_data.get("comment", ""):
        channel_data["comment"] = new_note
        changed = True
    for key, entry in channel_data["channels"].items():
        idx = int(key)
        new_q = quality_by_idx.get(idx, entry.get("quality", ""))
        new_c = comment_by_idx.get(idx, entry.get("comment", ""))
        new_inc = include_by_idx.get(idx, entry.get("include", True))
        if (new_q != entry.get("quality", "") or new_c != entry.get("comment", "")
                or new_inc != entry.get("include", True)):
            changed = True
        entry["quality"] = new_q
        entry["comment"] = new_c
        entry["include"] = new_inc
    channel_data["exemplar_channel_index"] = exemplar_idx

    if not changed:
        return no_update
    save_channels(session.pl2_path, channel_data)
    return channel_data.get("updated_at")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_NO_ANIMAL_HINT = "No animal ID (check the recording filename) — can't export."


def _ask_export_path(default_name):
    """Ask the user where to save an export; "" if they cancel.

    Opens in EXPORT_DIR, which is now the suggested home for exports rather than
    the only place they can go.
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return pick_save_path("Save CSV", "CSV (*.csv)", str(EXPORT_DIR), default_name)


def _write_export(df, out_path, comment=""):
    """Write a CSV to `out_path` and return it.

    An optional free-text comment is written as `#`-prefixed header lines above
    the data (skippable on import, e.g. JMP's comment-character setting).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        for line in (comment or "").splitlines():
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)
    return out


# Fixed preamble of the neural CSV header, in order, ending with the count of the
# per-channel rows that follow it. Copied from the PLX format's own layout (see
# neo.rawio.plexonrawio): a fixed-size GlobalHeader carrying NumDSPChannels /
# NumEventChannels / NumSlowChannels, then that many per-channel records. The header
# is variable in height but self-describing — a reader skips
# len(_CHANNEL_HEADER_FIELDS) + n_channel_rows instead of needing a fixed size.
_CHANNEL_HEADER_FIELDS = (
    "animal", "recorded", "source", "sampling_rate_hz", "duration_s",
    "comment", "exemplar", "n_channel_rows",
)


def _channel_header(session, channel_data, included):
    """Assemble the neural-CSV header block: animal, recording provenance, general
    comment, exemplar, and each exported channel's quality + comment. Returned as
    plain lines; `_write_export` adds the `#` prefixes.
    """
    channels = channel_data.get("channels", {})
    sig_info = session.analog_signal_summaries[0]
    rec = session.rec_datetime

    v = {k: "" for k in _CHANNEL_HEADER_FIELDS}
    v["animal"] = channel_data.get("animal") or "Unknown"
    if rec is not None:
        v["recorded"] = rec.strftime("%Y-%m-%d %H:%M")
    if session.pl2_path:
        v["source"] = Path(session.pl2_path).name
    v["sampling_rate_hz"] = f"{sig_info['sampling_rate_hz']:.0f}"
    v["duration_s"] = f"{sig_info['duration_sec']:.1f}"
    v["comment"] = _flatten(channel_data.get("comment"))

    exemplar = channel_data.get("exemplar_channel_index")
    if exemplar is not None:
        label = channels.get(str(exemplar), {}).get("label", str(exemplar))
        v["exemplar"] = label + ("" if exemplar in included else " (not exported)")

    v["n_channel_rows"] = str(len(included))
    lines = [f"{k}: {v[k]}" for k in _CHANNEL_HEADER_FIELDS]

    for idx in included:
        entry = channels.get(str(idx), {})
        label = entry.get("label", str(idx))
        quality = entry.get("quality") or "unrated"
        note = _flatten(entry.get("comment"))
        line = f"channel: {label} [{quality}]"
        if note:
            line += f": {note}"
        lines.append(line)

    return "\n".join(lines)


# Fields written in the analysis CSV header, in order. EVERY key is always emitted,
# blank when it doesn't apply, so the header is the same number of rows in every file
# neurodash writes — concatenating across animals can then skip a fixed row count.
# Adding a field here changes that count: append rather than insert, and expect older
# exports to be one row shorter.
_ANALYSIS_HEADER_FIELDS = (
    "animal", "neural_source", "spectrogram_channel", "theta_band_hz",
    "theta_estimator", "time_base",
    "behavior_source", "start_time", "experiment", "trial", "arena",
    "behavior_binning", "grid_note", "spectrogram_channel_comment", "behavior_comment",
    "theta_ratio_bands_hz",
)


def _flatten(text):
    """Squash a free-text comment onto one line so it stays one header row."""
    return " / ".join(ln.strip() for ln in (text or "").strip().splitlines() if ln.strip())


def _analysis_header(session, animal, behavior_path, channel_data,
                     channel_index, band, window, step, c, estimator=None,
                     ratio_low_band=None, ratio_high_band=None):
    """Header block for the analysis CSV — a fixed set of rows, always in this order.

    Everything needed to reproduce the table and to know what its `time` column
    means: provenance for both source files, the theta parameters, and — because
    the behavioral columns were resampled — the grid they were put on and the
    window they were averaged over.

    Fields that don't apply are written blank rather than omitted, so every export
    has an identical header height (see _ANALYSIS_HEADER_FIELDS).
    """
    v = {k: "" for k in _ANALYSIS_HEADER_FIELDS}
    v["animal"] = animal or "Unknown"

    if session.has_neural:
        label = session.analog_signal_summaries[0]["channel_labels"][channel_index]
        if session.pl2_path:
            v["neural_source"] = Path(session.pl2_path).name
        v["spectrogram_channel"] = label
        v["theta_band_hz"] = f"{band[0]}-{band[1]}"
        v["theta_estimator"] = estimator or ""
        # Both sub-bands on one row — the ratio is a contrast, so a band is
        # meaningless without its partner.
        r_low = tuple(ratio_low_band or DEFAULT_THETA_RATIO_LOW_BAND)
        r_high = tuple(ratio_high_band or DEFAULT_THETA_RATIO_HIGH_BAND)
        v["theta_ratio_bands_hz"] = (f"low {r_low[0]}-{r_low[1]}, "
                                     f"high {r_high[0]}-{r_high[1]}")
        v["time_base"] = f"spectral bins, step {step}s (window {window}s, c={c}) on {label}"
        # Only the spectrogram channel's own note belongs here — every neural column in
        # this table comes from that one channel. The general neural comment belongs to
        # the Channel Viewer export (_channel_header), which spans all saved channels.
        entry = (channel_data or {}).get("channels", {}).get(str(channel_index), {})
        v["spectrogram_channel_comment"] = _flatten(entry.get("comment"))
    else:
        v["time_base"] = "behavior sample times (no neural data loaded)"

    if session.has_behavior and behavior_path:
        info = get_display_metadata(session.behavior_metadata)
        start = info.get("start_time")
        v["behavior_source"] = Path(behavior_path).name
        v["start_time"] = (start.strftime("%Y-%m-%d %H:%M")
                           if hasattr(start, "strftime") else (start or ""))
        v["experiment"] = info.get("experiment") or ""
        v["trial"] = info.get("trial_name") or ""
        v["arena"] = info.get("arena_name") or ""
        if session.has_neural:
            t_behav = session.behavior_data["Recording time"].to_numpy(dtype=float)
            rate = 1.0 / np.median(np.diff(t_behav)) if len(t_behav) > 1 else float("nan")
            # The bin is closed at both ends, so it catches one more sample than
            # step*rate would suggest.
            n_per_bin = int(step * rate) + 1
            v["behavior_binning"] = (f"mean over the {step}s bin centred on each time "
                                     f"({n_per_bin} samples at {rate:.1f} Hz)")
            v["grid_note"] = (f"grid starts at window/2 = {window / 2:g}s; bins past the "
                              "end of the behavior recording are NaN")

    if behavior_path:
        v["behavior_comment"] = _flatten(load_behavior_notes(behavior_path).get("comment"))

    return "\n".join(f"{k}: {v[k]}" for k in _ANALYSIS_HEADER_FIELDS)


@callback(
    Output("div-channel-export-status", "children"),
    Input("btn-download-channel-csv", "n_clicks"),
    State("store-neural-path", "data"),
    prevent_initial_call=True,
)
def export_channel_csv(n_clicks, neural_path):
    """Write the LFP signal (animal, time, one column per checked channel) to CSV."""
    if not neural_path:
        return no_update
    session = load_session_from_paths(neural_path, "")
    channel_data = load_channels(session.pl2_path, session)
    if not channel_data.get("animal"):
        return _NO_ANIMAL_HINT
    included = sorted(
        int(key) for key, entry in channel_data.get("channels", {}).items()
        if entry.get("include", True)
    )
    if not included:
        return "No channels selected — check the Save box on the channels to export."
    out_path = _ask_export_path(f"{Path(neural_path).stem}_neural.csv")
    if not out_path:
        return ""
    df = build_neural_table(session, included, channel_data["animal"])
    out = _write_export(df, out_path,
                        comment=_channel_header(session, channel_data, included))
    return f"Saved to {out}"


@callback(
    Output("div-analysis-export-status", "children"),
    Input("btn-download-analysis-csv", "n_clicks"),
    State("store-neural-path", "data"),
    State("store-behavior-path", "data"),
    State("dropdown-spectrogram-channel", "value"),
    State("input-theta-low", "value"),
    State("input-theta-high", "value"),
    State("input-spect-window", "value"),
    State("input-spect-step", "value"),
    State("input-spect-c-param", "value"),
    State("input-spect-max-freq", "value"),
    State("radio-theta-estimator", "value"),
    State("input-theta-ratio-low-lo", "value"),
    State("input-theta-ratio-low-hi", "value"),
    State("input-theta-ratio-high-lo", "value"),
    State("input-theta-ratio-high-hi", "value"),
    prevent_initial_call=True,
)
def export_analysis_csv(n_clicks, neural_path, behavior_path, channel,
                        theta_low, theta_high,
                        spect_window, spect_step, spect_c, spect_max_freq,
                        theta_estimator,
                        ratio_low_lo, ratio_low_hi, ratio_high_lo, ratio_high_hi):
    """Write the one analysis table — theta channels plus behavior, on a single
    time base — to a CSV of the user's choosing.

    The spectrogram bins are the grid whenever neural data is loaded; behavior is
    averaged onto them. Nothing is written in two time bases.
    """
    if not neural_path and not behavior_path:
        return no_update

    session = load_session_from_paths(neural_path or "", behavior_path or "")

    # animal always comes from the neural sidecar — behavior files don't carry it.
    channel_data = {}
    animal = ""
    if neural_path:
        channel_data = load_channels(session.pl2_path, session)
        animal = channel_data.get("animal", "")
    if not animal:
        return _NO_ANIMAL_HINT

    ch = channel if channel is not None else 0
    band = (theta_low or DEFAULT_THETA_LOW_HZ, theta_high or DEFAULT_THETA_HIGH_HZ)
    window = spect_window or DEFAULT_SPECT_WINDOW_SEC
    step = spect_step or DEFAULT_SPECT_STEP_SEC
    c = spect_c or DEFAULT_SPECT_C_PARAM
    spect_params = {"window": window, "step": step, "c": c,
                    "max_freq": spect_max_freq or DEFAULT_SPECT_MAX_FREQ}

    stem = Path(neural_path or behavior_path).stem
    out_path = _ask_export_path(f"{stem}_analysis.csv")
    if not out_path:
        return ""

    # Same fallback the figure uses, so the exported ratio matches what is on screen.
    r_low = _ratio_band(ratio_low_lo, ratio_low_hi, DEFAULT_THETA_RATIO_LOW_BAND)
    r_high = _ratio_band(ratio_high_lo, ratio_high_hi, DEFAULT_THETA_RATIO_HIGH_BAND)

    df = build_analysis_table(session, animal, ch, band, spect_params,
                              estimator=theta_estimator,
                              ratio_low_band=r_low, ratio_high_band=r_high)
    out = _write_export(
        df, out_path,
        comment=_analysis_header(session, animal, behavior_path, channel_data,
                                 ch, band, window, step, c, theta_estimator,
                                 r_low, r_high),
    )
    return f"Saved to {out}"


# ---------------------------------------------------------------------------
# Behavioral data note — free-text comment on the whole recording, sidecar-persisted
# ---------------------------------------------------------------------------

@callback(
    Output("input-behavior-comment", "value"),
    Input("store-behavior-path", "data"),
)
def load_behavior_comment(behavior_path):
    """Populate the behavioral comment box from its sidecar when a file loads."""
    if not behavior_path:
        return ""
    return load_behavior_notes(behavior_path).get("comment", "")


@callback(
    Output("store-behavior-saved", "data"),
    Input("input-behavior-comment", "value"),
    State("store-behavior-path", "data"),
    prevent_initial_call=True,
)
def save_behavior_comment(comment, behavior_path):
    """Silently persist the behavioral comment; writes only when it changes."""
    if not behavior_path:
        return no_update
    new_comment = comment or ""
    if new_comment == load_behavior_notes(behavior_path).get("comment", ""):
        return no_update
    save_behavior_notes(behavior_path, {"comment": new_comment})
    return new_comment


# ---------------------------------------------------------------------------
# Merge — combine per-animal analysis exports into one table
#
# Independent of the loaded session: this reads already-exported CSVs from a
# folder. Scanning happens on folder selection (so the file list and any
# problems are visible before committing), compatibility is checked at merge
# time (so unticking a file can resolve a conflict without re-scanning).
# ---------------------------------------------------------------------------

@callback(
    Output("div-merge-folder", "children"),
    Output("check-merge-files", "options"),
    Output("check-merge-files", "value"),
    Output("div-merge-status", "children"),
    Input("btn-merge-folder", "n_clicks"),
    prevent_initial_call=True,
)
def browse_merge_folder(n_clicks):
    """Pick a folder and list the analysis exports in it, all ticked."""
    folder = pick_directory("Select folder of analysis exports", last_browse_dir())
    if not folder:
        return no_update, no_update, no_update, ""

    remember_browse_dir(folder)
    entries, problems = merge.scan_folder(folder)
    options = [{"label": f" {e['animal']} ({e['name']})", "value": str(e["path"])}
               for e in entries]
    value = [opt["value"] for opt in options]

    status = "\n".join(f"Skipped {p}" for p in problems)
    if not entries:
        status = (f"No {merge.ANALYSIS_GLOB} files in that folder.\n{status}").strip()
    return folder, options, value, status


@callback(
    Output("div-merge-status", "children", allow_duplicate=True),
    Input("btn-merge-run", "n_clicks"),
    State("check-merge-files", "value"),
    prevent_initial_call=True,
)
def run_merge(n_clicks, selected_paths):
    """Merge the ticked exports into one CSV, or explain why they can't be."""
    if not selected_paths:
        return "Select a folder and tick the files to merge."
    if len(selected_paths) < 2:
        return "Tick at least two files to merge."

    entries, problems = [], []
    for path in selected_paths:
        try:
            meta, df = merge.read_analysis_csv(path)
        except Exception as e:
            problems.append(f"{Path(path).name}: could not be read ({e})")
            continue
        entries.append({"path": Path(path), "name": Path(path).name,
                        "animal": meta.get("animal") or str(df["animal"].iloc[0]),
                        "meta": meta, "df": df})
    if problems:
        return "\n".join(problems)

    entries.sort(key=lambda e: e["animal"])
    reasons = merge.check_compatible(entries)
    if reasons:
        return merge.refusal_message(reasons)

    out_path = _ask_export_path("merged_analysis.csv")
    if not out_path:
        return ""

    df = merge.merge_tables(entries)
    out = _write_export(df, out_path, comment=merge.merged_header(entries))
    return f"Merged {len(entries)} animals ({len(df)} rows) to {out}"
