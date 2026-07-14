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
    DEFAULT_FILE_DIR, DEFAULT_VIEW_DURATION, DEFAULT_SPECT_MAX_FREQ,
    DEFAULT_SPECT_WINDOW_SEC, DEFAULT_SPECT_STEP_SEC, DEFAULT_SPECT_C_PARAM,
    AUTOLOAD_ON_STARTUP, AUTOLOAD_NEURAL_PATH, AUTOLOAD_BEHAVIOR_PATH,
    EXEMPLAR_SEEDS_DEFAULT_CHANNEL,
)
from neurodash.file_picker import pick_file
from neurodash.behavior_io import get_display_metadata
from neurodash.neural_io import get_analog_signal, extract_time_window
from neurodash.plot_utils import normalize_lfp_traces, plot_session_view, plot_qc_figure
from neurodash.session import load_session_from_paths, compute_spectrogram
from neurodash.qc_io import load_qc, save_qc, qc_to_dataframe
from neurodash.layout import build_qc_view, exemplar_glyph, exemplar_button_style


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
        path = pick_file("Select .pl2 file", "Plexon (*.pl2)", DEFAULT_FILE_DIR)
    if not path:
        return (no_update,) * 7

    session = load_session_from_paths(path, "")
    sig_info = session.analog_signal_summaries[0]
    options = [
        {"label": lbl, "value": idx}
        for idx, lbl in zip(sig_info["channel_indices"], sig_info["channel_labels"])
    ]

    # Build metadata display
    dur = sig_info["duration_sec"]
    minutes = int(dur // 60)
    seconds = dur % 60
    rec_dt = session.rec_datetime
    dt_str = rec_dt.strftime("%Y-%m-%d %H:%M") if rec_dt else "Unknown"

    metadata = html.Div([
        html.Div(f"Recorded: {dt_str}"),
        html.Div(f"Duration: {minutes}m {seconds:.1f}s"),
        html.Div(f"Sampling rate: {sig_info['sampling_rate_hz']:.0f} Hz"),
        html.Div(f"Channels: {sig_info['n_channels']}"),
    ])

    # Seed the default selected channel from a saved QC exemplar (if any).
    default_channel = 0
    if EXEMPLAR_SEEDS_DEFAULT_CHANNEL:
        ex = load_qc(path, session).get("exemplar_channel_index")
        if ex is not None and 0 <= ex < sig_info["n_channels"]:
            default_channel = ex

    return (path, Path(path).name, metadata, options,
            [default_channel], options, default_channel)


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
        path = pick_file("Select behavior file", "Excel (*.xlsx)", DEFAULT_FILE_DIR)
    if not path:
        return no_update, no_update, no_update

    session = load_session_from_paths("", path)
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


# Panning/zooming the combined QC figure updates the same shared window.
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
    Input("qc-plot", "relayoutData"),
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


# ---------------------------------------------------------------------------
# Spectrogram controls visibility — show/hide with toggle
# ---------------------------------------------------------------------------

clientside_callback(
    """
    function(toggleValue, tabValue) {
        var toggled = toggleValue && toggleValue.indexOf("on") !== -1;
        var onQC = tabValue === "qc";
        /* Params: shown when the Session Viewer spectrogram is on, and always
           on the Channel Viewer. Channel selector: Session Viewer only. */
        var showParams = toggled || onQC;
        var showChannel = toggled && !onQC;
        return [
            {"display": showParams ? "block" : "none"},
            {"display": showChannel ? "block" : "none"}
        ];
    }
    """,
    Output("div-spectrogram-controls", "style"),
    Output("div-spect-channel-select", "style"),
    Input("toggle-spectrogram", "value"),
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
            var center = (x0 + x1) / 2;
            x0 = center - windowDuration / 2;
            x1 = center + windowDuration / 2;
        }

        // Relayout the primary x-axis; matches='x' propagates to others
        var graphDiv = document.querySelector("#main-plot .js-plotly-plot");
        if (graphDiv) {
            Plotly.relayout(graphDiv, {"xaxis.range": [x0, x1], "xaxis.autorange": false});
        }
        var qcDiv = document.querySelector("#qc-plot .js-plotly-plot");
        if (qcDiv) {
            Plotly.relayout(qcDiv, {"xaxis.range": [x0, x1], "xaxis.autorange": false});
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
    State("store-view-range", "data"),
)
def update_figure(neural_path, behavior_path, selected_channels, spect_toggle,
                  spect_channel, spect_window, spect_step, spect_c, spect_max_freq,
                  view_range):
    if not neural_path and not behavior_path:
        return no_update, no_update, no_update

    session = load_session_from_paths(neural_path or "", behavior_path or "")
    channels = selected_channels or [0]
    controls = {
        "raw_channel_indices": channels,
        "show_spectrogram": "on" in (spect_toggle or []),
        "spectrogram_channel_index": spect_channel if spect_channel is not None else 0,
        "spect_window_sec": spect_window or DEFAULT_SPECT_WINDOW_SEC,
        "spect_step_sec": spect_step or DEFAULT_SPECT_STEP_SEC,
        "spect_c_param": spect_c or DEFAULT_SPECT_C_PARAM,
        "spect_max_freq": spect_max_freq or DEFAULT_SPECT_MAX_FREQ,
    }
    fig = plot_session_view(session, controls)
    if fig is None:
        return no_update, no_update, no_update

    x0, x1 = view_range or [0, DEFAULT_VIEW_DURATION]
    fig.update_xaxes(range=[x0, x1], autorange=False)
    return fig, {"display": "block"}, {"display": "none"}


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
    State("store-view-range", "data"),
    prevent_initial_call=True,
)
def launch_viewer(n_clicks, neural_path, behavior_path, existing_video_path,
                  selected_channels, spect_toggle, spect_channel,
                  spect_window, spect_step, spect_c, spect_max_freq,
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

    # Write handoff JSON
    handoff = {
        "behavior_path": behavior_path,
        "video_path": video_path,
        "has_neural": has_neural,
        "show_spectrogram": show_spect and has_neural,
        "spectrogram_channel_name": spect_ch_name if (show_spect and has_neural) else None,
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
# Channel QC — tab switching, lazy grid render, annotation persistence
# ---------------------------------------------------------------------------

# Show the active tab's content pane, hide the other (both stay mounted so the
# main-plot figure + viewport survive a tab switch).
clientside_callback(
    """
    function(tabValue) {
        var showQC = tabValue === "qc";
        return [
            {"display": showQC ? "none" : "block"},
            {"display": showQC ? "block" : "none"}
        ];
    }
    """,
    Output("div-viewer-content", "style"),
    Output("div-qc-wrap", "style"),
    Input("tabs-main", "value"),
)


def _spect_params(window, step, c, max_freq):
    return {"window": window, "step": step, "c": c, "max_freq": max_freq}


@callback(
    Output("qc-content", "children"),
    Output("store-qc-exemplar", "data"),
    Input("tabs-main", "value"),
    State("store-neural-path", "data"),
    State("store-view-range", "data"),
    State("input-spect-window", "value"),
    State("input-spect-step", "value"),
    State("input-spect-c-param", "value"),
    State("input-spect-max-freq", "value"),
    prevent_initial_call=True,
)
def render_qc_tab(tab_value, neural_path, view_range,
                  spect_window, spect_step, spect_c, spect_max_freq):
    """Lazily build the QC view when the QC tab is opened, at the current window."""
    if tab_value != "qc":
        return no_update, no_update
    if not neural_path:
        return html.Div("Load neural data to review channels.",
                        style={"color": "#999", "padding": "20px"}), no_update

    session = load_session_from_paths(neural_path, "")
    qc_data = load_qc(session.pl2_path, session)
    params = _spect_params(spect_window, spect_step, spect_c, spect_max_freq)
    return (build_qc_view(session, qc_data, view_range, params),
            qc_data.get("exemplar_channel_index"))


@callback(
    Output("qc-plot", "figure"),
    Input("input-spect-window", "value"),
    Input("input-spect-step", "value"),
    Input("input-spect-c-param", "value"),
    Input("input-spect-max-freq", "value"),
    State("tabs-main", "value"),
    State("store-neural-path", "data"),
    State("store-view-range", "data"),
    prevent_initial_call=True,
)
def update_qc_spectrograms(spect_window, spect_step, spect_c, spect_max_freq,
                           tab_value, neural_path, view_range):
    """Rebuild the QC figure when a spectrogram control changes (QC tab only).

    Only the figure is rebuilt — the controls gutter and its state are untouched.
    The current window is preserved.
    """
    if tab_value != "qc" or not neural_path:
        return no_update
    session = load_session_from_paths(neural_path, "")
    params = _spect_params(spect_window, spect_step, spect_c, spect_max_freq)
    return plot_qc_figure(session, view_range, params)


@callback(
    Output("store-qc-exemplar", "data", allow_duplicate=True),
    Input({"type": "qc-exemplar", "index": ALL}, "n_clicks"),
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
    Output({"type": "qc-exemplar", "index": ALL}, "children"),
    Output({"type": "qc-exemplar", "index": ALL}, "style"),
    Input("store-qc-exemplar", "data"),
    State({"type": "qc-exemplar", "index": ALL}, "id"),
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
    Output("div-qc-save-status", "children"),
    Input({"type": "qc-quality", "index": ALL}, "value"),
    Input({"type": "qc-comment", "index": ALL}, "value"),
    Input("store-qc-exemplar", "data"),
    State({"type": "qc-quality", "index": ALL}, "id"),
    State({"type": "qc-comment", "index": ALL}, "id"),
    State("store-neural-path", "data"),
    prevent_initial_call=True,
)
def save_qc_edits(qualities, comments, exemplar_idx, quality_ids, comment_ids, neural_path):
    """Persist the whole QC record whenever a rating/comment/exemplar changes.

    Rebuilds from the UI (which holds the complete state) and writes only when
    something actually differs from the sidecar — so re-opening the QC tab, which
    re-fires these inputs, does not trigger a spurious write.
    """
    if not neural_path or not quality_ids:
        return no_update

    session = load_session_from_paths(neural_path, "")
    qc_data = load_qc(session.pl2_path, session)
    quality_by_idx = {qid["index"]: (q or "") for qid, q in zip(quality_ids, qualities)}
    comment_by_idx = {cid["index"]: (c or "") for cid, c in zip(comment_ids, comments)}

    changed = qc_data.get("exemplar_channel_index") != exemplar_idx
    for key, entry in qc_data["channels"].items():
        idx = int(key)
        new_q = quality_by_idx.get(idx, entry.get("quality", ""))
        new_c = comment_by_idx.get(idx, entry.get("comment", ""))
        if new_q != entry.get("quality", "") or new_c != entry.get("comment", ""):
            changed = True
        entry["quality"] = new_q
        entry["comment"] = new_c
    qc_data["exemplar_channel_index"] = exemplar_idx

    if not changed:
        return no_update
    save_qc(session.pl2_path, qc_data)
    return "saved ✓"


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

@callback(
    Output("download-qc-csv", "data"),
    Input("btn-download-qc-csv", "n_clicks"),
    State("store-neural-path", "data"),
    prevent_initial_call=True,
)
def download_qc_csv(n_clicks, neural_path):
    if not neural_path:
        return no_update
    session = load_session_from_paths(neural_path, "")
    df = qc_to_dataframe(load_qc(session.pl2_path, session))
    return dcc.send_data_frame(df.to_csv, f"{Path(neural_path).stem}_qc.csv", index=False)


@callback(
    Output("download-behavior-csv", "data"),
    Input("btn-download-behavior-csv", "n_clicks"),
    State("store-behavior-path", "data"),
    prevent_initial_call=True,
)
def download_behavior_csv(n_clicks, behavior_path):
    if not behavior_path:
        return no_update
    session = load_session_from_paths("", behavior_path)
    return dcc.send_data_frame(
        session.behavior_data.to_csv, f"{Path(behavior_path).stem}_tracking.csv", index=False,
    )
