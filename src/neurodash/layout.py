"""Dash layout for neurodash."""

from pathlib import Path

from dash import dcc, html

from neurodash.config import (
    APP_TITLE,
    DEFAULT_VIEW_DURATION,
    DEFAULT_SPECT_MAX_FREQ,
    DEFAULT_SPECT_WINDOW_SEC,
    DEFAULT_SPECT_STEP_SEC,
    DEFAULT_SPECT_C_PARAM,
    LOGO_PATH,
    QC_QUALITY_OPTIONS,
    QC_ROW_HEIGHT,
)
from neurodash.plot_utils import plot_qc_figure

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_SIDEBAR_WIDTH = "260px"
_RIGHT_SIDEBAR_WIDTH = "240px"

_SECTION_STYLE_BASE = {
    "padding": "10px",
    "marginBottom": "8px",
    "borderRadius": "4px",
}

_NEURAL_SECTION = {**_SECTION_STYLE_BASE, "backgroundColor": "#eef3f8"}
_BEHAVIOR_SECTION = {**_SECTION_STYLE_BASE, "backgroundColor": "#eef8f0"}
_VIEWER_SECTION = {**_SECTION_STYLE_BASE, "backgroundColor": "#f5f5f5"}

_SECTION_HEADER = {
    "fontWeight": "bold",
    "fontSize": "0.85em",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "color": "#666",
    "marginBottom": "6px",
}

_META_STYLE = {"marginTop": "6px", "fontSize": "0.85em", "color": "#555"}


# ---------------------------------------------------------------------------
# Left sidebar — data loading & metadata
# ---------------------------------------------------------------------------

def _left_sidebar():
    return html.Div(
        [
            html.Img(src=LOGO_PATH, style={"width": "100%", "marginBottom": "12px"}),

            # --- Neural section ---
            html.Div(
                [
                    html.Div("Neural Data", style=_SECTION_HEADER),
                    html.Button("Load neural data", id="btn-browse-neural", n_clicks=0),
                    html.Div(id="div-neural-filename", style={"marginTop": "6px"}),
                    html.Div(id="div-neural-metadata", style=_META_STYLE),
                    dcc.Dropdown(
                        id="dropdown-channels",
                        multi=True,
                        placeholder="Select LFP channels...",
                        style={"marginTop": "6px"},
                    ),
                    dcc.Checklist(
                        id="toggle-spectrogram",
                        options=[{"label": " Show spectrogram", "value": "on"}],
                        value=[],
                        style={"marginTop": "6px"},
                    ),
                ],
                style=_NEURAL_SECTION,
            ),

            # --- Behavior section ---
            html.Div(
                [
                    html.Div("Behavioral Data", style=_SECTION_HEADER),
                    html.Button("Load behavior data", id="btn-browse-behavior", n_clicks=0),
                    html.Div(id="div-behavior-filename", style={"marginTop": "6px"}),
                    html.Div(id="div-behavior-metadata", style=_META_STYLE),
                    html.Button("Download behavior CSV", id="btn-download-behavior-csv",
                                n_clicks=0,
                                style={"marginTop": "8px", "fontSize": "0.85em"}),
                ],
                style=_BEHAVIOR_SECTION,
            ),

            # --- Video Viewer section ---
            html.Div(
                [
                    html.Div("Video Viewer", style=_SECTION_HEADER),
                    html.Button("Launch viewer", id="btn-launch-viewer", n_clicks=0, disabled=True),
                    html.Div(
                        "Load behavioral data first",
                        id="div-viewer-hint",
                        style={"marginTop": "4px", "fontSize": "0.8em", "color": "#999", "fontStyle": "italic"},
                    ),
                    html.Div(id="div-viewer-status", style={"marginTop": "6px"}),
                    html.Div(id="div-video-filename", style={"marginTop": "4px", "fontSize": "0.85em", "color": "#555"}),
                ],
                style=_VIEWER_SECTION,
            ),
        ],
        style={
            "width": _SIDEBAR_WIDTH,
            "padding": "12px",
            "borderRight": "1px solid #ccc",
            "flexShrink": 0,
            "overflowY": "auto",
        },
    )


# ---------------------------------------------------------------------------
# Right sidebar — display controls (collapsible)
# ---------------------------------------------------------------------------

def _right_sidebar():
    return html.Div(
        [
            html.Div("Controls", style=_SECTION_HEADER),

            # --- View controls ---
            html.Label("View duration (s)", style={"fontSize": "0.8em"}),
            dcc.Input(
                id="input-window-duration",
                type="number",
                value=DEFAULT_VIEW_DURATION,
                min=1, step=1,
                debounce=True,
                style={"width": "100%", "marginBottom": "6px"},
            ),
            html.Label("View time (s)", style={"fontSize": "0.8em"}),
            dcc.Input(
                id="input-jump-to-time",
                type="number",
                value=0,
                min=0, step=1,
                debounce=True,
                style={"width": "100%", "marginBottom": "12px"},
            ),

            # --- Spectrogram controls (visible when toggled on) ---
            html.Div(
                [
                    html.Div("Spectrogram", style=_SECTION_HEADER),
                    # Channel selector applies only to the Session Viewer's single
                    # spectrogram; hidden on the Channel Viewer (all channels shown).
                    html.Div(
                        [
                            html.Label("Channel", style={"fontSize": "0.8em"}),
                            dcc.Dropdown(
                                id="dropdown-spectrogram-channel",
                                placeholder="Channel...",
                                style={"marginBottom": "6px"},
                            ),
                        ],
                        id="div-spect-channel-select",
                    ),
                    html.Label("Window (s)", style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-window",
                        type="number",
                        value=DEFAULT_SPECT_WINDOW_SEC,
                        min=0.1, step=0.1,
                        debounce=True,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("Step (s)", style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-step",
                        type="number",
                        value=DEFAULT_SPECT_STEP_SEC,
                        min=0.01, step=0.01,
                        debounce=True,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("C parameter", style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-c-param",
                        type="number",
                        value=DEFAULT_SPECT_C_PARAM,
                        min=1, step=1,
                        debounce=True,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("Max freq (Hz)", style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-max-freq",
                        type="number",
                        value=DEFAULT_SPECT_MAX_FREQ,
                        min=1, step=1,
                        debounce=True,
                        style={"width": "100%"},
                    ),
                ],
                id="div-spectrogram-controls",
                style={"display": "none"},
            ),
        ],
        id="div-right-sidebar",
        style={
            "width": _RIGHT_SIDEBAR_WIDTH,
            "padding": "12px",
            "borderLeft": "1px solid #ccc",
            "flexShrink": 0,
            "overflowY": "auto",
        },
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def make_layout():
    return html.Div(
        [
            _left_sidebar(),

            # Main area — Session Viewer / Channel QC tabs
            html.Div(
                [
                    dcc.Tabs(
                        id="tabs-main",
                        value="viewer",
                        children=[
                            dcc.Tab(label="Session Viewer", value="viewer"),
                            dcc.Tab(label="Channel Viewer", value="qc"),
                        ],
                    ),

                    # Session Viewer pane — always mounted (display toggled by tab)
                    html.Div(
                        [
                            html.Div(
                                "Load data to begin",
                                id="div-plot-placeholder",
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "height": "100%",
                                    "color": "#999",
                                    "fontSize": "1.2em",
                                },
                            ),
                            dcc.Graph(
                                id="main-plot",
                                style={"display": "none"},
                                config={"toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "neurodash_session",
                                    "scale": 2,
                                }},
                            ),
                        ],
                        id="div-viewer-content",
                    ),

                    # Channel QC pane — always mounted, hidden until tab selected
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Button("Download QC table (CSV)",
                                                id="btn-download-qc-csv", n_clicks=0),
                                    html.Span(id="div-qc-save-status",
                                              style={"marginLeft": "10px",
                                                     "color": "#3a8a3a",
                                                     "fontSize": "0.85em"}),
                                ],
                                style={"marginBottom": "8px"},
                            ),
                            dcc.Loading(html.Div(id="qc-content")),
                        ],
                        id="div-qc-wrap",
                        style={"display": "none"},
                    ),
                ],
                style={"flex": 1, "padding": "8px", "minWidth": 0, "overflowY": "auto"},
            ),

            _right_sidebar(),

            # Stores
            dcc.Store(id="store-neural-path", data=""),
            dcc.Store(id="store-behavior-path", data=""),
            dcc.Store(id="store-video-path", data=""),
            dcc.Store(id="store-view-range", data=[0, DEFAULT_VIEW_DURATION]),
            dcc.Store(id="store-qc-exemplar", data=None),

            # Downloads
            dcc.Download(id="download-qc-csv"),
            dcc.Download(id="download-behavior-csv"),
        ],
        style={"display": "flex", "height": "100vh"},
    )


# ---------------------------------------------------------------------------
# Channel QC view — one synced figure + an aligned column of per-channel
# controls, built lazily when the QC tab is opened (see callbacks)
# ---------------------------------------------------------------------------

def exemplar_glyph(is_exemplar):
    return "★" if is_exemplar else "☆"


def exemplar_button_style(is_exemplar):
    return {
        "border": "none", "background": "none", "fontSize": "1.4em",
        "cursor": "pointer", "padding": "0",
        "color": "#e0a800" if is_exemplar else "#bbb",
    }


def _qc_control_block(idx, label, entry, is_exemplar):
    """One channel's controls, sized (flex:1) to line up with its figure row."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(label, style={"fontWeight": "bold"}),
                    html.Button(
                        exemplar_glyph(is_exemplar),
                        id={"type": "qc-exemplar", "index": idx},
                        title="Mark as exemplar channel",
                        n_clicks=0,
                        style=exemplar_button_style(is_exemplar),
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center"},
            ),
            dcc.RadioItems(
                id={"type": "qc-quality", "index": idx},
                options=[{"label": f" {q.capitalize()}", "value": q}
                         for q in QC_QUALITY_OPTIONS],
                value=entry.get("quality") or None,
                inline=True,
                style={"fontSize": "0.85em", "marginTop": "4px"},
            ),
            dcc.Input(
                id={"type": "qc-comment", "index": idx},
                type="text", value=entry.get("comment", ""),
                placeholder="comment...", debounce=True,
                style={"width": "100%", "marginTop": "4px", "fontSize": "0.85em"},
            ),
        ],
        style={
            "flex": "1 1 0", "display": "flex", "flexDirection": "column",
            "justifyContent": "center", "overflow": "hidden",
            "borderBottom": "1px solid #eee", "paddingRight": "10px",
        },
    )


def build_qc_view(session, qc_data, view_range=None, spect_params=None):
    """Build the QC view: a controls gutter aligned to one synced figure.

    The gutter's height + top/bottom padding mirror the figure's height +
    margins, and both divide into n equal rows, so each control block lines up
    with its channel's row in the figure.
    """
    sig_info = session.analog_signal_summaries[0]
    exemplar = qc_data.get("exemplar_channel_index")
    channels = qc_data.get("channels", {})
    n = len(sig_info["channel_indices"])

    gutter = html.Div(
        [
            _qc_control_block(
                idx, label,
                channels.get(str(idx), {"quality": "", "comment": ""}),
                idx == exemplar,
            )
            for idx, label in zip(sig_info["channel_indices"], sig_info["channel_labels"])
        ],
        style={
            "display": "flex", "flexDirection": "column",
            "width": "200px", "flexShrink": 0,
            "height": f"{QC_ROW_HEIGHT * n + 50}px",
            "paddingTop": "10px", "paddingBottom": "40px",
            "boxSizing": "border-box",
        },
    )
    stem = Path(session.pl2_path).stem
    graph = dcc.Graph(
        id="qc-plot",
        figure=plot_qc_figure(session, view_range, spect_params),
        style={"flex": 1, "minWidth": 0},
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": f"{stem}_qc", "scale": 2},
        },
    )
    return html.Div([gutter, graph], style={"display": "flex", "alignItems": "flex-start"})
