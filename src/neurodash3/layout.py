"""Dash layout for neurodash3."""

from dash import dcc, html

from neurodash3.config import (
    APP_TITLE,
    DEFAULT_VIEW_DURATION,
    DEFAULT_SPECT_MAX_FREQ,
    DEFAULT_SPECT_WINDOW_SEC,
    DEFAULT_SPECT_STEP_SEC,
    DEFAULT_SPECT_C_PARAM,
    LOGO_PATH,
)

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
                    html.Label("Channel", style={"fontSize": "0.8em"}),
                    dcc.Dropdown(
                        id="dropdown-spectrogram-channel",
                        placeholder="Channel...",
                        style={"marginBottom": "6px"},
                    ),
                    html.Label("Window (s)", style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-window",
                        type="number",
                        value=DEFAULT_SPECT_WINDOW_SEC,
                        min=0.1, step=0.1,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("Step (s)", style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-step",
                        type="number",
                        value=DEFAULT_SPECT_STEP_SEC,
                        min=0.01, step=0.01,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("C parameter", style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-c-param",
                        type="number",
                        value=DEFAULT_SPECT_C_PARAM,
                        min=1, step=1,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("Max freq (Hz)", style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-max-freq",
                        type="number",
                        value=DEFAULT_SPECT_MAX_FREQ,
                        min=1, step=1,
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

            # Main area — Session Viewer
            html.Div(
                [
                    html.Div("Session Viewer", style={
                        "fontWeight": "bold",
                        "fontSize": "1.0em",
                        "color": "#444",
                        "marginBottom": "4px",
                    }),
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
                    dcc.Graph(id="main-plot", style={"display": "none"}),
                ],
                style={"flex": 1, "padding": "8px", "minWidth": 0},
            ),

            _right_sidebar(),

            # Stores
            dcc.Store(id="store-neural-path", data=""),
            dcc.Store(id="store-behavior-path", data=""),
            dcc.Store(id="store-video-path", data=""),
            dcc.Store(id="store-view-range", data=[0, DEFAULT_VIEW_DURATION]),
        ],
        style={"display": "flex", "height": "100vh"},
    )
