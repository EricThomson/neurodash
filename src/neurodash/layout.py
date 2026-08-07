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
    DEFAULT_THETA_LOW_HZ,
    DEFAULT_THETA_HIGH_HZ,
    DEFAULT_THETA_DOT_SIZE,
    DEFAULT_THETA_ESTIMATOR,
    DEFAULT_THETA_RATIO_LOW_BAND,
    DEFAULT_THETA_RATIO_HIGH_BAND,
    LOGO_PATH,
    CHANNEL_QUALITY_OPTIONS,
    CHANNEL_ROW_HEIGHT,
)
from neurodash.plot_utils import plot_channel_figure

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
_MERGE_SECTION = {**_SECTION_STYLE_BASE, "backgroundColor": "#f8f4ee"}

_SECTION_HEADER = {
    "fontWeight": "bold",
    "fontSize": "0.85em",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "color": "#666",
    "marginBottom": "6px",
}

_META_STYLE = {"marginTop": "6px", "fontSize": "0.85em", "color": "#555"}

_CHANNEL_COL_HEADER = {"fontSize": "0.8em", "fontWeight": "bold", "color": "#555",
                       "marginBottom": "2px"}

_IDENTITY_LABEL = {"fontSize": "0.85em", "color": "#555", "width": "58px",
                   "flexShrink": "0"}
_IDENTITY_INPUT = {"flex": "1", "minWidth": "0", "fontSize": "0.9em"}


# ---------------------------------------------------------------------------
# Left sidebar — data loading & metadata
# ---------------------------------------------------------------------------

def _left_sidebar():
    return html.Div(
        [
            html.Img(src=LOGO_PATH, style={"width": "100%", "marginBottom": "12px"}),

            # --- Session identity ---
            # Above the data sections because it labels the whole recording, and
            # because it's what every export leads with. Inferred, but editable:
            # both fields come from free text somebody typed into EthoVision, so
            # they are wrong often enough that a correction has to be possible
            # without renaming files.
            html.Div(
                [
                    html.Div("Session Info", style=_SECTION_HEADER),
                    # title goes on the row, not the Input — dcc.Input has no title prop.
                    html.Div(
                        [
                            html.Label("Animal", style=_IDENTITY_LABEL),
                            dcc.Input(id="input-animal", type="text", disabled=True,
                                      debounce=True, style=_IDENTITY_INPUT),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                        title="Animal ID. Inferred from the EthoVision header, "
                              "else the .pl2 filename.",
                    ),
                    html.Div(
                        [
                            html.Label("Session", style=_IDENTITY_LABEL),
                            dcc.Input(id="input-session", type="text", disabled=True,
                                      debounce=True, style=_IDENTITY_INPUT),
                        ],
                        style={"display": "flex", "alignItems": "center",
                               "marginTop": "3px"},
                        title="Session label, e.g. hab1. Inferred from the "
                              "EthoVision Session field.",
                    ),
                    html.Button("Edit", id="btn-edit-identity", n_clicks=0,
                                style={"marginTop": "5px", "fontSize": "0.8em"},
                                title="Correct the animal or session label."),
                    html.Div(id="div-identity-status", style=_META_STYLE),
                ],
                style={"marginBottom": "12px"},
            ),

            # --- Neural section ---
            html.Div(
                [
                    html.Div("Neural Data", style=_SECTION_HEADER),
                    html.Button("Load neural data", id="btn-browse-neural", n_clicks=0),
                    html.Div(id="div-neural-filename", style={"marginTop": "6px"}),
                    html.Div(id="div-neural-metadata", style=_META_STYLE),
                    # Show and Save are deliberately independent: you rarely want
                    # to *look* at eight LFP traces, but you always want to keep
                    # the data. Two plain checklists side by side rather than one
                    # table of cells — each lays its own rows out identically, so
                    # they line up for free.
                    html.Div(
                        [
                            html.Div(
                                [html.Div("Show", style=_CHANNEL_COL_HEADER),
                                 dcc.Checklist(id="checklist-show-channels", value=[])],
                            ),
                            html.Div(
                                [html.Div("Save", style=_CHANNEL_COL_HEADER),
                                 dcc.Checklist(id="checklist-save-channels", value=[])],
                            ),
                        ],
                        id="div-channel-select",
                        style={"display": "flex", "gap": "18px", "marginTop": "6px"},
                        title="Show = plot the raw LFP here. "
                              "Save = include in the analysis CSV export.",
                    ),
                    dcc.Checklist(
                        id="toggle-spectrogram",
                        options=[{"label": " Show spectrogram", "value": "on"}],
                        value=[],
                        style={"marginTop": "6px"},
                    ),
                    dcc.Checklist(
                        id="toggle-theta",
                        options=[
                            {"label": " Show theta band", "value": "band"},
                            {"label": " Show theta peak", "value": "peak"},
                            {"label": " Show theta power", "value": "power"},
                            {"label": " Show theta ratio", "value": "ratio"},
                        ],
                        value=[],
                        style={"marginTop": "2px"},
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

            # --- Merge section: combine per-animal analysis exports into one
            # table. Always visible — this works on a folder of already-exported
            # CSVs and has nothing to do with the currently loaded session. ---
            html.Div(
                [
                    html.Div("Merge Data", style=_SECTION_HEADER),
                    html.Button("Select folder", id="btn-merge-folder", n_clicks=0,
                                title="Pick a folder of exported *_analysis.csv files "
                                      "to combine into one table."),
                    html.Div(id="div-merge-folder",
                             style={"marginTop": "4px", "fontSize": "0.8em", "color": "#555",
                                    "wordBreak": "break-all"}),
                    dcc.Checklist(id="check-merge-files", options=[], value=[],
                                  style={"marginTop": "6px", "fontSize": "0.85em"}),
                    html.Button("Merge selected", id="btn-merge-run", n_clicks=0,
                                style={"marginTop": "6px"}),
                    # Multi-line, unlike the single-line export status spans — a
                    # refusal lists which files disagree and what to do about it.
                    html.Div(id="div-merge-status",
                             style={"marginTop": "6px", "fontSize": "0.8em", "color": "#555",
                                    "whiteSpace": "pre-line"}),
                ],
                style=_MERGE_SECTION,
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
            html.Label("View duration (s)",
                       title="Width of the visible time window.",
                       style={"fontSize": "0.8em"}),
            dcc.Input(
                id="input-window-duration",
                type="number",
                value=DEFAULT_VIEW_DURATION,
                min=1, step=1,
                debounce=True,
                style={"width": "100%", "marginBottom": "6px"},
            ),
            html.Label("View time (s)",
                       title="Start time of the visible window.",
                       style={"fontSize": "0.8em"}),
            dcc.Input(
                id="input-jump-to-time",
                type="number",
                value=0,
                min=0, step=1,
                debounce=True,
                style={"width": "100%", "marginBottom": "12px"},
            ),

            # --- Analysis channel — the single channel the spectrogram and theta
            # channels are computed on. Session Viewer only (the Channel Viewer
            # shows all channels). Seeded from the starred exemplar. ---
            html.Div(
                [
                    html.Label("Spectrogram Channel",
                               title="Channel for displaying spectrogram. Used for "
                                     "calculating theta peak, power, etc. Defaults to "
                                     "exemplar unless other channel is selected.",
                               style={"fontSize": "0.8em"}),
                    dcc.Dropdown(
                        id="dropdown-spectrogram-channel",
                        placeholder="Channel...",
                        style={"marginBottom": "6px"},
                    ),
                ],
                id="div-analysis-channel",
                style={"display": "none"},
            ),

            # --- Spectrogram params (visible when the spectrogram or a theta
            # channel is on, and always on the Channel Viewer). Theta rides on
            # these same params. ---
            html.Div(
                [
                    html.Div("Spectrogram", style=_SECTION_HEADER),
                    html.Label("Window (s)",
                               title="Time window each spectrogram is computed over. Longer "
                                     "window will smooth across time, but give better "
                                     "frequency-domain resolution (1/window).",
                               style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-window",
                        type="number",
                        value=DEFAULT_SPECT_WINDOW_SEC,
                        min=0.1, step=0.1,
                        debounce=True,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("Step (s)",
                               title="Time resolution of spectrogram: the step size between "
                                     "spectrogram calculations.",
                               style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-step",
                        type="number",
                        value=DEFAULT_SPECT_STEP_SEC,
                        min=0.01, step=0.01,
                        debounce=True,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("C parameter",
                               title="How much the spectrum is averaged to reduce noise: "
                                     "higher C = cleaner but coarser, lower C = sharper "
                                     "but noisier.",
                               style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-spect-c-param",
                        type="number",
                        value=DEFAULT_SPECT_C_PARAM,
                        min=1, step=1,
                        debounce=True,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("Max freq (Hz)",
                               title="Highest frequency shown on the spectrogram.",
                               style={"fontSize": "0.8em"}),
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

            # --- Theta band + export (visible when a theta channel is on). ---
            html.Div(
                [
                    html.Div("Theta", style=_SECTION_HEADER),
                    html.Label("Low (Hz)",
                               title="Lower limit of the theta band.",
                               style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-theta-low",
                        type="number",
                        value=DEFAULT_THETA_LOW_HZ,
                        min=0.5, step=0.5,
                        debounce=True,
                        style={"width": "100%", "marginBottom": "6px"},
                    ),
                    html.Label("High (Hz)",
                               title="Upper limit of the theta band.",
                               style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-theta-high",
                        type="number",
                        value=DEFAULT_THETA_HIGH_HZ,
                        min=1, step=0.5,
                        debounce=True,
                        style={"width": "100%", "marginBottom": "8px"},
                    ),
                    # Estimator — argmax pins to the band edge when delta bleeds in;
                    # bandpass filters the LFP first, so there is no delta left to leak.
                    html.Label("Peak estimator",
                               title="Whether to bandpass filter the LFP around the theta "
                                     "band before finding the peak.",
                               style={"fontSize": "0.8em"}),
                    dcc.RadioItems(
                        id="radio-theta-estimator",
                        options=[{"label": " argmax", "value": "argmax"},
                                 {"label": " bandpass", "value": "bandpass"}],
                        value=DEFAULT_THETA_ESTIMATOR,
                        inline=True,
                        style={"fontSize": "0.85em", "marginBottom": "4px"},
                    ),
                    # Peak overlay appearance (line rides on the spectrogram)
                    html.Label("Peak line", style={"fontSize": "0.8em"}),
                    dcc.RadioItems(
                        id="radio-theta-peak-color",
                        options=[{"label": " Black", "value": "black"},
                                 {"label": " White", "value": "white"}],
                        value="black",
                        inline=True,
                        style={"fontSize": "0.85em"},
                    ),
                    dcc.Checklist(
                        id="toggle-theta-peak-markers",
                        options=[{"label": " Dots", "value": "dots"}],
                        value=["dots"],
                        style={"fontSize": "0.85em", "marginBottom": "4px"},
                    ),
                    html.Label("Dot size", style={"fontSize": "0.8em"}),
                    dcc.Input(
                        id="input-theta-dot-size",
                        type="number",
                        value=DEFAULT_THETA_DOT_SIZE,
                        min=1, max=20, step=0.5,
                        debounce=True,
                        style={"width": "100%", "marginBottom": "8px"},
                    ),

                    # Theta ratio sub-bands. Two narrow slices of the theta band:
                    # (low - high) / (low + high), so positive = slow theta dominant.
                    html.Div("Ratio bands", style={**_SECTION_HEADER,
                                                   "marginTop": "4px"}),
                    html.Label("Low band (Hz)",
                               title="Frequency bounds for low theta, used to calculate "
                                     "the theta ratio.",
                               style={"fontSize": "0.8em"}),
                    html.Div(
                        [
                            dcc.Input(
                                id="input-theta-ratio-low-lo",
                                type="number",
                                value=DEFAULT_THETA_RATIO_LOW_BAND[0],
                                min=0.5, step=0.1,
                                debounce=True,
                                style={"width": "50%"},
                            ),
                            dcc.Input(
                                id="input-theta-ratio-low-hi",
                                type="number",
                                value=DEFAULT_THETA_RATIO_LOW_BAND[1],
                                min=0.5, step=0.1,
                                debounce=True,
                                style={"width": "50%"},
                            ),
                        ],
                        style={"display": "flex", "marginBottom": "6px"},
                    ),
                    html.Label("High band (Hz)",
                               title="Frequency bounds for high theta, used to calculate "
                                     "the theta ratio.",
                               style={"fontSize": "0.8em"}),
                    html.Div(
                        [
                            dcc.Input(
                                id="input-theta-ratio-high-lo",
                                type="number",
                                value=DEFAULT_THETA_RATIO_HIGH_BAND[0],
                                min=0.5, step=0.1,
                                debounce=True,
                                style={"width": "50%"},
                            ),
                            dcc.Input(
                                id="input-theta-ratio-high-hi",
                                type="number",
                                value=DEFAULT_THETA_RATIO_HIGH_BAND[1],
                                min=0.5, step=0.1,
                                debounce=True,
                                style={"width": "50%"},
                            ),
                        ],
                        style={"display": "flex", "marginBottom": "8px"},
                    ),
                ],
                id="div-theta-controls",
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

            # Main area — Session Viewer / Channel Viewer tabs
            html.Div(
                [
                    dcc.Tabs(
                        id="tabs-main",
                        value="viewer",
                        children=[
                            dcc.Tab(label="Session Viewer", value="viewer"),
                            dcc.Tab(label="Channel Viewer", value="channels"),
                        ],
                    ),

                    # Session Viewer pane — always mounted (display toggled by tab)
                    html.Div(
                        [
                            # Analysis export (theta + behavior on one time base)
                            # + a free-text comment on the recording
                            html.Div(
                                [
                                    html.Button(
                                        "Export analysis CSV",
                                        id="btn-download-analysis-csv", n_clicks=0,
                                        title="Theta peak/power/ratio and behavior on one time base "
                                              "(the spectrogram bins), written to a CSV you choose.",
                                    ),
                                    html.Span(id="div-analysis-export-status",
                                              style={"marginLeft": "10px", "color": "#999",
                                                     "fontSize": "0.85em"}),
                                    dcc.Textarea(
                                        id="input-behavior-comment",
                                        placeholder="Comments on behavioral data...",
                                        style={"marginLeft": "16px", "flex": 1,
                                               "height": "32px", "fontSize": "0.85em",
                                               "resize": "vertical"},
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center",
                                       "marginBottom": "8px"},
                            ),
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
                                responsive=True,
                                style={"display": "none", "height": "calc(100vh - 120px)"},
                                config={
                                    # scroll = zoom x, drag = pan x. y-axes are all
                                    # fixedrange=True, so zoom is x-only by construction.
                                    "scrollZoom": True,
                                    "displaylogo": False,
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "neurodash_session",
                                        "scale": 2,
                                    },
                                },
                            ),
                        ],
                        id="div-viewer-content",
                    ),

                    # Channel Viewer pane — always mounted, hidden until tab selected
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(id="div-channel-animal",
                                              style={"fontSize": "0.85em",
                                                     "fontWeight": "bold",
                                                     "marginRight": "16px"}),
                                    html.Button("Download to CSV",
                                                id="btn-download-channel-csv", n_clicks=0),
                                    html.Span(id="div-channel-export-status",
                                              style={"marginLeft": "10px",
                                                     "color": "#999",
                                                     "fontSize": "0.85em"}),
                                    dcc.Textarea(
                                        id="input-channel-comment",
                                        placeholder="Comments on neural data...",
                                        style={"marginLeft": "16px", "flex": 1,
                                               "height": "32px", "fontSize": "0.85em",
                                               "resize": "vertical"},
                                    ),
                                ],
                                style={"marginBottom": "8px", "display": "flex",
                                       "alignItems": "center"},
                            ),
                            dcc.Loading(html.Div(id="channel-content")),
                        ],
                        id="div-channel-wrap",
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
            dcc.Store(id="store-channel-exemplar", data=None),
            dcc.Store(id="store-channel-saved"),  # sink for the silent annotation auto-save
            dcc.Store(id="store-behavior-saved"),  # sink for the silent behavior-note auto-save
        ],
        style={"display": "flex", "height": "100vh"},
    )


# ---------------------------------------------------------------------------
# Channel Viewer — one synced figure + an aligned column of per-channel
# controls, built lazily when the Channel Viewer tab is opened (see callbacks)
# ---------------------------------------------------------------------------

def exemplar_glyph(is_exemplar):
    return "★" if is_exemplar else "☆"


def exemplar_button_style(is_exemplar):
    return {
        "border": "none", "background": "none", "fontSize": "1.4em",
        "cursor": "pointer", "padding": "0",
        "color": "#e0a800" if is_exemplar else "#bbb",
    }


def _channel_control_block(idx, label, entry, is_exemplar):
    """One channel's controls, sized (flex:1) to line up with its figure row."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(label, style={"fontWeight": "bold"}),
                    dcc.Checklist(
                        id={"type": "channel-include", "index": idx},
                        options=[{"label": " Save", "value": "y"}],
                        value=["y"] if entry.get("include", True) else [],
                        style={"fontSize": "0.8em"},
                    ),
                    html.Button(
                        exemplar_glyph(is_exemplar),
                        id={"type": "channel-exemplar", "index": idx},
                        title="Mark as exemplar channel",
                        n_clicks=0,
                        style=exemplar_button_style(is_exemplar),
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center"},
            ),
            dcc.RadioItems(
                id={"type": "channel-quality", "index": idx},
                options=[{"label": f" {q.capitalize()}", "value": q}
                         for q in CHANNEL_QUALITY_OPTIONS],
                value=entry.get("quality") or None,
                inline=True,
                style={"fontSize": "0.85em", "marginTop": "4px"},
            ),
            dcc.Input(
                id={"type": "channel-comment", "index": idx},
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


def build_channel_view(session, channel_data, view_range=None, spect_params=None):
    """Build the Channel Viewer: a controls gutter aligned to one synced figure.

    The gutter's height + top/bottom padding mirror the figure's height +
    margins, and both divide into n equal rows, so each control block lines up
    with its channel's row in the figure.
    """
    sig_info = session.analog_signal_summaries[0]
    exemplar = channel_data.get("exemplar_channel_index")
    channels = channel_data.get("channels", {})
    n = len(sig_info["channel_indices"])

    gutter = html.Div(
        [
            _channel_control_block(
                idx, label,
                channels.get(str(idx), {"quality": "", "comment": "", "include": True}),
                idx == exemplar,
            )
            for idx, label in zip(sig_info["channel_indices"], sig_info["channel_labels"])
        ],
        style={
            "display": "flex", "flexDirection": "column",
            "width": "200px", "flexShrink": 0,
            "height": f"{CHANNEL_ROW_HEIGHT * n + 50}px",
            "paddingTop": "10px", "paddingBottom": "40px",
            "boxSizing": "border-box",
        },
    )
    stem = Path(session.pl2_path).stem
    graph = dcc.Graph(
        id="channel-plot",
        figure=plot_channel_figure(session, view_range, spect_params),
        style={"flex": 1, "minWidth": 0},
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": f"{stem}_channels", "scale": 2},
        },
    )
    return html.Div([gutter, graph], style={"display": "flex", "alignItems": "flex-start"})
