"""The session navigator strip: whole-session overview above the detail view.

The overview+detail pattern (Highcharts Stock's navigator, IGV's chromosome
ideogram): a thin strip showing the entire session, with a viewport box you can
drag to pan, resize by its edges to zoom, or click past to jump. This module
owns the whole feature except the browser-side interaction, which lives in
`assets/session_navigator.js`; the same shape as `arena_io.py`, which holds all
of arena calibration while the viewer is just the UI.

It does three separable jobs, and only the first needs epochs:

  1. orientation within trial structure — which epoch am I in, where is the
     next shock;
  2. navigation of a long recording — 1291 s is ~43 screens at the default 30 s
     window, and dragging a box beats repeatedly panning;
  3. proportion — the box's width against the strip says what FRACTION of the
     session is on screen, and the strip's extent makes session length a picture
     instead of a number in the sidebar.

Jobs 2 and 3 need only a time axis, which is why OPEN FIELD GETS THE STRIP TOO.
It is built data-driven for that reason: segments come from
`session.trial_events` when there are any and the list is simply empty
otherwise, so nothing here branches on session type. An open-field strip is a
plain bar with a viewport box, and that is a deliberate outcome rather than a
degenerate case.

**The design rule this must respect**: the strip is plain DOM, and the only
Plotly calls are DISCRETE relayouts. An earlier attempt at sticky epoch labels
called `Plotly.relayout` on every drag tick, re-rendering a million-point figure
continuously, and made the whole app unusable. During a drag only the box moves
(pure CSS); the plot is committed on a throttle and once more on release.
"""

from dash import html

from neurodash import config, epochs
from neurodash.behavior_io import behavior_time

# Segment fills are the config hex colour plus an alpha byte (8-digit hex CSS).
# The strip carries more saturation than the in-plot bands because it has no
# data underneath it to obscure.
SEGMENT_ALPHA_HEX = "59"          # ~35 percent

# Segments narrower than this fraction of the session get no text label: at
# realistic strip widths the text truncates to debris ("ton", "trac"). Narrow
# segments stay identifiable by colour and by their hover tooltip.
LABEL_MIN_FRACTION = 0.03


def session_extent(session):
    """How far the session runs. Thin alias for Session.extent_s.

    Kept as a name because the strip is entirely a function of this number, but
    the logic lives on Session — `_plot_events` needs the same answer, and the
    two had drifted apart with a bug on the plot_utils side.
    """
    return session.extent_s


def build_strip(session, epoch_params=None):
    """The navigator strip for this session, or None when there is nothing to show.

    Everything is positioned in percent of the session, so neither this function
    nor the CSS needs to know the strip's pixel width, and a window resize costs
    nothing. The viewport box is the ONLY element that ever moves;
    `assets/session_navigator.js` moves it by writing left/width.
    """
    t_end = session_extent(session)
    if t_end is None:
        return None

    def pct(t):
        return f"{100.0 * t / t_end:.4f}%"

    trial = session.trial_events
    tones, shocks = trial["tones"], trial["shocks"]

    children = []
    for band in epochs.build_epochs(tones, shocks, epoch_params, t_end):
        wide_enough = ((band["end"] - band["start"]) / t_end) >= LABEL_MIN_FRACTION
        children.append(html.Div(
            html.Span(band["label"]) if wide_enough else None,
            className="nav-segment",
            title=f'{band["label"]}: {band["start"]:.1f} - {band["end"]:.1f} s',
            style={
                "left": pct(band["start"]),
                "width": pct(band["end"] - band["start"]),
                "background": config.EPOCH_COLORS[band["kind"]] + SEGMENT_ALPHA_HEX,
            },
        ))

    # Shocks only. Tones already read as the start of their own tone segment,
    # whereas the shock falls in the guard band between trace and isi and so has
    # nothing else marking it (epochs.py: "the shock is never inside an epoch").
    for span in session.event_spans(epoch_params):
        if span["kind"] != "shock_event":
            continue
        children.append(html.Div(
            className="nav-shock-tick",
            title=f'shock: {span["start"]:.1f} s',
            style={"left": pct(span["start"]),
                   "width": pct(span["end"] - span["start"]),
                   "background": config.EVENT_SERIES_COLORS["shock"]},
        ))

    children.append(html.Div(id="nav-viewport"))

    return html.Div(
        html.Div(
            children,
            id="nav-strip",
            # Read by the JS. Passed as data attributes rather than hardcoded in
            # the .js so Python stays the single source for both.
            **{"data-tend": f"{t_end:.4f}",
               "data-throttle-ms": str(config.NAVIGATOR_LIVE_THROTTLE_MS)},
        ),
        # Must line up with the figure's plotting area, hence the shared margins.
        style={"paddingLeft": f"{config.PLOT_MARGIN_LEFT_PX}px",
               "paddingRight": f"{config.PLOT_MARGIN_RIGHT_PX}px",
               "marginBottom": "4px"},
    )
