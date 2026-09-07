/* Session navigator interaction — see src/neurodash/session_navigator.py.
 *
 * Vanilla JS on purpose: the strip is plain DOM and must stay out of Plotly's
 * render loop. During a drag only the viewport box moves (style writes, which
 * the browser does at 60fps); the plot gets a THROTTLED relayout mid-drag and
 * one more on release. An earlier sticky-label attempt relayouted on every drag
 * tick and made the app unusable, which is the rule this file exists to respect.
 *
 * Two entry points, both registered as clientside callbacks in callbacks.py:
 *
 *   init     - rebinds to a freshly rendered strip. Keyed on the strip's
 *              children, so Dash decides WHEN, and there is no readiness poll.
 *   syncBox  - paints the box from store-view-range, so the box follows native
 *              pan/zoom. Going through the existing store rather than binding
 *              plotly_relayout directly means no listener to lose when Dash
 *              re-renders the graph, and it cannot loop: this path only ever
 *              writes CSS.
 */
(function () {
    "use strict";

    var MIN_WINDOW_S = 2.0;     /* resize floor, keeps the box grabbable */
    var EDGE_ZONE_PX = 8;       /* how close to a box edge counts as resize */
    var MIN_MOVE_ZONE_PX = 16;  /* middle of the box always grabbable for a move */
    var CLICK_SLOP_PX = 3;      /* mouseup within this of mousedown = a click */

    /* How wide the resize zones may be on a box THIS wide.
     *
     * Fixed 8 px edges are wrong here, and badly: a 30 s window on a 1291 s
     * session makes the box ~2% of the strip, so two 8 px edges on a ~25 px box
     * leave under 10 px to grab and nearly every drag became a resize. The
     * narrower the box, the more it becomes move-only - which is the right
     * trade, because resizing by dragging a 3 px edge was never a real
     * interaction anyway, and at that zoom panning is what you want. Use the
     * View duration box or scroll-zoom to resize a tiny window.
     */
    function edgeZone(boxWidthPx) {
        return Math.max(0, Math.min(EDGE_ZONE_PX,
                                    (boxWidthPx - MIN_MOVE_ZONE_PX) / 2));
    }

    /* Module-level so a rebuilt strip reuses them. The document-level listeners
       are attached exactly once; binding them per init would stack a duplicate
       set every time a session is loaded. */
    var state = {
        strip: null, box: null, tEnd: 0, throttleMs: 125,
        drag: null, relayoutInFlight: false, lastCommit: 0, documentBound: false,
    };

    /* Resolved per use, never cached: the figure is rebuilt by its own callback
       and may not exist when the strip is first bound. */
    function plotDiv() {
        return document.querySelector("#main-plot .js-plotly-plot");
    }

    function currentRange() {
        var gd = plotDiv();
        if (!gd || !gd.layout || !gd.layout.xaxis || !gd.layout.xaxis.range) {
            return null;
        }
        var r = gd.layout.xaxis.range;
        return [Number(r[0]), Number(r[1])];
    }

    function paintBox(x0, x1) {
        if (!state.box || !state.tEnd) return;
        state.box.style.left = (100 * x0 / state.tEnd) + "%";
        state.box.style.width = (100 * (x1 - x0) / state.tEnd) + "%";
    }

    function commit(x0, x1) {
        var gd = plotDiv();
        if (!gd) return;
        state.relayoutInFlight = true;
        window.Plotly.relayout(gd, { "xaxis.range": [x0, x1] }).then(function () {
            state.relayoutInFlight = false;
        }, function () {
            state.relayoutInFlight = false;
        });
    }

    function clampedMove(range, dt) {
        var width = range[1] - range[0];
        var x0 = Math.min(Math.max(range[0] + dt, 0), Math.max(state.tEnd - width, 0));
        return [x0, x0 + width];
    }

    function clampedResize(range, dt, whichEdge) {
        var x0 = range[0], x1 = range[1];
        if (whichEdge === "left") {
            x0 = Math.min(Math.max(x0 + dt, 0), x1 - MIN_WINDOW_S);
        } else {
            x1 = Math.max(Math.min(x1 + dt, state.tEnd), x0 + MIN_WINDOW_S);
        }
        return [x0, x1];
    }

    /* Centre the current window on the pointer. Used by the click-to-place and
       by scrubbing the strip background. */
    function centredOnPointer(event, range) {
        var rect = state.strip.getBoundingClientRect();
        var frac = (event.clientX - rect.left) / rect.width;
        var center = Math.min(Math.max(frac, 0), 1) * state.tEnd;
        var width = range[1] - range[0];
        return clampedMove([center - width / 2, center + width / 2], 0);
    }

    /* Takes the drag explicitly rather than reading state.drag, so mouseup can
       clear the drag first and still ask where it ended. */
    function dragRange(event, drag) {
        if (drag.mode === "scrub") return centredOnPointer(event, drag.startRange);
        var rect = state.strip.getBoundingClientRect();
        var dt = (event.clientX - drag.startClientX) / rect.width * state.tEnd;
        if (drag.mode === "move") return clampedMove(drag.startRange, dt);
        return clampedResize(drag.startRange, dt, drag.mode);
    }

    function bindDocumentOnce() {
        if (state.documentBound) return;
        state.documentBound = true;

        document.addEventListener("mousemove", function (event) {
            if (!state.drag) return;
            /* Pressed on the background and now moving: this is a scrub, not a
               click. Promoting here is what makes the ENTIRE strip a sliding
               surface, so panning never depends on hitting a box that may be
               only a couple of percent wide. */
            if (state.drag.mode === "maybe-click") {
                if (Math.abs(event.clientX - state.drag.startClientX)
                        <= CLICK_SLOP_PX) {
                    return;
                }
                state.drag.mode = "scrub";
                state.strip.style.cursor = "grabbing";
            }
            var range = dragRange(event, state.drag);
            paintBox(range[0], range[1]);

            /* Live follow. The in-flight guard is what adapts to slow hardware:
               a machine that cannot finish a relayout cannot start the next, so
               follow degrades to a gentler rate instead of queueing. */
            if (state.relayoutInFlight) return;
            var now = performance.now();
            if (now - state.lastCommit >= state.throttleMs) {
                state.lastCommit = now;
                commit(range[0], range[1]);
            }
        });

        document.addEventListener("mouseup", function (event) {
            if (!state.drag) return;
            var finished = state.drag;
            state.drag = null;
            state.strip.style.cursor = "";

            /* Still "maybe-click" means it never moved: place the window here. */
            if (finished.mode === "maybe-click") {
                var placed = centredOnPointer(event, finished.startRange);
                paintBox(placed[0], placed[1]);
                commit(placed[0], placed[1]);
                return;
            }
            var range = dragRange(event, finished);
            paintBox(range[0], range[1]);
            commit(range[0], range[1]);
        });
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        session_navigator: {

            init: function () {
                var strip = document.getElementById("nav-strip");
                var box = document.getElementById("nav-viewport");
                /* No strip for this session (nothing loaded) - nothing to bind. */
                if (!strip || !box) return window.dash_clientside.no_update;

                state.strip = strip;
                state.box = box;
                state.tEnd = parseFloat(strip.dataset.tend) || 0;
                state.throttleMs = parseFloat(strip.dataset.throttleMs) || 125;
                state.drag = null;

                /* Per-element listeners. Safe to add unconditionally: these
                   nodes are new, the old ones went away with the old strip. */
                box.addEventListener("mousemove", function (event) {
                    if (state.drag) return;
                    var rect = box.getBoundingClientRect();
                    var zone = edgeZone(rect.width);
                    var nearEdge = zone > 0
                        && ((event.clientX - rect.left < zone)
                            || (rect.right - event.clientX < zone));
                    box.style.cursor = nearEdge ? "ew-resize" : "grab";
                });

                box.addEventListener("mousedown", function (event) {
                    var range = currentRange();
                    if (!range) return;
                    var rect = box.getBoundingClientRect();
                    var zone = edgeZone(rect.width);
                    var mode = "move";
                    if (zone > 0) {
                        if (event.clientX - rect.left < zone) mode = "left";
                        else if (rect.right - event.clientX < zone) mode = "right";
                    }
                    state.drag = { mode: mode, startClientX: event.clientX,
                                   startRange: range };
                    event.preventDefault();
                    event.stopPropagation();
                });

                /* Strip background: click to centre the view there. */
                strip.addEventListener("mousedown", function (event) {
                    if (event.target === box) return;
                    var range = currentRange();
                    if (!range) return;
                    state.drag = { mode: "maybe-click", startClientX: event.clientX,
                                   startRange: range };
                    event.preventDefault();
                });

                bindDocumentOnce();

                var initial = currentRange();
                if (initial) paintBox(initial[0], initial[1]);
                return window.dash_clientside.no_update;
            },

            syncBox: function (viewRange) {
                /* Mid-drag the box is already where the pointer put it, and a
                   lagging store update would snap it backwards. */
                if (state.drag || !viewRange) return window.dash_clientside.no_update;
                paintBox(Number(viewRange[0]), Number(viewRange[1]));
                return window.dash_clientside.no_update;
            },
        },
    });
})();
