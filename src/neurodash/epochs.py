"""Carving a trace-conditioning session into epochs.

`baseline`, `tone1-5`, `trace1-5`, `isi1-5`, derived from the TTL onsets plus a
handful of buffer parameters. See `sandbox/acquisition/epoch_windows.md` for the
rationale and `isi_window_bug.md` for the error this replaces.

Two things about these are easy to get wrong:

**They are not a partition of the session.** They are analysis windows with
deliberate gaps between them, and a large fraction of the recording belongs to no
epoch at all. The lab's `make_epoch_series` labels those bins `""`, and so does
`epoch_labels` here — a blank epoch means "deliberately excluded", not missing.

**The shock is never inside an epoch.** It sits in the gap between `traceI`
(which ends `trace_pad` before it) and `isiI` (which starts `post_shock_delay`
after it), so it needs its own marker anywhere epochs are drawn.
"""

import numpy as np

from neurodash import config


def epoch_params(overrides=None):
    """Buffer parameters, config defaults with any UI overrides applied.

    Kept as one dict rather than eight arguments because they travel together
    everywhere — figure, export header and epoch construction all need the same
    set, and a mismatch between them would be invisible.
    """
    params = dict(config.ACQUISITION_EPOCH_PARAMS)
    for key, value in (overrides or {}).items():
        if value is not None and key in params:
            params[key] = float(value)
    return params


def build_epochs(tones, shocks, params=None, session_end=None):
    """Epoch windows on the behavior clock.

    Parameters
    ----------
    tones, shocks : array-like or None — onsets in behavior time.
    params : dict — see epoch_params.
    session_end : float or None — clip windows to the recording.

    Returns
    -------
    list of {"label", "kind", "start", "end"}, in time order. ``kind`` is one of
    baseline/tone/trace/isi and drives colour; ``label`` is what goes in the CSV.

    A session with no tones (context) gets no epochs at all rather than an
    invented baseline: `baseline` is defined relative to the first tone, so
    without one there is nothing to anchor it to.
    """
    p = epoch_params(params)
    tones = None if tones is None else np.asarray(tones, dtype=float)
    shocks = None if shocks is None else np.asarray(shocks, dtype=float)
    if tones is None or not len(tones):
        return []

    out = [{"label": "baseline", "kind": "baseline",
            "start": p["baseline_start"],
            "end": float(tones[0]) - p["baseline_pad"]}]

    for i, tone in enumerate(tones, start=1):
        out.append({"label": f"tone{i}", "kind": "tone",
                    "start": tone + p["tone_pad"],
                    "end": tone + p["tone_duration"] - p["tone_pad"]})

    if shocks is not None and len(shocks) == len(tones):
        for i, (tone, shock) in enumerate(zip(tones, shocks), start=1):
            out.append({"label": f"trace{i}", "kind": "trace",
                        "start": tone + p["tone_duration"] + p["trace_pad"],
                        "end": shock - p["trace_pad"]})
        for i, shock in enumerate(shocks, start=1):
            # delay + duration, deliberately. The notebooks wrote this as
            # `delay + record_duration - trailing_buffer`, which subtracts the
            # buffer from a length that already excludes it; with a non-zero
            # trailing buffer that silently produced 90 s windows where 120 s was
            # intended, and it reached the saved results. See isi_window_bug.md.
            start = shock + p["post_shock_delay"]
            out.append({"label": f"isi{i}", "kind": "isi",
                        "start": start, "end": start + p["isi_duration"]})

    out = [e for e in out if e["end"] > e["start"]]
    if session_end is not None:
        for epoch in out:
            epoch["end"] = min(epoch["end"], float(session_end))
        out = [e for e in out if e["end"] > e["start"]]
    return sorted(out, key=lambda e: e["start"])


def event_spans(tones, shocks, params=None):
    """Tone and shock spans on the behavior clock — the stimuli themselves.

    Distinct from the epochs: the tone *epoch* is inset by `tone_pad`, and the
    shock has no epoch at all. These are what actually happened to the animal.
    """
    p = epoch_params(params)
    spans = []
    for onset in (np.asarray(tones, dtype=float) if tones is not None else []):
        spans.append({"kind": "tone_event", "start": float(onset),
                      "end": float(onset) + p["tone_duration"]})
    for onset in (np.asarray(shocks, dtype=float) if shocks is not None else []):
        spans.append({"kind": "shock_event", "start": float(onset),
                      "end": float(onset) + p["shock_duration"]})
    return sorted(spans, key=lambda s: s["start"])


def overlap_warnings(epochs, tones, params=None):
    """Ways the current buffers produce windows that don't make sense.

    Surfaced because the buffers are user-editable and the failure is quiet: an
    ISI window long enough to run into the next tone still plots, it just stops
    measuring what it claims to. This is the check the notebooks lacked.
    """
    warnings = []
    tones = np.asarray(tones, dtype=float) if tones is not None else np.array([])
    for epoch in epochs:
        later = tones[tones > epoch["start"]]
        if epoch["kind"] == "isi" and len(later) and epoch["end"] > later[0]:
            warnings.append(
                f"{epoch['label']} runs {epoch['end'] - later[0]:.1f} s past the "
                f"next tone — shorten the ISI duration or the post-shock delay.")
    for a, b in zip(epochs, epochs[1:]):
        if b["start"] < a["end"]:
            warnings.append(f"{a['label']} and {b['label']} overlap by "
                            f"{a['end'] - b['start']:.1f} s.")
    return warnings


def epoch_labels(times, epochs):
    """Epoch label per time point; "" where a point is in no epoch.

    Interval containment, never averaging — an epoch is a categorical label, so
    the notebook is explicit that upsampled epochs are "a farce". Blank is a real
    value here, meaning a bin the analysis deliberately excludes.
    """
    times = np.asarray(times, dtype=float)
    labels = np.full(len(times), "", dtype=object)
    for epoch in epochs:
        inside = (times >= epoch["start"]) & (times <= epoch["end"])
        labels[inside] = epoch["label"]
    return labels


def event_mask(times, spans, kind):
    """Binary 0/1 per time point for one event kind (tone or shock)."""
    times = np.asarray(times, dtype=float)
    mask = np.zeros(len(times), dtype=int)
    for span in spans:
        if span["kind"] == kind:
            mask[(times >= span["start"]) & (times <= span["end"])] = 1
    return mask
