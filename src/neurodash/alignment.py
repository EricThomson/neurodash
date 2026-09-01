"""Putting the neural and behavioral clocks on one time base.

Open field needed none of this: the Plexon and EthoVision recordings were started
together, so behavior t=0 is neural sample 0 and the two axes already agree.

Fear-conditioning sessions do not agree. On the acquisition test file the Plexon
recording starts 33.05 s *before* the behavior recording and stops 33.86 s before
it too, and nothing in either file states the relationship directly — there is no
behavior-onset pulse anywhere (RSTART/RSTOP are empty and the pl2 carries no
rec_datetime), so the alignment has to be inferred.

**Everything is displayed and exported on the behavior clock**, i.e. seconds from
the start of the behavior recording. That matches the video 1:1, matches the CSV,
matches the lab's own `Events.start` convention ("all other events are relative to
this"), and keeps `time` comparable across sessions — pl2 clock would put 5739 in
a column that means "seconds into the session" everywhere else in the app.

The anchor is the END of both recordings, because that is the only correspondence
the files support. Its specific weakness: anything that changes the behavior
file's *duration* — a truncated export, a trimmed CSV, dropped frames — shifts the
whole alignment silently. A start TTL would be immune. Hence `check_alignment`,
which re-derives the answer from the animal's startle response and is the only
thing standing between a bad file and plausible-looking nonsense.
"""

import numpy as np

from neurodash import config
from neurodash.behavior_io import behavior_time
from neurodash.freezeframe_io import MOTION_COLUMN, run_time_seconds


def _first(events, names):
    for name in names:
        times = events.get(name)
        if times is not None and len(times):
            return np.asarray(times, dtype=float)
    return None


def tone_times(events):
    """Tone onsets in pl2 clock, or None when this session has no tones."""
    return _first(events or {}, config.EVENT_TONE_CHANNELS)


def shock_times(events):
    """Shock onsets in pl2 clock, or None (a tone or context session has none)."""
    return _first(events or {}, config.EVENT_SHOCK_CHANNELS)


def stop_time(events):
    """The stop marker in pl2 clock — the end of the behavior recording.

    A lone event after all tones and shocks. The lab's `Events` table has both a
    `start` and a `stop`; this file only has the latter, so check other sessions
    for a start channel before assuming end-anchoring is the general rule.
    """
    times = _first(events or {}, config.EVENT_STOP_CHANNELS)
    return float(times[-1]) if times is not None else None


def behavior_start_in_pl2(events, behavior_metadata, segment=None):
    """Where behavior t=0 sits on the pl2 clock, or None if it can't be found.

    ``stop marker - behavior duration``. The segment's own ``t_stop`` is the
    free cross-check: the two agree to 0.05 s on the acquisition file (5578.37 vs
    5578.42), both well inside the 0.1-0.6 s startle lag that limits how precisely
    this can be known at all. Disagreement between them means something is wrong
    with the file, so it is reported rather than silently averaged.
    """
    duration = run_time_seconds(behavior_metadata)
    if duration is None:
        return None
    stop = stop_time(events)
    if stop is None:
        stop = segment[1] if segment else None
    return None if stop is None else float(stop) - duration


def neural_time_offset(events, behavior_metadata, segment):
    """Seconds to add to neural *sample* time to get behavior time.

    Neural sample i sits at `i / rate` seconds from the first sample, which is
    `t_start` on the pl2 clock; behavior time subtracts the anchor from that:

        offset = t_start - behavior_start_in_pl2

    Open field has no events, so there is nothing to anchor to and the offset is
    0 — which is exactly right there, because behavior t=0 *is* neural sample 0.
    That is also why the app has always been correct while ignoring `t_start`:
    sample-relative time is the right primitive, and `t_start` only matters for
    comparing against event times, which live on the pl2 clock.

    Measured on the acquisition file: t_start 5545.324 - anchor 5578.371 =
    **-33.047 s**, i.e. the neural recording began 33 s before the behavior one.
    """
    anchor = behavior_start_in_pl2(events, behavior_metadata, segment)
    if anchor is None or segment is None:
        return 0.0
    return float(segment[0]) - anchor


def check_alignment(events, behavior_metadata, behavior_data, segment=None):
    """Re-derive the offset from the animal's startle, and report disagreement.

    A shock makes the animal jump, so the largest isolated spikes in ``Motion
    Index`` must land on the shock TTLs. On the acquisition file those spikes are
    3-5x anything else in the session (4798-7810 against a next tier of ~1500), so
    they are unambiguous, and they match the end-anchored prediction to within
    0.10-0.60 s on all five trials.

    That residual is the startle *response* lagging the TTL, plus 33 ms frame
    binning — it is not clock error, so do not "improve" the alignment by fitting
    to the motion peaks. The check is for gross failure, which is what a truncated
    or trimmed behavior file produces.

    Returns ``(ok, message)``. ``ok`` is True when there is nothing to check
    against (no shocks, no motion column, non-FreezeFrame data) — this reports
    problems it can prove, not absence of evidence.
    """
    shocks = shock_times(events)
    anchor = behavior_start_in_pl2(events, behavior_metadata, segment)
    if shocks is None or anchor is None or behavior_data is None:
        return True, ""
    if MOTION_COLUMN not in behavior_data.columns:
        return True, ""

    predicted = np.asarray(shocks, dtype=float) - anchor
    times = behavior_time(behavior_data)
    motion = behavior_data[MOTION_COLUMN].to_numpy(dtype=float)

    # Checked first, and separately from the startle, because it is the *direct*
    # test of the thing end-anchoring is vulnerable to: the anchor is
    # `stop marker - Run Time`, so if Run Time disagrees with how far the data
    # actually runs, every derived time is shifted by that difference. It is also
    # unambiguous, which the startle test is not — so it still works on the
    # no-shock control animals, where there is no startle to look for.
    stated = run_time_seconds(behavior_metadata)
    actual = float(times[-1]) if len(times) else None
    if stated is not None and actual is not None:
        drift = actual - stated
        if abs(drift) > config.ALIGNMENT_DURATION_TOLERANCE_S:
            return False, (
                f"The behavior file says it ran {stated:.1f} s but its data runs to "
                f"{actual:.1f} s ({drift:+.1f} s). Alignment is anchored on "
                f"'Run Time', so every neural time is shifted by that much — the "
                f"export is probably truncated or trimmed.")

    # Ask whether a startle happens AT each shock TTL, rather than whether the
    # session's biggest spikes happen to be the shocks. Both alternatives were
    # tried and are worse:
    #
    #   - Largest-peak-near-each-prediction caps the error the check can report at
    #     the search-window width, so a 60 s truncation scored 1.99 s — barely
    #     distinguishable from a good file.
    #   - Rank the session's top spikes and match them to TTLs: the animal moves
    #     hard for its own reasons. This file has a 6145 spike at 221.0 s, bigger
    #     than three of the five shock responses, so the real shock at 644.9 s
    #     falls out of the top five and an intact file fails.
    #
    # A threshold instead of a ranking: the shock responses are 4798-7810 against
    # a 99.9th percentile of 3623, so all five clear it, while a misaligned window
    # lands on ordinary motion (median 79, 99th percentile 1033).
    threshold = float(np.nanpercentile(motion, config.ALIGNMENT_SPIKE_PERCENTILE))
    matched, worst_time = 0, None
    for t in predicted:
        window = (times >= t - config.ALIGNMENT_TOLERANCE_S) & \
                 (times <= t + config.ALIGNMENT_TOLERANCE_S)
        if window.any() and np.nanmax(motion[window]) >= threshold:
            matched += 1
        elif worst_time is None:
            worst_time = float(t)

    # None at all is ambiguous, and saying "misaligned" would be wrong half the
    # time. The shock TTL fires for the whole rig, but only one box is wired to
    # deliver it: on acquisition day the Box 1 animals are **no-shock controls**
    # (confirmed against the lab's session notes, where Box 1 is "No Shock" and
    # Box 2 "Shock" for every pair). A control animal has no startle to find, so
    # a confident failure here would condemn a perfectly good file.
    #
    # A *partial* match is different — it means startles exist, so the animal was
    # shocked, and some of them landing off the TTLs is real misalignment.
    if matched < config.ALIGNMENT_MIN_CONFIDENT_MATCHES:
        return False, (
            f"Startle found at only {matched} of {len(predicted)} shock TTLs. "
            f"Either this animal was not shocked (the no-shock control box), in "
            f"which case the alignment cannot be checked this way, or the neural "
            f"and behavior files do not belong together.")
    if matched < len(predicted):
        return False, (
            f"Only {matched} of {len(predicted)} shock TTLs coincide with a startle "
            f"in Motion Index (first miss at {worst_time:.1f} s). The neural and "
            f"behavior files may not belong together, or the behavior export may be "
            f"truncated — alignment is anchored on the end of both recordings, so a "
            f"changed duration shifts everything.")
    return True, (f"Alignment checked: all {len(predicted)} shock TTLs coincide "
                  f"with a startle.")


def trial_structure(events, anchor):
    """Tone/shock times on the behavior clock, plus the derived trace interval.

    Durations are not in the file — no event channel carries one — so tone and
    shock *lengths* are protocol constants from config. The trace interval is the
    one timing that is measured per session: shock onset minus tone offset.
    """
    result = {"tones": None, "shocks": None, "trace_s": None}
    if anchor is None:
        return result
    tones = tone_times(events)
    shocks = shock_times(events)
    if tones is not None:
        result["tones"] = tones - anchor
    if shocks is not None:
        result["shocks"] = shocks - anchor
    if tones is not None and shocks is not None and len(tones) == len(shocks):
        gaps = (shocks - tones) - config.TONE_DURATION_S
        result["trace_s"] = float(np.median(gaps))
    return result
