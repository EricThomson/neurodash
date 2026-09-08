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
from neurodash.freezeframe_io import run_time_seconds


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


def behavior_start_in_pl2(events, behavior_metadata, segment=None, fragments=None):
    """Where behavior t=0 sits on the pl2 clock, or None if it can't be found.

    **The two recordings START together; they do not stop together.** Behavior
    t=0 is the moment neural acquisition begins. The lab's wrangling notes say so
    outright: *"we often have 1291 measures of behavior, but the duration of the
    neuronal recordings is 1290, and Jesse said that time zero of the behavioral
    measure is the start time TTL."* Our file shows exactly that shape —
    1290.933 s of behavior against 1290.117 s of neural — the behavior simply
    running on after the neural stops. It is also what open field assumes, where
    neural and behavior share an origin and only the *video* needs an explicit
    delay (EthoVision's `Recording after`).

    So the anchor is the start of the analog data, taken from the fragment table,
    which is the only thing that states where the samples actually begin.

    **End-anchoring (`stop marker - behavior duration`) was wrong.** It put
    behavior t=0 at 5578.371 against the true 5579.344 — an 0.97 s error that
    drew each shock TTL almost a second after the animal's response to it. Motion
    rose 0.83 s *before* the shock on all five trials, and the response straddled
    the 2 s shock window instead of filling it. Kept only as a fallback for a file
    whose fragment table can't be read, where a rough anchor beats none.
    """
    if fragments and segment is not None:
        biggest = max(fragments, key=lambda f: f["n_samples"])
        return float(segment[0]) + biggest["start_s"]
    duration = run_time_seconds(behavior_metadata)
    if duration is None:
        return None
    stop = stop_time(events)
    if stop is None:
        stop = segment[1] if segment else None
    return None if stop is None else float(stop) - duration


def neural_time_offset(events, behavior_metadata, segment, neural_duration=None,
                       fragments=None, sampling_rate=1000.0):
    """Seconds to add to neural *sample* time to get behavior time.

    Both recordings are stopped together, so the END is the anchor for this as
    well as for `behavior_start_in_pl2`. Sample 0 sits `neural_duration` before
    the stop and behavior t=0 sits `behavior_duration` before it, so:

        offset = behavior_duration - neural_duration

    On the acquisition file that is 1290.933 - 1290.117 = **+0.82 s**: the analog
    recording started a fraction of a second after the behavior one.

    **Do not use `segment.t_start` for this.** That was the first implementation
    and it was wrong by 33 s. neo reports `t_start` 5545.324 for this file, but
    the analog samples do not begin there — anchoring on it put the shock
    artifacts 33 s away from the shock TTLs, which is exactly what the lab's own
    ingestion notes warn about: "you need to subtract `Start` for everything and
    the numbers will all work out and lock to ephys/artifacts properly in the
    time array you build". `t_start` is not that Start.

    Open field has no events, so there is nothing to anchor to and the offset is
    0 — right there, because its two recordings were started together and
    behavior t=0 *is* neural sample 0.
    """
    anchor = behavior_start_in_pl2(events, behavior_metadata, segment, fragments)
    duration = run_time_seconds(behavior_metadata)
    if anchor is None or duration is None:
        return 0.0

    # Behavior t=0 is the start of the analog data, so inside the main fragment
    # neural sample time already IS behavior time. What remains is neo's own
    # bookkeeping: it concatenates the fragments, so any earlier fragment's
    # samples shift the main one's indices, and the offset undoes exactly that.
    if fragments and segment is not None:
        biggest = max(range(len(fragments)), key=lambda k: fragments[k]["n_samples"])
        before = sum(f["n_samples"] for f in fragments[:biggest]) / float(sampling_rate)
        return -before

    # Fallback when the fragment table is unreadable. Approximate: it assumes the
    # recordings ended together, which they do not (see behavior_start_in_pl2).
    if neural_duration is None:
        return 0.0
    return float(duration) - float(neural_duration)


def check_alignment(events, behavior_metadata, behavior_data, segment=None,
                    fragments=None):
    """Report the one alignment failure this can prove: a truncated behavior file.

    The anchor is the behavior file's own duration, so if its stated `Run Time`
    disagrees with how far its data actually runs, every derived neural time is
    shifted by that difference. Direct, unambiguous, and it works on every animal.

    **The startle check was removed (Eric, Sep 2026).** It asked whether a jump in
    Motion Index happened at each shock TTL. That is only meaningful for shocked
    animals, and the shock TTL fires rig-wide while only one box delivers it — on
    acquisition day the Box 1 animals are no-shock controls, so roughly half of
    all sessions could only ever report "0 of 5 startles found, either this animal
    was not shocked or the files don't belong together". Correct, and useless: a
    message that fires on every control animal is noise the user learns to ignore,
    which is worse than no check.

    Two things worth not relearning, since the code is gone:

    * Ranking the session's biggest spikes and matching them to TTLs FAILS an
      intact file — the animal moves hard for its own reasons, and a 6145 spike at
      221.0 s outranked three of the five real shock responses.
    * Taking the largest peak near each prediction caps the error the check can
      report at the search-window width, so a 60 s truncation scored 1.99 s.

    The real instrument for the neural<->behavior clock is the shock-triggered
    motion average, and for the pl2-internal clock it is artifact causality; both
    are run by hand, in sandbox. See sandbox/docs/acquisition_timing_issues.md.

    Returns ``(ok, message)``. ``ok`` is True when there is nothing to check —
    this reports problems it can prove, not absence of evidence.
    """
    if behavior_data is None:
        return True, ""

    stated = run_time_seconds(behavior_metadata)
    times = behavior_time(behavior_data)
    actual = float(times[-1]) if len(times) else None
    if stated is None or actual is None:
        return True, ""

    drift = actual - stated
    if abs(drift) > config.ALIGNMENT_DURATION_TOLERANCE_S:
        return False, (
            f"The behavior file says it ran {stated:.1f} s but its data runs to "
            f"{actual:.1f} s ({drift:+.1f} s). Alignment is anchored on "
            f"'Run Time', so every neural time is shifted by that much — the "
            f"export is probably truncated or trimmed.")
    return True, ""


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
