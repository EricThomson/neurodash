"""Session type, and the one thing it decides: whether a shock was delivered.

A tone session's .pl2 is structurally identical to an acquisition one - same
EVT01 x5 tones, same EVT02 x5 forty seconds later, same 222 s ITI, same duration -
because a tone test runs the same program in a different chamber to see what the
animal learned. Nothing inside either file distinguishes them, so the filename
does, and this is the app's one deliberate exception to being data-driven.

The consequence that matters: a tone session's `shock` column must read zero. If
that were left to the No shock checkbox it would depend on somebody remembering
to tick a control that is about something else entirely - a control ANIMAL within
an acquisition session, not a session type that shocks nobody.
"""

import numpy as np
import pytest

from neurodash.channel_io import resolve_session_name
from neurodash.session import Session

TONES = np.array([159.933, 381.940, 603.945, 825.952, 1047.958])
SHOCKS = TONES + 39.994


class FakeSession(Session):
    """A Session with trial events but no .pl2 behind it."""

    def __init__(self, pl2_name):
        self.pl2_path = f"C:/data/{pl2_name}.pl2"
        self.behavior_path = ""
        self.events = {}
        self.segment = None
        self.behavior_metadata = {}
        self.behavior_data = None
        self.block = None
        self.analog_signal_summaries = []
        self.lfp_signal_index = 0
        self.bank_index = 0
        self.epoch_params = None
        self.no_shock = False

    @property
    def trial_events(self):
        return {"tones": TONES, "shocks": SHOCKS, "trace_s": 20.0}


def kinds(session):
    counts = {}
    for span in session.event_spans(None):
        counts[span["kind"]] = counts.get(span["kind"], 0) + 1
    return counts


# --- reading the type off the filename -------------------------------------

@pytest.mark.parametrize("pl2_name, expected", [
    ("acquisition G16-1 and G20-3", "acquisition"),
    ("tone G16-1 and G20-3", "tone"),
    ("context G16-1 and G20-3", "context"),
    ("170505 open field theta FC33-4", ""),
])
def test_session_type(pl2_name, expected):
    assert FakeSession(pl2_name).session_type == expected


# --- what it decides -------------------------------------------------------

def test_a_tone_session_delivers_no_shock_to_anyone():
    """Its EVT02 fires on acquisition's schedule; no animal is shocked."""
    session = FakeSession("tone G16-1 and G20-3")
    assert session.shock_delivered is False
    assert kinds(session) == {"tone_event": 5}


def test_an_acquisition_session_is_shocked_by_default():
    session = FakeSession("acquisition G16-1 and G20-3")
    assert session.shock_delivered is True
    assert kinds(session) == {"tone_event": 5, "shock_event": 5}


def test_the_no_shock_control_still_works_within_acquisition():
    """The checkbox marks a control ANIMAL, and that is a separate question."""
    session = FakeSession("acquisition G16-1 and G20-3")
    session.no_shock = True
    assert session.shock_delivered is False
    assert kinds(session) == {"tone_event": 5}


def test_the_checkbox_cannot_make_a_tone_session_shocked():
    """The two reasons must not be conflated: neither can override the other."""
    session = FakeSession("tone G16-1 and G20-3")
    session.no_shock = True
    assert session.shock_delivered is False
    session.no_shock = False
    assert session.shock_delivered is False, (
        "a tone session is unshocked because of what it IS, not because of a flag")


def test_tones_are_never_suppressed():
    """Only the shock is in question - the tone is the point of the session."""
    for name in ("tone G16-1 and G20-3", "acquisition G16-1 and G20-3"):
        assert kinds(FakeSession(name)).get("tone_event") == 5


# --- seeding the session label ---------------------------------------------

def test_the_filename_seeds_the_session_label():
    """The fear rig's second exporter states no Trial key at all."""
    assert resolve_session_name({}, "C:/d/tone G16-1 and G20-3.pl2") == "tone"
    assert resolve_session_name(None, "C:/d/context G16-1.pl2") == "context"


def test_a_stated_session_still_wins_over_the_filename():
    """Open field names no type, but more importantly the file's own field rules."""
    assert resolve_session_name({"Session": "hab 1"},
                                "C:/d/context G16-1.pl2") == "hab1"


def test_blank_when_nothing_names_a_session():
    """FC33-4 has no Session field and its filename names no type."""
    assert resolve_session_name({}, "C:/d/170505 open field theta FC33-4.pl2") == ""
    assert resolve_session_name(None, None) == ""
