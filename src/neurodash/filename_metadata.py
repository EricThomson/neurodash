"""What a recording's FILENAME says about it: which animals, which session type.

The filename is a real data source in this project and a distinct one. The .pl2
header does not name the animal, and on the fear rig the behavior export often
names the BOX where another names the animal, so for several sessions the only
statement of who was recorded is the name someone typed when saving the file:

    acquisition G16-1 and G20-3.pl2     two animals, acquisition session
    tone G16-1 and G20-3.pl2            the same two, tone test
    context G16-1 and G20-3.pl2         the same two, context test
    170505 open field theta FC33-4.pl2  one animal, no session type stated

It is also the WEAKEST source, and everything here is a starting guess that a
better source overrides. `Session.animal_id` states the precedence: the behavior
file's own header wins, these fall back.

This module is deliberately pure - stdlib only, no I/O, no imports from the rest
of the package - so it can be read and tested as what it is: string rules about
names. Storage-side normalization (`channel_io.canonical_id`) stays out of here,
because it also applies to values a human typed into the sidebar, which have
nothing to do with filenames.
"""

import re
from pathlib import Path


# The lab's Session.phase, plus the spellings seen in the wild. "aquisition" is
# not a typo on our side: the fear rig's own exports are named that way
# ("G16-1 Raw Aquisition.csv"), so a parser that insists on the correct spelling
# silently fails on real files.
SESSION_TYPE_ALIASES = {
    "acquisition": ("acquisition", "aquisition", "acq"),
    "tone": ("tone",),
    "context": ("context", "ctx"),
}


def _tokens(path):
    """A filename stem's tokens, splitting on whitespace *or* underscores.

    Both conventions are in use - '170505_open_field_theta_FC33-4' and
    '170505 open field theta FC33-4' - and neither is more correct.
    """
    stem = Path(path).stem.strip() if path else ""
    return [token for token in re.split(r"[\s_]+", stem) if token]


def _fold(text):
    """The form two names are compared in: case and punctuation flattened.

    `-` and `_` are folded together because the same animal is written both ways
    across one session's files: the raw FreezeFrame CSV names it `G20_3` where the
    .pl2 filename and the 1-s xlsx both write `G20-3`. A literal comparison misses
    that, which is how the two copies of this rule that preceded this module came
    to disagree - one folded punctuation and the other did not.

    Folding is for COMPARISON ONLY and never for storage: which separator an
    animal's name uses is a human's decision, not ours to rewrite.
    """
    return str(text if text is not None else "").strip().lower().replace("_", "-")


def parse_animal_id(pl2_path):
    """Best-effort animal ID from a pl2 filename: the last token.

    Single-animal only. '170505 open field theta FC33-4' -> 'FC33-4'. Returns ''
    when no path is given.

    Only a starting guess: the sidebar's Animal field can be corrected when a
    filename doesn't end in the animal token. Renaming the file used to be the
    only fix - see the filename-handling note in CLAUDE.md.
    """
    tokens = _tokens(pl2_path)
    return tokens[-1] if tokens else ""


def parse_animal_ids(pl2_path):
    """Every animal named in a pl2 filename, in order.

    Two-animal recordings are named "acquisition G16-1 and G20-3", so the stem is
    split on " and " and each part contributes its last token - the same rule
    `parse_animal_id` uses for a single animal.

    **The order is assumed to match the channel-bank order** (first animal named
    -> lowest-numbered headstage bank), which is what lets the app say "G20-3 is
    FP17-FP21" instead of making you work it out. That is a lab convention, not
    something any file states: it agrees with the 1-s xlsx putting G20-3 in
    `Box 2`, and it is being confirmed with NIH. Until then it seeds a default the
    user can override, and the behavior file still outranks it - see
    `Session.animal_id`.

    Returns [] when no path is given.
    """
    stem = Path(pl2_path).stem.strip() if pl2_path else ""
    if not stem:
        return []
    parts = re.split(r"\s+and\s+", stem, flags=re.IGNORECASE)
    names = []
    for part in parts:
        tokens = [t for t in re.split(r"[\s_]+", part.strip()) if t]
        if tokens:
            names.append(tokens[-1])
    return names


def parse_session_type(pl2_path):
    """Which of the lab's three session types this recording is, or ''.

    **This is the only thing that can tell a tone session from an acquisition
    session.** Their .pl2 files are identical in structure - same EVT01 x5 tones,
    same EVT02 x5 forty seconds later, same 222 s ITI, same duration - because a
    tone test runs the same program in a different chamber to see what the animal
    has learned. Nothing inside either file distinguishes them, so the name is it.

    Open field names no session type and correctly returns ''.

    Matched against every token rather than just the first, so 'G16-1 tone' works
    as well as 'tone G16-1'. A false positive would need an animal named after a
    session type.
    """
    tokens = {token.lower() for token in _tokens(pl2_path)}
    for session_type, aliases in SESSION_TYPE_ALIASES.items():
        if tokens.intersection(aliases):
            return session_type
    return ""


def same_animal(one, other):
    """Whether two written names refer to the same animal (`G20_3` == `G20-3`)."""
    folded = _fold(one)
    return bool(folded) and folded == _fold(other)


def filename_names_animal(path, name):
    """Whether `path`'s filename mentions `name`.

    A substring test rather than a token match: behavior files are named
    `G16-1 Raw Aquisition.csv`, so the animal is in there but not as the last
    token, and the candidates being tested come from the .pl2 rather than from
    carving a token out of this name.
    """
    if not path or not name:
        return False
    return _fold(name) in _fold(Path(path).stem)


def names_another_animal(pl2_path, bank_index, name):
    """True when `name` is this recording's OTHER animal.

    The behavior file's own header normally settles the animal, and on a
    single-animal file it should. On a two-animal .pl2 it can also be the wrong
    file for the channel group selected, and then it is worse than no answer: it
    labels one animal's ephys with its neighbour's name, silently. So the header
    still wins, but not against the recording itself.
    """
    names = parse_animal_ids(pl2_path)
    if bank_index is None or bank_index >= len(names):
        return False
    mine = _fold(names[bank_index])
    others = {_fold(n) for i, n in enumerate(names) if i != bank_index}
    return _fold(name) in others - {mine}


def animal_for_bank(pl2_path, behavior_path, bank_index):
    """Name the animal on this channel bank, when the behavior file's contents don't.

    Some FreezeFrame exports name the BOX on the row where others name the animal
    ("Box: Box 1"), so the usual source is simply absent.

    The candidate is `parse_animal_ids(pl2)[bank]` - the .pl2 filename lists its
    animals in bank order. That alone is only a naming habit, so the name is
    returned ONLY if the BEHAVIOR FILENAME corroborates it: those are usually named
    for their animal (`G16-1 Raw Aquisition.csv`), and it is matched against the
    names the .pl2 already offers rather than by carving a token out of the
    filename, so there is no rule about which token holds the ID and no chance of
    inventing a name that appears nowhere.

    **The stated box is deliberately NOT used.** It was, on the strength of
    Box N -> bank N being confirmed for one acquisition day - but a box is where
    the animal sat, and a bank is which headstage it wore, and nothing stops
    headstage 1 going into box 2. Treating the two as interchangeable made this
    both blank a CORRECT name (the box veto) and corroborate a wrong one, on any
    session cabled differently. The division that matters here is the headstage,
    which the channel numbering records directly.

    A behavior file naming the OTHER animal still vetoes the answer, because a
    behavior file paired with the wrong channel group is exactly the mix-up the
    bank filtering exists to prevent - one field over.
    """
    if not pl2_path or bank_index is None:
        return ""
    names = parse_animal_ids(pl2_path)
    if bank_index >= len(names):
        return ""

    others = [n for i, n in enumerate(names) if i != bank_index]
    if any(filename_names_animal(behavior_path, other) for other in others):
        return ""                       # behavior file names another animal
    candidate = names[bank_index]
    return candidate if filename_names_animal(behavior_path, candidate) else ""
