"""Tiny persisted UI state for neurodash.

Two things, both in ~/.neurodash, both best-effort — any I/O error falls back to
a default rather than raising:

  last_dir.txt      the folder the picker last browsed, so dialogs reopen there
  last_session.json the files last opened, so a restart reopens them

They're separate on purpose: the browsed folder also tracks the *merge* folder
picker, which has nothing to do with the loaded session.

Nothing machine-specific belongs in the repo — this is where it lives instead.
"""

import json
from pathlib import Path

from neurodash import config

_LAST_DIR_FILE = Path.home() / ".neurodash" / "last_dir.txt"
_LAST_SESSION_FILE = Path.home() / ".neurodash" / "last_session.json"


def last_browse_dir():
    """Folder the picker should open in — the last one browsed, or the config
    default when there's no valid saved folder."""
    try:
        saved = _LAST_DIR_FILE.read_text(encoding="utf-8").strip()
        if saved and Path(saved).is_dir():
            return saved
    except OSError:
        pass
    return config.DEFAULT_FILE_DIR


def remember_browse_dir(path):
    """Persist a browsed folder as the next picker default.

    Accepts either a picked *file* (stores its containing folder) or a picked
    *folder* (stores it as-is) — the merge picker returns a directory, and
    taking its parent would reopen one level too high.
    """
    try:
        path = Path(path)
        folder = path if path.is_dir() else path.parent
        _LAST_DIR_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_DIR_FILE.write_text(str(folder), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Last session — the files to reopen on startup
# ---------------------------------------------------------------------------

def last_session():
    """Files last opened, as ``{"neural": path|None, "behavior": path|None}``.

    A file that no longer exists comes back as None, so a moved or deleted
    recording just means no reopen rather than an error on startup.
    """
    try:
        saved = json.loads(_LAST_SESSION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return {key: (saved.get(key) if saved.get(key) and Path(saved[key]).is_file() else None)
            for key in ("neural", "behavior")}


def remember_session(neural=None, behavior=None):
    """Record a just-opened file. Only the arguments given are updated.

    Read-modify-write because the two files are picked by separate callbacks and
    neither knows the other's path; writing the whole record from one of them
    would clear the other.
    """
    try:
        saved = json.loads(_LAST_SESSION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    if neural:
        saved["neural"] = str(neural)
    if behavior:
        saved["behavior"] = str(behavior)
    try:
        _LAST_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_SESSION_FILE.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    except OSError:
        pass
