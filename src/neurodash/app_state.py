"""Tiny persisted UI state for neurodash.

Currently just the last folder the file picker browsed, so it reopens where you
left off instead of a hardcoded default (this replaces the retired startup
autoload). Deliberately minimal — one line in ~/.neurodash, best-effort: any I/O
error just falls back to the configured default.
"""

from pathlib import Path

from neurodash import config

_LAST_DIR_FILE = Path.home() / ".neurodash" / "last_dir.txt"


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


def remember_browse_dir(file_path):
    """Persist the folder containing ``file_path`` as the next picker default."""
    try:
        _LAST_DIR_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_DIR_FILE.write_text(str(Path(file_path).parent), encoding="utf-8")
    except OSError:
        pass
