"""Persisted UI state — the last folder browsed and the files to reopen.

All best-effort by design: a missing or corrupt file must degrade to a default
rather than raise, because this runs during app startup.
"""

from neurodash import app_state, config


def test_no_saved_session(isolated_state):
    assert app_state.last_session() == {"neural": None, "behavior": None}


def test_the_two_files_are_recorded_independently(isolated_state, tmp_path):
    """Separate callbacks pick them, so neither may clobber the other."""
    neural = tmp_path / "rec.pl2"
    behavior = tmp_path / "rec.xlsx"
    neural.touch()
    behavior.touch()

    app_state.remember_session(neural=neural)
    assert app_state.last_session() == {"neural": str(neural), "behavior": None}

    app_state.remember_session(behavior=behavior)
    assert app_state.last_session() == {"neural": str(neural), "behavior": str(behavior)}


def test_a_file_that_moved_is_not_reopened(isolated_state, tmp_path):
    behavior = tmp_path / "rec.xlsx"
    behavior.touch()
    app_state.remember_session(neural=tmp_path / "gone.pl2", behavior=behavior)

    saved = app_state.last_session()
    assert saved["neural"] is None
    assert saved["behavior"] == str(behavior)   # the other entry survives


def test_corrupt_state_does_not_break_startup(isolated_state):
    app_state._LAST_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    app_state._LAST_SESSION_FILE.write_text("{not json", encoding="utf-8")
    assert app_state.last_session() == {"neural": None, "behavior": None}


def test_browse_dir_round_trip(isolated_state, tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    app_state.remember_browse_dir(target / "file.pl2")     # a file: stores its folder
    assert app_state.last_browse_dir() == str(target)

    app_state.remember_browse_dir(target)                  # a folder: stores it as-is
    assert app_state.last_browse_dir() == str(target)


def test_browse_dir_falls_back_to_a_folder_that_exists(isolated_state):
    """The default is used once per user, so it must exist on any machine."""
    from pathlib import Path
    assert app_state.last_browse_dir() == config.DEFAULT_FILE_DIR
    assert Path(config.DEFAULT_FILE_DIR).is_dir()


def test_no_author_paths_are_shipped():
    """config used to hardcode the author's data folder and two of his files."""
    from pathlib import Path
    src = Path(config.__file__).read_text(encoding="utf-8").lower()
    assert "users/eric" not in src and "users\\eric" not in src
