"""The Dash wiring, not the functions behind it.

0.0.9 shipped with `_describe` accidentally carrying the `@callback` decorator, so
`browse_merge_folder` was never registered and its button 500'd for every user —
while every check passed, because they called the underlying functions directly.
These tests exist to catch that class of bug.
"""

import ast
from pathlib import Path

import pytest

from neurodash import callbacks
from neurodash.app import create_app

CALLBACKS_SRC = Path(callbacks.__file__)


def callback_function_names():
    """Names of every function actually carrying an @callback decorator."""
    tree = ast.parse(CALLBACKS_SRC.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if getattr(target, "id", None) == "callback":
                names.append(node.name)
    return names


def test_app_builds():
    create_app()


def test_no_private_helper_is_decorated():
    """A helper inserted between @callback and its function silently steals it."""
    stolen = [n for n in callback_function_names() if n.startswith("_")]
    assert not stolen, (
        f"{stolen} carries @callback — a helper was inserted between the decorator "
        f"and the function it was meant to decorate")


@pytest.mark.parametrize("name", [
    "browse_neural",
    "browse_behavior",
    "render_neural_metadata",
    "fill_identity",
    "browse_merge_folder",
    "run_merge",
    "export_analysis_csv",
    "populate_channel_pickers",
    "choose_bank",
])
def test_expected_callback_is_registered(name):
    assert name in callback_function_names()


def test_browse_merge_folder_runs(tmp_path, monkeypatch, write_export):
    """Invoke it the way the button does, rather than calling helpers."""
    write_export(tmp_path, "AAA_hab1_analysis.csv", animal="AAA", session="hab1")
    write_export(tmp_path, "AAA_hab2_analysis.csv", animal="AAA", session="hab2")
    monkeypatch.setattr(callbacks, "pick_directory", lambda title, start: str(tmp_path))
    monkeypatch.setattr(callbacks, "remember_browse_dir", lambda path: None)

    folder, options, value, status = callbacks.browse_merge_folder(1)

    assert str(tmp_path) in str(folder)
    assert len(options) == 2 and len(value) == 2
    assert not status
    # both sessions named, so a merged input can't masquerade as its first row
    assert {"hab1", "hab2"} <= {s for o in options for s in o["label"].split()}


def test_browse_merge_folder_cancelled(monkeypatch):
    monkeypatch.setattr(callbacks, "pick_directory", lambda title, start: "")
    assert callbacks.browse_merge_folder(1)[3] == ""


def test_identity_edit_button_toggles():
    assert callbacks.toggle_identity_edit(0) == (True, True, "Edit")
    assert callbacks.toggle_identity_edit(1) == (False, False, "Done")
    assert callbacks.toggle_identity_edit(2) == (True, True, "Edit")
