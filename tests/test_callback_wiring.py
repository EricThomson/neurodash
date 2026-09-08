"""The Dash wiring, not the functions behind it.

0.0.9 shipped with `_describe` accidentally carrying the `@callback` decorator, so
`browse_merge_folder` was never registered and its button 500'd for every user —
while every check passed, because they called the underlying functions directly.
These tests exist to catch that class of bug.
"""

import ast
from pathlib import Path

import pytest

from neurodash import callbacks, channel_io
from neurodash.app import create_app
from neurodash.config import ACQUISITION_EPOCH_PARAMS
from neurodash.layout import epoch_input_id

CALLBACKS_SRC = Path(callbacks.__file__)


def decorator_args(function_name):
    """The @callback(...) arguments for one callback, as source strings."""
    tree = ast.parse(CALLBACKS_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "callback":
                    return [ast.unparse(a).replace("'", '"') for a in dec.args]
    raise AssertionError(f"{function_name} carries no @callback")


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
    "render_epoch_controls",
    "update_figure",
    "collect_epoch_params",
    "render_navigator",
])
def test_expected_callback_is_registered(name):
    assert name in callback_function_names()


def test_every_epoch_param_has_a_control():
    """A config key with no sidebar control fires no callback and says nothing.

    `collect_epoch_params` derives its Inputs from ACQUISITION_EPOCH_PARAMS, and
    `suppress_callback_exceptions=True` (required, since the sidebar is built
    after file load) means an Input naming a component that doesn't exist fails
    silently: the parameter would simply never leave its default, with no error.
    """
    layout = create_app().layout
    present = {c.id for c in layout._traverse() if getattr(c, "id", None)}
    missing = [key for key in callbacks.EPOCH_PARAM_KEYS
               if epoch_input_id(key) not in present]
    assert not missing, f"epoch params with no control in the sidebar: {missing}"


def test_epoch_params_store_exists():
    """The one hop every epoch consumer depends on."""
    layout = create_app().layout
    assert "store-epoch-params" in {c.id for c in layout._traverse()
                                    if getattr(c, "id", None)}


def test_collect_epoch_params_falls_back_per_key():
    """A cleared number box sends None; the store must still carry a usable set."""
    values = [None] * len(callbacks.EPOCH_PARAM_KEYS)
    assert callbacks.collect_epoch_params(*values) == dict(ACQUISITION_EPOCH_PARAMS)

    values[0] = 99.0
    result = callbacks.collect_epoch_params(*values)
    assert result[callbacks.EPOCH_PARAM_KEYS[0]] == 99.0
    assert result[callbacks.EPOCH_PARAM_KEYS[1]] == \
        ACQUISITION_EPOCH_PARAMS[callbacks.EPOCH_PARAM_KEYS[1]]


def test_browse_merge_folder_runs(tmp_path, monkeypatch, write_export):
    """Invoke it the way the button does, rather than calling helpers."""
    write_export(tmp_path, "AAA_hab1_analysis.csv", animal="AAA", session="hab1")
    write_export(tmp_path, "AAA_hab2_analysis.csv", animal="AAA", session="hab2")
    monkeypatch.setattr(callbacks, "pick_directory", lambda title, start: str(tmp_path))
    monkeypatch.setattr(callbacks, "remember_export_dir", lambda path: None)

    folder, options, value, status = callbacks.browse_merge_folder(1)

    assert str(tmp_path) in str(folder)
    assert len(options) == 2 and len(value) == 2
    assert not status
    # both sessions named, so a merged input can't masquerade as its first row
    assert {"hab1", "hab2"} <= {s for o in options for s in o["label"].split()}


def test_browse_merge_folder_cancelled(monkeypatch):
    monkeypatch.setattr(callbacks, "pick_directory", lambda title, start: "")
    assert callbacks.browse_merge_folder(1)[3] == ""


def test_merge_picker_opens_where_exports_went(tmp_path, monkeypatch, write_export):
    """Its inputs are exports, so it must not open at the raw-data folder."""
    write_export(tmp_path, "AAA_hab1_analysis.csv", animal="AAA", session="hab1")
    monkeypatch.setattr(callbacks, "remember_export_dir", lambda path: None)
    monkeypatch.setattr(callbacks, "last_export_dir", lambda: str(tmp_path))
    monkeypatch.setattr(callbacks, "last_browse_dir",
                        lambda: "/raw/data/should/not/be/used")

    seen = {}
    def fake_pick(title, start):
        seen["start"] = start
        return str(tmp_path)
    monkeypatch.setattr(callbacks, "pick_directory", fake_pick)

    callbacks.browse_merge_folder(1)
    assert seen["start"] == str(tmp_path)


def test_identity_edit_button_toggles():
    """Session locks; Animal never does — it is a dropdown of known names."""
    assert callbacks.toggle_identity_edit(0, "hab1") == (False, True, "Edit")
    assert callbacks.toggle_identity_edit(1, "hab1") == (False, False, "Done")
    assert callbacks.toggle_identity_edit(2, "hab1") == (False, True, "Edit")


def test_a_blank_session_is_editable_without_hunting_for_edit():
    assert callbacks.toggle_identity_edit(0, "") == (False, False, "Edit")


# --- animal and channels are chosen independently -------------------------
# Which animal sits on which headstage is in the lab's notes, not in any file.
# So the app lists what it can read — the animals named in the filenames, and
# the channel groups in the .pl2 — and the user pairs them. Nothing infers one
# from the other.

def test_animal_options_come_from_the_pl2_which_names_both():
    options = callbacks._animal_options(
        "/data/acquisition G16-1 and G20-3.pl2",
        "/data/G16-1 Raw Aquisition.csv", "")
    assert [o["value"] for o in options] == ["G16-1", "G20-3"]


def test_behavior_filename_is_used_when_there_is_no_pl2():
    options = callbacks._animal_options("", "/data/FC33-4.xlsx", "")
    assert [o["value"] for o in options] == ["FC33-4"]


def test_animal_options_from_the_pl2_alone():
    """No behavior file loaded: both animals must still be offered."""
    options = callbacks._animal_options(
        "/data/acquisition G16-1 and G20-3.pl2", "", "")
    assert [o["value"] for o in options] == ["G16-1", "G20-3"]


def test_a_set_animal_stays_in_its_own_option_list():
    """A name from a behavior header may appear in no filename."""
    options = callbacks._animal_options("/data/rec.pl2", "", "G20_3")
    assert "G20_3" in [o["value"] for o in options]


def test_channel_options_do_not_name_the_animal():
    """Naming it there would imply the app knows the headstage mapping."""
    bank = {"label": "FP01-FP05", "indices": [0, 1, 2, 3, 4]}
    label = callbacks._bank_option_label(["G16-1", "G20-3"], 0, bank)
    assert "G16-1" not in label
    assert "Box 1" in label and "FP01-FP05" in label


def test_choosing_a_group_does_not_name_the_animal(tmp_path):
    """Decoupled: the channel group says nothing about who the animal is."""
    pl2 = tmp_path / "acquisition G16-1 and G20-3.pl2"
    pl2.write_bytes(b"")
    callbacks.choose_bank(0, str(pl2))
    assert channel_io.load_identity(pl2, bank=0)["animal"] == ""


def test_a_chosen_animal_survives_switching_groups(tmp_path):
    pl2 = tmp_path / "acquisition G16-1 and G20-3.pl2"
    pl2.write_bytes(b"")
    channel_io.save_identity(pl2, "G16-1", "acq", bank=0)
    callbacks.choose_bank(1, str(pl2))
    callbacks.choose_bank(0, str(pl2))
    assert channel_io.load_identity(pl2, bank=0)["animal"] == "G16-1"


def test_fill_identity_reacts_to_the_channel_group():
    """Without store-bank as an Input the field kept the previous animal."""
    args = decorator_args("fill_identity")
    assert any(a.startswith('Input("store-bank"') for a in args), (
        "fill_identity must re-read the animal when the channel group changes")
