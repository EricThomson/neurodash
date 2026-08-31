"""Shared fixtures.

Everything here is synthetic on purpose. Real `.pl2` files are ~300 MB and can't
live in the repo, and almost nothing worth testing needs them: the merge rules,
identity canonicalization, callback wiring and arena fitting are all pure logic
over small inputs. That keeps the suite fast and runnable on any machine.

Tests that touch persisted state must use `isolated_state` — the real paths are
under ``~/.neurodash`` and a test that writes there would clobber the user's own
arenas and session.
"""

import numpy as np
import pandas as pd
import pytest

from neurodash import app_state, arena_io


# Header fields a valid export carries. The four analysis parameters are what
# merge.BLOCKING_FIELDS compares, so they have to match across files to merge.
DEFAULT_META = {
    "animal": "AAA",
    "session": "hab1",
    "spectrogram_channel": "AI17",
    "theta_band_hz": "4.0-12.0",
    "theta_estimator": "bandpass",
    "time_base": "spectral bins, step 0.1s (window 1.5s, c=10)",
    "theta_ratio_bands_hz": "low 6.1-7.4, high 7.5-8.8",
}

N_BINS = 20


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect persisted state at ~/.neurodash into tmp_path."""
    monkeypatch.setattr(app_state, "_LAST_SESSION_FILE", tmp_path / "last_session.json")
    monkeypatch.setattr(app_state, "_LAST_DIR_FILE", tmp_path / "last_dir.txt")
    monkeypatch.setattr(arena_io, "ARENAS_DIR", tmp_path / "arenas")
    monkeypatch.setattr(arena_io, "_LAST_ARENA_FILE", tmp_path / "last_arena.txt")
    return tmp_path


def analysis_frame(animal="AAA", session="hab1", channels=("AI17",), n=N_BINS):
    """A table shaped like a real `_analysis.csv`: channels as columns."""
    time = np.round(np.arange(n) * 0.1, 6)
    columns = {"animal": animal, "session": session, "time": time}
    for variable in ("theta_peak", "theta_power", "theta_ratio"):
        for i, label in enumerate(channels):
            columns[f"{variable}_{label}"] = np.linspace(1 + i, 2 + i, n)
    for behavioral in ("velocity", "mobility", "x", "y"):
        columns[behavioral] = np.linspace(0, 1, n)
    return pd.DataFrame(columns)


@pytest.fixture
def write_export():
    """Factory: write an export CSV (``#`` header + body) into a folder."""
    def _write(folder, name, animal="AAA", session="hab1", channels=("AI17",),
               frame=None, **meta_overrides):
        meta = {**DEFAULT_META, "animal": animal, "session": session,
                "spectrogram_channel": ", ".join(channels), **meta_overrides}
        df = analysis_frame(animal, session, channels) if frame is None else frame
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_text("".join(f"#{k}: {v}\n" for k, v in meta.items())
                        + df.to_csv(index=False), encoding="utf-8")
        return path
    return _write
