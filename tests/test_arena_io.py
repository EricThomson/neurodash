"""Arena calibration: picking frames, fitting the affine, and the store.

A calibration is `px = a_x*x_cm + b_x`, `py = a_y*y_cm + b_y` — deliberately
independent of the data, unlike the data-bounds fallback whose scale depends on
how much of the arena the animal happened to cover.
"""

import numpy as np
import pytest

from neurodash import arena_io


A_X, B_X, A_Y, B_Y = 10.0, 265.0, -10.0, 249.0


def synthetic_track(n=2000):
    """A wandering path that reaches all four corners of a box."""
    rng = np.random.default_rng(0)
    angle = np.linspace(0, 8 * np.pi, n)
    x = 12 * np.cos(angle) + rng.normal(0, 0.2, n)
    y = 8 * np.sin(angle) + rng.normal(0, 0.2, n)
    return x, y


# --- fitting --------------------------------------------------------------

def test_fit_recovers_a_known_affine():
    x, y = synthetic_track()
    points = [(cx, cy, A_X * cx + B_X, A_Y * cy + B_Y) for cx, cy in zip(x[:4], y[:4])]
    fit = arena_io.fit_calibration(points)

    assert fit["a_x"] == pytest.approx(A_X)
    assert fit["b_x"] == pytest.approx(B_X)
    assert fit["a_y"] == pytest.approx(A_Y)
    assert fit["b_y"] == pytest.approx(B_Y)
    assert fit["residual_px_x"] == pytest.approx(0, abs=1e-6)
    assert fit["n_points"] == 4


def test_y_flip_sign_is_derived_not_assumed():
    """EthoVision Y is up, image Y is down — the fit must find that itself."""
    x, y = synthetic_track()
    points = [(cx, cy, A_X * cx + B_X, A_Y * cy + B_Y) for cx, cy in zip(x[:4], y[:4])]
    assert arena_io.fit_calibration(points)["a_y"] < 0


def test_click_noise_shows_up_as_residual():
    """Four points against two parameters per axis leaves room to detect error."""
    rng = np.random.default_rng(1)
    x, y = synthetic_track()
    noisy = [(cx, cy, A_X * cx + B_X + rng.normal(0, 4), A_Y * cy + B_Y + rng.normal(0, 4))
             for cx, cy in zip(x[:4], y[:4])]
    fit = arena_io.fit_calibration(noisy)
    assert fit["residual_px_x"] > 0.1


@pytest.mark.parametrize("points", [
    [(1.0, 1.0, 10.0, 10.0)],                               # one point
    [(1.0, 1.0, 10.0, 10.0), (1.0, 2.0, 10.0, 20.0)],       # no spread in x
])
def test_degenerate_input_is_refused(points):
    with pytest.raises(ValueError):
        arena_io.fit_calibration(points)


def test_apply_is_the_inverse_of_the_fit():
    x, y = synthetic_track()
    calib = {"a_x": A_X, "b_x": B_X, "a_y": A_Y, "b_y": B_Y}
    px, py = arena_io.apply_calibration(x, y, calib)
    assert np.allclose(px, A_X * x + B_X)
    assert np.allclose(py, A_Y * y + B_Y)


# --- picking the frames to click -----------------------------------------

def test_corner_candidates_are_actually_diagonal_extremes():
    x, y = synthetic_track()
    by_label = dict(arena_io.corner_candidates(x, y, min_separation=5))
    assert set(by_label) == {"bottom-left", "bottom-right", "top-left", "top-right"}

    first = {label: candidates[0] for label, candidates in by_label.items()}
    assert x[first["bottom-left"]] < x[first["bottom-right"]]
    assert x[first["top-left"]] < x[first["top-right"]]
    assert y[first["bottom-left"]] < y[first["top-left"]]


def test_candidates_for_one_corner_are_different_visits():
    """'Skip frame' must move to another visit, not the next frame of the same one."""
    x, y = synthetic_track()
    for _label, candidates in arena_io.corner_candidates(x, y, min_separation=50):
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                assert abs(candidates[i] - candidates[j]) >= 50


def test_no_finite_positions_is_refused():
    nan = np.full(10, np.nan)
    with pytest.raises(ValueError):
        arena_io.corner_candidates(nan, nan)


# --- the store ------------------------------------------------------------

def test_arena_round_trip(isolated_state):
    calib = {"a_x": A_X, "b_x": B_X, "a_y": A_Y, "b_y": B_Y,
             "residual_px_x": 0.5, "residual_px_y": 1.2, "n_points": 4}
    slug = arena_io.save_arena("Rat Box #2", calib, 640, 480)
    assert slug == "rat-box-2"

    loaded = arena_io.load_arena(slug)
    assert loaded["name"] == "Rat Box #2"
    assert loaded["a_x"] == pytest.approx(A_X)
    assert [a["slug"] for a in arena_io.list_arenas()] == [slug]


def test_frame_size_guard(isolated_state):
    """A 640x480 calibration on a 720p video is meaningless, and looks plausible."""
    calib = {"a_x": A_X, "b_x": B_X, "a_y": A_Y, "b_y": B_Y,
             "residual_px_x": 0.0, "residual_px_y": 0.0, "n_points": 4}
    arena = arena_io.load_arena(arena_io.save_arena("box", calib, 640, 480))
    assert arena_io.frame_size_matches(arena, 640, 480)
    assert not arena_io.frame_size_matches(arena, 1280, 720)


def test_missing_arena_is_none_not_an_error(isolated_state):
    assert arena_io.load_arena("nope") is None
    assert arena_io.list_arenas() == []


def test_last_arena_forgets_one_that_was_deleted(isolated_state):
    calib = {"a_x": A_X, "b_x": B_X, "a_y": A_Y, "b_y": B_Y,
             "residual_px_x": 0.0, "residual_px_y": 0.0, "n_points": 4}
    slug = arena_io.save_arena("box", calib, 640, 480)
    arena_io.remember_arena(slug)
    assert arena_io.last_arena_slug() == slug

    arena_io.arena_json_path(slug).unlink()
    assert arena_io.last_arena_slug() is None


def test_snapshot_applies_brightness_and_shrinks(isolated_state):
    """A raw frame from these dark videos saves as a near-black rectangle."""
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    name = arena_io.save_arena_image("box", frame, levels=(0, 80))
    assert name == "box.png"

    import imageio.v3 as iio
    thumb = iio.imread(arena_io.arena_image_path("box"))
    assert thumb.shape[1] == 480                # 1920 // 480 -> stride 4
    assert thumb.max() > 100                    # levels applied, not raw
