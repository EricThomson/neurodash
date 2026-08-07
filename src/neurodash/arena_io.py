"""arena_io.py — per-arena spatial calibration (EthoVision cm -> video pixels).

A calibration is four numbers, a direct affine per axis:

    px = a_x * x_cm + b_x
    py = a_y * y_cm + b_y        (a_y normally negative — EthoVision Y is up,
                                  image Y is down; the fit derives the sign)

That is deliberately NOT what `behavior_io.estimate_position_pixels` does. That
function stretches the *data bounds* (min/max of where the animal went) across
the frame, so its scale is a property of the animal's exploration rather than of
the arena: across five trials in one rat box the tracked X extent ranged 21.75 to
28.04 cm, so the same saved scale is ~29% wrong from one animal to the next. A
mapping you can save per arena has to be independent of the data, hence the
affine above. The data-bounds version stays as the uncalibrated fallback.

Calibration is a property of the arena AND the camera geometry (mount, zoom,
aim) — bump the tripod and a saved arena is stale. `frame_w`/`frame_h` and the
per-axis residuals are stored so that goes noticed rather than silent.

Layout, one JSON + one background image per arena:

    ~/.neurodash/arenas/<slug>.json
    ~/.neurodash/arenas/<slug>.png
    ~/.neurodash/last_arena.txt
"""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

ARENAS_DIR = Path.home() / ".neurodash" / "arenas"
_LAST_ARENA_FILE = Path.home() / ".neurodash" / "last_arena.txt"

# The four diagonal extremes of the tracked cloud, as (label, x weight, y weight)
# to maximize. Diagonal corners rather than per-axis min/max on purpose: each
# corner then contributes near-full spread to BOTH the x and the y fit, whereas
# argmin_y/argmax_y frames sit mid-range in x and barely constrain the x fit.
# Labels describe the arena as seen in the video and are only a hint to the user
# — the fit derives the signs, so a mirrored or rotated camera still calibrates.
CORNERS = (
    ("bottom-left", -1, -1),
    ("bottom-right", +1, -1),
    ("top-right", +1, +1),
    ("top-left", -1, +1),
)


# ---------------------------------------------------------------------------
# Picking frames to click
# ---------------------------------------------------------------------------

def _neighbour_jump(x, y, idx, neighbourhood):
    """Largest cm distance from sample idx to its temporal neighbours.

    Small means the sample is continuous with the frames around it — real
    movement. Large means an isolated jump, i.e. tracking briefly lost the
    animal, and the video frame will not show it where the cm says it is.
    """
    lo, hi = max(0, idx - neighbourhood), min(len(x), idx + neighbourhood + 1)
    return float(np.nanmax(np.hypot(x[lo:hi] - x[idx], y[lo:hi] - y[idx])))


def corner_candidates(x_cm, y_cm, n_per_corner=4, top_fraction=0.005,
                      min_separation=60, neighbourhood=3):
    """Frames to ask the user to click, as [(label, [index, ...]), ...].

    For each arena corner, take the samples furthest out in that diagonal
    direction and rank them by temporal continuity, so an outlier from a
    momentary tracking dropout loses to a real excursion. Taking the top
    `top_fraction` and tie-breaking costs almost no lever arm (on the test data
    the chosen X moved from 16.48 to 16.43 cm) and is cheap insurance.

    Candidates for one corner are forced `min_separation` samples apart, so
    "skip this frame" gets you a different visit to that corner rather than the
    neighbouring frame of the same one.
    """
    x = np.asarray(x_cm, dtype=float)
    y = np.asarray(y_cm, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError("no finite tracked positions to calibrate from")

    out = []
    for label, wx, wy in CORNERS:
        proj = np.where(finite, wx * x + wy * y, -np.inf)
        thresh = np.percentile(proj[finite], 100.0 * (1.0 - top_fraction))
        cands = np.flatnonzero(proj >= thresh)
        cands = cands[np.argsort([_neighbour_jump(x, y, int(i), neighbourhood) for i in cands])]

        picked = []
        for i in cands:
            if all(abs(int(i) - j) >= min_separation for j in picked):
                picked.append(int(i))
            if len(picked) >= n_per_corner:
                break
        out.append((label, picked or [int(cands[0])]))
    return out


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_calibration(points):
    """Least-squares affine from clicked points.

    points : sequence of (x_cm, y_cm, px, py)

    Returns a dict of a_x/b_x/a_y/b_y plus the RMS residual per axis. Per axis,
    not pooled: the tracked Y extent is often about half the X extent, so the
    same click noise buys roughly half the precision in Y and one combined
    number would hide that.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("need at least 2 points to fit a calibration")
    x_cm, y_cm, px, py = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    if np.ptp(x_cm) < 1e-6 or np.ptp(y_cm) < 1e-6:
        raise ValueError("clicked points span no distance in cm — pick opposite corners")

    a_x, b_x = np.polyfit(x_cm, px, 1)
    a_y, b_y = np.polyfit(y_cm, py, 1)
    return {
        "a_x": float(a_x), "b_x": float(b_x),
        "a_y": float(a_y), "b_y": float(b_y),
        "residual_px_x": float(np.sqrt(np.mean((px - (a_x * x_cm + b_x)) ** 2))),
        "residual_px_y": float(np.sqrt(np.mean((py - (a_y * y_cm + b_y)) ** 2))),
        "n_points": int(len(pts)),
    }


def apply_calibration(x_cm, y_cm, arena):
    """Map cm to pixels through a saved arena. Returns (px, py)."""
    px = arena["a_x"] * np.asarray(x_cm, dtype=float) + arena["b_x"]
    py = arena["a_y"] * np.asarray(y_cm, dtype=float) + arena["b_y"]
    return px, py


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def slugify(name):
    """'Rat Box #2' -> 'rat-box-2', so the name is also a safe filename."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    return slug or "arena"


def arena_json_path(slug):
    return ARENAS_DIR / f"{slug}.json"


def arena_image_path(slug):
    return ARENAS_DIR / f"{slug}.png"


def save_arena_image(slug, frame, levels=None, target_width=480):
    """Write <slug>.png — a snapshot for recognizing the arena. Returns the filename.

    One ordinary frame, animal and all: enough to answer "is this my box?", which
    is all it's for. `levels` is the (lo, hi) display range the viewer is using —
    pass it, because these recordings are dark enough that a raw frame saves as a
    near-black rectangle that identifies nothing.

    Shrinking is a plain integer stride (no filtering, no extra dependency), so the
    result lands in [target_width, 2*target_width): 640 px stays 640 (~100 KB),
    1920 goes to exactly 480. Big enough to recognize is the goal, not exact.

    Best-effort: returns None if the image can't be written, since a missing
    thumbnail must never cost you the calibration.
    """
    try:
        import imageio.v3 as iio

        img = np.asarray(frame, dtype=float)
        lo, hi = levels if levels else (float(img.min()), float(img.max()))
        img = np.clip((img - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)

        step = max(1, img.shape[1] // target_width)
        img = img[::step, ::step]

        ARENAS_DIR.mkdir(parents=True, exist_ok=True)
        iio.imwrite(arena_image_path(slug), img)
        return f"{slug}.png"
    except Exception:
        return None


def save_arena(name, calib, frame_w, frame_h, image=None, note=""):
    """Write <slug>.json. `image` is a filename in the same folder, or None.

    Returns the slug. Overwrites an existing arena of the same name — the
    caller is responsible for asking first.
    """
    slug = slugify(name)
    record = {
        "name": str(name).strip(),
        "slug": slug,
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
        "image": image,
        "note": note,
        "created": datetime.now().isoformat(timespec="seconds"),
        **calib,
    }
    ARENAS_DIR.mkdir(parents=True, exist_ok=True)
    arena_json_path(slug).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return slug


def load_arena(slug):
    """Read one arena, or None if it's missing or unreadable."""
    try:
        return json.loads(arena_json_path(slug).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_arenas():
    """Every readable arena, by name. Unreadable files are skipped, not raised."""
    if not ARENAS_DIR.is_dir():
        return []
    arenas = []
    for path in sorted(ARENAS_DIR.glob("*.json")):
        try:
            arenas.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(arenas, key=lambda a: a.get("name", "").lower())


def frame_size_matches(arena, frame_w, frame_h):
    """Was this arena calibrated at the current video's resolution?

    A calibration made at 640x480 applied to a 720p video is meaningless, and
    that is the one failure mode nothing on screen would otherwise reveal.
    """
    return (int(arena.get("frame_w", -1)) == int(frame_w)
            and int(arena.get("frame_h", -1)) == int(frame_h))


# --- last used, so the viewer reopens on the arena you were working in -----

def last_arena_slug():
    try:
        slug = _LAST_ARENA_FILE.read_text(encoding="utf-8").strip()
        return slug if slug and arena_json_path(slug).is_file() else None
    except OSError:
        return None


def remember_arena(slug):
    try:
        _LAST_ARENA_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_ARENA_FILE.write_text(slug or "", encoding="utf-8")
    except OSError:
        pass
