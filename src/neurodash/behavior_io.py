"""
behavior_io.py — EthoVision behavioral data loading for neurodash.

Loads EthoVision Excel exports and provides position/motion extraction
and neural sync utilities.

EthoVision Excel format:
  - Row 0: header row count (n_header)
  - Rows 1..n_header-3: key/value metadata pairs
  - Row n_header-2: column names
  - Row n_header-1: units
  - Row n_header onward: tracking data (~30 Hz)

Key columns: Recording time, X/Y center/nose/tail (cm), Velocity (cm/s),
Direction (deg), Distance moved (cm).
"""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_behavior_file(path):
    """Load an EthoVision Excel export.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    metadata : dict
        Key/value pairs from the header block.
    data : pd.DataFrame
        Tracking data, one row per frame. All numeric where possible;
        non-numeric EthoVision markers become np.nan.
        data.attrs['units'] holds a {column: unit_string} dict.
    """
    raw = pd.read_excel(path, header=None)
    n_header = int(raw.iloc[0, 1])

    metadata = {}
    for i in range(1, n_header - 2):
        key = str(raw.iloc[i, 0]).strip()
        val = raw.iloc[i, 1]
        if key and key != 'nan':
            metadata[key] = val if not _is_blank(val) else None

    col_names = raw.iloc[n_header - 2].tolist()
    units = raw.iloc[n_header - 1].tolist()

    data = raw.iloc[n_header:].copy()
    data.columns = col_names
    data = data.reset_index(drop=True)
    data.attrs['units'] = {col: str(u) for col, u in zip(col_names, units)}

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    return metadata, data


def _is_blank(val):
    if val is None:
        return True
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return str(val).strip() == ''


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def get_recording_delay(metadata):
    """Parse 'Recording after' delay from EthoVision metadata.

    Returns the number of seconds between trial start and when video
    recording began. Used to compute frame_offset = round(delay * fps).

    Parameters
    ----------
    metadata : dict
        Output of load_behavior_file.

    Returns
    -------
    delay_seconds : float
    """
    recording_after = str(metadata.get('Recording after', '+ 00:00:00.000'))
    time_part = recording_after.strip().lstrip('+ ').strip()
    h, m, s = time_part.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def get_display_metadata(metadata):
    """Extract the subset of metadata fields useful for sidebar display.

    Parameters
    ----------
    metadata : dict
        Output of load_behavior_file.

    Returns
    -------
    dict with keys:
        mouse_id, experiment, trial_name, start_time, recording_duration,
        arena_name, arena_id, video_stem,
        missed_samples_pct, subject_not_found_pct, interpolated_pct
    """
    def _duration_str(val):
        """Strip leading '+ ' from EthoVision duration strings."""
        if val is None:
            return None
        return str(val).strip().lstrip('+ ').strip()

    def _pct_float(val):
        """Parse '0.0 %' → 0.0."""
        if val is None:
            return None
        m = re.search(r'[\d.]+', str(val))
        return float(m.group()) if m else None

    video_path = metadata.get('Video file')
    video_filename = Path(video_path).name if video_path else None

    return {
        'mouse_id':              metadata.get('mouse ID'),
        'experiment':            metadata.get('Experiment'),
        'trial_name':            metadata.get('Trial name'),
        'start_time':            metadata.get('Start time'),
        'recording_duration':    _duration_str(metadata.get('Recording duration')),
        'arena_name':            metadata.get('Arena name'),
        'arena_id':              metadata.get('Arena ID'),
        'video_filename':        video_filename,
        'missed_samples_pct':    _pct_float(metadata.get('Missed samples')),
        'subject_not_found_pct': _pct_float(metadata.get('Subject not found')),
        'interpolated_pct':      _pct_float(metadata.get('Interpolated samples')),
    }


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

def extract_position(data, point='center'):
    """Extract x/y position time series for a tracked body point.

    Parameters
    ----------
    data : pd.DataFrame
        Output of load_behavior_file.
    point : {'center', 'nose', 'tail'}

    Returns
    -------
    t : np.ndarray — Recording time (s)
    x : np.ndarray — X position (cm)
    y : np.ndarray — Y position (cm)
    """
    point_map = {
        'center': ('X center', 'Y center'),
        'nose':   ('X nose',   'Y nose'),
        'tail':   ('X tail',   'Y tail'),
    }
    if point not in point_map:
        raise ValueError(f"point must be one of {list(point_map)}, got '{point}'")
    x_col, y_col = point_map[point]
    t = data['Recording time'].to_numpy(dtype=float)
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    return t, x, y


def estimate_position_pixels(x, y, video_width, video_height,
                             scale_x=1.0, scale_y=1.0,
                             offset_x=0.0, offset_y=0.0,
                             padding=0.1):
    """Map EthoVision cm coordinates to video pixel coordinates.

    Uses data bounds + padding as a proxy for arena extent (uncalibrated).
    Pass scale_x/y, offset_x/y from a saved arena calibration to refine.

    Y is flipped: EthoVision Y increases upward, image Y increases downward.

    Parameters
    ----------
    x, y : np.ndarray — position in cm
    video_width, video_height : int — frame dimensions in pixels
    scale_x, scale_y : float — calibration scale factors
    offset_x, offset_y : float — calibration pixel offsets
    padding : float — fractional padding around data bounds

    Returns
    -------
    px, py : np.ndarray — pixel coordinates
    """
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    x_pad = (x_max - x_min) * padding
    y_pad = (y_max - y_min) * padding
    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad

    px = ((x - x_min) / (x_max - x_min) * video_width) * scale_x + offset_x
    py = ((1.0 - (y - y_min) / (y_max - y_min)) * video_height) * scale_y + offset_y
    return px, py


# ---------------------------------------------------------------------------
# Behavior notes — a free-text comment on the whole recording, persisted as a
# JSON sidecar (<stem>.behavior.json) next to the Excel file, like the channel
# sidecar. Reloads on the next open.
# ---------------------------------------------------------------------------

def behavior_sidecar_path(behavior_path):
    """Return the notes sidecar path: <stem>.behavior.json beside the Excel file."""
    p = Path(behavior_path)
    return p.parent / (p.stem + ".behavior.json")


def load_behavior_notes(behavior_path):
    """Load the behavior notes sidecar, or a blank record if none exists."""
    path = behavior_sidecar_path(behavior_path)
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return {"comment": data.get("comment", "")}
        except (json.JSONDecodeError, OSError):
            pass
    return {"comment": ""}


def save_behavior_notes(behavior_path, notes):
    """Write the behavior notes sidecar next to the Excel file."""
    payload = {
        "comment": notes.get("comment", ""),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(behavior_sidecar_path(behavior_path), "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

def compute_motion(data, point='center'):
    """Compute frame-to-frame displacement and speed from tracked position.

    Re-derives from x/y coordinates rather than EthoVision's 'Distance moved'
    column, so the calculation is transparent and reproducible.

    Returns
    -------
    t : np.ndarray — Recording time (s)
    displacement : np.ndarray — distance moved since previous frame (cm); first value NaN
    speed : np.ndarray — instantaneous speed (cm/s); first value NaN
    """
    t, x, y = extract_position(data, point=point)
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    displacement = np.sqrt(dx**2 + dy**2)
    speed = displacement / dt
    displacement = np.concatenate([[np.nan], displacement])
    speed = np.concatenate([[np.nan], speed])
    return t, displacement, speed


# ---------------------------------------------------------------------------
# Neural sync
# ---------------------------------------------------------------------------

def get_recording_delay_from_data(data):
    """Return the offset between trial start and behavioral t=0.

    Alias for get_recording_delay that works directly from the data DataFrame
    when metadata is not available separately.
    """
    # Recording time in the DataFrame starts at 0 when video recording began.
    # The delay relative to trial start is already baked into the time axis.
    return float(data['Recording time'].iloc[0])


def sync_to_neural(data, neural_t_offset=0.0):
    """Align behavioral Recording time to a shared neural time axis.

    Parameters
    ----------
    data : pd.DataFrame
    neural_t_offset : float
        Seconds to add to behavioral Recording time to align with neural time.

    Returns
    -------
    t_neural : np.ndarray
    """
    t_behavior = data['Recording time'].to_numpy(dtype=float)
    return t_behavior + neural_t_offset


def get_behavior_at_neural_times(data, query_times, column, neural_t_offset=0.0):
    """Interpolate a behavioral variable onto arbitrary neural timepoints.

    Parameters
    ----------
    data : pd.DataFrame
    query_times : array-like — neural timepoints (s)
    column : str — column name (e.g. 'Velocity', 'X center')
    neural_t_offset : float

    Returns
    -------
    values : np.ndarray — interpolated values; NaN outside behavioral range
    """
    t_neural = sync_to_neural(data, neural_t_offset=neural_t_offset)
    values = data[column].to_numpy(dtype=float)
    query_times = np.asarray(query_times, dtype=float)
    return np.interp(query_times, t_neural, values, left=np.nan, right=np.nan)
