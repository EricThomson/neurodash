"""Central session object and file-loading machinery for neurodash.

A Session holds all loaded data for one recording — neural, behavioral, or both.
Files are loaded directly from filesystem paths — no copying.

Caching uses functools.lru_cache (process-level). The Dash app is a standard
Python process so module-level lru_cache works fine.
"""

from functools import lru_cache
from pathlib import Path

from neurodash.neural_io import load_pl2_block, list_analog_signal_summaries, get_analog_signal, extract_time_window
from neurodash.spectral_utils import compute_multitaper_spectrogram
from neurodash.behavior_io import load_behavior_file


class Session:
    """Container for one recording session.

    At least one of neural or behavioral data must be loaded.
    Check has_neural / has_behavior before accessing those attributes.
    """

    def __init__(self,
                 pl2_path=None, block=None, channel_names=None,
                 behavior_path=None, behavior_metadata=None, behavior_data=None):
        # Neural
        self.pl2_path = pl2_path
        self.block = block
        self.analog_signal_summaries = (
            list_analog_signal_summaries(block, channel_names) if block else []
        )
        self.rec_datetime = (
            block.annotations.get("m_CreatorDateTime", None) if block else None
        )

        # Behavioral
        self.behavior_path = behavior_path
        self.behavior_metadata = behavior_metadata
        self.behavior_data = behavior_data

    @property
    def has_neural(self):
        return self.block is not None

    @property
    def has_behavior(self):
        return self.behavior_data is not None


# ---------------------------------------------------------------------------
# Cached loaders — load directly from path, no copying
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _cached_load_block(pl2_path_str):
    block, channel_names = load_pl2_block(Path(pl2_path_str))
    return block, channel_names


@lru_cache(maxsize=4)
def _cached_load_behavior(behavior_path_str):
    metadata, data = load_behavior_file(Path(behavior_path_str))
    return metadata, data


@lru_cache(maxsize=8)
def compute_spectrogram(
    pl2_path_str,
    analog_signal_index,
    channel_index,
    start_time_sec,
    duration_sec,
    max_frequency,
    window_duration,
    step_duration,
    c_parameter,
):
    """Compute and cache spectrogram by pl2_path + parameters.

    Keys on pl2_path_str for hashability (lru_cache requirement).
    """
    block, _ = _cached_load_block(pl2_path_str)
    sig = get_analog_signal(block, analog_signal_index)
    t, y, sr = extract_time_window(sig, channel_index, start_time_sec, duration_sec)
    freqs, times, power_db = compute_multitaper_spectrogram(
        y,
        sampling_rate=sr,
        window_duration=window_duration,
        step_duration=step_duration,
        c_parameter=c_parameter,
        max_frequency=max_frequency,
    )
    return freqs, times, power_db


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

def load_session_from_paths(neural_path_str, behavior_path_str):
    """Load a Session directly from filesystem paths. No file copying.

    Parameters
    ----------
    neural_path_str : str — path to .pl2 file, or empty string / None
    behavior_path_str : str — path to .xlsx file, or empty string / None

    Returns
    -------
    Session
    """
    block = channel_names = pl2_path = None
    if neural_path_str:
        pl2_path = Path(neural_path_str)
        block, channel_names = _cached_load_block(str(pl2_path))

    behavior_metadata = behavior_data = behavior_path = None
    if behavior_path_str:
        behavior_path = Path(behavior_path_str)
        behavior_metadata, behavior_data = _cached_load_behavior(str(behavior_path))

    return Session(
        pl2_path=pl2_path,
        block=block,
        channel_names=channel_names,
        behavior_path=behavior_path,
        behavior_metadata=behavior_metadata,
        behavior_data=behavior_data,
    )
