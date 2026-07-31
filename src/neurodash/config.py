"""Central configuration and default parameters for neurodash."""

from pathlib import Path


APP_TITLE = "neurodash"

# Default starting directory for file picker dialogs
DEFAULT_FILE_DIR = "C:/Users/Eric/data/open_field_data"
LOGO_PATH = "/assets/logo/neurodash_logo_200.png"

# CSV exports are written here (server-side), NOT the browser's download folder.
# To save alongside the raw data instead, set this to the source file's directory
# at export time.
EXPORT_DIR = Path.home() / ".neurodash" / "data"

# Default initial time window shown in the main plot (seconds).
# Full session is always computed; this just sets the starting viewport.
DEFAULT_VIEW_DURATION = 30.0

# Session Viewer figure height: pixels per unit of panel "weight"
# (spectrogram=3, LFP=2, behavioral=1), capped at the viewport. So a lone LFP
# panel (~2 units) is a top strip instead of filling the whole screen, while a
# full stack still fits the viewport without scrolling.
SESSION_PANEL_UNIT_PX = 150

# Spectrogram defaults (opt-in). Theta rides on these, so they set the theta computation
# too. window 1.5 s, c 10, and max 40 Hz come from the analysis notebook (2.0 s over-smooths
# real fluctuations); step 0.1 s is a finer hop than the notebook's 0.25 (finer time
# sampling, suited to the higher-rate behavior). STEP is the hop (time resolution = 1/step).
DEFAULT_SHOW_SPECTROGRAM = False
DEFAULT_SPECT_MAX_FREQ = 40.0
DEFAULT_SPECT_WINDOW_SEC = 1.5
DEFAULT_SPECT_STEP_SEC = 0.1
DEFAULT_SPECT_C_PARAM = 10

# Channel Viewer
CHANNEL_QUALITY_OPTIONS = ["good", "fair", "bad"]  # per-channel quality rating
CHANNEL_ROW_HEIGHT = 160  # px per channel row in the combined channel figure
EXEMPLAR_SEEDS_DEFAULT_CHANNEL = True  # exemplar seeds the viewer's default channel

# Theta analysis (Phase 1: peak frequency + band power for one channel).
# Both are derived from the same spectrogram the Session Viewer already computes,
# on the shared "analysis channel" (the spectrogram channel selector).
DEFAULT_SHOW_THETA_PEAK = False
DEFAULT_SHOW_THETA_POWER = False
DEFAULT_THETA_LOW_HZ = 4.0          # theta band lower edge
DEFAULT_THETA_HIGH_HZ = 12.0        # theta band upper edge
DEFAULT_THETA_INTERP_STEP_HZ = 0.1  # fine grid for sub-bin peak-frequency precision
DEFAULT_THETA_SMOOTH_WIDTH = 5      # Hann window (bins) smoothing the output series
                                    # (=5 per the notebook; ports 1:1 now the hop is 0.25s)
THETA_SPECT_TIME_SMOOTH_WIDTH = 5   # Hann window (bins) smoothing the spectrogram in time
                                    # before peak extraction (stabilizes the argmax)
DEFAULT_THETA_DOT_SIZE = 4          # marker size for the theta-peak overlay dots

# Theta peak estimator. "argmax" is the original (largest power in band); "crest"
# removes the 1/f background first and takes the tallest local maximum, which stops
# delta's flank pinning the estimate to the 4 Hz edge. See
# sandbox/docs/delta_contamination.md. DEFAULT_THETA_WINDOW_SEC lets the peak be
# estimated on a wider window than the spectrogram is drawn with (None = share it);
# a wider window narrows the smoothing kernel that causes the contamination.
DEFAULT_THETA_ESTIMATOR = "crest"
DEFAULT_THETA_WINDOW_SEC = 2.5

# ---------------------------------------------------------------------------
# TEMPORARY dev convenience: autoload these files on startup so you don't have
# to browse for them every run. Flip AUTOLOAD_ON_STARTUP to False to disable,
# or delete this block once neurodash is where you want it (see the autoload
# branches in callbacks.browse_neural / browse_behavior).
# ---------------------------------------------------------------------------
AUTOLOAD_ON_STARTUP = True
AUTOLOAD_NEURAL_PATH = "C:/Users/Eric/data/fear/plexon/170505_open_field_theta_FC33-4.pl2"
AUTOLOAD_BEHAVIOR_PATH = "C:/Users/Eric/data/fear/plexon/Raw data-260129_Zhenglin_openfield_pipelinepilot-Trial 5.xlsx"
