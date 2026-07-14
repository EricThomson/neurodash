"""Central configuration and default parameters for neurodash."""


APP_TITLE = "neurodash"

# Default starting directory for file picker dialogs
DEFAULT_FILE_DIR = "C:/Users/Eric/data/fear/plexon"
LOGO_PATH = "/assets/logo/neurodash_logo_200.png"

# Default initial time window shown in the main plot (seconds).
# Full session is always computed; this just sets the starting viewport.
DEFAULT_VIEW_DURATION = 30.0

# Spectrogram defaults (opt-in)
DEFAULT_SHOW_SPECTROGRAM = False
DEFAULT_SPECT_MAX_FREQ = 90.0
DEFAULT_SPECT_WINDOW_SEC = 2.0
DEFAULT_SPECT_STEP_SEC = 0.1
DEFAULT_SPECT_C_PARAM = 20

# Channel QC
QC_QUALITY_OPTIONS = ["good", "fair", "bad"]  # per-channel quality rating
QC_ROW_HEIGHT = 160  # px per channel row in the combined QC figure
EXEMPLAR_SEEDS_DEFAULT_CHANNEL = True  # exemplar seeds the viewer's default channel

# ---------------------------------------------------------------------------
# TEMPORARY dev convenience: autoload these files on startup so you don't have
# to browse for them every run. Flip AUTOLOAD_ON_STARTUP to False to disable,
# or delete this block once neurodash is where you want it (see the autoload
# branches in callbacks.browse_neural / browse_behavior).
# ---------------------------------------------------------------------------
AUTOLOAD_ON_STARTUP = True
AUTOLOAD_NEURAL_PATH = "C:/Users/Eric/data/fear/plexon/170505_open_field_theta_FC33-4.pl2"
AUTOLOAD_BEHAVIOR_PATH = "C:/Users/Eric/data/fear/plexon/Raw data-260129_Zhenglin_openfield_pipelinepilot-Trial 5.xlsx"
