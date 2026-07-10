"""Central configuration and default parameters for neurodash3."""


APP_TITLE = "neurodash3"

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
