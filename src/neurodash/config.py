"""Central configuration and default parameters for neurodash."""

from pathlib import Path


APP_TITLE = "neurodash"

# Where file dialogs open before anything has been browsed — used exactly once
# per user, after which app_state.last_browse_dir takes over. Home, because it is
# the only folder guaranteed to exist on a machine that isn't the author's.
DEFAULT_FILE_DIR = str(Path.home())
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

# Which analog stream in a .pl2 holds the LFP, most-preferred first. A pl2 can
# carry several; which one is the LFP depends on the rig (the lab's schema calls
# this `setup: original or digiamp`). "FP" is Plexon's field-potential stream
# ("FPl-Low Pass Filtered") and wins wherever it exists; "AI" is auxiliary input,
# which is where the open-field rig put the LFP because it wrote no FP stream.
#
# This is not a preference so much as a correctness fix: the acquisition file has
# both, with the real LFP in FP (10 ch) and noise in AI (32 ch) at 1/10th the
# amplitude. Every caller used to assume analog signal 0, which is AI there.
LFP_STREAM_PREFERENCE = ("FP", "AI")

# Above this correlation, two channel "banks" (contiguous runs of channel numbers,
# normally one per animal in a two-animal recording) are not electrically
# independent and so are probably not separate subjects. Measured across-bank
# correlation on the acquisition test file is 0.001 mean / 0.008 max, and
# within-bank pairs run 0.28-0.99, so anything in between is a wide, safe gap.
CHANNEL_BANK_INDEPENDENCE_THRESHOLD = 0.05

# --- Fear-conditioning TTL events (acquisition / tone / context) --------------
# Which pl2 event channels carry what. Lists rather than single names so a rig
# that renumbers them needs a config edit, not a code change. Confirmed on the
# acquisition test file: EVT01 = 5 tone onsets, EVT02 = 5 shocks (each 39.994 s
# after its tone), EVT03 = a lone stop marker at the end of the behavior recording.
EVENT_TONE_CHANNELS = ("EVT01",)
EVENT_SHOCK_CHANNELS = ("EVT02",)
EVENT_STOP_CHANNELS = ("EVT03",)

# Tone and shock DURATIONS are protocol constants, not file facts — no pl2 event
# channel carries a duration, and the source notebook hardcodes `tones + 20` and
# `shocks + 2`. The lab's own `Events` table stores them as explicit fields, so
# they are treated the same way here: defaults that a session can override.
# The trace interval is NOT here: it is measured per session as shock onset minus
# tone offset (20.0 s on the test file). Confirm with NIH whether these ever vary.
TONE_DURATION_S = 20.0
SHOCK_DURATION_S = 2.0

# Alignment self-check. With no behavior-onset pulse anywhere, the neural/behavior
# offset is anchored on the END of both recordings, so anything that changes the
# behavior file's duration shifts everything silently. The check re-derives the
# answer from the animal's startle: the largest motion spike near each shock TTL.
# Tolerance is generous because the residual is the startle *response* lagging the
# TTL (0.10-0.60 s measured, plus 33 ms frame binning), not clock error — this is
# meant to catch gross failure, not to measure latency.
ALIGNMENT_TOLERANCE_S = 1.5
# How big a Motion Index value has to be, as a percentile of the session, to count
# as a startle. The five shock responses on the test file are 4798-7810 against a
# 99.9th percentile of 3623 and a median of 79, so they clear this comfortably
# while ordinary movement does not. Chosen empirically: a shift of one frame
# already fails, and the intact file passes with all five.
ALIGNMENT_SPIKE_PERCENTILE = 99.5

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

# Theta ratio — (low - high) / (low + high) of mean power in two sub-bands, so
# positive = slow theta dominates. Bands are fixed/manual here; deriving them from
# the movement-conditioned peak distribution (KDE) is separate, deferred work.
# Defaults are James Ward's bands, per the source notebook. They are sub-bands of
# the theta band, and must stay inside it: under the "bandpass" estimator the whole
# spectrogram is of LFP filtered to theta band +/- the margin, and a ratio band
# reaching outside that sits on the filter rolloff. Inside, this is a non-issue —
# measured power gain is 1.0000 at all four default edges and the ratio off the
# filtered vs unfiltered spectrogram matches at r = 1.0000 (sandbox/probe_ratio_bandpass.py),
# so the ratio rides on the same spectrogram peak and power already use.
DEFAULT_SHOW_THETA_RATIO = False
DEFAULT_THETA_RATIO_LOW_BAND = (6.1, 7.4)    # James low
DEFAULT_THETA_RATIO_HIGH_BAND = (7.5, 8.8)   # James high

# Theta peak estimator. Two options, see sandbox/docs/delta_contamination.md:
#   "argmax"   original — largest power in the band. Delta's flank pins it to the
#              4 Hz edge on 2-15% of bins depending on channel.
#   "bandpass" filter the LFP to the band +/- a margin, then argmax. DEFAULT.
#              Simplest, measured marginally best, and trivial to describe in a
#              methods section. Destructive: whatever the filter removes is gone
#              before any estimator sees it, so the margin must keep the edges
#              outside the real theta range (FC33-4 AI18 has genuine 4.1 Hz theta).
# A third "crest" estimator (whiten out 1/f, take the strongest local maximum) was
# built, measured and removed — same answers to within 0.03-0.15 Hz, far harder to
# explain. Kept for the record in sandbox/docs/delta_contamination.md.
DEFAULT_THETA_ESTIMATOR = "bandpass"

# "bandpass" estimator: bandpass the LFP just outside the theta band, then plain
# argmax. Measured slightly better on pinning than the alternatives, but it is
# DESTRUCTIVE — filter edges inside the real theta range silently mangle the signal
# and nothing downstream can detect it. FC33-4 AI18 has genuine theta at 4.1 Hz, so
# a 4.0 Hz edge would clip it; hence the margin, which keeps the filter outside
# whatever band is set rather than at a fixed frequency.
DEFAULT_THETA_BANDPASS_MARGIN_HZ = 0.5
DEFAULT_THETA_BANDPASS_ORDER = 8

# Reopen the files you had loaded when the app last ran (app_state.last_session).
# This replaces a block of hardcoded author paths behind an AUTOLOAD flag: the
# point was never those particular files, it was not re-picking two files on every
# restart — and with debug=False forcing a relaunch for every code change, that
# adds up. Generalizing it makes it a feature rather than a dev hack, and takes
# the last machine-specific paths out of the repo.
#
# Files that have moved or been deleted are simply not reopened.
REOPEN_LAST_SESSION = False
