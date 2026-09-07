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

# Colour limits for the spectrogram, as percentiles of the dB values on screen.
# Autoscaling to min/max lets one artifact set the scale for the whole session:
# a noise event on FP17 at 668.6 s reaches 33.8 dB above the session median, and
# the shock artifacts do the same, flattening everything else into the bottom of
# the colormap. Clipping to p1-p99 costs 2% of bins and compresses the range from
# 58.3 dB to 22.3 dB — 2.6x more colour resolution on the signal you came to read.
#
# Not acquisition-specific: open field gains 2.0x by the same measure, and both
# land at ~22 dB, which is a decent sign that's the real dynamic range of these
# recordings and the rest is outliers.
SPECT_COLOR_PERCENTILES = (1.0, 99.0)

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

# Epoch windows for a trace-conditioning session. Full rationale in
# sandbox/acquisition/epoch_windows.md; defaults from wt_learning_streamlined.ipynb
# and wt_acq_learning.ipynb.
#
# Two kinds of parameter, deliberately grouped that way in the sidebar:
#
#   TIMING HEDGES (baseline_pad, tone_pad, trace_pad) guard against clock slop.
#   They were chosen for 1 Hz behavior; with 30 Hz behavior, 0.1 s bins and exact
#   TTLs they may now be unnecessary. Tuning these is housekeeping.
#
#   DESIGN WINDOWS (baseline_start, post_shock_delay, isi_duration) are claims
#   about the animal — how long it takes to settle, and how long the UR lasts
#   ("that's how long it takes the freakout to stop"). Tuning these changes what
#   is being measured, so it is a scientific decision, not cleanup.
#
# The ISI window is delay + duration ON PURPOSE. The notebooks wrote it as
# `delay + record_duration - trailing_buffer`, which subtracts the buffer from a
# length that already excludes it; that silently produced 90 s windows where
# 120 s was intended and reached the saved results (isi_window_bug.md).
ACQUISITION_EPOCH_PARAMS = {
    "baseline_start":   10.0,   # skip the settling-in period
    "baseline_pad":      1.0,   # stop short of the first tone
    "tone_duration":    TONE_DURATION_S,
    "tone_pad":          0.25,  # inset both ends of the tone
    "trace_pad":         0.25,  # inset both ends of the trace
    "shock_duration":   SHOCK_DURATION_S,
    "post_shock_delay": 30.0,   # time for the UR to end
    "isi_duration":    120.0,   # length of the ISI measurement window
}

# Epoch/event band colours follow utilities.overlay_epochs so figures stay
# comparable with the notebooks. Low opacity because these tile most of the
# session and full-saturation fills would bury the LFP; the gaps between bands
# are the guard periods and are meant to read as gaps.
# Top of the Motion Index axis, as a percentile of the session. Lower than the
# 99.9th the open-field panels use for velocity, because Motion Index is far more
# spiky: median 79 against shock startles of 7810 on the test file, so p99.9
# still leaves ordinary behavior in the bottom fifth of the panel. p99 clips ~1%.
MOTION_YMAX_PERCENTILE = 99.0

DEFAULT_SHOW_EPOCHS = True

# Raw TTL onsets as black lines. A diagnostic rather than a display: the epoch and
# event bands are built from the TTLs *plus* assumed durations and an inferred
# clock offset, so they can look plausible while resting on a wrong assumption.
# The pulses themselves carry none of that — they are what the file says, on the
# behavior clock — which makes them the thing to check an artifact against. On by
# default while acquisition support is being built, since the alignment is the
# thing most worth watching; turn it off once the bands are trusted.
DEFAULT_SHOW_TTL_PULSES = True
TTL_LINE_COLOR = "black"
# THE single source of truth for epoch colors, read by the Dash bands, the
# navigator strip, and anything else that draws trial structure. Hex for the
# same reason as EVENT_SERIES_COLORS below. Values unchanged from the CSS names
# they replace (orangered, cornflowerblue, fuchsia, lime).
EPOCH_COLORS = {"baseline": "#ff4500", "tone": "#6495ed",
                "trace": "#ff00ff", "isi": "#00ff00"}
EPOCH_BAND_OPACITY = 0.10

# The stimuli themselves, drawn over the epochs. The shock has no epoch of its
# own (it falls in the guard gap between trace and isi), so without this it would
# be invisible.
# Bands are for EPOCHS ONLY. Tone and shock get their own 0/1 time series
# instead: a stimulus is a thing that is on or off, and a trace says that
# plainly, where a band over the LFP just tints the signal you are trying to
# read. (The tone band was also redundant with the tone epoch band, which covers
# the same span to within 0.25 s at each edge.)
#
# THE single source of truth for tone/shock colors, read by the Dash events
# panel, the pyqtdash TTL lines and events trace, and the session navigator.
# Hex rather than CSS names because every consumer parses hex: plotly directly,
# pyqtgraph via mkPen. A tone/shock color hardcoded anywhere else is a bug (DRY).
EVENT_SERIES_COLORS = {"tone": "#5a8cff", "shock": "#ff00ff"}

# Alignment self-check. With no behavior-onset pulse anywhere, the neural/behavior
# offset is anchored on the END of both recordings, so anything that changes the
# behavior file's duration shifts everything silently. The check re-derives the
# answer from the animal's startle: the largest motion spike near each shock TTL.
# Tolerance is generous because the residual is the startle *response* lagging the
# TTL (0.10-0.60 s measured, plus 33 ms frame binning), not clock error — this is
# meant to catch gross failure, not to measure latency.
# The startle window, relative to the shock TTL. Asymmetric on purpose: the
# response *follows* the shock, so looking backward only invites false matches on
# ordinary movement. It spans the 2 s shock plus margin. A symmetric +/-1.5 s
# window was used while the behavior clock was mis-anchored, and it stopped
# fitting once the response moved to its true position (+0.17 to +2.00 s).
ALIGNMENT_WINDOW_S = (-0.3, 2.6)
ALIGNMENT_TOLERANCE_S = 1.5   # legacy symmetric width, still used for reporting
# How big a Motion Index value has to be, as a percentile of the session, to count
# as a startle. The five shock responses on the test file are 4798-7810 against a
# 99.9th percentile of 3623 and a median of 79, so they clear this comfortably
# while ordinary movement does not.
#
# 99.9 rather than 99.5 because of the false-match rate. The test window is 3 s
# wide (+/- the tolerance) = ~90 frames at 30 fps, so the chance that ordinary
# motion clears the bar somewhere in it is 1-(1-p)^90: about 36% at the 99.5th
# percentile but only ~9% at the 99.9th. That matters for the no-shock control
# animals, which have no startle at all and would otherwise match one or two
# shocks by luck and be reported as misaligned.
ALIGNMENT_SPIKE_PERCENTILE = 99.9

# Below this many matching shocks the result is ambiguous rather than a failure:
# too few to distinguish "this animal was never shocked" from "the alignment is
# wrong". Above it, startles demonstrably exist, so the ones that miss are real.
ALIGNMENT_MIN_CONFIDENT_MATCHES = 2

# The behavior file's stated 'Run Time' must match how far its data actually runs.
# This is the direct test for the failure end-anchoring is exposed to — a trimmed
# or truncated export — and unlike the startle test it is unambiguous, so it also
# works for the no-shock control animals. The real file differs by 0.067 s (its
# last sample is 1291.0 against a stated 1290.933), so 1 s is plenty of slack.
ALIGNMENT_DURATION_TOLERANCE_S = 1.0

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
# Dots mark each time bin on the subsampled traces (theta channels and binned
# behavior). Off by default: they are a "which bin is this?" tool, and on a full
# session they crowd the line they are meant to annotate. The Markers section in
# the right sidebar turns them on.
DEFAULT_SHOW_MARKER_DOTS = False

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
