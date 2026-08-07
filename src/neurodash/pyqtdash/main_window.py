"""
main_window.py — neurodash video viewer.

Receives a handoff directory from neurodash containing pre-serialized arrays
and metadata. Displays synchronized video, behavioral traces, LFP, and
spectrogram. No data loading or computation — just display.

Handoff directory contents:
    handoff.json        — metadata and display settings
    lfp.npz             — pre-normalized LFP traces (if has_neural)
    spectrogram.npz     — pre-computed spectrogram (if show_spectrogram)
    theta.npz           — pre-computed theta peak frequency (if show_theta_peak)
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import av
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6.QtWidgets import (
    QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QSlider, QLabel, QCheckBox, QSplitter,
    QDoubleSpinBox, QPushButton, QComboBox, QInputDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QIcon

from neurodash import arena_io
from neurodash.plot_utils import velocity_ylim
from neurodash.behavior_io import (
    load_behavior_file, get_recording_delay, estimate_position_pixels
)

# Spatial calibration comes from ~/.neurodash/arenas via the Arena pulldown; see
# arena_io. The old hardcoded _SCALE_X/_OFFSET_X constants were removed: they were
# fitted against one session's *data bounds*, so they were a property of where
# FC33-4 happened to walk, not of the arena, and applying them to any other
# session was actively wrong. With no arena selected we now fall back to the
# plain data-bounds stretch, labelled "uncalibrated" in the UI.

# Max frames to decode-and-discard forward before preferring a seek. Keeps 2×
# playback and short forward scrubs on the cheap sequential decode path.
_MAX_SEQ_SKIP = 15

# Brightness slider half-range, in units where 50 = one doubling of gain
# (+50 -> 2× brighter, -50 -> half). ±100 is ±2 stops, which covers the dark
# recordings without letting the control run off into uselessness.
_BRIGHTNESS_RANGE = 100


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_behavior(path):
    """Load EthoVision Excel file. Returns (data DataFrame, recording_delay_s)."""
    metadata, data = load_behavior_file(path)
    recording_delay_s = get_recording_delay(metadata)
    return data, recording_delay_s


def _estimate_position_pixels(x, y, video_width, video_height):
    """Uncalibrated fallback: stretch the tracked data bounds across the frame.

    Only as good as the assumption that the animal visited the whole arena, which
    is why it can't be saved per arena — use the Arena pulldown for a real one.
    """
    return estimate_position_pixels(x, y, video_width, video_height)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class NeurodashViewer(QMainWindow):
    def __init__(self, handoff_dir: str):
        super().__init__()
        self.setWindowTitle("neurodash viewer")
        self._handoff_dir = handoff_dir
        self._calib = None   # in-progress calibration state, None when not calibrating
        self._arena = None   # the applied arena record, None when uncalibrated

        # Load handoff metadata
        with open(Path(handoff_dir) / "handoff.json") as f:
            handoff = json.load(f)

        video_path = handoff.get("video_path") or None
        behavior_path = handoff.get("behavior_path")
        has_neural = handoff.get("has_neural", False)
        show_spectrogram = handoff.get("show_spectrogram", False)
        spectrogram_channel_name = handoff.get("spectrogram_channel_name")
        # Theta peak is an overlay on the spectrogram, so it never shows without it.
        show_theta_peak = handoff.get("show_theta_peak", False) and show_spectrogram
        theta_peak_color = handoff.get("theta_peak_color", "black")
        theta_band = handoff.get("theta_band")
        t_start = handoff.get("t_start", 0.0)
        window_duration = handoff.get("window_duration", 30.0)

        has_video = bool(video_path)
        has_behavior = behavior_path is not None

        # --- Load behavioral data ---
        if has_behavior:
            print("Loading behavioral data...")
            behavior, recording_delay_s = _load_behavior(behavior_path)
            self.t_behav = behavior["Recording time"].to_numpy(dtype=float)
            self.x_cm = behavior["X center"].to_numpy(dtype=float)
            self.y_cm = behavior["Y center"].to_numpy(dtype=float)
            # Velocity, raw at the EthoVision rate. The viewer deliberately shows only
            # the raw trace — the raw-vs-subsampled comparison is a QC question that
            # lives in the Dash Session Viewer, not something you ask while scrubbing video.
            self.velocity = behavior["Velocity"].to_numpy(dtype=float)
        else:
            recording_delay_s = 0.0

        # --- Load pre-serialized neural data ---
        if has_neural:
            print("Loading LFP from handoff...")
            lfp_data = np.load(Path(handoff_dir) / "lfp.npz", allow_pickle=True)
            self.t_neural = lfp_data["t"]
            self.lfp_traces = lfp_data["traces"]  # already normalized
            self.ch_names = list(lfp_data["channel_names"])
            self.lfp_y_range = tuple(lfp_data["y_range"])

            if show_spectrogram:
                print("Loading spectrogram from handoff...")
                spec_data = np.load(Path(handoff_dir) / "spectrogram.npz")
                self.freqs = spec_data["freqs"]
                self.spec_times = spec_data["times"]
                self.power_db = spec_data["power_db"]

            if show_theta_peak:
                print("Loading theta peak from handoff...")
                theta_data = np.load(Path(handoff_dir) / "theta.npz")
                self.theta_times = theta_data["times"]
                self.theta_peak_hz = theta_data["peak_hz"]
        else:
            self.t_neural = self.t_behav if has_behavior else np.array([0.0])

        # --- Video metadata ---
        self.video_path = video_path
        if has_video:
            av_meta = iio.immeta(video_path, plugin="pyav")
            self.fps = av_meta.get("fps", 30.0)
            duration_s = av_meta.get("duration", None)
            total_frames = int(duration_s * self.fps) if duration_s else (len(self.t_behav) if has_behavior else 0)
            self.frame_offset = round(recording_delay_s * self.fps)
            self.n_frames = min(total_frames - self.frame_offset, len(self.t_behav) if has_behavior else total_frames)

            # Persistent decoder — open once and decode sequentially during
            # playback. The old reopen-and-seek-per-frame path capped playback
            # at ~7 fps; sequential decode restores real-time speed.
            self._container = av.open(video_path)
            self._video_stream = self._container.streams.video[0]
            self._video_stream.thread_type = "AUTO"
            self._decode_gen = None      # live sequential decode generator
            self._decode_next_idx = -1   # frame index the generator yields next
            self._base_levels = None     # (lo, hi) latched from the first frame
            self._img_levels = None      # those levels after the brightness gain
            self._last_frame = None      # last decoded frame, to repaint on a levels change
        else:
            self.fps = 30.0
            self.frame_offset = 0
            self.n_frames = len(self.t_behav) if has_behavior else (len(self.t_neural) if has_neural else 1)
            self._container = None

        if has_video:
            # Frame size is needed to reset the crop, so latch it whenever
            # there's video; the cm->pixel map only makes sense with behavior.
            first_frame = self._read_video_frame(self.frame_offset)
            self.frame_h, self.frame_w = first_frame.shape[:2]
            if has_behavior:
                self.px, self.py = _estimate_position_pixels(self.x_cm, self.y_cm, self.frame_w, self.frame_h)

        # --- Build UI ---
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(self.main_splitter)

        # Top: video | behavioral traces
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.top_splitter)

        if has_video:
            # Video pane: the image with its own image controls directly beneath.
            # Brightness belongs here rather than in the transport row below —
            # it acts on the picture, not on time.
            video_panel = QWidget()
            video_layout = QVBoxLayout(video_panel)
            video_layout.setContentsMargins(0, 0, 0, 0)
            video_layout.setSpacing(2)

            self.image_view = pg.ImageView()
            self.image_view.ui.roiBtn.hide()
            self.image_view.ui.menuBtn.hide()
            self.image_view.ui.histogram.hide()
            video_layout.addWidget(self.image_view)

            image_controls = QHBoxLayout()
            image_controls.setContentsMargins(0, 0, 0, 0)

            # Arena calibration. Only meaningful with behavior loaded — there's
            # nothing to map without tracked positions — so the controls simply
            # aren't built otherwise rather than sitting there greyed out.
            if has_behavior:
                image_controls.addWidget(QLabel("Arena:"))
                self.arena_combo = QComboBox()
                self.arena_combo.setToolTip("Which saved cm-to-pixel calibration to use.")
                self.arena_combo.setIconSize(QSize(64, 48))
                self.arena_combo.currentIndexChanged.connect(self.on_arena_changed)
                image_controls.addWidget(self.arena_combo)

                self.calibrate_button = QPushButton("Calibrate")
                self.calibrate_button.setToolTip(
                    "Click the animal in four auto-chosen frames to calibrate this arena.")
                self.calibrate_button.clicked.connect(self.on_calibrate_clicked)
                image_controls.addWidget(self.calibrate_button)

                self.arena_status = QLabel("")
                self.arena_status.setStyleSheet("color: #999;")
                image_controls.addWidget(self.arena_status)
            else:
                self.arena_combo = None

            image_controls.addSpacing(12)
            image_controls.addWidget(QLabel("Brightness:"))
            self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
            self.brightness_slider.setMinimum(-_BRIGHTNESS_RANGE)
            self.brightness_slider.setMaximum(_BRIGHTNESS_RANGE)
            self.brightness_slider.setValue(0)
            # Ticks at -range / 0 / +range, so the centre one marks the default
            # (the levels the video came in at) and you can find your way back.
            self.brightness_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.brightness_slider.setTickInterval(_BRIGHTNESS_RANGE)
            self.brightness_slider.setToolTip(
                "Brighten or darken the video image. Display only — the data is untouched."
            )
            self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
            image_controls.addWidget(self.brightness_slider, stretch=1)

            self.crop_roi = None
            self.crop_button = QPushButton("Crop")
            self.crop_button.setToolTip("Drag a box around the region to keep, then Apply.")
            self.crop_button.clicked.connect(self.on_crop_clicked)
            image_controls.addWidget(self.crop_button)

            self.reset_crop_button = QPushButton("Reset")
            self.reset_crop_button.setToolTip("Show the whole video frame again.")
            self.reset_crop_button.clicked.connect(self.on_reset_crop)
            image_controls.addWidget(self.reset_crop_button)

            video_layout.addLayout(image_controls)

            # Calibration prompt row — hidden until Calibrate is pressed, so the
            # instructions are in front of you exactly when they apply.
            if has_behavior:
                self.calib_row = QWidget()
                calib_layout = QHBoxLayout(self.calib_row)
                calib_layout.setContentsMargins(0, 0, 0, 0)
                self.calib_status = QLabel("")
                self.calib_status.setStyleSheet("color: #ffd400; font-weight: bold;")
                calib_layout.addWidget(self.calib_status, stretch=1)
                for text, slot, tip in (
                    ("Skip frame", self.on_calib_skip, "Show a different visit to this corner."),
                    ("Undo", self.on_calib_undo, "Take back the last click."),
                    ("Cancel", self.on_calib_cancel, "Leave calibration without saving."),
                ):
                    btn = QPushButton(text)
                    btn.setToolTip(tip)
                    btn.clicked.connect(slot)
                    calib_layout.addWidget(btn)
                self.calib_row.setVisible(False)
                video_layout.addWidget(self.calib_row)

            self.top_splitter.addWidget(video_panel)
            self.dot = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None), brush=pg.mkBrush(255, 50, 50, 220))
            self.image_view.getView().addItem(self.dot)

            if has_behavior:
                # Green crosses mark the points collected so far.
                self.calib_marks = pg.ScatterPlotItem(
                    size=14, symbol="crosshair", pen=pg.mkPen("w", width=1),
                    brush=pg.mkBrush(50, 255, 50, 200))
                self.image_view.getView().addItem(self.calib_marks)
                self.image_view.getView().scene().sigMouseClicked.connect(self.on_image_click)
        else:
            self.image_view = None
            self.dot = None
            self.arena_combo = None

        if has_behavior:
            behav_panel = QWidget()
            behav_layout = QVBoxLayout(behav_panel)
            behav_layout.setSpacing(2)
            self.top_splitter.addWidget(behav_panel)
            self.top_splitter.setSizes([650, 550])

            pos_ymin = np.nanpercentile(np.concatenate([self.x_cm, self.y_cm]), 10)
            pos_ymax = np.nanpercentile(np.concatenate([self.x_cm, self.y_cm]), 90)
            _, vel_ymax = velocity_ylim(self.velocity)

            self.pos_plot = pg.PlotWidget(title="Position (cm)")
            self.pos_plot.addLegend(offset=(-1, 1), brush=pg.mkBrush(50, 50, 50, 200))
            self.pos_plot.plot(self.t_behav, self.x_cm, pen=pg.mkPen("c", width=1), name="X")
            self.pos_plot.plot(self.t_behav, self.y_cm, pen=pg.mkPen("m", width=1), name="Y")
            self.pos_plot.setYRange(pos_ymin, pos_ymax, padding=0.05)
            self.pos_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", width=2))
            self.pos_plot.addItem(self.pos_cursor)
            behav_layout.addWidget(self.pos_plot)

            self.vel_plot = pg.PlotWidget(title="Velocity (cm/s)")
            self.vel_plot.plot(self.t_behav, self.velocity, pen=pg.mkPen("g", width=1))
            self.vel_plot.setYRange(0, vel_ymax, padding=0.05)
            self.vel_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", width=2))
            self.vel_plot.addItem(self.vel_cursor)
            behav_layout.addWidget(self.vel_plot)

        # Slider controls — three stacked rows:
        #   1) time label   2) play + 2× + scrub slider   3) window + zoom
        slider_row = QWidget()
        slider_layout = QVBoxLayout(slider_row)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(4)
        root.addWidget(slider_row)

        # Row 1: time label
        self.time_label = QLabel("t = 0.00 s  |  frame 0")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        slider_layout.addWidget(self.time_label)

        # Row 2: play | 2× | scrub slider
        transport_layout = QHBoxLayout()
        transport_layout.setContentsMargins(0, 0, 0, 0)
        self.play_button = QPushButton("▶ Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.on_play_clicked)
        transport_layout.addWidget(self.play_button)
        self.speed_checkbox = QCheckBox("2×")
        self.speed_checkbox.stateChanged.connect(self.on_speed_changed)
        transport_layout.addWidget(self.speed_checkbox)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_frames - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        transport_layout.addWidget(self.slider, stretch=1)
        slider_layout.addLayout(transport_layout)

        # Row 3: window duration + zoom toggle (left-packed)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(QLabel("Window (s):"))
        self.window_spinbox = QDoubleSpinBox()
        self.window_spinbox.setMinimum(1.0)
        self.window_spinbox.setMaximum(300.0)
        self.window_spinbox.setSingleStep(5.0)
        self.window_spinbox.setValue(window_duration)
        self.window_spinbox.valueChanged.connect(self.on_window_duration_changed)
        controls_layout.addWidget(self.window_spinbox)
        self.zoom_checkbox = QCheckBox("Zoom")
        self.zoom_checkbox.setChecked(True)
        self.zoom_checkbox.stateChanged.connect(self.on_zoom_mode_changed)
        controls_layout.addWidget(self.zoom_checkbox)
        controls_layout.addStretch(1)
        slider_layout.addLayout(controls_layout)

        # Playback timer — advances one frame per tick at real-time fps.
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)
        self._play_interval_ms = max(1, round(1000 / self.fps))
        # Wall-clock playback state (initialized here, before the initial
        # slider.setValue below fires on_slider_changed). Playback derives the
        # target frame from elapsed real time * speed and drops frames when
        # rendering can't keep up, so 1× / 2× stay true to the wall clock instead
        # of sliding into slow motion.
        self._play_speed = 1.0       # 1.0 (1×) or 2.0 (2×)
        self._play_t0 = None         # perf_counter latched at play start
        self._play_start_idx = 0     # slider index latched at play start
        self._advancing = False      # our own slider writes vs. a user scrub

        # LFP
        if has_neural:
            colors = ["c", "m", "y"]
            self.lfp_plot = pg.PlotWidget(title="LFP")
            self.lfp_plot.setLabel("left", "Amplitude (mV)")
            self.lfp_plot.setLabel("bottom", "Time (s)")
            self.lfp_plot.addLegend(offset=(-1, 1), brush=pg.mkBrush(50, 50, 50, 200))
            # Render fix: clip each LFP curve to the visible window and decimate to
            # ~screen resolution instead of re-rasterizing all ~480k points/channel on
            # every frame. Profiled: cuts show_frame ~60ms -> ~30ms, so 1x plays at
            # real time. 'peak' keeps the min/max envelope so oscillations read right.
            self.lfp_plot.setClipToView(True)
            self.lfp_plot.setDownsampling(auto=True, mode="peak")

            for i, trace in enumerate(self.lfp_traces):
                name = self.ch_names[i] if i < len(self.ch_names) else f"Ch {i+1}"
                color = colors[i % len(colors)]
                self.lfp_plot.plot(self.t_neural, trace, pen=pg.mkPen(color, width=1), name=name)

            self.lfp_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", width=2))
            self.lfp_plot.addItem(self.lfp_cursor)
            self.lfp_plot.getAxis("left").setTicks([])
            self.lfp_plot.setYRange(*self.lfp_y_range, padding=0)
            self.main_splitter.addWidget(self.lfp_plot)

        # Spectrogram (only if Dash had it enabled)
        if show_spectrogram:
            spec_title = f"Spectrogram — {spectrogram_channel_name}" if spectrogram_channel_name else "Spectrogram"
            self.spec_plot = pg.PlotWidget(title=spec_title)
            self.spec_plot.setLabel("left", "Frequency (Hz)")
            self.spec_plot.setLabel("bottom", "Time (s)")

            img = pg.ImageItem()
            img.setImage(self.power_db.T)
            img.setRect(QtCore.QRectF(
                self.spec_times[0], self.freqs[0],
                self.spec_times[-1] - self.spec_times[0],
                self.freqs[-1] - self.freqs[0]
            ))
            img.setColorMap(pg.colormap.get("inferno"))
            self.spec_plot.addItem(img)

            # Theta peak overlay — no markers, riding the spectrogram's own frequency
            # axis, in whichever colour the Session Viewer is using. Faint dashed
            # guides mark the band edges the peak is searched in (as in Dash).
            # connect="finite" leaves gaps rather than drawing through NaN bins. No
            # clip/downsample here: it's one point per spectrogram bin (~5k), not the
            # LFP's ~480k, so it's already cheap.
            if show_theta_peak:
                if theta_band:
                    guide_pen = pg.mkPen(
                        color=(255, 255, 255, 128), width=1,
                        style=Qt.PenStyle.DashLine,
                    )
                    for edge in theta_band:
                        self.spec_plot.addItem(
                            pg.InfiniteLine(pos=float(edge), angle=0, pen=guide_pen)
                        )
                self.theta_curve = pg.PlotDataItem(
                    self.theta_times, self.theta_peak_hz,
                    pen=pg.mkPen(theta_peak_color, width=1), connect="finite",
                )
                self.spec_plot.addItem(self.theta_curve)

            self.spec_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("w", width=2))
            self.spec_plot.addItem(self.spec_cursor)
            self.spec_plot.setYRange(0, self.freqs[-1], padding=0)
            self.main_splitter.addWidget(self.spec_plot)

        # Set splitter sizes based on what's shown
        if has_neural and show_spectrogram:
            self.main_splitter.setSizes([500, 200, 200])
        elif has_neural:
            self.main_splitter.setSizes([500, 200])

        # Collect all time-series plots for range updates
        self._time_plots = []
        if has_behavior:
            self._time_plots += [self.pos_plot, self.vel_plot]
        if has_neural:
            self._time_plots.append(self.lfp_plot)
        if show_spectrogram:
            self._time_plots.append(self.spec_plot)

        for plot in self._time_plots:
            plot.enableAutoRange(x=False, y=False)
            plot.setXRange(0, window_duration, padding=0)

        self._has_video = has_video
        self._has_behavior = has_behavior
        self._has_neural = has_neural
        self._show_spectrogram = show_spectrogram

        # Arena calibration — reopen on the arena you last used.
        if self.arena_combo is not None:
            self._populate_arena_combo(arena_io.last_arena_slug())
            self._apply_arena(self.arena_combo.currentData())

        # Seek to t_start
        if has_behavior:
            initial_idx = int(np.searchsorted(self.t_behav, t_start))
        else:
            initial_idx = int(round(t_start * self.fps))
        initial_idx = max(0, min(initial_idx, self.n_frames - 1))
        self.slider.setValue(initial_idx)

        self.resize(1400, 900)
        print("Ready.")
        QTimer.singleShot(100, lambda: self.show_frame(initial_idx))

    def _read_video_frame(self, video_idx):
        """Return the RGB frame (H, W, 3) at video_idx.

        Fast path: if video_idx continues the current decode sequence
        (playback), pull the next decoded frame. Slow path: on a jump/scrub,
        seek to the nearest keyframe and decode forward to the exact frame.
        """
        # Fast path: continue the current decode sequence, decoding and
        # discarding forward across small gaps so 2× playback and short forward
        # scrubs stay off the seek path.
        if (self._decode_gen is not None
                and self._decode_next_idx <= video_idx <= self._decode_next_idx + _MAX_SEQ_SKIP):
            frame = None
            while self._decode_next_idx <= video_idx:
                frame = next(self._decode_gen, None)
                if frame is None:
                    break
                self._decode_next_idx += 1
            if frame is not None:
                return frame.to_ndarray(format="rgb24")

        stream = self._video_stream
        start_ts = stream.start_time or 0
        target_pts = int(round(video_idx / self.fps / stream.time_base)) + start_ts
        self._container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        self._decode_gen = self._container.decode(stream)
        for frame in self._decode_gen:
            fidx = int(round((frame.pts - start_ts) * stream.time_base * self.fps))
            if fidx >= video_idx:
                self._decode_next_idx = fidx + 1
                return frame.to_ndarray(format="rgb24")
        return None  # past end of video

    def _draw_frame(self, frame):
        self.image_view.setImage(
            frame.transpose(1, 0, 2),
            autoLevels=False, autoRange=False, levels=self._img_levels,
        )

    def _apply_brightness(self):
        """Recompute display levels from the base levels and the slider.

        Brightness is a pure gain on the white point: the black point stays put
        and the top of the range slides down, which is what a too-dark video
        needs. Levels are applied by the ImageItem at paint time, so this costs
        nothing per frame — no numpy work in the playback path.
        """
        lo, hi = self._base_levels
        span = max(hi - lo, 1.0)
        gain = 2.0 ** (self.brightness_slider.value() / 50.0)
        self._img_levels = (lo, lo + span / gain)

    def on_brightness_changed(self):
        if self._base_levels is None:
            return
        self._apply_brightness()
        # Repaint the frame we already have — decoding it again would mean a
        # keyframe seek on every tick of the slider drag.
        if self._last_frame is not None:
            self._draw_frame(self._last_frame)

    # --- Crop ---------------------------------------------------------------
    # Cropping moves the view range only; the frame array is never sliced. So
    # image pixel coordinates are unchanged and the position dot needs no
    # correction — it lands where it always did. Aspect is locked, so the
    # applied view may show slightly more than the box on one axis.

    def on_crop_clicked(self):
        """Crop -> drop the ROI box; Apply -> zoom the view to it."""
        if self.crop_roi is None:
            self._show_crop_roi()
        else:
            self._apply_crop()

    def _show_crop_roi(self):
        # Seed the box just inside what's currently on screen, so it's always
        # visible and cropping an already-cropped view keeps working.
        (x0, x1), (y0, y1) = self.image_view.getView().viewRange()
        inset_x = (x1 - x0) * 0.1
        inset_y = (y1 - y0) * 0.1
        self.crop_roi = pg.RectROI(
            [x0 + inset_x, y0 + inset_y],
            [(x1 - x0) - 2 * inset_x, (y1 - y0) - 2 * inset_y],
            pen=pg.mkPen((255, 220, 0), width=2),
        )
        self.crop_roi.addScaleHandle([0, 0], [1, 1])  # top-left grip too
        self.crop_roi.setZValue(20)
        self.image_view.getView().addItem(self.crop_roi)
        self.crop_button.setText("Apply")

    def _apply_crop(self):
        pos, size = self.crop_roi.pos(), self.crop_roi.size()
        x0, x1 = sorted([pos.x(), pos.x() + size.x()])
        y0, y1 = sorted([pos.y(), pos.y() + size.y()])
        self._clear_crop_roi()
        if x1 - x0 > 1 and y1 - y0 > 1:
            self.image_view.getView().setRange(
                xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def _clear_crop_roi(self):
        if self.crop_roi is not None:
            self.image_view.getView().removeItem(self.crop_roi)
            self.crop_roi = None
            self.crop_button.setText("Crop")

    def on_reset_crop(self):
        self._clear_crop_roi()
        self.image_view.getView().setRange(
            xRange=(0, self.frame_w), yRange=(0, self.frame_h), padding=0)

    # --- Arena calibration --------------------------------------------------
    # Maps EthoVision cm to video pixels. The four frames you click are chosen
    # for you (arena_io.corner_candidates) rather than scrubbed for: the corners
    # of the tracked cloud give the longest lever arm, and hunting for them by
    # hand is the part of calibration nobody wants to do.

    def _populate_arena_combo(self, select_slug=None):
        self.arena_combo.blockSignals(True)
        self.arena_combo.clear()
        self.arena_combo.addItem("Uncalibrated", None)
        for arena in arena_io.list_arenas():
            self.arena_combo.addItem(arena["name"], arena["slug"])
            # Thumbnail on the item, full size on hover — so picking between
            # similar-sounding arena names is a look, not a guess.
            path = arena_io.arena_image_path(arena["slug"])
            if arena.get("image") and path.is_file():
                i = self.arena_combo.count() - 1
                self.arena_combo.setItemIcon(i, QIcon(str(path)))
                self.arena_combo.setItemData(
                    i, f'<img src="{path.as_uri()}" width="320">',
                    Qt.ItemDataRole.ToolTipRole)
        idx = self.arena_combo.findData(select_slug) if select_slug else 0
        self.arena_combo.setCurrentIndex(max(idx, 0))
        self.arena_combo.blockSignals(False)

    def _apply_arena(self, slug):
        """Recompute the dot's pixel track from the selected arena."""
        self._arena = arena_io.load_arena(slug) if slug else None
        if self._arena is None:
            self.px, self.py = _estimate_position_pixels(
                self.x_cm, self.y_cm, self.frame_w, self.frame_h)
            self.arena_status.setText("rough guess — not calibrated")
        elif not arena_io.frame_size_matches(self._arena, self.frame_w, self.frame_h):
            # Applying it anyway would look plausible and be meaningless.
            self.px, self.py = _estimate_position_pixels(
                self.x_cm, self.y_cm, self.frame_w, self.frame_h)
            self.arena_status.setText(
                f"⚠ calibrated at {self._arena['frame_w']}×{self._arena['frame_h']}, "
                f"video is {self.frame_w}×{self.frame_h} — not applied")
            self._arena = None
        else:
            self.px, self.py = arena_io.apply_calibration(self.x_cm, self.y_cm, self._arena)
            self.arena_status.setText(
                f"±{self._arena['residual_px_x']:.1f} px x, "
                f"±{self._arena['residual_px_y']:.1f} px y")

    def on_arena_changed(self):
        slug = self.arena_combo.currentData()
        self._apply_arena(slug)
        arena_io.remember_arena(slug)
        self.show_frame(self.slider.value())

    def on_calibrate_clicked(self):
        try:
            # Candidates for one corner are kept ~2 s apart so "Skip frame"
            # gets you a different visit, not the next frame of the same one.
            gap = int(round(2.0 / np.median(np.diff(self.t_behav))))
            targets = arena_io.corner_candidates(
                self.x_cm, self.y_cm, min_separation=max(1, gap))
        except (ValueError, FloatingPointError) as exc:
            QMessageBox.warning(self, "Can't calibrate", str(exc))
            return

        QMessageBox.information(
            self, "Calibrate arena",
            "You'll be shown four frames — one per corner of the area the animal "
            "actually covered.\n\n"
            "In each, click the CENTER OF THE ANIMAL'S BODY. EthoVision tracks the "
            "body center, so that's the point being matched — not the nose or tail. "
            "Click the same spot every time: being consistent matters more than "
            "being exact, because a steady bias only shifts the dot, while a jumpy "
            "one shows up as fit error.\n\n"
            "If the animal is hidden or hard to make out, press Skip frame for a "
            "different visit to that corner.")

        self._clear_crop_roi()
        self._calib = {"targets": targets, "corner": 0, "cand": 0, "points": []}
        self.dot.setData([], [])            # the old mapping would bias the clicks
        self.calib_row.setVisible(True)
        self.calibrate_button.setEnabled(False)
        self.arena_combo.setEnabled(False)
        self._show_calib_target()

    def _show_calib_target(self):
        c = self._calib
        label, cands = c["targets"][c["corner"]]
        idx = int(cands[c["cand"] % len(cands)])
        if self.slider.value() == idx:
            self.show_frame(idx)            # setValue wouldn't fire; draw it directly
        else:
            self.slider.setValue(idx)
        self.calib_status.setText(
            f"Click the center of the animal's body  —  {label} corner  "
            f"({c['corner'] + 1} of {len(c['targets'])})")

    def _calib_current_index(self):
        c = self._calib
        _, cands = c["targets"][c["corner"]]
        return cands[c["cand"] % len(cands)]

    def on_image_click(self, event):
        if self._calib is None:
            return
        p = self.image_view.getView().mapSceneToView(event.scenePos())
        px, py = p.x(), p.y()
        if not (0 <= px <= self.frame_w and 0 <= py <= self.frame_h):
            return
        idx = self._calib_current_index()
        self._calib["points"].append((self.x_cm[idx], self.y_cm[idx], px, py))
        self._redraw_calib_marks()
        self._calib["corner"] += 1
        self._calib["cand"] = 0
        if self._calib["corner"] >= len(self._calib["targets"]):
            self._finish_calibration()
        else:
            self._show_calib_target()

    def _redraw_calib_marks(self):
        pts = self._calib["points"]
        self.calib_marks.setData([p[2] for p in pts], [p[3] for p in pts])

    def on_calib_skip(self):
        if self._calib is None:
            return
        self._calib["cand"] += 1
        self._show_calib_target()

    def on_calib_undo(self):
        if self._calib is None or not self._calib["points"]:
            return
        self._calib["points"].pop()
        self._calib["corner"] = len(self._calib["points"])
        self._calib["cand"] = 0
        self._redraw_calib_marks()
        self._show_calib_target()

    def on_calib_cancel(self):
        self._end_calibration()
        self._apply_arena(self.arena_combo.currentData())
        self.show_frame(self.slider.value())

    def _finish_calibration(self):
        try:
            calib = arena_io.fit_calibration(self._calib["points"])
        except ValueError as exc:
            # Drop the offending click and re-ask rather than losing the lot.
            self._calib["points"].pop()
            self._calib["corner"] = len(self._calib["points"])
            self._calib["cand"] += 1
            self._redraw_calib_marks()
            self._show_calib_target()
            QMessageBox.warning(self, "Can't fit calibration", str(exc))
            return

        current = self.arena_combo.currentText() if self.arena_combo.currentData() else ""
        name, ok = QInputDialog.getText(
            self, "Name this arena",
            f"Fit: ±{calib['residual_px_x']:.1f} px in x, "
            f"±{calib['residual_px_y']:.1f} px in y.\n\n"
            "Name this arena (e.g. 'rat box', 'EthoVision chamber').\n"
            "Reuse the name to recalibrate an existing arena:",
            text=current)
        if not ok or not name.strip():
            self.on_calib_cancel()
            return

        slug = arena_io.slugify(name)
        if arena_io.arena_json_path(slug).is_file():
            reply = QMessageBox.question(
                self, "Replace calibration?",
                f"'{name.strip()}' already exists. Replace its calibration?")
            if reply != QMessageBox.StandardButton.Yes:
                self.on_calib_cancel()
                return

        # The frame you just clicked is already in memory, so the snapshot is free.
        image = (arena_io.save_arena_image(slug, self._last_frame, self._img_levels)
                 if self._last_frame is not None else None)
        arena_io.save_arena(name, calib, self.frame_w, self.frame_h, image=image)
        arena_io.remember_arena(slug)
        self._end_calibration()
        self._populate_arena_combo(slug)
        self._apply_arena(slug)
        self.show_frame(self.slider.value())
        print(f"Arena '{name.strip()}' saved to {arena_io.arena_json_path(slug)} "
              f"(residual {calib['residual_px_x']:.2f} px x, "
              f"{calib['residual_px_y']:.2f} px y)")
        QMessageBox.information(
            self, "Arena saved",
            f"'{name.strip()}' saved.\n\nScrub through the video — the dot should sit "
            f"on the animal. If it drifts, recalibrate with the same name.")

    def _end_calibration(self):
        self._calib = None
        self.calib_marks.setData([], [])
        self.calib_row.setVisible(False)
        self.calibrate_button.setEnabled(True)
        self.arena_combo.setEnabled(True)

    def show_frame(self, behavior_idx):
        if self._has_video:
            video_idx = behavior_idx + self.frame_offset
            frame = self._read_video_frame(video_idx)
            if frame is not None:
                if self._base_levels is None:
                    self._base_levels = (float(frame.min()), float(frame.max()))
                    self._apply_brightness()
                self._last_frame = frame
                self._draw_frame(frame)

        if self._has_behavior:
            t = self.t_behav[min(behavior_idx, len(self.t_behav) - 1)]
            # Hidden mid-calibration: showing the mapping you're replacing would
            # bias where you click.
            if self._has_video and self.dot is not None and self._calib is None:
                idx = min(behavior_idx, len(self.px) - 1)
                self.dot.setData([self.px[idx]], [self.py[idx]])
            self.pos_cursor.setValue(t)
            self.vel_cursor.setValue(t)
        else:
            t = behavior_idx / self.fps

        self.time_label.setText(f"t = {t:.2f} s  |  frame {behavior_idx}")

        if self._has_neural:
            self.lfp_cursor.setValue(t)
        if self._show_spectrogram:
            self.spec_cursor.setValue(t)

        if self.zoom_checkbox.isChecked():
            window = self.window_spinbox.value()
            half = window / 2
            x_left = max(0.0, t - half)
            x_right = x_left + window
            for plot in self._time_plots:
                plot.setXRange(x_left, x_right, padding=0)

    def on_window_duration_changed(self):
        if self.zoom_checkbox.isChecked():
            self.show_frame(self.slider.value())

    def on_zoom_mode_changed(self):
        if not self.zoom_checkbox.isChecked():
            t_max = max(
                self.t_behav[-1] if self._has_behavior else 0,
                self.t_neural[-1] if self._has_neural else 0
            )
            for plot in self._time_plots:
                plot.setXRange(0, t_max, padding=0.02)
        else:
            self.show_frame(self.slider.value())

    def on_slider_changed(self, behavior_idx):
        self.show_frame(behavior_idx)
        # If the user scrubbed while playing, re-baseline the wall clock to the
        # new position so the next tick continues from here instead of yanking
        # the slider back. (_advancing marks our own writes so they don't count.)
        if self.play_timer.isActive() and not self._advancing:
            self._play_start_idx = behavior_idx
            self._play_t0 = time.perf_counter()

    def on_play_clicked(self, checked):
        if checked:
            self._start_playback()
        else:
            self._stop_playback()

    def on_speed_changed(self):
        # Apply the new rate from now forward — re-baseline so already-elapsed
        # time isn't retroactively rescaled (which would jump the playhead).
        if self.play_timer.isActive():
            self._play_start_idx = self.slider.value()
            self._play_t0 = time.perf_counter()
        self._play_speed = 2.0 if self.speed_checkbox.isChecked() else 1.0

    def _start_playback(self):
        # Restart from the beginning if we're parked at the last frame
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        # Latch the wall-clock baseline. Re-latching on every start also makes
        # resume-after-pause correct — the paused seconds are excluded.
        self._play_speed = 2.0 if self.speed_checkbox.isChecked() else 1.0
        self._play_start_idx = self.slider.value()
        self._play_t0 = time.perf_counter()
        self.play_button.setChecked(True)
        self.play_button.setText("⏸ Pause")
        self.play_timer.start(self._play_interval_ms)

    def _stop_playback(self):
        self.play_timer.stop()
        self.play_button.setChecked(False)
        self.play_button.setText("▶ Play")

    def _advance_frame(self):
        # Wall-clock target: the frame the elapsed real time says we should be on.
        # Jump straight there and drop the intermediate paints, so playback holds
        # true speed instead of falling behind when a frame can't paint in time.
        elapsed = time.perf_counter() - self._play_t0
        target = self._play_start_idx + round(elapsed * self._play_speed * self.fps)
        max_idx = self.slider.maximum()
        if target >= max_idx:
            self._set_slider(max_idx)
            self._stop_playback()
            return
        if target <= self.slider.value():
            return  # not due yet (the timer can poll faster than frames fall due)
        # Safety belt: never jump past the sequential-decode window onto the
        # keyframe-seek path (a pathological stall would thrash). Snap to the
        # reachable frame and re-baseline so time-debt can't accumulate.
        reachable = self.slider.value() + _MAX_SEQ_SKIP
        if target > reachable:
            target = reachable
            self._play_start_idx = target
            self._play_t0 = time.perf_counter()
        self._set_slider(target)

    def _set_slider(self, value):
        # Guarded write so on_slider_changed can tell our playback writes from a
        # real user scrub; setValue fires on_slider_changed -> show_frame.
        self._advancing = True
        self.slider.setValue(value)
        self._advancing = False

    def closeEvent(self, event):
        """Clean up handoff directory on close."""
        self.play_timer.stop()
        if self._container is not None:
            self._container.close()
        if self._handoff_dir and Path(self._handoff_dir).exists():
            shutil.rmtree(self._handoff_dir, ignore_errors=True)
        super().closeEvent(event)
