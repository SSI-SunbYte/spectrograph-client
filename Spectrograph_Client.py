#!/usr/bin/env python3
"""
Data from the spectrograph can be imported from a file/SD Card
Assuming the data in the form of hexadecimal, raw binary, or one full frame. Can cope with an unbroken hex string, or separated values (each value must have the same number of hex digits)
Adjust parse_samples() below if the logger adds timestamps / frame headers.

"""
import sys
import os
import re
import json
import math
import time
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui

N_PIXELS = 3694
DUMMY_START = 32              # first effective pixel (S1) sits at index 32
ACTIVE = 3648                 # effective pixels; the other 46 elements are dummy / shielded
SHOW_DUMMY_AT_START = False   # default state of the "Dummy pixels" button
CLIP_MARGIN = 2               # counts from an ADC rail that still count as "clipped"
SEARCH_HALF = 10              # line finder looks this many pixels either side of the click
SMOOTH_PX = 5                 # boxcar smoothing (pixels) before looking for the extremum
FIT_HALF = 3                  # parabola fitted over +/- this many pixels around the extremum
SPEEDS = [1, 2, 5, 10, 25, 50, 100]      # playback speeds, frames per second
HERE = os.path.dirname(os.path.abspath(__file__))
CAL_FILE = os.path.join(HERE, "calibration.json")
SETTINGS_FILE = os.path.join(HERE, "import_settings.json")
DEFAULT_OPTS = {"format": "auto", "digits": 0, "byte_order": "auto", "frame_len": N_PIXELS,
                "skip": 0, "int_us": 0, "adc_bits": 12, "invert": True}


# ===== DATA LOADING ========================================================= #
_HEXLUT = np.full(256, 255, np.uint8)
for _i, _c in enumerate(b"0123456789"):
    _HEXLUT[_c] = _i
for _i, _c in enumerate(b"abcdef"):
    _HEXLUT[_c] = 10 + _i
for _i, _c in enumerate(b"ABCDEF"):
    _HEXLUT[_c] = 10 + _i
_TEXT_OK = np.zeros(256, bool)
for _c in b"0123456789abcdefABCDEFxX,;:_ \t\r\n":
    _TEXT_OK[_c] = True


def parse_samples(data, opts):
    """bytes -> (1-D uint16 array of raw ADC values, description)."""
    data = data[int(opts["skip"]):]
    if not data:
        raise ValueError("The file is empty (after skipping the header bytes).")
    arr = np.frombuffer(data, np.uint8)
    fmt, order = opts["format"], opts["byte_order"]
    if fmt == "auto":
        fmt = "hex" if bool(_TEXT_OK[arr[:200000]].all()) else "binary"

    if fmt == "binary":
        n = arr.size // 2
        if n == 0:
            raise ValueError("Not enough data for a single 16-bit value.")
        dt = ">u2" if order == "big" else "<u2"
        return (np.frombuffer(data, dtype=dt, count=n).astype(np.uint16),
                f"raw binary, {'big' if order == 'big' else 'little'}-endian")

    # ---- hexadecimal text
    if b"0x" in data[:2000].lower():
        data = data.replace(b"0x", b"").replace(b"0X", b"")
        arr = np.frombuffer(data, np.uint8)
    ndig = int(opts["digits"])
    if not ndig:
        toks = [t for t in re.split(rb"[^0-9A-Fa-f]+", data[:400]) if t]
        L = len(toks[0]) if toks else 4
        ndig = L if L in (2, 3, 4) else 4          # long unbroken string -> 16-bit values
    nib = _HEXLUT[arr]
    digits = nib[nib < 16]
    m = digits.size // ndig * ndig
    if m == 0:
        raise ValueError("No hexadecimal data found.")
    digits = digits[:m]
    vals = np.empty(m // ndig, np.uint16)
    step = (4_000_000 // ndig) * ndig
    for a in range(0, m, step):
        d = digits[a:a + step].reshape(-1, ndig).astype(np.uint16)
        v = np.zeros(d.shape[0], np.uint16)
        for k in range(ndig):
            v = (v << 4) | d[:, k]
        vals[a // ndig: a // ndig + v.size] = v
    if ndig == 2:                                   # byte tokens -> combine pairs
        k = vals.size // 2 * 2
        p = vals[:k].reshape(-1, 2)
        s = ((p[:, 0] << 8) | p[:, 1]) if order == "big" else ((p[:, 1] << 8) | p[:, 0])
    elif ndig == 4 and order == "little":           # bytes written in memory order
        s = ((vals & 0xFF) << 8) | (vals >> 8)
    else:                                           # value written as a number
        s = vals
    return s.astype(np.uint16), f"hex text, {ndig} digits per value"


def to_frames(samples, frame_len):
    """1-D samples -> (n_frames, 3694) array. Returns (frames, leftover_samples)."""
    n = samples.size // frame_len
    if n == 0:
        raise ValueError(f"Only {samples.size} values found: fewer than one frame "
                         f"({frame_len} values). Check the format settings.")
    frames = samples[: n * frame_len].reshape(n, frame_len)
    if frame_len == ACTIVE:                         # pad the 46 dummy elements
        frames = np.pad(frames, ((0, 0), (DUMMY_START, N_PIXELS - ACTIVE - DUMMY_START)),
                        mode="edge")
    return frames, samples.size - n * frame_len


# --- demo data: a simulated Sun ------------------------------------------- #
DEMO_BRIGHTNESS = 1.0                    # raise to ~1.6 to see the clipping alert on demo data
DEMO_WL0, DEMO_SLOPE = 672.919, -0.0568029    # demo wavelength scale: wl = WL0 + SLOPE * pixel
DEMO_DARK_FRAMES = 10                    # covered frames recorded before the Sun is exposed
# Approximate CCD response (relative), read off the TCD1304 datasheet graph
_CCD_RESP = [(400, 0.80), (450, 0.92), (500, 0.98), (550, 1.00), (600, 0.96),
             (650, 0.88), (700, 0.76)]
# Strongest Fraunhofer lines in 460-675 nm: (centre nm, approximate equivalent width nm)
SOLAR_LINES = [(486.13, 0.29), (495.76, 0.05), (516.73, 0.07), (517.27, 0.13), (518.36, 0.16),
               (526.95, 0.09), (532.80, 0.05), (588.99, 0.075), (589.59, 0.056), (612.22, 0.03),
               (616.22, 0.05), (643.91, 0.045), (649.50, 0.06), (656.28, 0.36)]


def make_demo_frames(n=160, adc_bits=12, brightness=None):
    """Simulated logger run looking at the Sun (inverted ADC output):
         * continuum = 5772 K black body x CCD response, with Fraunhofer absorption lines
           (H-alpha, Na D, Mg b, H-beta ...) smoothed to the spectrometer's resolution
         * the dummy / shielded elements see no light
         * the Sun is steady: only a slow few-% drift (changing atmosphere / pointing),
           a ~0.5 % frame-to-frame pointing jitter, shot noise and fixed pixel-to-pixel gain
           variations (PRNU), as on the real sensor
         * the first frames are dark frames (cap on)."""
    brightness = DEMO_BRIGHTNESS if brightness is None else brightness
    rng = np.random.default_rng(7)
    full = 2 ** adc_bits - 1
    x = np.arange(N_PIXELS)
    wl = DEMO_WL0 + DEMO_SLOPE * x
    # continuum: Planck(5772 K) x CCD response
    lam = wl * 1e-9
    planck = lam ** -5 / np.expm1(1.4388e-2 / (lam * 5772.0))
    resp = np.interp(wl, [w for w, _ in _CCD_RESP], [r for _, r in _CCD_RESP])
    cont = planck * resp
    eff = (x >= DUMMY_START) & (x < DUMMY_START + ACTIVE)
    cont = np.where(eff, cont / cont[eff].max(), 0.0)
    # absorption lines: Gaussian of fixed equivalent width at the instrument resolution
    sigma_nm = 8 * abs(DEMO_SLOPE)                       # ~8 px
    trans = np.ones(N_PIXELS)
    for w0, ew in SOLAR_LINES:
        depth = min(ew / (sigma_nm * math.sqrt(2 * math.pi)), 0.9)
        trans -= depth * np.exp(-0.5 * ((wl - w0) / sigma_nm) ** 2)
    prnu = 1.0 + 0.015 * rng.normal(size=N_PIXELS)       # fixed pattern, same every frame
    prnu[1::2] *= 1.004                                  # tiny odd/even register imbalance
    spectrum = cont * trans * prnu
    blue = np.clip((600.0 - wl) / 150.0, 0.0, 1.0)       # extinction falls with altitude: blue gains most
    level = 2600.0 * brightness                          # counts above the dark level at the peak
    dark_level, jitter = 500.0, 0.0
    frames = np.empty((n, N_PIXELS), np.uint16)
    for i in range(n):
        if i < DEMO_DARK_FRAMES:
            signal = np.zeros(N_PIXELS)
        else:
            t = (i - DEMO_DARK_FRAMES) / max(n - DEMO_DARK_FRAMES, 1)
            jitter = 0.7 * jitter + rng.normal(0, 0.004)
            gain = 1.0 + 0.015 * math.sin(2 * math.pi * 1.5 * t) + jitter
            signal = level * gain * (1.0 + 0.03 * t * blue) * spectrum
        noise = rng.normal(0, np.sqrt(25.0 + 0.25 * np.clip(signal, 0, None)))
        frames[i] = np.clip(full - (dark_level + signal + noise), 0, full)
    return frames
# ===== END DATA LOADING ===================================================== #


def wavelength_to_rgb(wl):
    """Approximate colour of a visible wavelength (nm); grey outside 380-780 nm."""
    if 380 <= wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif 440 <= wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif 490 <= wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif 510 <= wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif 580 <= wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    elif 645 <= wl <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        return (175, 175, 175)
    if wl < 420:
        f = 0.3 + 0.7 * (wl - 380) / 40
    elif wl > 700:
        f = 0.3 + 0.7 * (780 - wl) / 80
    else:
        f = 1.0
    return tuple(int(255 * (c * f) ** 0.8) for c in (r, g, b))


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #
class ScaleAxis(pg.AxisItem):
    """Vertical axis that rescales the plot when dragged (manual mode only).
       Drag up = zoom in, drag down = zoom out; the bottom of the axis stays put.
       The mouse wheel over the axis does the same."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manual = False

    def set_manual(self, on):
        self.manual = on
        self.setCursor(QtCore.Qt.SizeVerCursor if on else QtCore.Qt.ArrowCursor)

    def _rescale(self, factor):
        vb = self.linkedView()
        if vb is None:
            return
        lo, hi = vb.viewRange()[1]
        vb.setYRange(lo, lo + max((hi - lo) * factor, 1e-6), padding=0)

    def mouseDragEvent(self, ev):
        if not self.manual or ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        dy = ev.pos().y() - ev.lastPos().y()          # item coords: down is positive
        self._rescale(math.exp(dy * 0.01))

    def wheelEvent(self, ev):
        if not self.manual:
            ev.ignore()
            return
        ev.accept()
        d = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        self._rescale(0.9 ** (d / 120.0))


class ChartWidget(pg.PlotWidget):
    """White plot. Its buttons live in a toolbar row under the plot (so they never cover
       axis tick labels); the readout and clipping alert overlay the top of the chart."""
    def __init__(self):
        super().__init__(background="w", axisItems={"left": ScaleAxis("left")})
        self.save_btn = QtWidgets.QPushButton("Save chart")
        self.save_btn.setFixedSize(92, 26)
        self.auto_chk = QtWidgets.QPushButton("Auto-scale Y: ON")
        self.auto_chk.setCheckable(True)
        self.auto_chk.setChecked(True)
        self.auto_chk.setFixedSize(135, 26)
        self.auto_chk.setToolTip("Click to switch automatic vertical scaling on/off.\n"
                                 "When OFF, drag the vertical axis (or scroll over it) to rescale.")
        for b in (self.save_btn, self.auto_chk):
            b.setCursor(QtCore.Qt.PointingHandCursor)
        # clipping alert: small, top-right, ignores the mouse so it never blocks clicks
        self.alert = QtWidgets.QLabel("", self)
        self.alert.setStyleSheet("background: rgba(255,170,0,235); color: black; font-weight: bold;"
                                 " border: 1px solid #a66a00; padding: 3px;")
        self.alert.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.alert.hide()
        self.readout = QtWidgets.QLabel("Click the spectrum to read a pixel", self)
        self.readout.setStyleSheet("background: rgba(255,255,255,200); color: black;"
                                   " border: 1px solid #aaa; padding: 3px;")
        self.readout.move(70, 8)
        self.readout.adjustSize()

    def _place_alert(self):
        a = self.__dict__.get("alert")
        if a is not None:
            a.adjustSize()
            a.move(self.width() - a.width() - 10, 8)

    def set_alert(self, text):
        self.alert.setText(text)
        self.alert.setVisible(bool(text))
        self._place_alert()

    def set_readout(self, text):
        self.readout.setText(text)
        self.readout.adjustSize()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_alert()          # guarded: pyqtgraph calls this during its own __init__


class OpenDialog(QtWidgets.QDialog):
    """Asks how the SD-card file is encoded. Choices are remembered between runs."""
    def __init__(self, parent, path, opts):
        super().__init__(parent)
        self.setWindowTitle("Open data file")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(f"<b>{os.path.basename(path)}</b>  "
                                       f"({os.path.getsize(path) / 1e6:.2f} MB)"))
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        def combo(items, current):
            c = QtWidgets.QComboBox()
            for text, val in items:
                c.addItem(text, val)
            c.setCurrentIndex(max(c.findData(current), 0))
            return c

        self.fmt = combo([("Auto-detect", "auto"), ("Hex text", "hex"), ("Raw binary", "binary")],
                         opts["format"])
        self.digits = combo([("Auto", 0), ("2 (one byte per token)", 2), ("3", 3),
                             ("4 (16-bit per token)", 4)], opts["digits"])
        self.order = combo([("Auto", "auto"), ("Little-endian", "little"), ("Big-endian", "big")],
                           opts["byte_order"])
        self.order.setToolTip("Auto: binary = little-endian; byte tokens = low byte first;\n"
                              "4-digit hex tokens = read as the number written (0E3A = 3642).\n"
                              "Little-endian on 4-digit tokens swaps the two bytes.")
        self.flen = combo([("3694 (all elements)", N_PIXELS),
                           ("3648 (effective pixels only)", ACTIVE)], opts["frame_len"])
        self.skip = QtWidgets.QSpinBox()
        self.skip.setRange(0, 2_000_000_000)
        self.skip.setSuffix(" bytes")
        self.skip.setValue(int(opts["skip"]))
        self.int_us = QtWidgets.QSpinBox()
        self.int_us.setRange(0, 100_000_000)
        self.int_us.setSuffix(" us")
        self.int_us.setSpecialValueText("unknown")
        self.int_us.setValue(int(opts["int_us"]))
        self.int_us.setToolTip("Only used to warn when a dark frame was taken at a different "
                               "integration time.")
        self.adc = QtWidgets.QSpinBox()
        self.adc.setRange(8, 16)
        self.adc.setSuffix(" bit")
        self.adc.setValue(int(opts["adc_bits"]))
        self.invert = QtWidgets.QCheckBox("Invert (sensor output falls with light)")
        self.invert.setChecked(bool(opts["invert"]))
        form.addRow("File format", self.fmt)
        form.addRow("Hex digits per value", self.digits)
        form.addRow("Byte order", self.order)
        form.addRow("Values per frame", self.flen)
        form.addRow("Skip header", self.skip)
        form.addRow("Integration time", self.int_us)
        form.addRow("ADC resolution", self.adc)
        form.addRow("", self.invert)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def opts(self):
        return {"format": self.fmt.currentData(), "digits": self.digits.currentData(),
                "byte_order": self.order.currentData(), "frame_len": self.flen.currentData(),
                "skip": self.skip.value(), "int_us": self.int_us.value(),
                "adc_bits": self.adc.value(), "invert": self.invert.isChecked()}


class PlayButton(QtWidgets.QPushButton):
    def __init__(self, text, w, h=34):
        super().__init__(text)
        self.setFixedSize(w, h)
        self.setCursor(QtCore.Qt.PointingHandCursor)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TCD1304 Spectrometer")
        self.resize(1020, 680)
        self.setMinimumSize(900, 600)
        sb = self.statusBar()
        sb.setSizeGripEnabled(True)
        sb.showMessage("Click the spectrum to read a pixel  |  drag the bottom-right corner to resize")

        self.opts = dict(DEFAULT_OPTS)
        try:
            with open(SETTINGS_FILE) as f:
                self.opts.update(json.load(f))
        except (OSError, ValueError):
            pass
        self.frames = None            # (n, 3694) uint16
        self.idx = 0
        self.source = ""
        self.adc_bits = self.opts["adc_bits"]
        self.invert = self.opts["invert"]
        self.int_us = 0
        self.coef = None
        self.cur_raw = None
        self.last_y = None            # displayed data (after dark subtraction)
        self.last_base = None         # displayed frame before dark subtraction
        self.sel_px = None            # selected position, in (fractional) pixels
        self.dark = None
        self.dark_us = 0
        self.fill_key = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self._build_ui()
        self._load_calibration()

    # ---------------------------------------------------------------- UI --- #
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # ---- chart
        self.chart = ChartWidget()
        pi = self.chart.getPlotItem()
        pi.showGrid(x=True, y=True, alpha=0.25)
        pi.setMenuEnabled(False)
        self.chart.setMouseEnabled(False, False)
        for ax in ("left", "bottom"):
            a = pi.getAxis(ax)
            a.setPen(pg.mkPen("k"))
            a.setTextPen(pg.mkPen("k"))
        pi.setLabel("left", "Intensity [a.u.]", color="#000000")
        pi.setLabel("bottom", "Pixels", color="#000000")
        self.curve = pi.plot(pen=pg.mkPen("#1b2a3a", width=1), fillLevel=0,
                             brush=pg.mkBrush(125, 200, 255))
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen("#d32f2f", width=1.5, style=QtCore.Qt.DashLine))
        self.vline.setZValue(10)
        self.vline.hide()
        pi.addItem(self.vline)
        self.chart.scene().sigMouseClicked.connect(self.on_chart_click)
        pi.setXRange(DUMMY_START, DUMMY_START + ACTIVE - 1, padding=0)
        pi.setYRange(0, 4000, padding=0)
        self.chart.save_btn.clicked.connect(self.save_chart)
        self.chart.auto_chk.toggled.connect(self.autoscale_toggled)
        self.chart.setMinimumHeight(250)
        root.addWidget(self.chart, 1)

        tools = QtWidgets.QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        tools.addWidget(self.chart.save_btn)
        tools.addWidget(self.chart.auto_chk)
        self.btn_dummy = QtWidgets.QPushButton()
        self.btn_dummy.setCheckable(True)
        self.btn_dummy.setChecked(SHOW_DUMMY_AT_START)
        self.btn_dummy.setFixedSize(180, 26)
        self.btn_dummy.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_dummy.setToolTip("Show or hide the 46 dummy / light-shielded elements "
                                  "(pixels 0-31 and 3680-3693).\nPixel numbers do not change.")
        self.btn_dummy.toggled.connect(self.dummy_toggled)
        self.dummy_toggled(SHOW_DUMMY_AT_START)
        tools.addWidget(self.btn_dummy)
        self.btn_find = QtWidgets.QPushButton()
        self.btn_find.setCheckable(True)
        self.btn_find.setChecked(False)               # default: absorption (the Sun)
        self.btn_find.setFixedSize(230, 26)
        self.btn_find.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_find.setToolTip(f"What the line finder looks for within {SEARCH_HALF} px of your click:\n"
                                 "the lowest point (absorption line) or the highest point (emission line).")
        self.btn_find.toggled.connect(self.find_mode_toggled)
        self.find_mode_toggled(False)
        tools.addWidget(self.btn_find)
        tools.addStretch(1)
        root.addLayout(tools)

        bottom = QtWidgets.QHBoxLayout()
        root.addLayout(bottom)

        # ---- bottom left: playback controls
        left = QtWidgets.QGridLayout()
        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(450)
        bottom.addWidget(left_w)
        self.lbl_frames = QtWidgets.QLabel("Frame: - / -")
        left.addWidget(self.lbl_frames, 0, 0, 1, 2)
        left.addWidget(QtWidgets.QLabel("Playback speed"), 1, 1)

        self.rb_single = QtWidgets.QRadioButton("Single frame mode")
        self.rb_avg = QtWidgets.QRadioButton("Averaging mode")
        self.rb_single.setChecked(True)
        self.rb_avg.setToolTip("Shows the average of the last N frames up to the current one.")
        self.avg_n = QtWidgets.QSpinBox()
        self.avg_n.setRange(2, 1000)
        self.avg_n.setValue(10)
        self.avg_n.setPrefix("N = ")
        self.avg_n.setMinimumWidth(95)
        avg_row = QtWidgets.QHBoxLayout()
        self.rb_avg.setMinimumWidth(130)
        self.rb_single.setMinimumWidth(170)
        avg_row.addWidget(self.rb_avg)
        avg_row.addWidget(self.avg_n)
        avg_row.addStretch(1)
        radios = QtWidgets.QVBoxLayout()
        radios.addWidget(self.rb_single)
        radios.addLayout(avg_row)
        left.addLayout(radios, 2, 0)
        self.rb_single.toggled.connect(self.show_current)
        self.avg_n.valueChanged.connect(self.show_current)

        self.cb_speed = QtWidgets.QComboBox()
        for sp in SPEEDS:
            self.cb_speed.addItem(f"{sp} frames/s", sp)
        self.cb_speed.setCurrentIndex(SPEEDS.index(10))
        self.cb_speed.currentIndexChanged.connect(self._speed_changed)
        left.addWidget(self.cb_speed, 2, 1)

        nav = QtWidgets.QHBoxLayout()
        self.btn_prev = PlayButton("◀", 44)
        self.btn_play = PlayButton("Play", 80)
        self.btn_pause = PlayButton("Pause", 80)
        self.btn_next = PlayButton("▶", 44)
        self.btn_prev.setToolTip("Previous frame")
        self.btn_next.setToolTip("Next frame")
        for b in (self.btn_prev, self.btn_play, self.btn_pause, self.btn_next):
            nav.addWidget(b)
        nav.addStretch(1)
        left.addLayout(nav, 3, 0, 1, 2)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(1))
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(lambda v: self.goto(v))
        left.addWidget(self.slider, 4, 0, 1, 2)

        dark_row = QtWidgets.QHBoxLayout()
        self.btn_dark = QtWidgets.QPushButton("Set dark frame ▾")
        self.btn_dark.setCursor(QtCore.Qt.PointingHandCursor)
        menu = QtWidgets.QMenu(self.btn_dark)
        self.act_dark_cur = menu.addAction("From the current frame")
        self.act_dark_rng = menu.addAction("From a range of frames...")
        self.act_dark_file = menu.addAction("From another file...")
        self.act_dark_cur.triggered.connect(self.dark_from_current)
        self.act_dark_rng.triggered.connect(self.dark_from_range)
        self.act_dark_file.triggered.connect(self.dark_from_file)
        self.btn_dark.setMenu(menu)
        self.chk_dark = QtWidgets.QCheckBox("Subtract dark")
        self.chk_dark.setEnabled(False)
        self.chk_dark.toggled.connect(self.redisplay)
        self.lbl_dark = QtWidgets.QLabel("")
        dark_row.addWidget(self.btn_dark)
        dark_row.addWidget(self.chk_dark)
        dark_row.addWidget(self.lbl_dark)
        dark_row.addStretch(1)
        left.addLayout(dark_row, 5, 0, 1, 2)

        exp_row = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export frame to CSV...")
        self.btn_export.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_export.clicked.connect(self.export_csv)
        self.chk_loop = QtWidgets.QCheckBox("Loop playback")
        exp_row.addWidget(self.btn_export)
        exp_row.addWidget(self.chk_loop)
        exp_row.addStretch(1)
        left.addLayout(exp_row, 6, 0, 1, 2)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(70)
        left.addWidget(self.log_box, 7, 0, 1, 2)

        # ---- bottom middle: calibration
        cal = QtWidgets.QGroupBox("Calibration parameters")
        g = QtWidgets.QGridLayout(cal)
        self.wl_edits, self.px_edits = [], []
        for i in range(4):
            w, p = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
            w.setFixedWidth(90)
            p.setFixedWidth(90)
            self.wl_edits.append(w)
            self.px_edits.append(p)
            g.addWidget(QtWidgets.QLabel(f"Wavelength {i + 1} (nm)"), i, 0)
            g.addWidget(w, i, 1)
            g.addWidget(QtWidgets.QLabel(f"Pixel-peak {i + 1}"), i, 2)
            g.addWidget(p, i, 3)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply_calibration)
        #self.btn_apply.setFixedSize(100, 26)
        g.addWidget(self.btn_apply, 0, 4, 2, 1)
        self.lbl_fit = QtWidgets.QLabel("Fitting parameters:")
        self.lbl_fit.setWordWrap(True)
        g.addWidget(self.lbl_fit, 4, 0, 1, 3)
        self.chk_cal = QtWidgets.QCheckBox("Use calibration parameters")
        self.chk_cal.setEnabled(False)
        self.chk_cal.toggled.connect(self.update_axis)
        g.addWidget(self.chk_cal, 4, 3, 1, 2)
        bottom.addWidget(cal, 1)

        # ---- bottom right: data source
        right = QtWidgets.QVBoxLayout()
        right_w = QtWidgets.QWidget()
        right_w.setLayout(right)
        right_w.setFixedWidth(200)
        bottom.addWidget(right_w)
        right.addWidget(QtWidgets.QLabel("Data source"))
        self.btn_open = QtWidgets.QPushButton("Open file...")
        self.btn_demo = QtWidgets.QPushButton("Demo data")
        for b in (self.btn_open, self.btn_demo):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            right.addWidget(b)
        self.btn_open.clicked.connect(self.open_file)
        self.btn_demo.clicked.connect(self.load_demo)
        self.lbl_status = QtWidgets.QLabel("Status: no data loaded")
        self.lbl_status.setWordWrap(True)
        right.addWidget(self.lbl_status)
        right.addStretch(1)

        self._set_loaded(False)

    def _set_loaded(self, on):
        for w in (self.rb_single, self.rb_avg, self.avg_n, self.cb_speed, self.btn_prev,
                  self.btn_play, self.btn_next, self.slider, self.btn_dark, self.btn_export,
                  self.chk_loop):
            w.setEnabled(on)
        self.btn_pause.setEnabled(False)

    def log(self, text):
        self.log_box.appendPlainText(text)

    # -------------------------------------------------------- loading data -- #
    def read_frames(self, path, opts):
        """Read + parse a file. Returns frames; logs a summary."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            with open(path, "rb") as f:
                data = f.read()
            samples, how = parse_samples(data, opts)
            frames, left = to_frames(samples, int(opts["frame_len"]))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.log(f"{os.path.basename(path)}: {frames.shape[0]} frames ({how}).")
        if left:
            self.log(f"Note: {left} trailing values ignored (incomplete frame).")
        top = int(frames.max())
        if top > 2 ** opts["adc_bits"] - 1:
            self.log(f"Warning: values up to {top} exceed a {opts['adc_bits']}-bit ADC. "
                     "Check the byte order / digits settings.")
        return frames

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open SD-card data file", "",
                                                        "All files (*)")
        if not path:
            return
        dlg = OpenDialog(self, path, self.opts)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        opts = dlg.opts()
        try:
            frames = self.read_frames(path, opts)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Could not read file", str(e))
            return
        self.opts = opts
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(opts, f)
        except OSError:
            pass
        self.set_data(frames, os.path.basename(path), opts)

    def load_demo(self):
        opts = dict(self.opts, adc_bits=12, invert=True, int_us=1000)
        self.set_data(make_demo_frames(160, 12), "demo data (simulated Sun)", opts)
        marks = [("H-alpha", 656.28), ("Na D", 589.3), ("Mg b", 517.3), ("H-beta", 486.13)]
        txt = ", ".join(f"{n} {w} nm @ px {(w - DEMO_WL0) / DEMO_SLOPE:.0f}" for n, w in marks)
        self.log("Demo: simulated Sun. Frames 1-10 are dark frames (cap on); the rest show a "
                 "steady solar spectrum. Calibration lines: " + txt + ".")

    def set_data(self, frames, name, opts):
        self.pause()
        self.frames = frames
        self.source = name
        self.adc_bits = int(opts["adc_bits"])
        self.invert = bool(opts["invert"])
        self.int_us = int(opts["int_us"])
        self.idx = 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(frames) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.lbl_status.setText(f"Status: loaded\n{name}\n{len(frames)} frames"
                                + (f"\nIntegration: {self.int_us} us" if self.int_us else ""))
        self._set_loaded(True)
        self.update_dark_label()
        self.show_current()

    # ------------------------------------------------------------ playback -- #
    def goto(self, i):
        if self.frames is None:
            return
        self.idx = int(np.clip(i, 0, len(self.frames) - 1))
        self.show_current()

    def step(self, d):
        self.pause()
        self.goto(self.idx + d)

    def play(self):
        if self.frames is None:
            return
        if self.idx >= len(self.frames) - 1:
            self.idx = 0
        self.timer.start(int(1000 / self.cb_speed.currentData()))
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)

    def pause(self):
        self.timer.stop()
        if self.frames is not None:
            self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def _speed_changed(self):
        if self.timer.isActive():
            self.timer.start(int(1000 / self.cb_speed.currentData()))

    def tick(self):
        if self.idx + 1 >= len(self.frames):
            if self.chk_loop.isChecked():
                self.goto(0)
            else:
                self.pause()
            return
        self.goto(self.idx + 1)

    def show_current(self, *_):
        if self.frames is None:
            return
        n, i = len(self.frames), self.idx
        if self.rb_avg.isChecked():
            a = max(0, i - self.avg_n.value() + 1)
            raw = self.frames[a:i + 1].mean(axis=0)
            suffix = f"  (average of {i - a + 1})"
        else:
            raw = self.frames[i].astype(np.float64)
            suffix = ""
        self.cur_raw = raw
        self.check_clipping(raw)
        full = 2 ** self.adc_bits - 1
        base = (full - raw) if self.invert else raw
        self.last_base = base
        self.lbl_frames.setText(f"Frame: {i + 1} / {n}{suffix}")
        if self.slider.value() != i:
            self.slider.blockSignals(True)
            self.slider.setValue(i)
            self.slider.blockSignals(False)
        self.draw(self.processed(base))

    # ------------------------------------------------------------ clipping -- #
    def visible_slice(self):
        if self.btn_dummy.isChecked():
            return slice(0, N_PIXELS)
        return slice(DUMMY_START, DUMMY_START + ACTIVE)

    def check_clipping(self, raw):
        """Flag pixels sitting on an ADC rail: their true intensity is unknown."""
        full = 2 ** self.adc_bits - 1
        r = raw[self.visible_slice()]
        at_max = int(np.count_nonzero(r >= full - CLIP_MARGIN))
        at_min = int(np.count_nonzero(r <= CLIP_MARGIN))
        bright, dark = (at_min, at_max) if self.invert else (at_max, at_min)
        lines = []
        if bright:
            lines.append(f"⚠ CLIPPING: {bright} px saturated - reduce integration time on the logger")
        if dark:
            lines.append(f"⚠ {dark} px below the ADC floor")
        self.chart.set_alert("\n".join(lines))

    # ---------------------------------------------------------- dark frame --- #
    def processed(self, base):
        if self.chk_dark.isChecked() and self.dark is not None:
            return base - self.dark
        return base

    def redisplay(self, *_):
        if self.last_base is not None:
            self.draw(self.processed(self.last_base))

    def _set_dark(self, raw_mean, desc, int_us):
        full = 2 ** self.adc_bits - 1
        self.dark = (full - raw_mean) if self.invert else np.asarray(raw_mean, float)
        self.dark_us = int(int_us)
        self.chk_dark.setEnabled(True)
        self.chk_dark.setChecked(True)               # also redraws
        self.update_dark_label()
        self.log(f"Dark frame set from {desc} (mean level {self.dark.mean():.0f} counts).")

    def dark_from_current(self):
        if self.cur_raw is not None:
            self._set_dark(self.cur_raw, f"frame {self.idx + 1}", self.int_us)

    def dark_from_range(self):
        if self.frames is None:
            return
        n = len(self.frames)
        a, ok = QtWidgets.QInputDialog.getInt(self, "Dark frame",
                                              f"First dark frame (1-{n}):", 1, 1, n)
        if not ok:
            return
        b, ok = QtWidgets.QInputDialog.getInt(self, "Dark frame", f"Last dark frame ({a}-{n}):",
                                              min(a + 9, n), a, n)
        if ok:
            self._set_dark(self.frames[a - 1:b].mean(axis=0), f"frames {a}-{b}", self.int_us)

    def dark_from_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open dark-frame file", "",
                                                        "All files (*)")
        if not path:
            return
        dlg = OpenDialog(self, path, self.opts)
        dlg.setWindowTitle("Open dark-frame file")
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        opts = dlg.opts()
        try:
            frames = self.read_frames(path, opts)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Could not read file", str(e))
            return
        self._set_dark(frames.mean(axis=0), f"{os.path.basename(path)} "
                       f"({len(frames)} frames)", opts["int_us"])

    def update_dark_label(self, *_):
        if self.dark is None or not self.dark_us:
            self.lbl_dark.setText("")
            return
        same = (not self.int_us) or self.int_us == self.dark_us
        self.lbl_dark.setText(f"({self.dark_us} us)" + ("" if same else " ⚠"))
        self.lbl_dark.setStyleSheet("" if same else "color: #b36b00; font-weight: bold;")
        self.lbl_dark.setToolTip("" if same else
                                 f"Dark level grows with integration time: this dark frame is for "
                                 f"{self.dark_us} us but the data is {self.int_us} us.")

    # ------------------------------------------------------ colour fill ---- #
    def update_fill(self, x):
        cal = self.chk_cal.isChecked() and self.coef is not None
        key = (round(float(x.min()), 3), round(float(x.max()), 3)) if cal else None
        if key == self.fill_key:
            return
        self.fill_key = key
        if key is None or key[1] - key[0] < 1e-6:
            brush = pg.mkBrush(125, 200, 255)
        else:
            lo, hi = key
            grad = QtGui.QLinearGradient(lo, 0, hi, 0)      # data coordinates
            for t in np.linspace(0.0, 1.0, 150):
                grad.setColorAt(float(t), QtGui.QColor(*wavelength_to_rgb(lo + t * (hi - lo))))
            brush = QtGui.QBrush(grad)
        self.curve.setFillBrush(brush)

    # -------------------------------------------------------------- plot --- #
    def x_values(self):
        px = np.arange(N_PIXELS, dtype=float)
        if self.chk_cal.isChecked() and self.coef is not None:
            return np.polyval(self.coef, px)
        return px

    def draw(self, y):
        self.last_y = y
        sl = self.visible_slice()
        x = self.x_values()[sl]
        yv = y[sl]
        self.update_fill(x)
        self.curve.setData(x, yv)
        self.update_selection()
        if self.chart.auto_chk.isChecked():
            top = max(float(np.max(yv)) * 1.08, 10.0)
            bot = min(float(np.min(yv)), 0.0)
            self.chart.getPlotItem().setYRange(bot, top, padding=0)

    def dummy_toggled(self, shown):
        self.btn_dummy.setText("Dummy pixels: SHOWN" if shown else "Dummy pixels: HIDDEN")
        if hasattr(self, "chart") and hasattr(self, "chk_cal"):
            self.update_axis()

    def autoscale_toggled(self, auto):
        self.chart.auto_chk.setText("Auto-scale Y: ON" if auto else "Auto-scale Y: OFF")
        self.chart.getPlotItem().getAxis("left").set_manual(not auto)
        if auto:
            if self.last_y is not None:
                self.draw(self.last_y)
            self.log("Y axis: automatic scaling.")
        else:
            self.log("Y axis: fixed. Drag the vertical axis (or scroll over it) to rescale.")

    def update_axis(self, *_):
        pi = self.chart.getPlotItem()
        cal = self.chk_cal.isChecked() and self.coef is not None
        pi.setLabel("bottom", "Wavelength (nm)" if cal else "Pixels", color="#000000")
        x = self.x_values()[self.visible_slice()]
        pi.setXRange(float(x.min()), float(x.max()), padding=0)
        if self.last_y is not None:
            self.draw(self.last_y)
        else:
            self.update_selection()

    def save_chart(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save chart", time.strftime("spectrum_%Y%m%d_%H%M%S.png"), "PNG (*.png)")
        if fn:
            from pyqtgraph.exporters import ImageExporter
            ImageExporter(self.chart.getPlotItem()).export(fn)
            self.log(f"Chart saved: {os.path.basename(fn)}")

    def export_csv(self):
        if self.cur_raw is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export frame", f"frame_{self.idx + 1:05d}.csv", "CSV (*.csv)")
        if not fn:
            return
        cal = self.chk_cal.isChecked() and self.coef is not None
        x = self.x_values()
        data = np.column_stack([np.arange(N_PIXELS), x if cal else np.full(N_PIXELS, np.nan),
                                self.cur_raw, self.last_y])
        hdr = (f"TCD1304 spectrum | source={self.source} frame={self.idx + 1} "
               f"mode={'average of ' + str(self.avg_n.value()) if self.rb_avg.isChecked() else 'single'} "
               f"dark_subtracted={self.chk_dark.isChecked() and self.dark is not None}\n"
               "pixel,wavelength_nm,raw_counts,displayed")
        np.savetxt(fn, data, delimiter=",", header=hdr, comments="# ", fmt="%.6g")
        self.log(f"Exported {os.path.basename(fn)}")

    # ------------------------------------------------- click to read pixel --- #
    def px_to_x(self, px):
        return float(np.interp(px, np.arange(N_PIXELS), self.x_values()))

    def x_to_px(self, xv):
        """Axis coordinate -> fractional pixel (interpolated, so not limited to integers)."""
        if not (self.chk_cal.isChecked() and self.coef is not None):
            return float(np.clip(xv, 0, N_PIXELS - 1))
        x = self.x_values()
        order = np.argsort(x)
        return float(np.interp(xv, x[order], np.arange(N_PIXELS)[order]))

    @staticmethod
    def find_line_centre(y, c, vis, minimum):
        """Sub-pixel centre of the deepest dip (minimum=True, absorption) or the highest
           peak (minimum=False, emission) within +/-SEARCH_HALF pixels of pixel c.
           The data are lightly smoothed first (so noise can't pick the wrong pixel on a
           broad line), then a parabola is least-squares fitted around the extremum."""
        lo, hi = max(vis.start, c - SEARCH_HALF), min(vis.stop, c + SEARCH_HALF + 1)
        if hi <= lo:
            return None
        a, b = max(0, lo - SMOOTH_PX - FIT_HALF), min(len(y), hi + SMOOTH_PX + FIT_HALF)
        ys = np.convolve(y[a:b], np.ones(SMOOTH_PX) / SMOOTH_PX, mode="same")
        seg = ys[lo - a: hi - a]
        j = int(np.argmin(seg) if minimum else np.argmax(seg)) + (lo - a)
        if FIT_HALF <= j < len(ys) - FIT_HALF:
            qa, qb, _ = np.polyfit(np.arange(-FIT_HALF, FIT_HALF + 1),
                                   ys[j - FIT_HALF: j + FIT_HALF + 1], 2)
            if (qa > 0) if minimum else (qa < 0):         # curvature must match the line type
                off = -qb / (2 * qa)
                if abs(off) <= 1.5:
                    return a + j + off
        return float(a + j)

    def find_mode_toggled(self, emission):
        self.btn_find.setText("Find lines: EMISSION (peak)" if emission
                              else "Find lines: ABSORPTION (dip)")
        self.update_selection()

    def on_chart_click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.chart.getPlotItem().vb
        pos = ev.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        self.sel_px = self.x_to_px(vb.mapSceneToView(pos).x())
        self.chart.setFocus()                       # so Left/Right keys can nudge
        self.update_selection(announce=True)

    def update_selection(self, announce=False):
        if self.sel_px is None:
            return
        px = self.sel_px
        xs = self.px_to_x(px)
        cal = self.chk_cal.isChecked() and self.coef is not None
        self.vline.setValue(xs)
        self.vline.show()
        lines = [f"Pixel: {px:.2f}" + (f"    Wavelength: {xs:.2f} nm" if cal else "")]
        if self.last_y is not None:
            idx = np.arange(N_PIXELS)
            c = int(round(px))
            emission = self.btn_find.isChecked()
            txt = f"Intensity: {np.interp(px, idx, self.last_y):.0f}"
            pk = self.find_line_centre(self.last_y, c, self.visible_slice(), not emission)
            if pk is not None:
                txt += (f"    Local {'peak' if emission else 'dip'} (±{SEARCH_HALF} px): pixel {pk:.2f}"
                        + (f" ({self.px_to_x(pk):.2f} nm)" if cal else ""))
            lines.append(txt)
        lines.append("(←/→ nudge 0.1 px, Shift+←/→ nudge 1 px)")
        self.chart.set_readout("\n".join(lines))
        if announce:
            QtWidgets.QApplication.clipboard().setText(f"{px:.2f}")
            self.log(f"Selected pixel {px:.2f} (copied to clipboard)")

    def keyPressEvent(self, e):
        if self.sel_px is not None and e.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            step = 1.0 if e.modifiers() & QtCore.Qt.ShiftModifier else 0.1
            if e.key() == QtCore.Qt.Key_Left:
                step = -step
            self.sel_px = float(np.clip(round(self.sel_px + step, 3), 0, N_PIXELS - 1))
            self.update_selection(announce=True)
        else:
            super().keyPressEvent(e)

    # ------------------------------------------------------- calibration --- #
    def apply_calibration(self):
        pts = []
        for w, p in zip(self.wl_edits, self.px_edits):
            try:
                pts.append((float(p.text()), float(w.text())))
            except ValueError:
                pass
        if len(pts) < 2 or len({p for p, _ in pts}) < len(pts):
            self.log("Need at least 2 valid wavelength/pixel pairs (distinct pixels).")
            return
        px, wl = np.array(pts).T
        deg = min(2, len(pts) - 1)                    # linear with 2 pts, else quadratic
        self.coef = np.polyfit(px, wl, deg)
        names = "abc"[2 - deg:]                       # y = a x^2 + b x + c (or b x + c)
        self.lbl_fit.setText("Fitting parameters: " + "  ".join(
            f"{n}={c:.6g}" for n, c in zip(names, self.coef)))
        self.chk_cal.setEnabled(True)
        try:
            with open(CAL_FILE, "w") as f:
                json.dump({"wavelength": [e.text() for e in self.wl_edits],
                           "pixel": [e.text() for e in self.px_edits]}, f)
            self.log("Calibration applied and saved.")
        except OSError as e:
            self.log(f"Calibration applied (could not save file: {e})")
        self.update_axis()

    def _load_calibration(self):
        try:
            with open(CAL_FILE) as f:
                d = json.load(f)
            for e, t in zip(self.wl_edits, d["wavelength"]):
                e.setText(t)
            for e, t in zip(self.px_edits, d["pixel"]):
                e.setText(t)
            self.apply_calibration()
            self.log("Calibration file is loaded.")
        except (OSError, KeyError, ValueError):
            self.log("No calibration file found.")

    def closeEvent(self, e):
        self.timer.stop()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#f0f0f0"))
    app.setPalette(pal)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
