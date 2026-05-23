"""
PySide6 Fullscreen Transparent Overlay Window — Phase 2

Each OverlayWindow owns one vision mode and renders independently.
Multiple instances can coexist, each showing a different vision filter
over the full screen simultaneously.

Click-through / freeze notes (unchanged from Phase 1)
──────────────────────────────────────────────────────
Qt's WA_TransparentForMouseEvents operates at the Qt event level only.
Windows still delivers WM_NCHITTEST to the native HWND.  The correct fix
is WS_EX_LAYERED | WS_EX_TRANSPARENT via Win32, which answers NCHITTEST
with HTTRANSPARENT before the message reaches Qt.

Backpressure (per-overlay)
──────────────────────────
Each overlay has its own _busy flag.  submit_frame() drops the incoming
frame if _busy is True.  paintEvent() clears _busy after drawing.
This caps the Qt-event-queue depth at one frame per overlay, preventing
memory growth when the GUI thread is slower than the capture rate.

Thread model
────────────
  submit_frame()  — called from worker thread; emits _frame_signal
  _on_frame()     — slot; Qt auto-selects QueuedConnection (cross-thread)
                    and delivers to the GUI thread
  paintEvent()    — GUI thread; draws and releases backpressure
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
from typing import List, Optional, Tuple

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter

from modes.base_mode import BaseVisionMode

# ── Win32 constants ───────────────────────────────────────────────────────────
_GWL_EXSTYLE         = -20
_WS_EX_LAYERED       = 0x00080000   # Required companion for WS_EX_TRANSPARENT
_WS_EX_TRANSPARENT   = 0x00000020   # OS answers NCHITTEST before Qt sees it
_WDA_EXCLUDEFROMCAP  = 0x00000011   # Hide from BitBlt / screen-recording
_SWP_NOMOVE          = 0x0002
_SWP_NOSIZE          = 0x0001
_SWP_NOZORDER        = 0x0004
_SWP_FRAMECHANGED    = 0x0020       # Forces the style change to take effect


class OverlayWindow(QWidget):
    """
    Fullscreen always-on-top overlay rendering one vision mode independently.

    Frame pipeline:
        OverlayManager.distribute()
            └─► submit_frame(QImage)          [worker thread]
                    └─► _frame_signal.emit()  [QueuedConnection → GUI thread]
                            └─► _on_frame()   [GUI thread]
                                    └─► update() → paintEvent()
    """

    _frame_signal: Signal = Signal(QImage)

    def __init__(
        self,
        overlay_id: int,
        mode: BaseVisionMode,
        mode_index: int,
        all_modes: List[BaseVisionMode],
    ) -> None:
        """
        Args:
            overlay_id:  Unique integer ID (0-based) assigned by OverlayManager.
            mode:        Initial vision mode to apply.
            mode_index:  Index of mode in all_modes.
            all_modes:   Shared list of all available modes (for cycling).
        """
        super().__init__()
        self.overlay_id    = overlay_id
        self._mode         = mode
        self._mode_index   = mode_index
        self._all_modes    = all_modes
        self._pixmap: Optional[QPixmap] = None
        self._busy: bool   = False

        # Phase 4: optional clip rect for window-tracked overlays.
        # (x1, y1, x2, y2) in desktop coordinates; None = full-screen.
        self._clip_rect: Optional[Tuple[int, int, int, int]] = None
        # HWND of the tracked window, or None for a free/fullscreen overlay.
        self._tracked_hwnd: Optional[int] = None

        # Force QueuedConnection so the emit() in submit_frame() (worker thread)
        # always posts an event rather than calling _on_frame() synchronously.
        # Without this Qt uses auto-detection, which could fall back to a direct
        # call if thread affinity is ambiguous, blocking the worker thread.
        self._frame_signal.connect(self._on_frame, Qt.ConnectionType.QueuedConnection)
        self._setup_window()

    # ── Mode management ───────────────────────────────────────────────────

    @property
    def mode(self) -> BaseVisionMode:
        """Currently active vision mode."""
        return self._mode

    @property
    def mode_index(self) -> int:
        """Zero-based index of the active mode in all_modes."""
        return self._mode_index

    @property
    def mode_count(self) -> int:
        """Total number of available modes."""
        return len(self._all_modes)

    def set_mode(self, mode: BaseVisionMode, index: int) -> None:
        """
        Atomically swaps the active mode.

        Safe to call from the GUI thread while the worker thread reads
        self.mode: under CPython's GIL, attribute assignment is atomic.

        Args:
            mode:  New BaseVisionMode instance.
            index: Corresponding index in all_modes.
        """
        self._mode_index = index
        self._mode       = mode

    @property
    def clip_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Screen-coordinate clip rect (x1,y1,x2,y2) or None for full-screen."""
        return self._clip_rect

    @clip_rect.setter
    def clip_rect(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        self._clip_rect = rect

    @property
    def tracked_hwnd(self) -> Optional[int]:
        """HWND of the associated window, or None for a free overlay."""
        return self._tracked_hwnd

    @tracked_hwnd.setter
    def tracked_hwnd(self, hwnd: Optional[int]) -> None:
        self._tracked_hwnd = hwnd

    def cycle_mode(self) -> None:
        """Advances to the next mode in all_modes, wrapping at the end."""
        next_idx         = (self._mode_index + 1) % len(self._all_modes)
        self._mode_index = next_idx
        self._mode       = self._all_modes[next_idx]
        print(f"  ► Overlay {self.overlay_id + 1}: {self._mode.name}")

    # ── Frame submission (worker thread) ─────────────────────────────────

    def submit_frame(self, q_img: QImage) -> None:
        """
        Submits a fully-processed frame for display.

        Called from the FrameWorker thread.  Three early-exit guards:

        1. _busy=True  — previous frame not yet painted; drop to prevent
                          Qt-event-queue build-up (backpressure).
        2. not isVisible() — overlay is hidden (e.g. after hide_all() during
                              shutdown).  Stopping emission here prevents the
                              worker from flooding the queue after ESC, which
                              would cause Windows to declare the HWND "Not
                              Responding" and replace it with a ghost window
                              that lacks WS_EX_TRANSPARENT (blocking clicks).

        The emit() posts to the GUI event queue via the forced QueuedConnection
        established in __init__.
        """
        if self._busy or not self.isVisible():
            return
        self._busy = True
        self._frame_signal.emit(q_img)

    # ── Qt slots & events (GUI thread) ────────────────────────────────────

    @Slot(QImage)
    def _on_frame(self, q_img: QImage) -> None:
        """
        Receives a processed QImage from the worker and schedules a repaint.

        QPixmap construction must happen here (GUI thread) — QPixmap is not
        thread-safe.
        """
        self._pixmap = QPixmap.fromImage(q_img)
        self.update()   # Posts a paint event; does not block

    def paintEvent(self, event) -> None:
        """
        Draws the current frame and releases the per-overlay backpressure lock.

        When clip_rect is set (windowed overlay), the painter is clipped to
        that screen region before drawing so the rest of the overlay remains
        fully transparent.  The pixmap covers the full primary screen at 1:1
        scale, so clip rect coordinates align directly with pixmap coordinates.
        """
        if self._pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._clip_rect is not None:
            x1, y1, x2, y2 = self._clip_rect
            # Desktop coords == widget coords because the overlay is positioned
            # at the primary screen's origin (0, 0).
            painter.setClipRect(x1, y1, x2 - x1, y2 - y1)
        painter.drawPixmap(self.rect(), self._pixmap, self._pixmap.rect())
        painter.end()
        self._busy = False  # Worker may now submit the next frame

    # ── Window configuration ──────────────────────────────────────────────

    def _setup_window(self) -> None:
        """Applies Qt window flags and fullscreen geometry."""
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                      # Off the Alt-Tab list
            | Qt.WindowType.WindowTransparentForInput # Qt-level click-through
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setGeometry(QApplication.primaryScreen().geometry())

    # ── Native Win32 hardening ────────────────────────────────────────────

    def showEvent(self, event) -> None:
        """Applies Win32 extended styles once the native HWND exists."""
        super().showEvent(event)
        _ = self.winId()   # Force HWND creation if Qt deferred it
        self._apply_win32_clickthrough()
        self._apply_capture_exclusion()

    def _apply_win32_clickthrough(self) -> None:
        """
        Sets WS_EX_LAYERED | WS_EX_TRANSPARENT on the native HWND.

        WS_EX_TRANSPARENT instructs Win32 hit-testing to return HTTRANSPARENT
        so all mouse messages are forwarded to the window below before they
        ever reach the Qt event loop.  SetWindowPos with SWP_FRAMECHANGED
        flushes the style change immediately.
        """
        try:
            hwnd = int(self.winId())
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd,
                _GWL_EXSTYLE,
                ex_style | _WS_EX_LAYERED | _WS_EX_TRANSPARENT,
            )
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
            )
        except Exception as exc:
            print(f"[OverlayWindow] Win32 click-through error: {exc}")

    def _apply_capture_exclusion(self) -> None:
        """
        Hides this window from all screen-capture APIs.

        Without this, MSS would capture the overlay in every frame, causing
        recursive visual feedback where the filter is applied to its own output.
        Requires Windows 10 >= 2004 (19041); fails silently on older systems.
        """
        try:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, _WDA_EXCLUDEFROMCAP)
        except Exception:
            pass
