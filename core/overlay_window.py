"""
PySide6 Fullscreen Transparent Overlay Window — Phase 2 / Phase 6 patch

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

Phase 6 patch — pipeline state retention
─────────────────────────────────────────
cycle_mode() and set_mode() now delegate to core.pipeline_state so that
an active VisionPipeline effect chain (e.g., HD Matrix Analyzer) is preserved
when the user presses M or a digit hotkey to change the base vision mode.

Phase 6 patch — Win32 geometry robustness
──────────────────────────────────────────
showEvent() refreshes the overlay geometry at display time so monitor
configuration changes between __init__ and show() (multi-monitor setups,
DPI changes) are picked up.  WS_EX_TOOLWINDOW and WS_EX_NOACTIVATE are now
explicitly written into the extended style to prevent the 1-cm taskbar ghost
that appeared on machines where Qt's Tool flag was not translated correctly.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
from typing import List, Optional, Tuple

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter

from modes.base_mode import BaseVisionMode

# ── Win32 extended-style constants ────────────────────────────────────────────
_GWL_EXSTYLE         = -20
_WS_EX_LAYERED       = 0x00080000   # Required companion to WS_EX_TRANSPARENT
_WS_EX_TRANSPARENT   = 0x00000020   # OS answers NCHITTEST before Qt sees it
_WDA_EXCLUDEFROMCAP  = 0x00000011   # Hide from BitBlt / screen-recording APIs
_SWP_NOMOVE          = 0x0002
_SWP_NOSIZE          = 0x0001
_SWP_NOZORDER        = 0x0004
_SWP_FRAMECHANGED    = 0x0020       # Flushes extended-style change immediately

# ── NEW Win32 constants (Phase 6 patch) ───────────────────────────────────────
# WS_EX_TOOLWINDOW: marks the window as a tool window so Windows removes it
# from the taskbar button list and the Alt-Tab switcher.  Qt.WindowType.Tool
# should set this automatically, but on some Windows builds / driver combos
# it is not reliably applied.  We set it explicitly in showEvent() to fix the
# "1-cm ghost window on taskbar" artefact reported on the second machine.
_WS_EX_TOOLWINDOW    = 0x00000080   # removes window from taskbar and Alt-Tab

# WS_EX_NOACTIVATE: prevents the overlay from stealing keyboard focus or being
# activated when the user clicks through it.  Complements WS_EX_TRANSPARENT.
_WS_EX_NOACTIVATE    = 0x08000000   # window cannot be activated by mouse/key

# WS_EX_APPWINDOW: the OPPOSITE of what we want — forces a taskbar button.
# Qt can accidentally set this flag on certain Windows versions when the window
# has no owner.  We explicitly CLEAR it in _apply_win32_clickthrough().
_WS_EX_APPWINDOW     = 0x00040000   # we must remove this flag, never set it


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
        mode:       BaseVisionMode,
        mode_index: int,
        all_modes:  List[BaseVisionMode],
    ) -> None:
        """
        Args:
            overlay_id:  Unique integer ID (0-based) assigned by OverlayManager.
            mode:        Initial vision mode to apply.
            mode_index:  Index of mode in all_modes (or -1 for pipeline modes).
            all_modes:   Shared list of all available modes (for cycling).
        """
        super().__init__()
        self.overlay_id    = overlay_id       # unique numeric ID for this window
        self._mode         = mode             # currently active mode (or VisionPipeline)
        self._mode_index   = mode_index       # registry index; -1 when pipeline is active

        # ── NEW: _base_mode_index (Phase 6 patch) ─────────────────────────
        # Tracks the NUMERICAL base mode index independently of _mode_index.
        # When a VisionPipeline is applied via the ControlPanel, _mode_index
        # is set to -1 (the pipeline sentinel used by the HUD).  If the user
        # then presses M to cycle, we would compute ((-1)+1)%n == 0, always
        # restarting from index 0.  _base_mode_index remembers where we were
        # before the pipeline was installed so cycling advances correctly.
        # Updated by cycle_mode() and set_mode() on every mode change.
        self._base_mode_index: int = max(mode_index, 0)  # clamp -1 to 0 for safety

        self._all_modes    = all_modes        # shared reference to the full mode list
        self._pixmap: Optional[QPixmap] = None   # last rendered frame as Qt pixmap
        self._busy: bool   = False            # backpressure flag (see submit_frame)

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
        """Currently active vision mode (may be a VisionPipeline)."""
        return self._mode

    @property
    def mode_index(self) -> int:
        """Zero-based index of the active mode; -1 when a pipeline is active."""
        return self._mode_index

    @property
    def mode_count(self) -> int:
        """Total number of available modes."""
        return len(self._all_modes)

    def set_mode(self, mode: BaseVisionMode, index: int) -> None:
        """
        Assigns a mode or — when a VisionPipeline is currently active —
        patches only the pipeline's base_mode reference, keeping all effects.

        ── Phase 6 patch: pipeline preservation ──────────────────────────────
        Previous behaviour: `self._mode = mode` — unconditionally replaced the
        entire mode, discarding any active VisionPipeline and its effects.

        New behaviour:
          • If `mode` IS a VisionPipeline (user applied one from the UI):
              Store it directly; the old pipeline (if any) is fully replaced.
          • If `mode` is a plain BaseVisionMode AND current _mode is a pipeline:
              Patch pipeline.base_mode = mode in-place; effects survive.
          • If `mode` is a plain BaseVisionMode AND no pipeline is active:
              Store it directly, same as before.

        Thread safety: delegate to core.pipeline_state.apply_base_mode_change()
        which performs a single GIL-atomic attribute write.

        Args:
            mode:  New BaseVisionMode instance, or a VisionPipeline.
            index: Corresponding index in all_modes (-1 for pipeline modes).
        """
        # Import the pipeline utilities at call time, not at module load time.
        # This prevents the circular import:
        #   overlay_window → pipeline_state → VisionPipeline → overlay_window
        from core.pipeline_state import apply_base_mode_change  # call-time import
        from core.vision_pipeline import VisionPipeline          # call-time import

        # ── Case 1: caller is installing a brand-new VisionPipeline ───────
        # This happens when the user clicks "Apply Pipeline" in the ControlPanel.
        # We always store it directly; a new pipeline supersedes any previous one.
        if isinstance(mode, VisionPipeline):
            self._mode            = mode              # store new pipeline directly
            self._mode_index      = index             # store index (-1 for pipelines)
            self._base_mode_index = max(index, 0)     # clamp -1 → 0 to keep tracking valid
            return                                    # done — no further action needed

        # ── Case 2: caller is setting a plain base mode ────────────────────
        # Delegate to the authoritative patch function.  It checks whether the
        # current _mode is a VisionPipeline and either patches it or replaces it.
        self._mode, self._mode_index = apply_base_mode_change(
            self._mode,   # current mode — may be VisionPipeline or a plain mode
            mode,         # the new plain base mode selected by the user
            index,        # registry index to store when no pipeline is active
        )

        # Always update _base_mode_index so the next cycle_mode() call knows
        # which position to advance from, even when _mode_index is -1.
        self._base_mode_index = index                 # track numeric base regardless of pipeline

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
        """
        Advances to the next mode in all_modes, wrapping at the end.

        ── Phase 6 patch: pipeline preservation + index tracking ─────────────
        Previous behaviour: `self._mode = self._all_modes[next_idx]` — replaced
        the mode unconditionally, discarding any active VisionPipeline.

        New behaviour:
          1. Use _base_mode_index (not _mode_index) as the cycling origin.
             _mode_index is -1 when a pipeline is active; using it directly
             would always land on index 0.  _base_mode_index tracks the last
             known numeric base position independently.
          2. Delegate to apply_base_mode_change(), which patches the pipeline's
             base_mode in-place if one is active, or replaces _mode directly.
          3. Update _base_mode_index after every cycle so each subsequent M
             press advances by one instead of always cycling from the same spot.
        """
        # Import call-time to avoid circular dependency at module level.
        from core.pipeline_state import resolve_base_index, apply_base_mode_change  # local import

        # Step 1: Convert _base_mode_index to a guaranteed-valid origin.
        # resolve_base_index() handles the -1 sentinel (pipeline mode) and any
        # out-of-range value by falling back to 0 as the safe default.
        origin = resolve_base_index(
            self._base_mode_index,   # the cached numeric base index (not -1)
            self._mode,              # current mode object (may be VisionPipeline)
            len(self._all_modes),    # total number of available modes
        )                            # returns an int in [0, len-1]

        # Step 2: Advance to the next index using modulo wrap-around.
        # If we were at index 5 of 20, this produces 6.  At index 19: wraps to 0.
        next_idx = (origin + 1) % len(self._all_modes)  # advance one step, wrap at end

        # Step 3: Fetch the new plain base mode object from the shared list.
        # This is a plain BaseVisionMode instance — never a VisionPipeline.
        new_base = self._all_modes[next_idx]             # look up the new base mode

        # Step 4: Apply the mode change, preserving any active pipeline.
        # Returns (current_pipeline_with_new_base, -1) if pipeline is active,
        # or (new_base, next_idx) if no pipeline.  Unpack both values at once.
        self._mode, self._mode_index = apply_base_mode_change(
            self._mode,    # current mode — VisionPipeline or plain BaseVisionMode
            new_base,      # new plain base mode to activate
            next_idx,      # index to store when no pipeline is currently active
        )                  # single GIL-atomic write if patching pipeline.base_mode

        # Step 5: Always update _base_mode_index to the new numerical position.
        # Even when _mode_index was set to -1 by apply_base_mode_change (pipeline
        # path), _base_mode_index must remember the numeric position so the NEXT
        # cycle call advances from next_idx, not from index 0.
        self._base_mode_index = next_idx                 # persist numeric base position

        # Console feedback: show 1-based overlay number + mode name for legibility.
        print(f"  ► Overlay {self.overlay_id + 1}: {self._mode.name}")

    # ── Frame submission (worker thread) ─────────────────────────────────

    def submit_frame(self, q_img: QImage) -> None:
        """
        Submits a fully-processed frame for display.

        Called from the FrameWorker thread.  Two early-exit guards:

        1. _busy=True  — previous frame not yet painted; drop to prevent
                          Qt-event-queue build-up (backpressure).
        2. not isVisible() — overlay is hidden (e.g. after hide_all() during
                              shutdown, or by sync_split_screen_windows()).
                              Stopping emission here prevents the worker from
                              flooding the queue after ESC.

        The emit() posts to the GUI event queue via the forced QueuedConnection
        established in __init__.
        """
        if self._busy or not self.isVisible():
            return                          # drop frame — backpressure or hidden
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
        """Applies Qt window flags; geometry is set (and re-set) in showEvent()."""
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                      # maps to WS_EX_TOOLWINDOW on Win32
            | Qt.WindowType.WindowTransparentForInput # Qt-level click-through
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Set an initial geometry so the window is not zero-sized before show().
        # showEvent() will re-query and re-set this after the HWND is created,
        # picking up any monitor-config changes (multi-monitor, DPI changes).
        self.setGeometry(QApplication.primaryScreen().geometry())  # initial best-guess

    # ── Native Win32 hardening ────────────────────────────────────────────

    def showEvent(self, event) -> None:
        """
        Applies Win32 extended styles once the native HWND exists.

        ── Phase 6 patch: geometry refresh at show time ───────────────────────
        On machines where the monitor configuration differs from the machine
        used during development (different DPI scaling, second monitor as
        primary, etc.), the geometry set in _setup_window() may be stale by
        the time show() is called.  We re-query primaryScreen().geometry() here
        to pick up the current values.

        This also fixes the edge case where QApplication.primaryScreen() was
        not yet fully initialised at __init__ time (rare but observed on some
        driver stacks).
        """
        super().showEvent(event)

        # Force HWND creation before we try to read winId().
        # Qt defers native window creation until it is actually needed;
        # calling winId() triggers it synchronously.
        _ = self.winId()                              # ensure native HWND exists now

        # ── Re-apply geometry at show time ────────────────────────────────
        # Re-query the primary screen geometry so any monitor or DPI change
        # since __init__ is reflected.  This is especially important on the
        # second computer where the primary monitor may have different dimensions
        # or scaling from the development machine.
        current_screen_geo = QApplication.primaryScreen().geometry()  # re-query geometry
        self.setGeometry(current_screen_geo)          # apply the refreshed full-screen rect

        # Apply the Win32 extended styles (click-through + taskbar hiding).
        self._apply_win32_clickthrough()
        self._apply_capture_exclusion()

    def _apply_win32_clickthrough(self) -> None:
        """
        Sets Win32 extended styles to make this window fully click-through
        and remove it from the taskbar / Alt-Tab list.

        ── Phase 6 patch ──────────────────────────────────────────────────────
        Previous version only added WS_EX_LAYERED | WS_EX_TRANSPARENT.
        Qt.WindowType.Tool should map to WS_EX_TOOLWINDOW but this translation
        is not reliable on all Windows 10/11 driver + graphics-stack combos.
        On the second test machine, overlay windows appeared as a ~1-cm ghost
        entry on the taskbar because WS_EX_TOOLWINDOW was absent.

        Fixes applied:
          1. WS_EX_TOOLWINDOW   — explicitly added to guarantee taskbar removal.
          2. WS_EX_NOACTIVATE   — window cannot steal focus or activation.
          3. WS_EX_APPWINDOW    — cleared (bitwise AND NOT) in case Qt or the
             driver accidentally set it (WS_EX_APPWINDOW forces a taskbar button).

        SetWindowPos with SWP_FRAMECHANGED flushes all style changes immediately
        without moving, resizing, or reordering the window.
        """
        try:
            # Read the HWND of this Qt widget as an integer for Win32 API calls.
            hwnd = int(self.winId())                  # integer HWND for ctypes

            # Read the current extended style word from Win32.
            # GetWindowLongW(hwnd, GWL_EXSTYLE) returns all currently-set flags.
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)

            # Build the new style: add the bits we WANT set.
            # Bitwise OR accumulates flags without touching unrelated bits.
            new_style = (
                ex_style
                | _WS_EX_LAYERED       # required for WS_EX_TRANSPARENT to work
                | _WS_EX_TRANSPARENT   # OS answers NCHITTEST with HTTRANSPARENT
                | _WS_EX_TOOLWINDOW    # hide from taskbar and Alt-Tab list
                | _WS_EX_NOACTIVATE    # prevent this window from stealing focus
            )

            # Remove WS_EX_APPWINDOW if it is set.  This flag forces a taskbar
            # button even when WS_EX_TOOLWINDOW is present; clearing it ensures
            # the window truly disappears from the taskbar.
            # Bitwise AND with the complement (~) clears only that specific bit.
            new_style = new_style & ~_WS_EX_APPWINDOW  # clear the appwindow bit

            # Write the combined style back to Win32.
            ctypes.windll.user32.SetWindowLongW(
                hwnd,
                _GWL_EXSTYLE,          # index for the extended style word
                new_style,             # combined style with all changes applied
            )

            # Flush the style change without moving/resizing the window.
            # SWP_FRAMECHANGED tells Win32 to re-send WM_NCCALCSIZE immediately
            # so the new extended styles take effect for the next NCHITTEST.
            ctypes.windll.user32.SetWindowPos(
                hwnd,
                0,                     # hWndInsertAfter — ignored with NOZORDER
                0, 0, 0, 0,            # x, y, cx, cy — all ignored
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
            )
        except Exception as exc:
            # Non-fatal: the overlay still works without the Win32 hardening;
            # click-through may be imperfect and taskbar entry may appear.
            print(f"[OverlayWindow] Win32 click-through error: {exc}")

    def _apply_capture_exclusion(self) -> None:
        """
        Hides this window from all screen-capture APIs (MSS, BitBlt, etc.).

        Without this, MSS would capture the overlay in every frame, causing
        recursive visual feedback where the filter is applied to its own output.
        Requires Windows 10 >= 2004 (build 19041); fails silently on older builds.
        """
        try:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, _WDA_EXCLUDEFROMCAP)
        except Exception:
            pass   # silently ignore — older Windows versions lack this API
