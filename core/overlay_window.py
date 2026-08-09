"""
PySide6 Fullscreen Transparent Overlay Window

Each OverlayWindow owns one vision mode and renders independently.
Multiple instances can coexist, each showing a different vision filter
over the full screen simultaneously.

Cross-platform design
─────────────────────
All OS-specific window hardening is delegated to the platform abstraction
layer (platform/).  This file contains only Qt and business logic.

On Windows the platform layer applies:
  WS_EX_LAYERED | WS_EX_TRANSPARENT — click-through at the OS/kernel level
  WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE — hide from taskbar, prevent focus theft
  SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAP) — exclude from screen capture

On macOS the platform layer applies:
  Qt.WindowTransparentForInput — maps to NSWindow.setIgnoresMouseEvents_(YES)
  Qt.Tool — maps to NSWindowStyleMaskUtilityWindow (hides from Dock)
  NSWindow.setSharingType_(NSWindowSharingNone) — exclude from screen capture

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

Pipeline state retention
────────────────────────
cycle_mode() and set_mode() delegate to core.pipeline_state so that an
active VisionPipeline effect chain is preserved when the user presses M
or a digit hotkey to change the base vision mode.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter

from modes.base_mode import BaseVisionMode
from platform import get_platform


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

    # ── Platform hardening ────────────────────────────────────────────────

    def showEvent(self, event) -> None:
        """
        Applies platform-specific window hardening once the native handle exists.

        Geometry is re-queried at show time so any monitor or DPI change
        between __init__ and show() (multi-monitor setups, resolution changes)
        is picked up automatically.

        Platform details are handled by the AbstractPlatform implementation
        in platform/windows.py (Win32) or platform/macos.py (AppKit/Quartz).
        """
        super().showEvent(event)

        # Ensure the native window handle exists before calling winId().
        # Qt defers native window creation; winId() triggers it synchronously.
        _ = self.winId()

        # Re-read primary screen geometry at show time to pick up any
        # resolution or monitor-config changes since __init__.
        self.setGeometry(QApplication.primaryScreen().geometry())

        # Delegate OS-specific hardening to the platform abstraction layer.
        # On Windows: sets WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, etc.
        # On macOS:   Qt flags already handle click-through and Dock hiding;
        #             the platform call adds capture exclusion only.
        plat = get_platform()
        plat.apply_overlay_styles(self)   # click-through + taskbar/Dock hiding
        plat.exclude_from_capture(self)   # hide from MSS / screen-recording APIs
