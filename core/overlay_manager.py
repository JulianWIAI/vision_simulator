"""
Overlay Manager — Phase 4

Owns the ordered list of active OverlayWindow instances and the shared mode
list.  On each captured frame the worker calls distribute(), which applies
each overlay's individual vision mode and submits the result for display.

Phase 4 additions
─────────────────
  overlays_changed signal  — emitted whenever the overlay list changes so
                             the ControlPanel can update its Regions view
                             without polling.
  add_windowed_overlay()   — creates an overlay clipped to a window's rect,
                             with a tracked HWND for position updates.
  remove_overlay_by_id()   — remove any overlay by its numeric ID.
  set_mode_for_overlay()   — set mode on a specific overlay (not just last).
  get_overlay_infos()      — snapshot for UI display (no Qt objects exposed).
  update_window_rects()    — called by ControlPanel's 100 ms timer to refresh
                             clip_rects as tracked windows move on screen.
  show_all()               — counterpart to hide_all() for the Settings toggle.
  hud_enabled property     — lets Settings view hide the HUD bar.

Thread model
────────────
  Worker thread  →  distribute()                  — read-only snapshot, no Qt calls
  GUI thread     →  add_overlay()                 — creates and shows a QWidget
                    add_windowed_overlay()         — creates clipped QWidget
                    remove_last()                  — hides and pops last overlay
                    remove_overlay_by_id()         — hides and pops by ID
                    cycle_mode_for_last()          — advances mode on last overlay
                    set_mode_for_last(index)       — sets mode on last overlay
                    set_mode_for_overlay(id, idx)  — sets mode on specific overlay
                    hide_all() / show_all()        — connected to aboutToQuit / Settings
                    update_window_rects()          — clip-rect refresh (100 ms timer)

All methods that create, destroy, or show/hide widgets MUST run in the GUI
thread.  Register them via QTimer.singleShot(0, ...) from keyboard callbacks.
"""

from __future__ import annotations

import threading
import numpy as np
import cv2
from typing import List, Optional, Tuple, TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from utils.config import HUD_FONT, HUD_FONT_SCALE, HUD_COLOR, HUD_HEIGHT
from utils.image_utils import frame_to_qimage
from utils.mode_registry import get_all_modes
from modes.base_mode import BaseVisionMode
from core.overlay_window import OverlayWindow

from core.split_screen_manager import SplitScreenManager, LAYOUT_DISPLAY_NAMES

if TYPE_CHECKING:
    from utils.window_manager import WindowManager


class OverlayManager(QObject):
    """
    Manages a dynamic list of OverlayWindow instances.

    Key invariants:
    - _overlays is only mutated in the GUI thread (add/remove).
    - distribute() runs in the worker thread; it takes a brief lock-protected
      snapshot so list mutations during iteration are safe.
    - Processing (mode.apply, HUD, QImage conversion) happens unlocked so a
      slow mode on one overlay does not stall the others.
    - overlays_changed is emitted from the GUI thread only.
    """

    # Emitted (GUI thread) whenever the overlay list grows or shrinks.
    overlays_changed = Signal()

    def __init__(self, parent: QObject = None) -> None:
        super().__init__(parent)
        self._modes: List[BaseVisionMode] = get_all_modes()
        self._overlays: List[OverlayWindow] = []
        self._lock = threading.Lock()
        self._next_id: int = 0
        self._hud_enabled: bool = True
        self._split_screen = SplitScreenManager()
        # Tracks overlay IDs hidden because their tracked window was minimized
        self._minimized_overlay_ids: set = set()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def modes(self) -> List[BaseVisionMode]:
        """The shared list of all available vision modes."""
        return self._modes

    @property
    def hud_enabled(self) -> bool:
        """Whether the mode-name HUD bar is rendered on each frame."""
        return self._hud_enabled

    @hud_enabled.setter
    def hud_enabled(self, value: bool) -> None:
        self._hud_enabled = value

    @property
    def split_screen(self) -> SplitScreenManager:
        """The split-screen composition engine.  Configure from the GUI thread."""
        return self._split_screen

    # ── Overlay lifecycle (GUI thread only) ───────────────────────────────

    def add_overlay(self, mode_index: int = 0) -> None:
        """
        Creates a new OverlayWindow with the given mode and shows it.

        Must be called from the GUI thread (creates a QWidget).

        ── Phase 6 patch ──────────────────────────────────────────────────────
        When split-screen is active the composed frame renders into snapshot[0].
        A newly added overlay would be placed on top (later-created windows sit
        higher in the Win32 z-order for WS_EX_TOPMOST windows) and would cover
        the composed split-screen frame with a stale or blank layer.  We hide
        the new overlay immediately when split-screen is active so it does not
        interfere; it becomes visible again when sync_split_screen_windows() is
        called after the user disables split-screen.

        Args:
            mode_index: Index into self._modes for the initial vision mode.
        """
        # Clamp the requested mode index so it never exceeds the mode list length.
        mode_index = mode_index % len(self._modes)  # safe wrap-around

        # Assign a unique integer ID and immediately advance the counter so the
        # next overlay gets a different ID even if this one is removed quickly.
        overlay_id = self._next_id                   # capture current counter value
        self._next_id += 1                           # increment for the next overlay

        # Construct the OverlayWindow.  The widget is created but not yet shown.
        # _setup_window() inside OverlayWindow.__init__ sets the Qt flags and
        # an initial geometry; showEvent() will refine the geometry at display time.
        overlay = OverlayWindow(
            overlay_id=overlay_id,
            mode=self._modes[mode_index],            # look up the requested base mode
            mode_index=mode_index,                   # numeric index for HUD display
            all_modes=self._modes,                   # shared reference; never mutated
        )

        # ── Split-screen visibility guard ──────────────────────────────────
        # If split-screen is currently active, hide this new overlay immediately
        # after creation so it does not occlude the composed split-screen frame.
        # The overlay is still registered in _overlays so that when split-screen
        # is disabled, sync_split_screen_windows() can restore it.
        if self._split_screen.is_active:
            # Do NOT call show() — the overlay must remain invisible while
            # the primary overlay (snapshot[0]) is rendering the split frame.
            pass                                     # stay hidden during split-screen
        else:
            # Normal path: show the overlay immediately so it renders live frames.
            overlay.show()                           # make visible — triggers showEvent()

        # Append to the shared list under lock so distribute() (worker thread)
        # sees a consistent snapshot.  The lock is held for the minimal time.
        with self._lock:
            self._overlays.append(overlay)           # register in the managed list

        # Notify the ControlPanel's Regions view that the list changed.
        # This signal is emitted from the GUI thread (we are in a slot or timer
        # callback), so it is safe to connect to GUI-thread slots directly.
        self.overlays_changed.emit()                 # signal → ControlPanel._populate_regions_list

        # Console feedback for development / debugging.
        print(f"  + Overlay {overlay_id + 1}: {self._modes[mode_index].name}")

    def remove_last(self) -> None:
        """
        Hides and removes the most recently added overlay.

        Must be called from the GUI thread.  Does nothing if no overlays exist.
        """
        with self._lock:
            if not self._overlays:
                return
            overlay = self._overlays.pop()

        overlay.hide()
        self.overlays_changed.emit()
        print(f"  - Overlay {overlay.overlay_id + 1} removed")

    def cycle_mode_for_last(self) -> None:
        """Advances the vision mode on the most recently added overlay."""
        with self._lock:
            if not self._overlays:
                return
            overlay = self._overlays[-1]
        overlay.cycle_mode()

    def set_mode_for_last(self, mode_index: int) -> None:
        """
        Sets a specific mode index on the most recently added overlay.

        Args:
            mode_index: Zero-based index into self._modes.
        """
        if not 0 <= mode_index < len(self._modes):
            return
        with self._lock:
            if not self._overlays:
                return
            overlay = self._overlays[-1]
        overlay.set_mode(self._modes[mode_index], mode_index)
        print(f"  ► Overlay {overlay.overlay_id + 1}: {self._modes[mode_index].name}")

    def hide_all(self) -> None:
        """
        Hides every active overlay.

        Connected to app.aboutToQuit so all windows disappear before cleanup
        runs, preventing the Windows 'Not Responding' dialog.
        """
        with self._lock:
            snapshot = list(self._overlays)
        for overlay in snapshot:
            overlay.hide()

    def show_all(self) -> None:
        """
        Re-shows every active overlay.

        Counterpart to hide_all(), used by the Settings view visibility toggle.
        Must be called from the GUI thread.
        """
        with self._lock:
            snapshot = list(self._overlays)
        for overlay in snapshot:
            overlay.show()

    def sync_split_screen_windows(self) -> None:
        """
        Reconciles overlay visibility with the current split-screen state.

        ── Why this method exists ──────────────────────────────────────────────
        Win32 assigns z-order by creation sequence for WS_EX_TOPMOST windows:
        the overlay created last sits highest on screen.  When split-screen is
        active, distribute() composes all panels into one frame and delivers it
        to snapshot[0] (the first / lowest overlay).  If snapshot[1] or higher
        overlays remain visible they paint stale single-mode frames on top of the
        composed result, making the split-screen appear to show only one panel.

        ── What this method does ───────────────────────────────────────────────
        Split-screen active:
          • snapshot[0].show()  — primary surface must remain visible so the
                                  worker can call submit_frame() and trigger
                                  Qt.paintEvent on it.
          • snapshot[1:].hide() — all higher-z overlays are hidden so they
                                  cannot occlude the composed frame below them.

        Split-screen inactive (disabled or "none"):
          • All overlays are shown so each renders its individual vision mode.

        ── When to call it ─────────────────────────────────────────────────────
        Call immediately after writing ss.layout_mode in _on_apply_split_screen()
        so the visibility change takes effect on the very next frame cycle.

        Must be called from the GUI thread (hide()/show() are Qt GUI operations).
        """
        # Take a GIL-safe snapshot of the overlay list so we are not holding
        # _lock while calling hide()/show() — those are Qt GUI calls and must
        # not run while the worker thread is inside distribute().
        with self._lock:
            snapshot = list(self._overlays)  # shallow copy; safe cross-thread read

        # Nothing to do if there are no managed overlays.
        if not snapshot:
            return

        if self._split_screen.is_active:
            # ── Split-screen is active ──────────────────────────────────────
            # The primary overlay (index 0) carries the composed frame; it must
            # be visible so its Qt.paintEvent fires and the image reaches the screen.
            snapshot[0].show()                  # primary surface — must be visible

            # All overlays at index 1 and above have a higher Win32 z-order than
            # snapshot[0].  Hiding them removes their stale single-mode windows
            # from the screen so only the composed split frame is visible.
            for extra_overlay in snapshot[1:]:  # iterate every secondary overlay
                extra_overlay.hide()            # remove from screen — does not destroy
        else:
            # ── Split-screen is inactive ────────────────────────────────────
            # Restore every overlay to its normal visible state so each one
            # independently renders and displays its own vision-mode frame.
            for overlay in snapshot:            # every managed overlay
                overlay.show()                  # make visible — triggers showEvent()

    def count(self) -> int:
        """Returns the number of currently active overlays."""
        with self._lock:
            return len(self._overlays)

    # ── Phase 4: windowed overlays ────────────────────────────────────────

    def add_region_overlay(
        self,
        rect: Tuple[int, int, int, int],
        mode_index: int = 0,
    ) -> None:
        """
        Creates an overlay clipped to a static drawn rectangle.

        Unlike add_windowed_overlay(), the clip rect is fixed — it does not
        track any window HWND and never updates as windows move.

        Args:
            rect:       (x1, y1, x2, y2) in desktop coordinates.
            mode_index: Index into self._modes for the initial vision mode.
        """
        mode_index = mode_index % len(self._modes)
        overlay_id = self._next_id
        self._next_id += 1

        overlay = OverlayWindow(
            overlay_id=overlay_id,
            mode=self._modes[mode_index],
            mode_index=mode_index,
            all_modes=self._modes,
        )
        overlay.clip_rect    = rect
        overlay.tracked_hwnd = None        # static region — no window tracking
        overlay.show()

        with self._lock:
            self._overlays.append(overlay)

        self.overlays_changed.emit()
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        print(
            f"  + Region Overlay {overlay_id + 1}: "
            f"{w}×{h} px  →  {self._modes[mode_index].name}"
        )

    def add_windowed_overlay(
        self,
        hwnd: int,
        title: str,
        initial_rect: Tuple[int, int, int, int],
        mode_index: int = 0,
    ) -> None:
        """
        Creates an overlay clipped to a specific window's screen rectangle.

        The overlay renders the full-screen vision filter but is only visible
        within initial_rect.  ControlPanel's 100 ms timer calls
        update_window_rects() to keep the clip rect current as the window moves.

        Args:
            hwnd:         Native HWND of the target window.
            title:        Display title for console logging.
            initial_rect: (x1, y1, x2, y2) in desktop coordinates.
            mode_index:   Index into self._modes for the initial vision mode.
        """
        mode_index = mode_index % len(self._modes)
        overlay_id = self._next_id
        self._next_id += 1

        overlay = OverlayWindow(
            overlay_id=overlay_id,
            mode=self._modes[mode_index],
            mode_index=mode_index,
            all_modes=self._modes,
        )
        overlay.clip_rect    = initial_rect
        overlay.tracked_hwnd = hwnd
        overlay.show()

        with self._lock:
            self._overlays.append(overlay)

        self.overlays_changed.emit()
        print(
            f"  + Windowed Overlay {overlay_id + 1}: "
            f"{title!r} → {self._modes[mode_index].name}"
        )

    def remove_overlay_by_id(self, overlay_id: int) -> None:
        """
        Hides and removes a specific overlay by its numeric ID.

        Must be called from the GUI thread.  No-ops silently if the ID is
        not found (e.g. already removed via keyboard hotkey).
        """
        target: Optional[OverlayWindow] = None
        with self._lock:
            for i, ov in enumerate(self._overlays):
                if ov.overlay_id == overlay_id:
                    target = self._overlays.pop(i)
                    break

        if target is not None:
            target.hide()
            self.overlays_changed.emit()
            print(f"  - Overlay {target.overlay_id + 1} removed")

    def set_mode_for_overlay(self, overlay_id: int, mode_index: int) -> None:
        """
        Sets a specific mode on a specific overlay identified by ID.

        Thread-safe: attribute assignment under the GIL is atomic.

        Args:
            overlay_id:  The numeric ID assigned at creation.
            mode_index:  Zero-based index into self._modes.
        """
        if not 0 <= mode_index < len(self._modes):
            return
        with self._lock:
            target = next(
                (ov for ov in self._overlays if ov.overlay_id == overlay_id),
                None,
            )
        if target is not None:
            target.set_mode(self._modes[mode_index], mode_index)
            self.overlays_changed.emit()

    def set_custom_mode_for_overlay(
        self, overlay_id: int, mode: BaseVisionMode
    ) -> None:
        """
        Sets any BaseVisionMode instance (e.g. VisionPipeline) on a specific overlay.

        Unlike set_mode_for_overlay(), accepts an arbitrary mode rather than a
        registry index.  Stores mode_index=-1 so _draw_hud() knows to show the
        pipeline indicator ("≡") instead of a numbered counter.

        Thread-safe: single attribute assignment under the GIL.

        Args:
            overlay_id: The numeric ID assigned at creation.
            mode:       Any BaseVisionMode instance, including VisionPipeline.
        """
        with self._lock:
            target = next(
                (ov for ov in self._overlays if ov.overlay_id == overlay_id),
                None,
            )
        if target is not None:
            target.set_mode(mode, -1)
            self.overlays_changed.emit()

    def get_overlay_infos(self) -> List[dict]:
        """
        Returns a snapshot of current overlays as plain dicts for UI display.

        Safe to call from the GUI thread.  Never exposes QWidget references
        so the ControlPanel cannot accidentally call Qt methods off-thread.

        Returns:
            List of {"id", "mode_name", "mode_index", "tracked_hwnd", "has_clip"}.
        """
        with self._lock:
            return [
                {
                    "id":           ov.overlay_id,
                    "mode_name":    ov.mode.name,
                    "mode_index":   ov.mode_index,
                    "tracked_hwnd": ov.tracked_hwnd,
                    "has_clip":     ov.clip_rect is not None,
                    # True when the overlay is a drawn region (clip rect set, no HWND)
                    "is_region":    ov.clip_rect is not None and ov.tracked_hwnd is None,
                }
                for ov in self._overlays
            ]

    def update_window_rects(self, window_manager: "WindowManager") -> None:
        """
        Refreshes clip_rects for all window-tracked overlays.

        Called by ControlPanel's 100 ms QTimer — always on the GUI thread.
        Writes overlay.clip_rect which paintEvent reads; the write is safe
        under the GIL (single attribute assignment is atomic in CPython).

        Minimized windows: the overlay is hidden to prevent a ghost tile
        showing on the desktop.  It is re-shown when the window is restored.
        _minimized_overlay_ids tracks which overlays we hid so that we only
        re-show overlays that were hidden by this path (not by hide_all()).

        If a tracked window is no longer valid, the clip is cleared so the
        overlay becomes full-screen rather than disappearing entirely.
        """
        with self._lock:
            tracked = [
                (ov, ov.tracked_hwnd)
                for ov in self._overlays
                if ov.tracked_hwnd is not None
            ]

        for overlay, hwnd in tracked:
            if window_manager.is_minimized(hwnd):
                if overlay.overlay_id not in self._minimized_overlay_ids:
                    self._minimized_overlay_ids.add(overlay.overlay_id)
                    overlay.hide()
                continue

            # Window is not minimized — restore if we were the ones who hid it
            if overlay.overlay_id in self._minimized_overlay_ids:
                self._minimized_overlay_ids.discard(overlay.overlay_id)
                overlay.show()

            new_rect = window_manager.get_rect(hwnd)
            if new_rect is not None:
                overlay.clip_rect = new_rect
            elif not window_manager.is_valid(hwnd):
                # Window closed — fall back to full-screen rather than crashing
                overlay.clip_rect    = None
                overlay.tracked_hwnd = None
                self._minimized_overlay_ids.discard(overlay.overlay_id)

    # ── Frame distribution (worker thread) ────────────────────────────────

    def distribute(self, raw_frame: np.ndarray) -> None:
        """
        Applies each overlay's vision mode to raw_frame and submits the result.

        Called from the FrameWorker thread every capture cycle.

        The lock is held only for the snapshot copy (O(n), very brief).
        All heavy work — mode.apply(), HUD, QImage conversion — runs unlocked
        so a slow mode cannot stall frame delivery to other overlays.

        Frames are dropped per-overlay if that overlay's previous frame has
        not yet been painted (_busy=True), preventing queue build-up.

        Args:
            raw_frame: Raw BGR capture from ScreenCapture.
        """
        with self._lock:
            snapshot = list(self._overlays)

        if not snapshot:
            return

        # ── Split-screen bypass ────────────────────────────────────────────
        # When a split layout is active the composed frame replaces individual
        # per-overlay rendering.  All resize/stitch work happens inside
        # SplitScreenManager.compose() using NumPy/OpenCV — no Qt calls, no
        # Python pixel loops.  Only the first active overlay is used as the
        # display surface; the composed frame already fills the full screen.
        if self._split_screen.is_active:
            try:
                composed = self._split_screen.compose(raw_frame)
                if self._hud_enabled:
                    composed = self._draw_split_hud(composed)
                snapshot[0].submit_frame(frame_to_qimage(composed))
            except Exception as exc:
                print(f"[SplitScreen] Composition error: {exc}")
            return

        # ── Standard per-overlay pipeline ─────────────────────────────────
        for overlay in snapshot:
            try:
                processed = overlay.mode.apply(raw_frame)
                annotated = (
                    self._draw_hud(processed, overlay)
                    if self._hud_enabled
                    else processed
                )
                q_img = frame_to_qimage(annotated)
                overlay.submit_frame(q_img)
            except Exception as exc:
                print(f"[OverlayManager] Overlay {overlay.overlay_id + 1} error: {exc}")

    # ── HUD rendering ─────────────────────────────────────────────────────

    def _draw_split_hud(self, frame: np.ndarray) -> np.ndarray:
        """
        Composites a minimal HUD bar for split-screen mode.

        Called from distribute() (worker thread) when split-screen is active.
        The per-panel mode labels are already embedded by SplitScreenManager,
        so this bar shows only the layout name and the standard control hints.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        layout_name = LAYOUT_DISPLAY_NAMES.get(
            self._split_screen.layout_mode, "Split Screen"
        )

        strip = out.copy()
        cv2.rectangle(strip, (0, h - HUD_HEIGHT), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(strip, 0.55, out, 0.45, 0, out)

        label = f"[SPLIT]  {layout_name}"
        cv2.putText(
            out, label,
            (16, h - HUD_HEIGHT + 24),
            HUD_FONT, HUD_FONT_SCALE, HUD_COLOR, 1, cv2.LINE_AA,
        )

        hint = "N: new   M: cycle   X: remove   ESC: quit"
        (tw, _), _ = cv2.getTextSize(hint, HUD_FONT, HUD_FONT_SCALE * 0.75, 1)
        cv2.putText(
            out, hint,
            (w - tw - 16, h - 10),
            HUD_FONT, HUD_FONT_SCALE * 0.75, (140, 140, 140), 1, cv2.LINE_AA,
        )
        return out

    def _draw_hud(self, frame: np.ndarray, overlay: OverlayWindow) -> np.ndarray:
        """
        Composites the mode-name and control-hint bar at the bottom of frame.

        Args:
            frame:   Processed BGR frame (already has mode applied).
            overlay: The OverlayWindow whose metadata is displayed.

        Returns:
            New BGR array with the HUD bar composited at the bottom.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        mode_name = overlay.mode.name
        mode_idx  = overlay.mode_index    # -1 for custom/pipeline modes
        n         = overlay.mode_count
        ovl_num   = overlay.overlay_id + 1  # 1-based for display

        # Semi-transparent dark strip
        strip = out.copy()
        cv2.rectangle(strip, (0, h - HUD_HEIGHT), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(strip, 0.55, out, 0.45, 0, out)

        # Left: overlay ID + mode info + name
        # mode_idx == -1 means a custom/pipeline mode not in the registry
        if mode_idx == -1:
            label = f"[OVL {ovl_num}]  ≡  {mode_name}"
        else:
            label = f"[OVL {ovl_num}]  Mode [{mode_idx + 1}/{n}]:  {mode_name}"
        cv2.putText(
            out, label,
            (16, h - HUD_HEIGHT + 24),
            HUD_FONT, HUD_FONT_SCALE, HUD_COLOR, 1, cv2.LINE_AA,
        )

        # Right: control hints (right-aligned)
        hint = "N: new   M: cycle   X: remove   ESC: quit"
        (tw, _), _ = cv2.getTextSize(hint, HUD_FONT, HUD_FONT_SCALE * 0.75, 1)
        cv2.putText(
            out, hint,
            (w - tw - 16, h - 10),
            HUD_FONT, HUD_FONT_SCALE * 0.75, (140, 140, 140), 1, cv2.LINE_AA,
        )

        return out
