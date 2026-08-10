"""
platform/base.py

Abstract base classes that define the cross-platform interface used by the
rest of the application.  No OS-specific imports live here — only the
abstract contracts and any purely algorithmic logic that is identical on
all platforms (e.g. the inverse-scale coordinate math for mouse remapping).

Two abstract classes are defined:

  AbstractPlatform
    Window-level operations: apply styles, exclude from capture, enumerate
    open windows, query window geometry and state.

  AbstractMouseRemapper
    Lifecycle (start / stop) for the split-screen click-remapping hook.
    The coordinate maths (_map_coords) is a shared concrete method here
    because it involves no OS calls — only integer arithmetic that is
    identical on every platform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TYPE_CHECKING

# TYPE_CHECKING guard keeps this import annotation-only at runtime,
# preventing a circular dependency between platform.base and PySide6.
if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


# ── AbstractPlatform ───────────────────────────────────────────────────────────

class AbstractPlatform(ABC):
    """
    OS-level window and system operations.

    All methods that touch OS APIs are abstract; callers depend only on
    this interface, never on a concrete implementation.
    """

    @abstractmethod
    def apply_overlay_styles(self, widget: "QWidget") -> None:
        """
        Hardens an overlay window so it is:
          • Always on top of all other windows.
          • Fully click-through (mouse/keyboard input falls to the desktop).
          • Hidden from the taskbar / Dock / Alt-Tab / Mission Control.
          • Not activatable (cannot steal focus).

        Called from OverlayWindow.showEvent() after the native window handle
        has been created by Qt.

        Args:
            widget: The QWidget whose native window handle should be configured.
        """
        ...

    @abstractmethod
    def exclude_from_capture(self, widget: "QWidget") -> None:
        """
        Hides the window from all screen-capture APIs so it does not appear
        in the frames captured by MSS / BitBlt / CGDisplayCreateImage.

        Without this the overlay would appear in its own captured frame,
        causing recursive visual feedback.

        Args:
            widget: The QWidget to exclude.
        """
        ...

    @abstractmethod
    def apply_drawer_styles(self, widget: "QWidget") -> None:
        """
        Configures a drawing-canvas window: hidden from taskbar / Dock but
        still receives mouse events (unlike overlay windows which are
        click-through).

        Called from RegionDrawer.showEvent().

        Args:
            widget: The QWidget whose native window handle should be configured.
        """
        ...

    @abstractmethod
    def get_screen_size(self) -> Tuple[int, int]:
        """
        Returns the primary screen's physical pixel dimensions.

        'Physical' here means the actual pixel count reported by the OS,
        not the DPI-scaled logical size.  This matches the frame dimensions
        produced by MSS (which captures at physical resolution).

        Returns:
            (width, height) in physical pixels.
        """
        ...

    @abstractmethod
    def list_windows(self) -> List[dict]:
        """
        Enumerates visible, titled, non-zero-area top-level windows.

        Each entry follows the data contract:
            {"id": int, "title": str, "rect": (x1, y1, x2, y2)}

        where rect is in desktop/screen coordinates (physical pixels on
        Windows; logical points on macOS).

        Returns:
            List of window info dicts (may be empty if no windows found).
        """
        ...

    @abstractmethod
    def get_window_rect(self, window_id: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns the live on-screen rect of a specific window.

        Queries the OS directly (not from any cache) so callers can track
        position changes at sub-second intervals.

        Args:
            window_id: The "id" value from a list_windows() entry.

        Returns:
            (x1, y1, x2, y2) in screen coordinates, or None if the window
            no longer exists or the query failed.
        """
        ...

    @abstractmethod
    def is_window_valid(self, window_id: int) -> bool:
        """
        Returns True if the window still exists and is visible.

        Args:
            window_id: The "id" value from a list_windows() entry.
        """
        ...

    @abstractmethod
    def is_window_minimized(self, window_id: int) -> bool:
        """
        Returns True if the window is currently minimized / iconified.

        Args:
            window_id: The "id" value from a list_windows() entry.
        """
        ...

    @abstractmethod
    def create_mouse_remapper(self, split_screen) -> "AbstractMouseRemapper":
        """
        Factory: creates the platform-specific mouse remapper.

        Args:
            split_screen: SplitScreenManager instance (read-only; remapper
                          reads .is_active and .layout_mode from it).

        Returns:
            An AbstractMouseRemapper ready to be started.
        """
        ...


# ── AbstractMouseRemapper ──────────────────────────────────────────────────────

class AbstractMouseRemapper(ABC):
    """
    Intercepts physical mouse button events during split-screen mode and
    remaps their coordinates so that clicking at a visual panel position
    sends the click to the correct real-desktop position.

    Sub-classes implement start() / stop() using OS-specific hook APIs.
    The coordinate maths is OS-agnostic and lives here as a shared concrete
    method (_map_coords).
    """

    def __init__(self, split_screen, screen_w: int, screen_h: int) -> None:
        """
        Args:
            split_screen: Shared SplitScreenManager (read-only).
            screen_w:     Primary screen width  in physical pixels.
            screen_h:     Primary screen height in physical pixels.
        """
        self._split_screen = split_screen   # shared reference — never written by remapper
        self._screen_w = screen_w           # physical screen width
        self._screen_h = screen_h           # physical screen height

    @abstractmethod
    def start(self) -> None:
        """Installs the OS hook and starts the event-capture thread."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Removes the OS hook and waits for the thread to exit cleanly."""
        ...

    # ── Shared coordinate transform ────────────────────────────────────────────

    def _map_coords(self, cx: int, cy: int) -> Tuple[int, int]:
        """
        Converts a cursor position in the composed split-screen overlay frame
        (panel space) back to the corresponding position in real desktop space.

        This is the exact inverse of SplitScreenManager._process_panel()'s
        cv2.resize call.  The panel dimension formulas (h_top, h_bot, w_left,
        w_right) are byte-for-byte identical to those in split_screen_manager.py
        to guarantee pixel-perfect consistency — any drift would reintroduce
        a click-position offset.

        The math applies only integer arithmetic and floating-point scale
        factors; it contains zero OS API calls and is therefore identical on
        all platforms.

        Args:
            cx: Cursor x in overlay / composed frame pixels.
            cy: Cursor y in overlay / composed frame pixels.

        Returns:
            (mapped_x, mapped_y) in real desktop pixels, clamped to screen
            bounds.  Returns (cx, cy) unchanged when layout is "none" or
            unrecognised.
        """
        W: int = self._screen_w     # primary screen width  in physical pixels
        H: int = self._screen_h     # primary screen height in physical pixels

        # Single GIL-atomic str attribute read — no lock required.
        layout: str = self._split_screen.layout_mode  # e.g. "2x_horizontal"

        # ── Layout: Top / Bottom (2x_horizontal) ──────────────────────────
        if layout == "2x_horizontal":
            # The compose step shrinks H rows to h_top (top panel) or h_bot
            # (bottom panel).  The inverse stretches back to H rows.
            h_top = H - H // 2         # ceiling half — MUST match compose() exactly
            h_bot = H // 2             # floor   half — MUST match compose() exactly

            mapped_x = cx              # x axis is not scaled in this layout

            if cy < h_top:
                # Top panel: panel_y → real_y = panel_y × (H / h_top)
                mapped_y = int(cy * H / h_top)
            else:
                # Bottom panel: local_y within the bottom panel, then scale
                local_y  = cy - h_top
                mapped_y = int(local_y * H / h_bot)

        # ── Layout: Left / Right (2x_vertical) ────────────────────────────
        elif layout == "2x_vertical":
            w_left  = W - W // 2       # ceiling half — MUST match compose() exactly
            w_right = W // 2           # floor   half — MUST match compose() exactly

            mapped_y = cy              # y axis is not scaled in this layout

            if cx < w_left:
                # Left panel: panel_x → real_x = panel_x × (W / w_left)
                mapped_x = int(cx * W / w_left)
            else:
                # Right panel: local_x within the right panel, then scale
                local_x  = cx - w_left
                mapped_x = int(local_x * W / w_right)

        # ── Layout: 2×2 Grid (4x_grid) ────────────────────────────────────
        elif layout == "4x_grid":
            h_top   = H - H // 2       # ceiling half height
            h_bot   = H // 2           # floor   half height
            w_left  = W - W // 2       # ceiling half width
            w_right = W // 2           # floor   half width

            if cy < h_top:
                # Upper row — vertical inverse: real_y = cy × (H / h_top)
                mapped_y = int(cy * H / h_top)
                if cx < w_left:
                    # Top-Left quadrant
                    mapped_x = int(cx * W / w_left)
                else:
                    # Top-Right quadrant
                    local_x  = cx - w_left
                    mapped_x = int(local_x * W / w_right)
            else:
                # Lower row — vertical inverse: real_y = local_y × (H / h_bot)
                local_y  = cy - h_top
                mapped_y = int(local_y * H / h_bot)
                if cx < w_left:
                    # Bottom-Left quadrant
                    mapped_x = int(cx * W / w_left)
                else:
                    # Bottom-Right quadrant
                    local_x  = cx - w_left
                    mapped_x = int(local_x * W / w_right)

        else:
            # "none" or unknown layout — no remapping needed.
            return cx, cy

        # ── Clamp to valid screen bounds ──────────────────────────────────
        # Floating-point division can produce a value == W or == H at edge
        # pixels; clamping keeps SetCursorPos / CGEventSetLocation within range.
        mapped_x = max(0, min(mapped_x, W - 1))
        mapped_y = max(0, min(mapped_y, H - 1))

        return mapped_x, mapped_y
