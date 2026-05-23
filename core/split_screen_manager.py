"""
Split-Screen Comparison Engine — Phase 5

Composes a full-resolution output frame from multiple vision-mode copies of
the raw input.  The critical performance insight: each panel frame is resized
DOWN to its target dimensions BEFORE mode.apply() is called, so the vision
mode operates on a ~50% (2× layouts) or ~75% (4× grid) smaller array.  For a
1920×1080 source this reduces mode processing from ~2 M pixels per panel to
~1 M (2×) or ~518 K (4×), without any Python-level pixel loops.

Layout modes
────────────
  "none"            Pass-through — split engine inactive.
  "2x_horizontal"   Two panels stacked vertically   (top  = A, bottom = B).
  "2x_vertical"     Two panels placed side-by-side  (left = A, right  = B).
  "4x_grid"         2×2 quad grid  (TL=A, TR=B, BL=C, BR=D).

Odd-dimension safety
────────────────────
For any dimension D the panel pair is split as (D − D//2, D//2) so their
sum always equals D exactly, preventing off-by-one shape mismatches in
np.vstack / np.hstack at non-standard resolutions like 2560×1079.

Thread model
────────────
  GUI thread   — writes: layout_mode (str), mode slots (list element assign).
                 Both are single attribute/element assignments, atomic under GIL.
  Worker thread — compose() reads (never writes) these attributes.
  No explicit lock required — identical pattern to overlay mode-switching in
  Phase 2/4.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from modes.base_mode import BaseVisionMode


# ── Public constants ───────────────────────────────────────────────────────────

LAYOUT_NONE        = "none"
LAYOUT_2X_HORIZ    = "2x_horizontal"
LAYOUT_2X_VERT     = "2x_vertical"
LAYOUT_4X_GRID     = "4x_grid"

VALID_LAYOUTS: frozenset = frozenset(
    {LAYOUT_NONE, LAYOUT_2X_HORIZ, LAYOUT_2X_VERT, LAYOUT_4X_GRID}
)

# Human-readable names used for HUD / UI labels
LAYOUT_DISPLAY_NAMES: dict[str, str] = {
    LAYOUT_NONE:     "Off",
    LAYOUT_2X_HORIZ: "2× Horizontal",
    LAYOUT_2X_VERT:  "2× Vertical",
    LAYOUT_4X_GRID:  "4× Grid",
}


# ── Drawing constants ─────────────────────────────────────────────────────────

_LABEL_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE     = 0.55
_LABEL_THICKNESS = 1
_LABEL_PAD_X     = 10
_LABEL_PAD_Y     = 22
_LABEL_FG        = (220, 220, 220)   # BGR light grey
_LABEL_SHADOW    = (0,   0,   0)     # BGR black drop-shadow

_DIVIDER_COLOR   = (55, 55, 55)      # BGR dark grey
_DIVIDER_THICK   = 2


# ── Manager ────────────────────────────────────────────────────────────────────

class SplitScreenManager:
    """
    Manages split-screen layout state and composes the output frame.

    Typical GUI-thread setup:
        ss = SplitScreenManager()
        ss.layout_mode = LAYOUT_2X_VERT
        ss.set_mode(0, dog_vision)    # left panel
        ss.set_mode(1, pit_viper)     # right panel

    Typical worker-thread call:
        if ss.is_active:
            out = ss.compose(raw_frame)   # (H, W, 3) same shape as input
    """

    def __init__(self) -> None:
        self._layout_mode: str = LAYOUT_NONE
        # Slots 0=A, 1=B, 2=C, 3=D.  None → show raw (unfiltered) panel.
        self._modes: List[Optional[BaseVisionMode]] = [None, None, None, None]

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def layout_mode(self) -> str:
        """Active layout identifier.  Assign a LAYOUT_* constant."""
        return self._layout_mode

    @layout_mode.setter
    def layout_mode(self, value: str) -> None:
        if value not in VALID_LAYOUTS:
            raise ValueError(
                f"Unknown layout {value!r}.  Valid: {sorted(VALID_LAYOUTS)}"
            )
        self._layout_mode = value

    @property
    def modes(self) -> List[Optional[BaseVisionMode]]:
        """Four-slot list.  Index directly: modes[0] = panel A, etc."""
        return self._modes

    @property
    def is_active(self) -> bool:
        """True whenever any split layout is selected (layout_mode != 'none')."""
        return self._layout_mode != LAYOUT_NONE

    # ── Mode management ───────────────────────────────────────────────────

    def set_mode(self, slot: int, mode: Optional[BaseVisionMode]) -> None:
        """
        Assigns a vision mode to a panel slot.

        Args:
            slot: 0–3 (panels A–D respectively).
            mode: Any BaseVisionMode instance, or None for a raw/passthrough panel.

        Raises:
            IndexError: if slot is outside 0–3.
        """
        if not 0 <= slot <= 3:
            raise IndexError(f"Panel slot must be 0–3, got {slot!r}")
        self._modes[slot] = mode  # atomic list-element write under GIL

    # ── Composition entry point ────────────────────────────────────────────

    def compose(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Returns the composed split-screen frame.

        Reads layout_mode and modes[] once — both attribute reads are atomic
        under the GIL, so no lock is needed against GUI-thread writes.

        All heavy operations are NumPy/OpenCV C-level; no Python pixel loops.

        Args:
            raw_frame: BGR uint8 ndarray (H, W, 3) from ScreenCapture.

        Returns:
            BGR uint8 ndarray of identical shape (H, W, 3).  When layout_mode
            is "none", raw_frame is returned unchanged (zero copy).
        """
        layout = self._layout_mode          # single read, atomic
        if layout == LAYOUT_2X_HORIZ:
            return self._compose_2x_horizontal(raw_frame)
        if layout == LAYOUT_2X_VERT:
            return self._compose_2x_vertical(raw_frame)
        if layout == LAYOUT_4X_GRID:
            return self._compose_4x_grid(raw_frame)
        return raw_frame                    # LAYOUT_NONE pass-through

    # ── Layout implementations ─────────────────────────────────────────────

    def _compose_2x_horizontal(self, raw: np.ndarray) -> np.ndarray:
        """Top = mode[0] (A), bottom = mode[1] (B) — each at half height."""
        h, w  = raw.shape[:2]
        h_top = h - h // 2          # ceiling half, e.g. 540 for h=1079
        h_bot = h // 2              # floor  half, e.g. 539 for h=1079

        top = self._process_panel(raw, 0, (w, h_top))
        bot = self._process_panel(raw, 1, (w, h_bot))

        out = np.vstack([top, bot])                         # (h, w, 3) ✓
        _draw_label(out, "A", self._panel_name(0),   0, 0)
        _draw_label(out, "B", self._panel_name(1),   0, h_top)
        _draw_h_divider(out, h_top)
        return out

    def _compose_2x_vertical(self, raw: np.ndarray) -> np.ndarray:
        """Left = mode[0] (A), right = mode[1] (B) — each at half width."""
        h, w   = raw.shape[:2]
        w_left  = w - w // 2        # ceiling half
        w_right = w // 2            # floor  half

        left  = self._process_panel(raw, 0, (w_left,  h))
        right = self._process_panel(raw, 1, (w_right, h))

        out = np.hstack([left, right])                      # (h, w, 3) ✓
        _draw_label(out, "A", self._panel_name(0), 0,      0)
        _draw_label(out, "B", self._panel_name(1), w_left, 0)
        _draw_v_divider(out, w_left)
        return out

    def _compose_4x_grid(self, raw: np.ndarray) -> np.ndarray:
        """2×2 quad grid: TL=A, TR=B, BL=C, BR=D."""
        h, w    = raw.shape[:2]
        h_top   = h - h // 2        # ceiling half
        h_bot   = h // 2            # floor  half
        w_left  = w - w // 2
        w_right = w // 2

        tl = self._process_panel(raw, 0, (w_left,  h_top))
        tr = self._process_panel(raw, 1, (w_right, h_top))
        bl = self._process_panel(raw, 2, (w_left,  h_bot))
        br = self._process_panel(raw, 3, (w_right, h_bot))

        top_row = np.hstack([tl, tr])                       # (h_top, w, 3)
        bot_row = np.hstack([bl, br])                       # (h_bot, w, 3)
        out     = np.vstack([top_row, bot_row])             # (h,     w, 3) ✓

        _draw_label(out, "A", self._panel_name(0), 0,      0)
        _draw_label(out, "B", self._panel_name(1), w_left, 0)
        _draw_label(out, "C", self._panel_name(2), 0,      h_top)
        _draw_label(out, "D", self._panel_name(3), w_left, h_top)
        _draw_h_divider(out, h_top)
        _draw_v_divider(out, w_left)
        return out

    # ── Per-panel processing ───────────────────────────────────────────────

    def _process_panel(
        self,
        raw: np.ndarray,
        slot: int,
        size: Tuple[int, int],       # (width, height) — cv2 convention
    ) -> np.ndarray:
        """
        Resizes raw to the panel's target dimensions, then applies modes[slot].

        Resize-before-apply is the key performance invariant:
          1920×1080 source  →  960×540 panel  →  mode.apply(960×540)
        Each mode processes 518 K pixels instead of 2.07 M — a 4× reduction.
        cv2.INTER_LINEAR is the fastest interpolation that avoids aliasing.

        Args:
            raw:  Full-resolution source frame.
            slot: Mode slot index (0–3).
            size: (width, height) tuple for cv2.resize.

        Returns:
            Processed panel array of shape (height, width, 3).
        """
        panel = cv2.resize(raw, size, interpolation=cv2.INTER_LINEAR)
        mode  = self._modes[slot]      # single read, atomic under GIL
        if mode is not None:
            panel = mode.apply(panel)
        return panel

    def map_screen_point(
        self,
        x: int,
        y: int,
        screen_w: int,
        screen_h: int,
    ) -> Tuple[int, int, int]:
        """
        Maps a full-screen coordinate to a (slot, panel_x, panel_y) tuple.

        Given a click at (x, y) in the composed frame's coordinate space,
        returns which panel was hit and the coordinate in that panel's local
        space.  Useful for inspection mode: determine which sub-panel the user
        clicked without any Python-side pixel loop.

        Args:
            x, y:              Point in the full-screen overlay space.
            screen_w, screen_h: Dimensions of the full composed frame.

        Returns:
            (slot, panel_x, panel_y) — slot 0–3 for panels A–D, or -1 when
            split-screen is inactive or the layout is unknown.
        """
        layout = self._layout_mode
        if layout == LAYOUT_NONE:
            return (-1, x, y)

        if layout == LAYOUT_2X_HORIZ:
            h_top = screen_h - screen_h // 2
            if y < h_top:
                return (0, x, y)
            return (1, x, y - h_top)

        if layout == LAYOUT_2X_VERT:
            w_left = screen_w - screen_w // 2
            if x < w_left:
                return (0, x, y)
            return (1, x - w_left, y)

        if layout == LAYOUT_4X_GRID:
            h_top  = screen_h - screen_h // 2
            w_left = screen_w - screen_w // 2
            if y < h_top:
                if x < w_left:
                    return (0, x, y)                    # TL — slot A
                return (1, x - w_left, y)              # TR — slot B
            if x < w_left:
                return (2, x, y - h_top)               # BL — slot C
            return (3, x - w_left, y - h_top)          # BR — slot D

        return (-1, x, y)

    def _panel_name(self, slot: int) -> str:
        """Returns the mode name for a slot, or 'Raw' if no mode is assigned."""
        mode = self._modes[slot]
        return mode.name if mode is not None else "Raw"


# ── Module-level drawing utilities (stateless, no allocation) ─────────────────

def _draw_label(
    frame: np.ndarray,
    letter: str,
    mode_name: str,
    x_off: int,
    y_off: int,
) -> None:
    """
    Renders a panel identifier in the top-left corner of a panel region.

    Two-pass render (shadow first, then foreground) ensures legibility on
    both dark and bright image content without a background rectangle.
    Draws directly into `frame` (in-place, no allocation).
    """
    text = f"[{letter}] {mode_name}"
    x = x_off + _LABEL_PAD_X
    y = y_off + _LABEL_PAD_Y
    # Shadow pass — slightly offset, thicker stroke
    cv2.putText(
        frame, text, (x + 1, y + 1),
        _LABEL_FONT, _LABEL_SCALE, _LABEL_SHADOW,
        _LABEL_THICKNESS + 1, cv2.LINE_AA,
    )
    # Foreground pass
    cv2.putText(
        frame, text, (x, y),
        _LABEL_FONT, _LABEL_SCALE, _LABEL_FG,
        _LABEL_THICKNESS, cv2.LINE_AA,
    )


def _draw_h_divider(frame: np.ndarray, y: int) -> None:
    """Draws a horizontal divider at pixel row y. In-place, no allocation."""
    _, w = frame.shape[:2]
    cv2.line(frame, (0, y), (w - 1, y), _DIVIDER_COLOR, _DIVIDER_THICK)


def _draw_v_divider(frame: np.ndarray, x: int) -> None:
    """Draws a vertical divider at pixel column x. In-place, no allocation."""
    h, _ = frame.shape[:2]
    cv2.line(frame, (x, 0), (x, h - 1), _DIVIDER_COLOR, _DIVIDER_THICK)
