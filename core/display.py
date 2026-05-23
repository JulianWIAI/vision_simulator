"""
HUD Renderer

Draws the mode-name and navigation-hint bar onto a BGR numpy frame
using OpenCV.  All window-management code (cv2.namedWindow, cv2.imshow,
cv2.waitKey) has been removed; the PySide6 overlay in overlay_window.py
is now responsible for showing frames on screen.

The HUDRenderer class is kept here (rather than inlined into FrameWorker)
so that the HUD logic lives in one place and can be adjusted independently
of the capture/emit loop.
"""

import numpy as np
import cv2
from utils.config import HUD_FONT, HUD_FONT_SCALE, HUD_COLOR, HUD_HEIGHT


class HUDRenderer:
    """
    Composites a semi-transparent status bar onto a BGR frame.

    Kept as a class to make future extensions (FPS counter, animated
    transitions, per-mode colour themes) easy to add without touching
    FrameWorker.
    """

    def draw(
        self,
        frame: np.ndarray,
        mode_name: str,
        mode_index: int,
        total_modes: int,
    ) -> np.ndarray:
        """
        Returns a new frame with the HUD bar rendered at the bottom.

        Args:
            frame: BGR image array (H, W, 3) uint8.
            mode_name: Name of the currently active vision mode.
            mode_index: Zero-based index of the current mode.
            total_modes: Total number of registered modes.

        Returns:
            Copy of the input frame with the HUD composited onto it.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Semi-transparent dark strip
        strip = out.copy()
        cv2.rectangle(strip, (0, h - HUD_HEIGHT), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(strip, 0.55, out, 0.45, 0, out)

        # Left side: mode counter and name
        left_text = f"Mode [{mode_index + 1}/{total_modes}]:  {mode_name}"
        cv2.putText(
            out, left_text,
            (16, h - HUD_HEIGHT + 24),
            HUD_FONT, HUD_FONT_SCALE, HUD_COLOR, 1, cv2.LINE_AA,
        )

        # Right side: key hints (right-aligned)
        hint = "1-9/0: switch   N/P: next/prev   ESC: quit"
        (txt_w, _), _ = cv2.getTextSize(hint, HUD_FONT, HUD_FONT_SCALE * 0.75, 1)
        cv2.putText(
            out, hint,
            (w - txt_w - 16, h - 10),
            HUD_FONT, HUD_FONT_SCALE * 0.75, (140, 140, 140), 1, cv2.LINE_AA,
        )

        return out
