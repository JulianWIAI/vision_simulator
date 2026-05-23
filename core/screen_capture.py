"""
Screen Capture Module

Handles real-time screen capture using the `mss` library.
MSS (Multiple Screen Shots) is chosen for its speed and cross-platform
support: it reads directly from the graphics framebuffer rather than
going through a slower OS screenshot API.
"""

import numpy as np
import mss
from typing import Optional, Tuple


class ScreenCapture:
    """
    Captures the screen in real time using MSS.

    MSS is significantly faster than PIL.ImageGrab or pyautogui for
    screen capture because it avoids extra copies and compression steps.

    Usage:
        cap = ScreenCapture()
        frame = cap.capture()   # returns BGR numpy array
    """

    def __init__(
        self,
        monitor_index: int = 1,
        region: Optional[dict] = None,
    ) -> None:
        """
        Initialises the capture context.

        Args:
            monitor_index: Which monitor to capture (1 = primary,
                           2 = secondary, etc.).
            region: Optional dict with keys 'top', 'left', 'width',
                    'height' to capture a sub-region instead of the
                    full monitor.
        """
        self._sct = mss.mss()
        self.monitor_index = monitor_index
        self.region = region
        # Cache the capture bounds to avoid repeated dict lookups.
        self._monitor = self._resolve_region()

    # ── Private helpers ───────────────────────────────────────────────

    def _resolve_region(self) -> dict:
        """
        Returns the MSS monitor dict that defines the capture rectangle.

        If a custom region was provided at init time that region is used;
        otherwise the full primary monitor bounds are used.
        """
        if self.region:
            return self.region
        return self._sct.monitors[self.monitor_index]

    # ── Public interface ─────────────────────────────────────────────

    def capture(self) -> np.ndarray:
        """
        Grabs the current frame from the screen.

        MSS returns a BGRA image; the alpha channel is dropped here
        so the result is always a plain 3-channel BGR array compatible
        with OpenCV.

        Returns:
            np.ndarray: Frame in BGR format, shape (H, W, 3), dtype uint8.
        """
        raw = self._sct.grab(self._monitor)
        # Drop alpha: [:, :, :3] selects only B, G, R channels.
        return np.array(raw)[:, :, :3]

    def get_resolution(self) -> Tuple[int, int]:
        """
        Returns the capture resolution as (width, height).
        """
        return self._monitor["width"], self._monitor["height"]

    def set_region(self, region: dict) -> None:
        """
        Updates the capture region on the fly.

        Args:
            region: Dict with keys 'top', 'left', 'width', 'height'.
        """
        self.region = region
        self._monitor = self._resolve_region()

    def __del__(self) -> None:
        """Releases MSS resources when the object is garbage-collected."""
        try:
            self._sct.close()
        except Exception:
            pass
