"""
Frog Vision Mode

Frogs have a highly specialised retina with five distinct classes of
retinal ganglion cells (RGCs), described by Lettvin et al. (1959) in
their landmark paper "What the Frog's Eye Tells the Frog's Brain":

1. Boundary detectors — respond to sharp contrast edges.
2. Moving-edge detectors ("bug detectors") — fire only for small
   dark moving objects (i.e., insects).
3. Net dimming detectors — respond when large dark objects move
   (potential predators).
4. Net convexity detectors — respond to small convex shapes.
5. Darkness detectors — fire when overall illumination decreases.

The key insight: a stationary scene produces almost NO neural signal.
Frogs perceive static environments as featureless grey; only moving
objects generate clear, bright percepts.

This mode simulates that bias using frame-to-frame differencing.
"""

import numpy as np
import cv2
from typing import Optional
from modes.base_mode import BaseVisionMode
from effects.contrast import adjust_contrast_brightness
from effects.blur import gaussian_blur


class FrogVision(BaseVisionMode):
    """
    Simulates motion-biased frog retinal perception.

    Static scenes are heavily suppressed; moving regions appear bright
    green (the colour most frogs are sensitive to via their green rods).
    """

    # Motion detection threshold: pixel difference must exceed this
    # value to be counted as "movement".
    MOTION_THRESHOLD: int = 12

    def __init__(self) -> None:
        # Stores the previous greyscale frame for differencing.
        self._prev_gray: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Frog Vision"

    @property
    def description(self) -> str:
        return "Static = dim grey, motion = bright green (bug-detector retina)."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies motion-emphasising retinal simulation.

        On the first call there is no previous frame, so the static
        view is returned immediately.  From the second call onward,
        a per-pixel difference map highlights moving regions.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Processed BGR frame with frog-like visual bias.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return self._static_view(frame)

        # ── Motion detection ─────────────────────────────────────────
        # Absolute frame difference highlights pixels that changed.
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray.copy()

        _, motion_mask = cv2.threshold(
            diff, self.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY
        )
        # Dilate: extends the motion region to nearby pixels (frogs respond
        # to the full shape of a moving insect, not just its leading edge).
        motion_mask = cv2.dilate(
            motion_mask, np.ones((7, 7), np.uint8), iterations=2
        )

        # ── Compose static background and motion foreground ──────────
        static_bg  = self._static_view(frame)
        motion_fg  = self._motion_view(frame)

        weight = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        result = (motion_fg.astype(np.float32) * weight +
                  static_bg.astype(np.float32) * (1.0 - weight))
        return result.astype(np.uint8)

    # ── Private helpers ───────────────────────────────────────────────

    def _static_view(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns the dim, blurred, nearly featureless static background.

        In a real frog retina, an unchanging scene produces very little
        firing.  We simulate this with heavy blur and strong darkening.
        """
        blurred  = gaussian_blur(frame, kernel_size=13)
        gray_bgr = cv2.cvtColor(
            cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR,
        )
        return adjust_contrast_brightness(gray_bgr, alpha=0.35, beta=0)

    def _motion_view(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns the bright green motion foreground.

        Green dominance reflects the frog's green-rod photoreceptors
        and their role as the primary "bug detector" channel.
        """
        motion = np.zeros_like(frame)
        # Push all luminance into the green channel
        motion[:, :, 1] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return motion
