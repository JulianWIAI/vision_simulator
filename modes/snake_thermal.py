"""
Snake Thermal Vision Mode

Pythons, boas, and some vipers possess pit organs (loreal pits) that
detect long-wave infrared radiation (5–30 μm) from warm-bodied prey.
The pit organ produces a low-resolution, blurry thermal image with an
angular resolution of roughly 1°.

This mode simulates the snake's dual-sensory percept: a thermal IR
image blended with the normal visual field, representing the proposed
bimodal integration that occurs in the snake's optic tectum.
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode
from effects.color_filters import blend_frames
from effects.lut import build_thermal_lut, apply_grayscale_lut
from effects.blur import gaussian_blur


class SnakeThermalVision(BaseVisionMode):
    """
    Simulates heat-sensing perception blended with normal snake vision.

    Thermal overlay (60%) + original visual field (40%) reflects
    the dual sensory fusion proposed by Gracheva et al. (2010).
    """

    def __init__(self) -> None:
        # Build LUT once; it never changes between frames.
        self._lut = build_thermal_lut()

    @property
    def name(self) -> str:
        return "Snake Thermal"

    @property
    def description(self) -> str:
        return "Low-res infrared overlay blended with normal vision."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies blended thermal + visual perception.

        Steps:
        1. Convert frame to grayscale — luminance approximates heat signature.
        2. Blur the grayscale map — pit organ spatial resolution is very low.
        3. Apply thermal colour LUT.
        4. Blend 60% thermal with 40% original (bimodal fusion).

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Blended BGR frame with thermal overlay.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pit organs form a blurry thermal image (~1° angular resolution)
        blurred_gray = gaussian_blur(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), kernel_size=11
        )
        blurred_gray = cv2.cvtColor(blurred_gray, cv2.COLOR_BGR2GRAY)

        # Map heat signature to thermal colours
        thermal = apply_grayscale_lut(blurred_gray, self._lut)

        # Blend: dominant thermal + residual visual
        return blend_frames(thermal, frame, alpha=0.65)
