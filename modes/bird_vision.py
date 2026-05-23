"""
Bird Vision Mode

Most birds are tetrachromats — they have four cone types: S (UV, ~360 nm),
M (blue, ~455 nm), L (green, ~508 nm), and VS/UVS (red, ~565 nm) plus
additional oil-droplet filters that further sharpen colour discrimination.

A bird's perceivable colour space is vastly richer than the human trichromatic
space.  Plumage patterns, nectar guides on flowers, and UV-reflective urine
trails that are invisible to humans are clearly visible to birds.

Key characteristics simulated:
- Strong saturation boost (much wider colour gamut)
- UV sensitivity proxy (blue channel exaggerated)
- Slight hue shift toward shorter wavelengths
- High foveal acuity (CLAHE sharpening)
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode
from effects.color_filters import adjust_saturation, shift_hue
from effects.contrast import clahe_enhance


class BirdVision(BaseVisionMode):
    """
    Simulates tetrachromatic avian vision with UV sensitivity and vivid colour.
    """

    @property
    def name(self) -> str:
        return "Bird Vision"

    @property
    def description(self) -> str:
        return "Tetrachromatic vision with UV sensitivity and vivid colour."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies avian tetrachromatic vision simulation.

        Steps:
        1. Strong saturation boost — birds resolve colours humans cannot.
        2. Blue-channel boost — simulates UV fourth cone axis.
        3. Slight hue shift — brings the effective spectrum toward UV.
        4. CLAHE — mimics the high cone density of the avian fovea.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Processed BGR frame simulating avian vision.
        """
        # Birds see far more vivid colours due to four cone types + oil droplets
        saturated = adjust_saturation(frame, factor=2.2)

        # UV fourth cone → blue channel carries extra information
        boosted = saturated.astype(np.float32)
        boosted[:, :, 0] = np.clip(boosted[:, :, 0] * 1.5, 0, 255)  # Blue/UV
        boosted[:, :, 1] = np.clip(boosted[:, :, 1] * 1.1, 0, 255)  # Green
        uv_frame = boosted.astype(np.uint8)

        # Shift visible spectrum slightly toward UV (shorter wavelengths)
        shifted = shift_hue(uv_frame, degrees=-12)

        # Avian fovea has ~2–3× the cone density of the human fovea
        return clahe_enhance(shifted, clip_limit=2.5, tile_size=6)
