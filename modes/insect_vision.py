"""
Insect Vision Mode

Insects see through compound eyes composed of hundreds to tens of
thousands of individual optical units called ommatidia.  Each ommatidium
points in a slightly different direction and contributes a single
averaged colour/brightness pixel to the final percept.

Key characteristics:
- Very low spatial resolution (each ommatidium ≈ one pixel)
- Wide or near-360° field of view
- High temporal resolution (can detect flicker > 300 Hz)
- Sensitivity to UV and polarised light (not visible to humans)
- Green/UV-dominant spectral sensitivity (less red sensitivity)
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode
from effects.overlays import add_hexagonal_overlay


class InsectVision(BaseVisionMode):
    """
    Simulates compound-eye insect vision with pixelation and UV channel.
    """

    # Pixel-block size in the compound-eye simulation.
    # Each block represents one ommatidium covering ~5–10° of visual field.
    OMMATIDIUM_SIZE = 9

    @property
    def name(self) -> str:
        return "Insect Vision"

    @property
    def description(self) -> str:
        return "Compound eye: hexagonal facets, UV sensitivity, pixelated."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies insect compound-eye simulation.

        Steps:
        1. Pixelate — each block is one ommatidium (low spatial resolution).
        2. Spectral shift — boost UV-proxy (blue/green), suppress red.
        3. Hexagonal grid overlay — visual texture of the facet structure.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Processed BGR frame with compound-eye visual appearance.
        """
        h, w = frame.shape[:2]
        s = self.OMMATIDIUM_SIZE

        # Pixelate: downsample then upsample with NEAREST to get hard blocks
        small = cv2.resize(frame, (w // s, h // s), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Insects are most sensitive to UV (~340 nm), blue-green (~480 nm),
        # and green (~550 nm).  Red sensitivity is minimal.
        shifted = pixelated.astype(np.float32)
        shifted[:, :, 0] = np.clip(shifted[:, :, 0] * 1.35, 0, 255)  # Blue (UV proxy)
        shifted[:, :, 1] = np.clip(shifted[:, :, 1] * 1.20, 0, 255)  # Green
        shifted[:, :, 2] = np.clip(shifted[:, :, 2] * 0.45, 0, 255)  # Red (suppressed)
        shifted = shifted.astype(np.uint8)

        # Hexagonal grid represents the physical facet boundaries
        return add_hexagonal_overlay(shifted, hex_radius=s + 7, alpha=0.30)
