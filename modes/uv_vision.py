"""
UV Vision Mode

Ultraviolet (UV) light occupies wavelengths of roughly 300–400 nm —
just below the violet end of the human visible spectrum.  Humans lack
UV photoreceptors, but many species can perceive it:

- Bees & butterflies: navigate by UV nectar guides on flowers.
- Birds: UV-reflective plumage patterns encode mate quality.
- Reindeer: UV-absorbing lichen stands out sharply against UV-bright snow.
- Some fish and reptiles: UV-sensitive cone classes.

This mode simulates UV perception by exploiting the relationship between
the visible and ultraviolet spectra:
- UV is the opposite spectral end from red → invert the red channel.
- UV boosts blues and violets → amplify the blue channel strongly.
- The resulting purple/violet palette approximates what a UV-capable
  retina might report.
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode
from effects.color_filters import adjust_saturation
from effects.contrast import gamma_correction


class UVVision(BaseVisionMode):
    """
    Simulates ultraviolet spectral sensitivity.
    """

    @property
    def name(self) -> str:
        return "UV Vision"

    @property
    def description(self) -> str:
        return "UV-range perception: hidden patterns revealed in purple tones."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies UV-spectrum vision simulation.

        Steps:
        1. Invert red channel — UV is the spectral opposite of red.
        2. Strongly boost blue — UV sits just beyond the blue end of
           the visible range.
        3. Suppress green slightly — UV sensitivity is lower in mid-spectrum.
        4. Saturation boost — UV reveals extra structural patterns that
           appear highly saturated to UV-capable observers.
        5. Mild gamma darkening — UV images tend to look high-contrast
           and slightly underexposed to the human eye.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Processed BGR frame with UV-shifted colour perception.
        """
        result = frame.astype(np.float32)

        # Invert red (red = longest visible wavelength; UV = shortest)
        result[:, :, 2] = 255.0 - result[:, :, 2]

        # Boost blue (UV maps onto the blue-end of the shifted spectrum)
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.7, 0, 255)

        # Suppress green (minimum UV sensitivity in the mid-spectrum)
        result[:, :, 1] = np.clip(result[:, :, 1] * 0.65, 0, 255)

        uv_frame = result.astype(np.uint8)

        # UV reveals otherwise invisible structural patterns → extra saturation
        saturated = adjust_saturation(uv_frame, factor=1.6)

        # Slight darkening for the contrasty UV look
        return gamma_correction(saturated, gamma=0.88)
