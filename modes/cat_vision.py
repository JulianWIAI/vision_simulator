"""
Cat Vision Mode

Cats are also dichromats, but their spectral sensitivity differs from
dogs.  They are most sensitive to blue (~450 nm) and green/yellow
(~555 nm) wavelengths, with virtually no red sensitivity.

Key characteristics:
- Superior low-light vision: tapetum lucidum reflects light back
  through the retina, roughly doubling photon capture efficiency.
- High rod density → excellent motion detection, poorer static detail.
- Roughly 20/100 human-equivalent visual acuity in daylight.
- Blue-dominant colour perception.
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode
from effects.color_filters import adjust_saturation, apply_channel_weights
from effects.contrast import gamma_correction, clahe_enhance


class CatVision(BaseVisionMode):
    """
    Simulates feline dichromatic vision with enhanced low-light performance.
    """

    @property
    def name(self) -> str:
        return "Cat Vision"

    @property
    def description(self) -> str:
        return "Dichromatic blue-green vision with night-adapted contrast."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies cat vision simulation.

        Steps:
        1. Boost blue/green channels; suppress red (cats have no red cones).
        2. Gamma < 1.0 mimics the brightness enhancement of the tapetum.
        3. CLAHE — sharp local contrast in lit areas, black in dark areas.
        4. Saturation reduction — dichromatic colour space is narrower.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Processed BGR frame simulating feline vision.
        """
        # Suppress red (cat L-cone peaks ~555 nm, not in the red range)
        reweighted = apply_channel_weights(frame, b=1.3, g=1.1, r=0.4)

        # Tapetum lucidum effectively doubles available light → brighter image
        brightened = gamma_correction(reweighted, gamma=0.65)

        # CLAHE emphasises the sharp/dark boundary typical of cat vision
        contrasted = clahe_enhance(brightened, clip_limit=3.0, tile_size=8)

        # Narrow colour gamut of dichromatic vision
        return adjust_saturation(contrasted, factor=0.60)
