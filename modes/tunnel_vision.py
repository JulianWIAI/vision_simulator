"""
Tunnel Vision Mode

Tunnel vision (peripheral vision loss) is characterised by a restricted
visual field where only a central circular area is perceived clearly;
the periphery is absent or severely degraded.

Common causes:
- Glaucoma: progressive optic-nerve damage starting at the periphery.
- Retinitis pigmentosa: rod cell degeneration from the periphery inward.
- Severe hypoglycaemia or hypoxia: temporary peripheral suppression.
- Acute stress response: sympathetic nervous system narrows attention.

The simulation uses a smooth radial mask (Gaussian-feathered at the
boundary) to avoid the unnatural hard circle edge of a simple threshold.
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode


class TunnelVision(BaseVisionMode):
    """
    Simulates peripheral vision loss with a configurable tunnel radius.
    """

    def __init__(self, radius: float = 0.33) -> None:
        """
        Args:
            radius: Fraction of the frame half-diagonal that remains fully
                    visible (0.1 = very tight tunnel, 0.6 = mild restriction).
        """
        self.radius = radius
        # Cache the mask per frame size to avoid recomputing every tick.
        self._mask_cache: dict = {}

    @property
    def name(self) -> str:
        return "Tunnel Vision"

    @property
    def description(self) -> str:
        return "Peripheral vision loss: only a central circle remains clear."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies the radial tunnel-vision mask.

        The mask is a smooth radial falloff (centre = 1.0, border = 0.0)
        created with a large Gaussian blur over a hard binary circle so
        the transition zone mimics the gradual scotoma boundary of real
        tunnel-vision conditions.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Masked BGR frame — bright central circle, black periphery.
        """
        h, w = frame.shape[:2]
        key = (h, w)

        if key not in self._mask_cache:
            self._mask_cache[key] = self._build_mask(h, w)

        result = frame.astype(np.float32) * self._mask_cache[key]
        return result.astype(np.uint8)

    def _build_mask(self, h: int, w: int) -> np.ndarray:
        """
        Builds and caches the radial tunnel mask as a (H, W, 3) float32 array.

        A binary circle is drawn then blurred with a large kernel so the
        edge is soft, approximating the gradual transition found in
        clinical visual field tests (Humphrey perimetry).
        """
        mask = np.zeros((h, w), dtype=np.float32)

        # Compute the tunnel radius in pixels relative to the shorter axis.
        rx = int(w * self.radius)
        ry = int(h * self.radius)
        cv2.ellipse(
            mask,
            center=(w // 2, h // 2),
            axes=(rx, ry),
            angle=0, startAngle=0, endAngle=360,
            color=1.0, thickness=-1,
        )

        # Gaussian feathering — kernel size must be odd; use ~8 % of width.
        k = max(51, int(min(h, w) * 0.08) | 1)  # bitwise OR 1 ensures odd
        mask = cv2.GaussianBlur(mask, (k, k), 0)

        return np.stack([mask, mask, mask], axis=2)
