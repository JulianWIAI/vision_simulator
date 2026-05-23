"""
Depth Map Vision Mode

Depth perception can be inferred monocularly from several visual cues:
- Defocus blur: objects near the focal plane appear sharp; others blur.
- Texture gradient: fine texture → far; coarse texture → near.
- Edge density: high-frequency edge regions are typically near.

This mode uses edge-sharpness (Laplacian magnitude) as a depth proxy.
High sharpness (many edges, high spatial frequencies) → near → warm colour.
Low sharpness (smooth, blurry regions) → far → cool colour.

Note: this is a heuristic approximation, not true 3D reconstruction.
For real depth a stereo camera pair, structured light (e.g., RealSense),
or a ToF sensor would be required.  A future extension could replace this
stage with an AI depth estimator (e.g., MiDaS via ONNX / MediaPipe).
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode
from effects.lut import build_thermal_lut, apply_grayscale_lut


class DepthMapVision(BaseVisionMode):
    """
    Approximates monocular depth from edge sharpness, colour-coded as a
    thermal gradient (warm = near, cool = far).
    """

    @property
    def name(self) -> str:
        return "Depth Map"

    @property
    def description(self) -> str:
        return "Edge-sharpness depth proxy: near=warm/sharp, far=cool/smooth."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimates a per-pixel depth proxy and maps it to a thermal palette.

        Steps:
        1. Laplacian edge magnitude — high response at sharp edges (near).
        2. Heavy Gaussian smoothing — converts the edge map into a
           continuous surface estimate.
        3. Normalise to 0–255 uint8.
        4. Apply thermal LUT — warm (red/white) = near, cool (blue) = far.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Depth-coded BGR frame with thermal colour mapping.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Laplacian responds strongly at edges (high spatial frequency = near)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edge_energy = np.abs(laplacian)

        # Smooth to a continuous surface — 51-px kernel blends across regions
        depth_surface = cv2.GaussianBlur(edge_energy, (51, 51), 0)

        # Normalise to full uint8 range so the LUT maps the complete palette
        depth_norm = cv2.normalize(
            depth_surface, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Thermal LUT: bright (high edge energy = near) → red/white
        lut = build_thermal_lut()
        return apply_grayscale_lut(depth_norm, lut)
