"""
Octopus Polarized Vision Mode — Phase 6

Cephalopods (octopuses, squids, cuttlefish) are monochromatic — they have a
single type of photoreceptor and cannot discriminate wavelength.  Yet they
show remarkable colour-matching behaviour.  The leading hypothesis: they
detect the *polarization angle* of light, which varies by surface material and
orientation, giving them a dimension of information invisible to humans.

Pipeline
────────
1. Grayscale conversion         — strip wavelength; keep luminance structure.
2. Sobel gradient (X, Y)        — detect spatial transitions (edges, textures).
3. Polarization angle           — arctan2(Gy, Gx) ∈ [−π, π]; mapped to hue.
4. Gradient magnitude           — encodes signal strength as HSV Value.
5. HSV → BGR                    — hue = angle, saturation = 255, value = mag.

In the output: uniform flat regions appear nearly black (weak gradient →
weak polarization signal), while edges and textures burst with saturated
colour whose hue encodes the local polarization orientation.

References
──────────
Mäthger et al. (2009) "Evidence for polarisation vision in the octopus",
J. Exp. Biol. 212, 2133–2140.
"""

from __future__ import annotations

import cv2
import numpy as np

from modes.base_mode import BaseVisionMode


class PolarizedVision(BaseVisionMode):
    """
    Hue-encodes local polarization angle; brightness encodes gradient strength.

    Uses CV_32F throughout to avoid precision loss in intermediate steps.
    cv2.magnitude() is SIMD-accelerated and ~2× faster than
    np.sqrt(gx**2 + gy**2) at 1080p.
    """

    _SOBEL_KSIZE: int = 3

    @property
    def name(self) -> str:
        return "Octopus Polarized Vision"

    @property
    def description(self) -> str:
        return "Hue = polarization angle (arctan2 of Sobel gradients); brightness = gradient magnitude."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR uint8 (H, W, 3).

        Returns:
            BGR uint8 (H, W, 3) — hue-coded polarization angle map.
        """
        # ── Step 1: grayscale ──────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Step 2: Sobel gradients (float32 for precision) ────────────────
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self._SOBEL_KSIZE)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self._SOBEL_KSIZE)

        # ── Step 3: polarization angle → hue channel ───────────────────────
        # arctan2 in [-π, π]; shift to [0, 2π] then scale to OpenCV hue [0, 179]
        angle = np.arctan2(gy, gx)                              # float32
        hue   = ((angle + np.pi) * (179.0 / (2.0 * np.pi))).astype(np.uint8)

        # ── Step 4: gradient magnitude → value channel ─────────────────────
        mag = cv2.magnitude(gx, gy)                             # float32, SIMD
        # Normalize to [0, 255] — NORM_MINMAX adapts per-frame dynamic range
        mag_u8 = np.empty(gray.shape, dtype=np.uint8)
        cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)
        mag_u8 = mag.astype(np.uint8)

        # ── Step 5: compose HSV and convert ───────────────────────────────
        sat = np.full(gray.shape, 255, dtype=np.uint8)
        hsv = cv2.merge([hue, sat, mag_u8])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
