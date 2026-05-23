"""
AI Edge Detection Vision Mode

Edge detection is both a fundamental computer vision technique and a
plausible model of early mammalian visual cortex processing.
Simple cells in the primary visual cortex (V1) fire selectively to
oriented line segments — effectively acting as biological edge detectors.

This mode renders the world as glowing cyan edge lines on a black
background, as a machine vision system might represent a scene after
feature extraction.

Two complementary edge detectors are combined:
- Canny (cv2.Canny): precise, thin, well-localised edges.
- Sobel gradient magnitude: broader, softer gradient response.
Combining both produces richer edge coverage than either alone.
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode
from effects.contrast import clahe_enhance


class AIEdgeVision(BaseVisionMode):
    """
    Renders the scene as glowing neon edge lines on a black background.
    """

    @property
    def name(self) -> str:
        return "AI Edge Detection"

    @property
    def description(self) -> str:
        return "Machine-vision edge skeleton: cyan neon lines on black."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts edges and renders them with a neon glow.

        Steps:
        1. CLAHE — normalises local contrast so edges are detectable
           regardless of scene brightness variation.
        2. Canny — extracts thin, precise edge outlines.
        3. Sobel magnitude — captures softer gradient transitions.
        4. Blend both edge maps with weighted addition.
        5. Colour the result cyan (blue + green) on black.
        6. Add glow by blending a blurred copy — optical bloom effect.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Black-background BGR frame with glowing cyan edge lines.
        """
        # ── Pre-process for consistent edge detection ─────────────────
        enhanced = clahe_enhance(frame, clip_limit=2.0, tile_size=8)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # ── Canny edges (thin, precise) ───────────────────────────────
        edges_canny = cv2.Canny(gray, threshold1=45, threshold2=130)

        # ── Sobel gradient magnitude (broad, smooth) ─────────────────
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(gx**2 + gy**2)
        sobel_norm = cv2.normalize(
            sobel_mag, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # ── Combine: Canny for precision, Sobel for richness ──────────
        combined = cv2.addWeighted(edges_canny, 0.55, sobel_norm, 0.45, 0)

        # ── Render as cyan neon on black background ───────────────────
        edge_bgr = np.zeros_like(frame)
        edge_bgr[:, :, 0] = combined  # Blue component of cyan
        edge_bgr[:, :, 1] = combined  # Green component of cyan

        # ── Add neon glow via soft-light blending ─────────────────────
        glow_blur = cv2.GaussianBlur(edge_bgr, (7, 7), 0)
        result = np.clip(
            edge_bgr.astype(np.float32) + glow_blur.astype(np.float32) * 0.55,
            0, 255,
        )
        return result.astype(np.uint8)
