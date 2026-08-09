"""
Color Edge Overlay Vision Mode

A variation of AI edge detection that preserves full scene colour while
drawing thick dark outlines along detected edges — a cel-shading effect
similar to how some birds and cephalopods may enhance edge contrast in
their visual processing.

The same dual-detector pipeline (Canny + Sobel) used in AIEdgeVision
drives the edge map, but here edges are dilated into thick borders and
composited onto the original colour frame by darkening edge pixels rather
than replacing them with a monochrome overlay.
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode
from effects.contrast import clahe_enhance


class ColorEdgeVision(BaseVisionMode):
    """
    Full-colour scene with AI-detected edges rendered as thick dark outlines.
    """

    @property
    def name(self) -> str:
        return "Color Edge Overlay"

    @property
    def description(self) -> str:
        return "Full colour + thick AI edge outlines (cel-shading)."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Composites thick edge outlines onto the original colour frame.

        Steps:
        1. CLAHE  — normalises local contrast for reliable edge detection.
        2. Canny  — extracts thin, precise edge outlines.
        3. Sobel  — captures broader gradient transitions.
        4. Blend  — combine both maps for richer coverage.
        5. Threshold + dilate — convert to a thick binary edge mask.
        6. Composite — darken the original colour frame at edge pixels.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            BGR frame in full colour with thick dark outlines on edges.
        """
        # ── Pre-process ──────────────────────────────────────────────────
        enhanced = clahe_enhance(frame, clip_limit=2.0, tile_size=8)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # ── Canny edges (thin, precise) ──────────────────────────────────
        edges_canny = cv2.Canny(gray, threshold1=45, threshold2=130)

        # ── Sobel gradient magnitude (broad, smooth) ─────────────────────
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(gx ** 2 + gy ** 2)
        sobel_norm = cv2.normalize(
            sobel_mag, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # ── Combine: Canny for precision, Sobel for richness ─────────────
        combined = cv2.addWeighted(edges_canny, 0.55, sobel_norm, 0.45, 0)

        # ── Threshold → binary mask, then dilate for thick lines ─────────
        _, edge_mask = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thick_edges = cv2.dilate(edge_mask, kernel, iterations=2)

        # ── Darken original colour at edge positions ──────────────────────
        # weight: 1.0 on non-edges, down to 0.12 on thick edges
        weight = 1.0 - (thick_edges.astype(np.float32) / 255.0) * 0.88
        weight_3ch = weight[:, :, np.newaxis]
        result = np.clip(
            frame.astype(np.float32) * weight_3ch, 0, 255
        )
        return result.astype(np.uint8)
