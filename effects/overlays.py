"""
Overlay Effects

Graphical elements drawn on top of processed frames.
Overlays are the last stage in the pipeline so they always appear
above the mode's colour and filter transformations.

All heavy geometry (hex grids, radial maps) is computed once and
cached inside module-level dicts keyed by frame resolution, so the
cost is paid only on the first frame at each resolution.
"""

import numpy as np
import cv2
import math
from typing import Tuple

# Module-level caches keyed by (height, width) to avoid recomputation.
_hex_cache: dict = {}
_vignette_cache: dict = {}


def add_vignette(frame: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Darkens frame edges to create a vignette effect.

    A radial gradient multiplies the frame so the centre retains full
    brightness while the periphery fades to black.  The gradient is
    cached after the first call so subsequent frames are cheap.

    Args:
        frame: BGR image array.
        strength: Vignette darkness at the border (0 = none, 1 = black).

    Returns:
        Frame with vignette applied.
    """
    h, w = frame.shape[:2]
    key = (h, w, round(strength, 3))

    if key not in _vignette_cache:
        cx, cy = w / 2.0, h / 2.0
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt(((x_coords - cx) / cx) ** 2 +
                       ((y_coords - cy) / cy) ** 2)
        weight = 1.0 - np.clip(dist * strength, 0, 1)
        _vignette_cache[key] = np.stack([weight, weight, weight], axis=2).astype(np.float32)

    result = frame.astype(np.float32) * _vignette_cache[key]
    return result.astype(np.uint8)


def add_grid_overlay(
    frame: np.ndarray,
    cell_size: int = 20,
    color: Tuple[int, int, int] = (0, 200, 0),
    alpha: float = 0.2,
) -> np.ndarray:
    """
    Draws a semi-transparent rectangular grid over the frame.

    Args:
        frame: BGR image array.
        cell_size: Pixel size of each grid cell.
        color: BGR grid-line colour.
        alpha: Grid opacity (0 = invisible, 1 = fully opaque).

    Returns:
        Frame with grid overlay.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for x in range(0, w, cell_size):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, cell_size):
        cv2.line(overlay, (0, y), (w, y), color, 1)
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def add_hexagonal_overlay(
    frame: np.ndarray,
    hex_radius: int = 18,
    alpha: float = 0.30,
) -> np.ndarray:
    """
    Draws a semi-transparent hexagonal grid to simulate compound-eye facets.

    The grid image is cached per (resolution, radius) so the expensive
    Python-loop geometry is only computed once.

    Args:
        frame: BGR image array.
        hex_radius: Circumscribed radius of each hexagon in pixels.
        alpha: Grid opacity.

    Returns:
        Frame with hexagonal facet overlay.
    """
    h, w = frame.shape[:2]
    key = (h, w, hex_radius)

    if key not in _hex_cache:
        grid = np.zeros((h, w, 3), dtype=np.uint8)
        hex_h = int(hex_radius * math.sqrt(3))
        hex_w = hex_radius * 2

        rows = h // hex_h + 3
        cols = w // (hex_w * 3 // 4) + 3

        for row in range(-1, rows):
            for col in range(-1, cols):
                cx = col * hex_w * 3 // 4
                cy = row * hex_h + (hex_h // 2 if col % 2 else 0)
                pts = []
                for i in range(6):
                    angle = math.radians(60 * i)
                    px = int(cx + hex_radius * math.cos(angle))
                    py = int(cy + hex_radius * math.sin(angle))
                    pts.append([px, py])
                pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(grid, [pts_arr], True, (0, 180, 0), 1)

        _hex_cache[key] = grid

    return cv2.addWeighted(frame, 1.0, _hex_cache[key], alpha, 0)


def add_scan_lines(
    frame: np.ndarray,
    spacing: int = 4,
    alpha: float = 0.15,
) -> np.ndarray:
    """
    Darkens every `spacing`-th row to create a CRT / thermal-camera look.

    This is implemented by direct array indexing (no loop), so it has
    negligible overhead even at 4K resolution.

    Args:
        frame: BGR image array.
        spacing: Row interval between scan lines.
        alpha: Darkness of each scan line (0 = invisible, 1 = black).

    Returns:
        Frame with scan-line texture.
    """
    result = frame.copy()
    result[::spacing] = (result[::spacing].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return result


def add_glow(
    frame: np.ndarray,
    intensity: float = 0.4,
    blur_kernel: int = 21,
) -> np.ndarray:
    """
    Adds a bloom / halo effect around the brightest regions.

    Simulates optical halation — the light bleed visible around bright
    sources in real cameras and heat-sensor imagers.

    Args:
        frame: BGR image array.
        intensity: Glow brightness multiplier.
        blur_kernel: Size of the spreading Gaussian kernel.

    Returns:
        Frame with glow applied to bright areas.
    """
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_3ch = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    glow = blurred.astype(np.float32) * bright_3ch * intensity
    return np.clip(frame.astype(np.float32) + glow, 0, 255).astype(np.uint8)
