"""
LUT (Look-Up Table) Effects

LUTs provide fast, precomputed colour transformations.  Instead of
recalculating per-pixel colour mappings at runtime, a LUT maps every
possible input value (0–255) to its output colour in a single vectorised
step via cv2.LUT().

All LUTs produced here are shaped (256, 1, 3) in BGR order as expected
by cv2.LUT() when applied to 3-channel images.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional


def build_thermal_lut() -> np.ndarray:
    """
    Builds a custom thermal gradient LUT.

    Colour stops (cold → hot):
        Black → Deep blue → Blue → Cyan → Yellow → Orange → Red → White

    This gradient maps grayscale intensity to perceived temperature:
    - 0   = absolute cold (black)
    - ~64 = cold (blue)
    - ~128 = intermediate (green/yellow)
    - ~224 = hot (orange/red)
    - 255  = maximum heat (white)

    Returns:
        LUT array of shape (256, 1, 3) in BGR format, ready for cv2.LUT().
    """
    # Each stop: (position 0-255, Blue, Green, Red)
    stops = [
        (0,   0,   0,   0),      # Black        – absolute cold
        (40,  160, 0,   0),      # Dark blue
        (80,  255, 0,   0),      # Blue
        (110, 255, 140, 0),      # Cyan
        (140, 20,  255, 20),     # Green (brief transition)
        (170, 0,   220, 180),    # Yellow-green
        (200, 0,   160, 255),    # Yellow → orange
        (225, 0,   60,  255),    # Deep orange
        (240, 0,   0,   255),    # Red
        (255, 255, 255, 255),    # White         – maximum heat
    ]

    lut = np.zeros((256, 3), dtype=np.uint8)

    for i in range(len(stops) - 1):
        p0, b0, g0, r0 = stops[i]
        p1, b1, g1, r1 = stops[i + 1]
        span = max(p1 - p0, 1)
        for p in range(p0, p1 + 1):
            t = (p - p0) / span
            lut[p] = [
                int(b0 + t * (b1 - b0)),
                int(g0 + t * (g1 - g0)),
                int(r0 + t * (r1 - r0)),
            ]

    return lut.reshape(256, 1, 3)


def build_night_vision_lut() -> np.ndarray:
    """
    Builds a phosphor green night-vision LUT.

    Maps each intensity value to a green shade, mimicking the look
    of Generation-I image-intensifier night-vision goggles.

    Returns:
        LUT array of shape (256, 1, 3) in BGR format.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        # Slight blue-green tint to match real NVG phosphor screens
        lut[i] = [int(i * 0.2), i, int(i * 0.3)]
    return lut.reshape(256, 1, 3)


def apply_lut(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Applies a per-channel colour LUT to a BGR frame.

    cv2.LUT processes all channels simultaneously without Python loops,
    making it one of the fastest colour transforms available.

    Args:
        frame: BGR image array (H, W, 3) uint8.
        lut: LUT array of shape (256, 1, 3) uint8.

    Returns:
        LUT-transformed BGR frame.
    """
    return cv2.LUT(frame, lut)


def apply_grayscale_lut(gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Applies a colour LUT to a single-channel grayscale image.

    In thermal modes the grayscale channel encodes temperature, and
    the LUT maps each temperature level to a display colour.

    Args:
        gray: Grayscale image (H, W) uint8.
        lut: LUT of shape (256, 1, 3) uint8 in BGR format.

    Returns:
        Colour-mapped BGR image.
    """
    # cv2.LUT requires a 3-channel input; replicate the grayscale.
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.LUT(gray_3ch, lut)


def load_lut_from_file(path: str) -> Optional[np.ndarray]:
    """
    Loads a colour LUT from a .npy file saved by np.save().

    This is the hook for loading custom LUTs placed in assets/luts/.
    A future update can extend this to parse .cube (Adobe/DaVinci)
    and .3dl (Lustre) formats.

    Args:
        path: Path to the .npy file containing a (256, 1, 3) uint8 array.

    Returns:
        LUT array, or None if the file does not exist or cannot be loaded.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return np.load(str(p)).astype(np.uint8)
    except Exception:
        return None
