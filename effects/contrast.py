"""
Contrast and Brightness Effects

Functions that adjust luminance rather than chrominance.
Kept separate from colour filters to honour the single-responsibility
principle and to make it obvious which stage of the pipeline is
responsible for a given visual artefact.
"""

import numpy as np
import cv2


def adjust_contrast_brightness(
    frame: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    Applies a linear contrast/brightness transform: output = alpha * input + beta.

    Args:
        frame: BGR image array.
        alpha: Contrast multiplier (1.0 = unchanged, > 1.0 = more contrast).
        beta: Brightness additive offset (−255 to 255).

    Returns:
        Adjusted BGR frame clipped to uint8.
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def gamma_correction(frame: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Applies gamma correction via a precomputed lookup table.

    gamma < 1.0 → brighter image (useful for night / low-light modes).
    gamma > 1.0 → darker image.

    A lookup table makes this O(256) rather than O(H × W), which is
    critical for real-time performance.

    Args:
        frame: BGR image array.
        gamma: Gamma value (typical range 0.3 – 3.0).

    Returns:
        Gamma-corrected BGR frame.
    """
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame, table)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Stretches pixel values to fill the full 0–255 range.

    Useful after mathematical operations that compress dynamic range,
    such as channel mixing or LUT applications.

    Args:
        frame: BGR image array (uint8 or float32).

    Returns:
        Normalised uint8 BGR frame.
    """
    f = frame.astype(np.float32)
    f_min, f_max = f.min(), f.max()
    if f_max > f_min:
        f = (f - f_min) / (f_max - f_min) * 255.0
    return f.astype(np.uint8)


def clahe_enhance(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE boosts local contrast while preventing over-amplification in
    flat/uniform regions.  It operates on the L (lightness) channel of
    LAB colour space so that hue and saturation are unaffected.

    Args:
        frame: BGR image array.
        clip_limit: Upper limit for contrast amplification per tile.
        tile_size: Side length of each local histogram tile.

    Returns:
        Contrast-enhanced BGR frame.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    l_enhanced = clahe.apply(l_ch)

    return cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
