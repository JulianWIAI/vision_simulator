"""
Mathematical Utility Functions

Low-level numerical operations shared by image-processing modules.

Design intent: this module is the designated insertion point for
future SciPy or GPU-accelerated replacements.  When scipy.ndimage or
CuPy becomes available, swap the implementations here without touching
any mode or effect code.

Current implementations rely solely on NumPy for zero extra dependencies.
"""

import numpy as np
from typing import Tuple


def normalize_to_range(
    array: np.ndarray,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """
    Linearly stretches an array to the range [out_min, out_max].

    Args:
        array: Input array (any dtype).
        out_min: Lower bound of the output range.
        out_max: Upper bound of the output range.

    Returns:
        float32 array with the same shape as input.
    """
    arr = array.astype(np.float32)
    a_min, a_max = arr.min(), arr.max()
    if a_max == a_min:
        # Constant array — map to midpoint to avoid division by zero.
        return np.full_like(arr, (out_min + out_max) / 2.0)
    return (arr - a_min) / (a_max - a_min) * (out_max - out_min) + out_min


def gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """
    Creates a normalised 2-D Gaussian kernel.

    Note: OpenCV's GaussianBlur is faster for most real-time use cases.
    This function exists for situations that require precise sigma control
    or explicit kernel inspection (e.g., unit tests, visualisation).

    Args:
        size: Side length of the square kernel (must be odd).
        sigma: Standard deviation of the Gaussian distribution.

    Returns:
        float32 array of shape (size, size) whose values sum to 1.
    """
    k = size // 2
    y, x = np.ogrid[-k : k + 1, -k : k + 1]
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)).astype(np.float32)
    return kernel / kernel.sum()


def build_radial_weight_map(
    h: int,
    w: int,
    exponent: float = 2.0,
) -> np.ndarray:
    """
    Builds a 2-D radial weight map: 1.0 at the centre, ≈ 0.0 at the edges.

    Used by vignette and radial-blur effects.  Higher exponents create
    a sharper drop-off near the boundary.

    Args:
        h: Frame height in pixels.
        w: Frame width in pixels.
        exponent: Controls falloff steepness (2.0 = smooth quadratic).

    Returns:
        float32 array of shape (h, w) with values in [0, 1].
    """
    cy, cx = h / 2.0, w / 2.0
    y_coords, x_coords = np.ogrid[:h, :w]
    # Normalise distances so the corner of the frame maps to distance = 1.
    dist = np.sqrt(((x_coords - cx) / cx) ** 2 + ((y_coords - cy) / cy) ** 2)
    return np.clip(1.0 - dist**exponent, 0.0, 1.0).astype(np.float32)


def interpolate_lut(lut: np.ndarray, num_entries: int = 256) -> np.ndarray:
    """
    Resamples a colour LUT to a different number of entries.

    Useful when loading third-party LUTs (e.g., .cube files) that use
    a different quantisation than uint8 images.

    Args:
        lut: Input LUT of shape (N, 3).
        num_entries: Desired output length (default 256 for uint8 images).

    Returns:
        Resampled float32 LUT of shape (num_entries, 3).
    """
    n = len(lut)
    if n == num_entries:
        return lut.astype(np.float32)

    indices = np.linspace(0, n - 1, num_entries)
    resampled = np.zeros((num_entries, 3), dtype=np.float32)
    for ch in range(3):
        resampled[:, ch] = np.interp(indices, np.arange(n), lut[:, ch])
    return resampled
