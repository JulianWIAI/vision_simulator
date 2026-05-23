"""
Image Utility Functions

Shared helpers for common image operations used across multiple modules.
Centralising these prevents duplication and provides a single location
to apply performance optimisations.
"""

import numpy as np
import cv2
from typing import Tuple
from PySide6.QtGui import QImage


def resize_frame(
    frame: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resizes a frame to the given dimensions.

    Args:
        frame: Input image array (any depth/channels).
        width: Target width in pixels.
        height: Target height in pixels.
        interpolation: OpenCV interpolation method.

    Returns:
        Resized image with the same number of channels.
    """
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Guarantees the frame is a 3-channel BGR image.

    Handles two common edge cases:
    - Grayscale (2-D array) → converted to 3-channel BGR.
    - BGRA (4 channels, e.g. from MSS on some platforms) → alpha dropped.

    Args:
        frame: Input image, any channel count.

    Returns:
        3-channel BGR uint8 array.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return frame[:, :, :3]
    return frame


def safe_uint8(array: np.ndarray) -> np.ndarray:
    """
    Clips values to [0, 255] and casts to uint8.

    Used after float arithmetic to prevent wrap-around artefacts that
    arise from silent integer overflow in NumPy.

    Args:
        array: Float or integer array of any shape.

    Returns:
        uint8 array with identical shape.
    """
    return np.clip(array, 0, 255).astype(np.uint8)


def frame_info(frame: np.ndarray) -> Tuple[int, int, int]:
    """
    Returns (height, width, channels) for a frame.

    Args:
        frame: Image array (2-D or 3-D).

    Returns:
        Tuple of (height, width, channels).
    """
    h, w = frame.shape[:2]
    c = frame.shape[2] if frame.ndim == 3 else 1
    return h, w, c


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """
    Converts a BGR numpy array to a Qt-owned QImage (RGB888).

    1. cv2.COLOR_BGR2RGB  — fast C-level channel swap.
    2. ascontiguousarray  — guarantees row-major layout required by QImage.
    3. q_img.copy()       — deep-copy into Qt's heap; source array freed after.

    Args:
        frame: BGR uint8 image array, shape (H, W, 3).

    Returns:
        QImage in Format_RGB888 with independent Qt-owned pixel data.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    q_img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return q_img.copy()
