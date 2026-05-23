"""
Blur Effects

Reusable blur functions consumed by multiple vision modes.
Separating blur logic from mode logic means the same operations
can be shared without duplication.
"""

import numpy as np
import cv2


def _odd(k: int) -> int:
    """Returns k if it is odd, otherwise k + 1 (OpenCV kernel-size requirement)."""
    return k if k % 2 == 1 else k + 1


def gaussian_blur(
    frame: np.ndarray,
    kernel_size: int = 15,
    sigma: float = 0,
) -> np.ndarray:
    """
    Applies a uniform Gaussian blur to the entire frame.

    Gaussian blur is preferred over box blur because it avoids ringing
    artefacts and better models optical defocus.

    Args:
        frame: BGR image array.
        kernel_size: Blur kernel side length (will be made odd if even).
        sigma: Standard deviation. 0 = auto-derived from kernel_size.

    Returns:
        Blurred BGR frame.
    """
    k = _odd(kernel_size)
    return cv2.GaussianBlur(frame, (k, k), sigma)


def selective_blur(
    frame: np.ndarray,
    mask: np.ndarray,
    blur_strength: int = 21,
) -> np.ndarray:
    """
    Blurs only the pixels where `mask` is non-zero (white).

    Used in Pit Viper mode to keep hot (bright) regions sharp while
    blurring cold (dark) regions — simulating the coarse spatial
    resolution of pit organs.

    Args:
        frame: BGR image array.
        mask: Single-channel uint8 mask.
               White (255) = apply blur; Black (0) = keep sharp.
        blur_strength: Gaussian kernel size for the blur pass.

    Returns:
        Frame where blurred and sharp regions are composited via the mask.
    """
    k = _odd(blur_strength)
    blurred = cv2.GaussianBlur(frame, (k, k), 0)

    # Expand mask to 3 channels for per-pixel blending.
    weight = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # where weight=1 → blurred; where weight=0 → original
    result = (blurred.astype(np.float32) * weight +
              frame.astype(np.float32) * (1.0 - weight))
    return result.astype(np.uint8)


def radial_blur(
    frame: np.ndarray,
    intensity: float = 0.5,
) -> np.ndarray:
    """
    Applies blur that increases with distance from the frame centre.

    Simulates tunnel vision or myopia where peripheral regions are
    out of focus while the central field remains sharp.

    Args:
        frame: BGR image array.
        intensity: How rapidly blur increases toward the edges (0–1).

    Returns:
        Radially blurred BGR frame.
    """
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt(((x_coords - cx) / cx) ** 2 +
                   ((y_coords - cy) / cy) ** 2)
    dist = np.clip(dist * intensity, 0, 1).astype(np.float32)

    blurred_mild   = cv2.GaussianBlur(frame, (9, 9), 0)
    blurred_strong = cv2.GaussianBlur(frame, (31, 31), 0)

    alpha = np.stack([dist, dist, dist], axis=2)
    result = (blurred_strong.astype(np.float32) * alpha +
              blurred_mild.astype(np.float32) * (1.0 - alpha))
    return result.astype(np.uint8)


def motion_blur(
    frame: np.ndarray,
    kernel_size: int = 15,
    angle: float = 0.0,
) -> np.ndarray:
    """
    Applies directional motion blur along `angle` degrees.

    Simulates the smearing visible when a camera (or eye) pans rapidly,
    or models species with poor temporal resolution that integrate motion.

    Args:
        frame: BGR image array.
        kernel_size: Length of the motion trail in pixels.
        angle: Direction of the blur in degrees (0 = horizontal).

    Returns:
        Motion-blurred BGR frame.
    """
    k = _odd(kernel_size)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k

    M = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    return cv2.filter2D(frame, -1, kernel)
