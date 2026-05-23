"""
Color Filter Effects

Reusable colour transformation functions consumed by multiple vision modes.
All operations work directly on NumPy arrays via OpenCV and vectorised math —
no Python-level pixel loops.
"""

import numpy as np
import cv2


# ── Channel manipulation ──────────────────────────────────────────────────────

def apply_channel_weights(
    frame: np.ndarray,
    b: float,
    g: float,
    r: float,
) -> np.ndarray:
    """
    Scales each BGR channel independently.

    Simulates species-specific cone sensitivity differences by
    boosting or suppressing individual colour channels.

    Args:
        frame: BGR image array (H, W, 3) uint8.
        b: Blue channel multiplier  (0.0 – 2.0 typical).
        g: Green channel multiplier (0.0 – 2.0 typical).
        r: Red channel multiplier   (0.0 – 2.0 typical).

    Returns:
        Adjusted BGR frame, clipped to uint8 range.
    """
    result = frame.astype(np.float32)
    result[:, :, 0] *= b
    result[:, :, 1] *= g
    result[:, :, 2] *= r
    return np.clip(result, 0, 255).astype(np.uint8)


def shift_hue(frame: np.ndarray, degrees: float) -> np.ndarray:
    """
    Rotates the hue channel by the given number of degrees.

    Works in HSV colour space.  OpenCV stores hue in [0, 179] (half of
    the 360° circle), so a full rotation equals 180 OpenCV units.

    Args:
        frame: BGR image array.
        degrees: Degrees to rotate hue (negative = shift toward violet,
                 positive = shift toward red).

    Returns:
        Hue-rotated BGR frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + degrees / 2.0) % 180
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_saturation(frame: np.ndarray, factor: float) -> np.ndarray:
    """
    Multiplies the saturation channel by `factor`.

    factor < 1.0 → more muted / closer to grey
    factor > 1.0 → more vivid

    Args:
        frame: BGR image array.
        factor: Saturation multiplier.

    Returns:
        Saturation-adjusted BGR frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def to_grayscale_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Converts a BGR frame to greyscale while keeping 3 channels.

    Keeping 3 channels avoids shape mismatches when later blending with
    colour overlays.

    Args:
        frame: BGR image array.

    Returns:
        BGR frame where R == G == B == luminance.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── Colour-blindness simulation matrices ─────────────────────────────────────
# Source: Viénot, Brettel & Mollon (1999) and Brettel et al. (1997).
# Each matrix maps linear RGB to the RGB as perceived by the simulated observer.

_DEUTERANOPIA_MATRIX = np.array(
    [[0.625, 0.375, 0.000],
     [0.700, 0.300, 0.000],
     [0.000, 0.300, 0.700]],
    dtype=np.float32,
)

_PROTANOPIA_MATRIX = np.array(
    [[0.567, 0.433, 0.000],
     [0.558, 0.442, 0.000],
     [0.000, 0.242, 0.758]],
    dtype=np.float32,
)

_TRITANOPIA_MATRIX = np.array(
    [[0.950, 0.050, 0.000],
     [0.000, 0.433, 0.567],
     [0.000, 0.475, 0.525]],
    dtype=np.float32,
)


def _apply_cvd_matrix(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Applies a 3×3 colour-vision-deficiency simulation matrix.

    The matrix operates in linear RGB space (0–1 range), so the frame
    is normalised before multiplication and de-normalised after.

    Args:
        frame: BGR image array.
        matrix: 3×3 float32 simulation matrix (RGB → simulated RGB).

    Returns:
        Simulated BGR frame.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    simulated = np.clip(rgb @ matrix.T, 0, 1)
    return cv2.cvtColor((simulated * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def simulate_deuteranopia(frame: np.ndarray) -> np.ndarray:
    """
    Simulates deuteranopia (green-cone deficiency, ~1 % of males).

    Red and green hues become indistinguishable; the world appears
    in shades of blue and yellow.
    """
    return _apply_cvd_matrix(frame, _DEUTERANOPIA_MATRIX)


def simulate_protanopia(frame: np.ndarray) -> np.ndarray:
    """
    Simulates protanopia (red-cone deficiency, ~1 % of males).

    Red appears very dark; green and red are confused.
    """
    return _apply_cvd_matrix(frame, _PROTANOPIA_MATRIX)


def simulate_tritanopia(frame: np.ndarray) -> np.ndarray:
    """
    Simulates tritanopia (blue-cone deficiency, ~0.003 % of the population).

    Blue and green are confused; yellow and violet are confused.
    """
    return _apply_cvd_matrix(frame, _TRITANOPIA_MATRIX)


# ── Blending ──────────────────────────────────────────────────────────────────

def blend_frames(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linearly blends two frames: result = alpha * a + (1 − alpha) * b.

    Args:
        a: BGR frame.
        b: BGR frame with identical shape.
        alpha: Weight of frame `a` (0.0 = all b, 1.0 = all a).

    Returns:
        Blended BGR frame.
    """
    return cv2.addWeighted(a, alpha, b, 1.0 - alpha, 0)
