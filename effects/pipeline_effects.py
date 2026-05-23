"""
Pipeline Effect Wrappers — Phase 6

Thin BaseVisionMode subclasses that adapt the stateless functions in
effects/blur.py, effects/contrast.py, and effects/overlays.py so they
can participate in a VisionPipeline effect chain.

Each wrapper:
  - Has a short name (≤ 20 chars) for HUD / UI dropdowns.
  - Stores its configuration as constructor args (immutable after creation).
  - Delegates entirely to the underlying effect function.

PIPELINE_EFFECTS is the canonical ordered list used to populate the
effect QComboBox in the control panel.  Index 0 is always the "None"
sentinel so the combo defaults to "no effect applied".
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from modes.base_mode import BaseVisionMode
from effects.blur import gaussian_blur, radial_blur
from effects.contrast import clahe_enhance
from effects.overlays import add_vignette, add_scan_lines, add_glow


# ── Effect wrappers ────────────────────────────────────────────────────────────

class GaussianBlurEffect(BaseVisionMode):
    """Uniform Gaussian blur — softens the output of the base mode."""

    def __init__(self, kernel_size: int = 15) -> None:
        self._kernel_size = kernel_size

    @property
    def name(self) -> str:
        return "Gaussian Blur"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return gaussian_blur(frame, kernel_size=self._kernel_size)


class HighContrastEffect(BaseVisionMode):
    """CLAHE local contrast boost — preserves hue, sharpens local detail."""

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8) -> None:
        self._clip_limit = clip_limit
        self._tile_size = tile_size

    @property
    def name(self) -> str:
        return "High Contrast"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return clahe_enhance(
            frame, clip_limit=self._clip_limit, tile_size=self._tile_size
        )


class VignetteEffect(BaseVisionMode):
    """Radial edge-darkening that focuses attention on the frame centre."""

    def __init__(self, strength: float = 0.6) -> None:
        self._strength = strength

    @property
    def name(self) -> str:
        return "Vignette"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return add_vignette(frame, strength=self._strength)


class ScanlineEffect(BaseVisionMode):
    """CRT / thermal-camera scan-line texture on every nth row."""

    def __init__(self, spacing: int = 4, alpha: float = 0.15) -> None:
        self._spacing = spacing
        self._alpha = alpha

    @property
    def name(self) -> str:
        return "Scan Lines"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return add_scan_lines(frame, spacing=self._spacing, alpha=self._alpha)


class GlowEffect(BaseVisionMode):
    """Bloom / halo effect around bright image regions."""

    def __init__(self, intensity: float = 0.4, blur_kernel: int = 21) -> None:
        self._intensity = intensity
        self._blur_kernel = blur_kernel

    @property
    def name(self) -> str:
        return "Glow"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return add_glow(
            frame, intensity=self._intensity, blur_kernel=self._blur_kernel
        )


class RadialBlurEffect(BaseVisionMode):
    """Peripheral blur increasing with distance from the frame centre."""

    def __init__(self, intensity: float = 0.5) -> None:
        self._intensity = intensity

    @property
    def name(self) -> str:
        return "Radial Blur"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return radial_blur(frame, intensity=self._intensity)


# ── Registry ───────────────────────────────────────────────────────────────────

# Ordered (display_name, instance_or_None) pairs for QComboBox population.
# Index 0 is always the "None" sentinel (no effect applied).
PIPELINE_EFFECTS: List[Tuple[str, Optional[BaseVisionMode]]] = [
    ("None",          None),
    ("Gaussian Blur", GaussianBlurEffect()),
    ("High Contrast", HighContrastEffect()),
    ("Vignette",      VignetteEffect()),
    ("Scan Lines",    ScanlineEffect()),
    ("Glow",          GlowEffect()),
    ("Radial Blur",   RadialBlurEffect()),
]
