"""
Pit Viper Vision Mode  ★ HIGH-PRIORITY FEATURE ★

Pit vipers (rattlesnakes, copperheads, bushmasters, fer-de-lances)
have a pair of pit organs located in a cavity between the eye and
nostril.  Each organ contains a thin membrane densely packed with
heat-sensitive TRPA1 ion channels that respond to IR wavelengths of
5–30 μm — the thermal radiation emitted by warm-blooded prey.

Physiological properties accurately modelled here
─────────────────────────────────────────────────
1. Low spatial resolution
   The membrane contains only ~700 receptor cells per pit, giving an
   angular resolution of roughly 5°.  Cold background regions therefore
   appear blurry/diffuse while hot regions are relatively sharper
   (the temperature *gradient* drives the strongest neural response).

2. Differential temperature sensitivity
   Pit organs respond primarily to temperature *differences* rather
   than absolute values.  Histogram equalisation in this pipeline
   exaggerates subtle gradients, matching that differential sensitivity.

3. Custom thermal colour map
   A hand-crafted LUT maps cold → deep blue, intermediate → yellow/orange,
   hot → bright red/white — the conventional false-colour thermal palette.

4. Glow on hottest regions
   Real thermal imagers show halation (light bleed) around bright heat
   sources.  The add_glow pass replicates this optical artefact.

5. Scan-line texture
   Adds a subtle CRT/sensor-grid aesthetic seen in real FLIR imagery.
"""

import numpy as np
import cv2
from modes.base_mode import BaseVisionMode
from effects.lut import build_thermal_lut, apply_grayscale_lut
from effects.blur import selective_blur, gaussian_blur
from effects.overlays import add_scan_lines, add_glow


class PitViperVision(BaseVisionMode):
    """
    High-fidelity pit viper infrared vision simulation.

    Hot (bright) regions → sharp, red/white.
    Cold (dark) regions  → blurry, deep blue.
    """

    # Temperature thresholds for the cold/warm mask (0–255 normalised scale).
    COLD_THRESHOLD = 90    # Below this → "cold" → heavy blur
    HOT_THRESHOLD  = 170   # Above this → "hot"  → kept sharp

    def __init__(self) -> None:
        # Build the LUT once at construction time and reuse every frame.
        # This avoids rebuilding the 256-entry interpolation on every tick.
        self._lut = build_thermal_lut()

    @property
    def name(self) -> str:
        return "Pit Viper Vision"

    @property
    def description(self) -> str:
        return "IR pit-organ: hot=sharp/red, cold=blurry/blue, scan lines."

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pit viper infrared perception pipeline.

        Stage 1 — Extract thermal signal
            Convert to greyscale.  Luminance is the best single-channel
            proxy for surface temperature in a visible-light image.

        Stage 2 — Maximise differential contrast
            cv2.equalizeHist stretches the global histogram so subtle
            temperature differences become visible, mirroring the high
            differential (rather than absolute) sensitivity of the pit organ.

        Stage 3 — Thermal false-colour mapping
            Apply the custom LUT: cold = deep blue, intermediate = yellow,
            hot = red/white.

        Stage 4 — Build cold-region mask (NumPy thresholding)
            Pixels below COLD_THRESHOLD are labelled "cold" and will
            receive a heavy Gaussian blur — simulating the pit organ's
            inability to resolve fine spatial detail in low-heat zones.

        Stage 5 — Selective blur
            Hot regions stay sharp; cold regions lose spatial resolution.
            This is the single most physiologically important step.

        Stage 6 — Scan-line texture
            Adds a subtle horizontal darkening pattern mimicking the
            scan-line readout of a real FLIR/LWIR sensor array.

        Stage 7 — Glow on hottest pixels
            Bright heat sources show halation (optical bloom) in real
            thermal imagers.  add_glow replicates this.

        Args:
            frame: Raw BGR screen capture.

        Returns:
            Processed BGR frame simulating pit viper IR perception.
        """
        # ── Stage 1: Extract thermal signal ──────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Stage 2: Differential contrast via histogram equalisation ─
        # equalizeHist spreads the histogram across the full 0-255 range,
        # making fine temperature gradients visible — matching the known
        # differential (not absolute) sensitivity of pit-organ receptors.
        equalised = cv2.equalizeHist(gray)

        # ── Stage 3: False-colour thermal mapping ─────────────────────
        thermal = apply_grayscale_lut(equalised, self._lut)

        # ── Stage 4: Build cold-region mask ───────────────────────────
        # NumPy boolean → uint8 mask: 255 where cold, 0 where warm/hot.
        cold_mask = np.where(equalised < self.COLD_THRESHOLD, 255, 0).astype(np.uint8)

        # Dilate the mask slightly so the blur feathers into warm regions,
        # avoiding a hard boundary that would look artificial.
        kernel = np.ones((7, 7), np.uint8)
        cold_mask = cv2.dilate(cold_mask, kernel, iterations=1)

        # ── Stage 5: Selective blur (the physiological core) ──────────
        # Cold areas → 27-pixel Gaussian blur (low spatial resolution).
        # Hot areas  → untouched (sharp thermal gradient response).
        result = selective_blur(thermal, cold_mask, blur_strength=27)

        # ── Stage 6: Scan-line texture ────────────────────────────────
        result = add_scan_lines(result, spacing=3, alpha=0.10)

        # ── Stage 7: Glow on hottest pixels ───────────────────────────
        result = add_glow(result, intensity=0.35, blur_kernel=17)

        return result
