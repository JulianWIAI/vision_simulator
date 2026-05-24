"""
hd_engine/analyzer.py  —  v1.1
Core Computer Vision module — extracts dimensional feature vectors from image bytes.

v1.0 pipeline (unchanged):
  Dim 5  → Dominant colour channel (RGB mean analysis)
  Dim 6-8 → Edge density / texture rhythm (Sobel gradient)
  Dim 9  → Shape angularity (FFT high-vs-low frequency energy ratio)
  Dim 10 → Impulse rate derived from edge density
  SUSY   → Bilateral symmetry score (left/right half pixel diff)

v1.1 additions (Quantum Field analysis):
  Particles  → Circle (Stabilization Atom) and Square (Hyperdimensional Cube) density
  Stability  → Circle-complement pairing check → Stable / Unstable
  Gravity    → Black-zone density → Gravitational Field strength
  EM Break   → Yellow pixel ratio + jagged edge signature
  Oscillating→ Alternating 2-3-2-3 strip density rhythm
  Luxury     → Black + Silver + White simultaneous presence → High-Value multiplier
  QV         → Composite Quantum Value multiplier

All heavy lifting uses NumPy + Pillow only — no OpenCV dependency needed.
"""

import io
import numpy as np
from PIL import Image

from hd_engine.models import (
    ColorComboValue,
    DetectedObject,
    Dim5Color,
    Dim68Rhythm,
    Dim9Geometry,
    Dim10Frequency,
    LocalAsymmetry,
    ObjectAnalysis,
    ParticleCount,
    QuantumFieldMetrics,
    SymmetryAnalysis,
)


class HyperdimensionalAnalyzer:
    """
    Stateful analyzer: construct once per request with raw image bytes,
    then call run_full_analysis() to get all dimensional vectors.
    """

    # Maximum long-edge size before downscaling (keeps analysis fast on large uploads)
    MAX_DIMENSION = 512

    # ── Quantum-field detection thresholds ────────────────────────────────────
    _DARK_THRESHOLD: float = 38.0          # luminance below this = "Black" pixel
    _WHITE_THRESHOLD: float = 215.0        # luminance above this = "White" pixel
    _GRAY_CHANNEL_TOLERANCE: float = 18.0  # max channel deviation for "Silver/Gray"
    _YELLOW_R_MIN: float = 170.0           # Yellow: high red channel
    _YELLOW_G_MIN: float = 155.0           # Yellow: high green channel
    _YELLOW_B_MAX: float = 90.0            # Yellow: suppressed blue channel
    _SMOOTH_VARIANCE_THRESHOLD: float = 220.0  # patch variance below this = circular region
    _ANGULAR_VARIANCE_THRESHOLD: float = 600.0  # patch variance above this = rectangular region

    def __init__(self, image_bytes: bytes) -> None:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Downscale proportionally if the image is very large
        w, h = img.size
        if max(w, h) > self.MAX_DIMENSION:
            scale = self.MAX_DIMENSION / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        self.image = img
        self.array: np.ndarray = np.array(img, dtype=np.float32)   # shape (H, W, 3)
        self.height, self.width = self.array.shape[:2]

        # Pre-compute grayscale once — used by multiple v1.1 methods
        self._gray: np.ndarray = self.array.mean(axis=2)  # (H, W)

    # ──────────────────────────────────────────────────────────────────────────
    # v1.28  Core analysis helpers (shared by Dim 5, Dim 9, and object detector)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_focus_zone(self, sector_mode: str) -> tuple[int, int, int, int] | None:
        """
        Returns (x1, y1, x2, y2) in pixel coords for the sector's analysis core zone.

        CARDS:      Exclude 20 % left/right and 22 % top/bottom — large enough to fully
                    mask the coloured card frame on any standard trading card aspect ratio.
        INDIVIDUUM: Central body column; exclude far edges and the lower 15 % (legs/floor).
        """
        if sector_mode == "CARDS":
            # 20 % horizontal, 22 % vertical margins strip the card border reliably
            mx = int(self.width  * 0.20)
            my = int(self.height * 0.22)
            return (mx, my, self.width - mx, self.height - my)
        if sector_mode == "INDIVIDUUM":
            mx = int(self.width  * 0.15)
            my = int(self.height * 0.05)
            return (mx, my, int(self.width * 0.85), int(self.height * 0.85))
        return None

    def _fft_curvature(self, gray: np.ndarray) -> float:
        """High-freq / total-freq ratio via 2-D FFT — used for Dim 9 curvature index."""
        if gray.size < 16:
            return 0.5
        fft_magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 5
        yy, xx = np.ogrid[:h, :w]
        center_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        lo = float(fft_magnitude[center_mask].sum())
        hi = float(fft_magnitude[~center_mask].sum())
        t  = lo + hi
        return round(hi / t, 3) if t > 0 else 0.5

    def _hsv_sat_votes(self, array: np.ndarray) -> dict:
        """
        Saturation-weighted hue-bin voting for an RGB array (H×W×3, float32 0-255).

        v1.3: hue computed on the OpenCV 0-180° scale (each standard degree halved)
        so that all bin boundaries match the OpenCV convention used when specifying
        species-specific thresholds.  Raw S/V comparisons use normalised [0,1] floats
        that correspond to OpenCV's 0-255 uint8 scale via simple /255 division.

        Bin layout (OpenCV hue, [0,180]):
          yellow  — H 20-35,  S ≥ 0.549 (140/255), V ≥ 0.510 (130/255)   vivid yellow signal
          orange  — H  5-19,  S ≥ 0.588 (150/255), V ≥ 0.588             vivid orange / fire tones
          brown   — H  0-18,  warm hue not captured as yellow or orange    earth tones / tan / beige
          red_wrap— H 170-180 and 0-4, high S+V                           pure red edges
          green   — H 36-90
          cyan    — H 90-105
          blue    — H 105-133
          violet  — H 133-155
          pink    — H 155-170

        White exclusion mask (S < 0.118, V > 0.784) removes card text-field
        backgrounds before any vote is tallied.

        Returns dict keys: yellow, orange, brown, green, cyan, blue, violet, pink, total_sat
        """
        _EMPTY = {k: 0.0 for k in (
            "yellow", "orange", "brown", "green", "cyan", "blue", "violet", "pink", "total_sat"
        )}
        if array.size == 0:
            return _EMPTY

        # Normalise RGB channels to [0, 1]
        r = array[:, :, 0].astype(np.float32) / 255.0
        g = array[:, :, 1].astype(np.float32) / 255.0
        b = array[:, :, 2].astype(np.float32) / 255.0

        cmax  = np.maximum(np.maximum(r, g), b)   # HSV Value  ∈ [0,1]
        cmin  = np.minimum(np.minimum(r, g), b)
        delta = (cmax - cmin).astype(np.float32)
        sat   = np.where(cmax > 0, delta / cmax, 0.0).astype(np.float32)  # HSV Sat ∈ [0,1]
        val   = cmax                                                         # HSV Val ∈ [0,1]

        # ── White / off-white exclusion ────────────────────────────────────────
        # S < 30/255 = 0.118  AND  V > 200/255 = 0.784  →  card text-field background.
        # These pixels carry zero chromatic information and must not skew the vote.
        white_mask = (sat < 0.118) & (val > 0.784)

        # Chromatic gate: S ≥ 40/255 = 0.157, not white background
        chromatic = (sat >= 0.157) & (~white_mask)

        # ── Hue on OpenCV 0-180° scale  (standard 360° / 2) ───────────────────
        hue = np.zeros(array.shape[:2], dtype=np.float32)
        mr  = chromatic & (cmax == r) & (delta > 0)
        mg  = chromatic & (cmax == g) & (delta > 0)
        mb  = chromatic & (cmax == b) & (delta > 0)
        hue[mr] = (30.0 * ((g[mr] - b[mr]) / delta[mr])) % 180.0   # 60/2 = 30
        hue[mg] = (30.0 * ((b[mg] - r[mg]) / delta[mg]) + 60.0) % 180.0   # 120/2 = 60
        hue[mb] = (30.0 * ((r[mb] - g[mb]) / delta[mb]) + 120.0) % 180.0  # 240/2 = 120

        # Saturation-weighted vote: vivid pixels outweigh desaturated ones
        sw = (sat * chromatic.astype(np.float32)).astype(np.float32)

        def _w(mask: np.ndarray) -> float:
            return float((sw * mask.astype(np.float32)).sum())

        # ── Warm sub-bins — classification order: yellow → orange → brown ────────
        #
        # Priority logic:
        #   1. Yellow is defined first with tight S+V gates (vivid yellow hue range).
        #   2. Vivid orange is defined next with high S+V gates (fire-tone range).
        #   3. Brown captures ALL remaining warm pixels: any H in the warm range that
        #      is NOT yellow and NOT vivid orange.  This intentionally covers the full
        #      value spectrum — from dark chocolate to light tan/beige (earth tones)
        #      — without S/V upper-bound clipping that previously excluded lighter pixels.
        #
        # Example reference points (OpenCV hue scale, 0-180°):
        #   Vivid yellow    RGB(255,220,  0): H≈26, S=255, V=255  → YELLOW ✓
        #   Earth brown     RGB(150,115, 75): H≈16, S=128, V=150  → BROWN  ✓
        #   Light tan       RGB(190,160,120): H≈15, S= 67, V=190  → BROWN  ✓
        #   Vivid orange    RGB(230, 95, 15): H≈12, S=237, V=230  → ORANGE ✓
        #   Card frame yel  RGB(245,215, 60): H≈27, S=216, V=245  → YELLOW ✓
        #
        # S/V comparisons use 0-255 scaled floats to match OpenCV integer thresholds
        # and avoid floating-point rounding errors (e.g. 150/255 = 0.5882… > 0.588).

        sat_255 = sat * 255.0   # maps sat [0,1] → OpenCV S [0,255] scale
        val_255 = val * 255.0   # maps val [0,1] → OpenCV V [0,255] scale

        # Vivid yellow: H 20-35, S ≥ 140, V ≥ 130 — raised S gate separates vivid yellow
        # from muted cream/tan tones (S ≈ 100-130) that share the same hue range.
        yellow = (
            chromatic
            & (hue >= 20) & (hue <= 35)
            & (sat_255 >= 140.0)
            & (val_255 >= 130.0)
        )

        # Vivid orange: H 5-19, S ≥ 150, V ≥ 150
        # Red-wrap zone (H 170-180 or 0-4) also funnels into orange at high S+V.
        orange_mid = (
            chromatic & (hue >= 5) & (hue < 20)
            & (sat_255 >= 150.0) & (val_255 >= 150.0)
        )
        red_wrap = (
            chromatic & ((hue >= 170) | (hue < 5))
            & (sat_255 >= 150.0) & (val_255 >= 150.0)
        )
        orange = orange_mid | red_wrap

        # Brown / earth tones: any warm-hue pixel NOT already captured as yellow or orange.
        # Covers the full brightness range: dark chocolate, medium brown, light tan, beige.
        # No S/V upper bounds — those were a prior source of false negatives on
        # lighter earth tones (e.g. feathers or fur at V ≈ 190/255).
        warm_hue = chromatic & (hue >= 0) & (hue < 22)
        brown    = warm_hue & ~yellow & ~orange

        return {
            "yellow":    _w(yellow),
            "orange":    _w(orange),
            "brown":     _w(brown),
            "green":     _w(chromatic & (hue >= 36)  & (hue < 90)),
            "cyan":      _w(chromatic & (hue >= 90)  & (hue < 105)),
            "blue":      _w(chromatic & (hue >= 105) & (hue < 133)),
            "violet":    _w(chromatic & (hue >= 133) & (hue < 155)),
            "pink":      _w(chromatic & (hue >= 155) & (hue < 170)),
            "total_sat": float(sw.sum()),
        }

    def _classify_color_from_votes(
        self,
        votes: dict,
        source_array: np.ndarray | None = None,
    ) -> tuple[str, str, str, float]:
        """
        Maps saturation-weighted hue-bin votes to a 4-tuple:
            (dominant_frequency, color_label, semantic_state, vector_value)

        dominant_frequency — broad family (Orange | Green | Blue) kept for vocabulary lookup.
        color_label        — specific hue (Yellow | Orange | Brown | Pink | Green |
                             Cyan | Blue | Violet) used for UI colour rendering.
        semantic_state     — human-readable sub-classification string.
        vector_value       — normalised channel mean [0, 1].

        source_array: pixel array for channel-mean computation.
                      Falls back to self.array (full image) when None.
        """
        arr       = source_array if source_array is not None else self.array
        total_sat = max(votes.get("total_sat", 0.0), 1e-6)
        MIN_W     = total_sat * 0.07   # minimum 7 % of total saturation weight to qualify

        yellow = votes.get("yellow", 0.0)
        orange = votes.get("orange", 0.0)
        brown  = votes.get("brown",  0.0)
        green  = votes.get("green",  0.0)
        cyan   = votes.get("cyan",   0.0)
        blue   = votes.get("blue",   0.0)
        violet = votes.get("violet", 0.0)
        pink   = votes.get("pink",   0.0)

        warm   = orange + yellow          # combined warm signal
        cool   = blue + cyan
        pink_m = pink + violet * 0.4      # violet bleeds partially into pink/purple tier

        # ── Achromatic / near-neutral fallback ────────────────────────────────
        # No single bin clears the minimum threshold — fall back to raw channel means.
        if max(warm, cool, green, pink_m, brown) < MIN_W * 0.6:
            means = [float(arr[:, :, c].mean()) for c in range(3)]
            idx   = int(np.argmax(means))
            labels = [
                ("Orange", "Orange", "Thermal Baseline / Achromatic Warm — residual heat signature"),
                ("Green",  "Green",  "Subharmonic Baseline / Neutral Composite"),
                ("Blue",   "Blue",   "Achromatic Field / Cognitive Reset"),
            ]
            freq, label, state = labels[idx]
            return freq, label, state, round(means[idx] / 255.0, 3)

        # ── Brown / earth tones ────────────────────────────────────────────────
        # Evaluated before the general warm block so muted browns do not get
        # overridden by a small vivid-orange or yellow component.
        # Raised ratio to 1.5× to prevent a small yellow area from losing to a
        # large earth-tone background when the subject is genuinely yellow.
        if (
            brown > MIN_W
            and brown >= orange
            and brown >= yellow * 1.5
            and brown >= green
            and brown >= cool
        ):
            state = "Earth-Band / Stasis Field — subdued spectral expression, energetic stagnation"
            vv    = round(float(arr[:, :, 0].mean()) / 255.0, 3)
            return "Orange", "Brown", state, vv

        # ── Pink / Magenta ─────────────────────────────────────────────────────
        if pink_m > MIN_W and pink_m >= warm * 0.85 and pink_m >= green and pink_m >= cool:
            state = "Pink-Band / Magenta Interference — low-threat aesthetic attractor"
            vv    = round(min(pink_m / total_sat * 2.5, 0.95), 3)
            return "Orange", "Pink", state, vv
        pink_note = pink_m > orange * 0.45 and pink_m > MIN_W

        # ── Yellow ─────────────────────────────────────────────────────────────
        # Yellow must dominate over orange to avoid being masked by a mixed warm frame.
        if (
            yellow > MIN_W
            and yellow >= orange * 0.80
            and yellow >= green
            and yellow >= cool
            and yellow >= brown
        ):
            state = (
                "Yellow Frequency / High-Luminance Resonance — maximum signal visibility, stable energy output"
                if yellow > orange * 1.50
                else "Gold Frequency / Warm-Spectrum Composite — elevated visibility, positive impulse signal"
            )
            vv = round(float(arr[:, :, 0].mean()) / 255.0, 3)
            return "Orange", "Yellow", state, vv

        # ── Green wins ────────────────────────────────────────────────────────
        if green >= warm and green >= cool and green > MIN_W:
            state = (
                "Subharmonic Green / Cyan-Tinted Recessive Field"
                if cyan > green * 0.4
                else "Cognitive Stasis / Growth Vector"
            )
            vv = round(float(arr[:, :, 1].mean()) / 255.0, 3)
            return "Green", "Green", state, vv

        # ── Blue / Cyan wins ──────────────────────────────────────────────────
        if cool >= warm and cool >= green and cool > MIN_W:
            if cyan > blue:
                state = "Cyan Resonance / Interface Frequency — boundary negotiation active"
                label = "Cyan"
            elif violet > blue * 0.45:
                state = "Indigo-Blue Composite / Depth Field — high strategic density"
                label = "Violet"
            else:
                state = "Cognitive Openness / Negotiation Interface"
                label = "Blue"
            vv = round(float(arr[:, :, 2].mean()) / 255.0, 3)
            return "Blue", label, state, vv

        # ── Violet / Purple (maps to Blue family) ─────────────────────────────
        if violet > MIN_W and violet >= warm and violet >= green:
            state = "Violet Frequency / Prestige-Depth Composite — subliminal authority field"
            vv    = round(float(arr[:, :, 2].mean()) / 255.0, 3)
            return "Blue", "Violet", state, vv

        # ── Warm wins (orange dominating) ─────────────────────────────────────
        if warm > MIN_W:
            if yellow > orange * 1.25:
                state = "Gold Frequency / Warm-Spectrum Composite — elevated visibility, positive impulse signal"
                label = "Yellow"
            elif pink_note:
                state = "Orange-Pink Interference / High-Value Composite — aesthetic dominance overlay"
                label = "Orange"
            elif orange > total_sat * 0.35:
                state = "High-Energy Focus / Attack Vector — maximum observer attention command"
                label = "Orange"
            else:
                state = "Warm Thermal Field / Muted Expansion — low-entropy dominance signal"
                label = "Orange"
            vv = round(float(arr[:, :, 0].mean()) / 255.0, 3)
            return "Orange", label, state, vv

        # ── Fallback ──────────────────────────────────────────────────────────
        means = [float(arr[:, :, c].mean()) for c in range(3)]
        idx   = int(np.argmax(means))
        labels = [
            ("Orange", "Orange", "Residual Thermal Baseline / Muted Entropy Field"),
            ("Green",  "Green",  "Cognitive Stasis / Subdued Growth"),
            ("Blue",   "Blue",   "Achromatic Field / Cognitive Reset"),
        ]
        freq, label, state = labels[idx]
        return freq, label, state, round(means[idx] / 255.0, 3)

    # ──────────────────────────────────────────────────────────────────────────
    # v1.0 Methods (updated for v1.28)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_absolute_stasis(self, edge_density: float) -> bool:
        """
        Returns True when edge density is near zero — complete rhythmic absence.

        Threshold: fewer than 2.5 % of pixels are classified as active edges,
        indicating a blank, nearly uniform field with no discernible takt pattern.
        Maps to the '0-0-0-0' [ABSOLUTE_STASIS] signature.
        """
        return edge_density < 0.025

    def _detect_fibonacci_harmonic(self, gray: np.ndarray | None = None) -> bool:
        """
        Detects a 1-1-2-3 Fibonacci-like density progression across 4 horizontal strips.

        The pattern is valid when:
          • Strip 0 and Strip 1 have near-equal density  (ratio 1:1, ±45 % tolerance).
          • Strip 2 has roughly double the density of Strip 0 (ratio 2:1, ±45 %).
          • Strip 3 has roughly triple the density of Strip 0 (ratio 3:1, ±50 %).
          • Strip 0 has at least 2 % edge density (rule out near-stasis fields).

        Naturally-composed images (architecture, faces, artwork with graded depth)
        tend to exhibit this bottom-heavy Fibonacci scaling in their texture density.

        Parameters
        ----------
        gray : np.ndarray or None
            Grayscale sub-array to analyse (e.g. the CARDS focus zone).
            Falls back to the full-image self._gray when None.
        """
        g    = gray if gray is not None else self._gray
        h    = g.shape[0]
        strips   = 4
        strip_h  = max(1, h // strips)

        gx_g = np.gradient(g, axis=1)
        gy_g = np.gradient(g, axis=0)
        mag  = np.sqrt(gx_g ** 2 + gy_g ** 2)

        densities = [
            float((mag[i * strip_h : min((i + 1) * strip_h, h), :] > 20).mean())
            for i in range(strips)
        ]

        d0 = densities[0]
        if d0 < 0.02:
            return False   # too sparse to establish a meaningful base ratio

        # Ratio tolerances: ±45 % for the 1:1 and 2:1 steps, ±50 % for the 3:1 step.
        tol_eq  = 0.45
        tol_2x  = 0.45
        tol_3x  = 0.50

        ratio_1_ok = abs(densities[1] - d0)       / d0 < tol_eq
        ratio_2_ok = abs(densities[2] - d0 * 2.0) / (d0 * 2.0) < tol_2x
        ratio_3_ok = abs(densities[3] - d0 * 3.0) / (d0 * 3.0) < tol_3x

        return ratio_1_ok and ratio_2_ok and ratio_3_ok

    # ──────────────────────────────────────────────────────────────────────────
    # v1.0 Methods (updated for v1.35)
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_dimension_5(self, sector_mode: str | None = None) -> Dim5Color:
        """
        v1.28 — Saturation-weighted HSV hue-bin voting with sector-specific focus masking.

        For CARDS: outer card frame (12 % left/right, 15 % top/bottom) is excluded.
                   Inner artwork zone weighted at 75 %, outer context at 25 %.
        For INDIVIDUUM: central body area (85 % width, 80 % height) weighted at 75 %.

        This prevents the yellow/orange card border and warm background skin tones
        from dominating the color read of the actual subject.
        """
        if sector_mode in ("CARDS", "INDIVIDUUM"):
            zone = self._get_focus_zone(sector_mode)
            if zone is not None:
                x1, y1, x2, y2 = zone
                core_arr  = self.array[y1:y2, x1:x2]
                if core_arr.size > 0:
                    core_votes = self._hsv_sat_votes(core_arr)
                    full_votes = self._hsv_sat_votes(self.array)
                    # 95 % inner zone, 5 % full image — heavier core weighting strips
                    # the card border's colour contribution almost entirely.
                    combined = {
                        k: core_votes[k] * 0.95 + full_votes.get(k, 0.0) * 0.05
                        for k in core_votes
                    }
                    # Use core_arr for channel means so vector_value reflects the artwork,
                    # not the card frame / full-image average.
                    freq, label, state, vv = self._classify_color_from_votes(
                        combined, source_array=core_arr
                    )
                    return Dim5Color(
                        dominant_frequency=freq, color_label=label,
                        vector_value=vv, semantic_state=state,
                    )

        # Default: full-image saturation-weighted voting
        votes = self._hsv_sat_votes(self.array)
        freq, label, state, vv = self._classify_color_from_votes(votes)
        return Dim5Color(
            dominant_frequency=freq, color_label=label,
            vector_value=vv, semantic_state=state,
        )

    def _compute_edge_density(self, exclusion_mask: "np.ndarray | None" = None) -> float:
        """
        Sobel-gradient edge density over the grayscale image.

        exclusion_mask (optional): boolean array (H×W) where True marks zones that
            should be attenuated — currently used for the URBAN_AREA sky region.
            Masked pixels contribute only 15 % of their normal edge weight so that
            smooth sky gradients do not artificially inflate Dim-10 frequency.

        Returns fraction of effectively-active edge pixels (0-1).
        """
        gx = np.gradient(self._gray, axis=1)
        gy = np.gradient(self._gray, axis=0)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        active    = (magnitude > 20.0).astype(np.float32)

        if exclusion_mask is not None:
            # Sky-zone edges contribute only 15 % (85 % reduction per spec)
            weights = np.where(exclusion_mask, 0.15, 1.0).astype(np.float32)
            return float((active * weights).sum() / weights.sum()) if weights.sum() > 0 else 0.0

        return float(active.sum() / (self.width * self.height))

    def _detect_progressive_density(self) -> bool:
        """
        Returns True when edge density increases monotonically from left to right
        across 4 vertical slices — the '1-2-3-4 Progressive Vector' signature.
        """
        slices = 4
        slice_w = max(1, self.width // slices)
        gx_g = np.gradient(self._gray, axis=1)
        gy_g = np.gradient(self._gray, axis=0)
        mag = np.sqrt(gx_g ** 2 + gy_g ** 2)

        densities = [
            float((mag[:, i * slice_w : (i + 1) * slice_w] > 20).mean())
            for i in range(slices)
        ]
        return all(densities[i] < densities[i + 1] for i in range(slices - 1))

    def _detect_oscillating_4242(self, edge_density: float) -> bool:
        """
        Detects a 4-2-4-2 oscillating resonance: broad high-energy arcs alternating
        with narrow dense zones, at 6 horizontal strips. Distinct from the 2-3-2-3
        detector which requires ≥5 alternations across 8 strips.
        """
        if edge_density < 0.08:
            return False
        strips = 6
        strip_h = max(1, self.height // strips)
        gx_g = np.gradient(self._gray, axis=1)
        gy_g = np.gradient(self._gray, axis=0)
        mag = np.sqrt(gx_g ** 2 + gy_g ** 2)

        densities = [
            float((mag[i * strip_h : (i + 1) * strip_h, :] > 15).mean())
            for i in range(strips)
        ]
        mean_d = float(np.mean(densities))
        alternations = sum(
            1 for i in range(1, len(densities))
            if (densities[i] > mean_d) != (densities[i - 1] > mean_d)
        )
        # 4-2-4-2 produces 3-4 alternations; wider than 5+ of the 2-3-2-3 detector
        return 3 <= alternations <= 4 and mean_d < 0.40

    def analyze_dimensions_6_8(
        self,
        edge_density: float | None = None,
        sector_mode: str = "",
    ) -> Dim68Rhythm:
        """
        Maps edge density to takt signature and coherence score.

        v1.35 additions:
          • '0-0-0-0' [ABSOLUTE_STASIS]   — near-zero edge density, energetic null point.
            Checked first, before any other pattern, so blank images short-circuit cleanly.
          • '1-1-2-3' [FIBONACCI_HARMONIC] — Fibonacci density progression across 4 strips.
            Detected on the CARDS artwork core (focus zone) when sector_mode == 'CARDS';
            falls back to full-frame analysis for all other sectors.

        Full takt hierarchy (checked in order):
          0-0-0-0 → 1-1-1-1 → 1-2-3-4 → 4-2-4-2 → 1-1-2-3 → 2-2-2 → 3-3-3 → 4-4-4 → 5-5-5
        """
        if edge_density is None:
            edge_density = self._compute_edge_density()

        # ── v1.35  Absolute Stasis — checked before all other patterns ─────────
        # Near-zero edge density means no rhythmic structure can form.  This fires
        # for effectively blank images and short-circuits all downstream detectors.
        if self._detect_absolute_stasis(edge_density):
            state = {
                "CARDS":      "Artwork Void — absolute negative space, rarity encoded through silence",
                "INDIVIDUUM": "Absolute Stillness / Power Null — authority through energetic absence",
                "STOCKS":     "Pre-Breakout Absolute Coil — zero-signal compression at maximum density",
                "URBAN_AREA": "Urban Void — zero navigational stimulus, spatial authority vacuum",
                "TOYS":       "Deep Rest State / Reset Null — play object suspended in energetic void",
            }.get(sector_mode, "Absolute Stasis / Zero-Point Field — energetic ground state")
            return Dim68Rhythm(
                pattern_type="Zero-density void — complete rhythmic absence, energetic null point",
                frequency_takt="0-0-0-0",
                coherence_score=round(min(edge_density * 2.0, 0.05), 3),
                semantic_state=state,
            )

        # ── Pre-compute CARDS focus-zone grayscale ─────────────────────────────
        # Both the organic-center detector and the Fibonacci detector operate on
        # the isolated artwork core when sector_mode is CARDS, excluding the dense
        # card frame from influencing rhythm classification.
        _cards_gray: np.ndarray | None = None
        if sector_mode == "CARDS":
            _czone = self._get_focus_zone("CARDS")
            if _czone is not None:
                cx1, cy1, cx2, cy2 = _czone
                _slice = self._gray[cy1:cy2, cx1:cx2]
                _cards_gray = _slice if _slice.size >= 16 else None

        # ── v1.3  CARDS: detect organic artwork center ─────────────────────────
        # If the innermost core of the artwork is significantly softer than the
        # full-image density, the card exhibits an oscillating-containment rhythm
        # (dense frame / soft round subject, e.g. cards with circular artwork).
        if sector_mode == "CARDS" and 0.08 < edge_density < 0.50 and _cards_gray is not None:
            ch, cw = _cards_gray.shape
            qx = cw // 4
            qy = ch // 4
            core_gray = _cards_gray[qy : ch - qy, qx : cw - qx]
            if core_gray.size >= 16:
                cgx = np.gradient(core_gray, axis=1)
                cgy = np.gradient(core_gray, axis=0)
                core_density = float((np.sqrt(cgx ** 2 + cgy ** 2) > 20).mean())
                # Inner core is < 50 % as dense as the full image → soft subject
                # surrounded by a dense border → oscillating-containment rhythm.
                if core_density < edge_density * 0.50 and not self._detect_oscillating_4242(edge_density):
                    return Dim68Rhythm(
                        pattern_type=(
                            "Soft artwork core / dense peripheral frame — "
                            "organic-containment energy field"
                        ),
                        frequency_takt="4-2-4-2",
                        coherence_score=min(round(0.30 + edge_density * 1.4, 3), 1.0),
                        semantic_state="Oscillating Resonance / Cyclical Energy Transfer",
                    )

        # ── v1.2  Complex pattern detectors ───────────────────────────────────
        if edge_density > 0.55:
            takt     = "1-1-1-1"
            pattern  = "Maximum stimulation field — beyond saturation threshold"
            state    = "Dense Sensory Overload / System Saturation"
            coherence = round(min(edge_density * 1.6, 1.0), 3)

        elif 0.18 < edge_density < 0.48 and self._detect_progressive_density():
            takt     = "1-2-3-4"
            pattern  = "Progressive density gradient — ascending vector field"
            state    = "Progressive Vector / Directional Momentum Build-Up"
            coherence = round(min(0.35 + edge_density, 1.0), 3)

        elif self._detect_oscillating_4242(edge_density):
            takt     = "4-2-4-2"
            pattern  = "Wide-arc oscillating resonance — alternating pressure zones"
            state    = "Oscillating Resonance / Cyclical Energy Transfer"
            coherence = round(min(0.30 + edge_density * 1.5, 1.0), 3)

        # ── v1.35  Fibonacci Harmonic ──────────────────────────────────────────
        # Detected on the CARDS artwork core if available, otherwise full frame.
        # Fires when 4-strip density follows a 1:1:2:3 Fibonacci-like progression.
        elif self._detect_fibonacci_harmonic(gray=_cards_gray):
            takt    = "1-1-2-3"
            pattern = "Natural fractal density scaling — Fibonacci harmonic progression"
            state   = {
                "CARDS":      "Fibonacci Artwork Scaling / Natural Composition — unconscious aesthetic value elevation",
                "INDIVIDUUM": "Fibonacci Presence Field / Fractal Scaling — unconscious luxury attractor engaged",
                "STOCKS":     "Fibonacci Retracement Confirmed / Natural Price Structure — high-probability reversal zone",
                "URBAN_AREA": "Fibonacci Façade Rhythm / Fractal Urban Scaling — spatial aesthetic resonance",
                "TOYS":       "Fibonacci Play Flow / Natural Engagement Scaling — optimal exploration trigger",
            }.get(sector_mode, "Fibonacci Harmonic / Natural Fractal Scaling — aesthetic resonance field active")
            coherence = round(min(0.55 + edge_density * 0.60, 1.0), 3)

        # ── v1.0 / v1.3  Standard tiers (sector-aware labels) ─────────────────
        elif edge_density > 0.35:
            takt    = "2-2-2"
            pattern = "Dense stripe / high-frequency texture"
            state   = {
                "CARDS":      "High Entropy / Dense Artwork Field",
                "INDIVIDUUM": "High Entropy / Cognitive Load",
                "STOCKS":     "High Volatility / Noise-Mask Field",
                "URBAN_AREA": "Urban Saturation / Infrastructure Overload",
                "TOYS":       "Stimulation Peak / Attention Saturation",
            }.get(sector_mode, "High Entropy / Cognitive Load")
            coherence = round(min(edge_density * 1.8, 1.0), 3)

        elif edge_density > 0.22:
            takt    = "3-3-3"
            pattern = "Moderate structured field — rhythmic equilibrium"
            state   = {
                "CARDS":      "Spatial Resonance / Artwork Equilibrium Field",
                "INDIVIDUUM": "Balanced Signal / Measured Authority Projection",
                "STOCKS":     "Consolidation Phase / Institutional Equilibrium",
                "URBAN_AREA": "Architectural Rhythm / Urban Grid Resonance",
                "TOYS":       "Moderate Engagement / Play-State Equilibrium",
            }.get(sector_mode, "Balanced Entropy / Transitional")
            coherence = round(0.45 + edge_density * 0.8, 3)

        elif edge_density > 0.12:
            takt    = "4-4-4"
            pattern = "Open field / low-density structured presence"
            state   = {
                "CARDS":      "Coherent Artwork / Clean Power Statement",
                "INDIVIDUUM": "Coherent Emergence / Authority Broadcast",
                "STOCKS":     "Low Volatility / Coil Formation",
                "URBAN_AREA": "Sparse Urban Field / Territorial Control",
                "TOYS":       "Focused Play / Sustained Engagement",
            }.get(sector_mode, "Coherent Emergence / Authority Broadcast")
            coherence = round(0.35 + edge_density * 1.2, 3)

        else:
            takt    = "5-5-5"
            pattern = "Sparse geometric / clean field"
            state   = "Coherent Structure / Power Signaling"
            coherence = round(0.25 + edge_density * 2.5, 3)

        return Dim68Rhythm(
            pattern_type=pattern,
            frequency_takt=takt,
            coherence_score=min(coherence, 1.0),
            semantic_state=state,
        )

    def analyze_dimension_9(self, sector_mode: str | None = None) -> Dim9Geometry:
        """
        2D FFT splits image energy into low-freq (round) vs high-freq (angular).
        High-freq dominance → Triangle/Jagged; low-freq → Circle/Spiral.

        v1.28: For CARDS/INDIVIDUUM applies focus-zone weighting (75 % inner, 25 % full)
        using the shared _fft_curvature() helper to suppress frame/border bias.
        """
        if sector_mode in ("CARDS", "INDIVIDUUM"):
            zone = self._get_focus_zone(sector_mode)
            if zone is not None:
                x1, y1, x2, y2 = zone
                inner_gray = self._gray[y1:y2, x1:x2]
                if inner_gray.size >= 16:
                    inner_curvature = self._fft_curvature(inner_gray)
                    full_curvature  = self._fft_curvature(self._gray)
                    edge_curvature_index = round(inner_curvature * 0.75 + full_curvature * 0.25, 3)
                    if edge_curvature_index > 0.55:
                        shape, distribution = "Triangle / Jagged", "Singularity Focus at Edge Points"
                    else:
                        shape, distribution = "Circle / Spiral", "Endless Loop / Energy Containment"
                    return Dim9Geometry(
                        dominant_shape=shape,
                        edge_curvature_index=min(edge_curvature_index, 1.0),
                        energy_distribution=distribution,
                    )

        # Default: full-image FFT
        edge_curvature_index = self._fft_curvature(self._gray)

        if edge_curvature_index > 0.55:
            shape, distribution = "Triangle / Jagged", "Singularity Focus at Edge Points"
        else:
            shape, distribution = "Circle / Spiral", "Endless Loop / Energy Containment"

        return Dim9Geometry(
            dominant_shape=shape,
            edge_curvature_index=min(edge_curvature_index, 1.0),
            energy_distribution=distribution,
        )

    def analyze_dimension_10(self, edge_density: float) -> Dim10Frequency:
        """Derive impulse Hz and intensity multiplier from edge density scalar."""
        return Dim10Frequency(
            impulse_rate_hz=round(edge_density * 20.0, 1),
            intensity_multiplier=round(1.0 + edge_density, 3),
        )

    def analyze_symmetry(self) -> SymmetryAnalysis:
        """
        Compare left half to horizontally mirrored right half.
        Score > 0.65 → Symmetric; 0.40-0.65 → SUSY pairing.
        """
        left = self._gray[:, : self.width // 2]
        right = np.fliplr(self._gray[:, self.width // 2 :])
        w = min(left.shape[1], right.shape[1])
        diff = np.abs(left[:, :w] - right[:, :w])

        symmetry_score = max(0.0, min(1.0, round(float(1.0 - diff.mean() / 128.0), 3)))
        is_symmetric = symmetry_score > 0.65
        is_susy = (not is_symmetric) and symmetry_score > 0.40

        if is_susy:
            state = "SUSY Pairing — Perfect Duality, Resistant to Interference"
        elif is_symmetric:
            state = "Stability Matrix — Predictability and Emotional Stasis"
        else:
            state = "Asymmetric Field — Dynamic Instability, Unresolved Potential"

        return SymmetryAnalysis(
            is_symmetric=is_symmetric,
            is_susy=is_susy,
            symmetry_score=symmetry_score,
            semantic_state=state,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # v1.1 Quantum Field Analysis — private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _scan_patches(
        self,
        patch_size: int = 16,
        exclusion_mask: "np.ndarray | None" = None,
    ) -> tuple[int, int]:
        """
        Divides the image into non-overlapping square patches and classifies each
        as 'smooth/circular' (low variance) or 'angular/rectangular' (high variance
        combined with dominant axis-aligned gradient).

        exclusion_mask: boolean array (H×W) where True marks zones to skip.
            Used for URBAN_AREA to avoid counting cloud/sky regions as circles,
            ensuring Stabilization Atoms are detected only within architectural zones
            that exhibit genuine geometric complexity (windows, cables, trusses).

        Returns:
            (circle_count, square_count) — raw patch counts
        """
        circle_count = 0
        square_count = 0

        # Pre-compute gradients for axis-alignment check
        gx = np.abs(np.gradient(self._gray, axis=1))   # horizontal edges → vertical lines
        gy = np.abs(np.gradient(self._gray, axis=0))   # vertical edges → horizontal lines

        for y in range(0, self.height - patch_size, patch_size):
            for x in range(0, self.width - patch_size, patch_size):
                # Skip patches that are predominantly in the exclusion zone (sky)
                if exclusion_mask is not None:
                    patch_mask = exclusion_mask[y: y + patch_size, x: x + patch_size]
                    if float(patch_mask.mean()) > 0.60:   # > 60 % sky → skip
                        continue

                patch    = self._gray[y: y + patch_size, x: x + patch_size]
                mean_val = float(patch.mean())
                variance = float(((patch - mean_val) ** 2).mean())

                if variance < self._SMOOTH_VARIANCE_THRESHOLD:
                    # Low variance → smooth region, likely circular/curved
                    circle_count += 1
                elif variance > self._ANGULAR_VARIANCE_THRESHOLD:
                    # High variance → check for axis-aligned edges (rectangular)
                    patch_gx = gx[y: y + patch_size, x: x + patch_size].mean()
                    patch_gy = gy[y: y + patch_size, x: x + patch_size].mean()
                    if patch_gx > 8.0 and patch_gy > 8.0:
                        square_count += 1

        return circle_count, square_count

    def _detect_micro_cluster_density(self, square_count: int) -> int:
        """
        Counts bright micro-clusters (small point-like blobs) in the image.
        Squares attract micro-forms: the count is weighted by detected square density.
        Uses a simple threshold on small bright spots in a high-pass filtered image.
        """
        # High-pass: local pixel minus neighbourhood mean highlights point features
        from scipy.ndimage import uniform_filter  # type: ignore[import-not-found]
        try:
            blurred = uniform_filter(self._gray, size=5)
            high_pass = self._gray - blurred
            # Bright micro-spots appear as positive peaks in the high-pass image
            micro_spots = int((high_pass > 25.0).sum())
            total_pixels = self.width * self.height
            micro_density = micro_spots / total_pixels
            # Weight by square presence — more squares = more gravitational attraction
            weighted = micro_density * (1.0 + square_count * 0.05)
            return min(int(weighted * 1000), 999)
        except ImportError:
            # scipy not available — use simple variance-based fallback
            std_map = np.abs(self._gray - uniform_filter(self._gray, size=3)
                             if False else self._gray - self._gray.mean())
            return min(int((std_map > 30).sum() // 100), 999)

    def _detect_micro_cluster_density_simple(self, square_count: int) -> int:
        """
        scipy-free micro-cluster approximation using a manual 3×3 mean filter.
        Falls back to this if _detect_micro_cluster_density raises ImportError.
        """
        # 3×3 neighbourhood mean via stride tricks (manual box filter)
        padded = np.pad(self._gray, 1, mode='edge')
        neighbours = np.stack([
            padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:],
            padded[1:-1, 0:-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
            padded[2:,   0:-2], padded[2:,   1:-1], padded[2:,   2:],
        ], axis=0)
        local_mean = neighbours.mean(axis=0)
        high_pass = self._gray - local_mean
        micro_spots = int((high_pass > 22.0).sum())
        total = self.width * self.height
        density = micro_spots / total * (1.0 + square_count * 0.04)
        return min(int(density * 1000), 999)

    def _detect_gravitational_density(self) -> float:
        """
        Measures the fraction of pixels with luminance below the dark threshold.
        High fraction → strong Gravitational Field (black zones curve the steganographic space).
        """
        dark_pixels = (self._gray < self._DARK_THRESHOLD).sum()
        return round(float(dark_pixels) / (self.width * self.height), 3)

    def _detect_electromagnetic_break(self) -> bool:
        """
        Detects the Yellow / jagged-line EM-break signature.
        Yellow = high R, high G, suppressed B.
        Must also exceed an edge-density threshold to confirm the 'jagged' quality.
        """
        r = self.array[:, :, 0]
        g = self.array[:, :, 1]
        b = self.array[:, :, 2]

        yellow_mask = (
            (r > self._YELLOW_R_MIN) &
            (g > self._YELLOW_G_MIN) &
            (b < self._YELLOW_B_MAX)
        )
        yellow_ratio = float(yellow_mask.sum()) / (self.width * self.height)

        # Require significant yellow presence and high edge density (jagged)
        edge_density = self._compute_edge_density()
        return yellow_ratio > 0.04 or (yellow_ratio > 0.015 and edge_density > 0.30)

    def _detect_oscillating_field(self) -> bool:
        """
        Detects a 2-3-2-3 oscillating rhythm by dividing the image into 8 horizontal
        strips and testing for an alternating high/low edge-density pattern.
        5+ alternations out of 7 adjacent-strip transitions confirms the field.
        """
        STRIPS = 8
        strip_h = max(1, self.height // STRIPS)

        gx = np.gradient(self._gray, axis=1)
        gy = np.gradient(self._gray, axis=0)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        densities: list[float] = []
        for i in range(STRIPS):
            strip = magnitude[i * strip_h : (i + 1) * strip_h, :]
            densities.append(float((strip > 20.0).sum() / strip.size))

        if not densities:
            return False

        mean_d = float(np.mean(densities))
        # Count high→low and low→high transitions
        alternations = sum(
            1 for i in range(1, len(densities))
            if (densities[i] > mean_d) != (densities[i - 1] > mean_d)
        )
        return alternations >= 5

    def _detect_luxury_particles(self) -> tuple[bool, float]:
        """
        Checks for simultaneous presence of Black, Silver, and White pixels
        (the Luxury Particle combo that multiplies steganographic base value).

        Returns:
            (detected: bool, boost: float)  — boost in [0.0, 0.8]
        """
        lum = self._gray  # per-pixel luminance
        r, g, b = self.array[:, :, 0], self.array[:, :, 1], self.array[:, :, 2]
        total = self.width * self.height

        black_ratio = float((lum < self._DARK_THRESHOLD).sum()) / total
        white_ratio = float((lum > self._WHITE_THRESHOLD).sum()) / total

        # Silver: near-neutral hue + mid-range luminance
        channel_max_diff = np.maximum(
            np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b))
        )
        silver_mask = (
            (channel_max_diff < self._GRAY_CHANNEL_TOLERANCE) &
            (lum > 85.0) &
            (lum < self._WHITE_THRESHOLD)
        )
        silver_ratio = float(silver_mask.sum()) / total

        # All three must be present at meaningful thresholds
        detected = black_ratio > 0.05 and white_ratio > 0.03 and silver_ratio > 0.04
        if detected:
            # Boost scales with the strength of the triple-combo
            combo_strength = min(black_ratio, 0.4) + min(white_ratio, 0.3) + min(silver_ratio, 0.3)
            boost = round(min(combo_strength * 1.5, 0.8), 3)
        else:
            boost = 0.0

        return detected, boost

    def _classify_stability(
        self,
        circle_count: int,
        square_count: int,
        edge_density: float,
    ) -> str:
        """
        Applies the Stabilization-Atom rule:
        A circle (smooth region) needs a complementary element X nearby.
        Complement = a significant linear/rectangular structure (squares or high edge lines).

        If circles dominate but no complement exists → Unstable.
        """
        total_patches = max(circle_count + square_count, 1)
        circle_ratio = circle_count / total_patches

        # Complement availability: squares or sufficient axis-aligned edges
        has_complement = square_count > 2 or edge_density > 0.18

        if circle_ratio > 0.55 and not has_complement:
            return "Unstable / Structurally Weakened"
        return "Stable"

    # ──────────────────────────────────────────────────────────────────────────
    # v1.1 Public orchestrator
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_quantum_field(
        self,
        edge_density: float,
        symmetry: SymmetryAnalysis,
        exclusion_mask: "np.ndarray | None" = None,
    ) -> QuantumFieldMetrics:
        """
        Runs the complete v1.1 Quantum Field analysis pipeline.
        Receives pre-computed edge_density and symmetry to avoid redundant work.

        exclusion_mask: optional sky mask forwarded to _scan_patches() so that
            circle/square detection focuses on architectural zones for URBAN_AREA
            images rather than being misled by smooth sky regions.

        Returns:
            QuantumFieldMetrics ready for JSON serialisation.
        """
        # ── Particle scan ──────────────────────────────────────────────────────
        circle_count, square_count = self._scan_patches(exclusion_mask=exclusion_mask)
        micro_count = self._detect_micro_cluster_density_simple(square_count)

        # ── Individual field detectors ─────────────────────────────────────────
        grav_density = self._detect_gravitational_density()
        em_break = self._detect_electromagnetic_break()
        oscillating = self._detect_oscillating_field()
        luxury_detected, luxury_boost = self._detect_luxury_particles()

        # ── Stability classification ───────────────────────────────────────────
        stability = self._classify_stability(circle_count, square_count, edge_density)
        is_unstable = "Unstable" in stability

        # ── Active Hyperdimensional Compression ────────────────────────────────
        # Square surrounded by dense micro-forms → compression is active
        active_compression = square_count > 1 and micro_count > 80

        # ── Quantum Value Multiplier composition ───────────────────────────────
        qv = 1.0
        qv += luxury_boost                          # 0.0-0.80 from luxury particles
        qv += 0.25 if symmetry.is_susy else 0.0    # SUSY pairing bonus
        qv += 0.15 if not is_unstable else 0.0     # structural stability bonus
        qv -= 0.30 if is_unstable else 0.0         # instability penalty
        qv += 0.10 if active_compression else 0.0  # hyperdimensional compression bonus
        qv = round(max(0.5, min(qv, 3.0)), 3)      # clamp to [0.5, 3.0]

        return QuantumFieldMetrics(
            detected_particles=ParticleCount(
                circle_count=circle_count,
                square_count=square_count,
                micro_cluster_count=micro_count,
            ),
            stability_status=stability,
            gravitational_density=grav_density,
            electromagnetic_break_detected=em_break,
            oscillating_field_active=oscillating,
            luxury_particle_detected=luxury_detected,
            active_hyperdimensional_compression=active_compression,
            quantum_value_multiplier=qv,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # v1.3  Sector auto-detection helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_card_structure(self) -> bool:
        """
        Heuristically determines whether the image is a trading card regardless of
        the sector_mode the user selected.

        Detection criteria (all must pass):
          1. Portrait orientation — cards are taller than wide (H > W × 1.10).
          2. Border dominance   — edge density in the outer 10 % frame is at least
             1.4× the inner zone density (standardised rectangular card border).
          3. Bilateral symmetry — symmetry score > 0.62 (cards are left-right symmetric).

        When True, _resolve_sector_mode() switches INDIVIDUUM → CARDS internally.
        """
        # 1. Portrait orientation
        if self.height < self.width * 1.10:
            return False

        # 2. Outer frame edge density vs inner zone
        mx = max(1, int(self.width  * 0.10))
        my = max(1, int(self.height * 0.10))

        gx  = np.gradient(self._gray, axis=1)
        gy  = np.gradient(self._gray, axis=0)
        mag = (np.sqrt(gx ** 2 + gy ** 2) > 20).astype(float)

        outer = np.zeros_like(mag, dtype=bool)
        outer[:my, :] = True
        outer[-my:, :] = True
        outer[:, :mx] = True
        outer[:, -mx:] = True

        outer_density = float(mag[outer].mean())  if outer.any()  else 0.0
        inner_density = float(mag[~outer].mean()) if (~outer).any() else 0.0

        if outer_density < inner_density * 1.40:
            return False

        # 3. Symmetry check
        return self.analyze_symmetry().symmetry_score > 0.62

    def _resolve_sector_mode(self, sector_mode: str) -> str:
        """
        Auto-corrects the user-supplied sector_mode when structural evidence
        clearly contradicts the selection.

        Currently implemented rule:
          INDIVIDUUM → CARDS  when _detect_card_structure() returns True.

        This protects users who upload a trading card into the wrong mode:
        the analysis still applies the correct focus masking and QV clamps.
        The resolved mode is stored in the analysis dict so the translator
        can use it and the frontend can display it if desired.
        """
        if sector_mode == "INDIVIDUUM" and self._detect_card_structure():
            return "CARDS"
        return sector_mode

    def _detect_sky_mask(self) -> np.ndarray:
        """
        Identifies sky pixels in the top third of an URBAN_AREA image.

        Sky pixels satisfy ALL of:
          • Located in the top 1/3 of the image frame.
          • Chromatic blue/cyan sky: B channel dominant, V > 0.45, S < 0.75.
          • OR overcast / cloud white: V > 0.82, S < 0.18.
          • Low local texture: Sobel magnitude < 15 (homogeneous sky patch).

        The resulting mask is forwarded to:
          • _scan_patches()       — skip sky patches for particle detection.
          • _compute_edge_density() — attenuate sky edges by 85 % for Dim-10.

        Returns:
            Boolean array (H×W) where True = sky pixel.
        """
        sky_mask = np.zeros((self.height, self.width), dtype=bool)
        top_h    = self.height // 3
        if top_h == 0:
            return sky_mask

        top = self.array[:top_h]
        r   = top[:, :, 0].astype(np.float32) / 255.0
        g   = top[:, :, 1].astype(np.float32) / 255.0
        b   = top[:, :, 2].astype(np.float32) / 255.0

        cmax  = np.maximum(np.maximum(r, g), b)
        cmin  = np.minimum(np.minimum(r, g), b)
        sat   = np.where(cmax > 0, (cmax - cmin) / cmax, 0.0)
        val   = cmax

        # Blue sky: blue channel dominant, moderate-high value, not too saturated
        blue_sky = (b > r) & (b > g * 0.85) & (val > 0.45) & (sat < 0.75)
        # Overcast / clouds: very bright, nearly achromatic
        white_sky = (val > 0.82) & (sat < 0.18)
        # Cyan sky variant: both G and B above R
        cyan_sky  = (g > r * 1.05) & (b > r * 1.10) & (val > 0.40) & (sat < 0.65)

        # Low-texture gate: sky is smooth (no hard edges)
        gray_top  = self._gray[:top_h]
        gx_top    = np.gradient(gray_top, axis=1)
        gy_top    = np.gradient(gray_top, axis=0)
        low_tex   = (np.sqrt(gx_top ** 2 + gy_top ** 2) < 15.0)

        sky_mask[:top_h] = (blue_sky | white_sky | cyan_sky) & low_tex
        return sky_mask

    # ──────────────────────────────────────────────────────────────────────────
    # v1.2 Central Zone Geometry — Bug 3 fix
    # ──────────────────────────────────────────────────────────────────────────

    def _analyze_central_zone_geometry(self, zone_fraction: float = 0.33) -> Dim9Geometry:
        """
        Runs FFT geometry analysis on the centre crop only (default: central 33%).
        Used by the translator for CARDS and INDIVIDUUM sectors to prevent card
        frames and text borders from dominating the full-frame angular score.
        """
        h_margin = int(self.height * (1.0 - zone_fraction) / 2)
        w_margin = int(self.width  * (1.0 - zone_fraction) / 2)
        central_gray = self._gray[
            h_margin : self.height - h_margin,
            w_margin : self.width  - w_margin,
        ]

        ch, cw = central_gray.shape
        if ch < 4 or cw < 4:
            return self.analyze_dimension_9()

        fft_magnitude = np.abs(np.fft.fftshift(np.fft.fft2(central_gray)))
        cy, cx = ch // 2, cw // 2
        radius = min(ch, cw) // 5

        yy, xx = np.ogrid[:ch, :cw]
        center_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

        low_freq_energy  = float(fft_magnitude[center_mask].sum())
        high_freq_energy = float(fft_magnitude[~center_mask].sum())
        total = low_freq_energy + high_freq_energy

        edge_curvature_index = round(high_freq_energy / total, 3) if total > 0 else 0.5

        if edge_curvature_index > 0.55:
            shape, distribution = "Triangle / Jagged", "Singularity Focus at Edge Points"
        else:
            shape, distribution = "Circle / Spiral", "Endless Loop / Energy Containment"

        return Dim9Geometry(
            dominant_shape=shape,
            edge_curvature_index=min(edge_curvature_index, 1.0),
            energy_distribution=distribution,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # v1.2  New detectors
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_color_combo_value(self, dim5: Dim5Color) -> ColorComboValue:
        """
        Identifies high-order combinatorial color states that override the
        single-channel Dim-5 dominance with a compound value classification.

        Priority order: HIGH-VALUE > DYNAMIC-VALUE > INTELLECTUAL-VALUE > LOW-VALUE > NEUTRAL
        """
        lum   = self._gray
        r     = self.array[:, :, 0]
        g_ch  = self.array[:, :, 1]
        b_ch  = self.array[:, :, 2]
        total = self.width * self.height

        black_ratio  = float((lum < self._DARK_THRESHOLD).sum()) / total
        white_ratio  = float((lum > self._WHITE_THRESHOLD).sum()) / total

        ch_diff = np.maximum(np.abs(r - g_ch), np.maximum(np.abs(g_ch - b_ch), np.abs(r - b_ch)))
        silver_ratio = float(
            ((ch_diff < self._GRAY_CHANNEL_TOLERANCE) & (lum > 80) & (lum < self._WHITE_THRESHOLD)).sum()
        ) / total

        # Orange pixels: high R, moderate G, suppressed B
        orange_ratio = float(((r > 150) & (g_ch > 70) & (g_ch < 210) & (b_ch < 110)).sum()) / total
        # Blue-dominant pixels: B significantly above R and G
        blue_ratio   = float(((b_ch > r * 1.20) & (b_ch > g_ch * 1.15)).sum()) / total
        # Brown: earthy warm tone — R > G > B in mid-range
        brown_ratio  = float(
            ((r > 80) & (r < 200) & (g_ch > 45) & (g_ch < 165) & (b_ch < 130) &
             (r > g_ch * 1.10) & (g_ch > b_ch * 1.10)).sum()
        ) / total
        # Matte yellow: R ≈ G >> B with low saturation
        cmax = np.maximum(np.maximum(r, g_ch), b_ch).astype(np.float32)
        cmin = np.minimum(np.minimum(r, g_ch), b_ch).astype(np.float32)
        sat  = np.where(cmax > 0, (cmax - cmin) / cmax, 0.0)
        matte_yellow = float(
            ((r > 130) & (g_ch > 120) & (b_ch < 115) & (sat < 0.42) &
             (np.abs(r.astype(np.int32) - g_ch.astype(np.int32)) < 45)).sum()
        ) / total
        gray_ratio = float(((ch_diff < 28) & (lum > 55) & (lum < 195)).sum()) / total

        has_black  = black_ratio  > 0.05
        has_white  = white_ratio  > 0.03
        has_silver = silver_ratio > 0.04
        has_orange = orange_ratio > 0.04
        has_blue   = blue_ratio   > 0.04

        if has_black and has_silver and has_white:
            return ColorComboValue(
                label="HIGH-VALUE",
                description="Absolute structural integrity — maximum value retention and status encoding",
                components=["Black", "Silver", "White"],
            )
        if has_black and has_white and has_orange:
            return ColorComboValue(
                label="DYNAMIC-VALUE",
                description="Explosive expansion, unconscious dominance, immediate breakthrough vector",
                components=["Black", "White", "Orange"],
            )
        if has_black and has_white and has_blue:
            return ColorComboValue(
                label="INTELLECTUAL-VALUE",
                description="High strategic density, cognitive openness, negotiation interface",
                components=["Black", "White", "Blue"],
            )
        if brown_ratio > 0.08 and (gray_ratio > 0.10 or matte_yellow > 0.05):
            return ColorComboValue(
                label="LOW-VALUE",
                description="Energetic decay, stagnation, value erosion — entropy sink active",
                components=["Brown", "Gray", "Matte-Yellow"],
            )
        return ColorComboValue(
            label="NEUTRAL",
            description="Standard field composite — no exceptional value state detected",
            components=[],
        )

    def _detect_local_asymmetry(self) -> LocalAsymmetry:
        """
        Compares FFT curvature index between the left and right image halves.
        A delta > 0.12 declares a 'Local Symmetry Break' with a directed-flow description.
        """
        half_w = self.width // 2
        if half_w < 4:
            return LocalAsymmetry(
                detected=False, left_curvature=0.5, right_curvature=0.5,
                asymmetry_delta=0.0, description="",
            )

        def _half_curvature(arr: np.ndarray) -> float:
            fft = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
            h, w = arr.shape
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 5
            yy, xx = np.ogrid[:h, :w]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            lo = float(fft[mask].sum())
            hi = float(fft[~mask].sum())
            t  = lo + hi
            return round(hi / t, 3) if t > 0 else 0.5

        left_c  = _half_curvature(self._gray[:, :half_w])
        right_c = _half_curvature(self._gray[:, half_w:])
        delta   = round(abs(left_c - right_c), 3)
        detected = delta > 0.12

        if detected:
            if left_c > right_c:
                desc = (
                    f"Dominant angular vector left ({left_c:.2f}) vs "
                    f"compressed field right ({right_c:.2f}) — generates directed rightward flow"
                )
            else:
                desc = (
                    f"Compressed field left ({left_c:.2f}) vs "
                    f"dominant angular vector right ({right_c:.2f}) — generates directed leftward flow"
                )
        else:
            desc = ""

        return LocalAsymmetry(
            detected=detected,
            left_curvature=left_c,
            right_curvature=right_c,
            asymmetry_delta=delta,
            description=desc,
        )

    def compute_radar_axes(
        self,
        dim5: Dim5Color,
        dim68: Dim68Rhythm,
        dim9: Dim9Geometry,
        dim10: Dim10Frequency,
        symmetry: SymmetryAnalysis,
        qfm: QuantumFieldMetrics,
        local_asymmetry: "LocalAsymmetry | None" = None,
    ) -> list[float]:
        """
        Computes the 8-axis Energy Distribution Matrix vector (v1.35).

        All axes are normalised to [0, 1].

        Axis layout:
          0  Cognitive Load      — Dim 6-8 coherence score
          1  Aggressive Focus    — Dim 9 edge curvature index
          2  Structural Stability— symmetry score × instability penalty
          3  Information Entropy — Dim 10 impulse rate normalised (÷ 20 Hz)
          4  Calm / Absorption   — Green dominance or inverse warm-channel intensity
          5  Quantum Value       — QV multiplier mapped from [0.5, 3.0] → [0, 1]
          6  Gravitational Pull  — Gravitational Field density (dark pixel fraction)
          7  Symmetry-Break Vec. — Local asymmetry delta (0 when no break detected)

        Note: axis 5 (QV) intentionally stays at index 5 so the translator's
        QV-nuance patch (`radar_axes[5] = adjusted_qv_norm`) requires no change.
        """
        # Axis 0: Cognitive Load
        cognitive = dim68.coherence_score

        # Axis 1: Aggressive Focus
        aggression = dim9.edge_curvature_index

        # Axis 2: Structural Stability
        stability_base = 0.85 if symmetry.is_susy else (0.75 if symmetry.is_symmetric else 0.45)
        stability = stability_base * (0.65 if "Unstable" in qfm.stability_status else 1.0)

        # Axis 3: Information Entropy
        entropy = min(dim10.impulse_rate_hz / 20.0, 1.0)

        # Axis 4: Calm / Absorption
        calm = (
            dim5.vector_value if dim5.dominant_frequency == "Green"
            else max(0.0, 1.0 - dim5.vector_value * 0.7)
        )

        # Axis 5: Quantum Value
        qv_norm = min(max((qfm.quantum_value_multiplier - 0.5) / 2.5, 0.0), 1.0)

        # Axis 6: Gravitational Pull — already a fraction in [0, 1]
        grav_pull = qfm.gravitational_density

        # Axis 7: Symmetry-Break Vector — 0 when no local asymmetry is detected
        sym_break = (
            local_asymmetry.asymmetry_delta
            if local_asymmetry is not None and local_asymmetry.detected
            else 0.0
        )

        return [round(v, 3) for v in [
            cognitive, aggression, stability, entropy,
            calm, qv_norm, grav_pull, sym_break,
        ]]

    def _analyze_region_dim9(self, x1: int, y1: int, x2: int, y2: int) -> tuple[Dim9Geometry, float]:
        """
        Runs FFT geometry + edge coherence on a sub-region.
        Returns (Dim9Geometry, coherence_score).
        """
        region = self._gray[y1:y2, x1:x2]
        if region.size < 16:
            return self.analyze_dimension_9(), 0.5

        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(region)))
        rh, rw  = region.shape
        cy, cx  = rh // 2, rw // 2
        radius  = min(rh, rw) // 5
        yy, xx  = np.ogrid[:rh, :rw]
        cmask   = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        lo = float(fft_mag[cmask].sum())
        hi = float(fft_mag[~cmask].sum())
        t  = lo + hi
        curvature = round(hi / t, 3) if t > 0 else 0.5

        if curvature > 0.55:
            shape, dist = "Triangle / Jagged", "Singularity Focus at Edge Points"
        else:
            shape, dist = "Circle / Spiral", "Endless Loop / Energy Containment"

        gx_r = np.gradient(region, axis=1)
        gy_r = np.gradient(region, axis=0)
        coherence = round(min(float((np.sqrt(gx_r**2 + gy_r**2) > 20).mean()) * 2.0, 1.0), 3)

        return (
            Dim9Geometry(
                dominant_shape=shape,
                edge_curvature_index=min(curvature, 1.0),
                energy_distribution=dist,
            ),
            coherence,
        )

    def _analyze_region_dim5(self, x1: int, y1: int, x2: int, y2: int) -> Dim5Color:
        """
        Dim 5 color analysis for a detected-object sub-region.
        Uses the shared _hsv_sat_votes() + _classify_color_from_votes() pipeline
        (same white exclusion, yellow/brown separation as the full-image path).
        """
        region_rgb = self.array[y1:y2, x1:x2]
        if region_rgb.size == 0:
            return Dim5Color(
                dominant_frequency="Blue", color_label="Blue",
                vector_value=0.5, semantic_state="Cognitive Openness",
            )
        votes = self._hsv_sat_votes(region_rgb)
        freq, label, state, vv = self._classify_color_from_votes(votes, source_array=region_rgb)
        return Dim5Color(
            dominant_frequency=freq, color_label=label,
            vector_value=vv, semantic_state=state,
        )

    def detect_objects(self, sector_mode: str, sector_labels: list[str]) -> list[DetectedObject]:
        """
        Detects 2-3 visually significant regions using a 4×4 grid of edge-density cells.
        Adjacent high-density cells are merged into object bounding boxes, then each region
        receives a mini dimensional analysis (Dim 5 + Dim 9 + 6-axis radar).

        Returns up to 3 DetectedObject instances ordered by content density.
        """
        GRID = 4
        cell_w = self.width / GRID
        cell_h = self.height / GRID

        gx_g = np.gradient(self._gray, axis=1)
        gy_g = np.gradient(self._gray, axis=0)
        mag  = np.sqrt(gx_g ** 2 + gy_g ** 2)

        # Compute density per cell
        cells = []
        for gr in range(GRID):
            for gc in range(GRID):
                y1 = int(gr * cell_h);  y2 = int((gr + 1) * cell_h)
                x1 = int(gc * cell_w);  x2 = int((gc + 1) * cell_w)
                cell_mag = mag[y1:y2, x1:x2]
                density  = float((cell_mag > 20).mean()) if cell_mag.size else 0.0
                cells.append({"r": gr, "c": gc, "density": density, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        # Select top cells above threshold, then group into 2-3 bounding boxes
        threshold = max(np.mean([c["density"] for c in cells]) * 0.7, 0.05)
        hot = [c for c in cells if c["density"] >= threshold]
        if not hot:
            hot = sorted(cells, key=lambda c: -c["density"])[:3]

        # Merge into up to 3 objects by partitioning the hot set
        hot.sort(key=lambda c: -c["density"])
        object_boxes: list[dict] = []
        used = set()
        for seed in hot:
            key = (seed["r"], seed["c"])
            if key in used:
                continue
            # Merge seed with adjacent cells from hot
            members = [seed]
            used.add(key)
            for neighbor in hot:
                nk = (neighbor["r"], neighbor["c"])
                if nk in used:
                    continue
                if abs(neighbor["r"] - seed["r"]) <= 1 and abs(neighbor["c"] - seed["c"]) <= 1:
                    members.append(neighbor)
                    used.add(nk)
            x1 = min(m["x1"] for m in members)
            y1 = min(m["y1"] for m in members)
            x2 = max(m["x2"] for m in members)
            y2 = max(m["y2"] for m in members)
            avg_density = float(np.mean([m["density"] for m in members]))
            object_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "density": avg_density})
            if len(object_boxes) >= 3:
                break

        # Build DetectedObject for each box
        objects: list[DetectedObject] = []
        for i, box in enumerate(object_boxes):
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = sector_labels[i % len(sector_labels)]

            dim5_obj       = self._analyze_region_dim5(x1, y1, x2, y2)
            dim9_obj, coh  = self._analyze_region_dim9(x1, y1, x2, y2)

            density = box["density"]
            entropy = round(min(density * 2.0, 1.0), 3)
            calm    = round(max(0.0, 1.0 - dim5_obj.vector_value * 0.7), 3)
            aggr    = dim9_obj.edge_curvature_index
            stab    = round(0.6 + (1.0 - aggr) * 0.3, 3)
            qv_n    = round(min(dim5_obj.vector_value + coh * 0.3, 1.0), 3)

            # Gravitational pull proxy: dark pixel fraction within this bounding box
            region_gray = self._gray[y1:y2, x1:x2]
            grav_pull   = round(
                float((region_gray < self._DARK_THRESHOLD).sum()) / max(region_gray.size, 1), 3
            )

            # 8-axis radar: [CogLoad, AggrFocus, StructStab, InfoEntropy, Calm, QV, GravPull, SymBreak]
            # Symmetry-Break is not computed at per-object level — fixed at 0.0.
            radar = [round(coh, 3), aggr, stab, entropy, calm, qv_n, grav_pull, 0.0]

            interp = (
                f"{label}: {getattr(dim5_obj, 'color_label', dim5_obj.dominant_frequency)} "
                f"frequency zone. "
                f"{'Angular' if aggr > 0.55 else 'Round'} geometry "
                f"(curvature {aggr:.2f}) — "
                f"coherence {coh:.2f} | grav {grav_pull:.2f}."
            )

            objects.append(DetectedObject(
                id=f"obj_{i + 1}",
                label=label,
                bounding_box=[
                    round(x1 / self.width, 3),
                    round(y1 / self.height, 3),
                    round(x2 / self.width, 3),
                    round(y2 / self.height, 3),
                ],
                analysis=ObjectAnalysis(
                    dim_5_color=dim5_obj,
                    dim_9_geometry=dim9_obj,
                    coherence_score=coh,
                    radar_axes=radar,
                    interpretation=interp,
                ),
            ))

        return objects

    # ──────────────────────────────────────────────────────────────────────────
    # Full Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def run_full_analysis(self, sector_mode: str = "") -> dict:
        """
        Orchestrates the complete dimensional analysis pipeline (v1.0 → v1.3).

        v1.3 additions:
          • _resolve_sector_mode(): auto-corrects INDIVIDUUM → CARDS when a card
            structure is detected, regardless of the user's frontend selection.
          • Sky masking for URBAN_AREA: top-third sky pixels are identified and
            forwarded to edge-density computation (−85 % weight) and patch scanning
            (sky patches skipped) so clouds do not generate fake Stabilization Atoms.
          • sector_mode is forwarded to analyze_dimensions_6_8() for dynamic rhythm
            semantics; resolved_sector_mode is stored in the output dict so the
            translator can use the corrected value without re-running detection.

        Returns a plain dict consumed by translator.build_result().
        """
        # ── Sector resolution (auto-correct if needed) ─────────────────────────
        resolved = self._resolve_sector_mode(sector_mode)

        # ── Sky masking for URBAN_AREA ─────────────────────────────────────────
        sky_mask: "np.ndarray | None" = None
        if resolved == "URBAN_AREA":
            sky_mask = self._detect_sky_mask()

        # ── Core pipeline — edge density shared across all downstream consumers ─
        edge_density = self._compute_edge_density(exclusion_mask=sky_mask)

        dim5     = self.analyze_dimension_5(sector_mode=resolved or None)
        dim68    = self.analyze_dimensions_6_8(edge_density=edge_density, sector_mode=resolved)
        dim9     = self.analyze_dimension_9(sector_mode=resolved or None)
        dim10    = self.analyze_dimension_10(edge_density=edge_density)
        symmetry = self.analyze_symmetry()

        # Quantum field: passes sky mask so particle scan skips sky patches
        quantum_field = self.analyze_quantum_field(
            edge_density=edge_density,
            symmetry=symmetry,
            exclusion_mask=sky_mask,
        )

        # v1.2 supplemental detectors
        color_combo = self._detect_color_combo_value(dim5)
        local_asym  = self._detect_local_asymmetry()
        # local_asym must be computed before compute_radar_axes to populate axis 7
        radar_axes  = self.compute_radar_axes(dim5, dim68, dim9, dim10, symmetry, quantum_field, local_asym)

        return {
            "dim5":                 dim5,
            "dim68":                dim68,
            "dim9":                 dim9,
            "dim10":                dim10,
            "symmetry":             symmetry,
            "quantum_field":        quantum_field,
            "edge_density":         edge_density,
            "central_dim9":         self._analyze_central_zone_geometry(),
            "color_combo":          color_combo,
            "local_asymmetry":      local_asym,
            "radar_axes":           radar_axes,
            # v1.3 — resolved sector may differ from the caller's sector_mode
            "resolved_sector_mode": resolved,
            "sky_masked":           sky_mask is not None and bool(sky_mask.any()),
        }
