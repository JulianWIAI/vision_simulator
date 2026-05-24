"""
Hyperdimensional Matrix Analyzer — Pipeline Effect
effects/hd_matrix_analyzer.py

Real-time structural analysis overlay that draws dimensional markers on top of
ANY active base vision mode.  All heavy computation runs on a downscaled copy of
the frame; only OpenCV C++ draw calls touch the full-resolution output.

Ported and adapted from hd_engine/analyzer.py (NumPy/PIL) and the canvas layers
in ImageVisualizer.jsx.

Visual layer stack (drawn in order)
────────────────────────────────────
  Layer 1 — Patch curvature markers
      Angular patch (variance > 600 + axis-aligned gradient)
          → red crosshair    #ff2d55  BGR (85, 45, 255)
      Smooth / circular patch (variance < 220)
          → cyan ring        #00d4ff  BGR (255, 212, 0)

  Layer 2 — SUSY vector lines
      Bilateral symmetry score in [0.40, 0.65]
          → dashed cyan lines connecting smooth cluster centres to their
            nearest sharp angular complements

  Layer 3 — Gravitational aura
      Dark-zone density > 0.10
          → concentric purple / violet halo rings at dark centroid positions

  Layer 4 — Instability border
      Circles dominate (ratio > 0.55) without linear / edge complement
          → double red inset rectangle warning border

Performance contract
────────────────────
  Analysis frame : downscaled to MAX_ANALYSIS_DIM px on the long edge.
  Frame skip     : full analysis runs once every ANALYSIS_SKIP captured frames;
                   drawing uses the cached result on all intermediate frames.
  No Python loops over the full-resolution frame; all pixel writes go through
  OpenCV C++ primitives (drawMarker, circle, rectangle, addWeighted).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from modes.base_mode import BaseVisionMode


# ── Tunable constants ──────────────────────────────────────────────────────────

# Long-edge cap for the analysis frame (keeps FFT + gradient ops fast).
_MAX_ANALYSIS_DIM: int = 320

# Captured frames between full analysis passes.
# At 60 fps → value 6 gives ~10 analysis cycles / second.
_ANALYSIS_SKIP: int = 6

# Patch grid size on the *analysis-resolution* frame.
_PATCH_SIZE: int = 16

# ── Thresholds (mirrored from analyzer.py) ────────────────────────────────────
_SMOOTH_VARIANCE_THRESHOLD: float = 220.0   # variance below → circular patch
_ANGULAR_VARIANCE_THRESHOLD: float = 600.0  # variance above + axis check → angular
_DARK_THRESHOLD: float = 38.0               # luminance below → Black pixel
_GRAV_DENSITY_MIN: float = 0.10             # dark fraction above → render aura
_EDGE_DENSITY_MIN: float = 0.18             # edge density above → has complement
_SUSY_MIN: float = 0.40
_SUSY_MAX: float = 0.65
_CIRCLE_RATIO_UNSTABLE: float = 0.55        # circle fraction above → unstable

# ── Drawing colours (BGR) ─────────────────────────────────────────────────────
_C_ANGULAR  = ( 85,  45, 255)   # #ff2d55 — vivid red  (angular crosshair)
_C_SMOOTH   = (255, 212,   0)   # #00d4ff — cyan       (smooth ring)
_C_SUSY     = (255, 212,   0)   # #00d4ff — same cyan  (SUSY vector line)
_C_GRAV_IN  = (180,  40, 120)   # violet / purple      (inner aura ring)
_C_GRAV_OUT = (120,  20,  80)   # darker violet        (outer aura ring)
_C_UNSTABLE = ( 85,  45, 255)   # #ff2d55 — red        (instability border)


# ── Analysis result snapshot ───────────────────────────────────────────────────

class _AnalysisResult:
    """
    Immutable snapshot of one analysis pass.

    All pixel coordinates are already scaled to the *full* frame resolution
    so the draw pass never needs to know about the analysis downscale factor.
    """
    __slots__ = (
        "angular_pts", "smooth_pts",
        "circle_clusters", "sharp_clusters",
        "is_susy",
        "grav_density", "grav_centroids",
        "is_unstable",
        "frame_h", "frame_w",
    )

    def __init__(
        self,
        angular_pts:     List[Tuple[int, int]],
        smooth_pts:      List[Tuple[int, int]],
        circle_clusters: List[Tuple[int, int]],
        sharp_clusters:  List[Tuple[int, int]],
        is_susy:         bool,
        grav_density:    float,
        grav_centroids:  List[Tuple[int, int]],
        is_unstable:     bool,
        frame_h:         int,
        frame_w:         int,
    ) -> None:
        self.angular_pts     = angular_pts
        self.smooth_pts      = smooth_pts
        self.circle_clusters = circle_clusters
        self.sharp_clusters  = sharp_clusters
        self.is_susy         = is_susy
        self.grav_density    = grav_density
        self.grav_centroids  = grav_centroids
        self.is_unstable     = is_unstable
        self.frame_h         = frame_h
        self.frame_w         = frame_w


# ── Analysis engine (pure static helpers) ─────────────────────────────────────

class _HDEngine:
    """
    Stateless analysis helpers adapted from HyperdimensionalAnalyzer.

    All methods accept NumPy arrays and return plain Python types.
    No OpenCV imports — pure NumPy so the logic stays testable in isolation.
    """

    @staticmethod
    def scan_patches(
        gray: np.ndarray,
        patch_size: int,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int, int]:
        """
        Classify each non-overlapping patch as angular or smooth.

        Angular  : variance > _ANGULAR_VARIANCE_THRESHOLD AND both gx/gy means > 8.
        Smooth   : variance < _SMOOTH_VARIANCE_THRESHOLD.
        Middle   : ignored (not sharp enough to be angular, not flat enough for smooth).

        Returns (angular_centers, smooth_centers, smooth_count, angular_count).
        Coordinates are pixel centres within the *analysis-resolution* frame.
        """
        h, w   = gray.shape
        gx_mag = np.abs(np.gradient(gray.astype(np.float32), axis=1))
        gy_mag = np.abs(np.gradient(gray.astype(np.float32), axis=0))

        angular_pts: List[Tuple[int, int]] = []
        smooth_pts:  List[Tuple[int, int]] = []

        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch    = gray[y : y + patch_size, x : x + patch_size].astype(np.float32)
                mean_val = float(patch.mean())
                variance = float(((patch - mean_val) ** 2).mean())
                cx, cy   = x + patch_size // 2, y + patch_size // 2

                if variance < _SMOOTH_VARIANCE_THRESHOLD:
                    smooth_pts.append((cx, cy))
                elif variance > _ANGULAR_VARIANCE_THRESHOLD:
                    pgx = float(gx_mag[y : y + patch_size, x : x + patch_size].mean())
                    pgy = float(gy_mag[y : y + patch_size, x : x + patch_size].mean())
                    if pgx > 8.0 and pgy > 8.0:
                        angular_pts.append((cx, cy))

        return angular_pts, smooth_pts, len(smooth_pts), len(angular_pts)

    @staticmethod
    def compute_symmetry(gray: np.ndarray) -> Tuple[float, bool, bool]:
        """
        Left-vs-mirrored-right pixel difference score.

        Returns (symmetry_score, is_symmetric, is_susy).
        Mirrors analyze_symmetry() in analyzer.py.
        """
        h, w  = gray.shape
        left  = gray[:, : w // 2].astype(np.float32)
        right = np.fliplr(gray[:, w // 2 :]).astype(np.float32)
        col   = min(left.shape[1], right.shape[1])
        score = max(0.0, min(1.0, 1.0 - float(np.abs(left[:, :col] - right[:, :col]).mean()) / 128.0))
        is_symmetric = score > 0.65
        is_susy      = (not is_symmetric) and (_SUSY_MIN < score <= _SUSY_MAX)
        return score, is_symmetric, is_susy

    @staticmethod
    def compute_gravitational_density(gray: np.ndarray) -> float:
        """Dark-pixel fraction — mirrors _detect_gravitational_density()."""
        return float((gray < _DARK_THRESHOLD).sum()) / max(gray.size, 1)

    @staticmethod
    def find_dark_centroids(
        gray: np.ndarray,
        grid_cells: int = 8,
        max_centroids: int = 5,
    ) -> List[Tuple[int, int]]:
        """
        Coarse-grid centroid detector for dark zones.
        Each grid cell contributes one centroid if its mean luminance is below
        a relaxed dark threshold — keeps complexity at O(cells), not O(pixels).
        """
        h, w      = gray.shape
        cell_h    = max(1, h // grid_cells)
        cell_w    = max(1, w // grid_cells)
        threshold = _DARK_THRESHOLD * 1.8   # relaxed: include dim-but-not-black zones
        rows: List[Tuple[float, Tuple[int, int]]] = []

        for gy in range(grid_cells):
            for gx in range(grid_cells):
                y1, y2 = gy * cell_h, min((gy + 1) * cell_h, h)
                x1, x2 = gx * cell_w, min((gx + 1) * cell_w, w)
                cell_mean = float(gray[y1:y2, x1:x2].mean())
                if cell_mean < threshold:
                    cx = x1 + (x2 - x1) // 2
                    cy = y1 + (y2 - y1) // 2
                    rows.append((cell_mean, (cx, cy)))

        rows.sort(key=lambda t: t[0])   # darkest first
        return [pt for _, pt in rows[:max_centroids]]

    @staticmethod
    def compute_edge_density(gray: np.ndarray) -> float:
        """Sobel-gradient edge density — mirrors _compute_edge_density()."""
        g  = gray.astype(np.float32)
        gx = np.gradient(g, axis=1)
        gy = np.gradient(g, axis=0)
        return float((np.sqrt(gx ** 2 + gy ** 2) > 20.0).mean())

    @staticmethod
    def classify_stability(
        circle_count: int,
        square_count: int,
        edge_density: float,
    ) -> bool:
        """
        Returns True when the frame is structurally unstable.
        Mirrors _classify_stability() in analyzer.py.
        """
        total        = max(circle_count + square_count, 1)
        circle_ratio = circle_count / total
        has_comp     = square_count > 2 or edge_density > _EDGE_DENSITY_MIN
        return circle_ratio > _CIRCLE_RATIO_UNSTABLE and not has_comp

    @staticmethod
    def cluster_centers(
        pts: List[Tuple[int, int]],
        min_dist: int = 24,
    ) -> List[Tuple[int, int]]:
        """
        Greedy cluster merge: groups nearby points and returns their centroids.
        min_dist is expressed in analysis-frame pixels (scaled up externally).
        O(n²) — safe for the typical patch counts (< 300 per analysis frame).
        """
        if not pts:
            return []
        used: List[bool] = [False] * len(pts)
        clusters: List[List[Tuple[int, int]]] = []

        for i, p in enumerate(pts):
            if used[i]:
                continue
            grp = [p]
            used[i] = True
            for j in range(i + 1, len(pts)):
                if not used[j]:
                    q = pts[j]
                    if abs(p[0] - q[0]) + abs(p[1] - q[1]) < min_dist:
                        grp.append(q)
                        used[j] = True
            clusters.append(grp)

        return [
            (int(np.mean([p[0] for p in g])), int(np.mean([p[1] for p in g])))
            for g in clusters
        ]


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _dashed_line(
    frame:     np.ndarray,
    pt1:       Tuple[int, int],
    pt2:       Tuple[int, int],
    color:     Tuple[int, int, int],
    thickness: int = 1,
    dash_len:  int = 8,
    gap_len:   int = 6,
) -> None:
    """Draws a dashed line from pt1 to pt2 directly on frame (in-place)."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    total  = max(1, int(np.hypot(dx, dy)))
    period = dash_len + gap_len

    for start in range(0, total, period):
        t0 = start / total
        t1 = min(1.0, (start + dash_len) / total)
        p0 = (int(x1 + dx * t0), int(y1 + dy * t0))
        p1 = (int(x1 + dx * t1), int(y1 + dy * t1))
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)


def _nearest(
    origin:     Tuple[int, int],
    candidates: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Returns the candidate point closest to origin (Euclidean)."""
    if not candidates:
        return None
    ox, oy = origin
    return min(candidates, key=lambda p: (p[0] - ox) ** 2 + (p[1] - oy) ** 2)


# ── Pipeline effect wrapper ────────────────────────────────────────────────────

class HyperdimensionalMatrixEffect(BaseVisionMode):
    """
    Pipeline effect: Hyperdimensional Matrix Analyzer overlay.

    Subclasses BaseVisionMode so it plugs into the PIPELINE_EFFECTS registry
    and can be layered on top of any base vision mode by VisionPipeline.

    The effect is purely additive — it never transforms the colour space of
    the incoming frame, only draws structural annotation markers on top of it.
    This means "Dog Vision" stays dichromatic blue/yellow; this layer simply
    draws its crosses, rings, and lines on top of that canine colour field.

    Individual layers can be toggled at construction time (all on by default):
        enable_curvature         — red crosshairs / cyan rings per patch
        enable_susy_lines        — dashed cyan vector lines (SUSY pairs only)
        enable_grav_aura         — concentric halo at dark zone centroids
        enable_instability_border— double red inset rectangle when unstable

    Performance notes
    -----------------
    • Heavy analysis (FFT, gradient, patch scan) runs on a ≤320px downscale.
    • The result is cached for analysis_skip frames (~10 updates/sec at 60 fps).
    • Every captured frame still gets its draw pass (smooth visual).
    • The gravitational aura uses a single cv2.addWeighted call on the full frame
      because the circles are few (≤ 5 centroids) and the blend is cheap.
    """

    def __init__(
        self,
        enable_curvature:           bool = True,
        enable_susy_lines:          bool = True,
        enable_grav_aura:           bool = True,
        enable_instability_border:  bool = True,
        analysis_skip:              int  = _ANALYSIS_SKIP,
        patch_size:                 int  = _PATCH_SIZE,
    ) -> None:
        self._enable_curvature          = enable_curvature
        self._enable_susy_lines         = enable_susy_lines
        self._enable_grav_aura          = enable_grav_aura
        self._enable_instability_border = enable_instability_border
        self._analysis_skip             = max(1, analysis_skip)
        self._patch_size                = patch_size

        self._frame_count: int = 0
        self._cache: Optional[_AnalysisResult] = None

    # ── BaseVisionMode interface ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "HD Matrix Analyzer"

    @property
    def description(self) -> str:
        return (
            "Structural overlay: patch curvature, SUSY vector lines, "
            "gravitational aura, instability detection."
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw structural analysis markers onto frame (BGR uint8, H×W×3).

        Analysis re-runs every self._analysis_skip frames; drawing executes
        on every call using the cached result.  Returns the same frame array
        (modified in-place) so no extra allocation is needed.
        """
        self._frame_count += 1

        if self._cache is None or self._frame_count % self._analysis_skip == 1:
            self._cache = self._analyse(frame)

        # Skip draw if the frame dimensions changed since the last analysis
        # (e.g., window resize mid-flight).
        fh, fw = frame.shape[:2]
        if self._cache.frame_h == fh and self._cache.frame_w == fw:
            self._draw(frame, self._cache)

        return frame

    # ── Internal analysis ──────────────────────────────────────────────────

    def _analyse(self, frame: np.ndarray) -> _AnalysisResult:
        """
        Downscale → grayscale → run all analysis passes → scale coords back up.
        """
        fh, fw = frame.shape[:2]
        scale  = min(1.0, _MAX_ANALYSIS_DIM / max(fh, fw, 1))
        aw, ah = max(1, int(fw * scale)), max(1, int(fh * scale))

        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # ── Patch scan ────────────────────────────────────────────────────
        ang_a, smo_a, n_smooth, n_angular = _HDEngine.scan_patches(
            gray, self._patch_size
        )

        # ── Symmetry / SUSY ───────────────────────────────────────────────
        _, _, is_susy = _HDEngine.compute_symmetry(gray)

        # ── Gravitational field ───────────────────────────────────────────
        grav_density  = _HDEngine.compute_gravitational_density(gray)
        grav_centroids: List[Tuple[int, int]] = []
        if grav_density > _GRAV_DENSITY_MIN:
            grav_centroids = _HDEngine.find_dark_centroids(gray)

        # ── Structural stability ──────────────────────────────────────────
        edge_density = _HDEngine.compute_edge_density(gray)
        is_unstable  = _HDEngine.classify_stability(n_smooth, n_angular, edge_density)

        # ── Cluster centres for SUSY lines ────────────────────────────────
        # min_dist is 1.5× the patch size so adjacent patches merge into one cluster.
        cluster_gap   = int(self._patch_size * 1.5)
        circ_clusters = _HDEngine.cluster_centers(smo_a, min_dist=cluster_gap)
        sharp_clusters = _HDEngine.cluster_centers(ang_a, min_dist=cluster_gap)

        # ── Scale all analysis-frame coords → full-frame coords ───────────
        inv = 1.0 / scale if scale > 0 else 1.0

        def _up(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            return [(int(x * inv), int(y * inv)) for x, y in pts]

        return _AnalysisResult(
            angular_pts     = _up(ang_a),
            smooth_pts      = _up(smo_a),
            circle_clusters = _up(circ_clusters),
            sharp_clusters  = _up(sharp_clusters),
            is_susy         = is_susy,
            grav_density    = grav_density,
            grav_centroids  = _up(grav_centroids),
            is_unstable     = is_unstable,
            frame_h         = fh,
            frame_w         = fw,
        )

    # ── Draw dispatch ──────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, r: _AnalysisResult) -> None:
        """Call each enabled draw layer in the correct stacking order."""
        if self._enable_grav_aura:
            self._draw_grav_aura(frame, r)

        if self._enable_curvature:
            self._draw_curvature_markers(frame, r)

        if self._enable_susy_lines and r.is_susy:
            self._draw_susy_lines(frame, r)

        if self._enable_instability_border and r.is_unstable:
            self._draw_instability_border(frame)

    # ── Layer 1: Patch curvature markers ──────────────────────────────────

    def _draw_curvature_markers(self, frame: np.ndarray, r: _AnalysisResult) -> None:
        """
        Red crosshair at each angular patch centre; cyan hollow ring at each smooth
        patch centre.  Marker size is scaled to ~1/120th of the frame width so the
        markers remain proportional across resolutions.
        """
        fw         = frame.shape[1]
        cross_size = max(8, min(fw // 120, 18))
        ring_rad   = max(6, min(fw // 150, 14))
        thickness  = 1

        for cx, cy in r.angular_pts:
            cv2.drawMarker(
                frame, (cx, cy), _C_ANGULAR,
                cv2.MARKER_CROSS, cross_size, thickness, cv2.LINE_AA,
            )
            cv2.circle(frame, (cx, cy), 2, _C_ANGULAR, -1, cv2.LINE_AA)

        for cx, cy in r.smooth_pts:
            cv2.circle(frame, (cx, cy), ring_rad, _C_SMOOTH, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, _C_SMOOTH, -1, cv2.LINE_AA)

    # ── Layer 2: SUSY vector lines ─────────────────────────────────────────

    def _draw_susy_lines(self, frame: np.ndarray, r: _AnalysisResult) -> None:
        """
        Dashed cyan lines connecting each smooth cluster centre to its nearest
        sharp angular complement — the Stabilization-Atom ↔ Complement pairing.
        Links longer than 45 % of the frame width are suppressed (cross-frame
        connections add visual noise without meaningful spatial information).
        """
        max_dist = frame.shape[1] * 0.45

        for circ in r.circle_clusters:
            near = _nearest(circ, r.sharp_clusters)
            if near is None:
                continue
            if np.hypot(circ[0] - near[0], circ[1] - near[1]) > max_dist:
                continue
            _dashed_line(frame, circ, near, _C_SUSY, thickness=1, dash_len=8, gap_len=6)

    # ── Layer 3: Gravitational aura ────────────────────────────────────────

    def _draw_grav_aura(self, frame: np.ndarray, r: _AnalysisResult) -> None:
        """
        Concentric halo rings at each dark-zone centroid position.

        Rendered as a blended overlay so the glow has natural transparency:
          • outer filled disc  — diffuse violet ambient glow (low alpha)
          • inner core ring    — brighter, tight stroke circle
          • mid concentric ring — intermediate stroke at 2/3 of aura radius

        All circles for all centroids are batched onto one overlay copy before
        the single addWeighted call, keeping memory allocation at O(1) per frame.
        """
        if r.grav_density <= _GRAV_DENSITY_MIN or not r.grav_centroids:
            return

        intensity = min(1.0, (r.grav_density - _GRAV_DENSITY_MIN) / 0.40)
        fw        = frame.shape[1]
        base_r    = max(20, int(fw * 0.018))
        aura_r    = int(base_r + intensity * base_r * 2.2)
        alpha     = 0.10 + 0.22 * intensity   # blend fraction [0.10, 0.32]

        overlay = frame.copy()
        for cx, cy in r.grav_centroids:
            # Outer diffuse filled circle
            cv2.circle(overlay, (cx, cy), aura_r,          _C_GRAV_OUT, -1, cv2.LINE_AA)
            # Mid concentric ring
            cv2.circle(overlay, (cx, cy), aura_r * 2 // 3, _C_GRAV_IN,   1, cv2.LINE_AA)
            # Inner bright core ring
            cv2.circle(overlay, (cx, cy), max(4, aura_r // 3), _C_GRAV_IN, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # ── Layer 4: Instability border ────────────────────────────────────────

    def _draw_instability_border(self, frame: np.ndarray) -> None:
        """
        Double red inset rectangle drawn along the screen margins when the frame
        is classified as structurally unstable (circle patches dominate without
        any linear / sharp complement to anchor them).

        Inset distances scale proportionally so the border looks consistent
        across 720p, 1080p, and 1440p captures.
        """
        h, w   = frame.shape[:2]
        edge   = min(w, h)
        inset1 = max(4,  int(edge * 0.006))
        inset2 = inset1 + max(3, int(edge * 0.005))

        # Outer ring — heavier stroke
        cv2.rectangle(
            frame,
            (inset1, inset1), (w - inset1 - 1, h - inset1 - 1),
            _C_UNSTABLE, 2, cv2.LINE_AA,
        )
        # Inner ring — lighter stroke, tighter
        cv2.rectangle(
            frame,
            (inset2, inset2), (w - inset2 - 1, h - inset2 - 1),
            _C_UNSTABLE, 1, cv2.LINE_AA,
        )
