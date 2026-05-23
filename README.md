# Vision Simulator

A real-time screen-overlay application that transforms your display through
**20 biologically and scientifically grounded perception modes** — from animal
eyes and infrared sensing to human visual conditions and machine vision.

A fullscreen, click-through PySide6 overlay sits on top of your desktop and
applies the selected filter live to every pixel of your screen, at up to 60 fps,
with zero Python-level pixel loops.

---

## Features

- **20 vision modes** covering animal eyes, scientific imaging, human conditions, and AI
- **Multi-overlay engine** — run several filters simultaneously, each in its own window
- **Split-screen comparison** — view 2 or 4 modes side-by-side without switching
- **Window-aware filtering** — pin a filter to a specific application window
- **VisionPipeline** — combine any base mode with a post-processing effect (blur, glow, CLAHE, …)
- **Floating control panel** — sidebar-navigated GUI with live overlay management
- **Global hotkeys** — work from any application, no window focus required
- **Click-through overlay** — the filtered view never blocks mouse or keyboard input

---

## Vision Modes

| # | Key | Mode | Scientific basis |
|---|-----|------|-----------------|
| 1 | `1` | Dog Vision | Dichromatic S+L cones — blue/yellow |
| 2 | `2` | Cat Vision | Dichromatic + tapetum lucidum low-light boost |
| 3 | `3` | Bird Vision | Tetrachromatic + UV fourth cone channel |
| 4 | `4` | Insect Vision | Compound eye: hexagonal facets, UV shift |
| 5 | `5` | Snake Thermal | Blended IR pit-organ + normal visual field |
| 6 | `6` | **Pit Viper Vision ★** | Full thermal pipeline: cold=blurry/blue, hot=sharp/white |
| 7 | `7` | Shark Vision | Monochromatic blue + electroreception shimmer |
| 8 | `8` | Frog Vision | Motion-only retina: static=dim, moving=bright green |
| 9 | `9` | UV Vision | Ultraviolet spectral range mapped to visible |
| 10 | `0` | Depth Map | Edge-sharpness monocular depth, thermal colour-coded |
| 11 | N/P | AI Edge Vision | Canny edge detection — machine perception |
| 12–14 | N/P | Color Blind ×3 | Deuteranopia / Protanopia / Tritanopia (Viénot 1999) |
| 15 | N/P | Tunnel Vision | Radial peripheral blur + vignette |
| 16–19 | N/P | Emotional ×4 | Happy / Sad / Angry / Fearful colour-temperature shifts |
| 20 | N/P | **Octopus Polarized Vision** | Sobel → arctan2 → HSV hue encodes polarization angle |

> Modes 11–20 cycle via `N` (next) / `P` (previous) since they have no dedicated number key.

---

## Architecture

```
vision-simulator/
│
├── main.py                       ← Entry point · hotkeys · shutdown sequence
│
├── core/
│   ├── engine.py                 ← VisionEngine: opens screen-capture context
│   ├── screen_capture.py         ← MSS real-time screen capture
│   ├── overlay_window.py         ← Fullscreen click-through QWidget (Win32 WS_EX_TRANSPARENT)
│   ├── overlay_manager.py        ← Owns all OverlayWindows · frame distribution · HUD
│   ├── frame_worker.py           ← QThread: capture loop → manager.distribute()
│   ├── split_screen_manager.py   ← 4 layout modes · resize-before-apply optimisation
│   └── vision_pipeline.py        ← Chains base_mode + effects sequentially
│
├── modes/
│   ├── base_mode.py              ← Abstract BaseVisionMode (ABC)
│   ├── dog_vision.py             ← Dichromatic S+L
│   ├── cat_vision.py             ← Dichromatic + low-light
│   ├── bird_vision.py            ← Tetrachromatic + UV
│   ├── insect_vision.py          ← Compound eye
│   ├── snake_thermal.py          ← IR blend
│   ├── pit_viper.py              ← ★ Full thermal pipeline
│   ├── shark_vision.py           ← Monochromatic + shimmer
│   ├── frog_vision.py            ← Motion-only retina
│   ├── uv_vision.py              ← UV range
│   ├── depth_map.py              ← Monocular depth
│   ├── ai_edge.py                ← Canny edges
│   ├── colorblind.py             ← CVD simulation (3 types)
│   ├── tunnel_vision.py          ← Peripheral blur
│   ├── emotional.py              ← Colour-temperature affect
│   └── polarized_vision.py       ← Octopus polarization angle (Mäthger 2009)
│
├── effects/
│   ├── color_filters.py          ← Channel weights, hue shift, CVD matrices
│   ├── blur.py                   ← Gaussian, selective, radial, motion blur
│   ├── contrast.py               ← CLAHE, gamma correction, linear contrast
│   ├── overlays.py               ← Vignette, hex grid, scan lines, glow/bloom
│   ├── lut.py                    ← LUT builders + cv2.LUT application
│   └── pipeline_effects.py       ← BaseVisionMode wrappers for VisionPipeline
│
├── ui/
│   ├── __init__.py
│   └── main_window.py            ← ControlPanel: sidebar QMainWindow (5 views)
│
├── utils/
│   ├── config.py                 ← All constants and defaults
│   ├── mode_registry.py          ← Single source of truth for all registered modes
│   ├── image_utils.py            ← frame_to_qimage, resize helpers
│   ├── math_utils.py             ← Normalisation, radial maps
│   └── window_manager.py         ← Win32 ctypes window enumeration + tracking
│
├── assets/
│   ├── luts/                     ← Drop .npy LUT files here (gitignored)
│   └── textures/                 ← Drop texture overlays here (gitignored)
│
├── requirements.txt
└── README.md
```

### Frame pipeline

```
ScreenCapture.capture()  (worker thread)
    │
    ├─[split-screen active]──► SplitScreenManager.compose(raw)
    │                               ├─ resize panel to 50 % / 25 % area
    │                               ├─ mode.apply(panel)          ← C/SIMD speed
    │                               └─ stitch panels + draw labels
    │
    └─[standard]──► per overlay:
                        VisionMode.apply(raw)        ← or VisionPipeline chain
                            └─► HUD bar composite
                                └─► QImage → QueuedConnection → OverlayWindow.paintEvent()
```

### Adding a new vision mode (3 steps)

1. Create `modes/my_mode.py` subclassing `BaseVisionMode`.
2. Implement `name` (property) and `apply(frame) -> np.ndarray`.
3. Add `MyMode()` to the list in `utils/mode_registry.py`.

No other file changes required.

### Adding a pipeline effect (2 steps)

1. Add a wrapper class in `effects/pipeline_effects.py` subclassing `BaseVisionMode`.
2. Append `("Display Name", MyEffect())` to `PIPELINE_EFFECTS`.

It will appear automatically in the control panel's effect combo box.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/JulianWIAI/vision-simulator.git
cd vision-simulator

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

> **Windows note**: The `keyboard` library requires running the terminal as
> Administrator for global hotkey capture to work across all applications.

---

## Controls

### Global hotkeys (work from any application)

| Key | Action |
|-----|--------|
| `1` – `9` | Set mode 1–9 on the most recently added overlay |
| `0` | Set mode 10 on the most recently added overlay |
| `N` | Add a new overlay (starts at mode 1) |
| `M` | Cycle the last overlay forward through all modes |
| `X` | Remove the most recently added overlay |
| `C` | Toggle the Control Panel window |
| `ESC` | Exit split-screen (if active), or quit the application |

> Hotkeys `N / M / X / 1–9 / 0` are suppressed while the Control Panel has
> keyboard focus so that typing in the panel's fields does not accidentally
> trigger overlay actions.

### Control Panel — Regions view

| Control | Action |
|---------|--------|
| + Add Overlay | Create a new full-screen overlay |
| ✕ Remove Selected | Remove the selected overlay |
| Mode combo + Set Mode | Assign a standard mode to the selected overlay |
| Base + Effect combos + Apply Pipeline | Wrap a mode + effect into a VisionPipeline and apply |

### Control Panel — Split Screen view

| Layout | Panels |
|--------|--------|
| Off | Normal per-overlay rendering |
| Top / Bottom | Two panels stacked vertically (A top, B bottom) |
| Left / Right | Two panels side-by-side (A left, B right) |
| 2×2 Grid | Four panels (A=TL, B=TR, C=BL, D=BR) |

---

## Split-Screen Performance Model

Split-screen resizes each panel **before** calling `mode.apply()`, so the
vision mode processes a fraction of the original resolution:

| Layout | Per-panel pixels (1920×1080 source) |
|--------|-------------------------------------|
| 2× vertical / horizontal | ~1.04 M (50 % of 2.07 M) |
| 4× grid | ~518 K (25 % of 2.07 M) |

This gives a **2–4× throughput improvement** over rendering at full resolution
per panel, with no Python-level pixel loops anywhere in the pipeline.

---

## Pit Viper Vision — Feature Deep Dive

The flagship mode models all known physiological properties of pit-viper
infrared sensing (TRPA1 membrane receptors, Gracheva et al., 2010):

| Stage | Implementation | Physiology modelled |
|-------|---------------|---------------------|
| 1 | `cv2.cvtColor(BGR→GRAY)` | Luminance ≈ thermal proxy |
| 2 | `cv2.equalizeHist` | Differential temperature sensitivity |
| 3 | Custom thermal LUT | False-colour IR palette |
| 4 | NumPy threshold mask | Cold vs warm region segmentation |
| 5 | `selective_blur` | Low spatial resolution of pit organ (~700 receptors) |
| 6 | `add_scan_lines` | Sensor-readout grid texture |
| 7 | `add_glow` | Optical bloom / halation around heat sources |

---

## Octopus Polarized Vision — Feature Deep Dive

Octopuses are monochromatic yet show remarkable colour-matching behaviour.
The leading hypothesis (Mäthger et al., 2009): they detect the *polarization
angle* of light via orthogonal rhabdomere orientations.

| Stage | Implementation | Modelling |
|-------|---------------|-----------|
| 1 | Grayscale conversion | Strip wavelength; keep luminance |
| 2 | Sobel X/Y (CV_32F) | Detect spatial transitions |
| 3 | `arctan2(Gy, Gx)` → hue | Map angle [−π, π] → HSV hue [0, 179] |
| 4 | `cv2.magnitude()` | SIMD-accelerated gradient strength → HSV value |
| 5 | HSV → BGR | Saturated colour encodes polarization angle |

Flat regions appear black (weak gradient); edges burst with colour whose hue
encodes the local polarization orientation.

---

## VisionPipeline — Effect Layering

Any overlay can be assigned a `VisionPipeline` instead of a plain mode.
A pipeline chains a **base mode** and up to one **effect** sequentially:

```
raw frame  →  base_mode.apply()  →  effect.apply()  →  output
```

Available effects:

| Effect | Description |
|--------|-------------|
| Gaussian Blur | Softens output — models optical defocus |
| High Contrast | CLAHE local contrast boost on LAB L-channel |
| Vignette | Radial edge-darkening to focus attention centrally |
| Scan Lines | CRT / thermal-camera row texture |
| Glow | Bloom / halo around bright image regions |
| Radial Blur | Peripheral blur increasing toward frame edges |

---

## Libraries

| Library | Version | Role |
|---------|---------|------|
| `opencv-python` | ≥ 4.8 | Image processing, colour transforms, HUD rendering |
| `numpy` | ≥ 1.24 | Vectorised pixel math, masks, LUT arrays |
| `PySide6` | ≥ 6.6 | Qt overlay windows, control panel, signal/slot threading |
| `mss` | ≥ 9.0 | Fast real-time screen capture |
| `Pillow` | ≥ 10.0 | Asset loading (textures, LUT images) |
| `keyboard` | ≥ 0.13 | Global hotkeys (OS-level, works outside window focus) |

---

## Performance Notes

- **No Python-level pixel loops** — all transforms use NumPy slicing and
  OpenCV / SIMD-accelerated functions throughout.
- **Cached overlays** — hexagonal grids, vignette maps, and tunnel masks are
  computed once per frame resolution and reused on every subsequent frame.
- **LUT precomputation** — thermal and night-vision LUTs are built at mode
  instantiation, not per frame.
- **Resize-before-apply** — split-screen panels are downscaled before the mode
  runs, reducing pixel count by 50–75 %.
- **Per-overlay backpressure** — each overlay drops incoming frames if its
  previous frame has not yet been painted, capping queue depth at one and
  preventing memory build-up at high frame rates.
- Typical throughput: **30–60 fps** at 1920×1080 on a modern CPU.

---

## Thread Model

```
Main thread (Qt event loop)
  ├── OverlayManager          — owns mode list + OverlayWindow list
  │     └── OverlayWindow[N]  — each: fullscreen, click-through, own mode
  ├── ControlPanel            — floating GUI, timer-driven list refresh
  └── FrameWorker (QThread)   — capture → distribute() loop
        └─► QueuedConnection  → OverlayWindow.paintEvent() [GUI thread]

keyboard library thread       — fires hotkey callbacks; uses QTimer.singleShot
                                to marshal GUI-thread-required calls safely
```

---

## References

- Gracheva E.O. et al. (2010). *Molecular basis of infrared detection by snakes.* Nature 464, 1006–1011.
- Mäthger L.M. et al. (2009). *Evidence for polarisation vision in the octopus.* J. Exp. Biol. 212, 2133–2140.
- Lettvin J.Y. et al. (1959). *What the frog's eye tells the frog's brain.* Proc. IRE 47(11), 1940–1951.
- Viénot F., Brettel H. & Mollon J.D. (1999). *Digital video colourmaps for checking the legibility of displays by dichromats.* Color Research & Application 24(4), 243–252.
- Fredrickson B.L. (2001). *The role of positive emotions in positive psychology.* American Psychologist 56(3), 218–226.

---

*Built with Python 3.11+ · OpenCV · NumPy · PySide6 · MSS*
