# Vision Simulator

A real-time, fullscreen Windows screen-overlay application that renders live simulations of non-human and altered visual perception directly on top of the desktop — without interrupting click-through interaction with any underlying application.

Built with **PySide6 (Qt for Python)**, **OpenCV**, **mss**, and **NumPy**. Targets 60 FPS on a modern CPU with no GPU requirement and zero Python-level pixel loops.

---

## Feature Highlights

| Feature | Detail |
|---|---|
| **21 built-in vision modes** | Animal eyes, dichromacy, thermal, UV, AI edge, and more |
| **VisionPipeline effect chain** | Stack a post-processing effect on any base mode; effects survive mode switching |
| **Multi-overlay** | Run several independent overlays simultaneously with different modes |
| **Split-Screen Comparison** | Tile 2 or 4 modes side-by-side (Top/Bottom, Left/Right, 2×2 Grid) |
| **Click-through** | Overlay is fully transparent to mouse and keyboard input; desktop stays interactive |
| **Split-screen click remapping** | Win32 `WH_MOUSE_LL` hook corrects click positions inside scaled split panels |
| **Window-tracked overlays** | Clip any overlay to a specific Win32 window; the clip rect follows as the window moves |
| **Draw Region** | Click-drag to paint a free rectangle on screen — the filter applies only inside it; multiple regions supported |
| **HD Matrix Analyzer** | Spatial curvature analysis, SUSY symmetry lines, gravitational aura, and instability border |
| **Control Panel GUI** | Floating dark-theme sidebar panel for all settings (show/hide with `C`) |
| **Capture exclusion** | Overlay is hidden from MSS / BitBlt so it never feeds back into its own capture |

---

## System Requirements

| Item | Minimum |
|---|---|
| OS | Windows 10 version 2004 (build 19041) or later |
| Python | 3.10+ |
| CPU | Any modern x86-64 |
| RAM | 512 MB free |
| Display | Primary monitor (multi-monitor supported; overlay always covers the primary) |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/JulianWIAI/vision-simulator.git
cd vision-simulator/vision_simulator

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell
# or: .\.venv\Scripts\activate.bat  # Command Prompt

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

> **PowerShell execution policy:** if activation fails with "scripts are disabled", run once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.8 | Frame resizing, colour transforms, HUD text rendering |
| `numpy` | ≥ 1.24 | All per-pixel array math — zero Python-level pixel loops |
| `mss` | ≥ 9.0 | Fast real-time screen capture at physical pixels |
| `Pillow` | ≥ 10.0 | Asset loading (LUT images, textures) |
| `keyboard` | ≥ 0.13 | System-wide hotkeys — work without window focus |
| `PySide6` | ≥ 6.6 | Qt for Python — overlay window, control panel, signal/slot threading |

---

## Quick Start

**Option A — double-click launcher:**  
Double-click `run.bat` in the project folder. Windows may show a SmartScreen prompt the first time; click **More info → Run anyway**.

**Option B — terminal:**
```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

The terminal prints the full mode list and hotkey map. One overlay starts immediately in **mode 1** covering the primary screen. The **Control Panel** opens in the top-right corner.

Press `C` to toggle the Control Panel at any time.  
Press `°` (degree symbol) to quit, or close the Control Panel window.

---

## Keyboard Controls

| Key | Action |
|---|---|
| `N` | Add a new overlay |
| `M` | Cycle the last overlay forward through all 21 modes |
| `1` – `9` | Set mode 1–9 on the last overlay |
| `0` | Set mode 10 on the last overlay |
| `X` | Remove the last overlay |
| `C` | Toggle the Control Panel window |
| `°` | First press: disable split-screen. Second press (or when no split is active): quit |

> `N`, `M`, `X`, `1–9`, `0` are suppressed while the Control Panel has OS focus so typing in combo boxes does not fire overlay actions.

---

## Vision Modes

All 21 modes are available as base modes, in the per-overlay selector, in the Global Modes override, and in every split-screen panel slot.

| # | Mode | Description |
|---|---|---|
| 1 | **Dog Vision** | Dichromatic (S+L cones) — blue/yellow axis; reds appear as desaturated grey |
| 2 | **Cat Vision** | Dichromatic + peripheral blur; reduced acuity outside the central fovea |
| 3 | **Bee Vision** | UV-shifted trichromacy; UV-reflective patterns invisible to humans become visible |
| 4 | **Bull Vision** | Dichromatic (blue-yellow axis only); all reds collapse to grey |
| 5 | **Frog Vision** | High contrast + cold-hue shift; motion-emphasis colour mapping |
| 6 | **Pit Viper IR** | Thermal infrared simulation; luminance-to-heat-gradient mapping via custom LUT |
| 7 | **Mantis Shrimp** | 16-channel hyperspectral simulation; extreme hue rotation across the visible spectrum |
| 8 | **Eagle Vision** | 4–8× acuity boost via CLAHE sharpening + telephoto centre-crop |
| 9 | **Colour Blind — Protanopia** | Red-green deficiency (missing L-cone); Viénot 1999 simulation matrix |
| 10 | **Colour Blind — Deuteranopia** | Red-green deficiency (missing M-cone) |
| 11 | **Colour Blind — Tritanopia** | Blue-yellow deficiency (missing S-cone) |
| 12 | **Night Vision** | Phosphor-green amplification; simulates Gen-III image-intensifier tubes |
| 13 | **Thermal Camera** | False-colour heat map (cold = blue → warm = red); CLAHE normalisation |
| 14 | **Echolocation Map** | Depth-cue estimation from luminance gradients; bat sonar inspired |
| 15 | **Compound Eye** | Hexagonal facet grid tiling; approximates insect ommatidial array |
| 16 | **Infrared** | Near-IR simulation via inverted luminance + red-channel boost |
| 17 | **Ultraviolet** | UV-band simulation; shifts blue channel into near-violet and enhances surface patterning |
| 18 | **Monochrome** | Perceptual greyscale (BT.601 luminance coefficients: R×0.299, G×0.587, B×0.114) |
| 19 | **High Contrast** | CLAHE local contrast enhancement; preserves hue, sharpens local detail |
| 20 | **Inverted** | Photographic negative; all channel values flipped around 127 |
| 21 | **Color Edge Overlay** | Full-colour scene with AI-detected edges rendered as thick dark outlines (cel-shading); same Canny + Sobel dual-detector pipeline as AI Edge Detection |

---

## VisionPipeline Effect Chain

The **Regions** tab in the Control Panel lets you compose a pipeline by stacking one post-processing **effect** on top of any **base mode**:

```
raw frame  →  base_mode.apply()  →  effect.apply()  →  display
```

### Available Effects

| Effect | Description |
|---|---|
| **None** | No post-processing; base mode output displayed directly |
| **Gaussian Blur** | Uniform Gaussian softening (kernel 15×15) |
| **High Contrast** | CLAHE local contrast boost (clip limit 2.0, tile grid 8×8) |
| **Vignette** | Radial edge-darkening to focus attention on the frame centre |
| **Scan Lines** | CRT / thermal-camera row-stripe texture (every 4th row, α 0.15) |
| **Glow** | Bloom / halo around bright image regions (kernel 21, intensity 0.4) |
| **Radial Blur** | Peripheral blur increasing with distance from the frame centre |
| **HD Matrix Analyzer** | Hyperdimensional spatial analysis overlay (see below) |

### HD Matrix Analyzer

A research-grade spatial analysis overlay rendered in four layers on every frame:

| Layer | Marker | Criterion |
|---|---|---|
| **Curvature markers** | Red crosshair = angular patch | High gradient orientation variance in 16×16 patch |
| | Cyan ring = smooth patch | Low gradient orientation variance |
| **SUSY vector lines** | Dashed lines between pairs | Nearest-neighbour symmetry between angular ↔ smooth markers |
| **Gravitational aura rings** | Concentric rings at dark-zone centroids | Rings weighted by local edge density |
| **Instability border** | Double red border | Global symmetry score < threshold → high perceptual chaos |

Analysis runs on a max-320-px downscale every 6 frames and the result is cached for intermediate frames, keeping per-frame overhead negligible.

### Pipeline Persistence on Mode Switch

When a `VisionPipeline` is active, pressing `M` or a digit key swaps only the **base mode** — the effect chain is **preserved**. The HUD shows `≡` instead of a numeric counter to indicate a pipeline is active. This is implemented in `core/pipeline_state.py` via in-place patching of `VisionPipeline.base_mode`.

---

## Draw Region

The **Windows** tab in the Control Panel includes a **Draw Region** tool that lets you apply any vision filter to an arbitrary rectangle on the screen — not bound to any specific window.

**How to use it:**

1. Open the Control Panel → go to the **Windows** tab.
2. Pick a vision mode from the **Draw Region** dropdown.
3. Click **✦ Draw Region on Screen** — the panel hides and a full-screen drawing canvas appears with a crosshair cursor and a faint dark tint.
4. Click and drag to define a rectangle. A live size indicator (e.g. `640 × 360`) appears at the corner while dragging.
5. Release the mouse — the canvas closes, the panel reappears, and the region overlay is created silently with no visible border.
6. Press `ESC` or right-click at any point to cancel.

Multiple regions can be drawn. Each appears in the **Regions** list labelled `[Region]` and supports the same mode-switching and pipeline effects as any other overlay. A region overlay can be removed by selecting it in the Regions list and clicking **✕ Remove Selected**.

---

## Split-Screen Comparison

Open the **Split Screen** tab in the Control Panel to select a layout and assign modes to each panel slot.

| Layout | Panels | Scale per panel |
|---|---|---|
| Off | — | Full frame, individual overlay modes |
| Top / Bottom | A (top) + B (bottom) | Full width × ≈50% height each |
| Left / Right | A (left) + B (right) | ≈50% width × full height each |
| 2×2 Grid | A (TL), B (TR), C (BL), D (BR) | ≈50% width × ≈50% height each |

Press `°` once to disable split-screen without quitting the application.

### Performance Model

`SplitScreenManager._process_panel()` calls `cv2.resize()` to downscale the raw desktop frame to panel size **before** calling `mode.apply()`. For a 1920×1080 source:

| Layout | Pixels per mode call | Reduction vs full frame |
|---|---|---|
| 2× (any) | ≈1.04 M | 2× faster |
| 4× grid | ≈518 K | 4× faster |

`np.vstack` / `np.hstack` stitch the panels back into the full-resolution output frame in a single C-level call with no Python pixel loops.

### Click Remapping in Split-Screen

A Win32 `WH_MOUSE_LL` low-level mouse hook (managed by `core/mouse_remap.py`) corrects click coordinates when split-screen is active.

**Problem:** each panel renders a scaled-down copy of the full desktop. A desktop element at position `(real_x, real_y)` appears at `(panel_x, panel_y)` inside the panel. Because the overlay passes all clicks through (`WS_EX_TRANSPARENT`), clicking at `(panel_x, panel_y)` hits the wrong real-desktop position.

**Fix:** the hook intercepts every physical button event and applies the inverse of the panel's `cv2.resize` scale:

```
Compose scale  (forward):  panel_y = real_y × (h_panel / H)
Remap   scale  (inverse):  real_y  = panel_y × (H / h_panel)
```

The hook then moves the system cursor to `(real_x, real_y)` via `SetCursorPos`, synthesises an identical button event via `SendInput`, and returns `1` to suppress the original. Injected events carry `LLMHF_INJECTED` in `MSLLHOOKSTRUCT.flags` and are passed through unchanged, preventing infinite recursion.

---

## Architecture

```
vision_simulator/
│
├── main.py                      Entry point · hotkeys · startup/shutdown sequence
│
├── core/
│   ├── engine.py                VisionEngine — opens the MSS screen-capture context
│   ├── frame_worker.py          FrameWorker (QThread) — capture loop → distribute()
│   ├── overlay_manager.py       Owns the OverlayWindow list · frame distribution · HUD rendering
│   ├── overlay_window.py        Fullscreen click-through QWidget + Win32 extended-style hardening
│   ├── split_screen_manager.py  Panel composition engine — 4 layout modes
│   ├── vision_pipeline.py       VisionPipeline — chains base_mode → effect(s) sequentially
│   ├── pipeline_state.py        Stateless helpers for pipeline-safe mode switching
│   └── mouse_remap.py           SplitScreenMouseRemapper — WH_MOUSE_LL click coordinate fix
│
├── modes/
│   ├── base_mode.py             BaseVisionMode abstract base class
│   └── <21 mode files>          One file per vision mode
│
├── effects/
│   ├── blur.py                  gaussian_blur(), radial_blur()
│   ├── contrast.py              clahe_enhance()
│   ├── overlays.py              add_vignette(), add_scan_lines(), add_glow()
│   ├── hd_matrix_analyzer.py    HyperdimensionalMatrixEffect + _HDEngine analysis engine
│   └── pipeline_effects.py      PIPELINE_EFFECTS ordered registry · BaseVisionMode wrappers
│
├── ui/
│   ├── main_window.py           ControlPanel (QMainWindow) — 5-tab sidebar interface
│   └── region_drawer.py         RegionDrawer — full-screen drag-to-draw region selector
│
└── utils/
    ├── config.py                Global constants (HUD font, capture settings, colours)
    ├── image_utils.py           frame_to_qimage() — OpenCV BGR → Qt QImage (zero-copy path)
    ├── mode_registry.py         get_all_modes() — single authoritative ordered mode list
    └── window_manager.py        Win32 window enumeration and clip-rect tracking
```

### Multi-Threaded Frame Pipeline

```
OS mouse-hook thread  (SplitScreenMouseRemapper._pump_thread)
  └── WH_MOUSE_LL callback — pure Win32; no Qt calls; GIL released in GetMessageW

Main thread  (Qt event loop)
  ├── ControlPanel slots         — all GUI interactions
  ├── OverlayWindow.paintEvent() — draws QPixmap to screen, clears _busy backpressure flag
  └── QTimer callbacks           — track-timer (100 ms clip-rect refresh), refresh-timer (500 ms)

FrameWorker thread  (QThread)
  └── capture loop:
        mss.grab()                        (C extension, releases GIL during DXGI/BitBlt)
        → numpy.frombuffer               (zero-copy view — no allocation)
        → manager.distribute(raw_frame)
              ├── [split active] SplitScreenManager.compose()
              │       ├── cv2.resize → panel      (C/SIMD, releases GIL)
              │       ├── mode.apply(panel)        (C/SIMD, releases GIL)
              │       └── np.vstack / hstack      (C, single allocation)
              └── [standard] per overlay:
                      mode.apply(raw)             (C/SIMD, releases GIL)
                      HUD composite               (OpenCV C)
                      overlay.submit_frame()      (emits QueuedConnection signal)
                                                     │
                                              (GUI thread) _on_frame() → paintEvent()
```

**Backpressure:** each `OverlayWindow` has a `_busy: bool` flag. `submit_frame()` silently drops the incoming frame when `_busy` is `True`, capping the Qt event-queue depth at exactly one frame per overlay. This prevents memory growth when the GUI thread falls behind the capture rate.

**Cross-thread safety:** all mode switches are single attribute assignments under CPython's GIL (atomic). An explicit `threading.Lock` protects only the `_overlays` list in `OverlayManager` against concurrent add/remove and distribute iteration.

### Win32 Overlay Hardening

`OverlayWindow._apply_win32_clickthrough()` writes these extended styles via `SetWindowLongW` after every `showEvent()`:

| Flag | Hex | Effect |
|---|---|---|
| `WS_EX_LAYERED` | `0x00080000` | Required companion to `WS_EX_TRANSPARENT` |
| `WS_EX_TRANSPARENT` | `0x00000020` | OS answers `WM_NCHITTEST` with `HTTRANSPARENT` before Qt |
| `WS_EX_TOOLWINDOW` | `0x00000080` | Removes window from taskbar and Alt-Tab |
| `WS_EX_NOACTIVATE` | `0x08000000` | Prevents overlay from stealing keyboard focus |
| `~WS_EX_APPWINDOW` | cleared | Ensures no ghost taskbar entry appears |

`SetWindowPos` with `SWP_FRAMECHANGED` flushes all style changes synchronously without moving or resizing the window.

`SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAP)` hides the overlay from all capture APIs (MSS, BitBlt, OBS) to prevent recursive visual feedback.

---

## Extending the Project

### Adding a New Vision Mode

**Step 1** — create `modes/my_mode.py`:

```python
from modes.base_mode import BaseVisionMode
import numpy as np
import cv2

class MyMode(BaseVisionMode):

    @property
    def name(self) -> str:
        return "My Mode"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        # frame: BGR uint8 (H, W, 3) — return same shape
        return cv2.bitwise_not(frame)
```

**Step 2** — register in `utils/mode_registry.py`:

```python
from modes.my_mode import MyMode

def get_all_modes():
    return [
        ...,
        MyMode(),   # appears at index len-1; no other file changes needed
    ]
```

The new mode appears automatically in the HUD counter, all combo boxes, the split-screen panel selectors, and is reachable via `M`-cycling. If its index is ≤ 9, it also gets a digit hotkey.

### Adding a New Pipeline Effect

**Step 1** — subclass `BaseVisionMode` in a new file under `effects/`:

```python
from modes.base_mode import BaseVisionMode
import numpy as np, cv2

class SharpenEffect(BaseVisionMode):

    @property
    def name(self) -> str:
        return "Sharpen"

    def apply(self, frame: np.ndarray) -> np.ndarray:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        return cv2.filter2D(frame, -1, kernel)
```

**Step 2** — add to `PIPELINE_EFFECTS` in `effects/pipeline_effects.py`:

```python
from effects.my_sharpen import SharpenEffect

PIPELINE_EFFECTS = [
    ("None",    None),
    ...,
    ("Sharpen", SharpenEffect()),
]
```

The effect appears immediately in the Control Panel's **Apply Pipeline Effect** combo box.

---

## Performance Notes

- **Zero Python-level pixel loops.** All transforms are NumPy array operations or OpenCV C functions, both of which release the GIL and use native SIMD (SSE4.2 / AVX2).
- **Resize-before-apply.** Split-screen downscales each panel before the mode runs, reducing per-mode pixel count by up to 4× in the 2×2 grid layout.
- **Analysis caching.** `HyperdimensionalMatrixEffect` downscales to max 320 px and runs its full analysis pipeline at most once every 6 frames, caching results for the intermediate frames. Total overhead on a 1920×1080 source is < 2 ms per frame.
- **LUT precomputation.** Colour-mapping LUTs (thermal, night vision, UV) are built once at mode instantiation; per-frame cost is a single `cv2.LUT` call.
- **Per-overlay backpressure.** `_busy` flag per `OverlayWindow` drops frames the GUI thread cannot yet consume. Queue depth is capped at 1 per overlay; no memory growth under sustained load.
- **Typical throughput:** 30–60 FPS at 1920×1080 on a modern CPU.

---

## References

- Viénot F., Brettel H. & Mollon J.D. (1999). *Digital video colourmaps for checking the legibility of displays by dichromats.* Color Research & Application 24(4), 243–252.
- Gracheva E.O. et al. (2010). *Molecular basis of infrared detection by snakes.* Nature 464, 1006–1011.
- Lettvin J.Y. et al. (1959). *What the frog's eye tells the frog's brain.* Proc. IRE 47(11), 1940–1951.
- Mäthger L.M. et al. (2009). *Evidence for polarisation vision in the octopus.* J. Exp. Biol. 212, 2133–2140.

---

*Built with Python 3.10+ · PySide6 · OpenCV · NumPy · MSS · keyboard*

---

Development Note
This project was developed, polished, and refactored with the assistance of Artificial Intelligence.