"""
Application Configuration

Central location for all constants and application-wide defaults.
Changing a value here propagates automatically to every module that
imports from this file — no hunting for magic numbers across the codebase.
"""

import cv2

# ── Display ──────────────────────────────────────────────────────────────────
WINDOW_NAME: str = "Vision Simulator"

# ── HUD style ─────────────────────────────────────────────────────────────────
HUD_FONT       = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.65
HUD_COLOR      = (210, 210, 210)   # Light grey in BGR
HUD_HEIGHT     = 55                # Pixel height of the bottom HUD bar

# ── Control keys (OpenCV key codes) ──────────────────────────────────────────
EXIT_KEY_CODE  = 27    # ESC
NEXT_KEY_CODE  = ord('n')
PREV_KEY_CODE  = ord('p')

# ── Engine defaults ───────────────────────────────────────────────────────────
DEFAULT_MODE_INDEX = 0

# ── Hotkey map: printable character → mode index (0-based) ───────────────────
# Keys '1'–'9' switch to modes 0–8; '0' switches to mode 9.
HOTKEYS: dict[str, int] = {str(i + 1): i for i in range(9)}
HOTKEYS['0'] = 9
