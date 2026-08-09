"""
utils/window_manager.py

Enumerates and caches visible top-level windows for the window-tracked
overlay feature.

Cross-platform design
─────────────────────
All OS-specific enumeration and geometry queries are delegated to the
platform abstraction layer (platform/):

  Windows → platform.windows.WindowsPlatform
              Uses EnumWindows, GetWindowRect, IsWindow, IsWindowVisible,
              IsIconic — all via ctypes/Win32.

  macOS   → platform.macos.MacOSPlatform
              Uses Quartz.CGWindowListCopyWindowInfo for enumeration and
              per-window rect queries.

Data contract (unchanged from the original implementation)
─────────────────────────────────────────────────────────
Every window entry is a plain dict:
    {"id": int, "title": str, "rect": (x1, y1, x2, y2)}

"id" is the platform's native window handle (HWND on Windows, CGWindowID
on macOS).  Rect is in screen/desktop coordinates.

Thread safety
─────────────
A threading.Lock protects the cached list and refresh timestamp.
get_rect() / is_valid() / is_minimized() issue live platform queries with
no shared state, so they do not need the lock.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

from platform import get_platform


# How long (seconds) the cached window list is considered fresh.
# After this interval, the next get_windows() call re-enumerates.
_REFRESH_INTERVAL: float = 2.0


class WindowManager:
    """
    Enumerates and caches visible, titled top-level windows.

    All actual OS calls go through get_platform(), which returns the
    correct AbstractPlatform implementation for the current OS.
    """

    def __init__(self) -> None:
        self._lock         = threading.Lock()
        self._windows: List[dict] = []
        self._last_refresh: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def get_windows(self, force: bool = False) -> List[dict]:
        """
        Returns the cached window list, re-fetching if REFRESH_INTERVAL has elapsed.

        Args:
            force: Bypass the timestamp check and enumerate immediately.

        Returns:
            Shallow copy of the cached list so callers cannot mutate shared state.
        """
        now = time.monotonic()
        with self._lock:
            if force or (now - self._last_refresh) >= _REFRESH_INTERVAL:
                # Delegate enumeration to the platform layer.
                self._windows      = get_platform().list_windows()
                self._last_refresh = now
            return list(self._windows)   # return a copy, not the live list

    def get_rect(self, window_id: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns the current on-screen rect of a specific window — live, not cached.

        Suitable for per-100 ms tracking of position shifts.  Does not acquire
        the internal lock because no shared state is involved.

        Args:
            window_id: The "id" value from a get_windows() entry.

        Returns:
            (x1, y1, x2, y2) in screen coordinates, or None if stale / invalid.
        """
        return get_platform().get_window_rect(window_id)

    def is_valid(self, window_id: int) -> bool:
        """Returns True if the window still exists and is visible."""
        return get_platform().is_window_valid(window_id)

    def is_minimized(self, window_id: int) -> bool:
        """Returns True if the window is currently minimized / iconified."""
        return get_platform().is_window_minimized(window_id)
