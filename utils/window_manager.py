"""
Window Manager — Phase 4

Enumerates and caches visible Win32 windows using ctypes (no extra deps).
The full window list is refreshed at most every REFRESH_INTERVAL seconds,
never on every frame, satisfying the per-frame performance requirement.

Data contract (public):
    {"id": int, "title": str, "rect": (x1, y1, x2, y2)}

where rect is in desktop/screen coordinates as returned by GetWindowRect.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import threading
import time
from typing import List, Optional, Tuple


_REFRESH_INTERVAL: float = 2.0  # seconds between full-list refreshes

_EnumWindowsProc = ctypes.WINFUNCTYPE(
    ctypes.c_bool,
    ctypes.wintypes.HWND,
    ctypes.wintypes.LPARAM,
)


class WindowManager:
    """
    Enumerates and caches visible, titled top-level Win32 windows.

    Thread-safe: a single threading.Lock guards both the cached list and
    the last-refresh timestamp.  get_rect() issues a live Win32 call that
    touches no shared state and therefore needs no lock.

    All returned window entries are plain dicts:
        {"id": int, "title": str, "rect": (x1, y1, x2, y2)}
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: List[dict] = []
        self._last_refresh: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────

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
                self._windows = _enumerate_windows()
                self._last_refresh = now
            return list(self._windows)

    def get_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns the current on-screen rect of a specific window — live, not cached.

        Suitable for per-100ms tracking of position shifts.
        Does not acquire the internal lock (no shared state involved).

        Args:
            hwnd: Native HWND stored in a window entry's "id" field.

        Returns:
            (x1, y1, x2, y2) in desktop coordinates, or None if the HWND
            is stale / no longer valid.
        """
        rect = ctypes.wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return (rect.left, rect.top, rect.right, rect.bottom)

    def is_valid(self, hwnd: int) -> bool:
        """Returns True if the HWND still exists and is visible."""
        return bool(
            ctypes.windll.user32.IsWindow(hwnd)
            and ctypes.windll.user32.IsWindowVisible(hwnd)
        )

    def is_minimized(self, hwnd: int) -> bool:
        """Returns True if the window is currently minimized (iconic)."""
        return bool(ctypes.windll.user32.IsIconic(hwnd))


# ── Private enumeration ────────────────────────────────────────────────────────

def _enumerate_windows() -> List[dict]:
    """
    Synchronously enumerates all visible, titled, non-zero-area top-level windows.

    Called at most every REFRESH_INTERVAL seconds; never per-frame.
    Filters out: invisible windows, untitled windows, zero-area rects
    (minimised or off-screen pseudo-windows).
    """
    results: List[dict] = []

    def _callback(hwnd: int, _: int) -> bool:
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True

        length: int = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if not title:
            return True

        rect = ctypes.wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True

        if (rect.right - rect.left) <= 0 or (rect.bottom - rect.top) <= 0:
            return True

        results.append({
            "id":    hwnd,
            "title": title,
            "rect":  (rect.left, rect.top, rect.right, rect.bottom),
        })
        return True

    proc = _EnumWindowsProc(_callback)
    ctypes.windll.user32.EnumWindows(proc, 0)
    return results
