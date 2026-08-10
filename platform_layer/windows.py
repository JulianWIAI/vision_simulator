"""
platform/windows.py

Windows implementation of AbstractPlatform.

All Win32 API calls that were previously scattered across overlay_window.py,
region_drawer.py, and window_manager.py are now consolidated here.  The rest
of the application imports get_platform() and calls abstract methods — it never
imports anything from this file directly.

Win32 extended-style constants
──────────────────────────────
WS_EX_LAYERED    — Required companion to WS_EX_TRANSPARENT; enables the
                   layered-window compositing path.
WS_EX_TRANSPARENT — Causes the OS to answer WM_NCHITTEST with HTTRANSPARENT
                    before Qt even sees the message, making the window fully
                    click-through at the kernel level.
WS_EX_TOOLWINDOW — Marks the window as a tool window; Windows removes it from
                   the taskbar button list and the Alt-Tab switcher.
WS_EX_NOACTIVATE — Prevents the window from stealing keyboard focus or being
                   activated when the user clicks near it.
WS_EX_APPWINDOW  — The OPPOSITE of what we want: forces a taskbar button even
                   when WS_EX_TOOLWINDOW is set.  We explicitly CLEAR this bit.
WDA_EXCLUDEFROMCAP — Hides the window from all BitBlt / MSS / OBS capture
                     APIs.  Requires Windows 10 2004 (build 19041) or later.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
from typing import List, Optional, Tuple, TYPE_CHECKING

from platform_layer.base import AbstractPlatform, AbstractMouseRemapper

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


# ── Win32 extended-style constants ─────────────────────────────────────────────

_GWL_EXSTYLE        = -20             # index to pass GetWindowLongW / SetWindowLongW
_WS_EX_LAYERED      = 0x00080000     # required companion to WS_EX_TRANSPARENT
_WS_EX_TRANSPARENT  = 0x00000020     # OS answers NCHITTEST with HTTRANSPARENT
_WS_EX_TOOLWINDOW   = 0x00000080     # removes window from taskbar and Alt-Tab
_WS_EX_NOACTIVATE   = 0x08000000     # window cannot be activated by mouse/keyboard
_WS_EX_APPWINDOW    = 0x00040000     # forces taskbar button — we must CLEAR this
_WDA_EXCLUDEFROMCAP = 0x00000011     # hide from BitBlt / MSS / screen-recording APIs

# SetWindowPos flags — combine to flush style changes without moving/resizing.
_SWP_NOMOVE       = 0x0002
_SWP_NOSIZE       = 0x0001
_SWP_NOZORDER     = 0x0004
_SWP_FRAMECHANGED = 0x0020           # tells Win32 to re-send WM_NCCALCSIZE immediately

# GetSystemMetrics indices for the primary screen's physical pixel dimensions.
_SM_CXSCREEN = 0    # primary screen width  in physical pixels
_SM_CYSCREEN = 1    # primary screen height in physical pixels

# EnumWindows callback type: bool CALLBACK(HWND hwnd, LPARAM lParam)
_EnumWindowsProc = ctypes.WINFUNCTYPE(
    ctypes.c_bool,
    ctypes.wintypes.HWND,
    ctypes.wintypes.LPARAM,
)


# ── WindowsPlatform ─────────────────────────────────────────────────────────────

class WindowsPlatform(AbstractPlatform):
    """
    Windows implementation of AbstractPlatform.

    Uses ctypes to call Win32 APIs directly — no extra Python packages are
    required beyond the standard library.
    """

    # ── Window-style methods ───────────────────────────────────────────────

    def apply_overlay_styles(self, widget: "QWidget") -> None:
        """
        Sets Win32 extended styles on an overlay window to make it:
          • Click-through (WS_EX_TRANSPARENT)
          • Always composited correctly (WS_EX_LAYERED)
          • Hidden from taskbar and Alt-Tab (WS_EX_TOOLWINDOW, clears WS_EX_APPWINDOW)
          • Unable to steal focus (WS_EX_NOACTIVATE)

        SetWindowPos with SWP_FRAMECHANGED flushes all style changes
        synchronously without moving, resizing, or reordering the window.

        Errors are caught and printed non-fatally — the overlay still works
        without the Win32 hardening; click-through may be imperfect.
        """
        try:
            hwnd = int(widget.winId())   # integer HWND from the Qt widget

            # Read the current extended-style bitmask.
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)

            # OR in the bits we want set.
            new_style = (
                ex_style
                | _WS_EX_LAYERED      # required companion for WS_EX_TRANSPARENT
                | _WS_EX_TRANSPARENT  # click-through at the OS level
                | _WS_EX_TOOLWINDOW   # hide from taskbar / Alt-Tab
                | _WS_EX_NOACTIVATE   # prevent focus theft
            )

            # AND-NOT removes WS_EX_APPWINDOW, which forces a taskbar button
            # even when WS_EX_TOOLWINDOW is set.  Qt can accidentally set this
            # on some Windows versions when the window has no owner.
            new_style &= ~_WS_EX_APPWINDOW

            # Write the combined style back.
            ctypes.windll.user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, new_style)

            # Flush the change so it takes effect for the very next NCHITTEST.
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
            )

        except Exception as exc:
            print(f"[WindowsPlatform] apply_overlay_styles error: {exc}")

    def exclude_from_capture(self, widget: "QWidget") -> None:
        """
        Calls SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAP) to hide the window
        from MSS, BitBlt, OBS, and all other screen-capture APIs.

        Requires Windows 10 2004 (build 19041) or later.  Fails silently on
        older builds so the app still runs, just with potential feedback loops.
        """
        try:
            hwnd = int(widget.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, _WDA_EXCLUDEFROMCAP)
        except Exception:
            pass   # silently ignore — older Windows versions lack this API

    def apply_drawer_styles(self, widget: "QWidget") -> None:
        """
        Sets Win32 extended styles on the region-drawing canvas window.

        WS_EX_TRANSPARENT is intentionally NOT applied here — unlike the
        overlay windows, the drawer must receive all mouse events to track
        the drag gesture.

        WS_EX_LAYERED is still set so the translucent background renders
        correctly via the layered-window compositing path.
        """
        try:
            hwnd = int(widget.winId())
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)

            # Add: layered (for translucency), toolwindow, noactivate.
            # Do NOT add WS_EX_TRANSPARENT — the drawer needs mouse events.
            new_style = (
                ex_style
                | _WS_EX_LAYERED
                | _WS_EX_TOOLWINDOW
                | _WS_EX_NOACTIVATE
            ) & ~_WS_EX_APPWINDOW   # clear forced-taskbar-button flag

            ctypes.windll.user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, new_style)
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
            )

        except Exception as exc:
            print(f"[WindowsPlatform] apply_drawer_styles error: {exc}")

    # ── Screen geometry ────────────────────────────────────────────────────

    def get_screen_size(self) -> Tuple[int, int]:
        """
        Returns the primary screen's physical pixel dimensions via
        GetSystemMetrics.  This matches the frame dimensions produced by MSS
        because both operate at physical (not DPI-scaled) resolution.
        """
        w = ctypes.windll.user32.GetSystemMetrics(_SM_CXSCREEN)
        h = ctypes.windll.user32.GetSystemMetrics(_SM_CYSCREEN)
        return (w, h)

    # ── Window enumeration ─────────────────────────────────────────────────

    def list_windows(self) -> List[dict]:
        """
        Synchronously enumerates all visible, titled, non-zero-area top-level
        windows using EnumWindows.

        Filters out:
          • Invisible windows (IsWindowVisible == False)
          • Windows with an empty title (GetWindowTextLengthW == 0)
          • Zero-area rects (minimised or off-screen pseudo-windows)

        Returns:
            List of {"id": HWND, "title": str, "rect": (x1,y1,x2,y2)} dicts.
        """
        results: List[dict] = []

        def _callback(hwnd: int, _: int) -> bool:
            # Skip windows that are not visible on screen.
            if not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True   # returning True continues enumeration

            # Skip windows with no title text.
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True

            # Read the window title into a Unicode buffer.
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
            if not title:
                return True

            # Query the window's screen rect.
            rect = ctypes.wintypes.RECT()
            if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True

            # Skip windows with zero or negative area (e.g. minimised ghosts).
            if (rect.right - rect.left) <= 0 or (rect.bottom - rect.top) <= 0:
                return True

            results.append({
                "id":    hwnd,
                "title": title,
                "rect":  (rect.left, rect.top, rect.right, rect.bottom),
            })
            return True   # always return True to continue enumeration

        # Wrap the Python callback in the correct Win32 function-pointer type
        # and hold a reference to prevent garbage collection during the call.
        proc = _EnumWindowsProc(_callback)
        ctypes.windll.user32.EnumWindows(proc, 0)
        return results

    def get_window_rect(self, window_id: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns the live screen rect of a window via GetWindowRect.

        This is a direct OS query, not cached, so it reflects the window's
        current position even if it was moved after the last list_windows() call.
        """
        rect = ctypes.wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(window_id, ctypes.byref(rect)):
            return None
        return (rect.left, rect.top, rect.right, rect.bottom)

    def is_window_valid(self, window_id: int) -> bool:
        """Returns True if the HWND still names a visible window."""
        return bool(
            ctypes.windll.user32.IsWindow(window_id)
            and ctypes.windll.user32.IsWindowVisible(window_id)
        )

    def is_window_minimized(self, window_id: int) -> bool:
        """Returns True if the window is currently minimized (iconic)."""
        return bool(ctypes.windll.user32.IsIconic(window_id))

    # ── Mouse remapper factory ─────────────────────────────────────────────

    def create_mouse_remapper(self, split_screen) -> AbstractMouseRemapper:
        """
        Creates a WindowsMouseRemapper backed by a WH_MOUSE_LL hook.

        The screen size is queried here (once) and passed to the remapper so
        the constructor does not need to call Win32 APIs itself.
        """
        from platform_layer.win_mouse_remap import WindowsMouseRemapper
        screen_w, screen_h = self.get_screen_size()
        return WindowsMouseRemapper(split_screen, screen_w, screen_h)
