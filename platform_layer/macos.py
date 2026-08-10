"""
platform/macos.py

macOS implementation of AbstractPlatform.

Most overlay-window hardening is handled by Qt window flags on macOS — the
platform layer only needs to add capture exclusion (NSWindow.setSharingType_)
and window enumeration (CGWindowListCopyWindowInfo) on top.

Dependencies
────────────
  pyobjc-framework-Quartz  — CGWindowListCopyWindowInfo, CGEventTap (mouse remap)
  pyobjc-framework-Cocoa   — NSWindowSharingNone for capture exclusion

Install:
    pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa

Both packages are optional at import time; missing packages degrade gracefully
with a printed warning rather than a startup crash.

Coordinate system note
──────────────────────
macOS uses "logical points" for window coordinates (Quartz / AppKit), not
physical pixels.  On a Retina (HiDPI) display the physical pixel count is
screen_points × backingScaleFactor (typically 2×).  MSS captures at physical
pixels, so the mouse remapper must account for the scale factor when mapping
between CGEvent coordinates (points) and the capture frame (pixels).
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import List, Optional, Tuple, TYPE_CHECKING

from platform_layer.base import AbstractPlatform, AbstractMouseRemapper

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

# ── Optional pyobjc imports ────────────────────────────────────────────────────

# Quartz is used for CGWindowListCopyWindowInfo (window enumeration) and
# CGEventTap (mouse remapping).  We import it lazily and record availability.
try:
    import Quartz                          # pyobjc-framework-Quartz
    _QUARTZ_AVAILABLE = True
except ImportError:
    _QUARTZ_AVAILABLE = False
    print(
        "[macOS] pyobjc-framework-Quartz not found — window enumeration and "
        "click remapping will be unavailable.  Run:  pip install pyobjc-framework-Quartz"
    )

# NSWindowSharingNone (= 0) is from AppKit and used for capture exclusion.
try:
    from AppKit import NSWindowSharingNone   # pyobjc-framework-Cocoa
    _APPKIT_AVAILABLE = True
except ImportError:
    NSWindowSharingNone = 0                  # safe fallback: same numeric value
    _APPKIT_AVAILABLE = False
    print(
        "[macOS] pyobjc-framework-Cocoa not found — capture exclusion may be "
        "limited.  Run:  pip install pyobjc-framework-Cocoa"
    )


# ── ObjC runtime helpers (ctypes, no pyobjc required) ─────────────────────────

def _get_nswindow_ptr(widget: "QWidget") -> Optional[int]:
    """
    Retrieves the raw NSWindow pointer for a PySide6 QWidget.

    PySide6's winId() on macOS returns a pointer to the native NSView.
    We send the -window message (ObjC selector) to the NSView to get the
    NSWindow that contains it.

    Uses the Objective-C runtime directly via ctypes so that pyobjc is not
    a hard requirement for this helper.

    Args:
        widget: Any QWidget whose native NSWindow is needed.

    Returns:
        Raw NSWindow pointer as an integer, or None on failure.
    """
    try:
        libobjc_path = ctypes.util.find_library("objc")
        if not libobjc_path:
            return None
        libobjc = ctypes.cdll.LoadLibrary(libobjc_path)

        # sel_registerName returns the SEL (selector) for "-window".
        sel_window = libobjc.sel_registerName(b"window")

        # objc_msgSend dispatches an ObjC message.  We configure argtypes
        # so ctypes handles pointer sizes correctly on 64-bit macOS.
        libobjc.objc_msgSend.restype  = ctypes.c_void_p
        libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        ns_view_ptr = ctypes.c_void_p(int(widget.winId()))  # NSView pointer from Qt
        ns_win_ptr  = libobjc.objc_msgSend(ns_view_ptr, sel_window)  # → NSWindow*
        return int(ns_win_ptr) if ns_win_ptr else None

    except Exception as exc:
        print(f"[macOS] _get_nswindow_ptr failed: {exc}")
        return None


def _call_nswindow_set_sharing(ns_win_ptr: int, sharing_type: int) -> None:
    """
    Calls [NSWindow setSharingType: sharing_type] via the ObjC runtime.

    NSWindowSharingNone (0) tells the OS not to share this window's content
    with screen-capture APIs, effectively hiding it from MSS / screenshot tools.

    Args:
        ns_win_ptr:   Raw NSWindow pointer (integer).
        sharing_type: NSWindowSharingNone (0) to exclude, NSWindowSharingReadOnly
                      (1) to allow read-only sharing.
    """
    try:
        libobjc_path = ctypes.util.find_library("objc")
        if not libobjc_path:
            return
        libobjc = ctypes.cdll.LoadLibrary(libobjc_path)

        # Selector for the -setSharingType: method.
        sel = libobjc.sel_registerName(b"setSharingType:")

        # objc_msgSend with one NSUInteger argument.
        libobjc.objc_msgSend.restype  = None
        libobjc.objc_msgSend.argtypes = [
            ctypes.c_void_p,   # id (NSWindow*)
            ctypes.c_void_p,   # SEL
            ctypes.c_ulong,    # NSUInteger sharing type
        ]
        libobjc.objc_msgSend(
            ctypes.c_void_p(ns_win_ptr),
            sel,
            ctypes.c_ulong(sharing_type),
        )

    except Exception as exc:
        print(f"[macOS] setSharingType failed: {exc}")


# ── MacOSPlatform ──────────────────────────────────────────────────────────────

class MacOSPlatform(AbstractPlatform):
    """
    macOS implementation of AbstractPlatform.

    Window-style hardening is mostly handled by Qt flags set in the calling
    widget's _setup_window(); this class adds the two things Qt cannot do:
      1. Capture exclusion via NSWindow.setSharingType_(NSWindowSharingNone).
      2. Window enumeration via Quartz.CGWindowListCopyWindowInfo.

    On macOS, Qt.WindowTransparentForInput maps to NSWindow.setIgnoresMouseEvents_(YES),
    Qt.Tool maps to NSWindowStyleMaskUtilityWindow (hides from Dock / Mission Control),
    and Qt.WindowStaysOnTopHint sets the window level above normal windows.
    These are sufficient for the overlay use-case without any extra ObjC calls.
    """

    # ── Window-style methods ───────────────────────────────────────────────

    def apply_overlay_styles(self, widget: "QWidget") -> None:
        """
        On macOS, Qt flags set in _setup_window() already handle:
          • Click-through      → Qt.WindowTransparentForInput
          • Always-on-top      → Qt.WindowStaysOnTopHint
          • Hide from Dock     → Qt.Tool (NSWindowStyleMaskUtilityWindow)
          • No focus steal     → Qt.WindowTransparentForInput also sets ignoresMouseEvents

        This method therefore only calls exclude_from_capture() to hide the
        window from screenshot / screen-recording APIs — the one thing Qt
        cannot configure through its public API.
        """
        self.exclude_from_capture(widget)

    def exclude_from_capture(self, widget: "QWidget") -> None:
        """
        Calls NSWindow.setSharingType_(NSWindowSharingNone) to hide the overlay
        from macOS screen-capture APIs (screenshot, screen recording, MSS).

        NSWindowSharingNone = 0 prevents the window's backing store from being
        shared with any capture consumer.  This is the macOS equivalent of
        Windows' SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAP).
        """
        ns_win = _get_nswindow_ptr(widget)
        if ns_win:
            _call_nswindow_set_sharing(ns_win, NSWindowSharingNone)  # 0 = exclude

    def apply_drawer_styles(self, widget: "QWidget") -> None:
        """
        On macOS the region-drawer window uses the same Qt flags as the overlay
        (Tool + WindowStaysOnTopHint + FramelessWindowHint) but WITHOUT
        WindowTransparentForInput, so it receives mouse drag events.

        Qt flags alone are sufficient; no extra ObjC calls are needed here.
        We still call exclude_from_capture so the draw canvas does not feed
        back into MSS frames while the user is dragging.
        """
        self.exclude_from_capture(widget)

    # ── Screen geometry ────────────────────────────────────────────────────

    def get_screen_size(self) -> Tuple[int, int]:
        """
        Returns the primary screen's physical pixel dimensions.

        Uses Qt's primaryScreen() to stay dependency-free on this path.
        devicePixelRatio() converts logical points to physical pixels on
        Retina / HiDPI displays, matching the dimensions MSS captures.
        """
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        ratio  = screen.devicePixelRatio()
        # Multiply logical size by backing scale factor for physical pixels.
        w = int(screen.geometry().width()  * ratio)
        h = int(screen.geometry().height() * ratio)
        return (w, h)

    # ── Window enumeration ─────────────────────────────────────────────────

    def list_windows(self) -> List[dict]:
        """
        Enumerates visible on-screen windows via CGWindowListCopyWindowInfo.

        Requires pyobjc-framework-Quartz.  Returns an empty list if the
        package is not installed, gracefully degrading without a crash.

        CGWindowListCopyWindowInfo returns each window's metadata as a dict:
          kCGWindowName        → window title (may be None for untitled windows)
          kCGWindowOwnerName   → app name (fallback when title is None)
          kCGWindowBounds      → {"X", "Y", "Width", "Height"} in logical points
          kCGWindowNumber      → unique window ID (stable until window closes)
          kCGWindowIsOnscreen  → True if not minimized / hidden
        """
        if not _QUARTZ_AVAILABLE:
            return []   # graceful degradation — window-tracking disabled

        # Request all on-screen windows, excluding the desktop wallpaper layer.
        options = (
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements
        )
        raw_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
        if not raw_list:
            return []

        results: List[dict] = []
        for entry in raw_list:
            # Prefer the window's own title; fall back to the owner app name.
            title = (
                entry.get("kCGWindowName")
                or entry.get("kCGWindowOwnerName")
                or ""
            )
            if not title:
                continue   # skip windows with no meaningful name

            bounds = entry.get("kCGWindowBounds") or {}
            x = int(bounds.get("X",      0))
            y = int(bounds.get("Y",      0))
            w = int(bounds.get("Width",  0))
            h = int(bounds.get("Height", 0))

            if w <= 0 or h <= 0:
                continue   # skip zero-area / minimized windows

            # kCGWindowNumber is stable and usable as a platform window ID.
            win_id = entry.get("kCGWindowNumber", 0)

            results.append({
                "id":    win_id,
                "title": title,
                "rect":  (x, y, x + w, y + h),   # convert bounds → (x1,y1,x2,y2)
            })

        return results

    def get_window_rect(self, window_id: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns the live rect of a single window by its CGWindowID.

        Uses kCGWindowListOptionIncludingWindow to query only that specific
        window without re-enumerating the full list.
        """
        if not _QUARTZ_AVAILABLE:
            return None

        raw = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionIncludingWindow,
            window_id,
        )
        if not raw:
            return None

        bounds = raw[0].get("kCGWindowBounds") or {}
        x = int(bounds.get("X",      0))
        y = int(bounds.get("Y",      0))
        w = int(bounds.get("Width",  0))
        h = int(bounds.get("Height", 0))
        return (x, y, x + w, y + h) if (w > 0 and h > 0) else None

    def is_window_valid(self, window_id: int) -> bool:
        """Returns True if the window still exists and is visible on screen."""
        if not _QUARTZ_AVAILABLE:
            return False
        raw = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionIncludingWindow | Quartz.kCGWindowListOptionOnScreenOnly,
            window_id,
        )
        return bool(raw)   # non-empty list means the window is alive and visible

    def is_window_minimized(self, window_id: int) -> bool:
        """
        Returns True if the window is not currently on screen (minimized or hidden).

        A window that is off-screen will not appear in a kCGWindowListOptionOnScreenOnly
        query; we use that absence as the "minimized" signal.
        """
        return not self.is_window_valid(window_id)

    # ── Mouse remapper factory ─────────────────────────────────────────────

    def create_mouse_remapper(self, split_screen) -> AbstractMouseRemapper:
        """
        Creates a MacOSMouseRemapper backed by a CGEventTap.

        Requires pyobjc-framework-Quartz and Accessibility permission granted
        in System Preferences → Privacy & Security → Accessibility.
        """
        from platform_layer.mac_mouse_remap import MacOSMouseRemapper
        screen_w, screen_h = self.get_screen_size()
        return MacOSMouseRemapper(split_screen, screen_w, screen_h)
