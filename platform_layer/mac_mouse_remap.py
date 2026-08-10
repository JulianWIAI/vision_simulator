"""
platform/mac_mouse_remap.py

macOS implementation of AbstractMouseRemapper.

Uses a CGEventTap (Quartz framework) to intercept mouse button events
system-wide and remap their coordinates during split-screen mode — the
macOS equivalent of Windows' WH_MOUSE_LL hook.

How it works
────────────
A CGEventTap is a Quartz Core Graphics callback that fires for every matching
input event before it is delivered to any application.  The callback receives
a mutable CGEventRef; we update the event's kCGMouseEventX / kCGMouseEventY
fields in-place so the remapped position reaches the target application without
any event suppression/injection cycle.  This is simpler than the Windows approach
because CGEventTap allows direct mutation of the original event rather than
requiring suppress + re-inject.

The tap runs in a CFRunLoop on a dedicated background thread.  The loop is
driven in short 100 ms slices so we can check a stop flag without blocking
the thread indefinitely.

Permissions
───────────
CGEventTap with kCGEventTapOptionDefault (mutating mode) requires that the
process has been granted Accessibility permission in:
  System Settings → Privacy & Security → Accessibility

Without this permission, CGEventTapCreate returns None and click remapping
is silently disabled (the rest of the app continues normally).

Coordinate system
─────────────────
CGEvent coordinates are in logical points on the primary display.  On a Retina
(HiDPI) display with backingScaleFactor = 2, logical points ≠ physical pixels.
MSS captures at physical pixels, so the panel dimensions stored in _screen_w /
_screen_h (physical) are converted to logical points before the remapping math
runs and then converted back so the mapped result can be written into the event.

Dependency
──────────
    pip install pyobjc-framework-Quartz
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

from platform_layer.base import AbstractMouseRemapper

# Quartz is required; if absent the remapper starts but does nothing.
try:
    import Quartz
    _QUARTZ_AVAILABLE = True
except ImportError:
    _QUARTZ_AVAILABLE = False
    print(
        "[macOS] pyobjc-framework-Quartz not found — click remapping is disabled.\n"
        "Install with:  pip install pyobjc-framework-Quartz"
    )


class MacOSMouseRemapper(AbstractMouseRemapper):
    """
    CGEventTap-based split-screen click remapper for macOS.

    Lifecycle mirrors the Windows implementation:
        remapper = MacOSMouseRemapper(split_screen, screen_w, screen_h)
        remapper.start()    # spawns thread, installs tap
        ...
        remapper.stop()     # disables tap, joins thread
    """

    def __init__(self, split_screen, screen_w: int, screen_h: int) -> None:
        """
        Args:
            split_screen: Shared SplitScreenManager (read-only).
            screen_w:     Primary screen width  in physical pixels.
            screen_h:     Primary screen height in physical pixels.
        """
        super().__init__(split_screen, screen_w, screen_h)

        self._tap:    Optional[object] = None   # CFMachPort (the event tap handle)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Determine the backing scale factor so we can convert between
        # physical pixels (MSS frame dimensions) and logical points (CGEvent).
        # Computed once at construction; does not change while the app runs.
        self._scale: float = self._query_scale_factor()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawns the CFRunLoop thread and installs the CGEventTap."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop_thread,
            name="MacMouseRemapTap",
            daemon=True,   # killed automatically when the main thread exits
        )
        self._thread.start()
        print("[MacOSMouseRemapper] CGEventTap thread started — click remapping active")

    def stop(self) -> None:
        """Disables the CGEventTap and waits for the thread to exit cleanly."""
        self._stop_event.set()

        # Disable the tap so the CFRunLoop iteration in the thread wakes up
        # and the stop flag is noticed on the next slice boundary.
        if self._tap is not None and _QUARTZ_AVAILABLE:
            try:
                Quartz.CGEventTapEnable(self._tap, False)
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        print("[MacOSMouseRemapper] CGEventTap thread stopped")

    # ── CFRunLoop thread ────────────────────────────────────────────────────

    def _run_loop_thread(self) -> None:
        """
        Creates the CGEventTap and drives a CFRunLoop in 100 ms slices.

        The CFRunLoop must run on the same thread that created the tap.
        We use CFRunLoopRunInMode with a short timeout instead of CFRunLoopRun
        so we can check _stop_event without being blocked indefinitely.
        """
        if not _QUARTZ_AVAILABLE:
            return   # no-op: gracefully disabled at import time

        # Build the event mask — only intercept button press/release events.
        # Mouse-move and scroll events are not needed and adding them would
        # increase per-event overhead unnecessarily.
        mask = (
            Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp)
            | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseUp)
            | Quartz.CGEventMaskBit(Quartz.kCGEventOtherMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventOtherMouseUp)
        )

        # kCGEventTapOptionDefault = mutating mode: we can change the event.
        # kCGSessionEventTap       = system-wide (all applications).
        # kCGHeadInsertEventTap    = inserted at the head of the tap chain.
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            mask,
            self._event_callback,   # Python callable used as C callback
            None,                   # userInfo — not used; state lives on self
        )

        if self._tap is None:
            # Most likely cause: Accessibility permission not granted.
            print(
                "[MacOSMouseRemapper] CGEventTapCreate returned None.\n"
                "  → Grant Accessibility permission in System Settings → Privacy & Security."
            )
            return

        # Attach the tap to the current thread's CFRunLoop.
        run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        current_loop    = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(current_loop, run_loop_source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)

        # Drive the run loop in 100 ms slices so we can exit when stop() is called.
        while not self._stop_event.is_set():
            # kCFRunLoopDefaultMode processes events from the tap.
            # The 0.1 s timeout returns control to Python so we can check
            # the stop flag without blocking until the next mouse event.
            Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.1, False)

        # Disable and detach cleanly.
        Quartz.CGEventTapEnable(self._tap, False)
        Quartz.CFRunLoopRemoveSource(current_loop, run_loop_source, Quartz.kCFRunLoopCommonModes)

    # ── CGEventTap callback ─────────────────────────────────────────────────

    def _event_callback(self, proxy, event_type, event, refcon):
        """
        CGEventTap callback — fires synchronously for every matching event.

        Unlike the Windows approach (suppress + re-inject), on macOS we mutate
        the CGEventRef's location field directly.  The modified event then
        proceeds normally to the target application with the corrected position.

        Args:
            proxy:      CGEventTapProxy — used to pass event to next tap if needed.
            event_type: CGEventType integer (kCGEventLeftMouseDown, etc.).
            event:      Mutable CGEventRef.
            refcon:     User data pointer (None here; state is on self).

        Returns:
            The (possibly mutated) event, or None to suppress it.
        """
        # Skip remapping when split-screen is not active.
        if not self._split_screen.is_active:
            return event

        # Read the event's current location in logical points.
        loc = Quartz.CGEventGetLocation(event)
        cx_pts = loc.x   # logical x in points
        cy_pts = loc.y   # logical y in points

        # Convert logical points → physical pixels for the shared _map_coords math.
        # _map_coords was designed around physical pixel dimensions (matching MSS).
        cx_px = int(cx_pts * self._scale)
        cy_px = int(cy_pts * self._scale)

        # Apply the inverse panel scale (shared base-class method).
        mapped_x_px, mapped_y_px = self._map_coords(cx_px, cy_px)

        # If coordinates did not change, return the event unmodified.
        if mapped_x_px == cx_px and mapped_y_px == cy_px:
            return event

        # Convert mapped physical pixels back to logical points for the event.
        mapped_x_pts = mapped_x_px / self._scale
        mapped_y_pts = mapped_y_px / self._scale

        # Write the new position directly into the CGEventRef.
        new_loc = Quartz.CGPoint(mapped_x_pts, mapped_y_pts)
        Quartz.CGEventSetLocation(event, new_loc)

        return event   # return the mutated event so it reaches the application

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _query_scale_factor() -> float:
        """
        Returns the primary screen's backing scale factor (e.g. 2.0 on Retina).

        Uses Qt so no extra dependency is needed for this query.  The scale
        factor is used to convert between logical points (CGEvent) and physical
        pixels (MSS capture frame).
        """
        try:
            from PySide6.QtWidgets import QApplication
            screen = QApplication.primaryScreen()
            if screen is not None:
                return float(screen.devicePixelRatio())
        except Exception:
            pass
        return 1.0   # safe fallback: no HiDPI scaling assumed
