"""
core/mouse_remap.py

Cross-platform split-screen click-coordinate remapper.

This module is the single import point for the rest of the application.
It selects the correct concrete remapper class at import time based on
sys.platform and re-exports it as SplitScreenMouseRemapper so that callers
(main.py) never need to know which platform they are running on.

Platform implementations
────────────────────────
  Windows → platform.win_mouse_remap.WindowsMouseRemapper
               Uses a Win32 WH_MOUSE_LL low-level mouse hook installed in a
               dedicated background thread with a GetMessageW pump loop.

  macOS   → platform.mac_mouse_remap.MacOSMouseRemapper
               Uses a Quartz CGEventTap installed in a CFRunLoop thread.
               Requires pyobjc-framework-Quartz and Accessibility permission
               (System Settings → Privacy & Security → Accessibility).

Shared coordinate maths
───────────────────────
The inverse-scale remapping logic (_map_coords) lives in
platform.base.AbstractMouseRemapper and is identical on both platforms.
Only the hook-installation / event-injection mechanism is OS-specific.

Usage (unchanged from before the refactor)
──────────────────────────────────────────
    from core.mouse_remap import SplitScreenMouseRemapper

    remapper = SplitScreenMouseRemapper(manager.split_screen)
    remapper.start()         # spawns background thread, installs hook/tap
    app.aboutToQuit.connect(remapper.stop)   # clean teardown on exit
"""

from platform_layer import get_platform

# Ask the platform factory for the correct remapper class, then expose it
# under the canonical name so all existing imports continue to work.
#
# get_platform().create_mouse_remapper() is a factory *method*, not a class,
# so we define a thin wrapper class that matches the original constructor
# signature:  SplitScreenMouseRemapper(split_screen)
#
# The wrapper simply calls the factory and stores the result; start() and
# stop() delegate to the underlying platform remapper.

class SplitScreenMouseRemapper:
    """
    Public facade for the platform-specific mouse click remapper.

    Wraps whichever AbstractMouseRemapper the platform layer provides so
    that callers can use the same constructor signature on every OS:

        remapper = SplitScreenMouseRemapper(manager.split_screen)
        remapper.start()
        remapper.stop()
    """

    def __init__(self, split_screen) -> None:
        """
        Args:
            split_screen: SplitScreenManager instance (read-only).  The
                          remapper reads .is_active and .layout_mode from it.
        """
        # Delegate object creation to the platform layer.  On Windows this
        # returns a WindowsMouseRemapper; on macOS a MacOSMouseRemapper.
        self._impl = get_platform().create_mouse_remapper(split_screen)

    def start(self) -> None:
        """Installs the OS hook/tap and starts the background thread."""
        self._impl.start()

    def stop(self) -> None:
        """Removes the OS hook/tap and waits for the thread to exit."""
        self._impl.stop()
