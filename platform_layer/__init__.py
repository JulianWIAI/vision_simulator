"""
platform/__init__.py

Factory that returns the correct AbstractPlatform implementation for the
current operating system.  All other modules import get_platform() from
here rather than importing a concrete platform class directly, so the
rest of the codebase stays OS-agnostic.

Usage
─────
    from platform_layer import get_platform
    plat = get_platform()
    plat.apply_overlay_styles(my_widget)

Supported platforms
───────────────────
  win32  → WindowsPlatform  (core/overlay_window, ctypes/Win32)
  darwin → MacOSPlatform    (AppKit/Quartz via pyobjc)

Any other sys.platform value raises RuntimeError at startup rather than
failing silently at the first platform call.
"""

import sys
from typing import TYPE_CHECKING

# Lazy singleton: the platform object is created once on first call and
# cached here.  This avoids importing ctypes or AppKit at module level
# for every file that does `from platform_layer import get_platform`.
_platform_instance = None


def get_platform():
    """
    Returns the singleton AbstractPlatform for the current OS.

    Thread-safety: this is called at startup from the main thread before
    any worker threads exist, so no lock is needed around the singleton.

    Returns:
        AbstractPlatform subclass instance for the current OS.

    Raises:
        RuntimeError: if the current OS is not Windows or macOS.
    """
    global _platform_instance

    # Return the cached instance if it was already created.
    if _platform_instance is not None:
        return _platform_instance

    if sys.platform == "win32":
        # Import here so the Windows-specific ctypes code is never loaded
        # when running on macOS.
        from platform_layer.windows import WindowsPlatform
        _platform_instance = WindowsPlatform()

    elif sys.platform == "darwin":
        # Import here so pyobjc imports are never attempted on Windows.
        from platform_layer.macos import MacOSPlatform
        _platform_instance = MacOSPlatform()

    else:
        raise RuntimeError(
            f"Vision Simulator: unsupported platform '{sys.platform}'. "
            "Only Windows (win32) and macOS (darwin) are supported."
        )

    return _platform_instance
