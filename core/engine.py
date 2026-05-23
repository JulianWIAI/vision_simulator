"""
Vision Engine — Phase 2

In Phase 2 the engine is a thin wrapper around ScreenCapture.
Mode management and frame processing moved to OverlayManager so that each
overlay can apply its own vision mode independently.

Kept as a separate class to serve as a future entry point for global
pre-processing hooks (e.g. resolution normalisation, HDR tone-mapping)
that should apply before any per-overlay mode runs.
"""

from core.screen_capture import ScreenCapture


class VisionEngine:
    """Owns the screen-capture context for the application lifetime."""

    def __init__(self) -> None:
        self.capture = ScreenCapture()
