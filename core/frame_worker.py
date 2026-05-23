"""
Frame Worker Thread — Phase 2

Single background capture loop that feeds raw frames to OverlayManager.
Each overlay applies its own vision mode independently; per-overlay
backpressure prevents Qt-event-queue build-up without needing a shared
_busy flag here.

Threading model
───────────────
  FrameWorker (QThread / background)
      capture()
          └─► OverlayManager.distribute(raw_frame)
                  └─► [per overlay] mode.apply → HUD → QImage → submit_frame()
                              └─► [QueuedConnection] OverlayWindow._on_frame()  [GUI thread]

Shutdown (two-phase, same pattern as Phase 1)
  request_stop()  —  non-blocking; sets _running=False only.
                     Safe to call from app.aboutToQuit (GUI thread).
  stop()          —  calls request_stop() then wait(3 s).
                     Call ONLY after app.exec() has returned.
"""

from __future__ import annotations

from PySide6.QtCore import QThread

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.engine import VisionEngine
    from core.overlay_manager import OverlayManager


class FrameWorker(QThread):
    """
    Background QThread: capture → OverlayManager.distribute().

    All mode processing, HUD rendering, and QImage conversion happen inside
    distribute(), keeping this class minimal and focused on the capture loop.
    """

    def __init__(self, engine: "VisionEngine", manager: "OverlayManager") -> None:
        """
        Args:
            engine:  Provides the screen-capture context.
            manager: Receives raw frames and distributes to overlays.
        """
        super().__init__()
        self._engine   = engine
        self._manager  = manager
        self._running: bool = False

    # ── QThread entry point ───────────────────────────────────────────────

    def run(self) -> None:
        """
        Capture-distribute loop.  Runs until request_stop() is called.

        Exceptions are caught per-frame so a single bad capture never
        terminates the thread.  A short sleep on error prevents a tight
        spin loop if the capture device is temporarily unavailable.
        """
        self._running = True

        while self._running:
            try:
                raw_frame = self._engine.capture.capture()
                if not self._running:   # check after slow capture before distribute
                    break
                self._manager.distribute(raw_frame)
            except Exception as exc:
                print(f"[FrameWorker] Frame error: {exc}")
                self.msleep(10)
                continue
            # Yield 1 ms so the Qt GUI event loop can process pending events
            # (paint, signals, queued slots) between capture cycles.
            # Without this, a fast mode can starve the dispatcher and delay
            # the processing of QEvent::Quit, extending the ESC-to-hide gap.
            self.msleep(1)

    # ── Lifecycle (two-phase shutdown) ────────────────────────────────────

    def request_stop(self) -> None:
        """
        Non-blocking stop request.  Safe to call from any thread.

        Sets _running=False so the loop exits on its next iteration.
        """
        self._running = False

    def stop(self) -> None:
        """
        Requests stop then blocks until the thread exits (max 3 s).

        Call this ONLY after app.exec() has returned.  Calling it from
        within the event loop (e.g., an aboutToQuit handler) will deadlock.
        """
        self.request_stop()
        if not self.wait(3000):
            print("[FrameWorker] Thread did not stop within timeout; terminating.")
            self.terminate()
