"""
Vision Simulator — Entry Point

Shutdown sequence (critical)
─────────────────────────────
app.aboutToQuit (GUI thread):
  1. remapper.stop()        — removes the OS mouse hook; joins pump thread.
  2. manager.hide_all()     — all overlay windows disappear immediately.
  3. worker.request_stop()  — sets _running=False (non-blocking flag only).
                              Do NOT connect worker.stop() here — stop() calls
                              QThread.wait(3000) which blocks the main thread
                              while Qt is still dispatching queued signals → deadlock.
  4. panel.force_close()    — allows the control panel to close cleanly.
app.exec() returns:
  5. worker.wait(1000) / terminate() — join the capture thread.

Architecture
────────────
  QApplication (main thread / Qt event loop)
    ├── OverlayManager  — owns mode list + dynamic list of OverlayWindow instances
    │     └── OverlayWindow[0..N]  — each: fullscreen, click-through, own mode
    ├── FrameWorker     — QThread: capture → manager.distribute(raw_frame)
    ├── SplitScreenMouseRemapper — OS hook thread; remaps clicks in split-screen
    └── MiniHUD         — floating draggable control widget (bottom-center)
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from core.engine import VisionEngine
from core.overlay_manager import OverlayManager
from core.frame_worker import FrameWorker
from core.mouse_remap import SplitScreenMouseRemapper
from utils.window_manager import WindowManager
from ui.main_window import ControlPanel
from ui.mini_hud import MiniHUD


def main() -> None:
    """
    Application entry point.

    Startup sequence:
      1. Create QApplication.
      2. Create VisionEngine (opens screen-capture context).
      3. Create OverlayManager (loads all 21 modes).
      4. Create FrameWorker wired to engine + manager.
      5. Connect aboutToQuit shutdown hooks (order matters).
      6. Create MiniHUD — the floating visual control widget.
      7. Create initial overlay and start worker.
      8. Enter Qt event loop.
    """
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Vision Simulator")

    engine         = VisionEngine()
    manager        = OverlayManager()
    window_manager = WindowManager()
    worker         = FrameWorker(engine, manager)
    panel          = ControlPanel(manager, window_manager)

    # Create the split-screen click remapper after manager (needs manager.split_screen).
    # The remapper's background thread is started here and owns the OS mouse hook.
    # It reads manager.split_screen.is_active and .layout_mode on every button event;
    # both are plain attribute reads, safe under CPython's GIL without a lock.
    remapper = SplitScreenMouseRemapper(manager.split_screen)
    remapper.start()

    # ── Shutdown hooks (ORDER MATTERS) ────────────────────────────────────
    # 1. remapper.stop() FIRST — removes the OS mouse hook before any Qt widget
    #    is destroyed; the hook thread must not outlive the app.
    # 2. hide_all() second — all overlay windows vanish before further cleanup.
    # 3. request_stop() is non-blocking (flag only) — safe inside aboutToQuit.
    # 4. panel.force_close() allows the panel window to actually close.
    app.aboutToQuit.connect(remapper.stop)
    app.aboutToQuit.connect(manager.hide_all)
    app.aboutToQuit.connect(worker.request_stop)
    app.aboutToQuit.connect(panel.force_close)

    # Floating visual control widget — replaces all keyboard shortcuts.
    # Positioned at the bottom-center of the primary screen on startup.
    hud = MiniHUD(manager, app, panel)
    hud.show()

    # Create the initial overlay (starts with mode 0 — Dog Vision)
    manager.add_overlay(mode_index=0)
    panel.show()
    worker.start()

    print("Vision Simulator started.")
    print(f"  {len(manager.modes)} vision modes available.")
    print("  Use the mini HUD at the bottom of your screen to control overlays.\n")

    exit_code = app.exec()

    # ── Post-event-loop cleanup ───────────────────────────────────────────
    if not worker.wait(1000):
        worker.terminate()
        worker.wait(200)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
