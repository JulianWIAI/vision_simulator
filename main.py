"""
Vision Simulator — Entry Point (Phase 4: Window-Aware Filtering + Control Panel)

Shutdown sequence (critical — same two-phase pattern as Phase 1)
─────────────────────────────────────────────────────────────────
app.aboutToQuit (GUI thread):
  1. manager.hide_all()     — all overlay HWNDs disappear immediately;
                              Windows never sees an unresponsive window covering
                              the screen, so no 'Not Responding' dialog appears.
  2. worker.request_stop()  — sets _running=False (non-blocking flag only).
                              Do NOT connect worker.stop() here — stop() calls
                              QThread.wait(3000) which blocks the main thread
                              while Qt is still dispatching queued signals → deadlock.
app.exec() returns:
  3. keyboard.unhook_all()  — remove OS-level hooks before process exits.
  4. worker.wait(300)       — give thread up to 300 ms to exit cleanly.
  5. worker.terminate()     — force-kill if it didn't stop in time.

Architecture
────────────
  QApplication (main thread / Qt event loop)
    ├── OverlayManager  — owns mode list + dynamic list of OverlayWindow instances
    │     └── OverlayWindow[0..N]  — each: fullscreen, click-through, own mode
    └── FrameWorker     — QThread: capture → manager.distribute(raw_frame)

Keyboard hotkeys (all non-ESC use QTimer.singleShot for GUI-thread safety)
────────────────
  N         Add a new overlay (starts at mode 1 / Dog Vision)
  M         Cycle the last overlay's vision mode forward
  X         Remove the last overlay
  C         Toggle the Control Panel window
  1–9 / 0   Set mode 0–9 on the last overlay
  ESC       Quit (app.quit() is thread-safe in Qt 5+)
"""

import sys
import keyboard
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer

from core.engine import VisionEngine
from core.overlay_manager import OverlayManager
from core.frame_worker import FrameWorker
from utils.window_manager import WindowManager
from ui.main_window import ControlPanel


def _print_banner(manager: OverlayManager) -> None:
    """Prints a startup banner listing all registered modes and their hotkeys."""
    modes = manager.modes
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║            V I S I O N   S I M U L A T O R          ║")
    print("║           Phase 2 — Multi-Overlay Edition            ║")
    print("╠══════════════════════════════════════════════════════╣")
    for i, mode in enumerate(modes):
        key_label = f"[{i + 1}]" if i < 9 else ("[0]" if i == 9 else "    ")
        print(f"║  {key_label:<5}  {mode.name:<43}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  [N] New overlay   [M] Cycle mode   [X] Remove last ║")
    print("║  [1-9/0] Set mode on last overlay   [ESC] Quit      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def _register_hotkeys(
    manager: OverlayManager,
    app: QApplication,
    panel: ControlPanel,
) -> None:
    """
    Installs global keyboard hooks via the `keyboard` library.

    The keyboard library fires callbacks on its own OS thread (no Qt event loop).

    - Mode switching (M, 1–9, 0): only writes Python attributes under the GIL —
      safe to call directly, same pattern as Phase 1's engine.switch_mode().
    - Overlay add/remove (N, X): create/hide QWidgets, which MUST run on the
      GUI thread.  QTimer.singleShot(0, app, callable) posts the call to the
      event loop owned by `app` (the GUI thread).  The `app` context argument
      is required — without it Qt has no event loop to post to from a non-Qt
      thread and the callback silently never fires.
    - ESC / C: always active — ESC quits or exits split-screen; C re-shows the panel.
    - N/M/X/1-9/0: guarded — skipped when the ControlPanel has OS focus so that
      typing in the panel's text fields / combos doesn't fire overlay actions.
      panel._focused is a plain bool written by the GUI-thread changeEvent handler
      and read here under CPython's GIL (single-bool read is atomic).
    """

    def _active(fn):
        """Call fn() only when the ControlPanel does not have OS focus."""
        def _wrapped():
            if not panel._focused:
                fn()
        return _wrapped

    # 1–9 / 0 → set a specific mode on the last overlay.
    for i in range(min(9, len(manager.modes))):
        keyboard.on_press_key(
            str(i + 1),
            lambda _, idx=i: _active(lambda: manager.set_mode_for_last(idx))(),
            suppress=False,
        )
    if len(manager.modes) > 9:
        keyboard.on_press_key(
            "0",
            lambda _: _active(lambda: manager.set_mode_for_last(9))(),
            suppress=False,
        )

    # M: cycle mode
    keyboard.on_press_key(
        "m", lambda _: _active(manager.cycle_mode_for_last)(), suppress=False
    )

    # N / X: create or hide a QWidget — must post to the GUI thread.
    keyboard.on_press_key(
        "n",
        lambda _: _active(lambda: QTimer.singleShot(0, app, manager.add_overlay))(),
        suppress=False,
    )
    keyboard.on_press_key(
        "x",
        lambda _: _active(lambda: QTimer.singleShot(0, app, manager.remove_last))(),
        suppress=False,
    )

    # C: toggle the Control Panel — unguarded so the user can always show it.
    def _toggle_panel() -> None:
        if panel.isVisible():
            panel.hide()
        else:
            panel.show()

    keyboard.on_press_key(
        "c",
        lambda _: QTimer.singleShot(0, app, _toggle_panel),
        suppress=False,
    )

    # ESC: deactivate split-screen (first press) or quit (no split active).
    # Unguarded — must always be reachable to avoid getting stuck.
    def _on_esc() -> None:
        if manager.split_screen.is_active:
            from core.split_screen_manager import LAYOUT_NONE
            manager.split_screen.layout_mode = LAYOUT_NONE
            # Sync the radio-button UI on the GUI thread
            QTimer.singleShot(0, app, panel._sync_split_screen_ui)
        else:
            app.quit()

    keyboard.on_press_key("esc", lambda _: _on_esc(), suppress=False)


def main() -> None:
    """
    Application entry point.

    Startup sequence:
      1. Create QApplication.
      2. Create VisionEngine (opens screen-capture context).
      3. Create OverlayManager (loads all 19 modes).
      4. Create FrameWorker wired to engine + manager.
      5. Connect aboutToQuit shutdown hooks (order matters).
      6. Register global keyboard hotkeys.
      7. Create initial overlay and start worker.
      8. Enter Qt event loop.

    Shutdown sequence (after ESC or app.quit()):
      9.  Qt dispatches aboutToQuit → manager.hide_all() → worker.request_stop().
     10.  app.exec() returns.
     11.  keyboard.unhook_all() cleans up OS hooks.
     12.  worker.wait(300) / terminate() joins the thread.
     13.  sys.exit with the Qt exit code.
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

    # ── Shutdown hooks (ORDER MATTERS) ────────────────────────────────────
    # 1. hide_all() must be FIRST — all HWNDs vanish before any cleanup runs.
    # 2. request_stop() is non-blocking (flag only) — safe inside aboutToQuit.
    # 3. panel.force_close() allows the panel window to actually close.
    app.aboutToQuit.connect(manager.hide_all)
    app.aboutToQuit.connect(worker.request_stop)
    app.aboutToQuit.connect(panel.force_close)

    _register_hotkeys(manager, app, panel)
    _print_banner(manager)

    # Create the initial overlay (starts with mode 0 — Dog Vision)
    manager.add_overlay(mode_index=0)
    panel.show()
    worker.start()

    print(f"Starting with 1 overlay: {manager.modes[0].name}")
    print("Overlay active  |  fully click-through  |  global hotkeys enabled")
    print("Control Panel open  |  press C to toggle\n")

    exit_code = app.exec()

    # ── Post-event-loop cleanup ───────────────────────────────────────────
    keyboard.unhook_all()
    if not worker.wait(1000):
        worker.terminate()
        worker.wait(200)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
