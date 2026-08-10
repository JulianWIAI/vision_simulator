"""
ui/mini_hud.py

Floating, draggable mini-HUD for controlling overlays without keyboard shortcuts.

Layout (≈290 × 68 px)
─────────────────────
  ╭──────────────────────────────────────────────────╮
  │ ⠿  ◀  [    Dog Vision    ▾]  ▶   ＋   ✕        │  ← mode row
  │    ◀      Overlay  1 / 3     ▶   ⚙   ⏻        │  ← overlay row
  ╰──────────────────────────────────────────────────╯

Controls
────────
  ◀ / ▶ (top row)    — step to previous / next vision mode on the selected overlay
  [Mode Name ▾]       — click to open a scrollable popup of all 21 modes
  ＋                   — add a new fullscreen overlay
  ✕                   — remove the currently selected overlay
  ◀ / ▶ (bottom row) — select which overlay to control (when multiple exist)
  ⚙                   — show / hide the full Control Panel window
  ⏻                   — quit the application

Drag
────
  Click and drag anywhere on the widget background (not on a button) to
  reposition the HUD.  The position is clamped to the primary screen bounds.

Platform hardening
──────────────────
  apply_drawer_styles() is reused: the HUD hides from the taskbar / Dock and
  cannot steal keyboard focus, but unlike the overlay windows it IS visible
  to mouse events (no WS_EX_TRANSPARENT / WindowTransparentForInput).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore    import Qt, QPoint, QTimer, Signal
from PySide6.QtGui     import QPainter, QColor, QBrush, QPen, QFont
from PySide6.QtWidgets import (
    QWidget, QApplication, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QMenu, QSizePolicy,
)

from platform_layer import get_platform

# ── Colour palette ─────────────────────────────────────────────────────────────
_BG         = QColor(15,  15,  28,  220)   # near-black, semi-transparent
_BTN_NORMAL = QColor(30,  30,  58,  255)   # dark indigo
_BTN_HOVER  = QColor(50,  50,  90,  255)   # lighter indigo
_BTN_PRESS  = QColor(70,  70, 120,  255)
_ACCENT     = QColor(137, 180, 250, 255)   # soft blue — mode name highlight
_TEXT       = QColor(205, 214, 244, 255)   # off-white
_SUBTLE     = QColor(140, 150, 180, 200)   # dimmed — overlay counter
_RADIUS     = 10                           # corner radius of the HUD background

# ── Shared button stylesheet ────────────────────────────────────────────────────
_BTN_CSS = """
QPushButton {{
    background: rgba(30, 30, 58, 255);
    color: rgba(205, 214, 244, 255);
    border: none;
    border-radius: 5px;
    font-size: {size}px;
    padding: 0px;
}}
QPushButton:hover  {{ background: rgba(50,  50,  90, 255); }}
QPushButton:pressed {{ background: rgba(70, 70, 120, 255); }}
"""

_MODE_BTN_CSS = """
QPushButton {
    background: rgba(30, 30, 58, 255);
    color: rgba(137, 180, 250, 255);
    border: none;
    border-radius: 5px;
    font-size: 11px;
    padding: 0 6px;
    text-align: center;
}
QPushButton:hover  { background: rgba(50, 50, 90, 255); }
QPushButton:pressed { background: rgba(70, 70, 120, 255); }
"""

_MENU_CSS = """
QMenu {
    background: rgb(22, 22, 40);
    color: rgb(205, 214, 244);
    border: 1px solid rgb(50, 50, 90);
    border-radius: 6px;
    padding: 4px;
    font-size: 11px;
}
QMenu::item { padding: 4px 16px; border-radius: 4px; }
QMenu::item:selected { background: rgb(50, 50, 90); }
"""


def _make_btn(label: str, size: int = 13, min_w: int = 26, min_h: int = 26) -> QPushButton:
    """Creates a flat icon/text button with the shared dark style."""
    btn = QPushButton(label)
    btn.setFixedSize(min_w, min_h)
    btn.setStyleSheet(_BTN_CSS.format(size=size))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


# ── MiniHUD ────────────────────────────────────────────────────────────────────

class MiniHUD(QWidget):
    """
    Floating overlay-control widget — the sole interface for managing overlays
    now that keyboard shortcuts have been removed.
    """

    def __init__(self, manager, app: QApplication, panel) -> None:
        """
        Args:
            manager: OverlayManager instance (shared, read + write).
            app:     QApplication — needed to post GUI-thread calls and to quit.
            panel:   ControlPanel instance — toggled by the ⚙ button.
        """
        super().__init__()

        self._manager = manager
        self._app     = app
        self._panel   = panel

        # ID of the overlay currently selected in the HUD.
        # None means "no overlay exists yet" (empty state).
        self._selected_id: Optional[int] = None

        # Drag state: offset from widget top-left to the cursor at press time.
        self._drag_offset: Optional[QPoint] = None

        self._build_ui()
        self._setup_window()

        # Refresh whenever the overlay list changes.
        manager.overlays_changed.connect(self._on_overlays_changed)

        # Also refresh periodically so mode-name changes triggered by other UI
        # (e.g. the full Control Panel) are reflected in the HUD.
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._refresh_labels)
        self._poll.start(400)   # 400 ms is imperceptibly fast and cheap

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Creates all child widgets and layouts."""

        # ── Row 1: mode controls ──────────────────────────────────────────
        self._btn_prev_mode = _make_btn("◀", size=11)
        self._btn_prev_mode.setToolTip("Previous mode")
        self._btn_prev_mode.clicked.connect(self._prev_mode)

        # Mode-name button doubles as a dropdown trigger.
        self._btn_mode = QPushButton("—")
        self._btn_mode.setFixedSize(130, 26)
        self._btn_mode.setStyleSheet(_MODE_BTN_CSS)
        self._btn_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_mode.setToolTip("Click to choose a mode")
        self._btn_mode.clicked.connect(self._show_mode_menu)

        self._btn_next_mode = _make_btn("▶", size=11)
        self._btn_next_mode.setToolTip("Next mode")
        self._btn_next_mode.clicked.connect(self._next_mode)

        self._btn_add = _make_btn("＋", size=14)
        self._btn_add.setToolTip("Add overlay")
        self._btn_add.clicked.connect(self._add_overlay)

        self._btn_remove = _make_btn("✕", size=11)
        self._btn_remove.setToolTip("Remove this overlay")
        self._btn_remove.clicked.connect(self._remove_overlay)

        row1 = QHBoxLayout()
        row1.setContentsMargins(6, 0, 6, 0)
        row1.setSpacing(3)
        row1.addWidget(self._btn_prev_mode)
        row1.addWidget(self._btn_mode)
        row1.addWidget(self._btn_next_mode)
        row1.addSpacing(6)
        row1.addWidget(self._btn_add)
        row1.addWidget(self._btn_remove)

        # ── Row 2: overlay selector + utility buttons ─────────────────────
        self._btn_prev_ov = _make_btn("◀", size=11, min_w=22, min_h=22)
        self._btn_prev_ov.setToolTip("Previous overlay")
        self._btn_prev_ov.clicked.connect(self._prev_overlay)

        self._lbl_overlay = QLabel("No overlays")
        self._lbl_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_overlay.setStyleSheet(
            "color: rgba(140,150,180,200); font-size: 10px; background: transparent;"
        )
        self._lbl_overlay.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self._btn_next_ov = _make_btn("▶", size=11, min_w=22, min_h=22)
        self._btn_next_ov.setToolTip("Next overlay")
        self._btn_next_ov.clicked.connect(self._next_overlay)

        # Eye button: ● = overlay visible (click to hide), ○ = hidden (click to show)
        self._btn_eye = _make_btn("●", size=10, min_w=22, min_h=22)
        self._btn_eye.setToolTip("Hide this overlay")
        self._btn_eye.clicked.connect(self._toggle_visibility)

        self._btn_settings = _make_btn("⚙", size=12, min_w=22, min_h=22)
        self._btn_settings.setToolTip("Toggle Control Panel")
        self._btn_settings.clicked.connect(self._toggle_panel)

        self._btn_quit = _make_btn("⏻", size=12, min_w=22, min_h=22)
        self._btn_quit.setToolTip("Quit Vision Simulator")
        self._btn_quit.clicked.connect(self._app.quit)

        row2 = QHBoxLayout()
        row2.setContentsMargins(8, 0, 6, 0)
        row2.setSpacing(3)
        row2.addWidget(self._btn_prev_ov)
        row2.addWidget(self._lbl_overlay)
        row2.addWidget(self._btn_next_ov)
        row2.addSpacing(6)
        row2.addWidget(self._btn_eye)
        row2.addWidget(self._btn_settings)
        row2.addWidget(self._btn_quit)

        # ── Main layout ───────────────────────────────────────────────────
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 6, 0, 6)
        vbox.setSpacing(2)
        vbox.addLayout(row1)
        vbox.addLayout(row2)

        self.setFixedSize(290, 68)

    # ── Window setup ───────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        """Applies Qt window flags and positions the HUD at bottom-centre."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool          # hide from taskbar / Dock
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Start at bottom-centre of the primary screen with a small margin.
        screen = QApplication.primaryScreen().geometry()
        x = screen.x() + (screen.width() - self.width())  // 2
        y = screen.y() +  screen.height() - self.height() - 24
        self.move(x, y)

    def showEvent(self, event) -> None:
        """Apply platform-specific hardening (taskbar hiding, no-focus-steal)."""
        super().showEvent(event)
        _ = self.winId()   # ensure native handle exists
        # Reuse drawer styles: receive mouse events, hide from taskbar, no focus steal.
        get_platform().apply_drawer_styles(self)

    # ── Custom painting ────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        """Draws the rounded semi-transparent dark background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fill a rounded rectangle with the semi-transparent background colour.
        painter.setBrush(QBrush(_BG))
        painter.setPen(QPen(QColor(50, 50, 90, 180), 1))  # subtle border
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), _RADIUS, _RADIUS)

        painter.end()

    # ── Drag support ───────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        """Records the drag offset when the user presses on the background."""
        if event.button() == Qt.MouseButton.LeftButton:
            # globalPosition() → current cursor in screen coords.
            # frameGeometry().topLeft() → widget origin in screen coords.
            # Difference = cursor offset relative to widget top-left.
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Moves the widget, clamped within the primary screen."""
        if self._drag_offset and (event.buttons() & Qt.MouseButton.LeftButton):
            new_pos = event.globalPosition().toPoint() - self._drag_offset
            screen  = QApplication.primaryScreen().geometry()
            # Clamp so the HUD never moves off-screen.
            new_pos.setX(max(screen.x(), min(new_pos.x(), screen.right()  - self.width())))
            new_pos.setY(max(screen.y(), min(new_pos.y(), screen.bottom() - self.height())))
            self.move(new_pos)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        event.accept()

    # ── Mode controls ──────────────────────────────────────────────────────

    def _prev_mode(self) -> None:
        """Steps to the previous mode on the selected overlay."""
        info = self._selected_info()
        if info is None:
            return
        n       = len(self._manager.modes)
        new_idx = (info["mode_index"] - 1) % n
        self._manager.set_mode_for_overlay(self._selected_id, new_idx)
        self._refresh_labels()

    def _next_mode(self) -> None:
        """Steps to the next mode on the selected overlay."""
        info = self._selected_info()
        if info is None:
            return
        n       = len(self._manager.modes)
        new_idx = (info["mode_index"] + 1) % n
        self._manager.set_mode_for_overlay(self._selected_id, new_idx)
        self._refresh_labels()

    def _show_mode_menu(self) -> None:
        """Opens a popup menu listing every available vision mode."""
        info = self._selected_info()
        if info is None:
            return

        menu = QMenu(self)
        menu.setStyleSheet(_MENU_CSS)

        current_idx = info["mode_index"]
        for idx, mode in enumerate(self._manager.modes):
            action = menu.addAction(mode.name)
            # Bold the currently active mode so the user can see their position.
            if idx == current_idx:
                font = action.font()
                font.setBold(True)
                action.setFont(font)
            # Use default-argument capture to freeze idx in the closure.
            action.triggered.connect(lambda checked=False, i=idx: self._set_mode(i))

        # Show directly below the mode-name button.
        menu.exec(self._btn_mode.mapToGlobal(QPoint(0, self._btn_mode.height() + 2)))

    def _set_mode(self, mode_index: int) -> None:
        """Sets a specific mode by index on the selected overlay."""
        if self._selected_id is None:
            return
        self._manager.set_mode_for_overlay(self._selected_id, mode_index)
        self._refresh_labels()

    # ── Overlay controls ───────────────────────────────────────────────────

    def _add_overlay(self) -> None:
        """Adds a new fullscreen overlay (must run on the GUI thread)."""
        # QTimer.singleShot with the app as context posts safely to the GUI thread
        # even if this slot is somehow triggered from a non-GUI thread.
        QTimer.singleShot(0, self._app, self._manager.add_overlay)

    def _remove_overlay(self) -> None:
        """Removes the currently selected overlay."""
        if self._selected_id is None:
            return
        ov_id = self._selected_id   # capture before it may change
        QTimer.singleShot(0, self._app, lambda: self._manager.remove_overlay_by_id(ov_id))

    def _prev_overlay(self) -> None:
        """Selects the previous overlay in the list."""
        ids = [info["id"] for info in self._manager.get_overlay_infos()]
        if not ids:
            return
        if self._selected_id not in ids:
            self._selected_id = ids[0]
        else:
            idx = ids.index(self._selected_id)
            self._selected_id = ids[(idx - 1) % len(ids)]
        self._refresh_labels()

    def _next_overlay(self) -> None:
        """Selects the next overlay in the list."""
        ids = [info["id"] for info in self._manager.get_overlay_infos()]
        if not ids:
            return
        if self._selected_id not in ids:
            self._selected_id = ids[0]
        else:
            idx = ids.index(self._selected_id)
            self._selected_id = ids[(idx + 1) % len(ids)]
        self._refresh_labels()

    # ── Visibility toggle ──────────────────────────────────────────────────

    def _toggle_visibility(self) -> None:
        """Shows or hides the selected overlay; overlays_changed refreshes the eye button."""
        if self._selected_id is None:
            return
        self._manager.toggle_overlay_visibility(self._selected_id)

    # ── Settings / quit ────────────────────────────────────────────────────

    def _toggle_panel(self) -> None:
        """Shows or hides the full Control Panel window."""
        if self._panel.isVisible():
            self._panel.hide()
        else:
            self._panel.show()
            self._panel.raise_()

    # ── Internal state helpers ─────────────────────────────────────────────

    def _selected_info(self) -> Optional[dict]:
        """
        Returns the overlay-info dict for the currently selected overlay,
        or None if no overlay is selected / exists.
        """
        if self._selected_id is None:
            return None
        infos = self._manager.get_overlay_infos()
        return next((i for i in infos if i["id"] == self._selected_id), None)

    def _on_overlays_changed(self) -> None:
        """
        Called when overlays are added or removed.

        Ensures _selected_id points to a valid overlay.  If the selected
        overlay was removed, falls back to the last remaining one.
        """
        infos = self._manager.get_overlay_infos()
        ids   = [info["id"] for info in infos]

        if not ids:
            # All overlays removed.
            self._selected_id = None
        elif self._selected_id not in ids:
            # The previously selected overlay no longer exists; pick the newest.
            self._selected_id = ids[-1]

        self._refresh_labels()

    def _refresh_labels(self) -> None:
        """Updates all text labels and button enabled-states to reflect current state."""
        infos = self._manager.get_overlay_infos()
        count = len(infos)
        has   = count > 0

        # ── Mode row ──────────────────────────────────────────────────────
        if has and self._selected_id is not None:
            info = next((i for i in infos if i["id"] == self._selected_id), None)
            mode_name = info["mode_name"] if info else "—"
        else:
            mode_name = "—"

        # Truncate long names so the button stays compact.
        if len(mode_name) > 16:
            mode_name = mode_name[:14] + "…"
        self._btn_mode.setText(f"{mode_name} ▾")

        # ── Overlay row ───────────────────────────────────────────────────
        if has and self._selected_id is not None:
            ids      = [i["id"] for i in infos]
            sel_pos  = ids.index(self._selected_id) + 1 if self._selected_id in ids else 1
            self._lbl_overlay.setText(f"Overlay {sel_pos} / {count}")
        else:
            self._lbl_overlay.setText("No overlays")

        # ── Eye button state ──────────────────────────────────────────────
        if has and self._selected_id is not None:
            info = next((i for i in infos if i["id"] == self._selected_id), None)
            is_visible = info["visible"] if info else True
            self._btn_eye.setText("●" if is_visible else "○")
            self._btn_eye.setToolTip(
                "Hide this overlay" if is_visible else "Show this overlay"
            )

        # ── Button enabled states ─────────────────────────────────────────
        self._btn_prev_mode.setEnabled(has)
        self._btn_mode.setEnabled(has)
        self._btn_next_mode.setEnabled(has)
        self._btn_remove.setEnabled(has)
        self._btn_prev_ov.setEnabled(count > 1)
        self._btn_next_ov.setEnabled(count > 1)
        self._btn_eye.setEnabled(has)
