"""
Region Drawer
ui/region_drawer.py

Full-screen transparent input-capture window that lets the user drag a
rectangle to define a regional overlay area.

Usage
─────
    drawer = RegionDrawer()
    drawer.region_selected.connect(lambda x1,y1,x2,y2: ...)
    drawer.cancelled.connect(panel.show)
    drawer.show()

The caller hides the control panel before showing the drawer so the full
desktop is visible.  The drawer emits region_selected (or cancelled) and
hides itself; the caller's slot re-shows the panel and creates the overlay.

Win32 notes
───────────
WS_EX_TRANSPARENT is intentionally NOT applied here — unlike the overlay
windows, the drawer must receive all mouse events to track the drag.
WS_EX_LAYERED is still set so the translucent background renders correctly.
WS_EX_TOOLWINDOW + WS_EX_NOACTIVATE prevent the window from appearing in the
taskbar / Alt-Tab list and from stealing keyboard focus from the shell.
"""

from __future__ import annotations

import ctypes

from PySide6.QtCore    import Qt, Signal, QPoint
from PySide6.QtGui     import QPainter, QColor, QFont, QPen, QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget


# ── Win32 extended-style constants ────────────────────────────────────────────
_GWL_EXSTYLE      = -20
_WS_EX_LAYERED    = 0x00080000
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_APPWINDOW  = 0x00040000
_SWP_NOMOVE       = 0x0002
_SWP_NOSIZE       = 0x0001
_SWP_NOZORDER     = 0x0004
_SWP_FRAMECHANGED = 0x0020

# Minimum drag size that counts as a valid region (pixels)
_MIN_REGION_PX = 20


class RegionDrawer(QWidget):
    """
    Full-screen input-capture canvas for drawing a rectangular region.

    Signals
    ───────
    region_selected(x1, y1, x2, y2)  — fires on valid mouse release
    cancelled()                        — fires on ESC or right-click
    """

    region_selected: Signal = Signal(int, int, int, int)
    cancelled:       Signal = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._start:   QPoint | None = None
        self._end:     QPoint | None = None
        self._drawing: bool          = False
        self._setup_window()

    # ── Window initialisation ─────────────────────────────────────────────

    def _setup_window(self) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(QApplication.primaryScreen().geometry())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        _ = self.winId()                       # ensure native HWND exists
        self.setGeometry(QApplication.primaryScreen().geometry())
        self._apply_win32_styles()

    def _apply_win32_styles(self) -> None:
        """
        Applies Win32 extended styles.

        WS_EX_TRANSPARENT is intentionally omitted — we need to receive mouse
        events.  All other flags mirror the overlay windows' configuration so
        the drawer does not appear in the taskbar or Alt-Tab switcher.
        """
        try:
            hwnd      = int(self.winId())
            ex_style  = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            new_style = (
                ex_style
                | _WS_EX_LAYERED       # required for translucent background
                | _WS_EX_TOOLWINDOW    # hide from taskbar and Alt-Tab
                | _WS_EX_NOACTIVATE    # do not steal keyboard focus
            ) & ~_WS_EX_APPWINDOW      # remove forced taskbar button
            ctypes.windll.user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, new_style)
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
            )
        except Exception as exc:
            print(f"[RegionDrawer] Win32 error: {exc}")

    # ── Mouse events ──────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start   = event.position().toPoint()
            self._end     = self._start
            self._drawing = True
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self._cancel()

    def mouseMoveEvent(self, event) -> None:
        if self._drawing:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._drawing:
            return
        self._drawing = False
        self._end = event.position().toPoint()

        if self._start and self._end:
            x1 = min(self._start.x(), self._end.x())
            y1 = min(self._start.y(), self._end.y())
            x2 = max(self._start.x(), self._end.x())
            y2 = max(self._start.y(), self._end.y())

            if (x2 - x1) >= _MIN_REGION_PX and (y2 - y1) >= _MIN_REGION_PX:
                self.hide()
                self.region_selected.emit(x1, y1, x2, y2)
                return

        # Region too small — reset and let the user try again
        self._start   = None
        self._end     = None
        self._drawing = False
        self.update()

    # ── Keyboard ──────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _cancel(self) -> None:
        self._start   = None
        self._end     = None
        self._drawing = False
        self.hide()
        self.cancelled.emit()

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        # Faint dark tint so drawing mode is visually distinct but the desktop
        # remains readable underneath.
        painter.fillRect(self.rect(), QColor(0, 0, 0, 45))

        # Instruction text centred near the top
        painter.setPen(QColor(220, 220, 220, 220))
        font = QFont("Segoe UI", 13)
        painter.setFont(font)
        painter.drawText(
            self.rect().adjusted(0, 28, 0, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Click and drag to draw a region  ·  ESC or right-click to cancel",
        )

        # Rectangle preview while dragging
        if self._start and self._end:
            x1 = min(self._start.x(), self._end.x())
            y1 = min(self._start.y(), self._end.y())
            w  = abs(self._end.x() - self._start.x())
            h  = abs(self._end.y() - self._start.y())

            # Semi-transparent fill inside the selection
            painter.fillRect(x1, y1, w, h, QColor(137, 180, 250, 35))

            # Dashed outline while dragging; solid on release (briefly before hide)
            pen = QPen(QColor(137, 180, 250, 220), 2)
            pen.setStyle(
                Qt.PenStyle.DashLine if self._drawing else Qt.PenStyle.SolidLine
            )
            painter.setPen(pen)
            painter.drawRect(x1, y1, w, h)

            # Size indicator below the bottom-right corner
            if self._drawing and w > 60 and h > 30:
                painter.setPen(QColor(200, 200, 200, 180))
                small_font = QFont("Segoe UI", 10)
                painter.setFont(small_font)
                painter.drawText(
                    x1 + w + 6,
                    y1 + h + 16,
                    f"{w} × {h}",
                )

        painter.end()
