"""
ui/region_drawer.py

Full-screen transparent input-capture window that lets the user drag a
rectangle to define a regional overlay area.

Cross-platform design
─────────────────────
All OS-specific window hardening is delegated to the platform abstraction
layer via get_platform().apply_drawer_styles(self).

On Windows, the platform layer applies:
  WS_EX_LAYERED   — enables the layered-window compositing path (translucency)
  WS_EX_TOOLWINDOW — hides from taskbar / Alt-Tab
  WS_EX_NOACTIVATE — prevents focus theft from the shell
  WS_EX_TRANSPARENT is intentionally NOT applied here: unlike the overlay
  windows, the drawer must receive all mouse events to track the drag gesture.

On macOS, Qt.Tool + Qt.WindowStaysOnTopHint + WA_TranslucentBackground are
sufficient; the platform call adds capture exclusion so the canvas does not
appear in MSS frames while the user is drawing.

Usage
─────
    drawer = RegionDrawer()
    drawer.region_selected.connect(lambda x1,y1,x2,y2: ...)
    drawer.cancelled.connect(panel.show)
    drawer.show()

The caller hides the control panel before showing the drawer so the full
desktop is visible.  The drawer emits region_selected (or cancelled) and
hides itself; the caller's slot re-shows the panel and creates the overlay.
"""

from __future__ import annotations

from PySide6.QtCore    import Qt, Signal, QPoint
from PySide6.QtGui     import QPainter, QColor, QFont, QPen, QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget

from platform_layer import get_platform


# Minimum drag size that counts as a valid region (pixels).
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
        """
        Applies Qt window flags common to all platforms.

        Note: Qt.WindowTransparentForInput is intentionally omitted — the
        drawer must receive mouse press / move / release events.
        """
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool   # hides from taskbar (Win32) / Dock (macOS)
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(QApplication.primaryScreen().geometry())

    def showEvent(self, event) -> None:
        """
        Applies platform-specific window hardening once the native handle exists.

        Geometry is refreshed at show time to pick up any monitor or resolution
        changes since the widget was constructed.
        """
        super().showEvent(event)

        # Trigger native handle creation before calling winId() or platform APIs.
        _ = self.winId()

        # Refresh geometry in case the screen configuration changed.
        self.setGeometry(QApplication.primaryScreen().geometry())

        # Delegate OS-specific hardening to the platform abstraction layer.
        # On Windows: adds WS_EX_LAYERED, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE.
        # On macOS:   adds capture exclusion via NSWindow.setSharingType_.
        get_platform().apply_drawer_styles(self)

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

        # Region too small — reset and let the user try again.
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

        # Instruction text centred near the top.
        painter.setPen(QColor(220, 220, 220, 220))
        font = QFont("Segoe UI", 13)
        painter.setFont(font)
        painter.drawText(
            self.rect().adjusted(0, 28, 0, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Click and drag to draw a region  ·  ESC or right-click to cancel",
        )

        # Rectangle preview while dragging.
        if self._start and self._end:
            x1 = min(self._start.x(), self._end.x())
            y1 = min(self._start.y(), self._end.y())
            w  = abs(self._end.x() - self._start.x())
            h  = abs(self._end.y() - self._start.y())

            # Semi-transparent fill inside the selection.
            painter.fillRect(x1, y1, w, h, QColor(137, 180, 250, 35))

            # Dashed outline while dragging; solid on release (briefly before hide).
            pen = QPen(QColor(137, 180, 250, 220), 2)
            pen.setStyle(
                Qt.PenStyle.DashLine if self._drawing else Qt.PenStyle.SolidLine
            )
            painter.setPen(pen)
            painter.drawRect(x1, y1, w, h)

            # Size indicator below the bottom-right corner.
            if self._drawing and w > 60 and h > 30:
                painter.setPen(QColor(200, 200, 200, 180))
                small_font = QFont("Segoe UI", 10)
                painter.setFont(small_font)
                painter.drawText(x1 + w + 6, y1 + h + 16, f"{w} × {h}")

        painter.end()
