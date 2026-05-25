"""
Control Panel — Phase 4/5

QMainWindow with sidebar navigation and QStackedWidget content area.
Provides full GUI control over overlays, window tracking, global modes,
performance settings, and split-screen layout without blocking the UI thread.

Views (index in QStackedWidget)
────────────────────────────────
  0  Regions      — list active overlays; add/remove; assign modes
  1  Windows      — enumerate Win32 windows; create windowed overlays
  2  Global Modes — master mode override applied to all overlays at once
  3  Settings     — display, HUD, and performance toggles
  4  Split Screen — layout selector; per-panel mode assignment

Threading
─────────
  All slots and timer callbacks run on the GUI thread.

  _track_timer  (100 ms) — calls OverlayManager.update_window_rects()
                           to refresh clip_rects for window-tracked overlays.
  _refresh_timer (500 ms) — keeps the Regions list current when keyboard
                            hotkeys add/remove overlays while the panel is open.

Close behaviour
───────────────
  Pressing ✕ hides the panel rather than destroying it.
  Keyboard hotkey 'C' toggles visibility so the panel is never lost.
  force_close() is called by main.py's aboutToQuit handler for real shutdown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QEvent, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QCloseEvent

if TYPE_CHECKING:
    from core.overlay_manager import OverlayManager
    from utils.window_manager import WindowManager


# ── Layout constants ──────────────────────────────────────────────────────────
_PANEL_W  = 760
_PANEL_H  = 680     # increased from 520 — gives the Split Screen and Regions views breathing room
_SIDEBAR_W = 148


# ── Stylesheet ─────────────────────────────────────────────────────────────────
_DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
/* ── Sidebar nav ── */
QListWidget#sidebar_nav {
    background-color: #181825;
    border: none;
    border-right: 1px solid #313244;
    border-radius: 0;
    font-size: 14px;
    outline: none;
}
QListWidget#sidebar_nav::item {
    padding: 16px 20px;
    border-radius: 0;
    border-left: 3px solid transparent;
    color: #a6adc8;
}
QListWidget#sidebar_nav::item:selected {
    background-color: #1e1e2e;
    color: #89b4fa;
    border-left: 3px solid #89b4fa;
}
QListWidget#sidebar_nav::item:hover:!selected {
    background-color: #252538;
    color: #cdd6f4;
}
/* ── Content lists ── */
QListWidget {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    padding: 8px 12px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #313244;
    color: #89b4fa;
}
QListWidget::item:hover:!selected {
    background-color: #252538;
}
/* ── Buttons ── */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 500;
    min-width: 88px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QPushButton:disabled {
    background-color: #181825;
    color: #585b70;
    border-color: #313244;
}
QPushButton#accent {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-color: #89b4fa;
    font-weight: 600;
}
QPushButton#accent:hover {
    background-color: #b4befe;
    border-color: #b4befe;
}
QPushButton#accent:pressed {
    background-color: #7287fd;
    border-color: #7287fd;
}
QPushButton#danger {
    color: #f38ba8;
    border-color: #f38ba8;
}
QPushButton#danger:hover {
    background-color: #f38ba8;
    color: #1e1e2e;
    border-color: #f38ba8;
}
/* ── Combo box ── */
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 170px;
}
QComboBox:hover {
    border-color: #89b4fa;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #a6adc8;
    width: 0;
    height: 0;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    selection-background-color: #45475a;
    outline: none;
}
/* ── Checkboxes ── */
QCheckBox {
    spacing: 10px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #45475a;
    border-radius: 4px;
    background-color: #181825;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QCheckBox::indicator:disabled {
    background-color: #313244;
    border-color: #313244;
}
QCheckBox:disabled {
    color: #585b70;
}
/* ── Labels ── */
QLabel#section_label {
    color: #89b4fa;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 1.5px;
    margin-top: 4px;
}
QLabel#status_label {
    color: #6c7086;
    font-size: 12px;
}
QLabel#title_label {
    background-color: #181825;
    color: #89b4fa;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 14px 0 12px 20px;
    border-bottom: 1px solid #313244;
}
/* ── Radio buttons ── */
QRadioButton {
    spacing: 10px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #45475a;
    border-radius: 8px;
    background-color: #181825;
}
QRadioButton::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QRadioButton::indicator:hover {
    border-color: #89b4fa;
}
/* ── Panel slot labels ── */
QLabel#slot_label {
    color: #89b4fa;
    font-weight: 700;
    font-size: 13px;
    min-width: 28px;
}
/* ── Separators ── */
QFrame#h_sep {
    background-color: #313244;
    max-height: 1px;
    min-height: 1px;
    margin: 6px 0;
}
"""


class ControlPanel(QMainWindow):
    """
    Floating, interactive control panel for the Vision Simulator.

    Not click-through — the user interacts with it normally.
    Closing hides it (force_close() performs a real shutdown close).
    """

    def __init__(
        self,
        manager: "OverlayManager",
        window_manager: "WindowManager",
    ) -> None:
        super().__init__()
        self._manager             = manager
        self._window_manager      = window_manager
        self._force_closing       = False
        self._selected_overlay_id: int | None = None
        # Written by GUI-thread changeEvent; read GIL-safely by keyboard thread
        # to suppress hotkeys (N/M/X/1-9) while the user types in this panel.
        self._focused: bool = False

        self._setup_ui()
        self._setup_timers()

        # React to hotkey-driven overlay add/remove without polling
        self._manager.overlays_changed.connect(self._populate_regions_list)
        self._manager.overlays_changed.connect(self._update_overlay_counts)

        # Initial data fill
        self._populate_regions_list()
        self._update_overlay_counts()

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.setWindowTitle("Vision Simulator — Control Panel")
        self.setMinimumSize(_PANEL_W, _PANEL_H)
        self.resize(_PANEL_W, _PANEL_H)
        self.setStyleSheet(_DARK_STYLESHEET)

        # Position top-right so it doesn't cover the primary monitor center
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - _PANEL_W - 24, screen.top() + 40)

        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(_SIDEBAR_W)
        sidebar_vbox = QVBoxLayout(sidebar_container)
        sidebar_vbox.setContentsMargins(0, 0, 0, 0)
        sidebar_vbox.setSpacing(0)

        app_title = QLabel("VISION\nSIMULATOR")
        app_title.setObjectName("title_label")
        app_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        sidebar_vbox.addWidget(app_title)

        self._nav = QListWidget()
        self._nav.setObjectName("sidebar_nav")
        self._nav.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._nav.addItems(["Regions", "Windows", "Global Modes", "Settings", "Split Screen"])
        self._nav.setCurrentRow(0)
        sidebar_vbox.addWidget(self._nav, stretch=1)
        root.addWidget(sidebar_container)

        # Stacked content
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_regions_view())        # 0
        self._stack.addWidget(self._build_windows_view())        # 1
        self._stack.addWidget(self._build_global_modes_view())   # 2
        self._stack.addWidget(self._build_settings_view())       # 3
        self._stack.addWidget(self._build_split_screen_view())   # 4
        root.addWidget(self._stack, stretch=1)

        # Wire nav after stack is constructed
        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.currentRowChanged.connect(self._on_view_changed)

    def _setup_timers(self) -> None:
        # Update windowed-overlay clip rects as windows move
        self._track_timer = QTimer(self)
        self._track_timer.setInterval(100)
        self._track_timer.timeout.connect(self._update_tracked_window_rects)
        self._track_timer.start()

        # Keep Regions list in sync with keyboard-driven changes
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self._auto_refresh_regions)
        self._refresh_timer.start()

    # ── View builders ─────────────────────────────────────────────────────

    def _build_regions_view(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        layout.addWidget(_section_label("ACTIVE OVERLAYS"))

        self._regions_list = QListWidget()
        self._regions_list.setMinimumHeight(130)
        self._regions_list.currentItemChanged.connect(self._on_regions_selection_changed)
        self._regions_list.itemClicked.connect(self._on_regions_item_clicked)
        layout.addWidget(self._regions_list)

        btn_row = QHBoxLayout()
        self._add_overlay_btn = QPushButton("+ Add Overlay")
        self._add_overlay_btn.setObjectName("accent")
        self._add_overlay_btn.clicked.connect(self._on_add_overlay)
        btn_row.addWidget(self._add_overlay_btn)

        self._remove_overlay_btn = QPushButton("✕ Remove Selected")
        self._remove_overlay_btn.setObjectName("danger")
        self._remove_overlay_btn.clicked.connect(self._on_remove_overlay)
        btn_row.addWidget(self._remove_overlay_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addWidget(_separator())
        layout.addWidget(_section_label("ASSIGN MODE TO SELECTED"))

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._regions_mode_combo = QComboBox()
        _fill_mode_combo(self._regions_mode_combo, self._manager.modes)
        mode_row.addWidget(self._regions_mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        set_mode_btn = QPushButton("Set Mode")
        set_mode_btn.clicked.connect(self._on_set_mode_for_overlay)
        layout.addWidget(set_mode_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(_separator())
        layout.addWidget(_section_label("APPLY PIPELINE EFFECT"))

        pipeline_base_row = QHBoxLayout()
        pipeline_base_row.addWidget(QLabel("Base:"))
        self._pipeline_base_combo = QComboBox()
        _fill_mode_combo(self._pipeline_base_combo, self._manager.modes)
        pipeline_base_row.addWidget(self._pipeline_base_combo)
        pipeline_base_row.addStretch()
        layout.addLayout(pipeline_base_row)

        pipeline_effect_row = QHBoxLayout()
        pipeline_effect_row.addWidget(QLabel("Effect:"))
        self._pipeline_effect_combo = QComboBox()
        from effects.pipeline_effects import PIPELINE_EFFECTS
        for name, _ in PIPELINE_EFFECTS:
            self._pipeline_effect_combo.addItem(name)
        pipeline_effect_row.addWidget(self._pipeline_effect_combo)
        pipeline_effect_row.addStretch()
        layout.addLayout(pipeline_effect_row)

        pipeline_apply_btn = QPushButton("Apply Pipeline")
        pipeline_apply_btn.clicked.connect(self._on_apply_pipeline)
        layout.addWidget(pipeline_apply_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addStretch()

        self._regions_status = QLabel("")
        self._regions_status.setObjectName("status_label")
        layout.addWidget(self._regions_status)

        return widget

    def _build_windows_view(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.addWidget(_section_label("DETECTED WINDOWS"))
        header_row.addStretch()
        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.setFixedWidth(88)
        refresh_btn.clicked.connect(self._refresh_windows)
        header_row.addWidget(refresh_btn)
        layout.addLayout(header_row)

        self._windows_list = QListWidget()
        self._windows_list.setMinimumHeight(180)
        layout.addWidget(self._windows_list)

        layout.addWidget(_separator())
        layout.addWidget(_section_label("TRACK SELECTED WINDOW"))

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Vision Mode:"))
        self._windows_mode_combo = QComboBox()
        _fill_mode_combo(self._windows_mode_combo, self._manager.modes)
        mode_row.addWidget(self._windows_mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        track_btn = QPushButton("Track Selected Window")
        track_btn.setObjectName("accent")
        track_btn.clicked.connect(self._on_track_window)
        layout.addWidget(track_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(_separator())
        layout.addWidget(_section_label("DRAW REGION"))

        draw_mode_row = QHBoxLayout()
        draw_mode_row.addWidget(QLabel("Vision Mode:"))
        self._draw_region_mode_combo = QComboBox()
        _fill_mode_combo(self._draw_region_mode_combo, self._manager.modes)
        draw_mode_row.addWidget(self._draw_region_mode_combo)
        draw_mode_row.addStretch()
        layout.addLayout(draw_mode_row)

        draw_btn = QPushButton("✦  Draw Region on Screen")
        draw_btn.setObjectName("accent")
        draw_btn.clicked.connect(self._on_draw_region)
        layout.addWidget(draw_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addStretch()

        self._windows_status = QLabel(
            "Select a window and choose a mode, then click Track.\n"
            "The overlay will follow the window as it moves.\n"
            "Or draw a custom region directly on the screen."
        )
        self._windows_status.setObjectName("status_label")
        self._windows_status.setWordWrap(True)
        layout.addWidget(self._windows_status)

        return widget

    def _build_global_modes_view(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        layout.addWidget(_section_label("MASTER OVERRIDE"))

        self._global_override_chk = QCheckBox("Enable Global Override")
        self._global_override_chk.setToolTip(
            "Applies one mode to all overlays simultaneously, "
            "ignoring their individual mode settings."
        )
        layout.addWidget(self._global_override_chk)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._global_mode_combo = QComboBox()
        _fill_mode_combo(self._global_mode_combo, self._manager.modes)
        mode_row.addWidget(self._global_mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        apply_btn = QPushButton("Apply to All Overlays")
        apply_btn.setObjectName("accent")
        apply_btn.clicked.connect(self._on_apply_global_mode)
        layout.addWidget(apply_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(_separator())

        self._global_count_label = QLabel("")
        self._global_count_label.setObjectName("status_label")
        layout.addWidget(self._global_count_label)

        info = QLabel(
            "Global Override sets the selected mode on every active overlay.\n"
            "Individual overlay modes are unaffected in memory — switching back "
            "to any overlay's Regions entry restores control per-overlay."
        )
        info.setObjectName("status_label")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        return widget

    def _build_settings_view(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        layout.addWidget(_section_label("DISPLAY"))

        self._show_overlays_chk = QCheckBox("Show Overlays")
        self._show_overlays_chk.setChecked(True)
        self._show_overlays_chk.toggled.connect(self._on_toggle_overlays)
        layout.addWidget(self._show_overlays_chk)

        self._show_hud_chk = QCheckBox("Show Mode Name in HUD")
        self._show_hud_chk.setChecked(True)
        self._show_hud_chk.toggled.connect(self._on_toggle_hud)
        layout.addWidget(self._show_hud_chk)

        layout.addWidget(_separator())
        layout.addWidget(_section_label("PERFORMANCE"))

        backpressure_chk = QCheckBox("Frame Drop Protection  (always on)")
        backpressure_chk.setChecked(True)
        backpressure_chk.setEnabled(False)
        backpressure_chk.setToolTip(
            "Each overlay skips incoming frames while its previous frame is "
            "still being painted, capping the Qt event-queue depth at one "
            "frame per overlay and preventing memory build-up."
        )
        layout.addWidget(backpressure_chk)

        self._fps_cap_chk = QCheckBox("Cap Capture to 30 FPS")
        self._fps_cap_chk.toggled.connect(self._on_toggle_fps_cap)
        self._fps_cap_chk.setToolTip(
            "Adds a 33 ms sleep to the FrameWorker loop, reducing CPU usage "
            "at the cost of slightly higher input latency."
        )
        layout.addWidget(self._fps_cap_chk)

        layout.addWidget(_separator())
        layout.addWidget(_section_label("STATUS"))

        self._settings_count_label = QLabel("")
        self._settings_count_label.setObjectName("status_label")
        layout.addWidget(self._settings_count_label)

        layout.addStretch()
        return widget

    def _build_split_screen_view(self) -> QWidget:
        """
        Constructs the Split Screen view.

        Layout selector (radio buttons) drives panel visibility: Off shows
        nothing, 2× layouts show rows A+B, 4× grid shows all four rows.
        The Apply button writes to SplitScreenManager on the GUI thread;
        the worker thread reads those values next frame via compose().
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # ── Layout selector ───────────────────────────────────────────────
        layout.addWidget(_section_label("LAYOUT"))

        self._split_bg = QButtonGroup(self)
        grid = QGridLayout()
        grid.setSpacing(8)

        _radio_specs = [
            (0, "Off",              0, 0),
            (1, "Top / Bottom",     0, 1),
            (2, "Left / Right",     1, 0),
            (3, "2×2 Grid",         1, 1),
        ]
        for btn_id, label, row, col in _radio_specs:
            rb = QRadioButton(label)
            if btn_id == 0:
                rb.setChecked(True)
            self._split_bg.addButton(rb, btn_id)
            grid.addWidget(rb, row, col)

        layout.addLayout(grid)

        layout.addWidget(_separator())
        layout.addWidget(_section_label("PANEL MODES"))

        # ── Mode assignment rows (A–D) ─────────────────────────────────────
        _panel_letters = ["A", "B", "C", "D"]
        self._split_rows:   list[QWidget] = []
        self._split_combos: list[QComboBox] = []

        for letter in _panel_letters:
            row_w = QWidget()
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(10)

            lbl = QLabel(f"[{letter}]")
            lbl.setObjectName("slot_label")
            lbl.setFixedWidth(30)
            row_h.addWidget(lbl)

            combo = QComboBox()
            combo.addItem("Raw  (no filter)")
            for mode in self._manager.modes:
                combo.addItem(mode.name)
            row_h.addWidget(combo, stretch=1)

            self._split_rows.append(row_w)
            self._split_combos.append(combo)
            layout.addWidget(row_w)

        # Initial state: Off → no rows visible
        for row_w in self._split_rows:
            row_w.setVisible(False)

        # Wire layout-change signal AFTER rows are built
        self._split_bg.idClicked.connect(self._on_split_layout_changed)

        layout.addStretch()

        self._split_apply_btn = QPushButton("Disable Split Screen")
        self._split_apply_btn.setObjectName("accent")
        self._split_apply_btn.clicked.connect(self._on_apply_split_screen)
        layout.addWidget(self._split_apply_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self._split_status = QLabel("")
        self._split_status.setObjectName("status_label")
        self._split_status.setWordWrap(True)
        layout.addWidget(self._split_status)

        return widget

    # ── Slot handlers ─────────────────────────────────────────────────────

    def _on_view_changed(self, index: int) -> None:
        """Lazy-loads data when the user switches to a view."""
        if index == 0:
            self._populate_regions_list()
        elif index == 1 and self._windows_list.count() == 0:
            self._refresh_windows()
        elif index in (2, 3):
            self._update_overlay_counts()
        elif index == 4:
            self._sync_split_screen_ui()

    # ·· Regions view ·····················································

    def _on_add_overlay(self) -> None:
        mode_idx = self._regions_mode_combo.currentIndex()
        self._manager.add_overlay(mode_index=mode_idx)
        # overlays_changed → _populate_regions_list fires automatically

    def _on_regions_selection_changed(self, current, previous) -> None:
        if current is not None:
            self._selected_overlay_id = current.data(Qt.ItemDataRole.UserRole)

    def _on_regions_item_clicked(self, item) -> None:
        if item is not None:
            self._selected_overlay_id = item.data(Qt.ItemDataRole.UserRole)

    def _on_remove_overlay(self) -> None:
        overlay_id = self._selected_overlay_id
        if overlay_id is None:
            item = self._regions_list.currentItem()
            if item is None:
                self._regions_status.setText("Select an overlay to remove.")
                return
            overlay_id = item.data(Qt.ItemDataRole.UserRole)
        self._selected_overlay_id = None
        self._manager.remove_overlay_by_id(overlay_id)

    def _on_set_mode_for_overlay(self) -> None:
        overlay_id = self._selected_overlay_id
        if overlay_id is None:
            item = self._regions_list.currentItem()
            if item is None:
                self._regions_status.setText("Select an overlay first.")
                return
            overlay_id = item.data(Qt.ItemDataRole.UserRole)
        mode_idx = self._regions_mode_combo.currentIndex()
        self._manager.set_mode_for_overlay(overlay_id, mode_idx)

    def _on_apply_pipeline(self) -> None:
        overlay_id = self._selected_overlay_id
        if overlay_id is None:
            item = self._regions_list.currentItem()
            if item is None:
                self._regions_status.setText("Select an overlay first.")
                return
            overlay_id = item.data(Qt.ItemDataRole.UserRole)

        from effects.pipeline_effects import PIPELINE_EFFECTS
        from core.vision_pipeline import VisionPipeline

        base_idx   = self._pipeline_base_combo.currentIndex()
        effect_idx = self._pipeline_effect_combo.currentIndex()
        base_mode  = self._manager.modes[base_idx]
        _, effect  = PIPELINE_EFFECTS[effect_idx]

        pipeline = VisionPipeline(base_mode, [effect] if effect is not None else [])
        self._manager.set_custom_mode_for_overlay(overlay_id, pipeline)
        self._regions_status.setText(f"Pipeline applied: {pipeline.name}")

    # ·· Windows view ·····················································

    def _refresh_windows(self) -> None:
        windows = self._window_manager.get_windows(force=True)
        self._windows_list.clear()
        for w in windows:
            item = QListWidgetItem(w["title"])
            item.setData(Qt.ItemDataRole.UserRole, w)
            self._windows_list.addItem(item)
        self._windows_status.setText(f"{len(windows)} windows detected.")

    def _on_track_window(self) -> None:
        item = self._windows_list.currentItem()
        if item is None:
            self._windows_status.setText("Select a window from the list first.")
            return

        win_info = item.data(Qt.ItemDataRole.UserRole)
        hwnd     = win_info["id"]
        title    = win_info["title"]
        mode_idx = self._windows_mode_combo.currentIndex()

        # Live rect (window may have moved since the list was populated)
        rect = self._window_manager.get_rect(hwnd)
        if rect is None:
            self._windows_status.setText(f"'{title}' is no longer available.")
            return

        self._manager.add_windowed_overlay(hwnd, title, rect, mode_index=mode_idx)
        mode_name = self._manager.modes[mode_idx].name
        self._windows_status.setText(f"Tracking: {title!r}  —  {mode_name}")

    def _on_draw_region(self) -> None:
        """
        Hides the control panel and opens the full-screen region drawer.

        The drawer emits region_selected(x1, y1, x2, y2) on a valid drag, or
        cancelled() on ESC / right-click.  Both slots re-show the panel.
        """
        from ui.region_drawer import RegionDrawer

        mode_idx = self._draw_region_mode_combo.currentIndex()
        self.hide()

        self._region_drawer = RegionDrawer()
        self._region_drawer.region_selected.connect(
            lambda x1, y1, x2, y2: self._on_region_drawn(x1, y1, x2, y2, mode_idx)
        )
        self._region_drawer.cancelled.connect(self._on_region_cancelled)
        self._region_drawer.show()
        # Give the drawer keyboard focus so ESC is reliably received
        self._region_drawer.activateWindow()
        self._region_drawer.setFocus()

    def _on_region_drawn(
        self, x1: int, y1: int, x2: int, y2: int, mode_idx: int
    ) -> None:
        """Creates the region overlay and re-shows the panel."""
        self._manager.add_region_overlay((x1, y1, x2, y2), mode_index=mode_idx)
        mode_name = self._manager.modes[mode_idx].name
        self._windows_status.setText(
            f"Region created: {x2 - x1} × {y2 - y1} px  —  {mode_name}"
        )
        self.show()

    def _on_region_cancelled(self) -> None:
        """Re-shows the panel after the user cancels region drawing."""
        self._windows_status.setText("Region drawing cancelled.")
        self.show()

    # ·· Global Modes view ················································

    def _on_apply_global_mode(self) -> None:
        mode_idx = self._global_mode_combo.currentIndex()
        infos    = self._manager.get_overlay_infos()
        if not infos:
            self._global_count_label.setText("No active overlays.")
            return
        for info in infos:
            self._manager.set_mode_for_overlay(info["id"], mode_idx)
        mode_name = self._manager.modes[mode_idx].name
        self._global_count_label.setText(
            f"Applied '{mode_name}' to {len(infos)} overlay(s)."
        )

    # ·· Split Screen view ················································

    def _on_split_layout_changed(self, layout_id: int) -> None:
        """Shows/hides panel-mode rows to match the selected layout."""
        is_two_panel  = layout_id in (1, 2)
        is_four_panel = layout_id == 3

        self._split_rows[0].setVisible(is_two_panel or is_four_panel)
        self._split_rows[1].setVisible(is_two_panel or is_four_panel)
        self._split_rows[2].setVisible(is_four_panel)
        self._split_rows[3].setVisible(is_four_panel)

        apply_label = "Disable Split Screen" if layout_id == 0 else "▶  Apply Split Screen"
        self._split_apply_btn.setText(apply_label)

    def _on_apply_split_screen(self) -> None:
        """
        Writes layout + mode assignments to SplitScreenManager.

        All writes are GUI-thread attribute assignments (atomic under GIL),
        so no lock is needed against the worker thread reading compose().
        """
        _LAYOUT_MAP = {
            0: "none",
            1: "2x_horizontal",
            2: "2x_vertical",
            3: "4x_grid",
        }
        layout_id   = self._split_bg.checkedId()
        layout_name = _LAYOUT_MAP[layout_id]
        ss = self._manager.split_screen

        # Write layout first so is_active() is correct immediately after.
        # sync_split_screen_windows() reads is_active() so it must run AFTER this.
        ss.layout_mode = layout_name

        # Reconcile overlay visibility with the new split-screen state.
        # When split is active this hides overlays[1:] so they do not occlude
        # the composed frame that distribute() writes to overlays[0].
        # When split is off it restores all overlays to visible.
        self._manager.sync_split_screen_windows()  # hides/shows overlays per z-order fix

        if layout_name != "none":
            for slot, combo in enumerate(self._split_combos):
                # Index 0 = "Raw (no filter)" → None
                # Index n (n>=1) → modes[n-1]
                mode_idx = combo.currentIndex() - 1
                ss.set_mode(slot, self._manager.modes[mode_idx] if mode_idx >= 0 else None)

        if layout_name == "none":
            self._split_status.setText("Split screen disabled.")
        else:
            names = [
                (ss.modes[i].name if ss.modes[i] else "Raw")
                for i in range(4 if layout_id == 3 else 2)
            ]
            self._split_status.setText(
                f"Active: {layout_name}  —  " + "  /  ".join(names)
            )

    def _sync_split_screen_ui(self) -> None:
        """Refreshes the split-screen view to reflect current manager state."""
        ss = self._manager.split_screen
        layout_id_map = {
            "none":          0,
            "2x_horizontal": 1,
            "2x_vertical":   2,
            "4x_grid":       3,
        }
        lid = layout_id_map.get(ss.layout_mode, 0)
        btn = self._split_bg.button(lid)
        if btn:
            btn.setChecked(True)
        self._on_split_layout_changed(lid)

    # ·· Settings view ····················································

    def _on_toggle_overlays(self, visible: bool) -> None:
        if visible:
            self._manager.show_all()
        else:
            self._manager.hide_all()

    def _on_toggle_hud(self, enabled: bool) -> None:
        self._manager.hud_enabled = enabled

    def _on_toggle_fps_cap(self, enabled: bool) -> None:
        # Extension point: wire to FrameWorker.set_fps_cap() in Phase 5
        print(f"[ControlPanel] FPS cap {'enabled (30)' if enabled else 'disabled'} — "
              "FrameWorker integration pending.")

    # ── Periodic callbacks ────────────────────────────────────────────────

    def _update_tracked_window_rects(self) -> None:
        """100 ms timer: pushes fresh clip_rects to windowed overlays."""
        self._manager.update_window_rects(self._window_manager)

    def _auto_refresh_regions(self) -> None:
        """500 ms timer: refreshes regions list while that view is active."""
        if self._nav.currentRow() == 0:
            self._populate_regions_list()

    # ── Data helpers ──────────────────────────────────────────────────────

    def _populate_regions_list(self) -> None:
        infos = self._manager.get_overlay_infos()
        # Block signals during clear+rebuild so that currentItemChanged fired by
        # clear() and setCurrentRow() cannot overwrite _selected_overlay_id.
        # The user's explicit click (itemClicked / currentItemChanged) still
        # updates it because those events arrive outside this block.
        self._regions_list.blockSignals(True)
        try:
            self._regions_list.clear()
            restore_row = -1
            for row, info in enumerate(infos):
                if info.get("is_region"):
                    scope = "Region"
                elif info["has_clip"]:
                    scope = "Window"
                else:
                    scope = "Full Screen"
                text  = f"#{info['id'] + 1}  —  {info['mode_name']}   [{scope}]"
                item  = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, info["id"])
                self._regions_list.addItem(item)
                if info["id"] == self._selected_overlay_id:
                    restore_row = row
            if restore_row >= 0:
                self._regions_list.setCurrentRow(restore_row)
        finally:
            self._regions_list.blockSignals(False)
        count = len(infos)
        if hasattr(self, "_regions_status"):
            self._regions_status.setText(
                f"{count} active overlay{'s' if count != 1 else ''}."
            )

    def _update_overlay_counts(self) -> None:
        count = len(self._manager.get_overlay_infos())
        text  = f"Active overlays: {count}"
        if hasattr(self, "_global_count_label"):
            self._global_count_label.setText(text)
        if hasattr(self, "_settings_count_label"):
            self._settings_count_label.setText(text)

    # ── Qt overrides ──────────────────────────────────────────────────────

    def changeEvent(self, event) -> None:
        """Track OS-level activation so keyboard hotkeys can skip while panel is focused."""
        if event.type() == QEvent.Type.ActivationChange:
            self._focused = self.isActiveWindow()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """X button triggers a clean quit so overlays don't get stuck on screen."""
        if self._force_closing:
            event.accept()
        else:
            # Reject the close so Qt doesn't destroy the window early,
            # then request a proper shutdown via the normal aboutToQuit path.
            event.ignore()
            from PySide6.QtWidgets import QApplication
            QApplication.instance().quit()

    def force_close(self) -> None:
        """Called by main.py's aboutToQuit handler — stops timers, then closes."""
        self._track_timer.stop()
        self._refresh_timer.stop()
        self._force_closing = True
        self.close()


# ── Layout helpers (module-level, stateless) ──────────────────────────────────

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("section_label")
    return lbl


def _separator() -> QFrame:
    line = QFrame()
    line.setObjectName("h_sep")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


def _fill_mode_combo(combo: QComboBox, modes) -> None:
    combo.clear()
    for mode in modes:
        combo.addItem(mode.name)
