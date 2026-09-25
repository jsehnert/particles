"""Qt window and input layer for slice_analysis.

Images retain their source resolution. All windows share the analysis model,
zoom, and ROI; mouse coordinates are mapped through the painted image rectangle.
"""

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class ImageCanvas(QWidget):
    def __init__(self, viewer, name):
        super().__init__()
        self.viewer = viewer
        self.name = name
        self.image = QImage()
        self.press = None
        self.dragged = False
        self.setMinimumSize(240, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, array):
        # Match HighGUI's float-image display convention (0..1 -> 0..255).
        if array.dtype != np.uint8:
            array = np.clip(np.nan_to_num(array) * 255, 0, 255).astype(np.uint8)
        array = np.ascontiguousarray(array)
        self.image = QImage(
            array.data,
            array.shape[1],
            array.shape[0],
            array.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        self.update()

    def rectangles(self):
        data = self.viewer.model.global_data
        h, w = data.vol.shape[1:]
        cw, ch = w * data.zoom_scale, h * data.zoom_scale
        x = max(0, min(data.zoom_cx * w - cw / 2, w - cw))
        y = max(0, min(data.zoom_cy * h - ch / 2, h - ch))
        source = QRectF(x, y, cw, ch)
        scale = min(self.width() / cw, self.height() / ch)
        target = QRectF(
            (self.width() - cw * scale) / 2,
            (self.height() - ch * scale) / 2,
            cw * scale,
            ch * scale,
        )
        return source, target

    def image_point(self, position, clamp=False):
        source, target = self.rectangles()
        if not clamp and not target.contains(position):
            return None
        h, w = self.viewer.model.global_data.vol.shape[1:]
        x = source.x() + (position.x() - target.x()) * source.width() / target.width()
        y = source.y() + (position.y() - target.y()) * source.height() / target.height()
        return (max(0, min(w - 1, math.floor(x))), max(0, min(h - 1, math.floor(y))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        if self.image.isNull():
            return
        source, target = self.rectangles()
        painter.drawImage(target, self.image, source)
        roi = self.viewer.model.global_data.roi
        if roi is None:
            return

        def point(x, y):
            return QPointF(
                target.x() + (x + 0.5 - source.x()) * target.width() / source.width(),
                target.y() + (y + 0.5 - source.y()) * target.height() / source.height(),
            )

        painter.setClipRect(target)
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        a, b = point(*roi[:2]), point(*roi[2:])
        if roi[:2] == roi[2:]:
            painter.setBrush(QColor(0, 255, 0))
            painter.drawEllipse(a, 4, 4)
        else:
            painter.drawRect(QRectF(a, b).normalized())

    def mousePressEvent(self, event):
        point = self.image_point(event.position())
        if point is None:
            return
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            self.press = (event.position(), point)
            self.dragged = False
        elif event.button() == Qt.MouseButton.RightButton:
            data = self.viewer.model.global_data
            data.zoom_cx = point[0] / data.vol.shape[2]
            data.zoom_cy = point[1] / data.vol.shape[1]
            self.viewer.refresh()

    def mouseDoubleClickEvent(self, event):
        # A second press is handled identically; no double-click timing logic.
        self.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.press is None:
            return
        if (
            event.position() - self.press[0]
        ).manhattanLength() >= QApplication.startDragDistance():
            self.dragged = True
        if self.dragged:
            self.set_roi(self.image_point(event.position(), clamp=True))

    def set_roi(self, end):
        start = self.press[1]
        self.viewer.model.global_data.roi = (
            min(start[0], end[0]),
            min(start[1], end[1]),
            max(start[0], end[0]),
            max(start[1], end[1]),
        )
        self.viewer.refresh()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.press is None:
            return
        if (
            event.position() - self.press[0]
        ).manhattanLength() >= QApplication.startDragDistance():
            self.dragged = True
        point = self.image_point(event.position(), clamp=True)
        if self.dragged:
            self.set_roi(point)
        else:
            model = self.viewer.model
            model.on_mouse_singleclick(
                event=0,
                display_x=int(event.position().x()),
                display_y=int(event.position().y()),
                image_x=point[0],
                image_y=point[1],
                flags=0,
                window_name=self.name,
                z=model.global_data.z_current,
            )
        self.press = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            self.viewer.zoom(0.8 if delta > 0 else 1.25)
        event.accept()


class ViewerWindow(QMainWindow):
    def __init__(self, viewer, name):
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle(name)

    def closeEvent(self, event):
        self.viewer.application.quit()
        event.accept()


class SliceViewer:
    def __init__(self, model, application):
        self.model = model
        self.application = application
        self.windows = {}
        self.canvases = {}
        self.shortcuts = []
        self.controls = {}
        data = model.global_data
        for name in (model.WIN_SLICE, model.WIN_MAXPROJ, model.WIN_ENH):
            window = ViewerWindow(self, name)
            canvas = ImageCanvas(self, name)
            window.setCentralWidget(canvas)
            window.resize(900, 900)
            self.windows[name] = window
            self.canvases[name] = canvas

        self.info = ViewerWindow(self, "Info")
        self.info.resize(640, 1000)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.info.setCentralWidget(panel)
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setRange(data.z_min, data.z_max)
        self.z_slider.setValue(data.z_current)
        self.z_slider.setTracking(False)
        self.z_spin = QSpinBox()
        self.z_spin.setRange(data.z_min, data.z_max)
        self.z_spin.setValue(data.z_current)
        self.z_spin.setKeyboardTracking(False)
        row = QHBoxLayout()
        row.addWidget(QLabel("z slice"))
        row.addWidget(self.z_slider)
        row.addWidget(self.z_spin)
        layout.addLayout(row)
        self.z_slider.valueChanged.connect(self.set_z)
        self.z_spin.valueChanged.connect(self.set_z)
        form = QFormLayout()
        layout.addLayout(form)
        maximum = QCheckBox("Use maximum projection")
        maximum.setChecked(data.use_max)
        maximum.toggled.connect(lambda value: self.change("use_max", value))
        form.addRow(maximum)
        for attr, (label, options, _) in model.CTRL_CYCLE.items():
            combo = QComboBox()
            combo.addItems(options)
            combo.setCurrentText(getattr(data, attr))
            combo.currentTextChanged.connect(
                lambda value, attr=attr: self.change(attr, value)
            )
            form.addRow(label, combo)
        for attr, (label, step, lo, hi, _) in model.CTRL_FLOAT.items():
            value = getattr(data, attr)
            spin = QSpinBox() if isinstance(value, int) else QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            self.controls[attr] = spin
            spin.valueChanged.connect(lambda value, attr=attr: self.change(attr, value))
            form.addRow(label, spin)
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        self.next_button = QPushButton("Next particle")
        self.next_button.clicked.connect(self.next_particle)
        buttons.addWidget(self.next_button)
        record = QPushButton("Record (R)")
        record.clicked.connect(self.record)
        buttons.addWidget(record)
        self.text = QTextBrowser()
        layout.addWidget(self.text, 1)
        self.scan_timer = QTimer(self.info)
        self.scan_timer.setInterval(0)
        self.scan_timer.timeout.connect(self.scan_step)
        for keys, action in [
            (["A", "Left"], lambda: self.set_z(data.z_current - 1)),
            (["D", "Right"], lambda: self.set_z(data.z_current + 1)),
            (["R"], self.record),
            (["+", "="], lambda: self.zoom(0.8)),
            (["-", "_"], lambda: self.zoom(1.25)),
            (["0"], self.reset_zoom),
            (["Q", "Escape"], application.quit),
        ]:
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), self.info)
                shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
                shortcut.activated.connect(action)
                self.shortcuts.append(shortcut)

    def set_z(self, z):
        data = self.model.global_data
        z = max(data.z_min, min(data.z_max, z))
        if z != data.z_current:
            click = self.model.last_click_info
            for canvas in self.canvases.values():
                canvas.press = None
            data.update_slice_location(z)
            if click is not None and not data.auto_scroll:
                self.model.on_mouse_singleclick(
                    event=0,
                    display_x=int(click["display_x"]),
                    display_y=int(click["display_y"]),
                    image_x=int(click["clicked_x"]),
                    image_y=int(click["clicked_y"]),
                    flags=0,
                    window_name=self.model.WIN_SLICE,
                    z=z,
                )
            else:
                self.model._clear_pending_particle()
            self.model.update_windows()
        for control in (self.z_slider, self.z_spin):
            control.blockSignals(True)
            control.setValue(z)
            control.blockSignals(False)

    def change(self, attr, value):
        data = self.model.global_data
        if attr == "slab_thickness":
            value = min(61, int(value) | 1)
            self.controls[attr].blockSignals(True)
            self.controls[attr].setValue(value)
            self.controls[attr].blockSignals(False)
        setattr(data, attr, value)
        if attr in self.model.ENH_ATTRS:
            self.model.create_enhancement()
        elif attr in self.model.MAX_PROJ_ATTRS or attr == "use_max":
            self.model.compute_max_projection(data.vol, *data.slab_range)
        else:
            data.update_slice_location(data.z_current)
            data.update_slab_data()
        self.model.update_windows()

    def zoom(self, factor):
        data = self.model.global_data
        if data.roi is not None:
            x1, y1, x2, y2 = data.roi
            height, width = data.vol.shape[1:]
            data.zoom_cx = (x1 + x2) / (2 * width)
            data.zoom_cy = (y1 + y2) / (2 * height)
        data.zoom_scale = float(np.clip(data.zoom_scale * factor, 0.02, 1.0))
        self.refresh()

    def reset_zoom(self):
        self.model.global_data.zoom_scale = 1.0
        self.refresh()

    def record(self):
        before = len(self.model.particles_found)
        self.model._record_pending_particle()
        if self.model.pending_particle_x is None:
            message = "No pending particle selection."
        elif len(self.model.particles_found) == before:
            message = "Particle already recorded."
        else:
            message = "Particle recorded."
        self.info.statusBar().showMessage(message, 5000)

    def next_particle(self):
        if self.scan_timer.isActive():
            self.scan_timer.stop()
            self.next_button.setText("Next particle")
        else:
            self.next_button.setText("Stop search")
            self.scan_timer.start()

    def scan_step(self):
        data = self.model.global_data
        data.auto_scroll = True
        try:
            self.set_z(data.z_current + 1)
        finally:
            data.auto_scroll = False
        if data.n_particle_pixels > 0 or data.z_current >= data.z_max:
            self.scan_timer.stop()
            self.next_button.setText("Next particle")

    def refresh(self):
        data = self.model.global_data
        y = min(data.vol.shape[1] - 1, round(data.zoom_cy * data.vol.shape[1]))
        x = min(data.vol.shape[2] - 1, round(data.zoom_cx * data.vol.shape[2]))
        suffix = f"z={data.z_current} | center (row,col)=({y}, {x}) | gray={data.vol[data.z_current, y, x]}"
        for name, array in [
            (self.model.WIN_SLICE, data.display_slice),
            (self.model.WIN_MAXPROJ, data.display_maxproj),
            (self.model.WIN_ENH, data.display_enh),
        ]:
            if array.size:
                self.canvases[name].set_image(array)
            self.windows[name].setWindowTitle(f"{name} | {suffix}")
        self.info.setWindowTitle(f"Info | {suffix}")
        local = dict(data.slice_stats)
        local["sigma_residual"] = data.get_sigma_residual()
        sections = []
        for title, values in [
            ("Local noise", local),
            ("Global noise", data.global_stats),
        ]:
            rows = "".join(
                f"<tr><td>{key}</td><td>{value:.4g}</td></tr>"
                for key, value in values.items()
                if isinstance(value, (int, float))
            )
            sections.append(f"<h3>{title}</h3><table>{rows}</table>")
        click = self.model.last_click_info
        details = "<h3>Last click</h3>"
        if click:
            details += f"<p>Display (row,col): ({click['display_y']}, {click['display_x']})</p>"
            for label, prefix in [("Clicked", "clicked"), ("Refined", "refined")]:
                details += (
                    f"<p>{label} (z,row,col): ({click[prefix + '_z']}, "
                    f"{click[prefix + '_y']}, {click[prefix + '_x']})"
                    f"<br>Gray level: {click[prefix + '_gray']}</p>"
                )
            particle_found = click["particle_volume"] > 0
            found_color = "#22c55e" if particle_found else "#ef4444"
            details += (
                f'<p><span style="color: {found_color};">'
                f"Particle found: {particle_found}</span>"
                f"<br>Particle volume: {click['particle_volume']} voxels"
                f"<br>Noise level: {click['noise_level']:.3f}</p>"
            )
        else:
            details += "<p>No click selection.</p>"
        if self.model.pending_particle_x is None:
            details += "<p>No pending particle selection.</p>"
        self.text.setHtml(
            "<table><tr>"
            + "".join(f"<td>{s}</td>" for s in sections)
            + "</tr></table>"
            + details
        )

    def show(self):
        for index, window in enumerate(self.windows.values()):
            window.move(40 + index * 60, 60 + index * 40)
            window.show()
        self.info.show()

    def close(self):
        self.scan_timer.stop()
        for window in [*self.windows.values(), self.info]:
            window.close()
