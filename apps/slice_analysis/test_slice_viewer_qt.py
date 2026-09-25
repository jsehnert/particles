"""Headless interaction tests for the slice analysis Qt viewer."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from apps.slice_analysis import slice_analysis as model
from apps.slice_analysis.slice_viewer_qt import SliceViewer


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        volume = np.zeros((5, 60, 100), dtype=np.uint8)
        volume[3, 30, 51] = 200
        data = SimpleNamespace(
            vol=volume,
            vol_data_path=Path("example.raw"),
            z_min=1,
            z_max=3,
            z_current=2,
            zoom_scale=0.5,
            zoom_cx=0.5,
            zoom_cy=0.5,
            roi=(1, 1, 80, 50),
            display_slice=volume[2],
            display_maxproj=volume.max(axis=0),
            display_enh=np.zeros((60, 100), dtype=np.float32),
            slice_stats={},
            global_stats={},
            get_sigma_residual=lambda: 1.0,
            use_max=False,
            baseline_method="mean",
            n_particle_pixels=0,
            auto_scroll=False,
        )
        for attr, (_, step, lo, hi, _) in model.CTRL_FLOAT.items():
            setattr(data, attr, 3 if attr == "slab_thickness" else lo)
        data.update_slice_location = lambda z: setattr(data, "z_current", z)
        model.global_data = data
        model._clear_pending_particle()
        model.last_click_info = None
        self.viewer = SliceViewer(model, self.app)
        model.qt_viewer = self.viewer
        self.viewer.refresh()
        self.viewer.show()
        self.app.processEvents()
        self.canvas = self.viewer.canvases[model.WIN_SLICE]

    def tearDown(self):
        self.viewer.close()
        model.qt_viewer = None

    def position(self, x, y):
        source, target = self.canvas.rectangles()
        return QPoint(
            round(
                target.x() + (x + 0.5 - source.x()) * target.width() / source.width()
            ),
            round(
                target.y() + (y + 0.5 - source.y()) * target.height() / source.height()
            ),
        )

    def test_legacy_csv_coordinates_are_reordered_without_changing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "particles.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "volume_name",
                        "x",
                        "y",
                        "z",
                        "gray_level",
                        "noise_level",
                        "particle_volume",
                    )
                )
                writer.writerow(("example.raw", 51, 30, 3, 200, 1.5, 12))

            with patch.object(model, "PARTICLES_FOUND_PATH", path):
                rows = model._load_particles_found()

            with path.open(newline="") as stream:
                migrated_reader = csv.DictReader(stream)
                migrated_rows = list(migrated_reader)

            self.assertEqual(
                tuple(migrated_reader.fieldnames or ()), model.PARTICLES_FOUND_FIELDS
            )
            self.assertEqual(rows[0]["z"], "3")
            self.assertEqual(rows[0]["y"], "30")
            self.assertEqual(rows[0]["x"], "51")
            self.assertEqual(migrated_rows, rows)

    def test_click_refines_but_marker_follows_exact_pixel_and_records_once(self):
        with (
            patch.object(model.ParticleExtractor, "particle", np.ones((1, 1, 1))),
            patch.object(model.ParticleExtractor, "noise_estimate", 1.0),
        ):
            QTest.mouseClick(
                self.canvas, Qt.MouseButton.LeftButton, pos=self.position(50, 30)
            )
        self.assertEqual(model.global_data.roi, (50, 30, 50, 30))
        self.assertEqual(
            (
                model.pending_particle_z_max,
                model.pending_particle_y,
                model.pending_particle_x,
            ),
            (3, 30, 51),
        )
        with (
            patch.object(model.ParticleExtractor, "particle", np.ones((1, 1, 1))),
            patch.object(model.ParticleExtractor, "noise_estimate", 1.0),
        ):
            QTest.mouseClick(
                self.canvas, Qt.MouseButton.LeftButton, pos=self.position(51, 30)
            )
        self.assertEqual(model.global_data.roi, (51, 30, 51, 30))
        self.assertIn("#22c55e", self.viewer.text.toHtml())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "particles.csv"
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerow(model.PARTICLES_FOUND_FIELDS)
            with (
                patch.object(model, "PARTICLES_FOUND_PATH", path),
                patch.object(model, "particles_found", []),
                patch.object(model, "particles_found_keys", set()),
            ):
                self.viewer.record()
                self.viewer.record()
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual(
                    tuple(rows[0]),
                    (
                        "volume_name",
                        "z",
                        "y",
                        "x",
                        "gray_level",
                        "noise_level",
                        "particle_volume",
                    ),
                )
                self.assertEqual(rows[0]["gray_level"], "200")
                self.assertEqual(rows[0]["x"], "51")
                self.assertEqual(rows[0]["y"], "30")
                self.assertEqual(rows[0]["z"], "3")
                self.assertEqual(
                    rows[0]["noise_level"],
                    f"{model.last_click_info['noise_level']:.3f}",
                )
                self.assertEqual(
                    rows[0]["particle_volume"],
                    str(model.last_click_info["particle_volume"]),
                )
        with (
            patch.object(model, "update_windows"),
            patch.object(model.ParticleExtractor, "particle", np.zeros((1, 1, 1))),
            patch.object(model.ParticleExtractor, "noise_estimate", 1.0),
        ):
            self.viewer.set_z(3)
        self.assertEqual(model.last_click_info["clicked_z"], 3)
        self.assertEqual(model.last_click_info["refined_z"], 3)
        self.assertEqual(
            (model.pending_particle_y, model.pending_particle_x),
            (30, 51),
        )
        self.assertIn("#ef4444", self.viewer.text.toHtml())

    def test_drag_and_letterbox_mapping(self):
        QTest.mousePress(
            self.canvas, Qt.MouseButton.LeftButton, pos=self.position(40, 25)
        )
        QTest.mouseMove(self.canvas, self.position(60, 35))
        QTest.mouseRelease(
            self.canvas, Qt.MouseButton.LeftButton, pos=self.position(60, 35)
        )
        self.assertEqual(model.global_data.roi, (40, 25, 60, 35))
        self.assertIsNone(model.pending_particle_x)
        self.assertIsNone(self.canvas.image_point(QPoint(0, 0)))
        self.assertEqual(self.canvas.image.width(), 100)
        self.assertEqual(self.canvas.image.height(), 60)


class EnhancementStartupTests(unittest.TestCase):
    def test_interior_enhancement_matches_3d_neighborhood_reference(self):
        rng = np.random.default_rng(123)
        residual_volume = rng.normal(size=(3, 16, 24)).astype(np.float32)
        threshold = 1.0
        area_threshold = 2
        padded_mask = np.pad(residual_volume > threshold, 1)
        neighborhoods = np.lib.stride_tricks.sliding_window_view(padded_mask, (3, 3, 3))
        expected_mask = neighborhoods.sum(axis=(3, 4, 5))[1] > area_threshold
        expected_image = np.max(residual_volume, axis=0) * expected_mask
        data = SimpleNamespace(
            z_current=2,
            dead_zone_scale=threshold,
            area_threshold=area_threshold,
            auto_scroll=False,
            n_particle_pixels=0,
            _residual_cache={z: (residual_volume[z - 1], 1.0) for z in (1, 2, 3)},
            record_particles=lambda *_: None,
        )

        with (
            patch.object(model, "global_data", data, create=True),
            patch.object(
                model,
                "mask_core",
                np.ones(residual_volume.shape[1:], dtype=bool),
            ),
            patch.object(
                model.cv2,
                "GaussianBlur",
                side_effect=lambda image, *_args, **_kwargs: image.copy(),
            ),
        ):
            model.create_enhancement()

        np.testing.assert_array_equal(data.display_enh, expected_image)
        self.assertEqual(data.n_particle_pixels, np.count_nonzero(expected_mask))

    def test_enhancement_without_legacy_gui_globals(self):
        # Exercise the real enhancement path at a volume boundary, where only
        # the current slice and its next neighbour have residuals cached.
        residual = np.zeros((16, 24), dtype=np.float32)
        for scanning in (False, True):
            with self.subTest(scanning=scanning):
                data = SimpleNamespace(
                    z_current=2,
                    dead_zone_scale=1.0,
                    area_threshold=1,
                    auto_scroll=scanning,
                    n_particle_pixels=99,
                    _residual_cache={
                        2: (residual.copy(), 1.0),
                        3: (residual.copy(), 1.0),
                    },
                )
                with (
                    patch.object(model, "global_data", data, create=True),
                    patch.object(
                        model, "mask_core", np.ones_like(residual, dtype=bool)
                    ),
                ):
                    model.create_enhancement()
                self.assertEqual(data.display_enh.shape, residual.shape)
                self.assertFalse(data.display_enh.any())
                self.assertEqual(data.n_particle_pixels, 0)


if __name__ == "__main__":
    unittest.main()
