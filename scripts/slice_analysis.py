#!/usr/bin/env python3
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import skimage
import typer
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
sys.path.insert(0, str(PROJECT_ROOT))

from enhance import softmax_projection  # noqa: E402
from metal import (  # noqa: E402
    estimate_grayscale_range,
    estimate_metal_threshold,
    extract_metal_mask,
)
from noise import (  # noqa: E402
    estimate_slice_stats,
    estimate_volume_slice_stats,
    leavein_median_noise_scale_penta,
)
from volume import load_volume  # noqa: E402


@dataclass
class GlobalData:
    vol: NDArray[np.uint8] = field(default_factory=lambda: np.array([], dtype=np.uint8))
    slab_thickness: int = 11
    z_min: int = 0
    z_max: int = 0
    z_current: int = 0
    voxel_size_mm: float = 0.01640435

    use_max: bool = False

    # softmax parameters
    softmax_beta: float = 2.0
    softmax_window: int = 2

    # enhancement parameters
    baseline_method: Literal["mean", "median"] = "median"
    sigma_residual: float = float("nan")
    dead_zone_scale: float = 3.0
    contrast_scale: float = 4.0
    area_threshold: int = 5

    # metal mask parameters
    metal_min_area: int = 200
    metal_margin: int = 20

    # noise estimation parameters
    noise_clip_scale: float = 2.5

    global_stats: dict = field(default_factory=dict)
    slice_stats: dict = field(default_factory=dict)

    def update_slice_location(self, z: int) -> None:
        """
        Set the current slice index and update dependent slice state to extract the
        slice-level statistics.

        args:
            z: int - the new slice index

        side effects:
            Updates the current slice index and recalculates slice-level statistics.
        """
        if self.z_current != z and z >= 0 and z < self.vol.shape[0]:
            self.z_current = z
            self._update_slice_stats()

    def _update_slice_stats(self) -> None:
        z = self.z_current
        _zmin = z - 2 if z - 2 >= self.z_min else self.z_min
        _zmax = z + 2 if z + 2 <= self.z_max else self.z_max

        # Check for clipping - we need to send 5 slices in the noise estimation
        if _zmin == self.z_min:
            _zmax = _zmin + 4
        if _zmax == self.z_max:
            _zmin = _zmax - 4

        # Don't use the slabs metal mask for the slice noise estimation. The
        # slab variables are managed elsewhere.
        mm = extract_metal_mask(
            self.vol[_zmin : _zmax + 1].max(axis=0),
            self.metal_threshold,
            min_area=self.metal_min_area,
            margin=self.metal_margin,
        )

        self.slice_stats = estimate_slice_stats(
            self.vol[_zmin : _zmax + 1], mask=mm, trunc_k=self.noise_clip_scale
        )

    def update_slab_data(self) -> None:
        """
        Update the slab-level data (max projection, metal mask) and the dependent
        enhancement and noise stats.
        """

        slab_thickness = global_data.slab_thickness
        _zcur = global_data.z_current
        _zmin = _zcur - slab_thickness // 2
        _zmax = _zcur + slab_thickness // 2

        # clip to the jellyroll
        if _zmin < self.z_min:
            _zmin = self.z_min
            _zmax = self.z_min + slab_thickness
        elif _zmax > self.z_max:
            _zmax = self.z_max
            _zmin = _zmax - slab_thickness

        self.max_proj = compute_max_projection(self.vol, _zmin, _zmax)

        mm = extract_metal_mask(
            self.vol[_zmin : _zmax + 1].max(axis=0),
            global_data.metal_threshold,
            min_area=global_data.metal_min_area,
            margin=global_data.metal_margin,
        )
        self.metal_mask = cast(NDArray[np.bool_], mm)

        self.sigma_residual = compute_residual_sigma()
        self.display_enh = create_enhancement()

    max_proj: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    metal_mask: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )

    # Cached display-resolution (DISPLAY_PX × DISPLAY_PX) grayscale images for fast ROI redraws
    display_slice: NDArray[np.uint8] = field(
        default_factory=lambda: np.array([], dtype=np.uint8)
    )
    display_maxproj: NDArray[np.uint8] = field(
        default_factory=lambda: np.array([], dtype=np.uint8)
    )
    display_enh: NDArray[np.uint8] = field(
        default_factory=lambda: np.array([], dtype=np.uint8)
    )

    # ROI in image coords (x1, y1, x2, y2); None means no ROI selected
    roi: tuple[int, int, int, int] | None = None

    # Zoom state: center in image coords, scale factor (1.0 = full view)
    zoom_cx: float = 0.5  # normalised [0,1]
    zoom_cy: float = 0.5
    zoom_scale: float = 1.0  # fraction of image visible; 1.0 = full, 0.25 = 4x zoom

    @property
    def metal_threshold(self) -> int:
        return self.global_stats["metal_threshold"] if self.global_stats else 0

    @property
    def air_grayvalue(self) -> int:
        return self.global_stats["air_grayvalue"] if self.global_stats else 0

    @property
    def slab_range(self) -> tuple[int, int]:
        z_min = max(self.z_current - self.slab_thickness // 2, self.z_min)
        z_max = min(self.z_current + self.slab_thickness // 2, self.z_max)
        return z_min, z_max


global_data = GlobalData()

# Transient drag state for ROI drawing
_drag: dict = {"active": False, "sx": 0, "sy": 0}

"""
Observations:

 - There apper to be many artifacts in the Cu foil.
 - The particle size in slice 1106 of M50L-06.raw is about 4x4 pixels
 - There are many FP particles on the scale 1x1 pixels
   - For such particles, we need to understand their contrast.
   - If the size and contrast are sufficiently low, then we should be able to suppress
 - Calculating persistence blobs in the residual image is not informative
 - When increasing the slab size to 61, we begin to see windmill artifacts

Options to explore:
1. Look at the max projection method. Max - Median
2. Look at Slab - slab.median(axis=0) clipped at 0 follwed by max projection.
3. Look at variants substituting softmax for max to suppress artifacts

- When using max - avg, we are seeing artifacts from anode wires in the core region. Thes
  wires are not constrained and float from slice-to-slice.
- These artifacts are still apparent in the softmax projection
- The softmax projection significantly improves the background speckle noise.

- For each slab, compute the metal mask from the max projection (to suppress artifacts)
- For each slab, estimate the noise level from the histogram of differences between adjacent slices, excluding the metal mask
"""

# region GUI constants and utilities
WIN_SLICE = "Slice"
WIN_MAXPROJ = "Max Projection"
WIN_ENH = "Enhancement"
WIN_STATS = "Noise Statistics"
WIN_CONTROLS = "Controls"

DISPLAY_PX = 1200
ENH_INTERP = cv2.INTER_NEAREST

HIST_W, HIST_H = 600, 450
PAD_L, PAD_R, PAD_T, PAD_B = 70, 20, 40, 60

KEYS_RIGHT = {83, 3, 63235, ord("d")}
KEYS_LEFT = {81, 2, 63234, ord("a")}
KEYS_QUIT = {27, ord("q")}
KEYS_ZOOM_IN = {ord("+"), ord("=")}
KEYS_ZOOM_OUT = {ord("-"), ord("_")}
KEYS_ZOOM_RESET = {ord("0")}

# There are large structural highlights that need to be suppressed

CTRL_W, CTRL_H = 320, 525
CTRL_BTN = {"use_max": (10, 10, 300, 55)}  # (x1, y1, x2, y2)

# Float adjusters: attr -> (label, step, min, max, y_top)
S = -85
CTRL_FLOAT = {
    # label, step_size, min, max, y_top
    "slab_thickness": ("Slab thickness", 2, 3, 61, 120),
    "dead_zone_scale": ("Dead zone", 0.1, 0.0, 10.0, 170),
    "contrast_scale": ("Contrast", 0.2, 1.0, 20.0, 220),
    "metal_min_area": ("Metal min area", 10, 0, 2000, 270),
    "metal_margin": ("Metal margin", 1, 0, 40, 320),
    "softmax_beta": ("Softmax beta", 0.5, 0.5, 20.0, 370),
    "softmax_window": ("Softmax window", 1, 1, 21, 420),
    "area_threshold": ("Area threshold", 1, 0, 21, 470),
}
# Cycle controls: attr -> (label, options, y_top)
CTRL_CYCLE = {
    "baseline_method": ("Baseline", ["mean", "median"], 65),
}
# Button rects within each float row (relative to row y_top, width CTRL_W)
_BTN_W = 36
# endregion - GUI constants and utilities


def compute_residual_sigma() -> float:
    sigma = float(global_data.slice_stats["sigma_corrected"])
    rho1 = float(global_data.slice_stats["rho1"])
    rho2 = float(global_data.slice_stats["rho2"])
    s_thick = global_data.slab_thickness

    leave_out = True
    if global_data.baseline_method == "mean":
        num = np.sqrt(
            s_thick**2 - s_thick * (1 + 2 * (rho1 + rho2)) - 2 * (rho1 + 2 * rho2)
        )
        if leave_out:
            noise_scale = num / (s_thick - 1.0)
        else:
            noise_scale = num / s_thick
    elif global_data.baseline_method == "median":
        noise_scale = leavein_median_noise_scale_penta(s_thick, rho1, rho2)
    else:
        raise ValueError(f"unknown baseline method: {global_data.baseline_method}")

    return float(noise_scale * sigma)


def _float_btn_rects(y_top: int) -> dict[str, tuple[int, int, int, int]]:
    """Return {'-': rect, '+': rect} for a float adjuster row."""
    row_h = 30
    return {
        "-": (10, y_top, 10 + _BTN_W, y_top + row_h),
        "+": (CTRL_W - 10 - _BTN_W, y_top, CTRL_W - 10, y_top + row_h),
    }


def render_controls_panel() -> NDArray[np.uint8]:
    canvas: NDArray[np.uint8] = np.full((CTRL_H, CTRL_W, 3), 45, dtype=np.uint8)

    # Toggle buttons
    for attr, (bx1, by1, bx2, by2) in CTRL_BTN.items():
        active = bool(getattr(global_data, attr, False))
        fill = (0, 160, 80) if active else (80, 80, 80)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), fill, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
        label = f"{attr}: {'ON' if active else 'OFF'}"
        cv2.putText(
            canvas,
            label,
            (bx1 + 10, (by1 + by2) // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    # Float adjuster rows
    for attr, (label, step, lo, hi, y_top) in CTRL_FLOAT.items():
        val = float(getattr(global_data, attr, 0.0))
        btns = _float_btn_rects(y_top)
        for sym, (bx1, by1, bx2, by2) in btns.items():
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (80, 80, 160), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
            cv2.putText(
                canvas,
                sym,
                (bx1 + 9, by2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
        # Value label centered between buttons
        text = f"{label}: {val:.1f}"
        cv2.putText(
            canvas,
            text,
            (10 + _BTN_W + 8, y_top + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    # Cycle controls
    for attr, (label, options, y_top) in CTRL_CYCLE.items():
        cval = str(getattr(global_data, attr, options[0]))
        bx1, by1, bx2, by2 = 10, y_top, CTRL_W - 10, y_top + 45
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (100, 60, 140), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
        cv2.putText(
            canvas,
            f"{label}: {cval}",
            (bx1 + 10, (by1 + by2) // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    return canvas


def _make_controls_callback(vol, state):
    def _cb(event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Toggle buttons
        for attr, (bx1, by1, bx2, by2) in CTRL_BTN.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                setattr(global_data, attr, not getattr(global_data, attr))
                cv2.imshow(WIN_CONTROLS, render_controls_panel())
                update_windows(vol)
                return

        # Float adjuster buttons
        for attr, (label, step, lo, hi, y_top) in CTRL_FLOAT.items():
            for sym, (bx1, by1, bx2, by2) in _float_btn_rects(y_top).items():
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    cur = float(getattr(global_data, attr))
                    delta = step if sym == "+" else -step
                    new_val = float(np.clip(cur + delta, lo, hi))
                    # preserve int type for int fields
                    if isinstance(getattr(global_data, attr), int):
                        new_val = int(round(new_val))
                    setattr(global_data, attr, new_val)
                    cv2.imshow(WIN_CONTROLS, render_controls_panel())
                    update_windows(vol)
                    return

        # Cycle controls
        for attr, (label, options, y_top) in CTRL_CYCLE.items():
            bx1, by1, bx2, by2 = 10, y_top, CTRL_W - 10, y_top + 45
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                cur_cycle = getattr(global_data, attr)
                idx = options.index(cur_cycle) if cur_cycle in options else 0
                setattr(global_data, attr, options[(idx + 1) % len(options)])
                cv2.imshow(WIN_CONTROLS, render_controls_panel())
                update_windows(vol)
                return

    return _cb


# We need to ascertain the size and contrast of the underlying particles.


def create_enhancement() -> NDArray[np.uint8]:
    """
    Create the enhanced image at the current global slice location.

    Compute the local noise estimate from adjacent slab slices. (The stats window reveals how
    the local estimate deviates from the global estimate and we can investigate stationarity.)

    Depending on the baseline method, compute the residual image as the slice minus the slab
    mean or slab median.

    Compute a deadzone mask to exclude residual pixels that are are negative (looking for
    bright particles) or not above the global deadzone plus the noise floor computed as the
    global deadzone_scale times the local noise estimate. Also include the metal mask in the
    deadzone to suppress metal artifacts.

    Returns:
        NDArray[np.uint8]: The particle/artifact image.
    """

    # Extract parameters from global config object
    z_min, z_max = global_data.slab_range
    z_slice = global_data.vol[global_data.z_current]

    sigma = float(global_data.slice_stats["sigma_corrected"])
    n_sigma = global_data.contrast_scale
    s_thick = global_data.slab_thickness
    sigma_scaled = global_data.sigma_residual

    # Now establish the baseline (background for trend removal)
    slab = global_data.vol[z_min : z_max + 1]
    _bm: str = global_data.baseline_method
    leave_out = True
    if _bm == "mean":
        # We have to account for correlations. We are using the assumption that correlations beyond
        # lag 1 are negligible (which can lead to noise over-estimation)
        if leave_out:
            baseline = (slab.sum(axis=0).astype(np.float32) - z_slice) / (s_thick - 1)
        else:
            baseline = slab.mean(axis=0).astype(np.float32)
    elif _bm == "median":
        baseline = np.median(slab, axis=0).astype(np.float32)

    # Compute the residual image and it's neighbors forming a local volume
    residual = z_slice - baseline

    # Create dead-zone of pixels where if slice is within dead_zone_scale * sigma of the baseline, set to 0.
    # Include in the dead zone any pixels in the metal mask, to suppress metal artifacts.
    dead_zone_thresh = global_data.dead_zone_scale * sigma_scaled
    dead_zone_mask = (
        residual < dead_zone_thresh
    )  # We are only seeking bright particles so we don't need to include dark pixels.
    dead_zone_mask |= global_data.metal_mask

    # baseline = global_data.vol[z_min : z_max + 1].mean(axis=0).astype(np.float32)
    residual[dead_zone_mask] = 0  # suppress dead zone in the enhancement
    _remove_small_regions = global_data.area_threshold > 1

    labels = skimage.measure.label(residual > 0, connectivity=2)
    if _remove_small_regions:
        area_threshold = global_data.area_threshold
        labels = skimage.measure.label(residual > 0, connectivity=2)
        labels = skimage.morphology.remove_small_objects(
            labels, min_size=area_threshold, connectivity=2
        )

        unique_labels = np.unique(labels)
        residual[labels == 0] = 0

        for label in unique_labels:
            if label == 0:
                continue
            mask = labels == label
            area = mask.sum()
            avg = residual[mask].mean()
            print(
                f"Region {label}: area={area}, avg gray={avg:.1f}, contrast={avg / sigma_scaled:.1f}"
            )

    # Scale the image for rendering
    # img = residual / (n_sigma * sigma)

    img = cv2.GaussianBlur(residual / (n_sigma * sigma), (3, 3), 0)
    # img = (img - img.min()) / (img.max() - img.min() + 1e-9) * 255.0
    img = cv2.resize(
        img.astype(np.float32), (DISPLAY_PX, DISPLAY_PX), interpolation=ENH_INTERP
    )

    return img


def compute_max_projection(
    vol: NDArray[np.uint8],
    z_min: int,
    z_max: int,
) -> NDArray[np.float32]:
    """Return the max projection of vol over slices z_min to z_max (inclusive)."""
    if global_data.use_max:
        vol[z_min : z_max + 1].max(axis=0).astype(np.float32)

    # Pad the datat o account for the window
    win = global_data.softmax_window
    z_min = max(z_min - win // 2, global_data.z_min)
    z_max = min(z_max + win // 2, global_data.z_max)
    _max_proj = softmax_projection(
        vol[z_min : z_max + 1],
        beta=global_data.softmax_beta,
        window=win,
    )  # side-effect: suppresses outliers and makes the max projection more robust
    return _max_proj


def _draw_stats_column(
    canvas: NDArray[np.uint8],
    title: str,
    fields: list[tuple[str, object]],
    x: int,
) -> None:
    cv2.putText(
        canvas,
        title,
        (x, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    for i, (label, value) in enumerate(fields):
        value_text = f"{value:.4g}" if isinstance(value, int | float) else "n/a"
        cv2.putText(
            canvas,
            f"{label}: {value_text}",
            (x, 95 + i * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def render_noise_stats(local_stats: dict, global_stats: dict) -> NDArray[np.uint8]:
    canvas: NDArray[np.uint8] = np.full((HIST_H, HIST_W, 3), 30, dtype=np.uint8)
    mid_x = HIST_W // 2

    cv2.line(canvas, (mid_x, 24), (mid_x, HIST_H - 24), (90, 90, 90), 1)
    _draw_stats_column(
        canvas,
        "Local noise",
        [
            ("sigma_corrected", local_stats.get("sigma_corrected")),
            ("rho1", local_stats.get("rho1")),
            ("rho2", local_stats.get("rho2")),
            ("sigma_diff", local_stats.get("sigma_diff")),
            ("sigma_residual", global_data.sigma_residual),
        ],
        24,
    )
    _draw_stats_column(
        canvas,
        "Global noise",
        [
            (
                "sigma",
                global_stats.get("sigma_corrected_median"),
            ),
            ("sigma min", global_stats.get("sigma_corrected_min")),
            (
                "sigma max",
                global_stats.get("sigma_corrected_max"),
            ),
            ("rho1", global_stats.get("rho1_median")),
            ("rho1 min", global_stats.get("rho1_min")),
            ("rho1 max", global_stats.get("rho1_max")),
            ("rho2", global_stats.get("rho2_median")),
            ("rho2 min", global_stats.get("rho2_min")),
            ("rho2 max", global_stats.get("rho2_max")),
        ],
        mid_x + 24,
    )

    return canvas


def _zoom_window(w: int, h: int) -> tuple[int, int, int, int]:
    """Return (x1, y1, crop_w, crop_h) of the current zoom crop in display-image pixels."""
    crop_w = max(1, int(w * global_data.zoom_scale))
    crop_h = max(1, int(h * global_data.zoom_scale))
    cx = int(global_data.zoom_cx * w)
    cy = int(global_data.zoom_cy * h)
    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    return x1, y1, crop_w, crop_h


def _apply_zoom(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Crop *img* (DISPLAY_PX×DISPLAY_PX) around the current zoom center and
    rescale the crop back to DISPLAY_PX×DISPLAY_PX."""
    if global_data.zoom_scale >= 1.0:
        return img
    h, w = img.shape[:2]
    x1, y1, crop_w, crop_h = _zoom_window(w, h)
    crop = img[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cast(
        NDArray[np.uint8], cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    )


def _overlay_roi(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply zoom, convert to BGR, then draw the ROI rectangle."""
    zoomed = _apply_zoom(img)
    out = cast(
        NDArray[np.uint8],
        cv2.cvtColor(zoomed, cv2.COLOR_GRAY2BGR) if zoomed.ndim == 2 else zoomed.copy(),
    )
    if global_data.roi is None or not global_data.display_slice.size:
        return out
    # Map ROI image coords → display coords, accounting for zoom
    img_h, img_w = global_data.display_slice.shape[:2]
    h, w = img.shape[:2]
    vx1, vy1, crop_w, crop_h = _zoom_window(w, h)

    x1_img, y1_img, x2_img, y2_img = global_data.roi
    # image-coord → display-coord
    x1_disp = (x1_img * w / img_w - vx1) * w / crop_w
    y1_disp = (y1_img * h / img_h - vy1) * h / crop_h
    x2_disp = (x2_img * w / img_w - vx1) * w / crop_w
    y2_disp = (y2_img * h / img_h - vy1) * h / crop_h
    cv2.rectangle(
        out,
        (int(x1_disp), int(y1_disp)),
        (int(x2_disp), int(y2_disp)),
        (0, 255, 0),
        2,
    )
    return out


def _redraw_images() -> None:
    """Re-show all 3 image windows with the current ROI overlay (no recompute)."""
    z = global_data.z_current
    zoom_suffix = ""
    if global_data.zoom_scale < 1.0 and global_data.display_slice.size:
        img_h, img_w = global_data.display_slice.shape[:2]
        cx_px = int(global_data.zoom_cx * img_w)
        cy_px = int(global_data.zoom_cy * img_h)
        zoom_suffix = f"  center=({cx_px}, {cy_px})"
    roi_suffix = ""
    if global_data.roi is not None:
        x1_img, y1_img, x2_img, y2_img = global_data.roi
        roi_w = x2_img - x1_img
        roi_h = y2_img - y1_img
        label = "px^2" if roi_w > 0 and roi_h > 0 else "px"
        area = roi_w * roi_h
        if area == 0:
            area = max(roi_w, roi_h)
        roi_suffix = f"  ROI={area} {label})"
    sub_label = (
        f"[z={z}, {z * global_data.voxel_size_mm:.2f}mm] {zoom_suffix}{roi_suffix}"
    )
    if global_data.display_slice.size:
        cv2.setWindowTitle(WIN_SLICE, f"Slice {sub_label}")
        cv2.imshow(WIN_SLICE, _overlay_roi(global_data.display_slice))
    if global_data.display_maxproj.size:
        cv2.setWindowTitle(WIN_MAXPROJ, f"Max Projection  {sub_label}")
        cv2.imshow(WIN_MAXPROJ, _overlay_roi(global_data.display_maxproj))
    if global_data.display_enh.size:
        cv2.setWindowTitle(WIN_ENH, f"Enhanced  {sub_label}")
        cv2.imshow(WIN_ENH, _overlay_roi(global_data.display_enh))


def _make_mouse_callback():
    """Return a single mouse callback shared across all 3 image windows.

    Left-drag  : draw ROI rectangle (shown on all windows live).
    Right-click: clear ROI.
    """

    def _cb(event: int, x: int, y: int, flags: int, param) -> None:
        if not global_data.display_slice.size:
            return
        img_h, img_w = global_data.display_slice.shape[:2]
        # Map screen position → image coords using the same crop window as _apply_zoom
        h = w = DISPLAY_PX
        vx1, vy1, crop_w, crop_h = _zoom_window(w, h)
        # screen pos → absolute display-image pos → image pos
        ix = round((vx1 + x * crop_w / w) * img_w / w)
        iy = round((vy1 + y * crop_h / h) * img_h / h)
        ix = max(0, min(img_w - 1, ix))
        iy = max(0, min(img_h - 1, iy))

        if event == cv2.EVENT_LBUTTONDOWN:
            _drag["active"] = True
            _drag["sx"] = ix
            _drag["sy"] = iy
        elif event == cv2.EVENT_MOUSEMOVE and _drag["active"]:
            global_data.roi = (
                min(_drag["sx"], ix),
                min(_drag["sy"], iy),
                max(_drag["sx"], ix),
                max(_drag["sy"], iy),
            )
            _redraw_images()
        elif event == cv2.EVENT_LBUTTONUP and _drag["active"]:
            _drag["active"] = False
            roi = (
                min(_drag["sx"], ix),
                min(_drag["sy"], iy),
                max(_drag["sx"], ix),
                max(_drag["sy"], iy),
            )
            global_data.roi = roi
            x1_img, y1_img, x2_img, y2_img = roi
            global_data.zoom_cx = ((x1_img + x2_img) / 2.0) / img_w
            global_data.zoom_cy = ((y1_img + y2_img) / 2.0) / img_h
            _redraw_images()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click: set zoom center; double right-click resets zoom
            global_data.zoom_cx = ix / img_w
            global_data.zoom_cy = iy / img_h
            _redraw_images()
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Scroll up = zoom in (smaller scale fraction), scroll down = zoom out
            step = 0.8 if flags > 0 else 1.25
            global_data.zoom_scale = float(
                np.clip(global_data.zoom_scale * step, 0.05, 1.0)
            )
            _redraw_images()

    return _cb


def update_windows(vol: NDArray[np.uint8]) -> None:
    """Update the slab data and render the windows

    Args:
        vol (NDArray[np.uint8]): _description_
    """
    global global_data
    global_data.update_slab_data()

    z_slice = global_data.vol[global_data.z_current].astype(np.float32)
    img = (z_slice - z_slice.min()) / max(z_slice.max() - z_slice.min(), 1e-6) * 255.0
    global_data.display_slice = cv2.resize(
        img.astype(np.uint8), (DISPLAY_PX, DISPLAY_PX)
    )

    max_proj = global_data.max_proj
    img = (
        (max_proj - max_proj.min()) / max(max_proj.max() - max_proj.min(), 1e-6) * 255.0
    )
    global_data.display_maxproj = cv2.resize(
        img.astype(np.uint8), (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_LINEAR
    )

    stats_wnd = render_noise_stats(global_data.slice_stats, global_data.global_stats)
    cv2.imshow(WIN_STATS, stats_wnd)
    _redraw_images()


app = typer.Typer()


def _load_config() -> dict:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def _resolve_data_path(filename: str | None, cfg: dict) -> Path:
    if filename:
        filename_path = Path(filename)
        if filename_path.is_absolute():
            return filename_path
        if filename_path.parent == Path("."):
            return PROJECT_ROOT / "Data" / filename_path
        return PROJECT_ROOT / filename_path

    configured_path = cfg.get("paths", {}).get("vol_data")
    if not configured_path:
        typer.echo(
            "Error: no filename provided and paths.vol_data is missing.", err=True
        )
        raise typer.Exit(1)

    data_path = Path(configured_path)
    if data_path.is_absolute():
        return data_path

    candidates = [
        PROJECT_ROOT / data_path,
        Path.cwd() / data_path,
        Path(__file__).resolve().parent / data_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _init_globals(vol: NDArray[np.uint8], cfg: dict) -> None:
    analysis_cfg = cfg["analysis"]
    slab_thickness = analysis_cfg.get(
        "slab_thickness", analysis_cfg.get("slice_thickness", 11)
    )
    dead_zone = analysis_cfg.get("dead_zone_scale", 3.0)
    z0 = analysis_cfg.get("vol_top", 0)
    z1 = analysis_cfg.get("vol_bottom", vol.shape[0] - 1)
    noise_clip_val = cfg["noise"].get("clip_scale", 2.5)
    global global_data
    global_data = GlobalData(
        vol=vol,
        z_min=z0,
        z_max=z1,
        slab_thickness=slab_thickness,
        dead_zone_scale=dead_zone,
        noise_clip_scale=noise_clip_val,
    )

    typer.echo("Computing grayscale landmarks...")
    air_grayvalue, max_grayvalue = estimate_grayscale_range(vol, slice_step=200)
    metal_threshold = estimate_metal_threshold(vol, air_grayvalue, slice_step=200)

    typer.echo("Estimating global noise...")
    slice_levels = list(range(z0, z1 + 1, max(1, (z1 - z0) // 20)))
    if z1 - 1 not in slice_levels:
        slice_levels.append(z1 - 1)
    global_stats = estimate_volume_slice_stats(
        vol[z0:z1], slice_levels, global_data.noise_clip_scale, metal_threshold
    )

    global_stats["air_grayvalue"] = air_grayvalue
    global_stats["metal_threshold"] = metal_threshold
    global_stats["max_grayvalue"] = max_grayvalue
    global_stats["min_grayvalue"] = air_grayvalue
    typer.echo(
        f"Air grayscale: {global_stats['air_grayvalue']}\n"
        f"Max grayscale: {global_stats['max_grayvalue']}\n"
        f"Metal threshold: {global_stats['metal_threshold']}"
    )

    global_data.global_stats = global_stats


@app.command()
def main(
    filename: str | None = typer.Argument(
        None,
        help="Optional volume filename inside ./Data/. Defaults to paths.vol_data in config.toml.",
    ),
) -> None:
    cfg = _load_config()
    data_path = _resolve_data_path(filename, cfg)
    if not data_path.exists():
        typer.echo(
            f"Error: {data_path} not found. Check the filename or paths.vol_data in config.toml.",
            err=True,
        )
        raise typer.Exit(1)

    vol: NDArray[np.uint8] = load_volume(data_path)  # type: ignore[assignment]
    _init_globals(vol, cfg)

    state: dict[str, int] = {"z": global_data.z_min}

    for win in (WIN_MAXPROJ, WIN_SLICE, WIN_ENH, WIN_STATS, WIN_CONTROLS):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    typer.echo(
        "Use the slider or A/D keys to navigate through slices. Press Q to quit."
    )

    def on_trackbar(pos: int) -> None:
        new_z = pos + global_data.z_min
        state["z"] = new_z

        # Update the global slice location which updates the slice stats.
        global_data.update_slice_location(new_z)

        # Update the windows
        update_windows(vol)

    _cb = _make_mouse_callback()
    for win in (WIN_SLICE, WIN_MAXPROJ, WIN_ENH):
        cv2.setMouseCallback(win, _cb)

    cv2.imshow(WIN_CONTROLS, render_controls_panel())
    cv2.resizeWindow(WIN_CONTROLS, CTRL_W, CTRL_H)
    cv2.setWindowProperty(WIN_CONTROLS, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(WIN_CONTROLS, _make_controls_callback(vol, state))

    z0 = global_data.z_min
    z1 = global_data.z_max
    cv2.createTrackbar("z", WIN_SLICE, 0, z1 - z0, on_trackbar)

    cv2.moveWindow(WIN_STATS, 1500, 50)
    cv2.moveWindow(WIN_ENH, 50, 50)

    while True:
        key: int = cv2.waitKey(20)
        if key in KEYS_QUIT:
            break
        elif key in KEYS_RIGHT:
            cv2.setTrackbarPos("z", WIN_SLICE, min(state["z"] + 1, z1) - z0)
        elif key in KEYS_LEFT:
            cv2.setTrackbarPos("z", WIN_SLICE, max(state["z"] - 1, z0) - z0)
        elif key in KEYS_ZOOM_IN:
            global_data.zoom_scale = float(
                np.clip(global_data.zoom_scale * 0.8, 0.05, 1.0)
            )
            _redraw_images()
        elif key in KEYS_ZOOM_OUT:
            global_data.zoom_scale = float(
                np.clip(global_data.zoom_scale * 1.25, 0.05, 1.0)
            )
            _redraw_images()
        elif key in KEYS_ZOOM_RESET:
            global_data.zoom_scale = 1.0
            _redraw_images()
        elif key in (ord("r"), ord("R")):
            global_data.roi = None
            _redraw_images()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
