#!/usr/bin/env python3
# TODO:
#  - We need to get the enhancement to display properly.
#  - Simply run a check on the residual image at the current location.
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import skimage
import typer
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
sys.path.insert(0, str(PROJECT_ROOT))

from cfg_data import AnalysisBase  # noqa: E402
from enhance import softmax_projection  # noqa: E402
from metal import (  # noqa: E402
    estimate_metal_threshold,
    extract_metal_mask,
)
from noise import (  # noqa: E402
    estimate_slice_stats,
    estimate_volume_slice_stats,
    leavein_median_noise_scale_penta,
)
from utils import estimate_grayscale_range, identify_cylindrical_support  # noqa: E402


@dataclass
class GlobalData(AnalysisBase):
    z_current: int = 0

    use_max: bool = False
    slab_thickness_dirty: bool = (
        True  # whether the slab thickness has changed since last slab data update
    )

    # softmax parameters
    softmax_beta: float = 2.0
    softmax_window: int = 2

    # enhancement parameters
    sigma_residual: float = float("nan")
    contrast_scale: float = 1.0

    # Number of surviving pixels
    auto_scroll: bool = False
    n_particle_pixels: int = 1

    # Rolling window cache: z -> (residual_image, sigma_residual) for z_current-1..+1
    _residual_cache: dict = field(default_factory=dict)
    _cache_params: tuple = field(
        default_factory=tuple
    )  # (slab_thickness, baseline_method)

    global_stats: dict = field(default_factory=dict)
    slice_stats: dict = field(default_factory=dict)

    residual_image: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )

    slice_metal_mask: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    slab_metal_mask: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=bool)
    )

    def __init__(
        self, cfg_path: Path = CONFIG_PATH, vol_file_name: str | None = None
    ) -> None:
        super().__init__(cfg_path=cfg_path, vol_file_name=vol_file_name)

        shape = self.vol.shape
        hw = shape[1:]
        self.slice_stats: dict = {}
        self.global_stats: dict = {}
        self.residual_image: NDArray[np.float32] = np.zeros(hw, dtype=np.float32)
        self.slice_metal_mask: NDArray[np.bool_] = np.zeros(hw, dtype=bool)
        self.slab_metal_mask: NDArray[np.bool_] = np.zeros(hw, dtype=bool)
        self._residual_cache: dict[int, tuple[NDArray[np.float32], float]] = {}
        self._cache_params: tuple = ()

    @AnalysisBase.slab_thickness.setter  # type: ignore
    def slab_thickness(self, value: int) -> None:
        if value < 3 or value > 61 or value % 2 == 0:
            raise ValueError("slab_thickness must be an odd integer between 3 and 61")

        cur_val = self.slab_thickness
        if cur_val != value:
            self.slab_thickness_dirty = True
        AnalysisBase.slab_thickness.fset(self, value)  # type: ignore

    def get_current_slice(self) -> NDArray[np.uint8]:
        return self.vol[self.z_current]

    def update_slice_location(self, z: int) -> None:
        """
        Set the current slice index and update dependent slice state to extract the
        slice-level statistics.

        args:
            z: int - the new slice index

        side effects:
            Updates the current slice index and recalculates slice-level statistics.
        """
        if self.z_current != z and z >= self.z_min and z <= self.z_max:
            self.z_current = z
            # update the slice's metal mask
            max_proj = self.vol[self.z_current - 2 : self.z_current + 3].max(axis=0)
            mm = extract_metal_mask(
                max_proj,
                self.metal_threshold,
                min_area=self.metal_min_area,
                margin=self.metal_grayscale_margin,
            )
            self.slice_metal_mask = cast(NDArray[np.bool_], mm)

            # Update the slice-level noise statistics
            self._update_slice_stats()

            # update the slab data
            self.update_slab_data()

    def get_sigma_residual(self) -> float:
        sigma = float(self.slice_stats["sigma_corrected"])
        rho1 = float(self.slice_stats["rho1"])
        rho2 = float(self.slice_stats["rho2"])
        s_thick = self.slab_thickness

        leave_out = True
        if self.baseline_method == "mean":
            num = np.sqrt(
                s_thick**2 - s_thick * (1 + 2 * (rho1 + rho2)) - 2 * (rho1 + 2 * rho2)
            )
            if leave_out:
                noise_scale = num / (s_thick - 1.0)
            else:
                noise_scale = num / s_thick
        elif self.baseline_method == "median":
            noise_scale = leavein_median_noise_scale_penta(s_thick, rho1, rho2)
        else:
            raise ValueError(f"unknown baseline method: {self.baseline_method}")

        return noise_scale * sigma

    def _compute_residual_at(self, z: int) -> tuple[NDArray[np.float32], float]:
        """Compute residual image and sigma_residual for an arbitrary slice z.

        Self-contained: does not read or write z_current or any slab-level state,
        so it is safe to call for z_current±1 without disturbing existing state.
        """
        slab_half = self.slab_thickness // 2
        z_slice = self.vol[z].astype(np.float32)

        # Metal mask from a 5-slice window centred on z (mirrors _update_slice_stats)
        local_max = self.vol[z - 2 : z + 3].max(axis=0)
        metal_mask = cast(
            NDArray[np.bool_],
            extract_metal_mask(
                local_max,
                self.metal_threshold,
                min_area=self.metal_min_area,
                margin=self.metal_grayscale_margin,
            ),
        )

        # Slice stats for noise estimation
        cyl_support = identify_cylindrical_support(metal_mask)
        valid_mask = ~metal_mask & cyl_support
        stats = estimate_slice_stats(
            self.vol[z - 2 : z + 3],
            valid_mask=valid_mask,
            trunc_k=self.noise_clip_scale,
        )

        # sigma_residual (mirrors get_sigma_residual logic)
        sigma = float(stats["sigma_corrected"])
        rho1 = float(stats["rho1"])
        rho2 = float(stats["rho2"])
        s_thick = self.slab_thickness
        if self.baseline_method == "mean":
            num = np.sqrt(
                s_thick**2 - s_thick * (1 + 2 * (rho1 + rho2)) - 2 * (rho1 + 2 * rho2)
            )
            sigma_res = float(num / (s_thick - 1.0)) * sigma
        elif self.baseline_method == "median":
            sigma_res = leavein_median_noise_scale_penta(s_thick, rho1, rho2) * sigma
        else:
            raise ValueError(f"unknown baseline method: {self.baseline_method}")

        # Slab baseline (leave-one-out for mean, full median otherwise)
        slab = self.vol[z - slab_half : z + slab_half + 1]
        if self.baseline_method == "median":
            baseline = np.median(slab, axis=0).astype(np.float32)
        else:
            baseline = (slab.sum(axis=0).astype(np.float32) - z_slice) / (s_thick - 1)

        # Support mask from the slab metal mask
        slab_mm = cast(
            NDArray[np.bool_],
            extract_metal_mask(
                slab.max(axis=0),
                self.metal_threshold,
                min_area=self.metal_min_area,
                margin=self.metal_grayscale_margin,
            ),
        )
        support = identify_cylindrical_support(slab_mm)
        support &= ~slab_mm

        residual = z_slice - baseline
        # Update suppor to include only positive residuals (particles are bright)
        support &= residual > 0

        residual[~support] = 0

        return residual / sigma_res, sigma_res

    def _update_slice_stats(self) -> None:
        """
        Called when teh z-location changes to update the metal mask and slice statistics
        """
        z = self.z_current

        # Don't clip here, just use the fact that we are limiting the range of
        # z_current to the top and bottom of the volume and so have plenty of
        # margin to work with above and below the current level.
        _zmin = z - 2
        _zmax = z + 2

        cyl_support = identify_cylindrical_support(self.slice_metal_mask)
        valid_mask = ~self.slice_metal_mask & cyl_support
        self.slice_stats = estimate_slice_stats(
            self.vol[_zmin : _zmax + 1],
            valid_mask=valid_mask,
            trunc_k=self.noise_clip_scale,
        )

    def get_clipped_slab_range(self) -> tuple[int, int]:
        # z_min = max(self.z_current - self.slab_thickness // 2, self.z_min)
        # z_max = min(self.z_current + self.slab_thickness // 2, self.z_max)
        z_min = self.z_current - self.slab_thickness // 2
        z_max = self.z_current + self.slab_thickness // 2
        """if z_min == self.z_min:
            z_max = z_min + self.slab_thickness
        if z_max == self.z_max:
            z_min = z_max - self.slab_thickness"""

        return z_min, z_max

    def update_slab_data(self) -> None:
        """The slab has changed - update the relevant information"""
        # Update the max projection (sets self.max_proj and self.display_maxproj)
        compute_max_projection(self.vol, *self.slab_range)
        max_proj = self.vol[self.slab_range[0] : self.slab_range[1] + 1].max(axis=0)
        mm = extract_metal_mask(
            max_proj,
            self.metal_threshold,
            min_area=self.metal_min_area,
            margin=self.metal_grayscale_margin,
        )
        self.slab_metal_mask = cast(NDArray[np.bool_], mm)

        # Invalidate the rolling cache when slab thickness or baseline method changes
        current_params = (self.slab_thickness, self.baseline_method)
        if current_params != self._cache_params:
            self._residual_cache.clear()
            self._cache_params = current_params

        # Populate the length-3 rolling window, reusing any already-cached neighbours
        for z in (self.z_current - 1, self.z_current, self.z_current + 1):
            if z not in self._residual_cache and self.z_min <= z <= self.z_max:
                self._residual_cache[z] = self._compute_residual_at(z)

        # Evict entries that have scrolled out of the window
        window = {self.z_current - 1, self.z_current, self.z_current + 1}
        for z in [k for k in self._residual_cache if k not in window]:
            del self._residual_cache[z]

        # Expose the current slice's values as direct attributes
        self.residual_image, self.sigma_residual = self._residual_cache[self.z_current]

        create_enhancement()  # sets self.display_enh as a side effect
        self.slab_thickness_dirty = False

    max_proj: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )

    res_img: NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )

    # Cached display-resolution (DISPLAY_PX × DISPLAY_PX) grayscale images for fast ROI redraws
    display_slice: NDArray[np.uint8] = field(
        default_factory=lambda: np.array([], dtype=np.uint8)
    )
    display_maxproj: NDArray[np.uint8] | NDArray[np.float32] = field(
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
        """ "Return the slab range (z_min, z_max) for the current slice and slab thickness."""
        z_min = self.z_current - self.slab_thickness // 2
        z_max = self.z_current + self.slab_thickness // 2  # inclusive
        return z_min, z_max

    def compute_residual_image(self) -> None:
        """
        Compute the residual image from the current slice and slab data.

        The residual image is created by differencing the current slice with the slab mean or median, depending on the baseline method.

        The residual image is scaled by the local noise estimate to produce a contrast image that can be thresholded to identify particles.
        """
        z_slice = self.get_current_slice()

        slab = self.vol[self.slab_range[0] : self.slab_range[1] + 1]
        if self.baseline_method == "median":
            baseline = np.median(slab, axis=0).astype(np.float32)
        elif self.baseline_method == "mean":
            baseline = (slab.sum(axis=0).astype(np.float32) - z_slice) / (
                self.slab_thickness - 1
            )
        else:
            raise ValueError(f"unknown baseline method: {self.baseline_method}")

        residual = z_slice - baseline

        # Zero residual in the metal region and can exterior
        support_mask = identify_cylindrical_support(self.slab_metal_mask)
        support_mask &= ~self.slab_metal_mask
        residual[~support_mask] = 0

        self.residual_image = residual / self.get_sigma_residual()

    def record_particles(
        self, mask: NDArray[np.bool_], residual: NDArray[np.float32], slice: int
    ) -> None:
        """
        Record the number of particles in the current slice and update the global statistics.

        Args:
            mask: NDArray[np.bool_] - boolean mask of particle locations
            residual: NDArray[np.float32] - residual image for the current slice
        """
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )

        vol_path = Path(self.cfg["paths"]["vol_data_path"])
        out_dir = vol_path.parent / "particles"
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath: Path = out_dir / (vol_path.stem + "_particles.csv")
        fpath_exists = fpath.exists()
        # Load existing (slice, x, y) keys to avoid duplicate writes
        if not hasattr(self, "_recorded_particles"):
            self._recorded_particles: set[tuple[int, int, int]] = set()
            if fpath_exists:
                with open(fpath, "r", newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # skip header
                    for row in reader:
                        if len(row) >= 6:
                            self._recorded_particles.add(
                                (int(row[0]), int(float(row[4])), int(float(row[5])))
                            )

        new_rows: list[list] = []
        for n in range(1, n_labels):
            area = stats[n, cv2.CC_STAT_AREA]
            mean_contrast = residual[labels == n].mean()
            max_contrast = residual[labels == n].max()
            x, y = centroids[n]
            x = round(x)
            y = round(y)
            key = (slice, x, y)
            if key in self._recorded_particles:
                continue
            self._recorded_particles.add(key)
            grayvalue = self.vol[slice, y, x]
            new_rows.append([slice, grayvalue, area, mean_contrast, max_contrast, x, y])

        if not new_rows:
            return

        with open(fpath, "a", newline="") as f:
            writer = csv.writer(f)
            if not fpath_exists:
                writer.writerow(
                    [
                        "slice",
                        "grayvalue",
                        "area",
                        "mean_contrast",
                        "max_contrast",
                        "cx",
                        "cy",
                    ]
                )
            writer.writerows(new_rows)


global_data: GlobalData

# Transient drag state for ROI drawing
_drag: dict = {"active": False, "sx": 0, "sy": 0, "win": None}

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

CTRL_W, CTRL_H = 320, 625
CTRL_BTN = {"use_max": (10, 10, 300, 55)}  # (x1, y1, x2, y2)
CTRL_SCROLL_NEXT_BTN = (10, 525, 310, 570)  # "Scroll to next particle" button rect
CTRL_QUIT_BTN = (10, 575, 310, 620)  # Quit button rect

# Float adjusters: attr -> (label, step, min, max, y_top)
S = -85
CTRL_FLOAT = {
    # label, step_size, min, max, y_top
    "slab_thickness": ("Slab thickness", 2, 3, 61, 120),
    "dead_zone_scale": ("Dead zone", 0.1, 0.0, 10.0, 170),
    "contrast_scale": ("Contrast", 0.2, 1.0, 20.0, 220),
    "metal_min_area": ("Metal min area", 10, 0, 2000, 270),
    "metal_grayscale_margin": ("Metal margin", 1, 0, 40, 320),
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

# Attributes whose change only requires recomputing the enhancement image
# (not the slab data or residual image).
ENH_ATTRS = {"dead_zone_scale", "contrast_scale", "area_threshold"}
MAX_PROJ_ATTRS = {"softmax_beta", "softmax_window"}
# endregion - GUI constants and utilities


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

    # Scroll to next particle button
    bx1, by1, bx2, by2 = CTRL_SCROLL_NEXT_BTN
    fill = (0, 160, 80) if global_data.auto_scroll else (80, 80, 80)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), fill, -1)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
    cv2.putText(
        canvas,
        "Scroll to next particle",
        (bx1 + 30, (by1 + by2) // 2 + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    # Quit button
    bx1, by1, bx2, by2 = CTRL_QUIT_BTN
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (40, 40, 180), -1)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
    cv2.putText(
        canvas,
        "Quit",
        (bx1 + 120, (by1 + by2) // 2 + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    return canvas


def _make_controls_callback(vol, state):
    def _cb(event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Quit button
        bx1, by1, bx2, by2 = CTRL_QUIT_BTN
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            cv2.destroyAllWindows()
            sys.exit(0)

        # Scroll to next particle button
        bx1, by1, bx2, by2 = CTRL_SCROLL_NEXT_BTN
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            global_data.auto_scroll = not global_data.auto_scroll
            cv2.imshow(WIN_CONTROLS, render_controls_panel())
            return

        # Toggle buttons
        for attr, (bx1, by1, bx2, by2) in CTRL_BTN.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                setattr(global_data, attr, not getattr(global_data, attr))
                cv2.imshow(WIN_CONTROLS, render_controls_panel())
                update_windows()
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
                    if attr in ENH_ATTRS:
                        create_enhancement()
                    if attr in MAX_PROJ_ATTRS:
                        compute_max_projection(global_data.vol, *global_data.slab_range)
                    cv2.imshow(WIN_CONTROLS, render_controls_panel())
                    update_windows()
                    return

        # Cycle controls
        for attr, (label, options, y_top) in CTRL_CYCLE.items():
            bx1, by1, bx2, by2 = 10, y_top, CTRL_W - 10, y_top + 45
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                cur_cycle = getattr(global_data, attr)
                idx = options.index(cur_cycle) if cur_cycle in options else 0
                setattr(global_data, attr, options[(idx + 1) % len(options)])
                cv2.imshow(WIN_CONTROLS, render_controls_panel())
                update_windows()
                return

    return _cb


# We need to ascertain the size and contrast of the underlying particles.

_K3x3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)


def create_enhancement() -> None:
    """
    Here we filter the residual image to create a rendering by:
      - eliminating residuals that are below the dead zone threhold
      - eliminating connected regions of residuals that are below the area threshold
      - applying a mild blur to smooth out the speckle noise and make the particles more visible
    """

    residual_cache = global_data._residual_cache
    residual_0, sigma_0 = residual_cache.get(global_data.z_current - 1, (None, None))
    residual_1, sigma_1 = residual_cache.get(global_data.z_current, (None, None))
    residual_2, sigma_2 = residual_cache.get(global_data.z_current + 1, (None, None))

    assert residual_1 is not None, "current residual not in cache"

    # Create a mask where pixels in the current slice survive the dead zon.
    dead_zone_thresh = global_data.dead_zone_scale

    # Compute the mask for which the current slice is above the threshold dead zone. Clean the mask by removing small connected components below the area threshold.
    mask_cur = residual_1 > dead_zone_thresh

    # Now suppress small regions
    labels = cv2.connectedComponents(mask_cur.astype(np.uint8), connectivity=8)[1]
    if global_data.area_threshold > 1:
        # Compute all component areas in one pass, then build a single mask
        # of small regions to zero out.
        areas = np.bincount(labels.ravel())
        small_mask = areas < global_data.area_threshold
        small_mask[0] = False  # background label is never "small"
        mask_cur[small_mask[labels]] = 0
        labels[small_mask[labels]] = 0  # zero out small regions in the label image

    # Now create the mask from the adjacent slices. Dilate so that we get diagonal connections as well.
    mask_adj0 = np.ones_like(mask_cur, dtype=bool)
    mask_adj2 = np.ones_like(mask_cur, dtype=bool)
    if residual_0 is not None:
        mask_adj0 &= residual_0 > dead_zone_thresh
        mask_adj0 = skimage.morphology.dilation(mask_adj0, _K3x3)
    if residual_2 is not None:
        mask_adj2 &= residual_2 > dead_zone_thresh
        mask_adj2 = skimage.morphology.dilation(mask_adj2, _K3x3)

    # Use the adjacent masks to filter the current mask, leaving only those regions in the current slice that are connected to one of the adjacent masks.
    intersecting = np.unique(labels[mask_adj0 | mask_adj2])
    intersecting = intersecting[intersecting > 0]  # remove background label
    mask_cur = np.isin(labels, intersecting)

    img = cast(
        NDArray[np.float32], np.where(mask_cur, residual_1, 0.0).astype(np.float32)
    )
    blur_size = 5
    if blur_size > 1:
        scale = img.max()
        img = cast(
            NDArray[np.float32], cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        )
        scale /= max(img.max(), 1e-6) * dead_zone_thresh
        img *= np.float32(scale)
    img = cast(
        NDArray[np.float32],
        cv2.resize(img, (DISPLAY_PX, DISPLAY_PX), interpolation=ENH_INTERP),
    )
    # Normalise so the strongest surviving residual maps to 255; background stays 0.
    mx = float(img.max())
    if mx > 1e-6:
        pass  # img = img / mx * 255.0

    n_residual_pixels = int(np.count_nonzero(mask_cur))
    global_data.n_particle_pixels = n_residual_pixels
    if n_residual_pixels > 0:
        global_data.record_particles(mask_cur, residual_1, global_data.z_current)
    global_data.display_enh = cast(NDArray[np.float32], img)  # .astype(np.uint8)


def compute_max_projection(
    vol: NDArray[np.uint8],
    z_min: int,
    z_max: int,
) -> None:
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

    global_data.max_proj = _max_proj
    img = (
        (_max_proj - _max_proj.min())
        / max(_max_proj.max() - _max_proj.min(), 1e-6)
        * 255.0
    )
    global_data.display_maxproj = cv2.resize(
        img.astype(np.uint8), (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_LINEAR
    )


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
            ("sigma_residual", global_data.get_sigma_residual()),
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

    # Always compute the zoom/ROI centre in image-pixel coords
    cx_px = cy_px = 0
    if global_data.display_slice.size:
        img_h, img_w = global_data.display_slice.shape[:2]
        cx_px = int(global_data.zoom_cx * img_w)
        cy_px = int(global_data.zoom_cy * img_h)

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
    cx, cy = (
        round(cx_px * global_data.vol.shape[2] / img_w),
        round(cy_px * global_data.vol.shape[1] / img_h),
    )
    sub_label = (
        f"[z={z}, {z * global_data.voxel_size_mm:.2f}mm]"
        f"  center=({cx}, {cy}){roi_suffix}"
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
    if cv2.getWindowProperty(WIN_STATS, cv2.WND_PROP_VISIBLE) >= 0:
        cv2.setWindowTitle(WIN_STATS, f"Noise Statistics  {sub_label}")


def _make_mouse_callback():
    """Return a single mouse callback shared across all 3 image windows.

    Left-drag  : draw ROI rectangle (shown on all windows live).
    Right-click: set zoom center.
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
            _drag["win"] = param  # remember which window started the drag
        elif event == cv2.EVENT_LBUTTONUP:
            if not _drag["active"]:
                return
            _drag["active"] = False
            _drag["win"] = None
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
        elif event == cv2.EVENT_MOUSEMOVE and _drag["active"]:
            # Only update the ROI from the window that started the drag
            if _drag["win"] is not None and param != _drag["win"]:
                return
            global_data.roi = (
                min(_drag["sx"], ix),
                min(_drag["sy"], iy),
                max(_drag["sx"], ix),
                max(_drag["sy"], iy),
            )
            _redraw_images()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click: set zoom center
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


def update_windows() -> None:
    """Update the slab data and render the windows

    Args:
        vol (NDArray[np.uint8]): _description_
    """
    global global_data
    if global_data.slab_thickness_dirty:
        global_data.update_slab_data()

    z_slice = global_data.get_current_slice()
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


def _init_globals(vol_file_name: str | None = None) -> None:
    global global_data
    global_data = GlobalData(vol_file_name=vol_file_name)

    typer.echo("Volume shape: %s" % (global_data.vol.shape,))
    typer.echo("Computing grayscale landmarks...")
    z0, z1 = global_data.z_min, global_data.z_max
    air_grayvalue, core_grayvalue, max_grayvalue = estimate_grayscale_range(
        global_data.vol[z0 : z1 + 1], slice_step=global_data.grayscale_slice_step
    )
    metal_threshold = estimate_metal_threshold(
        global_data.vol[z0 : z1 + 1],
        air_grayvalue,
        slice_step=global_data.grayscale_slice_step,
    )

    typer.echo("Estimating global noise...")
    slice_levels = list(range(z0, z1 + 1, max(1, (z1 - z0) // 20)))
    if z1 - 1 not in slice_levels:
        slice_levels.append(z1 - 1)
    global_stats = estimate_volume_slice_stats(
        global_data.vol[z0:z1],
        slice_levels,
        global_data.noise_clip_scale,
        metal_threshold=metal_threshold,
    )

    global_stats["air_grayvalue"] = air_grayvalue
    global_stats["metal_threshold"] = metal_threshold
    global_stats["max_grayvalue"] = max_grayvalue
    global_stats["min_grayvalue"] = air_grayvalue
    global_stats["core_grayvalue"] = core_grayvalue
    typer.echo(
        f"Air grayscale, core grayscale, metal threshold, max grayscale: {global_stats['air_grayvalue']}, "
        f"{global_stats['core_grayvalue']}, "
        f"{global_stats['metal_threshold']}, "
        f"{global_stats['max_grayvalue']}"
    )

    global_data.global_stats = global_stats


@app.command()
def main(
    filename: str | None = typer.Argument(
        None,
        help="Optional volume filename inside ./Data/. Defaults to paths.vol_data_path in config.toml.",
    ),
) -> None:
    _init_globals(filename)
    # global_data.z_min = 3554
    state: dict[str, int] = {"z": global_data.z_min}

    for win in (WIN_MAXPROJ, WIN_SLICE, WIN_ENH, WIN_STATS, WIN_CONTROLS):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    typer.echo(
        "Use the slider or A/D keys to navigate through slices. Press Q to quit."
    )

    _auto_scrolling = False

    def on_trackbar(pos: int) -> None:
        nonlocal _auto_scrolling
        if _auto_scrolling:
            return  # ignore re-entrant call from setTrackbarPos

        new_z = pos + global_data.z_min
        state["z"] = new_z

        # Update the global slice location which updates the slice stats.
        global_data.update_slice_location(new_z)

        if global_data.auto_scroll:
            # Advance through slices until we find one with particles
            typer.echo(f"Auto-scrolling: starting at z={new_z}...")
            while global_data.n_particle_pixels <= 100000 and new_z < global_data.z_max:
                new_z += 1
                typer.echo(f"    ...Auto-scrolling: z={new_z}")
                global_data.update_slice_location(new_z)
            state["z"] = new_z
            typer.echo(
                f"Auto-scrolling stopped at z={new_z} with n_particle_pixels={global_data.n_particle_pixels}"
            )
            # Sync the trackbar to the final position; suppress the re-entrant callback
            _auto_scrolling = True
            cv2.setTrackbarPos("z", WIN_SLICE, new_z - global_data.z_min)
            _auto_scrolling = False

        # Update the windows
        update_windows()

    _cb = _make_mouse_callback()
    for win in (WIN_SLICE, WIN_MAXPROJ, WIN_ENH):
        cv2.setMouseCallback(win, _cb, param=win)

    cv2.imshow(WIN_CONTROLS, render_controls_panel())
    cv2.resizeWindow(WIN_CONTROLS, CTRL_W, CTRL_H)
    cv2.setWindowProperty(WIN_CONTROLS, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(WIN_CONTROLS, _make_controls_callback(global_data.vol, state))

    z0 = global_data.z_min
    z1 = global_data.z_max
    cv2.createTrackbar("z", WIN_SLICE, 0, z1 - z0, on_trackbar)

    cv2.moveWindow(WIN_STATS, 1500, 50)
    cv2.moveWindow(WIN_ENH, 50, 50)

    while True:
        key: int = cv2.waitKey(20)
        # Reset stuck drag state on any key press (handles missed LBUTTONUP)
        if key != -1 and _drag["active"]:
            _drag["active"] = False
            _drag["win"] = None
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
