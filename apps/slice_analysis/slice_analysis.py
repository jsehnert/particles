#!/usr/bin/env python3
# **Notes**
# - structural assymetry induces FP
# -   metal tabs
# -   a bend in the abode foil
# -   unconstrained anodes in the core and near the periphery (can wall)
# -   can tapering from the can wall below crimp region
# TODO:
#  -
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
PARTICLES_FOUND_PATH = PROJECT_ROOT / "Data" / "particles_found.csv"
PARTICLES_FOUND_FIELDS = (
    "volume_name",
    "z",
    "y",
    "x",
    "gray_level",
    "noise_level",
    "particle_volume",
)
LEGACY_PARTICLES_FOUND_FIELDS = {
    ("volume_name", "x", "y", "z", "gray_level"),
    (
        "volume_name",
        "x",
        "y",
        "z",
        "gray_level",
        "noise_level",
        "particle_volume",
    ),
}
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
from surfacing.particle_extractor import ParticleExtractor  # noqa: E402
from utils import estimate_grayscale_range, identify_cylindrical_support  # noqa: E402


def _load_particles_found() -> list[dict[str, str]]:
    """Load previously recorded particle locations, creating the CSV if needed."""
    PARTICLES_FOUND_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PARTICLES_FOUND_PATH.exists():
        with PARTICLES_FOUND_PATH.open("w", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=PARTICLES_FOUND_FIELDS).writeheader()
        return []

    with PARTICLES_FOUND_PATH.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [
            {field: row.get(field, "") for field in PARTICLES_FOUND_FIELDS}
            for row in reader
        ]

    if tuple(reader.fieldnames or ()) in LEGACY_PARTICLES_FOUND_FIELDS:
        with PARTICLES_FOUND_PATH.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=PARTICLES_FOUND_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    return rows


particles_found = _load_particles_found()
particles_found_keys = {
    (
        row.get("volume_name", ""),
        row.get("x", ""),
        row.get("y", ""),
        row.get("z", ""),
    )
    for row in particles_found
}


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

    def __init__(self, vol_index: int, cfg_path: Path = CONFIG_PATH) -> None:
        super().__init__(cfg_path=cfg_path, vol_index=vol_index, exp_index=0)

        if self.vol is None:
            raise ValueError(f"Volume data is not loaded for index {vol_index}")

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
        return (
            self.vol[self.z_current]
            if self.vol is not None
            else np.array([], dtype=np.uint8)
        )

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

    # Source-resolution grayscale images; Qt handles viewport scaling.
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

        vol_path = self.vol_data_path
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

# A single click creates a pending selection. It is written to the CSV only
# when the user confirms it with the ``r`` key.
pending_particle_z_max: int | None = None
pending_particle_y: int | None = None
pending_particle_x: int | None = None
last_click_info: dict[str, int | float] | None = None


def _clear_pending_particle() -> None:
    """Clear the particle selection awaiting confirmation."""
    global pending_particle_z_max, pending_particle_y, pending_particle_x
    pending_particle_z_max = None
    pending_particle_y = None
    pending_particle_x = None


def _record_particle_found(
    *,
    x: int,
    y: int,
    z: int,
    gray_level: int,
    noise_level: float,
    particle_volume: int,
) -> None:
    """Append one refined particle location to the project-level CSV.

    ``x`` is the image column and ``y`` is the image row, matching NumPy's
    ``volume[z, row, column]`` indexing convention. A refined coordinate is
    recorded at most once for each volume.
    """
    row = {
        "volume_name": global_data.vol_data_path.name,
        "x": str(x),
        "y": str(y),
        "z": str(z),
        "gray_level": str(gray_level),
        "noise_level": f"{noise_level:.3f}",
        "particle_volume": str(particle_volume),
    }
    key = (row["volume_name"], row["x"], row["y"], row["z"])
    if key in particles_found_keys:
        return

    with PARTICLES_FOUND_PATH.open("a", newline="") as csv_file:
        csv.DictWriter(csv_file, fieldnames=PARTICLES_FOUND_FIELDS).writerow(row)
    particles_found_keys.add(key)
    particles_found.append(row)


def _record_pending_particle() -> None:
    """Record the pending selection, or report that none is available."""
    if (
        pending_particle_z_max is None
        or pending_particle_y is None
        or pending_particle_x is None
        or last_click_info is None
    ):
        typer.echo("No pending particle selection.")
        return

    gray_level = int(
        global_data.vol[
            pending_particle_z_max,
            pending_particle_y,
            pending_particle_x,
        ]
    )
    _record_particle_found(
        x=pending_particle_x,
        y=pending_particle_y,
        z=pending_particle_z_max,
        gray_level=gray_level,
        noise_level=float(last_click_info["noise_level"]),
        particle_volume=int(last_click_info["particle_volume"]),
    )


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
WIN_STATS = "Info"
WIN_CONTROLS = "Controls"


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

# Attributes whose change only requires recomputing the enhancement image
# (not the slab data or residual image).
ENH_ATTRS = {"dead_zone_scale", "contrast_scale", "area_threshold"}
MAX_PROJ_ATTRS = {"softmax_beta", "softmax_window"}
# endregion - GUI constants and utilities


# We need to ascertain the size and contrast of the underlying particles.

_K3x3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

mask_core: NDArray | None = None


def create_enhancement() -> None:
    """
    Here we filter the residual image to create a rendering by:
      - eliminating residuals that are below the dead zone threshold
      - eliminating connected regions of residuals that are below the area threshold
      - applying a mild blur to smooth out the speckle noise and make the particles more visible
    """
    # Create a mask where pixels in the current slice survive the dead zon.
    dead_zone_thresh = global_data.dead_zone_scale

    residual_cache = global_data._residual_cache
    residual_0, sigma_0 = residual_cache.get(global_data.z_current - 1, (None, None))
    residual_1, sigma_1 = residual_cache.get(global_data.z_current, (None, None))
    residual_2, sigma_2 = residual_cache.get(global_data.z_current + 1, (None, None))
    assert residual_1 is not None, "current residual not in cache"

    # Carve out the core - which is right now a hack!
    global mask_core
    if mask_core is None:
        w, h = residual_1.shape[1], residual_1.shape[0]
        y, x = np.ogrid[:h, :w]
        radius = 100
        c_x, c_y = w / 2, h / 2
        mask_core = (x - c_x) ** 2 + (y - c_y) ** 2 >= radius**2

    if global_data.auto_scroll:
        mask_check = residual_1 > dead_zone_thresh
        mask_check &= mask_core
        if not mask_check.any():
            # Nothing in the current slice survives the dead zone threshold
            global_data.display_enh = np.zeros_like(residual_1, dtype=np.float32)
            global_data.n_particle_pixels = 0
            return

    if residual_0 is not None and residual_2 is not None:
        depth_counts = (residual_0 > dead_zone_thresh).astype(np.uint8)
        np.add(depth_counts, residual_1 > dead_zone_thresh, out=depth_counts)
        np.add(depth_counts, residual_2 > dead_zone_thresh, out=depth_counts)
        middle_counts = cv2.boxFilter(
            depth_counts,
            ddepth=-1,
            ksize=(3, 3),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        mask_cur = (
            middle_counts > global_data.area_threshold
        )  # Here we are interpreting the area threshold in 3D context
        mask_cur &= mask_core
        if not mask_cur.any():
            global_data.display_enh = np.zeros_like(residual_1, dtype=np.float32)
            global_data.n_particle_pixels = 0
            return
        # Project down the axis to pick up lateral shifting in adjacent regions
        img = np.maximum(residual_0, residual_1)
        np.maximum(img, residual_2, out=img)
        img *= mask_cur
    else:
        mask_cur = None
        img = None

    if img is None:
        # Compute the mask for which the current slice is above the threshold dead zone. Clean the mask by removing small connected components below the area threshold.
        mask_cur = (residual_1 > dead_zone_thresh) & mask_core

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

    blur_size = 3
    if blur_size > 1:
        scale = img.max()
        img = cast(
            NDArray[np.float32], cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        )
        scale /= max(img.max(), 1e-6) * dead_zone_thresh
        img *= np.float32(scale) * mask_cur
    img = cast(NDArray[np.float32], img)

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
    win = global_data.softmax_window
    z_min = max(z_min - win // 2, global_data.z_min)
    z_max = min(z_max + win // 2, global_data.z_max)
    if global_data.use_max:
        _max_proj = vol[z_min : z_max + 1].max(axis=0).astype(np.float32)
    else:
        _max_proj = softmax_projection(
            vol[z_min : z_max + 1],
            beta=global_data.softmax_beta,
            window=win,
        )

    global_data.max_proj = _max_proj
    img = (
        (_max_proj - _max_proj.min())
        / max(_max_proj.max() - _max_proj.min(), 1e-6)
        * 255.0
    )
    global_data.display_maxproj = img.astype(np.uint8)


def _redraw_images() -> None:
    """Refresh the Qt views without recomputing volume processing."""
    if qt_viewer is not None:
        qt_viewer.refresh()


def on_mouse_singleclick(
    *,
    event: int,
    display_x: int,
    display_y: int,
    image_x: int,
    image_y: int,
    flags: int,
    window_name: str,
    z: int,
) -> None:
    """Handle a stationary left-button click in one of the image windows.

    The clicked ``(y, x)`` coordinate is refined to the location of the maximum
    raw-volume value in the surrounding 3 × 3 × 3 neighborhood. The refined
    ``(z, y, x)`` is stored as a pending selection and is written only after
    the user presses ``r``. The displayed z-slice is not changed.

    Args:
        event: OpenCV mouse event code.
        display_x: Horizontal click coordinate in the displayed window.
        display_y: Vertical click coordinate in the displayed window.
        image_x: Horizontal coordinate mapped into the source image.
        image_y: Vertical coordinate mapped into the source image.
        flags: OpenCV mouse-event flags active at the time of the click.
        window_name: Name of the OpenCV window that received the click.
        z: Volume slice displayed when the click occurred.
    """
    global pending_particle_z_max, pending_particle_y, pending_particle_x
    global last_click_info

    if global_data.vol is None:
        raise RuntimeError("Volume data is not loaded.")

    clicked_x, clicked_y = image_x, image_y
    search_window = (5, 3, 3)
    z1, z2 = (
        max(0, z - search_window[0] // 2),
        min(global_data.vol.shape[0], z + search_window[0] // 2 + 1),
    )
    y1, y2 = (
        max(0, image_y - search_window[1] // 2),
        min(global_data.vol.shape[1], image_y + search_window[1] // 2 + 1),
    )
    x1, x2 = (
        max(0, image_x - search_window[2] // 2),
        min(global_data.vol.shape[2], image_x + search_window[2] // 2 + 1),
    )

    neighborhood = global_data.vol[z1:z2, y1:y2, x1:x2]
    local_z, local_y, local_x = np.unravel_index(
        int(np.argmax(neighborhood)), neighborhood.shape
    )
    max_z = z1 + int(local_z)
    image_y = y1 + int(local_y)
    image_x = x1 + int(local_x)

    # Keep the refined local maximum pending for confirmation with ``r``.
    pending_particle_z_max = max_z
    pending_particle_y = image_y
    pending_particle_x = image_x

    # Set the particle extractor's neighborhood size to abotu 250 um
    neighborhood_size_um = 250.0
    neighborhood_size_mm = neighborhood_size_um / 1000.0
    neighborhood_size_voxels = neighborhood_size_mm / global_data.voxel_size_mm
    neighborhood_size = np.rint(neighborhood_size_voxels).astype(int)
    # Ensure the neighborhood size is odd:
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1

    p_extractor = ParticleExtractor(
        global_data.vol, neighborhood_size=neighborhood_size
    )
    p_extractor.set_voxel_location((max_z, image_y, image_x))
    particle = p_extractor.particle
    noise_level = p_extractor.noise_estimate
    particle_volume = (particle > 0).sum()

    # The visual marker follows the exact location clicked by the user. It is
    # intentionally separate from the refined location stored above.
    global_data.roi = (clicked_x, clicked_y, clicked_x, clicked_y)

    global_data.zoom_cx = clicked_x / global_data.vol.shape[2]
    global_data.zoom_cy = clicked_y / global_data.vol.shape[1]
    clicked_gray_level = global_data.vol[z, clicked_y, clicked_x]
    max_gray_level = global_data.vol[max_z, image_y, image_x]
    last_click_info = {
        "display_x": display_x,
        "display_y": display_y,
        "clicked_z": z,
        "clicked_x": clicked_x,
        "clicked_y": clicked_y,
        "clicked_gray": int(clicked_gray_level),
        "refined_z": max_z,
        "refined_x": image_x,
        "refined_y": image_y,
        "refined_gray": int(max_gray_level),
        "particle_volume": int(particle_volume),
        "noise_level": float(noise_level),
    }
    _redraw_images()


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
    global_data.display_slice = img.astype(np.uint8)

    max_proj = global_data.max_proj
    img = (
        (max_proj - max_proj.min()) / max(max_proj.max() - max_proj.min(), 1e-6) * 255.0
    )
    global_data.display_maxproj = img.astype(np.uint8)

    _redraw_images()


app = typer.Typer()


def _init_globals(vol_index: int) -> None:
    global global_data, last_click_info
    _clear_pending_particle()
    last_click_info = None
    global_data = GlobalData(vol_index=vol_index)
    if global_data.vol is None:
        raise ValueError("Failed to load volume data")
    else:
        print(f"\n\n    ****Loaded volume {global_data.cfg['cell'][vol_index]}")

    typer.echo("Volume shape: %s" % (global_data.vol.shape,))
    typer.echo("Computing grayscale landmarks...")
    z0, z1 = global_data.z_min, global_data.z_max
    air_grayvalue, core_grayvalue, max_grayvalue = estimate_grayscale_range(
        global_data.vol[z0 : z1 + 1], slice_step=global_data.grayscale_slice_step
    )
    metal_threshold = estimate_metal_threshold(
        global_data.vol[z0 : z1 + 1],
        slice_step=global_data.grayscale_slice_step,
        offset_percentile=global_data.metal_offset_percentile,
    )

    typer.echo("Estimating global noise...")
    slice_levels = list(range(z0, z1 + 1, max(1, (z1 - z0) // 20)))
    if z1 - 1 not in slice_levels:
        slice_levels.append(z1 - 1)
    global_stats = estimate_volume_slice_stats(
        global_data.vol[z0:z1],
        slice_levels=slice_levels,
        trunc_k=global_data.noise_clip_scale,
        metal_threshold=metal_threshold,
        min_area=global_data.metal_min_area,
        grayscale_metal_margin=global_data.metal_grayscale_margin,
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


qt_viewer = None


@app.command()
def main(
    vol_index: int = typer.Argument(..., help="Index into the config's [[cell]] array"),
) -> None:
    """Run three independent Qt image windows and the Info/controls window."""
    from PySide6.QtWidgets import QApplication

    from apps.slice_analysis.slice_viewer_qt import SliceViewer

    global qt_viewer
    application = QApplication.instance() or QApplication(sys.argv[:1])
    _init_globals(vol_index)
    # Force the initial statistics/slab computation, including when z_min is zero.
    global_data.z_current = global_data.z_min - 1
    global_data.update_slice_location(global_data.z_min)
    qt_viewer = SliceViewer(sys.modules[__name__], application)
    update_windows()
    qt_viewer.show()
    try:
        application.exec()
    finally:
        qt_viewer.close()
        qt_viewer = None


if __name__ == "__main__":
    app()
