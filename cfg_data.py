from __future__ import annotations

import gc
import math
import tomllib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from metal import estimate_metal_threshold
from utils import estimate_grayscale_range


def _load_config(cfg_path: Path) -> dict:
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _load_volume(volume_path: Path, voxels_x: int, voxels_y: int) -> NDArray[np.uint8]:
    dtype: type[np.uint8] | type[np.float32] = (
        np.float32 if volume_path.suffix == ".vol" else np.uint8
    )

    z = (volume_path.stat().st_size) // (voxels_y * voxels_x * np.dtype(dtype).itemsize)
    return np.memmap(volume_path, dtype=dtype, mode="r", shape=(z, voxels_y, voxels_x))


class _CfgData:
    """
    Base class for loading and providing access to configuration data and the volume.
    """

    def __init__(self, cfg_path: Path, vol_index: int, exp_index: int) -> None:
        """Initialize the configuration data object.

        Args:
            cfg_path (Path): path to the toml config file
            vol_index (int): index into the [[cell]] array selecting the volume to load
            exp_index (int): index into the [[experiment]] array selecting the detection parameters to use
        """
        self.cfg_path = cfg_path
        self.cfg_dir = cfg_path.parent
        self.cfg = _load_config(self.cfg_path)
        self.vol_index = vol_index
        self.exp_index = exp_index

        self.vol: NDArray[np.uint8] | None = _load_volume(
            self.vol_data_path, self.voxels_x, self.voxels_y
        )

    def free_volume(self) -> None:
        """Free the memory used by the volume data."""
        self.vol = None
        self.vol_for_analysis = None
        gc.collect()

    @property
    def cell(self) -> dict:
        return self.cfg["cell"][self.vol_index]

    @property
    def voxels_x(self) -> int:
        return int(self.cell["voxels_x"])

    @property
    def voxels_y(self) -> int:
        return int(self.cell["voxels_y"])

    @property
    def experiment(self) -> dict:
        return self.cfg["experiment"][self.exp_index]

    @property
    def cell_name(self) -> str:
        return self.cell["name"]

    @property
    def vol_data_path(self) -> Path:
        vol_dir = Path(self.cfg["paths"]["vol_data_dir"])
        if not vol_dir.is_absolute():
            vol_dir = (self.cfg_dir / vol_dir).resolve()

        return vol_dir / f"{self.cell_name}.raw"

    @property
    def z_min(self) -> int:
        return int(self.cell["z_min"])

    @z_min.setter
    def z_min(self, value: int) -> None:
        self.cell["z_min"] = value

    @property
    def z_max(self) -> int:
        return int(self.cell["z_max"])

    @z_max.setter
    def z_max(self, value: int) -> None:
        self.cell["z_max"] = value

    @property
    def voxel_size_mm(self) -> float:
        return self.cell["voxel_size"]

    @property
    def form_factor(self) -> str:
        return self.cell["form_factor"]

    @property
    def nominal_diameter_mm(self) -> float:
        return float(self.cell["nominal_diameter_mm"])

    @property
    def nominal_height_mm(self) -> float:
        return float(self.cell["nominal_height_mm"])

    @property
    def slab_thickness(self) -> int:
        return int(self.experiment["slab_thickness"])

    @slab_thickness.setter
    def slab_thickness(self, value: int) -> None:
        self.experiment["slab_thickness"] = value

    @property
    def baseline_method(self) -> str:
        return self.experiment["baseline_method"]

    @baseline_method.setter
    def baseline_method(self, method: str) -> None:
        if method not in ("mean", "median"):
            raise ValueError(f"Invalid baseline method: {method}")

        self.experiment["baseline_method"] = method

    @property
    def dead_zone_scale(self) -> float:
        return self.cfg["analysis"]["dead_zone_scale"]

    @dead_zone_scale.setter
    def dead_zone_scale(self, value: float) -> None:
        self.cfg["analysis"]["dead_zone_scale"] = value

    @property
    def noise_slice_step(self) -> int:
        return self.cfg["noise"]["slice_step"]

    @property
    def noise_clip_scale(self) -> float:
        return self.cfg["noise"]["clip_scale"]

    @property
    def grayscale_slice_step(self) -> int:
        return self.cfg["grayscale_landmarks"]["slice_step"]

    @property
    def metal_offset_percentile(self) -> int:
        return self.cfg["metal"]["metal_offset_percentile"]

    @metal_offset_percentile.setter
    def metal_offset_percentile(self, value: int) -> None:
        self.cfg["metal"]["metal_offset_percentile"] = value

    @property
    def metal_grayscale_margin(self) -> int:
        # A uint8 grayscale delta for the metal-mask hysteresis (the low band is
        # metal_threshold - margin), NOT a spatial distance — resolution-independent
        # by nature, so it is stored and used directly with no voxel_size scaling.
        return self.cfg["metal"]["gray_margin"]

    @metal_grayscale_margin.setter
    def metal_grayscale_margin(self, margin: int) -> None:
        self.cfg["metal"]["gray_margin"] = margin

    @property
    def metal_min_area(self) -> int:
        # Stored as a physical area (mm^2); converted to pixel count with this
        # cell's voxel_size (area scales as 1/voxel_size**2).
        return round(self.cfg["metal"]["min_area_mm2"] / self.voxel_size_mm**2)

    @metal_min_area.setter
    def metal_min_area(self, value: int) -> None:
        # Callers speak pixels; store back as a physical area (mm^2).
        self.cfg["metal"]["min_area_mm2"] = value * self.voxel_size_mm**2

    @property
    def area_threshold(self) -> int:
        return self.cfg["analysis"]["area_threshold"]

    @area_threshold.setter
    def area_threshold(self, value: int) -> None:
        self.cfg["analysis"]["area_threshold"] = value

    @property
    def slices_per_chunk(self) -> int:
        return self.cfg["analysis"]["slices_per_chunk"]

    @property
    def storage_margin(self) -> int:
        return int(self.cfg["analysis"]["storage_margin"])

    @property
    def shell_radius(self) -> int:
        return int(self.cfg["analysis"]["shell_radius"])

    @property
    def enforce_shape_gates(self) -> bool:
        # When False (research default), detection skips the _axial_ok shape gates
        # and defers shape discrimination to the classifier; min_seed_voxels and
        # z_extent_max still gate.
        return bool(self.cfg["analysis"]["enforce_shape_gates"])

    @property
    def n_workers(self) -> int:
        return self.cfg["analysis"]["n_workers"]

    @property
    def preprocess_slices_per_chunk(self) -> int:
        return self.cfg["analysis"]["preprocess_slices_per_chunk"]

    @property
    def preprocess_n_workers(self) -> int:
        return self.cfg["analysis"]["preprocess_n_workers"]

    @property
    def max_axial_extent_mm(self) -> float:
        return float(self.experiment["max_axial_extent_mm"])

    @property
    def z_extent_max(self) -> int:
        # Physical axial ceiling converted to whole slices for THIS cell's voxel_size,
        # so the FP filter (and the slab halo it also sizes) is resolution-independent.
        # ceil, not round: rounding a ceiling down could clip a borderline real
        # particle, so err toward keeping candidates (sensitivity-first; FP is the
        # classifier's job). Isotropic reconstruction => voxel_size is the z spacing.
        return math.ceil(self.max_axial_extent_mm / self.voxel_size_mm)

    @property
    def min_seed_voxels(self) -> int:
        return self.experiment["min_seed_voxels"]

    @property
    def high_threshold_scale(self) -> float:
        return self.experiment["high_threshold_scale"]

    @property
    def low_threshold_scale(self) -> float:
        return self.experiment["low_threshold_scale"]

    @property
    def small_vol_cutoff(self) -> int:
        return self.experiment["small_vol_cutoff"]

    @property
    def aniso_factor(self) -> float:
        return self.experiment["aniso_factor"]

    @property
    def small_z_bounds(self) -> tuple[int, int]:
        bounds = self.experiment["small_z_bounds"]
        if isinstance(bounds, list) and len(bounds) == 2:
            return tuple(bounds)
        raise ValueError(
            f"Invalid small_z_bounds in config: {bounds}. Expected a list of two integers."
        )

    @property
    def z_pad(self) -> int:
        return self.experiment["z_pad"]


class AnalysisBase(_CfgData):
    """
    Base class for analysis, providing access to configuration and volume data, and performing
    basic pre-analysis operations.
    """

    def __init__(self, cfg_path: Path, vol_index: int, exp_index: int) -> None:
        super().__init__(cfg_path, vol_index, exp_index)
        self._grayscale_landmarks: dict[str, int] | None = None

    def histogram(self, slice_steps: int | None = None) -> NDArray:
        """
        Compute the histogram of the volume data.

        Parameters:
            slice_steps (int | None): The step size for slicing the volume. If None, use the
                                      default slice step from the configuration.

        Returns:
            NDArray: The histogram counts of the volume data.
        """
        if self.vol is None:
            raise ValueError("Volume data is not loaded.")

        if slice_steps is None:
            slice_steps = self.grayscale_slice_step

        # Slice the volume based on the specified step size
        sliced_volume = self.vol[::slice_steps]

        # Compute the histogram
        hist, _bin_edges = np.histogram(sliced_volume, bins=256, range=(0, 255))

        return hist

    @property
    def vol_for_analysis(self) -> NDArray[np.uint8]:
        """
        Returns the volume subset that should be used for analysis, based on the configured top and bottom slice indices.
        """
        if self.vol is None:
            raise ValueError("Volume data is not loaded.")

        return self.vol[self.z_min : self.z_max + 1]

    @vol_for_analysis.setter
    def vol_for_analysis(self, value: NDArray[np.uint8] | None) -> None:
        """
        Sets the volume subset for analysis. This allows for replacing the volume data with a modified version if needed.

        Args:
            value (NDArray[np.uint8] | None): The new volume data to be used for analysis.
        """
        self.vol = value

    @property
    def grayscale_landmarks(self) -> dict[str, int]:
        """
        Lazily compute the grayscale landmarks of the volume data.

        Returns:
            dict[str, int]: A dictionary containing the grayscale values of air, core, metal threshold, and max.
        """
        if self._grayscale_landmarks is not None:
            return self._grayscale_landmarks

        air_grayvalue, core_grayvalue, max_grayvalue = estimate_grayscale_range(
            self.vol_for_analysis, self.grayscale_slice_step
        )

        metal_threshold = estimate_metal_threshold(
            self.vol_for_analysis,
            slice_step=self.grayscale_slice_step,
            offset_percentile=self.metal_offset_percentile,
        )

        self._grayscale_landmarks = {
            "air_grayvalue": air_grayvalue,
            "core_grayvalue": core_grayvalue,
            "metal_threshold": metal_threshold,
            "max_grayvalue": max_grayvalue,
        }
        return self._grayscale_landmarks

    @property
    def metal_threshold(self) -> int:
        """
        Get the estimated metal threshold from the grayscale landmarks.

        Returns:
            int: The estimated metal threshold.

        The side-effect of this property is that it will compute the grayscale landmarks if they have not been computed yet.
        """
        return self.grayscale_landmarks["metal_threshold"]
