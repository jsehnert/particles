from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from metal import estimate_metal_threshold
from utils import estimate_grayscale_range


def _load_config(cfg_path: Path) -> dict:
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _parse_metadata(meta_path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not meta_path.exists():
        existing_stem = "M50L-06"
        new_meta_path = meta_path.with_name(f"{existing_stem}.xtekhelixct")

        print(
            f"Using metadata file {new_meta_path} instead of {meta_path}, exists={new_meta_path.exists()}"
        )
        if new_meta_path.exists():
            print(
                f"Warning: Metadata file {meta_path} not found. Using {new_meta_path} instead."
            )
            meta_path = new_meta_path

    with meta_path.open() as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, _, value = line.partition("=")
                metadata[key.strip()] = value.strip()
    return metadata


def _load_volume(volume_path: Path) -> NDArray[np.uint8]:
    meta_path = volume_path.with_suffix(".xtekhelixct")
    metadata = _parse_metadata(meta_path)

    x = int(metadata["VoxelsX"])
    y = int(metadata["VoxelsY"])
    z = int(metadata["VoxelsZ"])

    dtype: type[np.uint8] | type[np.float32] = (
        np.float32 if volume_path.suffix == ".vol" else np.uint8
    )

    return np.memmap(volume_path, dtype=dtype, mode="r", shape=(z, y, x))


class _CfgData:
    """
    Base class for loading and providing access to configuration data and the volume.
    """

    def __init__(self, cfg_path: Path, vol_file_name: str | None = None) -> None:
        """Initialize the configuration data object.

        Args:
            cfg_path (Path): path to the toml config file
            vol_file_name (str, optional): name of the volume data file to override the config. Defaults to None.
        """
        self.cfg_path = cfg_path
        self.cfg_dir = cfg_path.parent
        self.cfg = _load_config(self.cfg_path)

        self._vol_data_path: Path | None = None

        if vol_file_name is not None:
            self._vol_data_path = self.vol_data_path.with_name(vol_file_name)
        else:
            self._vol_data_path = self.vol_data_path

        self.vol: NDArray[np.uint8] = _load_volume(self._vol_data_path)

    @property
    def vol_data_path(self) -> Path:
        if self._vol_data_path is not None and self._vol_data_path.exists():
            return self._vol_data_path

        raw = Path(self.cfg["paths"]["vol_data_path"])
        if raw.is_absolute():
            self._vol_data_path = raw
        else:
            self._vol_data_path = (self.cfg_dir / raw).resolve()

        return self._vol_data_path

    @property
    def cell_name(self) -> str:
        return self.vol_data_path.stem

    @property
    def z_min(self) -> int:
        return int(self.cfg["analysis"]["vol_top"])

    @z_min.setter
    def z_min(self, value: int) -> None:
        self.cfg["analysis"]["vol_top"] = value

    @property
    def z_max(self) -> int:
        return int(self.cfg["analysis"]["vol_bottom"])

    @property
    def voxel_size_mm(self) -> float:
        return self.cfg["vol_info"]["voxel_size"]

    @property
    def slab_thickness(self) -> int:
        return int(self.cfg["analysis"]["slab_thickness"])

    @slab_thickness.setter
    def slab_thickness(self, value: int) -> None:
        self.cfg["analysis"]["slab_thickness"] = value

    @property
    def baseline_method(self) -> str:
        return self.cfg["analysis"]["baseline_method"]

    @baseline_method.setter
    def baseline_method(self, method: str) -> None:
        if method not in ("mean", "median"):
            raise ValueError(f"Invalid baseline method: {method}")

        self.cfg["analysis"]["baseline_method"] = method

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
    def metal_grayscale_margin(self) -> int:
        return self.cfg["metal"]["gray_margin"]

    @metal_grayscale_margin.setter
    def metal_grayscale_margin(self, margin: int) -> None:
        self.cfg["metal"]["gray_margin"] = margin

    @property
    def metal_min_area(self) -> int:
        return self.cfg["metal"]["min_area"]

    @metal_min_area.setter
    def metal_min_area(self, value: int) -> None:
        self.cfg["metal"]["min_area"] = value

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
    def z_extent_max(self) -> int:
        return self.cfg["analysis"]["z_extent_max"]

    @property
    def vol_threshold(self) -> int:
        return self.cfg["analysis"]["vol_threshold"]

    @property
    def high_threshold_scale(self) -> float:
        return self.cfg["analysis"]["high_threshold_scale"]

    @property
    def low_threshold_scale(self) -> float:
        return self.cfg["analysis"]["low_threshold_scale"]

    @property
    def small_vol_cutoff(self) -> int:
        return self.cfg["analysis"]["small_vol_cutoff"]

    @property
    def aniso_factor(self) -> float:
        return self.cfg["analysis"]["aniso_factor"]

    @property
    def small_z_bounds(self) -> tuple[int, int]:
        bounds = self.cfg["analysis"]["small_z_bounds"]
        if isinstance(bounds, list) and len(bounds) == 2:
            return tuple(bounds)
        raise ValueError(
            f"Invalid small_z_bounds in config: {bounds}. Expected a list of two integers."
        )

    @property
    def z_pad(self) -> int:
        return self.cfg["analysis"]["z_pad"]


class AnalysisBase(_CfgData):
    """
    Base class for analysis, providing access to configuration and volume data, and performing
    basic pre-analysis operations.
    """

    def __init__(self, cfg_path: Path, vol_file_name: str | None = None) -> None:
        super().__init__(cfg_path, vol_file_name)
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
        return self.vol[self.z_min : self.z_max + 1]

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
            self.vol_for_analysis, air_grayvalue, self.grayscale_slice_step
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
