#!/usr/bin/env python3
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

from metal import estimate_grayscale_range, estimate_metal_threshold  # noqa: E402
from noise import estimate_volume_slice_stats, leavein_median_noise_scale_penta
from volume import load_volume  # noqa: E402


# region ############################ Global Data #############################
class GlobalData:
    def __init__(self, vol: NDArray[np.uint8], cfg: dict):
        self.cfg = cfg
        z_top = self.cfg["analysis"].get("vol_top", 0)
        z_bottom = self.cfg["analysis"].get("vol_bottom", vol.shape[0] - 1)
        self.vol: NDArray[np.uint8] = vol[z_top : z_bottom + 1]
        self._on_startup()

        self.shape: tuple[int, int, int] = self.vol.shape

    def _on_startup(self) -> None:
        slice_step = self.cfg["analysis"].get("slice_step", 100)
        air_value, max_value = estimate_grayscale_range(self.vol, slice_step=slice_step)
        metal_threshold = estimate_metal_threshold(self.vol, air_value)
        print(
            f"Estimated air/metal/max gray value: "
            f"{air_value} {metal_threshold} {max_value}"
        )
        self.metal_threshold = metal_threshold

        self._set_stats()

    def _set_stats(self) -> None:
        """Sets the per-slice stats (sigma_corrected, rho1, rho2) on the GlobalData object, interpolating to all z-slices.

        Args:
            stats (dict[int, dict[str, float]]): _description_
        """
        # Set the analysis slice levels
        slice_step = self.cfg["analysis"].get("slice_step", 100)
        slice_levels: list[int] = []
        z0 = 0
        z1 = self.shape[0] - 1
        for n in range(z0, z1 + 1, slice_step):
            slice_levels.append(n)
        if slice_levels[-1] != z1:
            slice_levels.append(z1)

        slice_stats = estimate_volume_slice_stats(
            self.vol,
            slice_levels,
            trunc_k=self.noise_clip_level,
            metal_threshold=self.metal_threshold,
        )

        stats = slice_stats["slice_stats"]
        z_s = np.array(list(stats.keys()), dtype=np.float32)
        order = np.argsort(z_s)
        z_s = z_s[order]

        sigma_corrected = np.array(
            [s["sigma_corrected"] for s in stats.values()], dtype=np.float32
        )[order]
        rho1 = np.array([s["rho1"] for s in stats.values()], dtype=np.float32)[order]
        rho2 = np.array([s["rho2"] for s in stats.values()], dtype=np.float32)[order]

        all_z = np.arange(self.vol.shape[0], dtype=np.float32)
        self.sigma_corrected = np.interp(all_z, z_s, sigma_corrected)
        self.rho1 = np.interp(all_z, z_s, rho1)
        self.rho2 = np.interp(all_z, z_s, rho2)

        # Smooth the results over a significant sized window
        gs_sigmas = 50
        self.sigma_corrected = scipy.ndimage.gaussian_filter1d(
            self.sigma_corrected, sigma=gs_sigmas, mode="nearest"
        )
        self.rho1 = scipy.ndimage.gaussian_filter1d(
            self.rho1, sigma=gs_sigmas, mode="nearest"
        )
        self.rho2 = scipy.ndimage.gaussian_filter1d(
            self.rho2, sigma=gs_sigmas, mode="nearest"
        )

        self.plot_slice_stats()

    @property
    def noise_clip_level(self) -> float:
        return self.cfg["noise"].get("clip_scale", 2.5)

    def plot_slice_stats(self) -> None:
        import matplotlib.pyplot as plt

        plt.style.use("dark_background")

        plt.figure(figsize=(12, 12))
        R = 3
        plt.subplot(R, 1, 1)
        plt.plot(
            self.sigma_corrected,
        )
        plt.title("Sigma corrected")
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        plt.grid()

        plt.subplot(R, 1, 2)
        plt.plot(
            self.rho1,
        )
        plt.title("Rho1")
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        plt.grid()

        plt.subplot(R, 1, 3)
        plt.plot(
            self.rho2,
        )
        plt.title("Rho2")
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        plt.grid()

        plt.tight_layout()
        plt.show()

    def read(self, z0: int, z1: int
     ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Returns (sigma_corrected, rho) for slices [z0, z1)."""
        window_size = self.cfg.get("analysis", {}).get("slab_thickness", 11)
        method = self.cfg.get("analysis", {}).get("baseline_method", "median")
        sigma_scales: list[np.float32] = []
        if method == "median":
            for n in range(z0, z1):
                sr = leavein_median_noise_scale_penta(W, self.rho1[n], self.rho2[n], check_pd=False)
                sigma_scales.append(sr)
        else:
            for n in range(z0, z1):
                num = np.sqrt(
                    window_size**2 - window_size * (1 + 2 * (self.rho1[n] + self.rho2[n])) - 2 * (self.rho1[n] + 2 * self.rho2[n])
                )
                sr = num / (window_size - 1.0)
                sigma_scales.append(sr)

        #### NEED TO DO THIS...
        residual_block: NDArray[np.float32] = np.zeros((z1-z0,*self.shape[1:]), dtype = np.float32)  # or rho2, or a combination
        #
        sigma_residuals = self.sigma_corrected[z0:z1] * np.array(sigma_scales, dtype=np.float32)
        return sigma_residuals[:, None, None], residual_block

# endregion######################## Global Data ###############################


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def resolve_volume_path(cfg: dict) -> Path:
    configured_path = cfg.get("paths", {}).get("vol_data")
    if not configured_path:
        raise ValueError("config is missing paths.vol_data")

    data_path = Path(configured_path)
    if data_path.is_absolute():
        return data_path

    candidates = [
        CONFIG_PATH.parent / data_path,
        Path.cwd() / data_path,
        Path(__file__).resolve().parent / data_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


global_data: GlobalData


def main() -> None:
    global global_data
    cfg = load_config()
    volume_path = resolve_volume_path(cfg)
    if not volume_path.exists():
        raise FileNotFoundError(
            f"volume file not found: {volume_path}. Check paths.vol_data in {CONFIG_PATH}."
        )

    vol = load_volume(volume_path)
    global_data = GlobalData(vol, cfg)

    print(f"Config: {CONFIG_PATH}")
    print(f"Volume: {volume_path}")
    print(f"Shape: {vol.shape}")
    print(f"Dtype: {vol.dtype}")


if __name__ == "__main__":
    main()
