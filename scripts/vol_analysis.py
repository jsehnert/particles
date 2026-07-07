#!/usr/bin/env python3
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import cast

import numpy as np
import scipy
import typer
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
sys.path.insert(0, str(PROJECT_ROOT))

from cfg_data import AnalysisBase  # noqa: E402
from extract_candidates_3d import Candidate, detect_candidates_streaming
from metal import extract_metal_mask  # noqa: E402
from noise import (  # noqa: E402
    estimate_volume_slice_stats,
    leavein_median_noise_scale_penta,
)
from utils import identify_cylindrical_support  # noqa: E402


# region ############################ Global Data #############################
class GlobalData(AnalysisBase):
    shape: tuple[int, int, int]

    def __init__(self, cfg_path: Path, vol_file_name: str | None = None) -> None:
        super().__init__(cfg_path=cfg_path, vol_file_name=vol_file_name)

        z_span = self.z_max - self.z_min + 1
        self.shape: tuple[int, int, int] = (
            z_span,
            self.vol.shape[1],
            self.vol.shape[2],
        )

        self._on_startup()

    def _on_startup(self) -> None:
        self._set_stats()

        # The arrays were extended to the full volume, but we only need the slices for analysis. So we will slice them down to the analysis range.
        self.sigma_corrected_for_analysis = self.sigma_corrected[
            self.z_min : self.z_max + 1
        ]
        self.rho1_for_analysis = self.rho1[self.z_min : self.z_max + 1]
        self.rho2_for_analysis = self.rho2[self.z_min : self.z_max + 1]

    def _set_stats(self) -> None:
        """Sets the per-slice stats (sigma_corrected, rho1, rho2) on the GlobalData object, interpolating to all z-slices.

        Args:
            stats (dict[int, dict[str, float]]): _description_
        """
        # Set the analysis slice levels for analysis. Note that estimate_volume_slice_stats() will clip all values at the beginning and end so that 5 valid slicees are avaialble for estimating the stats - in particular rho_2.
        slice_step = self.noise_slice_step
        slice_levels: list[int] = []
        z0 = 0
        z1 = self.shape[0] - 1
        for n in range(z0, z1 + 1, slice_step):
            slice_levels.append(n)
        if slice_levels[-1] != z1:
            slice_levels.append(z1)

        slice_stats = estimate_volume_slice_stats(
            self.vol_for_analysis,
            slice_levels,
            trunc_k=self.noise_clip_scale,
            metal_threshold=self.metal_threshold,
        )

        stats = slice_stats["slice_stats"]
        # The z_s are clipped at the beginning and the end to allow a buffer of slices for the complete stats.
        z_s = np.array(list(stats.keys()), dtype=np.float32)

        order = np.argsort(z_s)
        z_s = z_s[order]

        sigma_corrected = np.array(
            [s["sigma_corrected"] for s in stats.values()], dtype=np.float32
        )[order]
        rho1 = np.array([s["rho1"] for s in stats.values()], dtype=np.float32)[order]
        rho2 = np.array([s["rho2"] for s in stats.values()], dtype=np.float32)[order]

        # The z's from teh stats are relative to z_min, so we need to account for this
        z_s += self.z_min  # Adjust z_s to be relative to the full volume

        all_z = np.arange(self.vol.shape[0], dtype=np.float32)
        self.sigma_corrected = np.interp(all_z, z_s, sigma_corrected)
        self.rho1 = np.interp(all_z, z_s, rho1)
        self.rho2 = np.interp(all_z, z_s, rho2)

        # Smooth the results over a significant sized window to reduce change in the global stats as we move through the volume. This is important to avoid abrupt changes in the residual source stats which would cause false positives in the streaming detection. The sigma is chosen for the Gaussian FWHM at 1/2 of the slice_step.
        gs_sigmas = slice_step / (2 * np.sqrt(np.log(4)))  # 50
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

    def plot_slice_stats(self) -> None:
        return
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

    def _compute_residual_noise_estimates(
        self, z0: int, z1: int
    ) -> NDArray[np.float32]:
        """Computes the residual noise estimates for the given analysis slice range [z0, z1) using the global stats (sigma_corrected, rho1, rho2) and the leave-in median noise scale or leave-out average noise scale.

        Args:
            z0 (int): The starting slice index (inclusive) in the analysis volume.
            z1 (int): The ending slice index (exclusive) in the analysis volume.

        Returns:
            NDArray[float]: A 1D array of residual noise estimates for each slice in the range.
        """
        window_size = self.slab_thickness  # self.cfg["analysis"]["slab_thickness"]
        method = self.baseline_method  # self.cfg["analysis"]["baseline_method"]
        sigma_scales: list[np.float32] = []

        # For the parameters, use the for_analysis versions which are synched to the analysis range indexing at 0.
        rho1 = self.rho1_for_analysis
        rho2 = self.rho2_for_analysis
        sigma_corrected = self.sigma_corrected_for_analysis

        if method == "median":
            for n in range(z0, z1):
                sr = leavein_median_noise_scale_penta(
                    window_size, rho1[n], rho2[n], check_pd=False
                )
                sigma_scales.append(np.float32(sr))
        else:
            for n in range(z0, z1):
                num = np.sqrt(
                    window_size**2
                    - window_size * (1 + 2 * (rho1[n] + rho2[n]))
                    - 2 * (rho1[n] + 2 * rho2[n])
                )
                sr = num / (window_size - 1.0)
                sigma_scales.append(np.float32(sr))

        sigma_residuals = sigma_corrected[z0:z1] * np.array(
            sigma_scales, dtype=np.float32
        )
        return cast(NDArray[np.float32], sigma_residuals)

    def _compute_residual(
        self, n: int, method: str, window_size: int
    ) -> NDArray[np.float32]:
        """Computes the residual for the given slice index n using the global stats (rho1, rho2) and the leave-in median or leave-out average filter.

        Args:
            n (int): The slice index for which to compute the residual.
        returns:
            NDArray[float]: A 2D array representing the residual for the given slice index.
        """
        z0 = n - window_size // 2
        z1 = n + window_size // 2 + 1  # Exclusive

        if z0 < 0:
            z0 = 0
            z1 = window_size
        elif z1 > self.vol_for_analysis.shape[0]:
            z1 = self.vol_for_analysis.shape[0]
            z0 = z1 - window_size

        if method == "median":
            baseline = np.median(self.vol_for_analysis[z0:z1], axis=0).astype(
                np.float32
            )
        else:
            # use the leave-out average to improve particle contrast
            baseline = (
                self.vol_for_analysis[z0:z1].sum(axis=0, dtype=np.float32)
                - self.vol_for_analysis[n]
            ) / (window_size - 1)

        metal_mask = extract_metal_mask(
            self.vol_for_analysis[z0:z1].max(axis=0),
            self.metal_threshold,
            min_area=self.metal_min_area,
            margin=self.metal_grayscale_margin,
        )

        # We are only interested in positive residuals outside the metal regions
        """residual = np.clip(
            self.vol_for_analysis[n].astype(np.float32) - baseline, 0, None
        )"""

        # Clipping has been removed for robust contrast calculations in the near surrounding
        # regions of candidate particles.
        residual = self.vol_for_analysis[n].astype(np.float32) - baseline
        residual[metal_mask] = 0
        return residual

    def read(self, z0: int, z1: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Returns (sigmas, residuals) for slices [z0, z1).
        args:
            z0 (int): The starting slice index (inclusive) in the analysis volume.
            z1 (int): The ending slice index (exclusive) in the analysis volume.
        returns:
            tuple[NDArray[float], NDArray[float]]: A tuple containing:
                - residuals: A 3D array of residuals for each slice in the range [z0, z1), with shape (z1 - z0, H, W).
                - sigmas: A 1D array of residual noise estimates for each slice in the range [z0, z1).
        """

        # Compute the residual noise estimates - for each residual slice
        window_size = self.slab_thickness
        method = self.baseline_method
        sigma_residuals = self._compute_residual_noise_estimates(z0, z1)

        residual_block: NDArray[np.float32] = np.zeros(
            (z1 - z0, *self.shape[1:]), dtype=np.float32
        )

        # Compute the cylindrial support mask using onlt the first 3 slices - this is a short-cut that we might want to think more about as we move forward.
        mm = extract_metal_mask(
            self.vol_for_analysis[z0 : z0 + 3].max(axis=0),
            self.metal_threshold,
            min_area=self.metal_min_area,
            margin=self.metal_grayscale_margin,
        )
        support_mask = identify_cylindrical_support(cast(NDArray[np.bool_], mm))

        n_slices = z1 - z0
        n_workers = min(os.cpu_count() or 1, n_slices)

        if n_workers < 2:
            # Sequential path — avoids thread overhead for tiny chunks
            for n in range(z0, z1):
                residual_block[n - z0] = self._compute_residual(
                    n, method=method, window_size=window_size
                )
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(self._compute_residual, n, method, window_size): n
                    for n in range(z0, z1)
                }
                wait(futures)
                for future, n in futures.items():
                    residual = future.result()
                    residual_block[n - z0] = residual

        residual_block[:, ~support_mask] = 0

        sigma_residuals = sigma_residuals[:, None, None]  # Expand dims for broadcasting

        return residual_block, sigma_residuals


# endregion ######################## Global Data ###############################

global_data: GlobalData

_CANDIDATE_FIELDS = [
    "n_voxels",
    "n_seed",
    "z_min",
    "z_max",
    "z_extent",
    "y_extent",
    "x_extent",
    "fill",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "snr_cluster",
    "r_peak",
    "r_peak_ratio",
    "peak_offset",
    "snr_peak",
    "linearity",
    "planarity",
    "sphericity",
    "axis_z",
    "normal_z",
    "edge_contrast",
    "decay_drop",
    "seed_grown_ratio",
]


def write_candidates_csv(candidates: list[Candidate], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CANDIDATE_FIELDS)
        writer.writeheader()
        for c in candidates:
            cz, cy, cx = c.centroid
            writer.writerow(
                {
                    "n_voxels": c.n_voxels,
                    "n_seed": c.n_seed,
                    "z_min": c.z_min,
                    "z_max": c.z_max,
                    "z_extent": c.z_extent,
                    "y_extent": c.y_extent,
                    "x_extent": c.x_extent,
                    "fill": f"{c.fill:.4f}",
                    "centroid_z": f"{cz:.3f}",
                    "centroid_y": f"{cy:.3f}",
                    "centroid_x": f"{cx:.3f}",
                    "snr_cluster": f"{c.snr_cluster:.4f}",
                    "r_peak": f"{c.r_peak:.4f}",
                    "r_peak_ratio": f"{c.r_peak_ratio:.4f}",
                    "peak_offset": f"{c.peak_offset:.4f}",
                    "snr_peak": f"{c.snr_peak:.4f}",
                    "linearity": f"{c.linearity:.4f}",
                    "planarity": f"{c.planarity:.4f}",
                    "sphericity": f"{c.sphericity:.4f}",
                    "axis_z": f"{c.axis_z:.4f}",
                    "normal_z": f"{c.normal_z:.4f}",
                    "edge_contrast": f"{c.edge_contrast:.4f}",
                    "decay_drop": f"{c.decay_drop:.4f}",
                    "seed_grown_ratio": f"{c.seed_grown_ratio:.4f}",
                }
            )


def main(
    vol_file_name: str = typer.Option(
        None, help="Volume file name override (default: from config)"
    ),
) -> None:
    all_vol_files = [f for f in (PROJECT_ROOT / "Data").glob("*.raw")]
    all_vol_files = sorted(all_vol_files, key=lambda f: f.stem)

    for f in all_vol_files:
        vol_file_name = f.name
        global global_data
        typer.echo(f"Loading global data from {CONFIG_PATH}...")
        global_data = GlobalData(CONFIG_PATH, vol_file_name=vol_file_name)

        typer.echo(f"Loading volume from {global_data.vol_data_path}...")
        typer.echo(
            f"Volume loaded with shape {global_data.vol.shape} and dtype {global_data.vol.dtype}. Initializing global data..."
        )

        typer.echo(
            "Running through the volume computing residual slices for analysis..."
        )

        begin = time.time()
        candidates = detect_candidates_streaming(
            src=global_data,
            k_high=global_data.high_threshold_scale,
            k_low=global_data.low_threshold_scale,
            min_voxels=global_data.vol_threshold,
            z_extent_max=global_data.z_extent_max,
            aniso_factor=global_data.aniso_factor,
            z_offset=global_data.z_min,
            slices_per_chunk=global_data.slices_per_chunk,
            small_z_bounds=global_data.small_z_bounds,
            small_voxel_cutoff=global_data.small_vol_cutoff,
            z_pad=global_data.z_pad,
        )
        end = time.time()
        typer.echo(f"Candidate detection completed in {end - begin:.2f} seconds.")

        print(f"Detected {len(candidates)} candidates in the volume.")
        total_voxels = np.sum([c.n_voxels for c in candidates])
        print(f"Total voxels in detected candidates: {total_voxels}")
        candidate_volumes = [c.n_voxels for c in candidates]
        largest_candidate_idx = np.argmax(candidate_volumes)
        largest_candidate = candidates[largest_candidate_idx]
        largest_location = candidates[largest_candidate_idx].centroid
        lc_z = largest_location[0]
        lc_x = largest_location[1]
        lc_y = largest_location[2]
        print(
            f"Largest candidate has {largest_candidate.n_voxels} voxels at (z, x, y) = ({lc_z:.1f}, {lc_x:.1f}, {lc_y:.1f})."
        )

        fname: str = str(global_data.vol_data_path.stem) + "_candidates.csv"
        print(f"Writing candidates to {fname}...")
        out_path = PROJECT_ROOT / "Data" / "particles" / Path(fname)

        write_candidates_csv(candidates, out_path)
        typer.echo(f"Wrote {len(candidates)} candidates to {out_path}")


if __name__ == "__main__":
    typer.run(main)
