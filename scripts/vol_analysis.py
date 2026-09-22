#!/usr/bin/env python3
import csv
import math
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import cast

import numpy as np
import scipy
import typer
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

# cv2.setNumThreads(1)  # Disable OpenCV multithreading to avoid oversubscription
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
sys.path.insert(0, str(PROJECT_ROOT))

import experiment_db as db  # noqa: E402
from cfg_data import AnalysisBase  # noqa: E402
from extract_candidates_3d import (  # noqa: E402
    EXPERIMENT_PARAM_FIELDS,
    Candidate,
    ResidualSource,
    detect_candidates_parallel,
    detect_candidates_streaming,
    precompute_residual,
)
from metal import extract_metal_mask  # noqa: E402
from noise import (  # noqa: E402
    estimate_volume_slice_stats,
    leavein_median_noise_scale_penta,
)
from utils import median_max_baseline_block  # noqa: E402


# region ############################ Global Data #############################
class GlobalData(AnalysisBase):
    shape: tuple[int, int, int]

    def __init__(self, cfg_path: Path, vol_index: int, exp_index: int) -> None:
        super().__init__(cfg_path=cfg_path, vol_index=vol_index, exp_index=exp_index)

        z_span = self.z_max - self.z_min + 1
        self.shape: tuple[int, int, int] = (
            z_span,
            self.vol.shape[1],
            self.vol.shape[2],
        )
        self._residual: np.memmap | None = None
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
        window_size = self.slab_thickness
        method = self.baseline_method
        if method != "median":
            raise ValueError(f"Unsupported baseline method: {method}. must be 'median'")

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
    ) -> NDArray[np.int16]:
        """Computes the residual for the given slice index n using the global stats (rho1, rho2) and the leave-in median or leave-out average filter.

        Args:
            n (int): The slice index for which to compute the residual.
        returns:
            NDArray[np.int16]: A 2D array representing the residual for the given slice index.
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
            baseline = np.median(self.vol_for_analysis[z0:z1], axis=0).astype(np.uint8)
        else:
            # use the leave-out average to improve particle contrast
            assert 0, "Leave-out average baseline not implemented yet."
            baseline = (
                self.vol_for_analysis[z0:z1].sum(axis=0, dtype=np.float32)
                - self.vol_for_analysis[n]
            ) / (window_size - 1)

        max_proj = self.vol_for_analysis[z0:z1].max(axis=0)
        metal_mask = extract_metal_mask(
            max_proj,
            self.metal_threshold,
            min_area=self.metal_min_area,
            margin=self.metal_grayscale_margin,
        )

        residual = self.vol_for_analysis[n].astype(np.int16) - baseline.astype(np.int16)
        residual[metal_mask] = 0
        return residual

    def _compute_residual2(
        self,
        residual_slice: NDArray[np.int16],
        max_slice: NDArray[np.uint8],
    ) -> NDArray[np.int16]:
        """Computes the residual for the given slice index n using the global stats (rho1, rho2) and the leave-in median or leave-out average filter.

        Args:
            residual_slice (NDArray[np.int16]): The residual slice to be processed.
            max_slice (NDArray[np.uint8]): The corresponding maximum intensity slice.
        returns:
            NDArray[np.int16]: A 2D array representing the residual for the given slice index.
        """
        metal_mask = extract_metal_mask(
            max_slice,
            self.metal_threshold,
            min_area=self.metal_min_area,
            margin=self.metal_grayscale_margin,
        )
        residual = residual_slice
        residual[metal_mask] = 0
        return residual

    def read(self, z0: int, z1: int) -> tuple[NDArray[np.int16], NDArray[np.float32]]:
        """Returns (sigmas, residuals) for slices [z0, z1).
        args:
            z0 (int): The starting slice index (inclusive) in the analysis volume.
            z1 (int): The ending slice index (exclusive) in the analysis volume.
        returns:
            tuple[NDArray[float], NDArray[float]]: A tuple containing:
                - residuals: A 3D array of residuals for each slice in the range [z0, z1), with shape (z1 - z0, H, W).
                - sigmas: A 1D array of residual noise estimates for each slice in the range [z0, z1).
        """
        # TODO: Compute the cylinder support mask over the current chunk computing over the entire volume inflates the support mask from bulges.

        # Compute the residual noise estimates - for each residual slice
        window_size = self.slab_thickness
        sigma_residuals = self._compute_residual_noise_estimates(z0, z1)

        median_vol, max_vol = median_max_baseline_block(
            self.vol_for_analysis, z0, z1, window_size
        )

        residual_block: NDArray[np.int16] = np.empty(
            (z1 - z0, *self.shape[1:]), dtype=np.int16
        )
        np.subtract(
            self.vol_for_analysis[z0:z1],
            median_vol,
            out=residual_block,
            dtype=np.int16,
            casting="unsafe",
        )

        n_workers = self.preprocess_n_workers
        if n_workers < 2:
            # Sequential path — avoids thread overhead for tiny chunks
            for n in range(z0, z1):
                residual_block[n - z0] = self._compute_residual2(
                    residual_slice=residual_block[n - z0],
                    max_slice=max_vol[n - z0],
                )
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        self._compute_residual2,
                        residual_slice=residual_block[n - z0],
                        max_slice=max_vol[n - z0],
                    ): n
                    for n in range(z0, z1)
                }
                wait(futures)
                for future, n in futures.items():
                    residual = future.result()
                    residual_block[n - z0] = residual

        # NOTE: culling to the jelly-roll interior is no longer done here. It now
        # happens per-slab in extract_candidates_3d._process_slab, gated on each
        # slab's OWN fitted can contour — more accurate than the single global,
        # axially-invariant support mask this used to apply (which distorted in
        # the taper region and could retain particles outside the local cell).
        # Bright metal is still zeroed per slice above (_compute_residual2).

        sigma_residuals = sigma_residuals[:, None, None]  # Expand dims for broadcasting
        return residual_block, sigma_residuals


# endregion ######################## Global Data ###############################

global_data: GlobalData

# Historical default: no caller ever passed min_fill explicitly (neither
# _run_one's detect_candidates_parallel/streaming calls, nor config.toml,
# which has no min_fill field at all), so every experiment run so far used
# whatever detect_candidates_parallel/streaming default to -- 0.15. Recorded
# here as an explicit constant, and threaded through explicitly below, so
# it's no longer an implicit function default nobody can see from the config.
HISTORICAL_MIN_FILL = 0.15

_CANDIDATE_FIELDS = [
    "experiment_number",
    "volume_name",
    "particle",
    "n_voxels",
    "n_seed",
    "n_seed_regions",
    "n_seed_total",
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
    "fill_pca",
    "diag",
    "radial_pos",
    "peak_z",
    "peak_y",
    "peak_x",
    "gs_median",
    "gs_p90",
    "gs_peak",
    "gs_shell_median",
    "gs_contrast",
    "metal_threshold",
    *EXPERIMENT_PARAM_FIELDS,
]


def _experiment_param_values(
    experiment: dict, min_fill: float, voxel_size_mm: float, enforce_shape_gates: bool
) -> dict[str, object]:
    """Build the EXPERIMENT_PARAM_FIELDS row from a config.toml [[experiment]] entry.

    Two derived (non-config-key) fields are filled specially:
      * min_fill -- the value actually used (not yet a config field).
      * z_extent_max -- the EFFECTIVE axial ceiling in slices for THIS cell,
        ceil(max_axial_extent_mm / voxel_size_mm). It is no longer a config key
        (the config holds the physical max_axial_extent_mm); it is logged as the
        slice count that actually gated detection, so the row is self-describing
        across cell formats with differing voxel_size.
    Used for the per-cell candidate / voxel rows, so it is called from _run_one
    with that cell's voxel_size. (_experiment_row_values writes the per-
    experiment `experiments`-table row separately and carries the physical
    spec, not this cell-dependent slice count.)"""
    values: dict[str, object] = {}
    for name in EXPERIMENT_PARAM_FIELDS:
        if name == "min_fill":
            values[name] = min_fill
            continue
        if name == "enforce_shape_gates":
            # global [analysis] run setting, not an [[experiment]] key
            values[name] = enforce_shape_gates
            continue
        if name == "z_extent_max":
            values[name] = math.ceil(
                float(experiment["max_axial_extent_mm"]) / voxel_size_mm
            )
            continue
        value = experiment[name]
        values[name] = (
            ";".join(str(v) for v in value) if isinstance(value, list) else value
        )
    return values


def write_candidate_rows(
    writer: "csv.DictWriter[str]",
    candidates: list[Candidate],
    volume_name: str,
    experiment_number: int,
    experiment_params: dict[str, object],
) -> None:
    """Append one row per candidate to an already-open CSV writer.

    The 'particle' column is left blank; it is filled in later by a manual
    labeling pass. experiment_number and experiment_params (see
    _experiment_param_values) are embedded on every row so that combining
    candidates.csv files across experiments is a plain concatenation, with no
    later join against params.csv needed.
    """
    for c in candidates:
        cz, cy, cx = c.centroid
        row = {
            "experiment_number": experiment_number,
            "volume_name": volume_name,
            "particle": "",
            "n_voxels": c.n_voxels,
            "n_seed": c.n_seed,
            "n_seed_regions": c.n_seed_regions,
            "n_seed_total": c.n_seed_total,
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
            "fill_pca": f"{c.fill_pca:.4f}",
            "diag": f"{c.diag:.4f}",
            "radial_pos": f"{c.radial_pos:.4f}",
            "peak_z": c.peak_z,
            "peak_y": c.peak_y,
            "peak_x": c.peak_x,
            "gs_median": f"{c.gs_median:.4f}",
            "gs_p90": f"{c.gs_p90:.4f}",
            "gs_peak": f"{c.gs_peak:.4f}",
            "gs_shell_median": f"{c.gs_shell_median:.4f}",
            "gs_contrast": f"{c.gs_contrast:.4f}",
            "metal_threshold": f"{c.metal_threshold:.4f}",
        }
        row.update(experiment_params)
        writer.writerow(row)


def _experiment_row_values(
    experiment: dict, min_fill: float, enforce_shape_gates: bool
) -> dict[str, object]:
    """Build the `experiments`-table row (experiment_db.EXPERIMENT_FIELDS) from
    a config.toml [[experiment]] entry.

    Unlike _experiment_param_values, this excludes z_extent_max -- of the
    historical EXPERIMENT_PARAM_FIELDS it is the one field that depends on a
    cell's voxel_size_mm rather than the experiment alone, so it isn't
    experiment-level (see experiment_db module docstring) and this helper needs
    no GlobalData/voxel_size_mm to run. Called once per experiment_number in
    main(), before the per-cell loop.
    """
    values: dict[str, object] = {}
    for name in db.EXPERIMENT_FIELDS:
        if name == "min_fill":
            values[name] = min_fill
            continue
        if name == "enforce_shape_gates":
            values[name] = enforce_shape_gates
            continue
        value = experiment[name]
        values[name] = list(value) if isinstance(value, list) else value
    return values


def _load_raw_config(cfg_path: Path) -> dict:
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _run_one(
    exp_index: int,
    vol_index: int,
    recorder: db.ExperimentRecorder,
    run_parallel: bool = True,
) -> None:
    global global_data
    typer.echo(
        f"\nLoading global data from {CONFIG_PATH} (exp_index={exp_index}, vol_index={vol_index})..."
    )
    global_data = GlobalData(CONFIG_PATH, vol_index=vol_index, exp_index=exp_index)

    typer.echo(f"Volume loaded with shape {global_data.vol.shape}")

    # region Testing
    TEST = False
    if TEST:
        msg = (
            f"{'%' * 80}\n\n"
            f"Running candidate detection for {global_data.cell_name} (exp_index={exp_index}, "
            f"k_high={global_data.high_threshold_scale}, k_low={global_data.low_threshold_scale}, "
            f"slab_thickness={global_data.slab_thickness}, baseline_method={global_data.baseline_method}, "
            f"min_seed_voxels={global_data.min_seed_voxels}"
            f"{'% ' * 80}\n"
        )
        typer.echo(msg)
        return
    # endregion Testing

    typer.echo("Running through the volume computing residual slices for analysis...")
    begin = time.time()

    memmap_path = PROJECT_ROOT / "Data" / "_tmp"
    if not memmap_path.exists():
        memmap_path.mkdir(parents=True, exist_ok=True)

    residual_memmap_path = memmap_path / "residual_memmap.dat"
    sigma_memmap_path = memmap_path / "sigma_memmap.npy"
    src = None
    try:
        vol: np.memmap = cast(np.memmap, global_data.vol)
        if vol is None:
            raise RuntimeError(
                f"Volume data is None for {global_data.cell_name} (exp_index={exp_index}, vol_index={vol_index})"
            )
        src = precompute_residual(
            cast(ResidualSource, global_data),
            residual_path=str(residual_memmap_path),
            sigma_path=str(sigma_memmap_path),
            chunk=global_data.preprocess_slices_per_chunk,
            residual_dtype=np.dtype(np.int16),
            flush_every=0,
            grayscale_path=vol.filename,
            grayscale_shape=vol.shape,
            grayscale_dtype=vol.dtype,
            grayscale_z_offset=global_data.z_min,
        )
        print(f"Time to execute precompute_residual: {time.time() - begin:.2f} seconds")
        # We don't need the raw volume data anymore, so we can free it to reduce memory usage during candidate detection.
        global_data.free_volume()

        experiment_params = _experiment_param_values(
            global_data.experiment,
            HISTORICAL_MIN_FILL,
            global_data.voxel_size_mm,
            global_data.enforce_shape_gates,
        )
        # z_extent_max is the only EXPERIMENT_PARAM_FIELDS value that's
        # cell-dependent (see experiment_db module docstring) -- echo it onto
        # this cell's candidates via the recorder; volume_name likewise varies
        # per cell within one experiment_number.
        recorder.begin_volume(global_data.cell_name, experiment_params["z_extent_max"])

        if run_parallel:
            _begin = time.time()
            candidates = detect_candidates_parallel(
                src=src,
                k_high=global_data.high_threshold_scale,
                k_low=global_data.low_threshold_scale,
                min_seed_voxels=global_data.min_seed_voxels,
                z_extent_max=global_data.z_extent_max,
                aniso_factor=global_data.aniso_factor,
                z_offset=global_data.z_min,
                slices_per_chunk=global_data.slices_per_chunk,
                small_z_bounds=global_data.small_z_bounds,
                small_voxel_cutoff=global_data.small_vol_cutoff,
                z_pad=global_data.z_pad,
                min_fill=HISTORICAL_MIN_FILL,
                enforce_shape_gates=global_data.enforce_shape_gates,
                metal_threshold=global_data.metal_threshold,
                metal_min_area=global_data.metal_min_area,
                metal_grayscale_margin=global_data.metal_grayscale_margin,
                storage_margin=global_data.storage_margin,
                shell_radius=global_data.shell_radius,
                n_workers=global_data.n_workers,
                on_slab_result=recorder.on_slab_result,
            )
            typer.echo(
                f"Time to execute detect_candidates_parallel: {time.time() - _begin:.2f} seconds"
            )
        else:
            candidates = detect_candidates_streaming(
                src=src,
                k_high=global_data.high_threshold_scale,
                k_low=global_data.low_threshold_scale,
                min_seed_voxels=global_data.min_seed_voxels,
                z_extent_max=global_data.z_extent_max,
                aniso_factor=global_data.aniso_factor,
                z_offset=global_data.z_min,
                slices_per_chunk=global_data.slices_per_chunk,
                small_z_bounds=global_data.small_z_bounds,
                small_voxel_cutoff=global_data.small_vol_cutoff,
                z_pad=global_data.z_pad,
                min_fill=HISTORICAL_MIN_FILL,
                enforce_shape_gates=global_data.enforce_shape_gates,
                metal_threshold=global_data.metal_threshold,
                metal_min_area=global_data.metal_min_area,
                metal_grayscale_margin=global_data.metal_grayscale_margin,
                storage_margin=global_data.storage_margin,
                shell_radius=global_data.shell_radius,
                on_slab_result=recorder.on_slab_result,
            )
    finally:
        # Free the memory used by the residual data to reduce memory usage after candidate detection
        if src is not None:
            src.free_volume()
        residual_memmap_path.unlink(missing_ok=True)
        sigma_memmap_path.unlink(missing_ok=True)

    print(f"Detected {len(candidates)} candidates in the volume.")
    if candidates:
        total_voxels = np.sum([c.n_voxels for c in candidates])
        print(f"Total voxels in detected candidates: {total_voxels}")
        candidate_volumes = [c.n_voxels for c in candidates]
        largest_candidate_idx = np.argmax(candidate_volumes)
        largest_candidate = candidates[largest_candidate_idx]
        largest_location = candidates[largest_candidate_idx].centroid
        lc_z, lc_x, lc_y = largest_location
        print(
            f"Largest candidate has {largest_candidate.n_voxels} voxels at (z, x, y) = ({lc_z:.1f}, {lc_x:.1f}, {lc_y:.1f})."
        )

    # Candidates/features/voxels were already persisted to DuckDB per-slab via
    # recorder.on_slab_result as detection produced them. finish_volume() is
    # the last step -- it writes the `runs` row that marks this
    # (experiment, volume_name) pair done, so it must only happen here, after
    # detection has fully succeeded (not in a `finally`): if anything above
    # raised, we want this volume left un-marked so a retry picks it back up.
    recorder.finish_volume()
    typer.echo(
        f"Recorded {len(candidates)} candidates for {global_data.cell_name} in {time.time() - begin:.2f} seconds."
    )


def main(
    exp_number: int | None = typer.Option(
        None,
        help="Restrict to a single experiment_number from [[experiment]]; defaults to running every experiment",
    ),
    vol_index: int | None = typer.Option(
        None,
        help="Restrict to a single index into [[cell]]; defaults to running every cell",
    ),
    run_parallel: bool = typer.Option(
        True,
        help="Run candidate detection in parallel (default True). Set to False to run in streaming mode, which is slower but uses less memory.",
    ),
) -> None:
    cfg = _load_raw_config(CONFIG_PATH)
    all_experiments = list(enumerate(cfg["experiment"]))
    exp_entries = (
        [
            (e, exp)
            for e, exp in all_experiments
            if exp["experiment_number"] == exp_number
        ]
        if exp_number is not None
        else all_experiments
    )
    if exp_number is not None and not exp_entries:
        raise typer.BadParameter(
            f"No [[experiment]] entry with experiment_number={exp_number}"
        )
    vol_indices = [vol_index] if vol_index is not None else range(len(cfg["cell"]))

    con = db.connect()

    with typer.progressbar(exp_entries, label="Experiments") as exp_progress:
        for e, experiment in exp_progress:
            exp_num = experiment["experiment_number"]

            experiment_row = _experiment_row_values(
                experiment, HISTORICAL_MIN_FILL, cfg["analysis"]["enforce_shape_gates"]
            )
            existing_id = db.get_experiment_id(con, exp_num)
            if existing_id is None:
                experiment_id = db.insert_experiment(con, exp_num, experiment_row)
            else:
                # Resuming: this experiment_number already has some volumes
                # recorded. Reuse its experiment_id rather than inserting a
                # duplicate row (experiment_number is UNIQUE) -- but first make
                # sure config.toml hasn't drifted since those volumes were
                # run, since candidates for different config under one
                # experiment_id would be silently inconsistent.
                stored_row = db.get_experiment_row(con, exp_num)
                if stored_row != experiment_row:
                    raise typer.BadParameter(
                        f"experiment_number={exp_num} already has experiment_id="
                        f"{existing_id} in {db.DB_PATH}, but its recorded config "
                        f"differs from config.toml's current [[experiment]] entry:\n"
                        f"  stored in DB:  {stored_row}\n"
                        f"  in config.toml: {experiment_row}\n"
                        "Either revert config.toml, or clear the whole experiment "
                        f"(`uv run scripts/clear_experiment.py {exp_num}`) before "
                        "adding more volumes to it."
                    )
                experiment_id = existing_id

            already_done = db.recorded_volumes(con, experiment_id)
            pending = [
                v for v in vol_indices if cfg["cell"][v]["name"] not in already_done
            ]
            skipped_names = [
                cfg["cell"][v]["name"]
                for v in vol_indices
                if cfg["cell"][v]["name"] in already_done
            ]
            if skipped_names:
                typer.echo(
                    f"Experiment {exp_num}: skipping {len(skipped_names)} "
                    f"already-recorded volume(s): {skipped_names}"
                )
            if not pending:
                typer.echo(
                    f"Experiment {exp_num}: nothing to do -- every requested "
                    "volume is already recorded."
                )
                continue

            recorder = db.ExperimentRecorder(con, experiment_id)
            typer.echo(f"--- Experiment {exp_num}: experiment_id={experiment_id} ---")
            with typer.progressbar(
                pending, label=f"Experiment {exp_num} cells"
            ) as vol_progress:
                for v in vol_progress:
                    _run_one(e, v, recorder, run_parallel=run_parallel)
            typer.echo(
                f"Experiment {exp_num}: {recorder.n_candidates} candidates, "
                f"{recorder.n_voxels} voxels recorded in {db.DB_PATH.name}."
            )
    con.close()


if __name__ == "__main__":
    typer.run(main)
