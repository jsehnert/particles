"""Standalone particle-noise and rotation experiment tools.

This module owns measurement persistence, reference validation, z-slab noise
rotation, and particle-neighborhood rotation. Sweep configuration and CLI
execution live in :mod:`scripts.particle_sim_sweep`.
"""

from __future__ import annotations

import copy
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Callable

import duckdb
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from numpy.typing import DTypeLike, NDArray
from scipy.ndimage import binary_erosion
from tqdm.auto import tqdm

from particle_sim_tools import (
    Particle,
    ParticleStateMeasurement,
    ParticleTransformationMeasurement,
    ParticleVolume,
    ParticleVolumeUint,
    StandardNoiseVolume,
    ValidXBounds,
    Volume,
    backward_rotation_map,
    create_uniform_particle_volume,
    estimate_volume_noise,
    forward_align_points_zyx,
    valid_x_bounds_after_alignment,
)

if TYPE_CHECKING:
    from scripts.particle_sim_sweep import ParticleRotationSweepConfig


def get_mean_from_volume(volume_data: NDArray) -> float:
    """
    Return the mean from the uint volume accounting for the zero padding that is induced from any rotations

    Since the data sits well above zero, using the assumed zero padding is safe. We buffer a bit to account
    for high order interpolation errors.
    """
    valid_mask = volume_data > 0
    valid_mask = binary_erosion(valid_mask, structure=np.ones((5, 5, 5)))
    return volume_data[valid_mask].mean()


NO_NOISE_SEED = -1

def measure_particle_states(
    particle_volume: ParticleVolumeUint,
    observed_data: NDArray,
    noise_mean: float,
    noise_profile: NDArray,
    state: tuple[DTypeLike, float, float],
) -> list[ParticleStateMeasurement]:
    """Measure observed data at locations ranked by particle-only signal."""
    return [
        ParticleStateMeasurement.from_particle(
            particle,
            particle_only_data=particle_volume.get_array(),
            combined_data=observed_data,
            state=state,
            noise_mean=noise_mean,
            noise_std_profile=noise_profile,
            top_k=particle.measurement_top_k,
        )
        for particle in particle_volume.particles
    ]


def add_experiment_row(
    rows: list[dict[str, object]],
    seed: int,
    bit_depth: type,
    particle_sigma: float,
    amplitude_snr: float,
    interpolation_order: int,
    theta: float,
    phi: float,
    volume_center_yx: np.ndarray,
    clean_before: ParticleStateMeasurement,
    clean_after: ParticleStateMeasurement,
    noisy_before: ParticleStateMeasurement,
    noisy_after: ParticleStateMeasurement,
) -> None:
    has_noise = seed != NO_NOISE_SEED
    before = noisy_before if has_noise else clean_before
    after = noisy_after if has_noise else clean_after
    change = ParticleTransformationMeasurement(before, after)
    position = before.position_zyx
    rotated_position = after.position_zyx

    rows.append(
        {
            "noise_seed": seed,
            "noise_condition": "realized" if has_noise else "flat_mean",
            "bit_depth": np.dtype(bit_depth).name,
            "noise_mean_before": before.noise_mean,
            "noise_mean_after": after.noise_mean,
            "particle_sigma": particle_sigma,
            "amplitude_snr": amplitude_snr,
            "interpolation_order": interpolation_order,
            "theta_deg": theta,
            "phi_deg": phi,
            "particle_id": before.particle_id,
            "top_k": before.top_k,
            "z": position[0],
            "y": position[1],
            "x": position[2],
            "radius_xy": float(np.linalg.norm(position[1:] - volume_center_yx)),
            "rotated_z": rotated_position[0],
            "rotated_y": rotated_position[1],
            "rotated_x": rotated_position[2],
            "peak_contrast_before": before.peak_contrast,
            "peak_contrast_after": after.peak_contrast,
            "peak_contrast_retention": change.peak_contrast_retention,
            "peak_contrast_drop": change.peak_contrast_drop,
            "integrated_contrast_before": before.integrated_contrast,
            "integrated_contrast_after": after.integrated_contrast,
            "integrated_contrast_retention": change.integrated_contrast_retention,
            "integrated_contrast_drop": change.integrated_contrast_drop,
            "peak_cnr_before": before.peak_cnr,
            "peak_cnr_after": after.peak_cnr,
            "peak_cnr_retention": change.peak_cnr_retention,
            "integrated_cnr_before": before.integrated_cnr,
            "integrated_cnr_after": after.integrated_cnr,
            "integrated_cnr_retention": change.integrated_cnr_retention,
        }
    )


@dataclass(frozen=True)
class RotatedNoiseState:
    mean: float
    profile: NDArray[np.float64]


def theta_stage_volume(source: Volume, theta: float, order: int) -> Volume:
    """Apply the first alignment rotation, or reuse the source at zero theta."""
    return source if theta == 0.0 else source.align(theta=theta, phi=0.0, order=order)


def finish_phi_rotation(
    theta_stage: Volume,
    theta: float,
    phi: float,
    order: int,
) -> Volume:
    """Apply phi to a reusable theta-stage volume."""
    if phi == 0.0:
        return theta_stage

    rotated = copy.copy(theta_stage)
    rotated.data = ndimage.rotate(
        theta_stage.data,
        angle=-phi,
        axes=(0, 2),
        reshape=False,
        order=order,
        mode="constant",
        cval=0.0,
        output=theta_stage.data.dtype,
        prefilter=order > 1,
    )
    interpolation_margin = 1.0 if order == 1 else float(2 * order)
    rotated.valid_bounds = valid_x_bounds_after_alignment(
        theta_stage.data.shape,
        theta,
        phi,
        margin=interpolation_margin,
    )
    if theta_stage.cylinder is not None:
        cylinder = copy.copy(theta_stage.cylinder)
        cylinder.phi -= phi
        rotated.cylinder = cylinder
    return rotated


def theta_stage_particles(
    source: ParticleVolumeUint,
    theta: float,
    order: int,
) -> ParticleVolumeUint:
    """Apply theta while preserving particle measurement metadata."""
    return source if theta == 0.0 else source.rotate(theta=theta, phi=0.0, order=order)


def finish_particle_phi_rotation(
    source: ParticleVolumeUint,
    theta_stage: ParticleVolumeUint,
    theta: float,
    phi: float,
    order: int,
) -> ParticleVolumeUint:
    """Apply phi and rebuild particle metadata for the complete rotation."""
    if phi == 0.0:
        return theta_stage

    rotated = copy.copy(theta_stage)
    rotated.volume = finish_phi_rotation(theta_stage.volume, theta, phi, order)
    rotated.particles = ParticleVolume.rotate_particles(
        particles=source.particles,
        rotated_particle_data=rotated.get_array(),
        theta=theta,
        phi=phi,
        order=order,
    )
    return rotated


def sample_observed_states(
    references: list[ParticleStateMeasurement],
    observed_data: NDArray,
    noise_state: RotatedNoiseState,
) -> list[ParticleStateMeasurement]:
    """Sample an observed volume at locations fixed by clean references."""
    noise_std_by_z = ParticleStateMeasurement._to_plane_noise_std(
        noise_state.profile,
        observed_data.shape[0],
    )
    measurements = []
    for reference in references:
        indices = reference.top_k_zyx
        observed_values = observed_data[indices[:, 0], indices[:, 1], indices[:, 2]]
        measurements.append(
            ParticleStateMeasurement(
                particle_id=reference.particle_id,
                state=reference.state,
                position_zyx=reference.position_zyx,
                top_k=reference.top_k,
                top_k_zyx=indices,
                particle_only_top_k_values=reference.particle_only_top_k_values,
                combined_top_k_values=observed_values,
                noise_mean=noise_state.mean,
                noise_std_by_z=noise_std_by_z,
            )
        )
    return measurements


PARTICLE_SWEEP_OUTPUT_CSV = Path(
    "Data/simulated_particles/particle_experiment_metrics.csv"
)


@dataclass(frozen=True)
class ParticleSweepRun:
    output_csv: Path
    rows_written: int
    elapsed_seconds: float
    timings: pd.DataFrame = field(repr=False)


def rotation_worker_count(order: int, n_theta: int) -> int:
    return min(3 if order == 1 else 1, n_theta)


def run_theta_workers(
    theta_angles: tuple[float, ...],
    order: int,
    worker,
) -> tuple[list, int]:
    workers = rotation_worker_count(order, len(theta_angles))
    if workers == 1:
        return [worker(theta) for theta in theta_angles], workers
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, theta_angles)), workers


def measure_clean_theta_group(
    clean_base: ParticleVolumeUint,
    clean_before: list[ParticleStateMeasurement],
    clean_background: float,
    flat_noise_profile: NDArray,
    interpolation_order: int,
    phi_angles: tuple[float, ...],
    theta: float,
) -> tuple[float, dict[float, list[ParticleStateMeasurement]]]:
    theta_stage = theta_stage_particles(clean_base, theta, interpolation_order)
    bit_depth = clean_base.get_array().dtype.type
    measurements_by_phi = {}
    for phi in phi_angles:
        if theta == 0.0 and phi == 0.0:
            measurements_by_phi[phi] = clean_before
            continue
        rotated = finish_particle_phi_rotation(
            clean_base,
            theta_stage,
            theta,
            phi,
            interpolation_order,
        )
        measurements_by_phi[phi] = measure_particle_states(
            rotated,
            rotated.get_array(),
            clean_background,
            flat_noise_profile,
            (bit_depth, theta, phi),
        )
    return theta, measurements_by_phi


def measure_noise_theta_group(
    noise_base: Volume,
    base_noise_state: RotatedNoiseState,
    interpolation_order: int,
    phi_angles: tuple[float, ...],
    theta: float,
) -> tuple[float, dict[float, RotatedNoiseState]]:
    theta_stage = theta_stage_volume(noise_base, theta, interpolation_order)
    states_by_phi = {}
    for phi in phi_angles:
        if theta == 0.0 and phi == 0.0:
            states_by_phi[phi] = base_noise_state
            continue
        rotated = finish_phi_rotation(
            theta_stage,
            theta,
            phi,
            interpolation_order,
        )
        noise_profile, _ = estimate_volume_noise(rotated)
        states_by_phi[phi] = RotatedNoiseState(
            mean=get_mean_from_volume(rotated.data),
            profile=noise_profile,
        )
    return theta, states_by_phi


def measure_noisy_theta_group(
    noisy_base: Volume,
    clean_states: dict[tuple[int, float, float], list[ParticleStateMeasurement]],
    noise_states: dict[tuple[int, float, float], RotatedNoiseState],
    interpolation_order: int,
    phi_angles: tuple[float, ...],
    theta: float,
) -> tuple[float, dict[float, list[ParticleStateMeasurement]]]:
    theta_stage = theta_stage_volume(noisy_base, theta, interpolation_order)
    measurements_by_phi = {}
    for phi in phi_angles:
        rotated = (
            noisy_base
            if theta == 0.0 and phi == 0.0
            else finish_phi_rotation(
                theta_stage,
                theta,
                phi,
                interpolation_order,
            )
        )
        state_key = (interpolation_order, theta, phi)
        measurements_by_phi[phi] = sample_observed_states(
            clean_states[state_key],
            rotated.data,
            noise_states[state_key],
        )
    return theta, measurements_by_phi


def write_condition_rows(
    output_csv: Path,
    seed: int,
    bit_depth: type,
    particle_sigma: float,
    amplitude_snr: float,
    interpolation_order: int,
    theta: float,
    phi: float,
    volume_center_yx: NDArray,
    clean_before: list[ParticleStateMeasurement],
    clean_after: list[ParticleStateMeasurement],
    observed_before: list[ParticleStateMeasurement],
    observed_after: list[ParticleStateMeasurement],
) -> tuple[int, float]:
    rows = []
    for measurements in zip(
        clean_before,
        clean_after,
        observed_before,
        observed_after,
        strict=True,
    ):
        add_experiment_row(
            rows,
            seed,
            bit_depth,
            particle_sigma,
            amplitude_snr,
            interpolation_order,
            theta,
            phi,
            volume_center_yx,
            *measurements,
        )

    write_started = time.perf_counter()
    pd.DataFrame.from_records(rows).to_csv(
        output_csv,
        mode="a",
        header=not output_csv.exists(),
        index=False,
    )
    return len(rows), time.perf_counter() - write_started


SWEEP_CONDITION_COLUMNS = (
    "noise_seed",
    "bit_depth",
    "particle_sigma",
    "amplitude_snr",
    "interpolation_order",
    "theta_deg",
    "phi_deg",
    "top_k",
)


def expected_particle_ids(config: ParticleRotationSweepConfig) -> frozenset[int]:
    """Return the particle IDs produced by the configured uniform grid."""
    radius = config.voxels_xy // 2
    delta_radius = config.voxels_xy // 8
    radii = []
    value = delta_radius
    while value < radius:
        radii.append(value)
        value += delta_radius
    angles = np.arange(
        0,
        np.deg2rad(config.quadrant_stop_deg)
        - np.deg2rad(config.radial_arm_spacing_deg)
        + 1e-6,
        np.deg2rad(config.radial_arm_spacing_deg),
    )
    z_positions = np.arange(
        round(config.z_start_rel * config.voxels_z),
        round(config.z_end_rel * config.voxels_z),
        round(config.z_delta_rel * config.voxels_z),
    )
    return frozenset(range(len(radii) * len(angles) * len(z_positions)))


def sweep_condition_key(
    seed: int,
    bit_depth: type,
    particle_sigma: float,
    amplitude_snr: float,
    interpolation_order: int,
    theta: float,
    phi: float,
    top_k: int,
) -> tuple[object, ...]:
    """Normalize a persisted sweep-condition key for resume checks."""
    return (
        int(seed),
        np.dtype(bit_depth).name,
        float(particle_sigma),
        float(amplitude_snr),
        int(interpolation_order),
        float(theta),
        float(phi),
        int(top_k),
    )


def prepare_resumable_output(
    output_csv: Path,
    expected_ids: frozenset[int],
    reset_output: bool,
) -> set[tuple[object, ...]]:
    """Reset or sanitize the CSV and return conditions with complete rows."""
    if reset_output:
        output_csv.unlink(missing_ok=True)
        return set()
    if not output_csv.exists():
        return set()

    existing = pd.read_csv(output_csv)
    missing_columns = set(SWEEP_CONDITION_COLUMNS) | {"particle_id"}
    missing_columns -= set(existing.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot resume {output_csv}: missing columns {sorted(missing_columns)}. "
            "Use reset_output=True to create a fresh results table."
        )

    complete = set()
    for key, group in existing.groupby(list(SWEEP_CONDITION_COLUMNS), dropna=False):
        ids = frozenset(group["particle_id"].astype(int))
        if ids == expected_ids and len(group) == len(expected_ids):
            complete.add(tuple(key))

    row_keys = list(existing[list(SWEEP_CONDITION_COLUMNS)].itertuples(index=False, name=None))
    keep = np.fromiter((key in complete for key in row_keys), dtype=bool)
    if not keep.all():
        removed = int((~keep).sum())
        existing.loc[keep].to_csv(output_csv, index=False)
        print(f"Removed {removed} incomplete or duplicate rows before resuming.")
    return complete


def update_sweep_progress(
    progress_bar: tqdm,
    phase: str,
    seed: int,
    bit_depth: type,
    particle_sigma: float | None,
    amplitude_snr: float | None,
    interpolation_order: int,
    theta: float,
    phi: float,
    advance: bool,
) -> None:
    progress_bar.set_postfix(
        phase=phase,
        seed=seed,
        dtype=np.dtype(bit_depth).name,
        sigma="-" if particle_sigma is None else f"{particle_sigma:g}",
        snr="-" if amplitude_snr is None else f"{amplitude_snr:g}",
        order=interpolation_order,
        theta=f"{theta:g}",
        phi=f"{phi:g}",
        refresh=False,
    )
    if advance:
        progress_bar.update()


def timing_record(
    phase: str,
    bit_depth: type,
    noise_seed: int,
    interpolation_order: int | None,
    workers: int,
    conditions_written: int,
    compute_seconds: float,
    write_seconds: float = 0.0,
    particle_sigma: float | None = None,
    amplitude_snr: float | None = None,
) -> dict[str, object]:
    return {
        "phase": phase,
        "bit_depth": np.dtype(bit_depth).name,
        "noise_seed": noise_seed,
        "particle_sigma": particle_sigma,
        "amplitude_snr": amplitude_snr,
        "interpolation_order": interpolation_order,
        "workers": workers,
        "conditions_written": conditions_written,
        "compute_seconds": compute_seconds,
        "write_seconds": write_seconds,
        "elapsed_seconds": compute_seconds + write_seconds,
    }


def run_particle_rotation_sweep(
    config: ParticleRotationSweepConfig,
    output_csv: Path = PARTICLE_SWEEP_OUTPUT_CSV,
) -> ParticleSweepRun:
    """Run the sweep with reusable rotations and condition-level CSV writes."""
    output_csv = Path(output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.unlink(missing_ok=True)
    print(f"Writing particle metrics to {output_csv}")

    total_conditions = (
        len(config.noise_seeds)
        * len(config.bit_depths)
        * len(config.particle_sigmas)
        * len(config.amplitude_snrs)
        * len(config.interpolation_orders)
        * len(config.theta_angles_deg)
        * len(config.phi_angles_deg)
    )
    volume_center_yx = np.array([config.voxels_xy // 2] * 2, dtype=np.float64)
    realized_seeds = tuple(seed for seed in config.noise_seeds if seed != NO_NOISE_SEED)
    rows_written = 0
    timing_records = []
    sweep_started = time.perf_counter()

    with tqdm(
        total=total_conditions,
        desc="Particle sweep",
        unit="condition",
        dynamic_ncols=True,
    ) as progress_bar:
        for bit_depth in config.bit_depths:
            clean_cache = {}

            for particle_sigma in config.particle_sigmas:
                for amplitude_snr in config.amplitude_snrs:
                    setup_started = time.perf_counter()
                    particle_fp32 = create_uniform_particle_volume(
                        voxels_z=config.voxels_z,
                        voxels_xy=config.voxels_xy,
                        sigma=particle_sigma,
                        amplitude=amplitude_snr,
                        z_start_rel=config.z_start_rel,
                        z_end_rel=config.z_end_rel,
                        delta_z_rel=config.z_delta_rel,
                        delta_angle_deg=config.radial_arm_spacing_deg,
                        stop_angle_deg=config.quadrant_stop_deg,
                        measurement_top_k=config.measurement_top_k,
                    )
                    clean_base = ParticleVolumeUint(
                        particle_fp32,
                        type=bit_depth,
                        target_std_scale=config.target_std_scale,
                    )
                    clean_background = float(np.rint(np.iinfo(bit_depth).max / 2))
                    flat_noise_profile = np.full(config.voxels_z, np.nan)
                    clean_before = measure_particle_states(
                        clean_base,
                        clean_base.get_array(),
                        clean_background,
                        flat_noise_profile,
                        (bit_depth, 0.0, 0.0),
                    )
                    timing_records.append(
                        timing_record(
                            phase="clean_setup",
                            bit_depth=bit_depth,
                            noise_seed=NO_NOISE_SEED,
                            interpolation_order=None,
                            workers=1,
                            conditions_written=0,
                            compute_seconds=time.perf_counter() - setup_started,
                            particle_sigma=particle_sigma,
                            amplitude_snr=amplitude_snr,
                        )
                    )

                    clean_states = {}
                    for interpolation_order in config.interpolation_orders:
                        compute_started = time.perf_counter()
                        worker = partial(
                            measure_clean_theta_group,
                            clean_base,
                            clean_before,
                            clean_background,
                            flat_noise_profile,
                            interpolation_order,
                            config.phi_angles_deg,
                        )
                        theta_results, workers = run_theta_workers(
                            config.theta_angles_deg,
                            interpolation_order,
                            worker,
                        )
                        compute_seconds = time.perf_counter() - compute_started
                        write_seconds = 0.0
                        conditions_written = 0

                        for theta, measurements_by_phi in theta_results:
                            for phi in config.phi_angles_deg:
                                state_key = (interpolation_order, theta, phi)
                                clean_after = measurements_by_phi[phi]
                                clean_states[state_key] = clean_after
                                if not condition_is_pending(
                                    NO_NOISE_SEED,
                                    bit_depth,
                                    particle_sigma,
                                    amplitude_snr,
                                    interpolation_order,
                                    theta,
                                    phi,
                                ):
                                    continue
                                row_count, condition_write_seconds = (
                                    write_condition_rows(
                                        output_csv,
                                        NO_NOISE_SEED,
                                        bit_depth,
                                        particle_sigma,
                                        amplitude_snr,
                                        interpolation_order,
                                        theta,
                                        phi,
                                        volume_center_yx,
                                        clean_before,
                                        clean_after,
                                        clean_before,
                                        clean_after,
                                    )
                                )
                                rows_written += row_count
                                write_seconds += condition_write_seconds
                                conditions_written += 1
                                update_sweep_progress(
                                    progress_bar,
                                    "clean",
                                    NO_NOISE_SEED,
                                    bit_depth,
                                    particle_sigma,
                                    amplitude_snr,
                                    interpolation_order,
                                    theta,
                                    phi,
                                    advance=True,
                                )

                        timing_records.append(
                            timing_record(
                                phase="clean_rotation",
                                bit_depth=bit_depth,
                                noise_seed=NO_NOISE_SEED,
                                interpolation_order=interpolation_order,
                                workers=workers,
                                conditions_written=conditions_written,
                                compute_seconds=compute_seconds,
                                write_seconds=write_seconds,
                                particle_sigma=particle_sigma,
                                amplitude_snr=amplitude_snr,
                            )
                        )

                    clean_cache[(particle_sigma, amplitude_snr)] = (
                        clean_before,
                        clean_states,
                    )
                    del clean_base, particle_fp32

            for seed in realized_seeds:
                noise_setup_started = time.perf_counter()
                np.random.seed(seed)
                standard_noise = StandardNoiseVolume(
                    voxels_z=config.voxels_z,
                    voxels_xy=config.voxels_xy,
                    rho1_z=config.rho1_z,
                    rho1_xy=config.rho1_xy,
                )
                target_std = np.iinfo(bit_depth).max * config.target_std_scale
                noise_base = standard_noise.get_quantized_noise(
                    target_std=target_std,
                    uint_type=bit_depth,
                )
                base_noise_profile, _ = estimate_volume_noise(noise_base)
                base_noise_state = RotatedNoiseState(
                    mean=get_mean_from_volume(noise_base.data),
                    profile=base_noise_profile,
                )
                timing_records.append(
                    timing_record(
                        phase="noise_setup",
                        bit_depth=bit_depth,
                        noise_seed=seed,
                        interpolation_order=None,
                        workers=1,
                        conditions_written=0,
                        compute_seconds=time.perf_counter() - noise_setup_started,
                    )
                )

                noise_states = {}
                for interpolation_order in config.interpolation_orders:
                    update_sweep_progress(
                        progress_bar,
                        "noise cache",
                        seed,
                        bit_depth,
                        None,
                        None,
                        interpolation_order,
                        0.0,
                        0.0,
                        advance=False,
                    )
                    compute_started = time.perf_counter()
                    worker = partial(
                        measure_noise_theta_group,
                        noise_base,
                        base_noise_state,
                        interpolation_order,
                        config.phi_angles_deg,
                    )
                    theta_results, workers = run_theta_workers(
                        config.theta_angles_deg,
                        interpolation_order,
                        worker,
                    )
                    for theta, states_by_phi in theta_results:
                        for phi, noise_state in states_by_phi.items():
                            noise_states[(interpolation_order, theta, phi)] = (
                                noise_state
                            )
                    timing_records.append(
                        timing_record(
                            phase="noise_rotation",
                            bit_depth=bit_depth,
                            noise_seed=seed,
                            interpolation_order=interpolation_order,
                            workers=workers,
                            conditions_written=0,
                            compute_seconds=time.perf_counter() - compute_started,
                        )
                    )
                del noise_base

                for particle_sigma in config.particle_sigmas:
                    for amplitude_snr in config.amplitude_snrs:
                        noisy_setup_started = time.perf_counter()
                        particle_fp32 = create_uniform_particle_volume(
                            voxels_z=config.voxels_z,
                            voxels_xy=config.voxels_xy,
                            sigma=particle_sigma,
                            amplitude=amplitude_snr,
                            z_start_rel=config.z_start_rel,
                            z_end_rel=config.z_end_rel,
                            delta_z_rel=config.z_delta_rel,
                            delta_angle_deg=config.radial_arm_spacing_deg,
                            stop_angle_deg=config.quadrant_stop_deg,
                            measurement_top_k=config.measurement_top_k,
                        )
                        particle_fp32.get_array()[:] += standard_noise.get_array()
                        noisy_base = ParticleVolumeUint(
                            particle_fp32,
                            type=bit_depth,
                            target_std_scale=config.target_std_scale,
                        )
                        clean_before, clean_states = clean_cache[
                            (particle_sigma, amplitude_snr)
                        ]
                        noisy_before = sample_observed_states(
                            clean_before,
                            noisy_base.get_array(),
                            base_noise_state,
                        )
                        timing_records.append(
                            timing_record(
                                phase="noisy_setup",
                                bit_depth=bit_depth,
                                noise_seed=seed,
                                interpolation_order=None,
                                workers=1,
                                conditions_written=0,
                                compute_seconds=(
                                    time.perf_counter() - noisy_setup_started
                                ),
                                particle_sigma=particle_sigma,
                                amplitude_snr=amplitude_snr,
                            )
                        )

                        for interpolation_order in config.interpolation_orders:
                            compute_started = time.perf_counter()
                            worker = partial(
                                measure_noisy_theta_group,
                                noisy_base.volume,
                                clean_states,
                                noise_states,
                                interpolation_order,
                                config.phi_angles_deg,
                            )
                            theta_results, workers = run_theta_workers(
                                config.theta_angles_deg,
                                interpolation_order,
                                worker,
                            )
                            compute_seconds = time.perf_counter() - compute_started
                            write_seconds = 0.0
                            conditions_written = 0

                            for theta, measurements_by_phi in theta_results:
                                for phi in config.phi_angles_deg:
                                    state_key = (interpolation_order, theta, phi)
                                    if not condition_is_pending(
                                        seed,
                                        bit_depth,
                                        particle_sigma,
                                        amplitude_snr,
                                        interpolation_order,
                                        theta,
                                        phi,
                                    ):
                                        continue
                                    row_count, condition_write_seconds = (
                                        write_condition_rows(
                                            output_csv,
                                            seed,
                                            bit_depth,
                                            particle_sigma,
                                            amplitude_snr,
                                            interpolation_order,
                                            theta,
                                            phi,
                                            volume_center_yx,
                                            clean_before,
                                            clean_states[state_key],
                                            noisy_before,
                                            measurements_by_phi[phi],
                                        )
                                    )
                                    rows_written += row_count
                                    write_seconds += condition_write_seconds
                                    conditions_written += 1
                                    update_sweep_progress(
                                        progress_bar,
                                        "noisy",
                                        seed,
                                        bit_depth,
                                        particle_sigma,
                                        amplitude_snr,
                                        interpolation_order,
                                        theta,
                                        phi,
                                        advance=True,
                                    )

                            timing_records.append(
                                timing_record(
                                    phase="noisy_rotation",
                                    bit_depth=bit_depth,
                                    noise_seed=seed,
                                    interpolation_order=interpolation_order,
                                    workers=workers,
                                    conditions_written=conditions_written,
                                    compute_seconds=compute_seconds,
                                    write_seconds=write_seconds,
                                    particle_sigma=particle_sigma,
                                    amplitude_snr=amplitude_snr,
                                )
                            )

                        del noisy_base, particle_fp32

                del standard_noise, noise_states

            del clean_cache

    elapsed_seconds = time.perf_counter() - sweep_started
    timings = pd.DataFrame.from_records(timing_records)
    timing_summary = timings.groupby(
        ["phase", "interpolation_order"],
        dropna=False,
    )[["compute_seconds", "write_seconds", "elapsed_seconds"]].sum()
    print("\nSweep timing summary (seconds):")
    print(timing_summary.to_string())
    return ParticleSweepRun(
        output_csv=output_csv,
        rows_written=rows_written,
        elapsed_seconds=elapsed_seconds,
        timings=timings,
    )


def summarize_particle_metrics_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    condition_columns = [
        "noise_condition",
        "bit_depth",
        "particle_sigma",
        "amplitude_snr",
        "interpolation_order",
        "theta_deg",
        "phi_deg",
        # Preserve the original particle location: rotation effects are anisotropic.
        "particle_id",
        "z",
        "y",
        "x",
        "radius_xy",
    ]
    retention_columns = [
        "peak_contrast_retention",
        "integrated_contrast_retention",
        "peak_cnr_retention",
        "integrated_cnr_retention",
    ]
    per_seed = metrics.groupby(
        condition_columns + ["noise_seed"], as_index=False, dropna=False
    )[retention_columns].mean()
    long_per_seed = per_seed.melt(
        id_vars=condition_columns + ["noise_seed"],
        value_vars=retention_columns,
        var_name="metric",
        value_name="seed_mean",
    )
    summary = long_per_seed.groupby(
        condition_columns + ["metric"], as_index=False, dropna=False
    )["seed_mean"].agg(mean="mean", std="std", n_seeds="count")
    summary["sem"] = summary["std"] / np.sqrt(summary["n_seeds"])
    summary["ci95_half_width"] = 1.96 * summary["sem"]
    return summary


def summarize_particle_rotation_sweep(
    metrics: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Summarize a metrics frame or query the persisted CSV directly."""
    if isinstance(metrics, pd.DataFrame):
        return summarize_particle_metrics_frame(metrics)

    query = """
        WITH per_seed AS (
            SELECT
                noise_condition,
                bit_depth,
                particle_sigma,
                amplitude_snr,
                interpolation_order,
                theta_deg,
                phi_deg,
                particle_id,
                z,
                y,
                x,
                radius_xy,
                noise_seed,
                avg(peak_contrast_retention) AS peak_contrast_retention,
                avg(integrated_contrast_retention) AS integrated_contrast_retention,
                avg(peak_cnr_retention) AS peak_cnr_retention,
                avg(integrated_cnr_retention) AS integrated_cnr_retention
            FROM read_csv_auto(?)
            GROUP BY ALL
        ),
        long_per_seed AS (
            SELECT *, 'peak_contrast_retention' AS metric,
                peak_contrast_retention AS seed_mean FROM per_seed
            UNION ALL
            SELECT *, 'integrated_contrast_retention' AS metric,
                integrated_contrast_retention AS seed_mean FROM per_seed
            UNION ALL
            SELECT *, 'peak_cnr_retention' AS metric,
                peak_cnr_retention AS seed_mean FROM per_seed
            UNION ALL
            SELECT *, 'integrated_cnr_retention' AS metric,
                integrated_cnr_retention AS seed_mean FROM per_seed
        ),
        aggregated AS (
            SELECT
                noise_condition,
                bit_depth,
                particle_sigma,
                amplitude_snr,
                interpolation_order,
                theta_deg,
                phi_deg,
                particle_id,
                z,
                y,
                x,
                radius_xy,
                metric,
                avg(seed_mean) AS mean,
                stddev_samp(seed_mean) AS std,
                count(seed_mean) AS n_seeds
            FROM long_per_seed
            GROUP BY ALL
        )
        SELECT
            *,
            std / sqrt(n_seeds) AS sem,
            1.96 * std / sqrt(n_seeds) AS ci95_half_width
        FROM aggregated
        ORDER BY
            noise_condition,
            bit_depth,
            particle_sigma,
            amplitude_snr,
            interpolation_order,
            theta_deg,
            phi_deg,
            particle_id,
            z,
            y,
            x,
            radius_xy,
            metric
    """
    with duckdb.connect() as connection:
        return connection.execute(query, [str(Path(metrics).resolve())]).fetchdf()


# Z-slab optimization

OPTIMIZED_PARTICLE_SWEEP_OUTPUT_CSV = PARTICLE_SWEEP_OUTPUT_CSV.with_name(
    "particle_experiment_metrics_slab_optimized.csv"
)
REFERENCE_SECONDS_PER_CONDITION = 392.0


@dataclass(frozen=True)
class ZSlabOptimizationSettings:
    """Controls for the moderate z-slab optimization."""

    rotation_workers: int = 2
    spline_halo: int = 32
    profile_guard: int = 2

    def __post_init__(self) -> None:
        if self.rotation_workers < 1:
            raise ValueError("rotation_workers must be at least one")
        if self.spline_halo < 0:
            raise ValueError("spline_halo must be nonnegative")
        if self.profile_guard < 1:
            raise ValueError("profile_guard must be at least one")


@dataclass(frozen=True)
class ParticleZSlabBounds:
    """Half-open global z intervals used by the slab rotations."""

    input_z_start: int
    input_z_stop: int
    output_z_start: int
    output_z_stop: int
    global_shape: tuple[int, int, int]

    @property
    def input_depth(self) -> int:
        return self.input_z_stop - self.input_z_start

    @property
    def output_depth(self) -> int:
        return self.output_z_stop - self.output_z_start


def slab_reference_particles(config: ParticleRotationSweepConfig) -> list[Particle]:
    """Construct only particle metadata for conservative slab-bound calculation."""
    z_start = round(config.z_start_rel * config.voxels_z)
    z_end = round(config.z_end_rel * config.voxels_z)
    delta_z = round(config.z_delta_rel * config.voxels_z)
    positions_z = np.arange(z_start, z_end, delta_z)

    delta_radius = config.voxels_xy // 8
    radii = np.arange(delta_radius, config.voxels_xy // 2, delta_radius)
    angle_stop = np.deg2rad(config.quadrant_stop_deg)
    angle_step = np.deg2rad(config.radial_arm_spacing_deg)
    angles = np.arange(0.0, angle_stop - angle_step + 1e-6, angle_step)

    positions_yx = [
        (
            radius * np.sin(angle) + config.voxels_xy // 2,
            radius * np.cos(angle) + config.voxels_xy // 2,
        )
        for angle in angles
        for radius in radii
    ]
    sigma = max(config.particle_sigmas)
    amplitude = max(config.amplitude_snrs)
    return [
        Particle(
            position_zyx=np.asarray((z, y, x), dtype=np.float64),
            amplitude_snr=amplitude,
            spatial_sigma=sigma,
            particle_id=particle_id,
            measurement_top_k=config.measurement_top_k,
        )
        for particle_id, (z, (y, x)) in enumerate(
            itertools.product(positions_z, positions_yx)
        )
    ]


def particle_support_box_corners(particle: Particle) -> NDArray[np.float64]:
    lower = particle.support_zyx.min(axis=0)
    upper = particle.support_zyx.max(axis=0)
    return np.asarray(
        list(itertools.product(*zip(lower, upper, strict=True))),
        dtype=np.float64,
    )


def calculate_particle_z_slab_bounds(
    config: ParticleRotationSweepConfig,
    settings: ZSlabOptimizationSettings = ZSlabOptimizationSettings(),
) -> ParticleZSlabBounds:
    """Calculate conservative input/output z slabs in full-volume coordinates."""
    global_shape = (config.voxels_z, config.voxels_xy, config.voxels_xy)
    particles = slab_reference_particles(config)
    support_corners = np.concatenate(
        [particle_support_box_corners(particle) for particle in particles]
    )

    transformed_z_min = np.inf
    transformed_z_max = -np.inf
    for theta in config.theta_angles_deg:
        for phi in config.phi_angles_deg:
            transformed = forward_align_points_zyx(
                support_corners,
                theta,
                phi,
                global_shape,
            )
            transformed_z_min = min(transformed_z_min, transformed[:, 0].min())
            transformed_z_max = max(transformed_z_max, transformed[:, 0].max())

    measurement_halo = max(
        1 if order == 1 else 2 * order for order in config.interpolation_orders
    )
    # The estimator copies its first and last valid interior estimates onto the
    # endpoint planes. Keep those copied values outside the particle/measurement
    # region by retaining one additional output plane on each side.
    noise_endpoint_padding = 1
    output_padding = measurement_halo + settings.profile_guard + noise_endpoint_padding
    output_z_start = max(0, int(np.floor(transformed_z_min)) - output_padding)
    output_z_stop = min(
        config.voxels_z,
        int(np.ceil(transformed_z_max)) + output_padding + 1,
    )

    # Back-project the complete output x extent through every phi transform.
    # The theta rotation does not mix z with another coordinate.
    input_z_min = np.inf
    input_z_max = -np.inf
    output_z_edges = (output_z_start, output_z_stop - 1)
    output_x_edges = (0, config.voxels_xy - 1)
    output_corners = np.asarray(
        [(z, 0.0, x) for z, x in itertools.product(output_z_edges, output_x_edges)],
        dtype=np.float64,
    )
    for phi in config.phi_angles_deg:
        matrix, offset = backward_rotation_map(-phi, (0, 2), global_shape)
        input_corners = output_corners @ matrix.T + offset
        input_z_min = min(input_z_min, input_corners[:, 0].min())
        input_z_max = max(input_z_max, input_corners[:, 0].max())

    spline_padding = (
        settings.spline_halo
        if any(order > 1 for order in config.interpolation_orders)
        else 1
    )
    input_z_start = max(0, int(np.floor(input_z_min)) - spline_padding)
    input_z_stop = min(
        config.voxels_z,
        int(np.ceil(input_z_max)) + spline_padding + 1,
    )
    return ParticleZSlabBounds(
        input_z_start=input_z_start,
        input_z_stop=input_z_stop,
        output_z_start=output_z_start,
        output_z_stop=output_z_stop,
        global_shape=global_shape,
    )


def theta_stage_z_slab(
    source: Volume,
    theta: float,
    order: int,
    bounds: ParticleZSlabBounds,
) -> NDArray:
    """Rotate the input z slab in x-y; z and the global center are unchanged."""
    source_slab = source.data[bounds.input_z_start : bounds.input_z_stop]
    if theta == 0.0:
        return source_slab
    return ndimage.rotate(
        source_slab,
        angle=theta,
        axes=(1, 2),
        reshape=False,
        order=order,
        mode="constant",
        cval=0.0,
        output=source.data.dtype,
        prefilter=order > 1,
    )


def finish_phi_z_slab(
    theta_stage: NDArray,
    phi: float,
    order: int,
    bounds: ParticleZSlabBounds,
) -> NDArray:
    """Produce the output z slab while rotating about the full-volume center."""
    if phi == 0.0:
        local_start = bounds.output_z_start - bounds.input_z_start
        local_stop = local_start + bounds.output_depth
        if local_start < 0 or local_stop > theta_stage.shape[0]:
            raise RuntimeError("The input z slab does not contain the output z slab")
        return theta_stage[local_start:local_stop]

    matrix_3d, offset_3d = backward_rotation_map(
        -phi,
        (0, 2),
        bounds.global_shape,
    )
    zx_axes = np.asarray((0, 2))
    matrix_zx = matrix_3d[np.ix_(zx_axes, zx_axes)]
    output_origin_zx = np.asarray((bounds.output_z_start, 0.0))
    input_origin_zx = np.asarray((bounds.input_z_start, 0.0))
    local_offset_zx = (
        matrix_zx @ output_origin_zx + offset_3d[zx_axes] - input_origin_zx
    )

    rotated = np.empty(
        (
            bounds.output_depth,
            bounds.global_shape[1],
            bounds.global_shape[2],
        ),
        dtype=theta_stage.dtype,
    )
    output_shape_zx = (bounds.output_depth, bounds.global_shape[2])
    for y in range(bounds.global_shape[1]):
        rotated[:, y, :] = ndimage.affine_transform(
            theta_stage[:, y, :],
            matrix_zx,
            offset=local_offset_zx,
            output_shape=output_shape_zx,
            order=order,
            mode="constant",
            cval=0.0,
            output=theta_stage.dtype,
            prefilter=order > 1,
        )
    return rotated


def global_valid_bounds_for_z_slab(
    theta: float,
    phi: float,
    order: int,
    bounds: ParticleZSlabBounds,
) -> ValidXBounds:
    interpolation_margin = 1.0 if order == 1 else float(2 * order)
    full_bounds = valid_x_bounds_after_alignment(
        bounds.global_shape,
        theta,
        phi,
        margin=interpolation_margin,
    )
    z_slice = slice(bounds.output_z_start, bounds.output_z_stop)
    return ValidXBounds(
        x_min=full_bounds.x_min[z_slice],
        x_max=full_bounds.x_max[z_slice],
    )


def noise_state_from_z_slab(
    source_template: Volume,
    rotated_slab: NDArray,
    theta: float,
    phi: float,
    order: int,
    bounds: ParticleZSlabBounds,
) -> RotatedNoiseState:
    """Estimate the noise mean and profile over the particle-bearing z slab."""
    slab_volume = copy.copy(source_template)
    slab_volume.data = rotated_slab
    slab_volume.voxels_z = rotated_slab.shape[0]
    slab_volume.valid_bounds = global_valid_bounds_for_z_slab(
        theta,
        phi,
        order,
        bounds,
    )
    slab_volume.cylinder = None
    local_profile, _ = estimate_volume_noise(slab_volume)
    global_profile = np.full(bounds.global_shape[0], np.nan, dtype=np.float64)
    profile_start = bounds.output_z_start
    profile_stop = profile_start + len(local_profile)
    global_profile[profile_start:profile_stop] = local_profile
    # Use every valid x-y sample in the output slab. In particular, do not
    # include the larger input slab: its extra z halo exists only to support
    # high-order interpolation at the output-slab boundaries.
    mean = get_mean_from_volume(rotated_slab)
    return RotatedNoiseState(mean=mean, profile=global_profile)


def rotated_particle_candidates(
    particle: Particle,
    theta: float,
    phi: float,
    order: int,
    global_shape: tuple[int, int, int],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return the global rotated center and search candidates for one particle."""
    rotated_position = forward_align_points_zyx(
        particle.position_zyx[None, :],
        theta,
        phi,
        global_shape,
    )[0]
    transformed_support = forward_align_points_zyx(
        particle.support_zyx,
        theta,
        phi,
        global_shape,
    )
    interpolation_halo = 1 if order == 1 else 2 * order
    lower_zyx = (
        np.floor(transformed_support.min(axis=0)).astype(np.int64) - interpolation_halo
    )
    upper_zyx = (
        np.ceil(transformed_support.max(axis=0)).astype(np.int64) + interpolation_halo
    )
    shape_zyx = np.asarray(global_shape, dtype=np.int64)
    lower_zyx = np.maximum(lower_zyx, 0)
    upper_zyx = np.minimum(upper_zyx, shape_zyx - 1)
    axes = [np.arange(lower_zyx[axis], upper_zyx[axis] + 1) for axis in range(3)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    return rotated_position, np.column_stack((zz.ravel(), yy.ravel(), xx.ravel()))


def measure_clean_state_from_z_slab(
    particles: list[Particle],
    rotated_slab: NDArray,
    theta: float,
    phi: float,
    order: int,
    bit_depth: type,
    noise_mean: float,
    bounds: ParticleZSlabBounds,
) -> list[ParticleStateMeasurement]:
    """Select and measure clean top-k locations using global coordinates."""
    flat_noise_by_z = np.full(bounds.global_shape[0], np.nan, dtype=np.float64)
    measurements = []
    for particle in particles:
        rotated_position, candidate_zyx = rotated_particle_candidates(
            particle,
            theta,
            phi,
            order,
            bounds.global_shape,
        )
        local_z = candidate_zyx[:, 0] - bounds.output_z_start
        if np.any(local_z < 0) or np.any(local_z >= bounds.output_depth):
            raise RuntimeError(
                "Calculated output z slab does not contain a particle search region"
            )
        candidate_values = rotated_slab[
            local_z,
            candidate_zyx[:, 1],
            candidate_zyx[:, 2],
        ]
        retained_k = min(particle.measurement_top_k, len(candidate_values))
        selected = np.argpartition(
            candidate_values,
            len(candidate_values) - retained_k,
        )[-retained_k:]
        selected = selected[np.argsort(candidate_values[selected])[::-1]]
        top_k_zyx = candidate_zyx[selected]
        top_k_values = candidate_values[selected]
        measurements.append(
            ParticleStateMeasurement(
                particle_id=particle.particle_id,
                state=(bit_depth, theta, phi),
                position_zyx=rotated_position,
                top_k=retained_k,
                top_k_zyx=top_k_zyx,
                particle_only_top_k_values=top_k_values,
                combined_top_k_values=top_k_values,
                noise_mean=noise_mean,
                noise_std_by_z=flat_noise_by_z,
            )
        )
    return measurements


def sample_observed_states_from_z_slab(
    references: list[ParticleStateMeasurement],
    observed_slab: NDArray,
    noise_state: RotatedNoiseState,
    bounds: ParticleZSlabBounds,
) -> list[ParticleStateMeasurement]:
    """Sample a noisy output slab at clean, globally indexed top-k locations."""
    noise_std_by_z = ParticleStateMeasurement._to_plane_noise_std(
        noise_state.profile,
        bounds.global_shape[0],
    )
    measurements = []
    for reference in references:
        indices = reference.top_k_zyx
        local_z = indices[:, 0] - bounds.output_z_start
        if np.any(local_z < 0) or np.any(local_z >= bounds.output_depth):
            raise RuntimeError("A clean top-k location lies outside the output z slab")
        observed_values = observed_slab[
            local_z,
            indices[:, 1],
            indices[:, 2],
        ]
        measurements.append(
            ParticleStateMeasurement(
                particle_id=reference.particle_id,
                state=reference.state,
                position_zyx=reference.position_zyx,
                top_k=reference.top_k,
                top_k_zyx=indices,
                particle_only_top_k_values=reference.particle_only_top_k_values,
                combined_top_k_values=observed_values,
                noise_mean=noise_state.mean,
                noise_std_by_z=noise_std_by_z,
            )
        )
    return measurements


def map_phi_tasks(
    phi_angles: tuple[float, ...],
    workers: int,
    task: Callable[[float], object],
) -> list[object]:
    if workers == 1:
        return [task(phi) for phi in phi_angles]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(task, phi_angles))


def run_z_slab_theta_groups(
    theta_angles: tuple[float, ...],
    phi_angles: tuple[float, ...],
    max_workers: int,
    worker: Callable[[float, int], object],
) -> tuple[list[object], int]:
    """Parallelize theta groups, or phi when a pilot has only one theta."""
    if len(theta_angles) == 1:
        phi_workers = min(max_workers, len(phi_angles))
        return [worker(theta_angles[0], phi_workers)], phi_workers

    theta_workers = min(max_workers, len(theta_angles))
    if theta_workers == 1:
        return [worker(theta, 1) for theta in theta_angles], 1
    with ThreadPoolExecutor(max_workers=theta_workers) as executor:
        return list(
            executor.map(lambda theta: worker(theta, 1), theta_angles)
        ), theta_workers


def measure_clean_theta_group_z_slab(
    clean_base: ParticleVolumeUint,
    clean_before: list[ParticleStateMeasurement],
    clean_background: float,
    interpolation_order: int,
    phi_angles: tuple[float, ...],
    bounds: ParticleZSlabBounds,
    theta: float,
    phi_workers: int,
) -> tuple[float, dict[float, list[ParticleStateMeasurement]]]:
    theta_stage = theta_stage_z_slab(
        clean_base.volume,
        theta,
        interpolation_order,
        bounds,
    )
    bit_depth = clean_base.get_array().dtype.type

    def measure_phi(phi: float) -> list[ParticleStateMeasurement]:
        if theta == 0.0 and phi == 0.0:
            return clean_before
        rotated_slab = finish_phi_z_slab(
            theta_stage,
            phi,
            interpolation_order,
            bounds,
        )
        return measure_clean_state_from_z_slab(
            clean_base.particles,
            rotated_slab,
            theta,
            phi,
            interpolation_order,
            bit_depth,
            clean_background,
            bounds,
        )

    results = map_phi_tasks(phi_angles, phi_workers, measure_phi)
    return theta, dict(zip(phi_angles, results, strict=True))


def measure_noise_theta_group_z_slab(
    noise_base: Volume,
    base_noise_state: RotatedNoiseState,
    interpolation_order: int,
    phi_angles: tuple[float, ...],
    bounds: ParticleZSlabBounds,
    theta: float,
    phi_workers: int,
) -> tuple[float, dict[float, RotatedNoiseState]]:
    theta_stage = theta_stage_z_slab(
        noise_base,
        theta,
        interpolation_order,
        bounds,
    )

    def measure_phi(phi: float) -> RotatedNoiseState:
        if theta == 0.0 and phi == 0.0:
            return base_noise_state
        rotated_slab = finish_phi_z_slab(
            theta_stage,
            phi,
            interpolation_order,
            bounds,
        )
        return noise_state_from_z_slab(
            noise_base,
            rotated_slab,
            theta,
            phi,
            interpolation_order,
            bounds,
        )

    results = map_phi_tasks(phi_angles, phi_workers, measure_phi)
    return theta, dict(zip(phi_angles, results, strict=True))


def measure_noisy_theta_group_z_slab(
    noisy_base: Volume,
    clean_states: dict[tuple[int, float, float], list[ParticleStateMeasurement]],
    noise_states: dict[tuple[int, float, float], RotatedNoiseState],
    interpolation_order: int,
    phi_angles: tuple[float, ...],
    bounds: ParticleZSlabBounds,
    theta: float,
    phi_workers: int,
) -> tuple[float, dict[float, list[ParticleStateMeasurement]]]:
    theta_stage = theta_stage_z_slab(
        noisy_base,
        theta,
        interpolation_order,
        bounds,
    )

    def measure_phi(phi: float) -> list[ParticleStateMeasurement]:
        rotated_slab = finish_phi_z_slab(
            theta_stage,
            phi,
            interpolation_order,
            bounds,
        )
        state_key = (interpolation_order, theta, phi)
        return sample_observed_states_from_z_slab(
            clean_states[state_key],
            rotated_slab,
            noise_states[state_key],
            bounds,
        )

    results = map_phi_tasks(phi_angles, phi_workers, measure_phi)
    return theta, dict(zip(phi_angles, results, strict=True))


def run_particle_rotation_sweep_z_slab(
    config: ParticleRotationSweepConfig,
    output_csv: Path = OPTIMIZED_PARTICLE_SWEEP_OUTPUT_CSV,
    settings: ZSlabOptimizationSettings = ZSlabOptimizationSettings(),
) -> ParticleSweepRun:
    """Run the existing experiment using full-center, particle-bearing z slabs."""
    output_csv = Path(output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.unlink(missing_ok=True)
    bounds = calculate_particle_z_slab_bounds(config, settings)
    print(f"Writing optimized particle metrics to {output_csv}")
    print(
        "z slabs: "
        f"input=[{bounds.input_z_start}, {bounds.input_z_stop}) "
        f"({bounds.input_depth}/{config.voxels_z}), "
        f"output=[{bounds.output_z_start}, {bounds.output_z_stop}) "
        f"({bounds.output_depth}/{config.voxels_z}); "
        f"rotation workers={settings.rotation_workers}"
    )

    total_conditions = (
        len(config.noise_seeds)
        * len(config.bit_depths)
        * len(config.particle_sigmas)
        * len(config.amplitude_snrs)
        * len(config.interpolation_orders)
        * len(config.theta_angles_deg)
        * len(config.phi_angles_deg)
    )
    volume_center_yx = np.array([config.voxels_xy // 2] * 2, dtype=np.float64)
    realized_seeds = tuple(seed for seed in config.noise_seeds if seed != NO_NOISE_SEED)
    rows_written = 0
    timing_records = []
    sweep_started = perf_counter()

    with tqdm(
        total=total_conditions,
        desc="Particle z-slab sweep",
        unit="condition",
        dynamic_ncols=True,
    ) as progress_bar:
        for bit_depth in config.bit_depths:
            clean_cache = {}

            for particle_sigma in config.particle_sigmas:
                for amplitude_snr in config.amplitude_snrs:
                    if not any(
                        condition_is_pending(
                            seed,
                            bit_depth,
                            particle_sigma,
                            amplitude_snr,
                            interpolation_order,
                            theta,
                            phi,
                        )
                        for seed in config.noise_seeds
                        for interpolation_order in config.interpolation_orders
                        for theta in config.theta_angles_deg
                        for phi in config.phi_angles_deg
                    ):
                        continue
                    setup_started = perf_counter()
                    particle_fp32 = create_uniform_particle_volume(
                        voxels_z=config.voxels_z,
                        voxels_xy=config.voxels_xy,
                        sigma=particle_sigma,
                        amplitude=amplitude_snr,
                        z_start_rel=config.z_start_rel,
                        z_end_rel=config.z_end_rel,
                        delta_z_rel=config.z_delta_rel,
                        delta_angle_deg=config.radial_arm_spacing_deg,
                        stop_angle_deg=config.quadrant_stop_deg,
                        measurement_top_k=config.measurement_top_k,
                    )
                    clean_base = ParticleVolumeUint(
                        particle_fp32,
                        type=bit_depth,
                        target_std_scale=config.target_std_scale,
                    )
                    clean_background = float(np.rint(np.iinfo(bit_depth).max / 2))
                    flat_noise_profile = np.full(config.voxels_z, np.nan)
                    clean_before = measure_particle_states(
                        clean_base,
                        clean_base.get_array(),
                        clean_background,
                        flat_noise_profile,
                        (bit_depth, 0.0, 0.0),
                    )
                    timing_records.append(
                        timing_record(
                            phase="clean_setup_z_slab",
                            bit_depth=bit_depth,
                            noise_seed=NO_NOISE_SEED,
                            interpolation_order=None,
                            workers=1,
                            conditions_written=0,
                            compute_seconds=perf_counter() - setup_started,
                            particle_sigma=particle_sigma,
                            amplitude_snr=amplitude_snr,
                        )
                    )

                    clean_states = {}
                    for interpolation_order in config.interpolation_orders:
                        compute_started = perf_counter()
                        worker = partial(
                            measure_clean_theta_group_z_slab,
                            clean_base,
                            clean_before,
                            clean_background,
                            interpolation_order,
                            config.phi_angles_deg,
                            bounds,
                        )
                        theta_results, workers = run_z_slab_theta_groups(
                            config.theta_angles_deg,
                            config.phi_angles_deg,
                            settings.rotation_workers,
                            worker,
                        )
                        compute_seconds = perf_counter() - compute_started
                        write_seconds = 0.0
                        conditions_written = 0
                        for theta, measurements_by_phi in theta_results:
                            for phi in config.phi_angles_deg:
                                state_key = (interpolation_order, theta, phi)
                                clean_after = measurements_by_phi[phi]
                                clean_states[state_key] = clean_after
                                row_count, condition_write_seconds = (
                                    write_condition_rows(
                                        output_csv,
                                        NO_NOISE_SEED,
                                        bit_depth,
                                        particle_sigma,
                                        amplitude_snr,
                                        interpolation_order,
                                        theta,
                                        phi,
                                        volume_center_yx,
                                        clean_before,
                                        clean_after,
                                        clean_before,
                                        clean_after,
                                    )
                                )
                                rows_written += row_count
                                write_seconds += condition_write_seconds
                                conditions_written += 1
                                update_sweep_progress(
                                    progress_bar,
                                    "clean z-slab",
                                    NO_NOISE_SEED,
                                    bit_depth,
                                    particle_sigma,
                                    amplitude_snr,
                                    interpolation_order,
                                    theta,
                                    phi,
                                    advance=True,
                                )
                        timing_records.append(
                            timing_record(
                                phase="clean_rotation_z_slab",
                                bit_depth=bit_depth,
                                noise_seed=NO_NOISE_SEED,
                                interpolation_order=interpolation_order,
                                workers=workers,
                                conditions_written=conditions_written,
                                compute_seconds=compute_seconds,
                                write_seconds=write_seconds,
                                particle_sigma=particle_sigma,
                                amplitude_snr=amplitude_snr,
                            )
                        )

                    clean_cache[(particle_sigma, amplitude_snr)] = (
                        clean_before,
                        clean_states,
                    )
                    del clean_base, particle_fp32

            for seed in realized_seeds:
                if not any(
                    condition_is_pending(
                        seed,
                        bit_depth,
                        particle_sigma,
                        amplitude_snr,
                        interpolation_order,
                        theta,
                        phi,
                    )
                    for particle_sigma in config.particle_sigmas
                    for amplitude_snr in config.amplitude_snrs
                    for interpolation_order in config.interpolation_orders
                    for theta in config.theta_angles_deg
                    for phi in config.phi_angles_deg
                ):
                    continue
                noise_setup_started = perf_counter()
                np.random.seed(seed)
                standard_noise = StandardNoiseVolume(
                    voxels_z=config.voxels_z,
                    voxels_xy=config.voxels_xy,
                    rho1_z=config.rho1_z,
                    rho1_xy=config.rho1_xy,
                )
                target_std = np.iinfo(bit_depth).max * config.target_std_scale
                noise_base = standard_noise.get_quantized_noise(
                    target_std=target_std,
                    uint_type=bit_depth,
                )
                base_slab = noise_base.data[
                    bounds.output_z_start : bounds.output_z_stop
                ]
                base_noise_state = noise_state_from_z_slab(
                    noise_base,
                    base_slab,
                    theta=0.0,
                    phi=0.0,
                    order=max(config.interpolation_orders),
                    bounds=bounds,
                )
                timing_records.append(
                    timing_record(
                        phase="noise_setup_z_slab",
                        bit_depth=bit_depth,
                        noise_seed=seed,
                        interpolation_order=None,
                        workers=1,
                        conditions_written=0,
                        compute_seconds=perf_counter() - noise_setup_started,
                    )
                )

                noise_states = {}
                for interpolation_order in config.interpolation_orders:
                    compute_started = perf_counter()
                    worker = partial(
                        measure_noise_theta_group_z_slab,
                        noise_base,
                        base_noise_state,
                        interpolation_order,
                        config.phi_angles_deg,
                        bounds,
                    )
                    theta_results, workers = run_z_slab_theta_groups(
                        config.theta_angles_deg,
                        config.phi_angles_deg,
                        settings.rotation_workers,
                        worker,
                    )
                    for theta, states_by_phi in theta_results:
                        for phi, noise_state in states_by_phi.items():
                            noise_states[(interpolation_order, theta, phi)] = (
                                noise_state
                            )
                    timing_records.append(
                        timing_record(
                            phase="noise_rotation_z_slab",
                            bit_depth=bit_depth,
                            noise_seed=seed,
                            interpolation_order=interpolation_order,
                            workers=workers,
                            conditions_written=0,
                            compute_seconds=perf_counter() - compute_started,
                        )
                    )
                del noise_base

                for particle_sigma in config.particle_sigmas:
                    for amplitude_snr in config.amplitude_snrs:
                        if not any(
                            condition_is_pending(
                                seed,
                                bit_depth,
                                particle_sigma,
                                amplitude_snr,
                                interpolation_order,
                                theta,
                                phi,
                            )
                            for interpolation_order in config.interpolation_orders
                            for theta in config.theta_angles_deg
                            for phi in config.phi_angles_deg
                        ):
                            continue
                        noisy_setup_started = perf_counter()
                        particle_fp32 = create_uniform_particle_volume(
                            voxels_z=config.voxels_z,
                            voxels_xy=config.voxels_xy,
                            sigma=particle_sigma,
                            amplitude=amplitude_snr,
                            z_start_rel=config.z_start_rel,
                            z_end_rel=config.z_end_rel,
                            delta_z_rel=config.z_delta_rel,
                            delta_angle_deg=config.radial_arm_spacing_deg,
                            stop_angle_deg=config.quadrant_stop_deg,
                            measurement_top_k=config.measurement_top_k,
                        )
                        particle_fp32.get_array()[:] += standard_noise.get_array()
                        noisy_base = ParticleVolumeUint(
                            particle_fp32,
                            type=bit_depth,
                            target_std_scale=config.target_std_scale,
                        )
                        clean_before, clean_states = clean_cache[
                            (particle_sigma, amplitude_snr)
                        ]
                        noisy_before = sample_observed_states(
                            clean_before,
                            noisy_base.get_array(),
                            base_noise_state,
                        )
                        timing_records.append(
                            timing_record(
                                phase="noisy_setup_z_slab",
                                bit_depth=bit_depth,
                                noise_seed=seed,
                                interpolation_order=None,
                                workers=1,
                                conditions_written=0,
                                compute_seconds=perf_counter() - noisy_setup_started,
                                particle_sigma=particle_sigma,
                                amplitude_snr=amplitude_snr,
                            )
                        )

                        for interpolation_order in config.interpolation_orders:
                            compute_started = perf_counter()
                            worker = partial(
                                measure_noisy_theta_group_z_slab,
                                noisy_base.volume,
                                clean_states,
                                noise_states,
                                interpolation_order,
                                config.phi_angles_deg,
                                bounds,
                            )
                            theta_results, workers = run_z_slab_theta_groups(
                                config.theta_angles_deg,
                                config.phi_angles_deg,
                                settings.rotation_workers,
                                worker,
                            )
                            compute_seconds = perf_counter() - compute_started
                            write_seconds = 0.0
                            conditions_written = 0
                            for theta, measurements_by_phi in theta_results:
                                for phi in config.phi_angles_deg:
                                    state_key = (interpolation_order, theta, phi)
                                    row_count, condition_write_seconds = (
                                        write_condition_rows(
                                            output_csv,
                                            seed,
                                            bit_depth,
                                            particle_sigma,
                                            amplitude_snr,
                                            interpolation_order,
                                            theta,
                                            phi,
                                            volume_center_yx,
                                            clean_before,
                                            clean_states[state_key],
                                            noisy_before,
                                            measurements_by_phi[phi],
                                        )
                                    )
                                    rows_written += row_count
                                    write_seconds += condition_write_seconds
                                    conditions_written += 1
                                    update_sweep_progress(
                                        progress_bar,
                                        "noisy z-slab",
                                        seed,
                                        bit_depth,
                                        particle_sigma,
                                        amplitude_snr,
                                        interpolation_order,
                                        theta,
                                        phi,
                                        advance=True,
                                    )
                            timing_records.append(
                                timing_record(
                                    phase="noisy_rotation_z_slab",
                                    bit_depth=bit_depth,
                                    noise_seed=seed,
                                    interpolation_order=interpolation_order,
                                    workers=workers,
                                    conditions_written=conditions_written,
                                    compute_seconds=compute_seconds,
                                    write_seconds=write_seconds,
                                    particle_sigma=particle_sigma,
                                    amplitude_snr=amplitude_snr,
                                )
                            )
                        del noisy_base, particle_fp32
                del standard_noise, noise_states
            del clean_cache

    elapsed_seconds = perf_counter() - sweep_started
    timings = pd.DataFrame.from_records(timing_records)
    timing_summary = timings.groupby(
        ["phase", "interpolation_order"],
        dropna=False,
    )[["compute_seconds", "write_seconds", "elapsed_seconds"]].sum()
    print("\nOptimized sweep timing summary (seconds):")
    print(timing_summary.to_string())
    return ParticleSweepRun(
        output_csv=output_csv,
        rows_written=rows_written,
        elapsed_seconds=elapsed_seconds,
        timings=timings,
    )


SLAB_COMPARISON_KEYS = [
    "noise_seed",
    "bit_depth",
    "particle_sigma",
    "amplitude_snr",
    "interpolation_order",
    "theta_deg",
    "phi_deg",
    "particle_id",
    "top_k",
]

SLAB_COMPARISON_COLUMNS = [
    "noise_mean_before",
    "noise_mean_after",
    "rotated_z",
    "rotated_y",
    "rotated_x",
    "peak_contrast_before",
    "peak_contrast_after",
    "peak_contrast_retention",
    "peak_contrast_drop",
    "integrated_contrast_before",
    "integrated_contrast_after",
    "integrated_contrast_retention",
    "integrated_contrast_drop",
    "peak_cnr_before",
    "peak_cnr_after",
    "peak_cnr_retention",
    "integrated_cnr_before",
    "integrated_cnr_after",
    "integrated_cnr_retention",
]


def compare_z_slab_results(
    optimized_csv: str | Path,
    reference_csv: str | Path = PARTICLE_SWEEP_OUTPUT_CSV,
    retention_tolerance: float = 5e-3,
) -> pd.DataFrame:
    """Compare optimized rows with matching persisted full-volume rows."""
    optimized = pd.read_csv(optimized_csv)
    reference = pd.read_csv(reference_csv)
    merged = optimized.merge(
        reference,
        on=SLAB_COMPARISON_KEYS,
        how="inner",
        suffixes=("_optimized", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(optimized):
        raise AssertionError(
            f"Matched {len(merged)} of {len(optimized)} optimized particle rows"
        )

    comparison_values = {
        column: (
            merged[f"{column}_optimized"].to_numpy(dtype=np.float64),
            merged[f"{column}_reference"].to_numpy(dtype=np.float64),
        )
        for column in SLAB_COMPARISON_COLUMNS
    }
    # These reconstruct the actual sampled uint values. They distinguish a
    # rotation mismatch from the small expected change caused by estimating
    # the background mean over only the particle-bearing z interval.
    for state in ("before", "after"):
        comparison_values[f"observed_peak_{state}"] = tuple(
            merged[f"peak_contrast_{state}_{suffix}"].to_numpy(dtype=np.float64)
            + merged[f"noise_mean_{state}_{suffix}"].to_numpy(dtype=np.float64)
            for suffix in ("optimized", "reference")
        )
        comparison_values[f"observed_top_k_sum_{state}"] = tuple(
            merged[f"integrated_contrast_{state}_{suffix}"].to_numpy(dtype=np.float64)
            + merged["top_k"].to_numpy(dtype=np.float64)
            * merged[f"noise_mean_{state}_{suffix}"].to_numpy(dtype=np.float64)
            for suffix in ("optimized", "reference")
        )

    records = []
    for column, (optimized_values, reference_values) in comparison_values.items():
        finite = np.isfinite(optimized_values) & np.isfinite(reference_values)
        both_nan = np.isnan(optimized_values) & np.isnan(reference_values)
        if not np.all(finite | both_nan):
            max_abs_difference = np.inf
            mean_abs_difference = np.inf
        elif finite.any():
            difference = np.abs(optimized_values[finite] - reference_values[finite])
            max_abs_difference = float(difference.max())
            mean_abs_difference = float(difference.mean())
        else:
            max_abs_difference = 0.0
            mean_abs_difference = 0.0
        tolerance = retention_tolerance if column.endswith("_retention") else np.nan
        within_tolerance = (
            max_abs_difference <= tolerance if np.isfinite(tolerance) else None
        )
        records.append(
            {
                "metric": column,
                "max_abs_difference": max_abs_difference,
                "mean_abs_difference": mean_abs_difference,
                "retention_tolerance": tolerance,
                "within_retention_tolerance": within_tolerance,
            }
        )
    comparison = pd.DataFrame.from_records(records)
    retention_rows = comparison["metric"].str.endswith("_retention")
    if not comparison.loc[retention_rows, "within_retention_tolerance"].all():
        failures = comparison.loc[
            retention_rows & ~comparison["within_retention_tolerance"].astype(bool)
        ]
        raise AssertionError(
            "Optimized retention metrics exceeded the comparison tolerance:\n"
            + failures.to_string(index=False)
        )
    return comparison


def z_slab_timing_comparison(
    run: ParticleSweepRun,
    config: ParticleRotationSweepConfig,
    reference_seconds_per_condition: float = REFERENCE_SECONDS_PER_CONDITION,
) -> pd.DataFrame:
    """Report mean optimized condition time against the saved 392-second reference."""
    condition_count = (
        len(config.noise_seeds)
        * len(config.bit_depths)
        * len(config.particle_sigmas)
        * len(config.amplitude_snrs)
        * len(config.interpolation_orders)
        * len(config.theta_angles_deg)
        * len(config.phi_angles_deg)
    )
    optimized_seconds = run.elapsed_seconds / condition_count
    return pd.DataFrame.from_records(
        [
            {
                "reference_seconds_per_condition": reference_seconds_per_condition,
                "optimized_seconds_per_condition": optimized_seconds,
                "seconds_saved_per_condition": (
                    reference_seconds_per_condition - optimized_seconds
                ),
                "speedup": reference_seconds_per_condition / optimized_seconds,
                "percent_reduction": 100.0
                * (1.0 - optimized_seconds / reference_seconds_per_condition),
            }
        ]
    )


def validate_particle_z_slab_optimization() -> pd.DataFrame:
    """Compare full and slab paths on a small, quickly evaluated sweep."""
    from scripts.particle_sim_sweep import particle_sweep_config

    validation_config = replace(
        particle_sweep_config,
        voxels_z=64,
        voxels_xy=64,
        particle_sigmas=(1.0,),
        amplitude_snrs=(5.0,),
        interpolation_orders=(1, 5),
        bit_depths=(np.uint8,),
        theta_angles_deg=(15.0,),
        phi_angles_deg=(0.1, 5.0),
        noise_seeds=(NO_NOISE_SEED, 10),
    )
    validation_directory = Path("Data/_tmp").resolve()
    reference_csv = validation_directory / "particle_slab_reference.csv"
    optimized_csv = validation_directory / "particle_slab_optimized.csv"
    reference_run = run_particle_rotation_sweep(
        validation_config,
        output_csv=reference_csv,
    )
    optimized_run = run_particle_rotation_sweep_z_slab(
        validation_config,
        output_csv=optimized_csv,
    )
    comparison = compare_z_slab_results(
        optimized_run.output_csv,
        reference_run.output_csv,
        retention_tolerance=3e-3,
    )
    reconstructed_samples = comparison["metric"].isin(
        [
            "observed_peak_before",
            "observed_peak_after",
            "observed_top_k_sum_before",
            "observed_top_k_sum_after",
        ]
    )
    if not (comparison.loc[reconstructed_samples, "max_abs_difference"] < 1e-10).all():
        raise AssertionError("The slab path changed sampled uint particle values")
    print(
        "Small-volume validation: "
        f"full={reference_run.elapsed_seconds:.3f}s, "
        f"z-slab={optimized_run.elapsed_seconds:.3f}s"
    )
    reference_csv.unlink(missing_ok=True)
    optimized_csv.unlink(missing_ok=True)
    return comparison


# Particle-neighborhood optimization

PARTICLE_NEIGHBORHOOD_SWEEP_OUTPUT_CSV = PARTICLE_SWEEP_OUTPUT_CSV.with_name(
    "particle_experiment_metrics_neighborhood_optimized.csv"
)
PARTICLE_NEIGHBORHOOD_REFERENCE_SECONDS = 130.4


@dataclass(frozen=True)
class ParticleNeighborhoodOptimizationSettings:
    """Controls for the noise-slab and particle-neighborhood work."""

    noise_rotation_workers: int = 2
    particle_workers: int = 8
    particle_spline_halo: int = 32
    noise_spline_halo: int = 32
    profile_guard: int = 2

    def __post_init__(self) -> None:
        if self.noise_rotation_workers < 1:
            raise ValueError("noise_rotation_workers must be at least one")
        if self.particle_workers < 1:
            raise ValueError("particle_workers must be at least one")
        if self.particle_spline_halo < 0:
            raise ValueError("particle_spline_halo must be nonnegative")
        if self.noise_spline_halo < 0:
            raise ValueError("noise_spline_halo must be nonnegative")
        if self.profile_guard < 1:
            raise ValueError("profile_guard must be at least one")

    def noise_slab_settings(self) -> ZSlabOptimizationSettings:
        return ZSlabOptimizationSettings(
            rotation_workers=self.noise_rotation_workers,
            spline_halo=self.noise_spline_halo,
            profile_guard=self.profile_guard,
        )


@dataclass(frozen=True)
class VoxelBox:
    """Half-open integer box in global z-y-x coordinates."""

    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.start_zyx) != 3 or len(self.stop_zyx) != 3:
            raise ValueError("VoxelBox coordinates must have length three")
        if any(start >= stop for start, stop in zip(self.start_zyx, self.stop_zyx)):
            raise ValueError("VoxelBox must have positive extent on every axis")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(
            stop - start
            for start, stop in zip(self.start_zyx, self.stop_zyx, strict=True)
        )

    @property
    def voxel_count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(
            slice(start, stop)
            for start, stop in zip(self.start_zyx, self.stop_zyx, strict=True)
        )


def box_from_points(points_zyx: NDArray[np.int64]) -> VoxelBox:
    points = np.asarray(points_zyx, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_zyx must have nonempty shape (N, 3)")
    start = points.min(axis=0)
    stop = points.max(axis=0) + 1
    return VoxelBox(tuple(start.tolist()), tuple(stop.tolist()))


def box_corners(box: VoxelBox) -> NDArray[np.float64]:
    endpoints = [
        (float(start), float(stop - 1))
        for start, stop in zip(box.start_zyx, box.stop_zyx, strict=True)
    ]
    return np.asarray(list(itertools.product(*endpoints)), dtype=np.float64)


def clip_box(
    start_zyx: NDArray[np.int64],
    stop_zyx: NDArray[np.int64],
    global_shape: tuple[int, int, int],
) -> VoxelBox:
    shape = np.asarray(global_shape, dtype=np.int64)
    start = np.clip(start_zyx, 0, shape - 1)
    stop = np.clip(stop_zyx, start + 1, shape)
    return VoxelBox(tuple(start.tolist()), tuple(stop.tolist()))


def union_boxes(
    boxes: list[VoxelBox],
    global_shape: tuple[int, int, int],
) -> VoxelBox:
    if not boxes:
        raise ValueError("at least one box is required")
    starts = np.asarray([box.start_zyx for box in boxes], dtype=np.int64)
    stops = np.asarray([box.stop_zyx for box in boxes], dtype=np.int64)
    return clip_box(starts.min(axis=0), stops.max(axis=0), global_shape)


def backproject_box(
    output_box: VoxelBox,
    matrix: NDArray[np.float64],
    offset: NDArray[np.float64],
    halo_zyx: tuple[int, int, int],
    global_shape: tuple[int, int, int],
) -> VoxelBox:
    mapped = box_corners(output_box) @ matrix.T + offset
    halo = np.asarray(halo_zyx, dtype=np.int64)
    start = np.floor(mapped.min(axis=0)).astype(np.int64) - halo
    stop = np.ceil(mapped.max(axis=0)).astype(np.int64) + halo + 1
    return clip_box(start, stop, global_shape)


def theta_neighborhood(
    source_data: NDArray,
    source_box: VoxelBox,
    output_box: VoxelBox,
    theta: float,
    order: int,
    global_shape: tuple[int, int, int],
) -> NDArray:
    """Evaluate theta locally, plane by plane, using the global center."""
    source_patch = source_data[source_box.slices]
    if theta == 0.0:
        offsets = tuple(
            output_box.start_zyx[axis] - source_box.start_zyx[axis] for axis in range(3)
        )
        slices = tuple(
            slice(offset, offset + output_box.shape[axis])
            for axis, offset in enumerate(offsets)
        )
        return source_patch[slices]

    matrix_3d, offset_3d = backward_rotation_map(
        theta,
        (1, 2),
        global_shape,
    )
    axes = np.asarray((1, 2))
    matrix_yx = matrix_3d[np.ix_(axes, axes)]
    output_origin_yx = np.asarray(output_box.start_zyx[1:], dtype=np.float64)
    input_origin_yx = np.asarray(source_box.start_zyx[1:], dtype=np.float64)
    local_offset_yx = matrix_yx @ output_origin_yx + offset_3d[axes] - input_origin_yx

    rotated = np.empty(output_box.shape, dtype=source_data.dtype)
    for local_output_z, global_z in enumerate(
        range(output_box.start_zyx[0], output_box.stop_zyx[0])
    ):
        local_input_z = global_z - source_box.start_zyx[0]
        rotated[local_output_z] = ndimage.affine_transform(
            source_patch[local_input_z],
            matrix_yx,
            offset=local_offset_yx,
            output_shape=output_box.shape[1:],
            order=order,
            mode="constant",
            cval=0.0,
            output=source_data.dtype,
            prefilter=order > 1,
        )
    return rotated


def phi_neighborhood(
    theta_patch: NDArray,
    theta_box: VoxelBox,
    output_box: VoxelBox,
    phi: float,
    order: int,
    global_shape: tuple[int, int, int],
) -> NDArray:
    """Evaluate phi locally, plane by plane, using the global center."""
    if phi == 0.0:
        offsets = tuple(
            output_box.start_zyx[axis] - theta_box.start_zyx[axis] for axis in range(3)
        )
        slices = tuple(
            slice(offset, offset + output_box.shape[axis])
            for axis, offset in enumerate(offsets)
        )
        return theta_patch[slices]

    matrix_3d, offset_3d = backward_rotation_map(
        -phi,
        (0, 2),
        global_shape,
    )
    axes = np.asarray((0, 2))
    matrix_zx = matrix_3d[np.ix_(axes, axes)]
    output_origin_zx = np.asarray(
        (output_box.start_zyx[0], output_box.start_zyx[2]),
        dtype=np.float64,
    )
    input_origin_zx = np.asarray(
        (theta_box.start_zyx[0], theta_box.start_zyx[2]),
        dtype=np.float64,
    )
    local_offset_zx = matrix_zx @ output_origin_zx + offset_3d[axes] - input_origin_zx

    rotated = np.empty(output_box.shape, dtype=theta_patch.dtype)
    output_shape_zx = (output_box.shape[0], output_box.shape[2])
    for local_output_y, global_y in enumerate(
        range(output_box.start_zyx[1], output_box.stop_zyx[1])
    ):
        local_input_y = global_y - theta_box.start_zyx[1]
        rotated[:, local_output_y, :] = ndimage.affine_transform(
            theta_patch[:, local_input_y, :],
            matrix_zx,
            offset=local_offset_zx,
            output_shape=output_shape_zx,
            order=order,
            mode="constant",
            cval=0.0,
            output=theta_patch.dtype,
            prefilter=order > 1,
        )
    return rotated


def neighborhood_boxes_for_points(
    output_points_by_phi: dict[float, NDArray[np.int64]],
    theta: float,
    order: int,
    halo: int,
    global_shape: tuple[int, int, int],
) -> tuple[VoxelBox, VoxelBox, dict[float, VoxelBox]]:
    """Return source, theta-stage, and final boxes for one particle."""
    final_boxes = {
        phi: box_from_points(points) for phi, points in output_points_by_phi.items()
    }
    effective_halo = max(1 if order == 1 else 2 * order, halo)
    intermediate_boxes = []
    for phi, final_box in final_boxes.items():
        if phi == 0.0:
            intermediate_boxes.append(final_box)
            continue
        matrix, offset = backward_rotation_map(-phi, (0, 2), global_shape)
        intermediate_boxes.append(
            backproject_box(
                final_box,
                matrix,
                offset,
                (effective_halo, 0, effective_halo),
                global_shape,
            )
        )
    theta_box = union_boxes(intermediate_boxes, global_shape)

    if theta == 0.0:
        source_box = theta_box
    else:
        matrix, offset = backward_rotation_map(theta, (1, 2), global_shape)
        source_box = backproject_box(
            theta_box,
            matrix,
            offset,
            (0, effective_halo, effective_halo),
            global_shape,
        )
    return source_box, theta_box, final_boxes


def sample_particle_neighborhoods(
    source_data: NDArray,
    output_points_by_phi: dict[float, NDArray[np.int64]],
    theta: float,
    order: int,
    halo: int,
    global_shape: tuple[int, int, int],
) -> dict[float, NDArray]:
    """Sample one source particle through the original two rotation stages."""
    source_box, theta_box, final_boxes = neighborhood_boxes_for_points(
        output_points_by_phi,
        theta,
        order,
        halo,
        global_shape,
    )
    theta_patch = theta_neighborhood(
        source_data,
        source_box,
        theta_box,
        theta,
        order,
        global_shape,
    )

    values_by_phi = {}
    for phi, points in output_points_by_phi.items():
        final_box = final_boxes[phi]
        final_patch = phi_neighborhood(
            theta_patch,
            theta_box,
            final_box,
            phi,
            order,
            global_shape,
        )
        local_points = points - np.asarray(final_box.start_zyx, dtype=np.int64)
        values_by_phi[phi] = final_patch[
            local_points[:, 0],
            local_points[:, 1],
            local_points[:, 2],
        ]
    return values_by_phi


def map_particle_tasks(
    particles: list[Particle],
    max_workers: int,
    task,
) -> tuple[list, int]:
    workers = min(max_workers, len(particles))
    if workers == 1:
        return [task(particle) for particle in particles], workers
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(task, particles)), workers


def clean_particle_measurement(
    particle: Particle,
    candidate_zyx: NDArray[np.int64],
    candidate_values: NDArray,
    rotated_position: NDArray[np.float64],
    bit_depth: type,
    theta: float,
    phi: float,
    clean_background: float,
    voxels_z: int,
) -> ParticleStateMeasurement:
    retained_k = min(particle.measurement_top_k, len(candidate_values))
    selected = np.argpartition(
        candidate_values,
        len(candidate_values) - retained_k,
    )[-retained_k:]
    selected = selected[np.argsort(candidate_values[selected])[::-1]]
    top_k_zyx = candidate_zyx[selected]
    top_k_values = candidate_values[selected]
    return ParticleStateMeasurement(
        particle_id=particle.particle_id,
        state=(bit_depth, theta, phi),
        position_zyx=rotated_position,
        top_k=retained_k,
        top_k_zyx=top_k_zyx,
        particle_only_top_k_values=top_k_values,
        combined_top_k_values=top_k_values,
        noise_mean=clean_background,
        noise_std_by_z=np.full(voxels_z, np.nan, dtype=np.float64),
    )


def measure_clean_theta_group_particle_neighborhoods(
    clean_base: ParticleVolumeUint,
    clean_before: list[ParticleStateMeasurement],
    clean_background: float,
    interpolation_order: int,
    phi_angles: tuple[float, ...],
    settings: ParticleNeighborhoodOptimizationSettings,
    theta: float,
) -> tuple[float, dict[float, list[ParticleStateMeasurement]], int]:
    global_shape = clean_base.get_array().shape
    bit_depth = clean_base.get_array().dtype.type
    before_by_id = {
        measurement.particle_id: measurement for measurement in clean_before
    }

    def measure_particle(particle: Particle):
        measurements = {}
        points_by_phi = {}
        geometry_by_phi = {}
        for phi in phi_angles:
            if theta == 0.0 and phi == 0.0:
                measurements[phi] = before_by_id[particle.particle_id]
                continue
            rotated_position, candidate_zyx = rotated_particle_candidates(
                particle,
                theta,
                phi,
                interpolation_order,
                global_shape,
            )
            points_by_phi[phi] = candidate_zyx
            geometry_by_phi[phi] = (rotated_position, candidate_zyx)

        if points_by_phi:
            values_by_phi = sample_particle_neighborhoods(
                clean_base.get_array(),
                points_by_phi,
                theta,
                interpolation_order,
                settings.particle_spline_halo,
                global_shape,
            )
            for phi, values in values_by_phi.items():
                rotated_position, candidate_zyx = geometry_by_phi[phi]
                measurements[phi] = clean_particle_measurement(
                    particle,
                    candidate_zyx,
                    values,
                    rotated_position,
                    bit_depth,
                    theta,
                    phi,
                    clean_background,
                    global_shape[0],
                )
        return measurements

    particle_results, workers = map_particle_tasks(
        clean_base.particles,
        settings.particle_workers,
        measure_particle,
    )
    by_phi = {phi: [result[phi] for result in particle_results] for phi in phi_angles}
    return theta, by_phi, workers


def measure_noisy_theta_group_particle_neighborhoods(
    noisy_base: ParticleVolumeUint,
    clean_states: dict[tuple[int, float, float], list[ParticleStateMeasurement]],
    noisy_before: list[ParticleStateMeasurement],
    noise_states: dict[tuple[int, float, float], RotatedNoiseState],
    interpolation_order: int,
    phi_angles: tuple[float, ...],
    settings: ParticleNeighborhoodOptimizationSettings,
    theta: float,
) -> tuple[float, dict[float, list[ParticleStateMeasurement]], int]:
    source_data = noisy_base.get_array()
    global_shape = source_data.shape
    before_by_id = {
        measurement.particle_id: measurement for measurement in noisy_before
    }
    references_by_phi = {
        phi: {
            measurement.particle_id: measurement
            for measurement in clean_states[(interpolation_order, theta, phi)]
        }
        for phi in phi_angles
    }
    plane_noise_by_phi = {
        phi: ParticleStateMeasurement._to_plane_noise_std(
            noise_states[(interpolation_order, theta, phi)].profile,
            global_shape[0],
        )
        for phi in phi_angles
    }

    def measure_particle(particle: Particle):
        measurements = {}
        points_by_phi = {}
        for phi in phi_angles:
            reference = references_by_phi[phi][particle.particle_id]
            if theta == 0.0 and phi == 0.0:
                measurements[phi] = before_by_id[particle.particle_id]
            else:
                points_by_phi[phi] = reference.top_k_zyx

        if points_by_phi:
            values_by_phi = sample_particle_neighborhoods(
                source_data,
                points_by_phi,
                theta,
                interpolation_order,
                settings.particle_spline_halo,
                global_shape,
            )
            for phi, observed_values in values_by_phi.items():
                reference = references_by_phi[phi][particle.particle_id]
                noise_state = noise_states[(interpolation_order, theta, phi)]
                measurements[phi] = ParticleStateMeasurement(
                    particle_id=reference.particle_id,
                    state=reference.state,
                    position_zyx=reference.position_zyx,
                    top_k=reference.top_k,
                    top_k_zyx=reference.top_k_zyx,
                    particle_only_top_k_values=reference.particle_only_top_k_values,
                    combined_top_k_values=observed_values,
                    noise_mean=noise_state.mean,
                    noise_std_by_z=plane_noise_by_phi[phi],
                )
        return measurements

    particle_results, workers = map_particle_tasks(
        noisy_base.particles,
        settings.particle_workers,
        measure_particle,
    )
    by_phi = {phi: [result[phi] for result in particle_results] for phi in phi_angles}
    return theta, by_phi, workers


def particle_neighborhood_footprint_summary(
    config: ParticleRotationSweepConfig,
    settings: ParticleNeighborhoodOptimizationSettings,
) -> pd.DataFrame:
    """Summarize conservative source-neighborhood sizes without rotating data."""
    global_shape = (config.voxels_z, config.voxels_xy, config.voxels_xy)
    full_voxels = int(np.prod(global_shape, dtype=np.int64))
    records = []
    for particle in slab_reference_particles(config):
        for order in config.interpolation_orders:
            for theta in config.theta_angles_deg:
                points_by_phi = {
                    phi: rotated_particle_candidates(
                        particle,
                        theta,
                        phi,
                        order,
                        global_shape,
                    )[1]
                    for phi in config.phi_angles_deg
                    if theta != 0.0 or phi != 0.0
                }
                if not points_by_phi:
                    continue
                source_box, theta_box, _ = neighborhood_boxes_for_points(
                    points_by_phi,
                    theta,
                    order,
                    settings.particle_spline_halo,
                    global_shape,
                )
                records.append(
                    {
                        "interpolation_order": order,
                        "theta_deg": theta,
                        "particle_id": particle.particle_id,
                        "source_shape": source_box.shape,
                        "theta_shape": theta_box.shape,
                        "source_voxels": source_box.voxel_count,
                        "source_fraction_of_volume": source_box.voxel_count
                        / full_voxels,
                    }
                )
    return pd.DataFrame.from_records(records)


def run_particle_rotation_sweep_particle_neighborhoods(
    config: ParticleRotationSweepConfig,
    output_csv: Path = PARTICLE_SWEEP_OUTPUT_CSV,
    settings: ParticleNeighborhoodOptimizationSettings = (
        ParticleNeighborhoodOptimizationSettings()
    ),
    reset_output: bool = False,
) -> ParticleSweepRun:
    """Run or resume noise-in-slab, particle-neighborhood sweep conditions."""
    output_csv = Path(output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed_conditions = prepare_resumable_output(
        output_csv,
        expected_particle_ids(config),
        reset_output,
    )
    noise_slab_settings = settings.noise_slab_settings()
    bounds = calculate_particle_z_slab_bounds(config, noise_slab_settings)
    print(f"Writing particle-neighborhood metrics to {output_csv}")
    print(
        "noise z slabs: "
        f"input=[{bounds.input_z_start}, {bounds.input_z_stop}) "
        f"({bounds.input_depth}/{config.voxels_z}), "
        f"output=[{bounds.output_z_start}, {bounds.output_z_stop}) "
        f"({bounds.output_depth}/{config.voxels_z}); "
        f"noise workers={settings.noise_rotation_workers}, "
        f"particle workers={settings.particle_workers}, "
        f"particle halo={settings.particle_spline_halo}"
    )

    total_conditions = (
        len(config.noise_seeds)
        * len(config.bit_depths)
        * len(config.particle_sigmas)
        * len(config.amplitude_snrs)
        * len(config.interpolation_orders)
        * len(config.theta_angles_deg)
        * len(config.phi_angles_deg)
    )
    requested_conditions = {
        sweep_condition_key(
            seed,
            bit_depth,
            particle_sigma,
            amplitude_snr,
            interpolation_order,
            theta,
            phi,
            config.measurement_top_k,
        )
        for seed in config.noise_seeds
        for bit_depth in config.bit_depths
        for particle_sigma in config.particle_sigmas
        for amplitude_snr in config.amplitude_snrs
        for interpolation_order in config.interpolation_orders
        for theta in config.theta_angles_deg
        for phi in config.phi_angles_deg
    }
    pending_conditions = requested_conditions - completed_conditions

    def condition_is_pending(
        seed: int,
        bit_depth: type,
        particle_sigma: float,
        amplitude_snr: float,
        interpolation_order: int,
        theta: float,
        phi: float,
    ) -> bool:
        return (
            sweep_condition_key(
                seed,
                bit_depth,
                particle_sigma,
                amplitude_snr,
                interpolation_order,
                theta,
                phi,
                config.measurement_top_k,
            )
            in pending_conditions
        )

    volume_center_yx = np.array([config.voxels_xy // 2] * 2, dtype=np.float64)
    realized_seeds = tuple(seed for seed in config.noise_seeds if seed != NO_NOISE_SEED)
    rows_written = 0
    timing_records = []
    sweep_started = perf_counter()

    if not pending_conditions:
        print("All requested sweep conditions are already complete; nothing to run.")
        return ParticleSweepRun(
            output_csv=output_csv,
            rows_written=0,
            elapsed_seconds=perf_counter() - sweep_started,
            timings=pd.DataFrame(),
        )

    with tqdm(
        total=len(pending_conditions),
        desc="Particle-neighborhood sweep",
        unit="condition",
        dynamic_ncols=True,
    ) as progress_bar:
        for bit_depth in config.bit_depths:
            clean_cache = {}

            for particle_sigma in config.particle_sigmas:
                for amplitude_snr in config.amplitude_snrs:
                    setup_started = perf_counter()
                    particle_fp32 = create_uniform_particle_volume(
                        voxels_z=config.voxels_z,
                        voxels_xy=config.voxels_xy,
                        sigma=particle_sigma,
                        amplitude=amplitude_snr,
                        z_start_rel=config.z_start_rel,
                        z_end_rel=config.z_end_rel,
                        delta_z_rel=config.z_delta_rel,
                        delta_angle_deg=config.radial_arm_spacing_deg,
                        stop_angle_deg=config.quadrant_stop_deg,
                        measurement_top_k=config.measurement_top_k,
                    )
                    clean_base = ParticleVolumeUint(
                        particle_fp32,
                        type=bit_depth,
                        target_std_scale=config.target_std_scale,
                    )
                    clean_background = float(np.rint(np.iinfo(bit_depth).max / 2))
                    flat_noise_profile = np.full(config.voxels_z, np.nan)
                    clean_before = measure_particle_states(
                        clean_base,
                        clean_base.get_array(),
                        clean_background,
                        flat_noise_profile,
                        (bit_depth, 0.0, 0.0),
                    )
                    timing_records.append(
                        timing_record(
                            phase="clean_setup_particle_neighborhood",
                            bit_depth=bit_depth,
                            noise_seed=NO_NOISE_SEED,
                            interpolation_order=None,
                            workers=1,
                            conditions_written=0,
                            compute_seconds=perf_counter() - setup_started,
                            particle_sigma=particle_sigma,
                            amplitude_snr=amplitude_snr,
                        )
                    )

                    clean_states = {}
                    for interpolation_order in config.interpolation_orders:
                        compute_started = perf_counter()
                        theta_results = [
                            measure_clean_theta_group_particle_neighborhoods(
                                clean_base,
                                clean_before,
                                clean_background,
                                interpolation_order,
                                config.phi_angles_deg,
                                settings,
                                theta,
                            )
                            for theta in config.theta_angles_deg
                        ]
                        compute_seconds = perf_counter() - compute_started
                        write_seconds = 0.0
                        conditions_written = 0
                        workers = 1
                        for theta, measurements_by_phi, workers in theta_results:
                            for phi in config.phi_angles_deg:
                                state_key = (interpolation_order, theta, phi)
                                clean_after = measurements_by_phi[phi]
                                clean_states[state_key] = clean_after
                                row_count, condition_write_seconds = (
                                    write_condition_rows(
                                        output_csv,
                                        NO_NOISE_SEED,
                                        bit_depth,
                                        particle_sigma,
                                        amplitude_snr,
                                        interpolation_order,
                                        theta,
                                        phi,
                                        volume_center_yx,
                                        clean_before,
                                        clean_after,
                                        clean_before,
                                        clean_after,
                                    )
                                )
                                rows_written += row_count
                                write_seconds += condition_write_seconds
                                conditions_written += 1
                                update_sweep_progress(
                                    progress_bar,
                                    "clean neighborhoods",
                                    NO_NOISE_SEED,
                                    bit_depth,
                                    particle_sigma,
                                    amplitude_snr,
                                    interpolation_order,
                                    theta,
                                    phi,
                                    advance=True,
                                )
                        timing_records.append(
                            timing_record(
                                phase="clean_rotation_particle_neighborhood",
                                bit_depth=bit_depth,
                                noise_seed=NO_NOISE_SEED,
                                interpolation_order=interpolation_order,
                                workers=workers,
                                conditions_written=conditions_written,
                                compute_seconds=compute_seconds,
                                write_seconds=write_seconds,
                                particle_sigma=particle_sigma,
                                amplitude_snr=amplitude_snr,
                            )
                        )

                    clean_cache[(particle_sigma, amplitude_snr)] = (
                        clean_before,
                        clean_states,
                    )
                    del clean_base, particle_fp32

            for seed in realized_seeds:
                noise_setup_started = perf_counter()
                np.random.seed(seed)
                standard_noise = StandardNoiseVolume(
                    voxels_z=config.voxels_z,
                    voxels_xy=config.voxels_xy,
                    rho1_z=config.rho1_z,
                    rho1_xy=config.rho1_xy,
                )
                target_std = np.iinfo(bit_depth).max * config.target_std_scale
                noise_base = standard_noise.get_quantized_noise(
                    target_std=target_std,
                    uint_type=bit_depth,
                )
                base_slab = noise_base.data[
                    bounds.output_z_start : bounds.output_z_stop
                ]
                base_noise_state = noise_state_from_z_slab(
                    noise_base,
                    base_slab,
                    theta=0.0,
                    phi=0.0,
                    order=max(config.interpolation_orders),
                    bounds=bounds,
                )
                timing_records.append(
                    timing_record(
                        phase="noise_setup_z_slab",
                        bit_depth=bit_depth,
                        noise_seed=seed,
                        interpolation_order=None,
                        workers=1,
                        conditions_written=0,
                        compute_seconds=perf_counter() - noise_setup_started,
                    )
                )

                noise_states = {}
                for interpolation_order in config.interpolation_orders:
                    compute_started = perf_counter()
                    worker = partial(
                        measure_noise_theta_group_z_slab,
                        noise_base,
                        base_noise_state,
                        interpolation_order,
                        config.phi_angles_deg,
                        bounds,
                    )
                    theta_results, workers = run_z_slab_theta_groups(
                        config.theta_angles_deg,
                        config.phi_angles_deg,
                        settings.noise_rotation_workers,
                        worker,
                    )
                    for theta, states_by_phi in theta_results:
                        for phi, noise_state in states_by_phi.items():
                            noise_states[(interpolation_order, theta, phi)] = (
                                noise_state
                            )
                    timing_records.append(
                        timing_record(
                            phase="noise_rotation_z_slab",
                            bit_depth=bit_depth,
                            noise_seed=seed,
                            interpolation_order=interpolation_order,
                            workers=workers,
                            conditions_written=0,
                            compute_seconds=perf_counter() - compute_started,
                        )
                    )
                del noise_base

                for particle_sigma in config.particle_sigmas:
                    for amplitude_snr in config.amplitude_snrs:
                        noisy_setup_started = perf_counter()
                        particle_fp32 = create_uniform_particle_volume(
                            voxels_z=config.voxels_z,
                            voxels_xy=config.voxels_xy,
                            sigma=particle_sigma,
                            amplitude=amplitude_snr,
                            z_start_rel=config.z_start_rel,
                            z_end_rel=config.z_end_rel,
                            delta_z_rel=config.z_delta_rel,
                            delta_angle_deg=config.radial_arm_spacing_deg,
                            stop_angle_deg=config.quadrant_stop_deg,
                            measurement_top_k=config.measurement_top_k,
                        )
                        particle_fp32.get_array()[:] += standard_noise.get_array()
                        noisy_base = ParticleVolumeUint(
                            particle_fp32,
                            type=bit_depth,
                            target_std_scale=config.target_std_scale,
                        )
                        clean_before, clean_states = clean_cache[
                            (particle_sigma, amplitude_snr)
                        ]
                        noisy_before = sample_observed_states(
                            clean_before,
                            noisy_base.get_array(),
                            base_noise_state,
                        )
                        timing_records.append(
                            timing_record(
                                phase="noisy_setup_particle_neighborhood",
                                bit_depth=bit_depth,
                                noise_seed=seed,
                                interpolation_order=None,
                                workers=1,
                                conditions_written=0,
                                compute_seconds=perf_counter() - noisy_setup_started,
                                particle_sigma=particle_sigma,
                                amplitude_snr=amplitude_snr,
                            )
                        )

                        for interpolation_order in config.interpolation_orders:
                            compute_started = perf_counter()
                            theta_results = [
                                measure_noisy_theta_group_particle_neighborhoods(
                                    noisy_base,
                                    clean_states,
                                    noisy_before,
                                    noise_states,
                                    interpolation_order,
                                    config.phi_angles_deg,
                                    settings,
                                    theta,
                                )
                                for theta in config.theta_angles_deg
                            ]
                            compute_seconds = perf_counter() - compute_started
                            write_seconds = 0.0
                            conditions_written = 0
                            workers = 1
                            for theta, measurements_by_phi, workers in theta_results:
                                for phi in config.phi_angles_deg:
                                    state_key = (interpolation_order, theta, phi)
                                    row_count, condition_write_seconds = (
                                        write_condition_rows(
                                            output_csv,
                                            seed,
                                            bit_depth,
                                            particle_sigma,
                                            amplitude_snr,
                                            interpolation_order,
                                            theta,
                                            phi,
                                            volume_center_yx,
                                            clean_before,
                                            clean_states[state_key],
                                            noisy_before,
                                            measurements_by_phi[phi],
                                        )
                                    )
                                    rows_written += row_count
                                    write_seconds += condition_write_seconds
                                    conditions_written += 1
                                    update_sweep_progress(
                                        progress_bar,
                                        "noisy neighborhoods",
                                        seed,
                                        bit_depth,
                                        particle_sigma,
                                        amplitude_snr,
                                        interpolation_order,
                                        theta,
                                        phi,
                                        advance=True,
                                    )
                            timing_records.append(
                                timing_record(
                                    phase="noisy_rotation_particle_neighborhood",
                                    bit_depth=bit_depth,
                                    noise_seed=seed,
                                    interpolation_order=interpolation_order,
                                    workers=workers,
                                    conditions_written=conditions_written,
                                    compute_seconds=compute_seconds,
                                    write_seconds=write_seconds,
                                    particle_sigma=particle_sigma,
                                    amplitude_snr=amplitude_snr,
                                )
                            )
                        del noisy_base, particle_fp32
                del standard_noise, noise_states
            del clean_cache

    elapsed_seconds = perf_counter() - sweep_started
    timings = pd.DataFrame.from_records(timing_records)
    timing_summary = timings.groupby(
        ["phase", "interpolation_order"],
        dropna=False,
    )[["compute_seconds", "write_seconds", "elapsed_seconds"]].sum()
    print("\nParticle-neighborhood timing summary (seconds):")
    print(timing_summary.to_string())
    return ParticleSweepRun(
        output_csv=output_csv,
        rows_written=rows_written,
        elapsed_seconds=elapsed_seconds,
        timings=timings,
    )


def validate_particle_neighborhood_optimization() -> pd.DataFrame:
    """Compare neighborhood rotations with z-slab rotations on a quick sweep."""
    from scripts.particle_sim_sweep import particle_sweep_config

    validation_config = replace(
        particle_sweep_config,
        voxels_z=192,
        voxels_xy=192,
        particle_sigmas=(2.0,),
        amplitude_snrs=(5.0,),
        interpolation_orders=(1, 5),
        bit_depths=(np.uint8,),
        theta_angles_deg=(15.0,),
        phi_angles_deg=(0.1, 5.0),
        noise_seeds=(NO_NOISE_SEED, 10),
    )
    settings = ParticleNeighborhoodOptimizationSettings(
        noise_rotation_workers=2,
        particle_workers=4,
        particle_spline_halo=32,
        noise_spline_halo=32,
    )
    validation_directory = Path("Data/_tmp").resolve()
    slab_csv = validation_directory / "particle_neighborhood_slab_reference.csv"
    neighborhood_csv = validation_directory / "particle_neighborhood_optimized.csv"
    slab_run = run_particle_rotation_sweep_z_slab(
        validation_config,
        output_csv=slab_csv,
        settings=settings.noise_slab_settings(),
    )
    neighborhood_run = run_particle_rotation_sweep_particle_neighborhoods(
        validation_config,
        output_csv=neighborhood_csv,
        settings=settings,
    )
    comparison = compare_z_slab_results(
        neighborhood_run.output_csv,
        slab_run.output_csv,
        retention_tolerance=1e-10,
    )
    reconstructed_samples = comparison["metric"].isin(
        [
            "observed_peak_before",
            "observed_peak_after",
            "observed_top_k_sum_before",
            "observed_top_k_sum_after",
        ]
    )
    if not (comparison.loc[reconstructed_samples, "max_abs_difference"] < 1e-10).all():
        raise AssertionError("Neighborhood rotations changed sampled uint values")
    print(
        "Neighborhood validation: "
        f"z-slab={slab_run.elapsed_seconds:.3f}s, "
        f"neighborhood={neighborhood_run.elapsed_seconds:.3f}s"
    )
    slab_csv.unlink(missing_ok=True)
    neighborhood_csv.unlink(missing_ok=True)
    return comparison
