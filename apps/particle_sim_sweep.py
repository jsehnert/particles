"""Run the optimized particle-noise and rotation sweep.

Examples
--------
Inspect the default production configuration without allocating volumes::

    python apps/particle_sim_sweep.py --dry-run

Run a small uint8 pilot::

    python apps/particle_sim_sweep.py \
        --bit-depths uint8 --noise-seeds -1 10 \
        --particle-sigmas 2 --amplitude-snrs 5 --interpolation-orders 3 \
        --theta-angles-deg 15 --phi-angles-deg 0.1 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

# Make direct execution (``python apps/particle_sim_sweep.py``) resolve the
# project-root modules in the same way as an import from the notebook.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


NO_NOISE_SEED = -1


@dataclass(frozen=True)
class ParticleRotationSweepConfig:
    """Immutable configuration for the particle noise-and-rotation sweep."""

    voxels_z: int = 1024
    voxels_xy: int = 1024
    particle_sigmas: tuple[float, ...] = (0.5,)  # (1.0, 2.0, 4.0)
    amplitude_snrs: tuple[float, ...] = (10.0,)
    measurement_top_k: int = 7
    interpolation_orders: tuple[int, ...] = (1, 3, 5)

    # Four planes: center, then three evenly spaced toward the top.
    z_start_rel: float = 1 / 2
    z_end_rel: float = 0.0
    z_delta_rel: float = -1 / 8

    # Three noncentral radii are produced by create_uniform_particle_volume.
    radial_arm_spacing_deg: float = 15.0
    quadrant_stop_deg: float = 90.0

    bit_depths: tuple[type, ...] = (np.uint8, np.uint16)
    theta_angles_deg: tuple[float, ...] = (
        0.0,
        0.1,
        1.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
    )
    phi_angles_deg: tuple[float, ...] = (0.0, 0.1, 0.2, 1.0, 5.0)

    target_std_scale: float = 4 / 255.0
    rho1_z: float = 0.5
    rho1_xy: float = 0.5
    # Start with five paired realizations; increase until the summary CIs stabilize.
    noise_seeds: tuple[int, ...] = (NO_NOISE_SEED, 10, 11, 12, 13, 14)


# The notebook imports this exact default. CLI values create a replacement
# configuration and never mutate it.
particle_sweep_config = ParticleRotationSweepConfig()


from particle_experiment_tools import (  # noqa: E402
    PARTICLE_SWEEP_OUTPUT_CSV,
    ParticleNeighborhoodOptimizationSettings,
    run_particle_rotation_sweep_particle_neighborhoods,
)

BIT_DEPTHS = {"uint8": np.uint8, "uint16": np.uint16}


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the canonical sweep runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PARTICLE_SWEEP_OUTPUT_CSV,
        help="Destination CSV (default: %(default)s).",
    )
    parser.add_argument("--voxels-z", type=int)
    parser.add_argument("--voxels-xy", type=int)
    parser.add_argument("--particle-sigmas", type=float, nargs="+")
    parser.add_argument("--amplitude-snrs", type=float, nargs="+")
    parser.add_argument("--measurement-top-k", type=int)
    parser.add_argument("--interpolation-orders", type=int, nargs="+")
    parser.add_argument("--z-start-rel", type=float)
    parser.add_argument("--z-end-rel", type=float)
    parser.add_argument("--z-delta-rel", type=float)
    parser.add_argument("--radial-arm-spacing-deg", type=float)
    parser.add_argument("--quadrant-stop-deg", type=float)
    parser.add_argument("--bit-depths", choices=tuple(BIT_DEPTHS), nargs="+")
    parser.add_argument("--theta-angles-deg", type=float, nargs="+")
    parser.add_argument("--phi-angles-deg", type=float, nargs="+")
    parser.add_argument("--target-std-scale", type=float)
    parser.add_argument("--rho1-z", type=float)
    parser.add_argument("--rho1-xy", type=float)
    parser.add_argument("--noise-seeds", type=int, nargs="+")
    parser.add_argument("--noise-rotation-workers", type=int)
    parser.add_argument("--particle-workers", type=int)
    parser.add_argument("--particle-spline-halo", type=int)
    parser.add_argument("--noise-spline-halo", type=int)
    parser.add_argument("--profile-guard", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and exit without running.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete the existing output CSV before the run instead of resuming it.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ParticleRotationSweepConfig:
    """Apply only explicitly supplied CLI values to ``particle_sweep_config``."""
    scalar_fields = (
        "voxels_z",
        "voxels_xy",
        "measurement_top_k",
        "z_start_rel",
        "z_end_rel",
        "z_delta_rel",
        "radial_arm_spacing_deg",
        "quadrant_stop_deg",
        "target_std_scale",
        "rho1_z",
        "rho1_xy",
    )
    updates = {
        field: value
        for field in scalar_fields
        if (value := getattr(args, field)) is not None
    }
    for field in (
        "particle_sigmas",
        "amplitude_snrs",
        "interpolation_orders",
        "theta_angles_deg",
        "phi_angles_deg",
        "noise_seeds",
    ):
        value = getattr(args, field)
        if value is not None:
            updates[field] = tuple(value)
    if args.bit_depths is not None:
        updates["bit_depths"] = tuple(BIT_DEPTHS[name] for name in args.bit_depths)
    return replace(particle_sweep_config, **updates)


def settings_from_args(
    args: argparse.Namespace,
) -> ParticleNeighborhoodOptimizationSettings:
    """Apply optimization-specific CLI overrides to the default settings."""
    defaults = ParticleNeighborhoodOptimizationSettings()
    updates = {
        field: value
        for field in (
            "noise_rotation_workers",
            "particle_workers",
            "particle_spline_halo",
            "noise_spline_halo",
            "profile_guard",
        )
        if (value := getattr(args, field)) is not None
    }
    return replace(defaults, **updates)


def condition_count(config: ParticleRotationSweepConfig) -> int:
    """Return the number of sweep conditions before expansion by particle ID."""
    return int(
        np.prod(
            (
                len(config.noise_seeds),
                len(config.bit_depths),
                len(config.particle_sigmas),
                len(config.amplitude_snrs),
                len(config.interpolation_orders),
                len(config.theta_angles_deg),
                len(config.phi_angles_deg),
            )
        )
    )


def printable_config(config: ParticleRotationSweepConfig) -> dict[str, object]:
    """Convert NumPy dtype classes to stable names for CLI reporting."""
    values = asdict(config)
    values["bit_depths"] = [np.dtype(dtype).name for dtype in config.bit_depths]
    return values


def main(argv: list[str] | None = None) -> int:
    """Run the configured canonical particle-neighborhood sweep."""
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    settings = settings_from_args(args)
    print(json.dumps(printable_config(config), indent=2))
    print(f"Conditions: {condition_count(config):,}")
    print(f"Output: {args.output_csv}")
    print("Mode: reset output" if args.reset_output else "Mode: resume output")
    if args.dry_run:
        return 0

    run = run_particle_rotation_sweep_particle_neighborhoods(
        config,
        output_csv=args.output_csv,
        settings=settings,
        reset_output=args.reset_output,
    )
    print(
        f"Completed {run.rows_written:,} particle rows in {run.elapsed_seconds:.1f}s "
        f"({run.elapsed_seconds / condition_count(config):.1f}s per condition)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
