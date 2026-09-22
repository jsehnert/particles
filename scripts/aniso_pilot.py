#!/usr/bin/env python3
"""
Standalone pilot: rerun candidate detection with the aniso_factor shape gate
relaxed (effectively disabled) against a small, targeted subset of
(slab_thickness, cell) combinations, without touching vol_analysis.py or
config.toml.

Why this exists
----------------
`_axial_ok`'s large-component branch rejects a candidate whenever
``z_extent > aniso_factor * min(y_extent, x_extent) + z_pad``. This is meant
to cut wall/winding remnants -- thin in cross-section, long in z. But the
check only looks at the axis-ALIGNED bounding box, so it can't tell a real
tube (thin in *both* lateral directions) from a disk-shaped particle whose
flat face happens to lie in, say, the Y-Z plane: that disk has one small
lateral extent (its thickness, along whichever axis its normal points down)
and one LARGE lateral extent (its diameter), and z_extent large too (also its
diameter) -- the exact same "z_extent large, lat_min small" signature as a
wall, purely because of how it's oriented relative to the scan axes, not
because of its actual shape.

Checking the labeled TP population confirms disk-shaped true particles are a
real, recurring morphology here (see chat discussion): 157 confirmed TPs
(n_voxels >= 15, to exclude small-N eigenvalue noise) show a clear
plate/disk PCA signature (planarity > 0.5 and planarity > linearity), spread
across all 19 volumes and most experiments. Their current bounding boxes are
only mildly anisotropic (avg max/min extent ratio ~1.46) -- consistent with
them being caught in a favorable orientation. Since real particles have no
reason to prefer one orientation over another relative to the scanner axes,
similarly-shaped disks caught less favorably (edge-on) are plausibly being
rejected by this gate right now, and we'd never know: `_process_slab` only
computes the PCA shape features (linearity/planarity/sphericity) for
candidates that SURVIVE `_axial_ok`, so a rejected disk never gets far enough
to reveal what it actually was.

This script reruns detection with aniso_factor pushed very high (effectively
disabling that one check, while leaving z_extent_max, min_fill, and every
other gate untouched) so every candidate that would normally be rejected by
the aniso ratio alone gets through to feature computation. The analysis step
(after this script runs) then uses the now-available planarity/linearity to
separate recovered disks from recovered noise (streaks/walls the gate was
correctly built to reject) -- this script's job is only to generate that
population, not to classify it.

Cell/slab_thickness selection (see chat discussion for the full analysis):
  - aniso_factor (1.2) and z_pad (2) are constant across the entire historical
    20-experiment sweep -- this parameter has never been varied, so there's no
    existing data on what a relaxed gate would surface. Has to be piloted.
  - slab_thickness=23 has the richest confirmed disk-shaped TP population
    (35 candidates meeting the n_voxels>=15 / planarity>0.5 / planarity>linearity
    filter above, spread across 13 volumes and all 5 of its low_threshold_scale
    experiments), giving the best odds that similarly-shaped, differently
    -oriented disks exist nearby in the same cells.
  - M50L-01, M50L-06, M50L-12 are the top 3 cells by that same disk-TP count
    within slab_thickness=23 (6, 4, 4 respectively).
These are just the defaults below -- override with --slab-thickness / --cell.

Schema: candidates.csv/voxels.csv rows carry the same experiment_number +
EXPERIMENT_PARAM_FIELDS columns (baked in at write time) as the production
Data/experiments/<experiment_number>/ output -- experiment_number is the real
production experiment_number for the (slab_thickness, low_threshold_scale)
combo being rerun here (just with aniso_factor relaxed and everything else,
including min_fill, matching production), so pilot and production rows are
directly concatenable without a join against params.csv.

Efficiency: stage-1 residual precomputation depends only on slab_thickness +
baseline_method, not on low_threshold_scale or aniso_factor. This script
precomputes the residual ONCE per (cell, slab_thickness) and reuses it across
every low_threshold_scale already defined for that family in config.toml, at
the relaxed aniso_factor -- so stage 1 is paid once per cell here, not once
per (cell, k_low) pair.

Usage:
    python scripts/aniso_pilot.py \
        --slab-thickness 23 \
        --cell M50L-01 --cell M50L-06 --cell M50L-12 \
        --aniso-factor 1000

Reruns are safe by default: for each (slab_thickness, low_threshold_scale)
combo, a cell already present in that combo's candidates.csv is skipped
rather than re-appended (would duplicate rows) or overwritten (would destroy
whatever other cells' data is already in that file). So if you already ran
--cell M50L-01 and now want to add the other two, just rerun with all three
--cell flags (or any --cell list you like) -- M50L-01 will be skipped, and
only the others will actually be processed. Pass --force if you specifically
want to redo a cell that's already there (its old rows are removed first,
then replaced -- never duplicated).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
sys.path.insert(0, str(PROJECT_ROOT))

from extract_candidates_3d import (  # noqa: E402
    VOXEL_FIELDS,
    Candidate,
    detect_candidates_parallel,
    detect_candidates_streaming,
    precompute_residual,
)
from scripts.vol_analysis import (  # noqa: E402
    _CANDIDATE_FIELDS,
    HISTORICAL_MIN_FILL,
    GlobalData,
    _experiment_param_values,
    _load_raw_config,
    write_candidate_rows,
)

DEFAULT_CELLS: list[str] = ["M50L-01", "M50L-06", "M50L-12"]
DEFAULT_SLAB_THICKNESSES: list[int] = [23]
OUT_ROOT: Path = PROJECT_ROOT / "Data" / "experiments_aniso_pilot"
TMP_ROOT: Path = PROJECT_ROOT / "Data" / "_tmp_aniso_pilot"


def _experiments_for_slab_thickness(
    cfg: dict, slab_thickness: int
) -> list[tuple[int, dict]]:
    """All [[experiment]] entries (index, entry) sharing this slab_thickness,
    i.e. the existing low_threshold_scale sweep for that background window."""
    matches = [
        (i, exp)
        for i, exp in enumerate(cfg["experiment"])
        if int(exp["slab_thickness"]) == slab_thickness
    ]
    if not matches:
        raise ValueError(
            f"No [[experiment]] entries with slab_thickness={slab_thickness}"
        )
    return matches


def _vol_index_for_cell(cfg: dict, cell_name: str) -> int:
    for i, cell in enumerate(cfg["cell"]):
        if cell["name"] == cell_name:
            return i
    raise ValueError(f"No [[cell]] entry named {cell_name!r}")


def _write_params_csv(
    out_path: Path, experiment: dict, aniso_factor: float, low_threshold_scale: float
) -> None:
    """Same convention as vol_analysis.write_experiment_params_csv, plus the
    min_fill column the production schema never captured (fixed at the
    historical production value here, since this pilot only varies
    aniso_factor). `experiment` must be this specific combo's own
    [[experiment]] entry (not the sweep's basis entry) so experiment_number
    and low_threshold_scale are both correct for this output directory.
    aniso_factor is overridden to the relaxed test value."""
    row: dict[str, object] = {}
    for key, value in experiment.items():
        if isinstance(value, list):
            value = ";".join(str(v) for v in value)
        row[key] = value
    row["low_threshold_scale"] = low_threshold_scale
    row["aniso_factor"] = aniso_factor
    row["min_fill"] = HISTORICAL_MIN_FILL
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _open_mode(csv_path: Path) -> str:
    """'w' (fresh, header included) if this CSV doesn't exist yet ON DISK, 'a'
    (append, no header) if it does -- checked against the filesystem, not an
    in-process set, so a SECOND, LATER invocation of this script that adds new
    cells for a combo it already has data for appends instead of clobbering
    what an earlier run wrote."""
    return "a" if csv_path.exists() and csv_path.stat().st_size > 0 else "w"


def _cell_already_present(csv_path: Path, cell_name: str) -> bool:
    """True if csv_path already has rows for this cell_name in its
    'volume_name' column (candidates.csv and voxels.csv both use that column
    name). Used to make reruns idempotent instead of silently duplicating or
    clobbering a previous run's data for a DIFFERENT cell."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return any(row.get("volume_name") == cell_name for row in reader)


def _strip_cell_rows(csv_path: Path, cell_name: str) -> None:
    """Rewrite csv_path in place, dropping every row whose volume_name is
    cell_name. Used by --force to replace a cell's existing rows rather than
    duplicate them."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get("volume_name") != cell_name]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_cell_slab(
    cfg: dict,
    cell_name: str,
    slab_thickness: int,
    aniso_factor: float,
    run_parallel: bool,
    n_workers_override: int | None,
    force: bool = False,
) -> None:
    """Precompute the residual once for (cell_name, slab_thickness), then run
    stage-2 detection at the relaxed aniso_factor (min_fill and every other
    gate left at production values) for every low_threshold_scale value
    already swept for that slab_thickness family in config.toml.

    Idempotent by default: if cell_name already has rows in a combo's
    candidates.csv (from an earlier invocation of this script), that combo is
    skipped rather than re-appended (which would duplicate rows) or
    overwritten (which would destroy other cells' data already in that file).
    Pass force=True to replace an existing cell's rows instead of skipping.
    """
    vol_index = _vol_index_for_cell(cfg, cell_name)
    exp_matches = _experiments_for_slab_thickness(cfg, slab_thickness)
    basis_exp_index, basis_experiment = exp_matches[0]
    low_values = sorted(float(exp["low_threshold_scale"]) for _, exp in exp_matches)

    def combo_dir(low: float) -> Path:
        return OUT_ROOT / f"slab{slab_thickness}_low{low}_aniso{aniso_factor:g}"

    if not force:
        pending = [
            low
            for low in low_values
            if not _cell_already_present(combo_dir(low) / "candidates.csv", cell_name)
        ]
        if not pending:
            typer.echo(
                f"\n=== {cell_name} @ slab_thickness={slab_thickness}: already present in "
                f"all {len(low_values)} combo(s), skipping (pass --force to redo) ==="
            )
            return
        if len(pending) < len(low_values):
            typer.echo(
                f"\n=== {cell_name} @ slab_thickness={slab_thickness}: already present in "
                f"{len(low_values) - len(pending)}/{len(low_values)} combo(s); "
                f"only running the missing low_threshold_scale values {pending} ==="
            )

    typer.echo(
        f"\n=== {cell_name} @ slab_thickness={slab_thickness}: "
        f"low_threshold_scale sweep {low_values}, aniso_factor={aniso_factor:g} "
        f"(min_fill left at production {HISTORICAL_MIN_FILL}) ==="
    )
    typer.echo(f"Loading volume + noise stats for {cell_name}...")
    global_data = GlobalData(
        CONFIG_PATH, vol_index=vol_index, exp_index=basis_exp_index
    )

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tag = f"{cell_name}_slab{slab_thickness}"
    residual_memmap_path = TMP_ROOT / f"residual_{tag}.dat"
    sigma_memmap_path = TMP_ROOT / f"sigma_{tag}.npy"

    begin = time.time()
    vol: np.memmap = global_data.vol
    src = precompute_residual(
        global_data,
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
    typer.echo(
        f"precompute_residual: {time.time() - begin:.2f}s "
        f"(computed once, reused across all {len(low_values)} low_threshold_scale values)"
    )

    # capture every scalar detection parameter BEFORE free_volume(): metal_threshold
    # is lazily computed from vol_for_analysis, which becomes unusable once freed.
    # (precompute_residual above already forces this computation as a side effect
    # of residual generation, but capturing explicitly here doesn't depend on that.)
    k_high = float(global_data.high_threshold_scale)
    min_seed_voxels = int(global_data.min_seed_voxels)
    z_extent_max = int(global_data.z_extent_max)
    small_z_bounds = global_data.small_z_bounds
    small_voxel_cutoff = int(global_data.small_vol_cutoff)
    z_pad = int(global_data.z_pad)
    metal_threshold = float(global_data.metal_threshold)
    z_offset = int(global_data.z_min)
    slices_per_chunk = int(global_data.slices_per_chunk)
    n_workers = int(n_workers_override or global_data.n_workers)
    # NB: global_data.aniso_factor (the production value, 1.2) is deliberately
    # NOT used below -- this pilot overrides it with the relaxed test value.

    global_data.free_volume()

    try:
        for _, experiment in exp_matches:
            low = float(experiment["low_threshold_scale"])
            # The real production experiment_number for this exact
            # (slab_thickness, low_threshold_scale) combo -- config.toml
            # defines one per [[experiment]] entry, and exp_matches already
            # filtered to this slab_thickness family, so each entry here maps
            # 1:1 to the production Data/experiments/<experiment_number>/ dir
            # sharing these params. Baking it (and the rest of
            # EXPERIMENT_PARAM_FIELDS) into every candidates.csv/voxels.csv
            # row keeps pilot output directly concatenable with production
            # rows -- same schema, real experiment_number, just a different
            # aniso_factor.
            experiment_number = int(experiment["experiment_number"])
            experiment_params = _experiment_param_values(experiment, HISTORICAL_MIN_FILL)
            experiment_params["aniso_factor"] = aniso_factor
            experiment_params_list = list(experiment_params.values())
            out_dir = combo_dir(low)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path_features = out_dir / "candidates.csv"
            out_path_voxels = out_dir / "voxels.csv"
            # NB: use this combo's own `experiment` entry here, not
            # `basis_experiment` -- each low_threshold_scale value in the
            # sweep is a DIFFERENT [[experiment]] entry with its own
            # experiment_number (basis_experiment is only used above to seed
            # GlobalData/residual precomputation, which is shared).
            _write_params_csv(out_dir / "params.csv", experiment, aniso_factor, low)

            already_present = _cell_already_present(out_path_features, cell_name)
            if already_present and not force:
                typer.echo(f"  low_threshold_scale={low}: already present, skipping")
                continue
            if already_present and force:
                typer.echo(
                    f"  low_threshold_scale={low}: --force set, replacing existing rows for {cell_name}"
                )
                _strip_cell_rows(out_path_features, cell_name)
                _strip_cell_rows(out_path_voxels, cell_name)

            # NB: mode is derived from candidates.csv only; voxels.csv is always
            # written/stripped in lockstep with it, so the two never disagree.
            mode = _open_mode(out_path_features)
            with (
                out_path_features.open(mode, newline="") as f_features,
                out_path_voxels.open(mode, newline="") as f_voxels,
            ):
                feature_writer = csv.DictWriter(
                    f_features, fieldnames=_CANDIDATE_FIELDS
                )
                voxel_writer = csv.writer(f_voxels)
                if mode == "w":
                    feature_writer.writeheader()
                    voxel_writer.writerow(VOXEL_FIELDS)

                begin = time.time()
                if run_parallel:
                    candidates: list[Candidate] = detect_candidates_parallel(
                        src=src,
                        k_high=k_high,
                        k_low=low,
                        min_seed_voxels=min_seed_voxels,
                        z_extent_max=z_extent_max,
                        small_z_bounds=small_z_bounds,
                        small_voxel_cutoff=small_voxel_cutoff,
                        aniso_factor=aniso_factor,
                        z_pad=z_pad,
                        min_fill=HISTORICAL_MIN_FILL,
                        slices_per_chunk=slices_per_chunk,
                        z_offset=z_offset,
                        n_workers=n_workers,
                        metal_threshold=metal_threshold,
                        voxel_writer=voxel_writer,
                        volume_name=cell_name,
                        storage_margin=3,
                        experiment_number=experiment_number,
                        experiment_params=experiment_params_list,
                    )
                else:
                    candidates = detect_candidates_streaming(
                        src=src,
                        k_high=k_high,
                        k_low=low,
                        min_seed_voxels=min_seed_voxels,
                        z_extent_max=z_extent_max,
                        small_z_bounds=small_z_bounds,
                        small_voxel_cutoff=small_voxel_cutoff,
                        aniso_factor=aniso_factor,
                        z_pad=z_pad,
                        min_fill=HISTORICAL_MIN_FILL,
                        slices_per_chunk=slices_per_chunk,
                        z_offset=z_offset,
                        metal_threshold=metal_threshold,
                        voxel_writer=voxel_writer,
                        volume_name=cell_name,
                        storage_margin=3,
                        experiment_number=experiment_number,
                        experiment_params=experiment_params_list,
                    )
                write_candidate_rows(
                    feature_writer,
                    candidates,
                    cell_name,
                    experiment_number,
                    experiment_params,
                )
                typer.echo(
                    f"  low_threshold_scale={low}: {len(candidates)} candidates "
                    f"({time.time() - begin:.2f}s) -> {out_dir}"
                )
    finally:
        src.free_volume()
        residual_memmap_path.unlink(missing_ok=True)
        sigma_memmap_path.unlink(missing_ok=True)


def main(
    slab_thickness: list[int] = typer.Option(
        DEFAULT_SLAB_THICKNESSES,
        "--slab-thickness",
        help="slab_thickness value(s) to test",
    ),
    cell: list[str] = typer.Option(
        DEFAULT_CELLS, "--cell", help="cell name(s) to test, e.g. M50L-01"
    ),
    aniso_factor: float = typer.Option(
        1000.0,
        "--aniso-factor",
        help="relaxed aniso_factor to test (large value effectively disables the z_extent/lat_min gate)",
    ),
    run_parallel: bool = typer.Option(
        True, "--run-parallel/--no-run-parallel", help="use the loky-parallel detector"
    ),
    n_workers: int | None = typer.Option(
        None, "--n-workers", help="override config.toml's [analysis].n_workers"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Replace a cell's existing rows in a combo instead of skipping it. "
            "Default behavior is idempotent: rerunning with cells already present "
            "in a combo's candidates.csv skips them, so it's always safe to pass "
            "the full --cell list again to pick up ones you haven't run yet."
        ),
    ),
) -> None:
    cfg = _load_raw_config(CONFIG_PATH)
    for st in slab_thickness:
        for c in cell:
            run_cell_slab(cfg, c, st, aniso_factor, run_parallel, n_workers, force=force)


if __name__ == "__main__":
    typer.run(main)
