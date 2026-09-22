#!/usr/bin/env python3
"""
Standalone pilot: rerun candidate detection with a relaxed min_fill gate
against a small, targeted subset of (slab_thickness, cell) combinations,
without touching vol_analysis.py or config.toml.

Why this exists
----------------
Every real entry point (`DetectParams`, `detect_candidates_streaming`,
`detect_candidates_parallel`) defaults `min_fill` to 0.15; `_axial_ok`'s own
internal default of 0.1 is dead code in practice because `_process_slab`
always passes `min_fill=p.min_fill` explicitly. `vol_analysis.py::_run_one`
never overrides `min_fill` either, so the existing 20-experiment sweep in
config.toml ran at 0.15. Checking the TP-labeled (particle in (1,2,3))
population against that boundary showed a meaningful number of confirmed
true particles sitting right at it (fill 0.15-0.20), which suggests real
particles are being rejected by the gate just below it. This script tests
min_fill=0.10 on the (slab_thickness, cell) combinations where that effect
should be most visible, writing to its own output tree so pilot results
never mix with the production Data/experiments/<experiment_number>/ set.

Cell/slab_thickness selection (see chat discussion for the full analysis):
  - slab_thickness=31 showed the highest share of candidates sitting in the
    fill 0.15-0.20 band (0.65%, vs 0.20% at slab_thickness=11), so it's
    where loosening min_fill should have the most visible effect per volume
    processed.
  - M50L-22, M50L-21, M50L-20 have the most already-confirmed TP candidates
    sitting within 0.25 fill (123 / 86 / 68 respectively), giving the best
    odds of the experiment being informative against real ground truth.
These are just the defaults below -- override with --slab-thickness / --cell.

Schema: candidates.csv/voxels.csv rows carry the same experiment_number +
EXPERIMENT_PARAM_FIELDS columns (baked in at write time) as the production
Data/experiments/<experiment_number>/ output -- experiment_number is the real
production experiment_number for the (slab_thickness, low_threshold_scale)
combo being rerun here (just with a different min_fill), so pilot and
production rows are directly concatenable without a join against params.csv.

Efficiency: stage-1 residual precomputation depends only on slab_thickness +
baseline_method, not on low_threshold_scale or min_fill. The production
driver reconstructs GlobalData and reruns precompute_residual once per
experiment_number even when several experiment_numbers share a
slab_thickness. This script instead precomputes the residual ONCE per
(cell, slab_thickness) and reuses it across every low_threshold_scale
already defined for that family in config.toml, at the new min_fill -- so
stage 1 is paid once per cell here, not once per (cell, k_low) pair.

Usage:
    python scripts/min_fill_pilot.py \
        --slab-thickness 31 \
        --cell M50L-22 --cell M50L-21 --cell M50L-20 \
        --min-fill 0.10

Reruns are safe by default: for each (slab_thickness, low_threshold_scale)
combo, a cell already present in that combo's candidates.csv is skipped
rather than re-appended (would duplicate rows) or overwritten (would destroy
whatever other cells' data is already in that file). So if you already ran
--cell M50L-22 and now want to add the other two, just rerun with all three
--cell flags (or any --cell list you like) -- M50L-22 will be skipped, and
only M50L-21 / M50L-20 will actually be processed. Pass --force if you
specifically want to redo a cell that's already there (its old rows are
removed first, then replaced -- never duplicated).
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
    GlobalData,
    _experiment_param_values,
    _load_raw_config,
    write_candidate_rows,
)

DEFAULT_CELLS: list[str] = ["M50L-22", "M50L-21", "M50L-20"]
DEFAULT_SLAB_THICKNESSES: list[int] = [31]
OUT_ROOT: Path = PROJECT_ROOT / "Data" / "experiments_min_fill_pilot"
TMP_ROOT: Path = PROJECT_ROOT / "Data" / "_tmp_min_fill_pilot"


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
    out_path: Path, experiment: dict, min_fill: float, low_threshold_scale: float
) -> None:
    """Same convention as vol_analysis.write_experiment_params_csv, plus the
    min_fill column the production schema never captured. `experiment` must be
    this specific combo's own [[experiment]] entry (not the sweep's basis
    entry) so experiment_number and low_threshold_scale are both correct for
    this output directory."""
    row: dict[str, object] = {}
    for key, value in experiment.items():
        if isinstance(value, list):
            value = ";".join(str(v) for v in value)
        row[key] = value
    row["low_threshold_scale"] = low_threshold_scale
    row["min_fill"] = min_fill
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
    min_fill: float,
    run_parallel: bool,
    n_workers_override: int | None,
    force: bool = False,
) -> None:
    """Precompute the residual once for (cell_name, slab_thickness), then run
    stage-2 detection at the new min_fill for every low_threshold_scale value
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
        return OUT_ROOT / f"slab{slab_thickness}_low{low}_minfill{min_fill}"

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
        f"low_threshold_scale sweep {low_values}, min_fill={min_fill} ==="
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
    aniso_factor = float(global_data.aniso_factor)
    z_pad = int(global_data.z_pad)
    metal_threshold = float(global_data.metal_threshold)
    z_offset = int(global_data.z_min)
    slices_per_chunk = int(global_data.slices_per_chunk)
    n_workers = int(n_workers_override or global_data.n_workers)

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
            # min_fill.
            experiment_number = int(experiment["experiment_number"])
            experiment_params = _experiment_param_values(experiment, min_fill)
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
            _write_params_csv(out_dir / "params.csv", experiment, min_fill, low)

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
                        min_fill=min_fill,
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
                        min_fill=min_fill,
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
        DEFAULT_CELLS, "--cell", help="cell name(s) to test, e.g. M50L-22"
    ),
    min_fill: float = typer.Option(0.10, "--min-fill", help="relaxed min_fill to test"),
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
            run_cell_slab(cfg, c, st, min_fill, run_parallel, n_workers, force=force)


if __name__ == "__main__":
    typer.run(main)
