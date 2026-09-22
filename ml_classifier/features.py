#!/usr/bin/env python3
"""
Shared feature-engineering logic for the TP/FP classifier, used by BOTH
prepare_training_data.py (builds the labeled training table) and
score_backlog.py (scores unlabeled candidates for review triage).

Kept in one place deliberately: if training-time and inference-time feature
engineering ever drift apart (train/serve skew), the model silently degrades
in production without any error being raised. See prepare_training_data.py's
module docstring for the full design rationale (unit normalization,
radial_pos_frac, the spiked-cell confound, the z-boundary risk).

No imputation of any kind happens here. combined_candidates.csv has zero NaN
values across all 502,618 rows / every feature column (verified empirically,
2026-07-26) -- the degenerate-value code paths in extract_candidates_3d.py
(fill_pca, edge_contrast, decay_drop, gs_*) exist but never actually trigger
on this dataset. Even if they did, HistGradientBoostingClassifier natively
supports NaN (it learns a split direction for missing values during training,
verified empirically) -- so the right answer is to leave any future NaN as
-is, not impute it with 0 or a median. An explicit fillna(0) precedent exists
in the older exploratory notebook Notebooks/experiment_analysis.ipynb, but
that predates and is unrelated to this training pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Raw input columns that get unit-normalized below. NOTE: radial_pos is no
# longer a voxel-unit quantity -- detection now emits it already normalized to
# the per-slab fitted can radius (0 = axis, ~1 = wall). It is kept in this list
# only because it is a required input column; it is passed through unchanged as
# radial_pos_frac (no rescale). l1/l2/l3 (PCA eigenvalues of the grown
# component's voxel coordinates) are voxel^2 quantities -- same generalization
# concern as n_voxels/z_extent, just squared instead of linear/cubic.
RAW_VOXEL_UNIT_COLUMNS: list[str] = [
    "n_voxels",
    "z_extent",
    "y_extent",
    "x_extent",
    "radial_pos",
    "l1",
    "l2",
    "l3",
]
OTHER_FEATURE_COLUMNS: list[str] = [
    "n_seed",
    "n_seed_regions",
    "n_seed_total",
    "fill_pca",
    "snr_cluster",
    "r_peak",
    "r_peak_ratio",
    "peak_offset",
    # NOTE: peak_offset/seed_offset are raw voxel-unit distances, same
    # generalization concern as the RAW_VOXEL_UNIT_COLUMNS above -- but
    # peak_offset has always lived here unconverted, so seed_offset matches
    # that existing precedent rather than introducing an inconsistency
    # between two otherwise-identical distance features. Worth revisiting
    # both together if/when other cell formats actually enter the pool.
    "seed_offset",
    "snr_peak",
    "linearity",
    "planarity",
    "sphericity",
    "axis_z",
    "normal_z",
    "edge_contrast",
    "decay_drop",
    "seed_grown_ratio",
    "diag",
    "gs_median",
    "gs_p90",
    "gs_peak",
    "gs_shell_median",
    "gs_contrast",
    "gs_iqr",  # grayscale texture (p75-p25 over the grown mask) -- already
    # normalized upstream, same as the other gs_* level features, no
    # per-volume calibration correction needed here.
    "surface_ratio",  # boundary_voxels / n_voxels -- dimensionless (a ratio
    # of voxel counts), no unit conversion needed across cell formats.
]
# final feature set used for training/inference -- unit-normalized names in
# place of the raw voxel-unit columns they replace.
FEATURE_COLUMNS: list[str] = OTHER_FEATURE_COLUMNS + [
    "volume_mm3",  # replaces n_voxels
    "z_extent_mm",  # replaces z_extent
    "y_extent_mm",  # replaces y_extent
    "x_extent_mm",  # replaces x_extent
    "radial_pos_frac",  # pass-through alias of the already-normalized radial_pos
    "dist_to_z_boundary_mm",  # engineered, replaces voxel-unit version
    "l1_mm2",  # replaces l1 -- see add_unit_normalized_features
    "l2_mm2",  # replaces l2
    "l3_mm2",  # replaces l3
]

# columns that must be selected from combined_candidates.csv for
# add_unit_normalized_features() to work (raw feature columns + the peak
# coordinate needed for dist_to_z_boundary).
REQUIRED_RAW_COLUMNS: list[str] = (
    OTHER_FEATURE_COLUMNS + RAW_VOXEL_UNIT_COLUMNS + ["peak_z"]
)

# --- ground truth -- shared between score_candidates.py and
# prepare_training_data.py for the exact same train/serve-skew reason this
# module exists at all (see module docstring): if the two ever computed "is
# this physical particle a TP" differently, that would corrupt what the
# model is trained against and what it's scored/reported against, which is
# worse than a feature-engineering mismatch. ---

TP_LABELS: frozenset[int] = frozenset({1, 2, 3})
FP_LABEL = 0
# Cells believed to have had test particles deliberately spiked into the
# outer jellyroll layers (see prepare_training_data.py's original module
# docstring, "SPIKED-CELL CONFOUND"). TP rows from these cells are excluded
# from in_sample; FP rows are kept -- a rejected candidate is still a
# legitimate negative example regardless of the spiking.
SPIKED_CELLS: frozenset[str] = frozenset(f"M50L-{n:02d}" for n in range(16, 24))


def ground_truth_label(particle_labels: set[int]) -> int | None:
    """Binary TP/FP label for a physical particle from its full historical
    review (labels 1/2/3 grouped as TP). None if ambiguous (both a TP and a
    non-TP review present) or uninformative (only -1/10/4)."""
    has_tp = bool(particle_labels & TP_LABELS)
    has_fp = FP_LABEL in particle_labels
    if has_tp and has_fp:
        return None
    if has_tp:
        return 1
    if has_fp:
        return 0
    return None


def compute_ground_truth(con, full_df: pd.DataFrame) -> pd.DataFrame:
    """Add `component`, `cluster_ml_label`, `in_sample` columns to `full_df`
    (an experiment_db.load_candidates_dataframe() result; mutates and returns
    it).

    Ground truth spans a physical particle's FULL historical review across
    every experiment, regardless of slab_thickness -- realness is a property
    of the particle, not of which experiment happened to review it.
    `component` groups candidates into physical particles via shared grown
    voxels (experiment_db.assign_components()) -- ids are only stable within
    this one call, never persist or compare them across calls.

    `con` is a duckdb connection (see experiment_db.connect()). experiment_db
    is imported lazily here, not at module level, so importing this module
    for just its column-name constants (e.g. from a notebook) never pulls in
    duckdb/scipy at all.
    """
    import experiment_db as db

    full_df["component"] = db.assign_components(con, full_df)
    label_by_component = full_df.groupby("component")["particle"].apply(
        lambda labels: ground_truth_label(set(labels.tolist()))
    )
    full_df["cluster_ml_label"] = full_df["component"].map(label_by_component)
    spiked_tp = full_df["volume_name"].isin(SPIKED_CELLS) & (full_df["cluster_ml_label"] == 1)
    full_df["in_sample"] = full_df["cluster_ml_label"].notna() & ~spiked_tp
    return full_df


def load_cell_meta(config_path: Path) -> dict[str, dict]:
    """Per-cell z-bounds + physical-unit conversion constants (voxel_size,
    nominal outer radius). See prepare_training_data.py "UNIT NORMALIZATION"."""
    sys.path.insert(0, str(config_path.parent))
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    return {
        c["name"]: {
            "z_min": int(c["z_min"]),
            "z_max": int(c["z_max"]),
            "voxel_size_mm": float(c["voxel_size"]),
            "nominal_radius_mm": float(c["nominal_diameter_mm"]) / 2.0,
        }
        for c in cfg["cell"]
    }


def add_unit_normalized_features(df: pd.DataFrame, cell_meta: dict[str, dict]) -> pd.DataFrame:
    """Add volume_mm3 / z_extent_mm / y_extent_mm / x_extent_mm /
    radial_pos_frac / dist_to_z_boundary_mm / l1_mm2 / l2_mm2 / l3_mm2 columns
    (mutates and returns df). Requires df to have REQUIRED_RAW_COLUMNS plus a
    "volume_name" column."""
    voxel_size = df["volume_name"].map(lambda v: cell_meta[v]["voxel_size_mm"])
    z_min = df["volume_name"].map(lambda v: cell_meta[v]["z_min"])
    z_max = df["volume_name"].map(lambda v: cell_meta[v]["z_max"])

    df["volume_mm3"] = df["n_voxels"] * voxel_size**3
    df["z_extent_mm"] = df["z_extent"] * voxel_size
    df["y_extent_mm"] = df["y_extent"] * voxel_size
    df["x_extent_mm"] = df["x_extent"] * voxel_size
    # radial_pos is now emitted already normalized as a fraction of the per-slab
    # fitted can radius (0 = axis, ~1 = wall, unclamped; see
    # extract_candidates_3d._process_slab), so it is used directly here.
    # Previously it was a raw voxel distance from the volume centre and was
    # converted with (radial_pos * voxel_size) / nominal_radius; that conversion
    # would now double-normalize.
    df["radial_pos_frac"] = df["radial_pos"]

    dist_to_z_boundary_vox = pd.concat(
        [df["peak_z"] - z_min, z_max - df["peak_z"]], axis=1
    ).min(axis=1)
    df["dist_to_z_boundary_mm"] = dist_to_z_boundary_vox * voxel_size

    # l1/l2/l3 are PCA eigenvalues of voxel COORDINATES -- variance, i.e.
    # voxel^2 units (unlike radial_pos, these were never emitted pre
    # -normalized) -- squared, not linear/cubed like the extent/volume
    # conversions above.
    voxel_size_sq = voxel_size**2
    df["l1_mm2"] = df["l1"] * voxel_size_sq
    df["l2_mm2"] = df["l2"] * voxel_size_sq
    df["l3_mm2"] = df["l3"] * voxel_size_sq
    return df
