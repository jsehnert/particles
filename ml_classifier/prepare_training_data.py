#!/usr/bin/env python3
"""
Build the training table for the TP/FP gradient-boosting classifier.

Design decisions this encodes:

  - Binary target only (particle in {1,2,3} -> TP, particle == 0 -> FP). The
    anode/cathode/interface sub-label has been applied "sloppily" and isn't
    trustworthy standalone; the TP/not-TP call itself is.

  - Training rows are restricted to slab_thickness in {21, 23, 25, 31} -- the
    window production is expected to run in. Two reasons: (1) slab_thickness
    changes the Stage-1 residual/background estimate, so a candidate's
    fill/SNR numbers aren't quite comparable across very different
    slab_thickness values; (2) production will run a SINGLE (slab_thickness,
    low_threshold_scale) per volume, so the model must be trained on raw,
    un-aggregated single-instance feature vectors -- not features averaged
    across a particle's multiple sweep detections -- to match what it will
    actually see at inference. Within this window there are 14 distinct
    (slab_thickness, low_threshold_scale) experiment configurations
    (low_threshold_scale 2.5-3.5), giving the model exposure to a spread of
    k_low without committing to whichever exact value production ends up
    choosing. 31 was added 2026-07-30 after a widened-scope CV experiment
    showed it separates TP/FP better than 21/23/25, not worse (out-of-fold
    AUC 0.990 vs 0.976-0.982) -- the smaller windows (11, 15) were not added,
    since the same experiment gave no reason to expect they'd help and
    including them was a deliberate scope decision, not an oversight.

  - The GROUND-TRUTH LABEL for a physical particle uses its full historical
    review across ALL experiments -- not just whatever was reviewed within
    the 21/23/25/31 subset -- realness is a property of the particle, not of
    which experiment a human happened to review it under. Physical particles
    are answered live via features.compute_ground_truth() (shared grown
    voxels, experiment_db.assign_components()) rather than a stored
    clusters.csv -- link_candidates.py is retired; see experiment_db's
    module docstring. Particles with contradictory labels (both a TP and a
    non-TP review) are dropped as ambiguous (empirically rare: 4 of 85,859,
    as of the old clusters.csv-based pipeline -- see
    candidate_linking/find_tp_contradictions.py to re-surface these).
    Particles with no informative review (only -1/10/unlabeled or
    4/needs-review) are dropped -- no usable label yet.

  - One TRAINING ROW per (physical particle, qualifying single instance in
    the 21/23/25/31 pool) -- not one aggregated row per particle. Per-particle
    sample weight = 1 / (number of that particle's qualifying instances in
    the pool), so a particle detected under many of the 14 configurations
    doesn't dominate the loss over one only ever caught once, while every
    instance still gets to contribute its own (unaggregated, production
    -realistic) feature vector.

  - Feature set: fill_pca (not fill -- shown to separate TP/FP in the
    expected direction while raw bbox fill doesn't), the existing shape/
    intensity descriptors, radial_pos (the normalized per-slab can-radius
    fraction measured from the fitted jelly-roll centre: 0 = axis, ~1 = can
    wall), and a new engineered dist_to_z_boundary feature (voxel distance to
    the nearer of the volume's z_min/z_max, from config.toml -- constant
    voxel_size across all 23 cells, so this is comparable unnormalized).
    Excluded: volume_name, experiment_number, every EXPERIMENT_PARAM_FIELDS
    column (detection *settings*, not particle properties), metal_threshold
    (per-volume calibration constant), and raw peak_z/y/x / centroid_z/y/x /
    z_min/z_max (volume-specific absolute coordinates that don't generalize
    across volumes with different z ranges). axis_y/axis_x were considered
    and rejected -- cell rotational orientation isn't controlled at scan
    time, so there's no stable lab-frame meaning to a "which lateral axis"
    split.

  - `volume_name` is retained in the output ONLY as the GroupKFold group key,
    not as a model feature. `component` is retained only as the physical-
    particle grouping id for diagnostics (e.g. nunique() counts) -- unlike
    the old `cluster_root`, it is a scipy-assigned integer stable only within
    one run of this script, not a durable identifier.

  - UNIT NORMALIZATION (voxel_size / cell size vary across cell formats):
    every M50L cell today is a 2170-format cell (21mm nominal diameter, 70mm
    nominal height) reconstructed at the same voxel_size (0.01640435mm), so
    raw voxel-unit features have been fine so far. But other formats (e.g.
    18650 at ~0.014mm voxels, 4860 at ~0.035mm voxels) will eventually enter
    the training pool with different voxel_size AND different can radius, so
    features that are fundamentally lengths/volumes in voxel units won't be
    physically comparable across formats unless corrected now (see
    features.add_unit_normalized_features for the exact conversions).

  - KNOWN RISK -- z_min/z_max are hand-picked per cell by visual inspection
    (z_min near the top of the jellyroll where all cathodes have appeared in
    cross-section despite not being uniformly aligned; z_max analogously at
    the bottom). Everything the pipeline relies on -- slab_thickness
    background estimation, and this feature -- assumes adjacent-slice
    similarity holds throughout [z_min, z_max], which breaks down outside it.
    In this curated training data, dist_to_z_boundary_mm carries real signal
    because that breakdown is real: FP rate spikes hard in the nearest 2mm to
    a boundary, and the feature is the model's 3rd-most-important by
    permutation importance (~0.033). But z_min/z_max are only as good as the
    human judgment that drew them -- production may select these bounds less
    precisely than the curated M50L training cells were. No fix is
    implemented for this yet (per team decision, 2026-07-26) -- flagged here
    to revisit once production's z_min/z_max workflow is decided.

  - SPIKED-CELL CONFOUND (radial_pos): cells M50L-16 through M50L-23 are
    believed to have had test particles deliberately added into the outer
    jellyroll layers. Empirically, TP radial_pos in these 8 cells is heavily
    compressed against the outer wall vs. a much wider, lower spread in the
    other 15 "natural" cells -- consistent with deliberate outer-layer
    placement rather than natural occurrence. Since radial_pos is kept as a
    feature (it also carries a genuine, generalizable signal: FP rate spikes
    at both radial extremes from jellyroll symmetry breakdown near the core
    and the can wall), training on the spiked cells' TP rows as-is would
    teach the model an experiment-design artifact ("high radial_pos -> TP")
    that will not hold in production (non-spiked) volumes. Fix: TP-labeled
    rows from the 8 spiked cells are excluded from training (this is exactly
    features.SPIKED_CELLS / the in_sample=False case computed by
    features.compute_ground_truth). FP-labeled rows from those same cells
    are kept -- a rejected candidate is still a legitimate negative example
    regardless of the spiking, and there's no reason to believe FP
    placement/character in those cells is atypical.

Usage:
    uv run ml_classifier/prepare_training_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import experiment_db as db  # noqa: E402

try:
    # Works when invoked as a module: `python -m ml_classifier.prepare_training_data`
    from .features import (
        FEATURE_COLUMNS,
        add_unit_normalized_features,
        compute_ground_truth,
        load_cell_meta,
    )
except ImportError:
    # Backward-compatible fallback for direct script execution.
    from features import (  # type: ignore
        FEATURE_COLUMNS,
        add_unit_normalized_features,
        compute_ground_truth,
        load_cell_meta,
    )

CONFIG_PATH = PROJECT_ROOT / "config.toml"
OUT_DIR = Path(__file__).resolve().parent

TRAINING_SLAB_THICKNESSES: tuple[int, ...] = (21, 23, 25, 31)


def main() -> None:
    cell_meta = load_cell_meta(CONFIG_PATH)
    con = db.connect()

    full_df = db.load_candidates_dataframe(con)
    print(f"total candidates in DB: {len(full_df)}")
    if full_df.empty:
        print("nothing to train on.")
        return

    full_df = compute_ground_truth(con, full_df)
    print(
        f"{full_df['component'].nunique()} distinct physical particles (full history); "
        f"{full_df['cluster_ml_label'].isna().sum()} candidates belong to an "
        "ambiguous/unreviewed particle (dropped regardless of slab_thickness below)"
    )

    window = full_df[
        full_df["slab_thickness"].isin(TRAINING_SLAB_THICKNESSES) & full_df["in_sample"]
    ].copy()
    print(
        f"training rows (slab_thickness {TRAINING_SLAB_THICKNESSES}, in-sample): {len(window)} "
        f"({(window['cluster_ml_label'] == 1).sum()} TP, {(window['cluster_ml_label'] == 0).sum()} FP)"
    )

    window["ml_label"] = window["cluster_ml_label"].astype(int)

    # unit normalization -- see module docstring "UNIT NORMALIZATION" (logic
    # lives in features.py, shared with score_candidates.py). A no-op
    # rescale on the current single-format/single-voxel_size dataset; needed
    # once other cell formats (different voxel_size, different can radius)
    # enter the training pool.
    window = add_unit_normalized_features(window, cell_meta)

    # per-particle sample weight: 1 / (qualifying instances for this particle
    # in this pool), so no particle's row count in the 21/23/25/31 window
    # dominates the loss over a particle only caught once.
    counts = window.groupby("component")["component"].transform("count")
    window["sample_weight"] = 1.0 / counts

    out_cols = ["component", "volume_name", *FEATURE_COLUMNS, "ml_label", "sample_weight"]
    out = window.reset_index()[out_cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "training_table.csv"
    out.to_csv(out_path, index=False)

    print(
        f"\nwrote {len(out)} rows, {out['component'].nunique()} unique particles -> {out_path}"
    )
    print("\nrows per volume:")
    print(out.groupby("volume_name").size().to_string())
    print("\nweight sums per volume (should equal unique-particle count per volume):")
    check = out.groupby("volume_name").agg(
        weight_sum=("sample_weight", "sum"), n_particles=("component", "nunique")
    )
    print(check.to_string())

    con.close()


if __name__ == "__main__":
    main()
