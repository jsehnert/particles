#!/usr/bin/env python3
"""
Score candidates with the trained TP/FP classifier (model.joblib, from
train_classifier.py) and store the results in the DuckDB `scores` table.

Replaces the former score_backlog.py + update_combined_candidates_scores.py
split. That split existed only because both scripts wrote to the same
Data/experiments/combined_candidates.csv that scripts/vol_data_review.py was
also rewriting wholesale on every label -- update_combined_candidates_scores.py
overwrote the live master in place (a real corruption risk if a labeling
session was running concurrently, hence its "run manually, infrequently"
warning), while score_backlog.py existed alongside it specifically so
backlog-triage scoring could happen without touching the master at all.
DuckDB doesn't have that hazard: vol_data_review.py does small `UPDATE ...
WHERE candidate_id = ?` statements, not full-table rewrites, so there's no
reason to keep two scripts computing the same thing for two different
destinations. This one always (re)writes the DB.

Scope: every candidate with slab_thickness in SCORING_SLAB_THICKNESSES gets a
fresh `scores` row -- TP/FP/backlog alike, matching
update_combined_candidates_scores.py's superset behavior (score_backlog.py's
narrower "only unreviewed" scope is now just a filter on the output, see
"triage_queue" below). The model has never seen other slab_thicknesses (see
features.py / prepare_training_data.py) so they're left unscored.

Ground truth (cluster_ml_label / in_sample) uses each physical particle's
FULL historical review across every experiment, not just the scoring window
-- realness is a property of the particle, not of which experiment happened
to review it (see prepare_training_data.py's SPIKED-CELL CONFOUND and
ground-truth sections, replicated here). "Physical particle" is answered live
via experiment_db.assign_components() over EVERY candidate for the volumes
involved, not a stored clusters.csv/instance_to_cluster.csv --
link_candidates.py is retired; see experiment_db's module docstring.

Also writes ml_classifier/triage_queue.csv: one row per physical particle
with no informative review anywhere in its history, ranked by
ml_prob_particle_max descending -- the human-facing review queue. This is a
disposable report snapshot (component ids from connected_components aren't
stable across runs), not a source of truth -- re-run this script to refresh
it. The old triage queue could resurface a particle that had actually already
been reviewed under an experiment outside the scoring window; this one can't,
since the "unreviewed" filter now uses the same full-history label used for
cluster_ml_label.

Usage:
    uv run ml_classifier/score_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import experiment_db as db  # noqa: E402

try:
    # Works when invoked as a module: `python -m ml_classifier.score_candidates`
    from .features import add_unit_normalized_features, compute_ground_truth, load_cell_meta
except ImportError:
    # Backward-compatible fallback for direct script execution.
    from features import (  # type: ignore
        add_unit_normalized_features,
        compute_ground_truth,
        load_cell_meta,
    )

CONFIG_PATH = PROJECT_ROOT / "config.toml"
OUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUT_DIR / "model.joblib"
QUEUE_PATH = OUT_DIR / "triage_queue.csv"

SCORING_SLAB_THICKNESSES: tuple[int, ...] = (21, 23, 25, 31)


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    print(f"loaded model: {len(feature_columns)} features")

    cell_meta = load_cell_meta(CONFIG_PATH)
    con = db.connect()

    full_df = db.load_candidates_dataframe(con)
    print(f"total candidates in DB: {len(full_df)}")
    if full_df.empty:
        print("nothing to score.")
        return

    # Ground truth from each particle's FULL historical review, not just the
    # scoring window -- see features.compute_ground_truth / module docstring.
    full_df = compute_ground_truth(con, full_df)
    print(f"{full_df['component'].nunique()} distinct physical particles (full history)")
    label_by_component = full_df.groupby("component")["cluster_ml_label"].first()

    window = full_df[full_df["slab_thickness"].isin(SCORING_SLAB_THICKNESSES)].copy()
    print(f"scoring window (slab_thickness {SCORING_SLAB_THICKNESSES}): {len(window)} candidates")
    if window.empty:
        print("nothing in the scoring window.")
        return

    window = add_unit_normalized_features(window, cell_meta)
    X = window[feature_columns].to_numpy()
    window["ml_prob"] = model.predict_proba(X)[:, 1]

    # per-particle max (within the scoring window only -- that's the only
    # place ml_prob exists) + which instance produced it.
    per_component_max = window.groupby("component")["ml_prob"].max()
    per_component_source = window.groupby("component")["ml_prob"].idxmax()
    window["ml_prob_particle_max"] = window["component"].map(per_component_max)
    window["ml_prob_max_source_candidate_id"] = window["component"].map(per_component_source)

    print("\nml_prob_particle_max distribution (one value per particle):")
    print(per_component_max.describe().to_string())
    print(
        f"\nin_sample: {int(window['in_sample'].sum())} rows "
        f"({int((~window['in_sample']).sum())} genuinely held-out)"
    )

    scores_df = window.reset_index()[
        [
            "candidate_id",
            "ml_prob",
            "ml_prob_particle_max",
            "ml_prob_max_source_candidate_id",
            "cluster_ml_label",
            "in_sample",
        ]
    ]
    db.replace_scores(con, scores_df)
    print(f"\nwrote {len(scores_df)} rows to the scores table")

    # --- triage_queue.csv: physical particles with no informative review
    # anywhere in their history, ranked by ml_prob_particle_max ---
    n_qualifying = window.groupby("component").size().rename("n_qualifying_instances")
    queue = pd.DataFrame(
        {
            "ml_prob_particle_max": per_component_max,
            "ml_prob_max_source_candidate_id": per_component_source,
        }
    ).join(n_qualifying)
    queue = queue[label_by_component.reindex(queue.index).isna()]
    queue = queue.join(
        full_df[["volume_name", "experiment_number", "peak_z", "peak_y", "peak_x"]],
        on="ml_prob_max_source_candidate_id",
    )
    queue = queue.sort_values("ml_prob_particle_max", ascending=False)
    queue.to_csv(QUEUE_PATH, index=False)
    print(f"wrote {len(queue)}-row triage queue -> {QUEUE_PATH}")

    con.close()


if __name__ == "__main__":
    main()
