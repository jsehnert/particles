#!/usr/bin/env python3
"""
Find physical particles with CONTRADICTORY ground-truth review labels: at
least one instance labeled a definitive TP (1/2/3) AND at least one instance
of the SAME physical particle labeled FP (0). A physical particle can only
really be one or the other, so this is a real labeling mistake somewhere --
exactly what prepare_training_data.py drops as "ambiguous" when building the
training set (empirically rare: ~4 of 85,859 clusters, as of the old
clusters.csv-based pipeline). This script surfaces those clusters -- rather
than letting them silently vanish from training -- so a human can go look and
fix the mislabel.

"Physical particle" is answered live via experiment_db.assign_components()
(shared grown voxels across the WHOLE candidate pool, every experiment, not
just any one slab_thickness window) -- there is no stored clusters.csv/
instance_to_cluster.csv anymore; link_candidates.py is retired. See
experiment_db's module docstring.

Output: Data/candidate_linking/tp_fp_contradictions.csv -- one row per
candidate instance belonging to a contradictory physical particle.
`component` numbers are only stable within one run of this script (scipy
connected-components ids, reassigned fresh each time) -- don't compare them
across runs; `candidate_id` is the stable identity.

Usage:
    uv run candidate_linking/find_tp_contradictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import experiment_db as db  # noqa: E402

TP_LABELS: frozenset[int] = frozenset({1, 2, 3})
FP_LABEL = 0

OUT_PATH = PROJECT_ROOT / "Data" / "candidate_linking" / "tp_fp_contradictions.csv"

DETAIL_COLUMNS = [
    "candidate_id",
    "component",
    "volume_name",
    "experiment_number",
    "peak_z",
    "peak_y",
    "peak_x",
    "particle",
    "n_voxels",
    "snr_peak",
    "fill_pca",
    "ml_prob",
    "ml_prob_particle_max",
    "slab_thickness",
    "low_threshold_scale",
]


def main() -> None:
    con = db.connect()
    full_df = db.load_candidates_dataframe(con)
    if full_df.empty:
        pd.DataFrame(columns=DETAIL_COLUMNS).to_csv(OUT_PATH, index=False)
        print("no candidates in the DB; wrote empty report")
        return

    full_df["component"] = db.assign_components(con, full_df)

    label_sets = full_df.groupby("component")["particle"].apply(lambda s: set(s.tolist()))
    is_contradiction = label_sets.apply(lambda labels: FP_LABEL in labels and bool(labels & TP_LABELS))
    contradictory_components = label_sets.index[is_contradiction]

    rows = full_df[full_df["component"].isin(contradictory_components)]
    rows = (
        rows.reset_index()[DETAIL_COLUMNS]
        .sort_values(["volume_name", "component", "experiment_number"])
    )
    rows.to_csv(OUT_PATH, index=False)
    print(f"true TP-vs-FP contradiction clusters: {len(contradictory_components)}")
    print(f"wrote detailed report: {OUT_PATH} ({len(rows)} rows)")

    con.close()


if __name__ == "__main__":
    main()
