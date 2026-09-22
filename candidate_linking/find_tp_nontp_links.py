#!/usr/bin/env python3
"""
Find physical particles with a MIXED review history: at least one instance
labeled a definitive TP (1/2/3) and at least one instance of the SAME
physical particle carrying any other label (0/-1/4/10/11). This is the exact
"mixed_tp_nontp" category link_candidates.py used to compute (see its
retired tp_status(), replicated here) -- a superset of the hard TP-vs-FP
contradictions find_tp_contradictions.py reports on. Most of it is the softer
case: a particle confirmed TP under one detection but never given a
definitive call (still -1/4/10) under another, not necessarily an error --
just unfinished review.

"Physical particle" is answered live via experiment_db.assign_components()
(shared grown voxels across the WHOLE candidate pool, every experiment) --
there is no stored clusters.csv anymore; link_candidates.py is retired. See
experiment_db's module docstring.

Output: Data/candidate_linking/mixed_tp_nontp_drilldown.csv -- one row per
candidate instance belonging to a mixed physical particle, with that
particle's full label set repeated on every row for context. `component`
numbers are only stable within one run of this script (scipy
connected-components ids, reassigned fresh each time) -- don't compare them
across runs; `candidate_id` is the stable identity.

Usage:
    uv run candidate_linking/find_tp_nontp_links.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import experiment_db as db  # noqa: E402

TP_LABELS: frozenset[int] = frozenset({1, 2, 3})

OUT_PATH = PROJECT_ROOT / "Data" / "candidate_linking" / "mixed_tp_nontp_drilldown.csv"

DETAIL_COLUMNS = [
    "candidate_id",
    "component",
    "volume_name",
    "experiment_number",
    "peak_z",
    "peak_y",
    "peak_x",
    "particle",
    "particle_labels",
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
    is_mixed = label_sets.apply(lambda labels: bool(labels & TP_LABELS) and bool(labels - TP_LABELS))
    mixed_components = label_sets.index[is_mixed]

    full_df["particle_labels"] = full_df["component"].map(label_sets.apply(sorted))

    rows = full_df[full_df["component"].isin(mixed_components)]
    rows = (
        rows.reset_index()[DETAIL_COLUMNS]
        .sort_values(["volume_name", "component", "experiment_number"])
    )
    rows.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(rows)} rows to {OUT_PATH}")

    print("\nlabel counts:")
    print(rows["particle"].value_counts(dropna=False).sort_index())
    print("\nindeterminate counts:")
    print(rows[rows["particle"].isin([4])]["particle"].value_counts(dropna=False).sort_index())
    total_inspected = len(rows) + len(rows[rows["particle"].isin([4])])
    print(
        "Total Inspected:", total_inspected,
        total_inspected / len(rows) if len(rows) else float("nan"),
    )

    con.close()


if __name__ == "__main__":
    main()
