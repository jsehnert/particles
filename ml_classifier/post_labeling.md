# After a labeling session

Reviewing candidates (`scripts/vol_data_review.py`) only changes the `particle`
column in `Data/experiments/combined_candidates.csv`. Nothing downstream picks
that up automatically -- clustering, the training table, the deployed model,
and the master file's own `ml_prob` column are all snapshots that go stale the
moment new labels land. Run the steps below, in order, to bring everything
back in sync.

Note what does **not** trigger this: running more sweep experiments. The
clustering step depends on voxel overlap (`combined_voxels.parquet`) and
candidate identity, not on labels -- so "new experiments, no new labels"
doesn't require step 1 (though it does mean `combined_candidates.csv` /
`combined_voxels.parquet` themselves changed, which does). What actually
triggers the need to re-run is new values in the `particle` column.

## 1. Relink candidates into physical particles

```bash
uv run python -m candidate_linking.link_candidates \
    --data-dir Data/experiments \
    --out Data/candidate_linking/clusters.csv \
    --instance-map-out Data/candidate_linking/instance_to_cluster.csv
```

Run from the repo root. **Always pass both `--out` and `--instance-map-out`
explicitly** -- `--out` otherwise defaults to `clusters.csv` in whatever
directory you happen to run the command from (not
`Data/candidate_linking/`), and `--instance-map-out` defaults to `None`,
meaning `instance_to_cluster.csv` silently isn't written at all. Both
defaults have already caused a real stale-data incident once.

Why this has to run first: `clusters.csv` records each physical particle's
`particle_labels` / `label_consistent` / `tp_status` as a snapshot taken at
generation time. Every step below depends on that snapshot being current.
Cheap to re-run (~10s) -- when in doubt, just run it.

## 2. Rebuild the training table

```bash
uv run python -m ml_classifier.prepare_training_data
```

Regenerates `ml_classifier/training_table.csv` from the fresh
`clusters.csv` / `instance_to_cluster.csv`. Restricted to slab_thickness
21/23/25/31 by design (see the script's docstring) -- 11/15 are never part
of training. (31 was added 2026-07-30 after a widened-scope CV experiment
showed it separates TP/FP better than 21/23/25, not worse.)

## 3. Retrain the classifier

```bash
uv run python -m ml_classifier.train_classifier
```

Fits a fresh `model.joblib` on the rebuilt training table and prints
cross-validated PR-AUC / calibration / permutation importance for the new
data. Depends on step 2's output.

## 4. Update the master file's scores

```bash
uv run python -m ml_classifier.update_combined_candidates_scores
```

Writes the new model's `ml_prob` / `ml_prob_particle_max` /
`cluster_root` / `cluster_ml_label` / `in_sample` back into
`Data/experiments/combined_candidates.csv` in place (backs up to
`combined_candidates.csv.bak` first, writes via a temp file + atomic
rename). Depends on step 1 (fresh clusters) and step 3 (fresh model).

**Do not run this while a labeling session is active.** It reads the whole
master file, scores it, then overwrites it -- any labels written by a
concurrent `vol_data_review.py` session during that window can be lost when
the overwrite lands. The backup is a recovery path, not a guard against
that race.

## 5. (Optional) Refresh the review triage queue

```bash
uv run python -m ml_classifier.score_backlog
```

Rescores the unreviewed backlog (slab_thickness 21/23/25/31, unreviewed
labels only) with the new model and rewrites `triage_queue.csv` --
worth doing if reviewers are working off that queue, so the next round of
labeling is prioritized by the latest model rather than a stale one.

## Summary

| Step | Script | Depends on |
|---|---|---|
| 1 | `candidate_linking.link_candidates` | new labels in `combined_candidates.csv` |
| 2 | `ml_classifier.prepare_training_data` | step 1 |
| 3 | `ml_classifier.train_classifier` | step 2 |
| 4 | `ml_classifier.update_combined_candidates_scores` | steps 1 and 3 |
| 5 (optional) | `ml_classifier.score_backlog` | step 3 |

Step 1 is cheap and safe to run any time labels might have changed. Steps
2-4 are heavier and touch the deployed model / master file -- run them
deliberately, not reflexively after every single review.
