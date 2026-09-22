#!/usr/bin/env python3
"""
Train the TP/FP gradient-boosting classifier on ml_classifier/
training_table.csv (see prepare_training_data.py for how that table is built).

Evaluation strategy (see chat discussion for the full reasoning):
  - StratifiedGroupKFold by volume_name, 5 folds. Grouping by volume is free
    leakage protection (a physical particle lives in exactly one volume, so
    grouping by volume can't split a particle across train/test) and the
    honest generalization test (does this work on a cell it's never seen).
  - All metrics use the per-particle sample_weight from prepare_training_data
    .py, so a particle detected under many of the (slab_thickness,
    low_threshold_scale) configurations in the pool doesn't get counted
    multiple times over one only ever caught once.
  - Reported metrics emphasize ranking quality (PR-AUC) and calibration over
    a single threshold's accuracy, since this model's initial role is Phase 1
    review-triage (ranking/prioritizing the unlabeled backlog), not an
    autonomous filter yet -- see chat discussion. Precision/recall at a
    default 0.5 threshold and a couple of alternates are reported for
    reference only.
  - Feature importance is permutation importance computed on each fold's
    HELD-OUT test split (not the training data), averaged across folds --
    avoids the optimistic bias of evaluating importance on data the model was
    fit on.
  - A final model, fit on the FULL training table, is saved for actually
    scoring the unlabeled backlog. Its own training-set metrics are not a
    generalization estimate; the CV numbers above are.

Calibration (added 2026-07-30): every model here is wrapped in
CalibratedClassifierCV(method="sigmoid") -- see CALIBRATION_METHOD below for
why sigmoid specifically, and why it's applied this way rather than as a
separate manual step.

Model: sklearn HistGradientBoostingClassifier -- gradient boosting, and the
only boosting implementation currently available in this project's
dependencies (only scikit-learn is declared; no xgboost/lightgbm/catboost).

Usage:
    python ml_classifier/train_classifier.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedGroupKFold

OUT_DIR = Path(__file__).resolve().parent
TABLE_PATH = OUT_DIR / "training_table.csv"
MODEL_PATH = OUT_DIR / "model.joblib"

NON_FEATURE_COLUMNS = {"component", "volume_name", "ml_label", "sample_weight"}
N_FOLDS = 5
RANDOM_STATE = 0

# Raw HistGradientBoostingClassifier probabilities are miscalibrated:
# underconfident around 0.1-0.4, overconfident around 0.6-0.9 (see calibration
# table below). Compared uncalibrated / isotonic / sigmoid head-to-head (chat
# discussion, 2026-07-30): sigmoid won on both PR-AUC (0.943 vs 0.936
# uncalibrated) and Brier (0.043 vs 0.047) -- isotonic improved Brier too but
# cost real PR-AUC (0.925), overfitting the sparse middle-probability bins.
CALIBRATION_METHOD = "sigmoid"

# CalibratedClassifierCV's inner cv: plain integer (ordinary StratifiedKFold),
# not a grouped splitter -- CalibratedClassifierCV.fit() doesn't route a
# `groups` array through to it (confirmed empirically: passing a
# StratifiedGroupKFold here fails with "number of splits greater than number
# of groups: 1"). That's fine: this inner split's only job is keeping the
# calibrator off the exact rows the base model trained on -- not testing
# cross-volume generalization, which is what the OUTER StratifiedGroupKFold
# (below, grouped by volume_name) is for. Every reported metric stays fully
# volume-held-out regardless.
INNER_CALIBRATION_FOLDS = 3


def make_model() -> CalibratedClassifierCV:
    """The one place the model is constructed -- used identically for every
    CV fold and for the final deployed model, so what gets evaluated is
    exactly what gets deployed."""
    base = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    return CalibratedClassifierCV(base, method=CALIBRATION_METHOD, cv=INNER_CALIBRATION_FOLDS)


def _weighted_calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, weight: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    bins = pd.cut(y_prob, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    df = pd.DataFrame({"bin": bins, "y": y_true, "p": y_prob, "w": weight})
    rows = []
    for b, g in df.groupby("bin", observed=True):
        if g["w"].sum() == 0:
            continue
        rows.append(
            {
                "bin": str(b),
                "n_rows": len(g),
                "weighted_n": g["w"].sum(),
                "mean_predicted": np.average(g["p"], weights=g["w"]),
                "actual_tp_rate": np.average(g["y"], weights=g["w"]),
            }
        )
    return pd.DataFrame(rows)


def _precision_recall_at_thresholds(
    y_true, y_prob, weight, thresholds=(0.3, 0.5, 0.7)
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = np.sum(weight * ((pred == 1) & (y_true == 1)))
        fp = np.sum(weight * ((pred == 1) & (y_true == 0)))
        fn = np.sum(weight * ((pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else float("nan")
        )
        rows.append(
            {"threshold": t, "precision": precision, "recall": recall, "f1": f1}
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(TABLE_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].to_numpy()
    y = df["ml_label"].to_numpy()
    w = df["sample_weight"].to_numpy()
    groups = df["volume_name"].to_numpy()

    print(
        f"training table: {len(df)} rows, {df['component'].nunique()} unique particles, "
        f"{len(feature_cols)} features, {len(set(groups))} volumes (CV groups)"
    )

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_prob = np.zeros(len(df))
    importances_per_fold: list[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        held_out_volumes = sorted(set(groups[test_idx]))
        model = make_model()
        model.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        oof_prob[test_idx] = prob

        fold_ap = average_precision_score(y[test_idx], prob, sample_weight=w[test_idx])
        print(
            f"fold {fold}: held out {len(held_out_volumes)} volumes {held_out_volumes}, "
            f"{len(test_idx)} rows, PR-AUC={fold_ap:.3f}"
        )

        perm = permutation_importance(
            model,
            X[test_idx],
            y[test_idx],
            sample_weight=w[test_idx],
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="average_precision",
        )
        importances_per_fold.append(perm.importances_mean)

    print("\n=== overall (out-of-fold) evaluation ===")
    overall_ap = average_precision_score(y, oof_prob, sample_weight=w)
    overall_brier = brier_score_loss(y, oof_prob, sample_weight=w)
    print(f"PR-AUC (weighted): {overall_ap:.3f}")
    print(f"Brier score (weighted, lower is better): {overall_brier:.3f}")

    print("\n=== calibration (out-of-fold predictions, weighted) ===")
    calib = _weighted_calibration_table(y, oof_prob, w)
    print(calib.to_string(index=False))

    print("\n=== precision/recall at reference thresholds (out-of-fold, weighted) ===")
    pr_table = _precision_recall_at_thresholds(y, oof_prob, w)
    print(pr_table.to_string(index=False))

    print(
        "\n=== permutation importance (avg PR-AUC drop across folds, held-out data) ==="
    )
    avg_importance = np.mean(importances_per_fold, axis=0)
    imp_df = pd.DataFrame(
        {"feature": feature_cols, "importance": avg_importance}
    ).sort_values("importance", ascending=False)
    print(imp_df.to_string(index=False))

    # final deployment model: same construction as every CV fold above (see
    # make_model), fit on ALL labeled data. Its own training-set metrics are
    # not a generalization estimate; the out-of-fold numbers above are.
    final_model = make_model()
    final_model.fit(X, y, sample_weight=w)
    joblib.dump({"model": final_model, "feature_columns": feature_cols}, MODEL_PATH)
    print(f"\nfinal model (fit on all {len(df)} rows, {CALIBRATION_METHOD}-calibrated) saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
