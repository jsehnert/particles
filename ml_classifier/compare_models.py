#!/usr/bin/env python3
"""
Fair head-to-head: the deployed HistGradientBoosting classifier vs a PyTorch
Lightning MLP, on the SAME training table, folds, weights, and metrics.

Fairness contract (why each choice is made):
  - Identical CV: StratifiedGroupKFold(n_splits, shuffle, random_state) grouped
    by volume_name -- reused from train_classifier so both models see the exact
    same train/test partitions and neither can leak a particle across folds.
  - Identical features / target / per-particle sample_weight.
  - Same objective family: the MLP minimizes sample-weighted BCE-with-logits =
    weighted log loss, matching what the HGB optimizes. pos_weight defaults to
    1.0 so, like the HGB (which has no class_weight), the TP/FP imbalance is NOT
    rebalanced. Pass --pos-weight to deviate (e.g. ~3.4 fully balances the
    current table) as a separate curiosity.
  - Same held-out evaluation: out-of-fold weighted PR-AUC (average precision),
    Brier, calibration, and permutation importance -- the identical helpers used
    for the HGB, computed on raw (unscaled) features for both so the importance
    comparison is apples-to-apples.
  - Symmetric internal model selection: the HGB uses early_stopping='auto'
    (10% internal val, best iteration by loss). The MLP mirrors this -- a 10%
    internal val split of each train fold, best epoch restored by weighted val
    loss -- so both lose the same slice to selection and pick their iteration
    count the same way. (The OneCycle schedule always runs the full max_epochs;
    only the best snapshot is kept.)

MLP-only, legitimate (not leakage): a per-fold StandardScaler fit on the fold's
training portion only, applied to that fold's internal-val and outer-test rows.
MLPs are not scale-invariant the way the trees are.

Architecture (per spec): Linear(n_features, 256) -> LayerNorm -> GELU ->
Dropout(0.1) -> Linear(256, 1); AdamW(lr=1e-3, weight_decay=1e-5); OneCycleLR
(max_lr=1e-3, pct_start=0.3, cosine); batch 32; 250 epochs.

This script does NOT touch model.joblib or the deployed HGB -- it only prints a
comparison. Deep-learning deps are required: `uv add torch lightning`.

Usage:
    python ml_classifier/compare_models.py [--pos-weight 1.0] [--batch-size 32]
        [--max-epochs 250] [--accelerator cpu] [--seed 0]
"""

from __future__ import annotations

import argparse
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:  # unified `lightning` package (uv add lightning)
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:  # fall back to the older distribution name
    import pytorch_lightning as L  # type: ignore
    from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore

# Reuse the HGB harness's constants + metric helpers so the two models are
# scored by identical code.
try:
    # Works when invoked as a module: `python -m ml_classifier.compare_models`
    from .train_classifier import (
        N_FOLDS,
        NON_FEATURE_COLUMNS,
        RANDOM_STATE,
        TABLE_PATH,
        _precision_recall_at_thresholds,
        _weighted_calibration_table,
    )
except ImportError:
    # Backward-compatible fallback for direct script execution.
    from train_classifier import (  # type: ignore
        N_FOLDS,
        NON_FEATURE_COLUMNS,
        RANDOM_STATE,
        TABLE_PATH,
        _precision_recall_at_thresholds,
        _weighted_calibration_table,
    )


class MLP(L.LightningModule):
    """Single-hidden-layer MLP with weighted BCE, AdamW, and OneCycle (cosine)."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        pct_start: float = 0.3,
        max_epochs: int = 250,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits

    def _weighted_bce(self, batch) -> torch.Tensor:
        x, y, w = batch
        logits = self(x)
        pw = torch.as_tensor(
            self.hparams.pos_weight, dtype=logits.dtype, device=logits.device
        )
        per = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pw, reduction="none"
        )
        return (per * w).sum() / w.sum()  # sample-weighted mean == weighted log loss

    def training_step(self, batch, _):
        loss = self._weighted_bce(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        self.log("val_loss", self._weighted_bce(batch), prog_bar=False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # exact number of optimizer steps Lightning will take (epochs x batches),
        # so OneCycle's total_steps can't over/under-shoot and crash mid-run.
        total_steps = int(self.trainer.estimated_stepping_batches)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.lr,
            total_steps=total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy="cos",
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


class TorchProbaAdapter(ClassifierMixin, BaseEstimator):
    """sklearn-compatible predict_proba over (scaler -> trained net), so the
    SAME permutation_importance path (which permutes RAW features) works for the
    MLP exactly as for the HGB."""

    def __init__(self, model: MLP, scaler: StandardScaler) -> None:
        self.model = model.eval().cpu()
        self.scaler = scaler
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):  # already trained; permutation_importance needs the API
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(Xs))).numpy()
        return np.column_stack([1.0 - p, p])


def _loader(X, y, w, batch_size, shuffle, seed):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def train_mlp_fold(X_tr, y_tr, w_tr, args, n_features, fold):
    """Fit one MLP on a train fold with an internal 10% val split; return the
    best-epoch model (restored by weighted val loss)."""
    xt, xv, yt, yv, wt, wv = train_test_split(
        X_tr, y_tr, w_tr, test_size=args.val_frac, random_state=args.seed, stratify=y_tr
    )

    L.seed_everything(args.seed + fold, workers=True)
    model = MLP(
        n_features=n_features,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        pct_start=args.pct_start,
        max_epochs=args.max_epochs,
    )
    with tempfile.TemporaryDirectory() as ckpt_dir:
        ckpt = ModelCheckpoint(
            dirpath=ckpt_dir, monitor="val_loss", mode="min", save_top_k=1
        )
        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=1,
            deterministic=True,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[ckpt],
            num_sanity_val_steps=0,
        )
        trainer.fit(
            model,
            _loader(xt, yt, wt, args.batch_size, True, args.seed + fold),
            _loader(
                xv, yv, wv, len(xv), False, args.seed
            ),  # full-batch val = exact weighted val loss
        )
        best = (
            MLP.load_from_checkpoint(ckpt.best_model_path)
            if ckpt.best_model_path
            else model
        )
    return best.eval()


def evaluate(name, oof_prob, y, w) -> None:
    print(f"\n=== {name}: out-of-fold evaluation (weighted) ===")
    print(f"PR-AUC:  {average_precision_score(y, oof_prob, sample_weight=w):.4f}")
    print(f"Brier:   {brier_score_loss(y, oof_prob, sample_weight=w):.4f}")
    print("calibration:")
    print(_weighted_calibration_table(y, oof_prob, w).to_string(index=False))
    print("precision/recall at thresholds:")
    print(_precision_recall_at_thresholds(y, oof_prob, w).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help="BCE positive-class weight (1.0 = match the HGB / no rebalancing)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=250)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--pct-start", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--accelerator", default="auto", help="cpu | mps | gpu | auto")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    df = pd.read_csv(TABLE_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["ml_label"].to_numpy()
    w = df["sample_weight"].to_numpy()
    groups = df["volume_name"].to_numpy()
    assert not np.isnan(X).any(), "MLP path requires NaN-free features (found NaN)"
    print(
        f"training table: {len(df)} rows, {len(feature_cols)} features, "
        f"{len(set(groups))} volumes; pos_weight={args.pos_weight}, batch={args.batch_size}, "
        f"epochs={args.max_epochs}, accelerator={args.accelerator}"
    )

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_hgb = np.zeros(len(df))
    oof_mlp = np.zeros(len(df))
    imp_hgb: list[np.ndarray] = []
    imp_mlp: list[np.ndarray] = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups)):
        held = sorted(set(groups[te]))
        # ---- HGB (identical to train_classifier's per-fold fit) ----
        hgb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        hgb.fit(X[tr], y[tr], sample_weight=w[tr])
        oof_hgb[te] = hgb.predict_proba(X[te])[:, 1]
        imp_hgb.append(
            permutation_importance(
                hgb,
                X[te],
                y[te],
                sample_weight=w[te],
                n_repeats=10,
                random_state=RANDOM_STATE,
                scoring="average_precision",
            ).importances_mean
        )

        # ---- MLP (per-fold scaler fit on train portion only) ----
        scaler = StandardScaler().fit(X[tr])
        model = train_mlp_fold(
            scaler.transform(X[tr]), y[tr], w[tr], args, len(feature_cols), fold
        )
        adapter = TorchProbaAdapter(model, scaler)
        oof_mlp[te] = adapter.predict_proba(X[te])[:, 1]
        imp_mlp.append(
            permutation_importance(
                adapter,
                X[te],
                y[te],
                sample_weight=w[te],
                n_repeats=10,
                random_state=RANDOM_STATE,
                scoring="average_precision",
            ).importances_mean
        )

        print(
            f"fold {fold}: held out {held}  "
            f"HGB PR-AUC={average_precision_score(y[te], oof_hgb[te], sample_weight=w[te]):.3f}  "
            f"MLP PR-AUC={average_precision_score(y[te], oof_mlp[te], sample_weight=w[te]):.3f}"
        )

    evaluate("HGB", oof_hgb, y, w)
    evaluate("MLP", oof_mlp, y, w)

    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "hgb_importance": np.mean(imp_hgb, axis=0),
            "mlp_importance": np.mean(imp_mlp, axis=0),
        }
    ).sort_values("hgb_importance", ascending=False)
    print("\n=== permutation importance (avg PR-AUC drop, held-out) ===")
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
