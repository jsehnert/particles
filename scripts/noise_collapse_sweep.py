#!/usr/bin/env python3
"""Empirical noise-collapse sweep: measure how the candidate seed count collapses with
the connectivity floor, on REAL residual volumes.

For a grid of (k_high, min_seed_voxels), this counts connected high-threshold seed
components (6-connectivity) of size >= min_seed_voxels across a volume's residual. Because
genuine particles are rare (dozens per volume), that count is, to good approximation, the
noise / false-positive behaviour at each operating point -- with all the real
non-stationarity, streaks, and ring artifacts baked in, unlike a synthetic Gaussian model.

PURPOSE (and its limits). This locates the NOISE-COLLAPSE FLOOR: the smallest
min_seed_voxels at which speckle stops dominating the candidate set. It is NOT a predictor
of the classifier's FP workload -- that is dominated by *structured* artifacts (windings,
walls, streaks), which are not i.i.d. noise. Read the knee LOCATION (which m collapses the
count), not the absolute numbers, as the robust output. The m=1 measured count is printed
next to the analytical N*p(k_high) for stationary Gaussian noise: their ratio is a direct,
quantitative read on how heavy/structured the real tail is (>>1 means the stationary model
underestimates, exactly the caveat that motivates measuring empirically).

Run it on ~3 volumes spanning conditions (prefer "natural" cells over the spiked
M50L-16..23 so extra TPs don't muddy the count) and check the knee is consistent; a knee
that moves across volumes means min_seed_voxels should be set for the noisiest one.

The residual is independent of k_high (k_high applies to snr = R/sigma afterward), so each
volume is precomputed once and the whole k_high grid is swept on the stored residual.

Usage:
    python scripts/noise_collapse_sweep.py --vols 0 5 10
    python scripts/noise_collapse_sweep.py --vols 0 5 10 --k-high 3.0 3.5 4.0 4.2 4.5 5.0 \
        --m 1 2 3 4 5 --out /tmp/noise_collapse.csv

Boundary note: seed components are counted per non-overlapping z-chunk, so a component
straddling a chunk boundary is split. Speckle components are 1-3 voxels so this is
negligible for the floor; long structured components (streaks) may fragment -- consistent
with this tool measuring the speckle floor, not accounting for structured artifacts.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# repo root for cfg_data / extract_candidates_3d, and this script's own dir (scripts/) for
# vol_analysis — so it runs both as `python scripts/noise_collapse_sweep.py` and as
# `python -m scripts.noise_collapse_sweep`.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_candidates_3d import precompute_residual  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config.toml"
_STRUCT = ndimage.generate_binary_structure(3, 1)  # 6-connectivity, matches detection


def _sweep_volume(
    src,
    k_highs: list[float],
    ms: list[int],
    chunk: int,
    verbose: bool,
) -> tuple[dict[tuple[float, int], int], dict[float, int], int]:
    """Return (counts[(k_high, m)] , seed1[k_high] , n_support).

    counts: components of size >= m at each k_high.
    seed1: total seed voxels above k_high (the m=1 single-voxel exceedance count).
    n_support: number of finite-sigma voxels swept (for the analytical reference).
    """
    Z, H, W = src.shape
    counts: dict[tuple[float, int], int] = {(k, m): 0 for k in k_highs for m in ms}
    seed1: dict[float, int] = {k: 0 for k in k_highs}
    n_support = 0
    for z0 in range(0, Z, chunk):
        z1 = min(z0 + chunk, Z)
        R, sigma = src.read(z0, z1)
        snr = R / sigma  # sigma is (nz,1,1); broadcasts
        n_support += int(R.size)
        for k in k_highs:
            seed = snr >= k
            s = int(seed.sum())
            seed1[k] += s
            if s == 0:
                continue
            labels, n = ndimage.label(seed, structure=_STRUCT)
            if n == 0:
                continue
            sizes = np.bincount(labels.ravel())[1:]  # drop background (label 0)
            for m in ms:
                counts[(k, m)] += int(np.count_nonzero(sizes >= m))
        if verbose:
            print(f"    z=[{z0},{z1})  ({z1}/{Z})", flush=True)
    return counts, seed1, n_support


def _build_source(vol_index: int, exp_index: int, tmp_dir: Path):
    """Precompute the residual for one volume and return a PrecomputedResidualSource.

    Imported lazily so `--help` etc. do not construct GlobalData / load volumes.
    """
    from vol_analysis import GlobalData  # type: ignore  (scripts/ is on sys.path)

    gd = GlobalData(cfg_path=CONFIG_PATH, vol_index=vol_index, exp_index=exp_index)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    vol = gd.vol
    src = precompute_residual(
        gd,
        residual_path=str(tmp_dir / "residual_memmap.dat"),
        sigma_path=str(tmp_dir / "sigma_memmap.npy"),
        chunk=gd.preprocess_slices_per_chunk,
        residual_dtype=np.dtype(np.int16),
        verbose=False,
    )
    name = gd.cell_name
    slab_thickness = int(gd.slab_thickness)
    baseline_method = str(gd.baseline_method)
    gd.free_volume()
    return src, name, slab_thickness, baseline_method


def _resolve_exp_index(cfg_path: Path, slab_thickness: int, baseline_method: str) -> int:
    """Find an [[experiment]] with this slab_thickness + baseline_method. The residual
    depends only on those two (not k_low/k_high), so any match gives an identical residual;
    we take the first."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    for i, e in enumerate(cfg["experiment"]):
        if int(e["slab_thickness"]) == slab_thickness and str(e["baseline_method"]) == baseline_method:
            return i
    raise SystemExit(
        f"No [[experiment]] with slab_thickness={slab_thickness}, "
        f"baseline_method={baseline_method!r} in {cfg_path}"
    )


def _cleanup_residual(tmp_dir: Path) -> None:
    """Delete the residual scratch files (and the dir if now empty)."""
    removed = False
    for fn in ("residual_memmap.dat", "sigma_memmap.npy"):
        p = tmp_dir / fn
        if p.exists():
            p.unlink()
            removed = True
    try:
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except OSError:
        pass
    if removed:
        print(f"Cleaned up residual scratch in {tmp_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vols", type=int, nargs="+", required=True, help="cell indices to sweep")
    ap.add_argument(
        "--slab-thickness", type=int, default=25,
        help="baseline window; selects the residual config (default 25)",
    )
    ap.add_argument("--baseline-method", default="median", help="default median")
    ap.add_argument(
        "--exp-index", type=int, default=None,
        help="advanced: use this [[experiment]] index directly, overriding "
        "--slab-thickness/--baseline-method",
    )
    ap.add_argument("--k-high", type=float, nargs="+", default=[3.0, 3.5, 4.0, 4.2, 4.5, 5.0])
    ap.add_argument("--m", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--chunk", type=int, default=128, help="z-slices per processing chunk")
    ap.add_argument("--tmp-dir", type=Path, default=PROJECT_ROOT / "Data" / "_tmp_noise_sweep")
    ap.add_argument("--out", type=Path, default=None, help="optional CSV output path")
    ap.add_argument(
        "--keep-residual", action="store_true",
        help="keep the ~16GB residual scratch (default: delete it when done)",
    )
    args = ap.parse_args()

    exp_index = (
        args.exp_index
        if args.exp_index is not None
        else _resolve_exp_index(CONFIG_PATH, args.slab_thickness, args.baseline_method)
    )
    print(
        f"Using exp-index {exp_index} "
        f"(slab_thickness={args.slab_thickness}, baseline={args.baseline_method})"
    )
    # Create the output dir up front so a bad --out path fails now, not after the
    # (slow) per-volume precompute.
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    k_highs = sorted(args.k_high)
    ms = sorted(args.m)
    rows: list[dict] = []

    try:
        for vi in args.vols:
            t0 = time.time()
            print(f"\n=== volume index {vi} (exp {exp_index}): precomputing residual ...", flush=True)
            src, name, slab_thickness, baseline_method = _build_source(vi, exp_index, args.tmp_dir)
            counts, seed1, n_support = _sweep_volume(src, k_highs, ms, args.chunk, verbose=True)
            print(
                f"--- {name}  slab_thickness={slab_thickness} baseline={baseline_method}  "
                f"(n_support={n_support:,}, {time.time() - t0:.0f}s)"
            )
            # header
            print("  k_high  " + "  ".join(f"m>={m:<8d}" for m in ms) + "   | m=1 measured vs N*p")
            for k in k_highs:
                p = float(norm.sf(k))
                analytic_m1 = n_support * p
                meas_m1 = seed1[k]
                ratio = (meas_m1 / analytic_m1) if analytic_m1 > 0 else float("nan")
                cells = "  ".join(f"{counts[(k, m)]:<9d}" for m in ms)
                print(
                    f"  {k:5.2f}   {cells}   | {meas_m1:,} vs {analytic_m1:,.0f}  (x{ratio:.1f})"
                )
                for m in ms:
                    rows.append(
                        {
                            "volume_index": vi,
                            "volume_name": name,
                            "slab_thickness": slab_thickness,
                            "baseline_method": baseline_method,
                            "k_high": k,
                            "min_seed_voxels": m,
                            "n_components": counts[(k, m)],
                            "seed_voxels_m1": meas_m1,
                            "analytic_np_m1": round(analytic_m1, 3),
                            "tail_ratio_m1": round(ratio, 3),
                            "n_support": n_support,
                        }
                    )
            src.free_volume()

        if args.out is not None and rows:
            with args.out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\nwrote {len(rows)} rows -> {args.out}")

        # Cross-volume consistency: for each k_high, the smallest m whose count is <= 1 in
        # EVERY volume (a practical "collapsed" knee), and whether it agrees across volumes.
        if len(args.vols) > 1:
            print("\n=== cross-volume knee (smallest m with <=1 component, per volume) ===")
            by_vol: dict[int, dict[float, int]] = {}
            for r in rows:
                by_vol.setdefault(r["volume_index"], {})
                if r["n_components"] <= 1:
                    d = by_vol[r["volume_index"]]
                    d[r["k_high"]] = min(d.get(r["k_high"], 10**9), r["min_seed_voxels"])
            for k in k_highs:
                knees = [by_vol.get(vi, {}).get(k) for vi in args.vols]
                shown = ", ".join(str(x) if x is not None else ">max" for x in knees)
                agree = "consistent" if len(set(knees)) == 1 else "VARIES across volumes"
                print(f"  k_high={k:5.2f}: knee m per volume = [{shown}]  ({agree})")
    finally:
        if args.keep_residual:
            print(f"\nResidual scratch kept in {args.tmp_dir} (--keep-residual)")
        else:
            _cleanup_residual(args.tmp_dir)


if __name__ == "__main__":
    main()
