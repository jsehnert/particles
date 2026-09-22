**Purpose & context**

Jim is a researcher building a micro-CT-based foreign particle detection pipeline for lithium-ion battery failure analysis. The goal is reliably detecting small foreign high-Z metallic contaminants (Cu, Al, Fe, Ni; ~40–75 µm, 2–3 voxels) within cylindrical jelly-roll structures, distinguishing them from structural artifacts and active-material features. This is an R&D failure analysis context, not inline inspection. Success means a pipeline with high sensitivity to genuine contaminants and tractable false-positive rates, with features sufficient for downstream classification.

The work spans the full pipeline: background estimation → residual computation → candidate detection → feature extraction → classification. Jim has strong domain knowledge and drives architectural decisions; Claude serves as a technical collaborator and implementer.

**Current state**

The pipeline is organized around two primary modules: `extract_candidates_3d.py` (detection) and `vol_analysis.py` (driver). The two-stage architecture is: (1) precompute residuals to int16 memmap (background subtraction via leave-in median baseline, per-slice metal masking, noise estimation); (2) parallel candidate detection via loky workers (hysteresis thresholding with k_high ≈ 4.0, k_low ≈ 2.99; connected component labeling; shape/intensity feature extraction). Jelly-roll interior culling is now done in stage 2 per slab (see below), not in stage 1.

Volume specs: 4600×1425×1425 voxels, uint8. Runtime environment: macOS (Apple M4 Max, 48 GB RAM, 16 logical cores) for development; Linux cloud with variable resources for deployment — portable optimizations are a priority. Optimal parallelism profiled at ~6 loky workers with chunk size 64.

Recent completed work:
- Custom numba sliding-histogram median+max kernel (`median_max_baseline_block`) verified bit-exact against numpy, achieving ~40× speedup
- Detection parallelized via loky with memory-constrained worker tuning
- `extract_metal_mask` optimized (BBDT algorithm, LUT-based label filtering, sparse seed indexing)
- `binary_propagation` replaced with label-then-scatter hysteresis reconstruction
- Slab-level two-labeling approach replacing per-candidate `ndimage.label` calls, with vectorized seed statistics
- Grayscale feature extraction added: `PrecomputedResidualSource` gains `read_grayscale(z0, z1)` via raw volume file path (`np.memmap.filename`), with `grayscale_z_offset` for frame alignment
- New `Candidate` fields: `peak_z/y/x` (integer global-frame join key), five grayscale scalars (`gs_median`, `gs_p90`, `gs_peak`, `gs_shell_median`, `gs_contrast`), `metal_threshold`
- Per-candidate voxel dump to `voxels.csv` keyed on `(volume_name, peak_z, peak_y, peak_x)` with columns `(z, y, x, residual, grayscale, in_grown)` over margin-padded bbox (`voxel_margin=3`), enabling offline shell feature recomputation
- Seed morphology features added: `n_seed` (largest connected high-threshold component), `n_seed_regions`, `n_seed_total`
- Radial position reworked: per-slab jelly-roll can geometry from `_slab_can_geometry` (`taubinSVD` circle fit on the largest external metal contour of a 9-slice max projection — 3 slices each top/mid/bottom of the slab). `radial_pos` is now normalized `d/r` (0 = axis, ~1 = wall, **unclamped** — the fitted radius is the outer wall surface and near-wall FP concentration carries signal), replacing the old absolute distance from the volume centre `(H/2, W/2)`. Downstream `ml_classifier/features.py` `radial_pos_frac` now aliases `radial_pos` (was `radial_pos * voxel_size / nominal_radius`, which would double-normalize). `metal_min_area`/`metal_grayscale_margin` threaded into `DetectParams` and both detection drivers.
- Interior culling moved out of precompute into detection. `GlobalData.read` no longer zeros the residual outside the global cylindrical support (the `cylinder_support_mask`/`not_`/`i16_` properties + caches and the `identify_cylindrical_support` import were removed). Instead `_process_slab` zeros `R` outside each slab's OWN fitted can contour (strict, actual filled contour) before thresholding, so seeds/candidates/features are all gated to the local can. Per-slice bright-metal zeroing (`_compute_residual2`) is retained. Live geometry helper split into `_slab_can_geometry` (9-slice projection) + `_can_geometry_from_projection` (mask→contour→taubin fit→interior mask), the latter shared with the offline backfill.
- Retroactive backfill EXECUTED and verified. `scripts/backfill_radial_pos.py` was run: `combined_candidates.csv` (502,618→501,066 rows), all 20 per-experiment `candidates.csv`, and all per-experiment `voxels.csv` updated (radial_pos normalized, 1,552 outside-can candidates deleted; per-exp deletions sum to 1,552; 0 orphan voxels). The `combined_voxels.parquet` step did NOT run in that pass (no pyarrow / --skip-parquet), so it was fixed separately with `scripts/fix_parquet_radial.py` (keys reconstructed from the per-exp candidate `.bak_radial_*` backups minus current; DuckDB anti-join): 272,376,056→271,531,815 rows (844,241 voxel rows removed), 0 orphan keys remaining, original backed up to `*.bak_radial_*`. Dataset is now fully consistent. Remaining: refit the ML models on the new radial_pos (see On the horizon). NB pyarrow row-group streaming works on a normal box but was reaped in the constrained Cowork sandbox; the robust recipe there was DuckDB foreground with `SET preserve_insertion_order=false`, writing to local disk then copying to the mount.
- (superseded) `scripts/backfill_radial_pos.py` originally written and validated before running — Recomputes `radial_pos` and drops outside-can candidates in the historical tables to match the new detection semantics WITHOUT re-detecting: per volume it replays the live chunking (`slices_per_chunk`=64 from `[analysis]`, `halo=z_extent_max`=16), fits one can circle per slab from the same 9-slice projection via `_can_geometry_from_projection`, sets `radial_pos = hypot(cy−cyc, cx−cxc)/r` (unclamped), and hard-deletes candidates whose centroid is outside the slab contour. Touches `combined_candidates.csv`, every `Data/experiments/<n>/candidates.csv`, and matching voxel rows in `combined_voxels.parquet` (streamed by row group) + each `voxels.csv` (streamed in chunks); each file backed up to `*.bak_radial_<timestamp>` first, in place. Geometry is per-volume (independent of `slab_thickness`, which is only the baseline window), so a location gets an identical `radial_pos` across experiments. Dry-run over all 23 volumes / 20 experiments: 1,552 / 502,618 rows deleted (0.31%), 0 slabs unfit; fitted radius ≈10.52 mm vs 10.5 mm nominal; can axis ~8 vox off image centre (the bias the old H/2,W/2 assumption carried). Needs `pyarrow` for the parquet step. Retroactive correction only — straddling candidates keep their original (ungated) shape/intensity features; only `radial_pos` + row membership change. Jim will run it on his workstation.

Known issues / recent findings:
- `n_voxels` has inherent soft chunk-sensitivity at knife-edge contested boundaries; invariant features (`n_seed`, `snr_peak`) should anchor detection decisions
- Docstring/signature drift flagged (k_high 4.0 vs. documented 4.7); dead code identified (`_to_int_elevation`, commented watershed block)
- Taper region (first ~266 slices) produces anomalous noise parameter estimates (ρ₂ out of fitted range); clipping to fitted box is the adopted solution
- `radial_pos` semantics changed from absolute voxels to a normalized radius fraction. Any code treating it as voxels will silently misbehave — notably `aggregate_experiments.ipynb` (`df[df["radial_pos"] >= 142]`) and the radial-band logic in `experiment_analysis.ipynb` (not updated). Already-written `candidates.csv` hold old absolute values — do not mix old and freshly-generated rows.
- Per-slab culling now depends on a grayscale slab + finite `metal_threshold` being present (always true on the `vol_analysis` path); a residual-only source gets no interior culling. The 9-slice max projection is a union envelope, so the gate is slightly loose at a strongly tapering slab's narrow end (still far tighter than the old global mask).

**On the horizon**

- Re-run the training/classification scripts once the `radial_pos` backfill (`scripts/backfill_radial_pos.py`) has been executed — the ML models were trained against the OLD absolute `radial_pos` and must be refit to the new normalized/gated values (and the reduced candidate set after outside-can deletions). Deliberately deferred until the backfill finishes.
- Parameter sweep validation: reruns at slab sizes 15 and 25 (previously executed incorrectly due to plumbing bug; fix confirmed, overnight fleet run pending)
- Downstream classification using extracted feature set
- Grouped cross-validation accounting for volume-level grouping structure (explicitly deferred, flagged as future work)
- Formal feature selection: univariate AUC ranking → Spearman clustermap → L1-logistic + tree ensemble reconciliation

**Key learnings & principles**

- **Connectivity requirement is mathematically essential**: at 5σ per-voxel threshold, ~1,030 expected false positives per volume; p² scaling from 2-voxel connectivity collapses this to near zero. The `n_seed` p³ argument extends this to the high-threshold core.
- **`n_seed` must be largest connected component, not pooled total**: pooled counting breaks the p² FP-collapse guarantee. Corrected implementation uses `ndimage.label` to gate on the largest connected component.
- **Fill and anisotropy are orthogonal gates**: fill catches sparse curved sheets; anisotropy ratio catches dense elongated slabs. Neither alone is sufficient — both must pass in `_axial_ok`.
- **`decay_drop` replaces `decay_ratio`**: near-zero ring-1 medians caused `decay_ratio` blow-up (values of −20,000+); `decay_drop` is the stable formulation.
- **Pass pre-clip residuals into detection**: the pipeline clips negatives to zero as the last step; passing pre-clip residuals restores the R==0 metal indicator and removes boundary contrast floor bias at no additional cost (σ is estimated pre-clip).
- **Texture features unsupportable at blob sizes 3–80 voxels**: GLCM/Haralick rejected; intensity skewness identified as the one distribution statistic with non-redundant content.
- **`snr_cluster` is inflated under selection**: all summed voxels exceed k_low by construction; pedestal-corrected statistic uses truncated-normal mean/variance at the low threshold. At k_low=2.99: μ_sel≈3.2738, s_sel≈0.2662.
- **Axial background estimation preferred over polar**: polar background approach retired; axial leave-in mean via cumulative sums adopted. Winding is an extruded 2D spiral (not a helix) — axially invariant by construction.
- **Interior culling belongs per-slab, not global**: one cylindrical-support mask derived from the full-cell max projection distorts (bulges) and can retain particles outside the local cell, especially in the taper. Gating each slab against its own fitted can contour is cleaner and strictly local.
- **System is isotropic**: z-blooming concerns for shell features do not apply.
- **Bit-identical outputs across configurations signal a plumbing bug**: used to detect the slab-15/25 misconfiguration during parameter sweep.
- **`voxels.csv` dump enables offline recomputation**: any shell feature can be recomputed without re-running detection, using `metal_threshold` surrogate (`grayscale > metal_threshold`) consistently for live and offline paths.

**Approach & patterns**

- Jim prefers concise, direct technical communication; pushes back on over-elaboration and unnecessary boilerplate
- Prefers methodology/architecture discussion before implementation
- Will implement straightforward aggregation code independently; Claude should provide conceptual guidance rather than boilerplate snippets unless asked
- Corrects Claude precisely when analysis or code misreads the implementation — Claude should update immediately without hedging
- Keeps distinct subtopics in separate conversations for organizational clarity
- Validates single-cell behavior before fleet runs
- Uses Monte Carlo verification for analytical estimates (e.g., noise scale factors, lag-correlation bounds)

**Tools & resources**

- Python with numpy, scipy.ndimage, numba, joblib/loky, OpenCV (connected components)
- `np.memmap` for large volume I/O; int16 memmap for residual storage
- `uv` for package management
- **All Python functions must use type hints** (`numpy.typing.NDArray` annotations; `FloatArray = NDArray[np.float32]` alias pattern)
- Noise model: pentadiagonal covariance, lag-1 (ρ₁≈0.516) and lag-2 (ρ₂≈0.092) inter-slice correlations; surface-fit σ_R/σ as function of N, ρ₁, ρ₂; operating point N=11, σ_R/σ≈0.909
- Literature: Chernov *Circular and Linear Regression* (2010) for circle fitting; `circle-fit` package (`taubinSVD()`) used for the per-slab can circle