# Linking Particle Candidates Across the Parameter Sweep

## Background

The candidate detection pipeline was run as a 20-configuration sweep over two parameters: `slab_thickness` (6 values: 11, 15, 21, 23, 25, 31, controlling the background baseline estimation window) and `low_threshold_scale` (k_low, controlling how far hysteresis growth extends from a seed once it exceeds the fixed seed threshold, k_high = 4.2). Every configuration was run against the same 23 volumes, producing 502,618 accepted candidate rows in `combined_candidates.csv`, each backed by a per-candidate voxel dump in `combined_voxels.parquet` (272 million voxel rows, of which 2.86 million are flagged `in_grown = 1`, i.e. actually part of the grown region rather than the padded margin around it).

Because the same physical particle can seed and grow into a distinct accepted candidate under many of the 20 configurations, the raw candidate count is not a particle count — it's an overcount by an unknown, particle-dependent factor. The goal of this analysis was to collapse the sweep into unique physical particles per volume, with enough metadata per particle to support a downstream sensitivity calculation (how detection probability varies with threshold parameters).

## Method

The first approach tried was to link candidates strictly within a `slab_thickness` family, walking from the strictest `low_threshold_scale` to the laxest and testing whether a stricter candidate's peak voxel fell inside a laxer candidate's grown region. This worked but had two limitations. First, `peak_z/y/x` is the location of the maximum residual value within a grown region, and because residuals are quantized, the maximum is often reached at more than one voxel; which tied voxel gets recorded as "the peak" is not guaranteed to be stable across otherwise-identical reruns, which produced spurious mismatches. Second, and more importantly, it only linked within a single `slab_thickness` family — it had no mechanism for recognizing that a candidate from `slab_thickness=11` and a candidate from `slab_thickness=31` are the same physical particle, since each family uses its own background estimate and therefore its own residual field.

The method was revised to use voxel-set overlap instead of peak-coordinate equality: two candidates, from any of the 20 experiments, are treated as the same physical particle if their grown regions (`in_grown = 1` voxels) share at least one voxel. This was checked directly against the data before relying on it — for well-detected particles, a single voxel coordinate is in fact shared by the grown region across all 20 experiments, including across different `slab_thickness` values, so the overlap criterion links across both sweep dimensions in a single pass without a separate proximity-matching step for the background-estimate differences. Practically, this is a union-find computed over roughly 480,000 voxels that are touched by more than one candidate's grown region; running it over the full dataset takes under ten seconds.

Each resulting cluster (unique physical particle) carries: the number of raw candidate-instances that collapsed into it, the number and fraction of the 20 experiments that produced a detection (`detection_rate`), which `slab_thickness` values detected it, and — where a `slab_thickness` family detected it in at least one experiment but not all — which specific experiment numbers within that family missed it. That last field is the gate-rejection diagnostic: a miss within an otherwise-detecting family usually means the seed was present but the grown region failed a downstream shape or size gate (volume threshold, anisotropy, fill, or the z-extent cap) under that particular `low_threshold_scale`, rather than the particle genuinely not being there. The existing ground-truth review label (`particle`, where -1/10 are unlabeled, 0 is not-a-particle, 1/2/3 are true positives in anode/cathode/interface, and 4 is needs-review) is carried through per cluster as a consistency check, and per your current convention labels 1/2/3 are grouped into a single "TP" bucket.

## Findings

Collapsing the sweep reduces 502,618 raw candidate-instances to 85,859 unique physical particles across the 23 volumes — roughly a 5.9x reduction. Per-volume counts range from about 3,175 to 4,546, shown below.

| volume | unique particles | volume | unique particles |
|---|---|---|---|
| M50L-01 | 3175 | M50L-13 | 3842 |
| M50L-02 | 3288 | M50L-14 | 3343 |
| M50L-03 | 3711 | M50L-15 | 3695 |
| M50L-04 | 3640 | M50L-16 | 3214 |
| M50L-05 | 4382 | M50L-17 | 3618 |
| M50L-06 | 3573 | M50L-18 | 4184 |
| M50L-07 | 3622 | M50L-19 | 4063 |
| M50L-08 | 3866 | M50L-20 | 3633 |
| M50L-09 | 3387 | M50L-21 | 4546 |
| M50L-10 | 3628 | M50L-22 | 3777 |
| M50L-11 | 4146 | M50L-23 | 3952 |
| M50L-12 | 3574 | | |

Detection is highly uneven across the sweep. The median particle is recovered in only 3 of the 20 configurations (`detection_rate` = 0.15), and the interquartile range runs from 0.15 to 0.35. At the extremes, 2,568 particles are detected in every single configuration — these are the robust, unambiguous candidates — while 2,285 are detected in exactly one configuration, which is more consistent with marginal, threshold-dependent detections (or possibly noise) than with a stable physical particle. Coverage across `slab_thickness` values follows the same pattern: 50,597 particles (59%) show up under only one of the six background-estimation settings, while progressively fewer show up under two through all six (15,923 / 7,228 / 5,433 / 3,624 / 3,054 respectively).

The ground-truth label check is reassuring. 85,696 of 85,859 clusters (99.8%) have a single consistent label across every candidate-instance that fed into them. Of the 163 inconsistent clusters, none mix a genuinely conflicting pair (for example a "not-a-particle" call alongside a TP call); they're all combinations involving unlabeled or needs-review instances sitting alongside a definite call, which is expected given partial review coverage rather than a sign that unrelated particles were merged. Grouping 1/2/3 as TP per your convention, 605 clusters are TP-only, 85,250 are non-TP-only, and only 4 mix a TP label with an unlabeled or needs-review instance — again no contradictions.

The gate-rejection diagnostic flags 12,173 particles (about 14% of the total) that were missed by at least one experiment within a `slab_thickness` family that otherwise detected them. This is the clearest evidence that a meaningful share of the sweep's variance in candidate counts comes from downstream shape/size gates rejecting an over-grown region at a given `low_threshold_scale`, not from the underlying seed appearing or disappearing.

## Caveats and open questions

This analysis only describes particles that were detected by at least one of the 20 configurations — it says nothing about particles that no configuration ever seeded, so it can't by itself measure absolute sensitivity against ground truth, only relative sensitivity across the parameter grid. The `detection_rate` distribution is heavily skewed toward low values, and it's worth deciding together whether a floor (for example, particles detected in only one or two configurations) should be treated as noise and excluded, or kept and treated as the least-sensitive tail of real detections. It's also worth sanity-checking a handful of the 2,285 single-configuration particles by hand against the volumes to see which of those two interpretations holds. Finally, the voxel-overlap criterion is currently "any shared voxel" with no minimum overlap size — that was validated against the strongest particles but hasn't been stress-tested against particles sitting close enough together that their grown regions might touch without being the same physical object; worth discussing whether a minimum-overlap threshold is needed.

Code lives in `candidate_linking/` (`link_candidates.py`); outputs (`clusters.csv`, `instance_to_cluster.csv`) live in `Data/candidate_linking/`.
