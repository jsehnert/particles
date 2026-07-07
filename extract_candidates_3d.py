from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

FloatArray: TypeAlias = NDArray[np.float32]


class ResidualSource(Protocol):
    """Lazy provider of residual + noise for a z-range. Backed by memmap,
    HDF5, zarr, or on-the-fly background subtraction — caller's choice."""

    shape: tuple[int, int, int]  # (Z, H, W)

    def read(self, z0: int, z1: int) -> tuple[FloatArray, FloatArray]:
        """Return (R, sigma_R) for slices [z0, z1).
        sigma_R shaped (z1-z0,1,1), (z1-z0,H,W), or broadcastable."""
        ...


class MyResidualSource:
    def __init__(self, vol: NDArray[np.uint8]):
        self.vol: NDArray[np.uint8] = vol
        self.shape: tuple[int, int, int] = vol.shape

    def read(self, z0: int, z1: int) -> tuple[FloatArray, FloatArray]:
        # Placeholder implementation: return the raw volume as residual and a constant noise level.
        R = self.vol[z0:z1].astype(np.float32)
        sigma_R = np.full(
            (z1 - z0, 1, 1), 10.0, dtype=np.float32
        )  # Example noise level
        return R, sigma_R


@dataclass
class Candidate:
    n_voxels: int  # grown component size (k_low)
    n_seed: int  # high-threshold core size (k_high) — detection basis
    z_min: int
    z_max: int
    z_extent: int
    y_extent: int
    x_extent: int
    fill: float  # fill fraction of the bounding box
    centroid: tuple[float, float, float]  # global (z, y, x), intensity-weighted
    snr_cluster: float
    r_peak: float
    r_peak_ratio: float
    peak_offset: float
    snr_peak: float
    linearity: float
    planarity: float
    sphericity: float
    axis_z: float  # |z-component| of the principal eigenvector (long axis)
    normal_z: float  # |z-component| of the minor eigenvector (plate normal)
    edge_contrast: float  # SNR drop across the component's boundary, in sigma units
    decay_drop: float
    seed_grown_ratio: float


def iter_slabs(Z: int, chunk: int, halo: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield (read_z0, read_z1, chunk_z0, chunk_z1).
    Chunk is the ownership region; [read_z0,read_z1) includes the halo."""
    for c0 in range(0, Z, chunk):
        c1 = min(c0 + chunk, Z)
        r0 = max(0, c0 - halo)
        r1 = min(Z, c1 + halo)
        yield r0, r1, c0, c1


def _axial_ok(
    z_extent: int,
    y_extent: int,
    x_extent: int,
    n_vox: int,
    fill: float,
    z_lo: int,
    z_hi_small: int,
    small_voxel_cutoff: int,
    aniso_factor: float,
    z_pad: int,
    min_fill: float = 0.1,
) -> bool:
    """Axial-transience test with two regimes.

    Small components (n_vox < cutoff): fixed bound [z_lo, z_hi_small]. At 2-3 voxels
    the lateral dims are too small for a meaningful shape ratio, and real particles
    this size are axially short, so the original fixed window is correct.

    Large components (n_vox >= cutoff): admit z proportional to the *thinnest* lateral
    dimension. An isotropic blob has z_extent ~ lat_min and passes; a wall/winding
    remnant is thin in cross-section but long in z (z_extent >> lat_min) and is cut.
    """
    if z_extent < z_lo:
        return False
    if n_vox < small_voxel_cutoff:
        return z_extent <= z_hi_small
    lat_min = min(y_extent, x_extent)
    if z_extent > aniso_factor * lat_min + z_pad:
        return False
    return fill >= min_fill


def _to_int_elevation(R: FloatArray) -> NDArray[np.int16]:
    """Map residual to an int16 elevation for watershed_ift, with bright regions
    as basin bottoms (negated). Robust-scaled to use the int16 range without
    overflow; only the ordering matters to the watershed, not absolute values."""
    finite = R[np.isfinite(R)]
    if finite.size == 0:
        return np.zeros(R.shape, dtype=np.int16)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros(R.shape, dtype=np.int16)
    # normalize to [0, 1], negate so high-R → low elevation, scale to int16 span
    norm = (R - lo) / (hi - lo)
    elev = (-(norm) * 32000.0).astype(np.int16)  # bright = deep basin
    return elev


# Eignevalue-based shape features could be added here, e.g., via PCA on the voxel coordinates.
def _eigen_features(sub_grown: NDArray[np.bool_]) -> tuple[float, float, float]:
    coords = np.argwhere(sub_grown).astype(np.float64)  # (N,3), bbox-local z,y,x
    coords -= coords.mean(axis=0)
    C = (coords.T @ coords) / coords.shape[0]  # covariance, ÷N
    C[np.diag_indices(3)] += 1.0 / 12.0  # Sheppard: unit-cube self-variance
    l3, l2, l1 = np.linalg.eigvalsh(C)  # ASCENDING → unpack small→large
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1
    return linearity, planarity, sphericity


def _eigen_features_with_axes(
    sub_grown: NDArray[np.bool_],
) -> tuple[float, float, float, float, float]:
    """Rotation-invariant shape triple plus orientation of a voxel component.

    Returns ``(linearity, planarity, sphericity, axis_z, normal_z)``, all from the
    Sheppard-corrected covariance of the voxel coordinates.

    linearity/planarity/sphericity: the tensor-shape triple (sum to 1). High
        sphericity = blob, high planarity = plate/sheet, high linearity = filament.

    axis_z: |z-component| of the PRINCIPAL eigenvector (largest eigenvalue) — the
        object's long axis. Near 1 = elongated along z (winding-aligned, the
        suspicious direction for a filament); near 0 = long axis lies in-plane.
    normal_z: |z-component| of the MINOR eigenvector (smallest eigenvalue) — for a
        planar object this is the plate normal. Near 1 = plate lies in the (y,x)
        plane, i.e. its face is perpendicular to z; near 0 = plate stands on edge,
        its face containing the z-axis. Only meaningful when planarity is high.

    Absolute values are taken because eigenvector sign is arbitrary; only the axis
    orientation matters, not its direction.
    """
    coords = np.argwhere(sub_grown).astype(np.float64)  # (N,3), bbox-local z,y,x
    coords -= coords.mean(axis=0)
    C = (coords.T @ coords) / coords.shape[0]  # covariance, ÷N
    C[np.diag_indices(3)] += 1.0 / 12.0  # Sheppard: unit-cube self-variance

    # eigh: ascending eigenvalues, eigenvectors as COLUMNS in the same order.
    evals, evecs = np.linalg.eigh(C)
    l3, l2, l1 = evals  # small → large
    v_minor = evecs[:, 0]  # eigenvector of smallest eigenvalue (plate normal)
    v_major = evecs[:, 2]  # eigenvector of largest eigenvalue (long axis)

    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1

    # z is axis 0 of coords, so the z-component is row 0 of each eigenvector.
    axis_z = float(abs(v_major[0]))
    normal_z = float(abs(v_minor[0]))

    return linearity, planarity, sphericity, axis_z, normal_z


def _shell_features(
    labels: NDArray[np.integer],
    snr: FloatArray,
    R: FloatArray,
    k_sigma: float,
    sl: tuple[slice, slice, slice],
    lid: int,
    structure: NDArray[np.bool_],
    shell_radius: int = 3,
) -> tuple[float, float]:
    """Boundary sharpness of a component from concentric SNR shells.

    Returns ``(edge_contrast, decay_drop)`` — the level and the shape of the SNR
    falloff across the grown blob's surface, both in sigma units, computed on one
    shared ring stack so they characterize the same surface.

    edge_contrast: median inner-edge SNR minus median SNR on a baseline shell at
        distance ``2..shell_radius``. Large for a sharp (metal) edge, near zero for
        a diffuse bump. The pinned ``k_low`` ring (distance 1) is excluded — it
        carries no dynamic range as an absolute level. Retains a mild dependence on
        brightness (the inner edge floats up with peak SNR).
    decay_drop: median SNR on ring 2 minus median SNR on ring 1. Measures falloff
        *magnitude* only — the ``k_low`` pinning common to both rings cancels, so it is
        brightness-independent. Near 0 for a sharp edge, larger for a diffuse
        bump. The pinned ring that disqualifies distance 1 from edge_contrast is
        exactly what makes it a valid decay denominator.

    Exact-zero voxels (zeroed metal in the unclipped residual) are excluded from all
    outer shells; the inner edge is grown (>= k_low), never metal. Ring statistics
    are medians, robust to a directional winding pedestal. Either value is NaN when
    its shells are empty after exclusion (ringed by metal / clipped by the frame) or,
    for decay_drop, when the ring-1 median is non-positive — read as 'no evidence',
    never a rejection.
    """
    psl = tuple(
        slice(max(0, s.start - shell_radius), min(dim, s.stop + shell_radius))
        for s, dim in zip(sl, snr.shape)
    )
    comp = labels[psl] == lid
    snr_p = snr[psl]
    R_p = R[psl]
    nonzero = R_p != 0.0  # exclude zeroed metal from outer shells

    # shared dilation stack — every ring is carved from these
    d1 = ndimage.binary_dilation(comp, structure, iterations=1)
    d2 = ndimage.binary_dilation(comp, structure, iterations=2)
    dR = ndimage.binary_dilation(comp, structure, iterations=shell_radius)

    inner_edge = comp & ~ndimage.binary_erosion(comp, structure)
    ring1 = (d1 & ~comp) & nonzero  # pinned isosurface ring
    ring2 = (d2 & ~d1) & nonzero
    baseline = (dR & ~d1) & nonzero  # distance 2..shell_radius

    # edge_contrast: inner-edge level minus outer-baseline level
    if baseline.any():
        edge_contrast = float(np.median(snr_p[inner_edge]) - np.median(snr_p[baseline]))
    else:
        edge_contrast = float("nan")
    # decay_ratio: shape of falloff, brightness-independent
    if ring1.any() and ring2.any():
        m1 = float(np.median(snr_p[ring1]))
        m2 = float(np.median(snr_p[ring2]))
        decay_drop = float(m1 - m2)
    else:
        decay_drop = float("nan")

    return edge_contrast, decay_drop


def detect_candidates_streaming(
    src: ResidualSource,
    k_high: float = 4.0,  # High threshold for candidate identifiaction
    k_low: float = 2.99,  # Lower hysteresis threshold for candidate growth
    min_voxels: int = 2,  # Min volume voxels at high threshold to be considered a candidate
    z_extent_max: int = 10,  # Longest admissible axial extent of a candidate in slices
    small_z_bounds: tuple[int, int] = (1, 4),
    small_voxel_cutoff: int = 5,
    aniso_factor: float = 1.0,
    z_pad: int = 2,
    min_fill: float = 0.15,  # Minimum fill fraction for large components
    slices_per_chunk: int = 128,  # Subvolume number of slices for one loop
    z_offset: int = 0,  # Offset to add to the z coordinates of the candidates
) -> list[Candidate]:
    """Detect high-Z foreign-particle candidates in a residual volume via slab-streamed
    3D blob detection with hysteresis growth.

    Reads the residual volume in overlapping z-slabs through ``src``, seeds components at a
    high per-voxel SNR threshold, grows each seed into connected lower-SNR voxels
    (hysteresis), groups the grown voxels into 3D connected components (6-connectivity),
    filters by core size and axial extent, and computes per-component features. Slabs overlap
    by a halo so that components straddling a slab boundary are fully reconstructed; each
    component is claimed by exactly one slab via intensity-weighted-centroid ownership,
    preventing double-counting without cross-slab label stitching.

    The detection model targets compact attenuation anomalies (foreign high-Z particles) and
    is not a general segmenter; it assumes the residual is a background-subtracted,
    noise-normalizable field with metal regions pre-zeroed by ``src``.

    Hysteresis separates detection from measurement. A component is *detected* only if its
    high-threshold core (voxels above ``k_high``) is itself at least ``min_voxels`` connected
    voxels — this preserves the ~p**2 false-positive collapse that single-voxel speckle
    rejection relies on, because growth into the low-threshold skirt cannot manufacture a
    seed. All component *features* (size, extents, centroid, integrated and peak SNR) are
    then measured on the grown blob (voxels above ``k_low`` connected to the core), recovering
    the partial-volume / blooming skirt that a single threshold would truncate. The maxim is
    detect strict, measure generous.

    Axial-extent filtering uses two regimes so that large, genuinely isotropic particles are
    not clipped while axially-elongated structure (winding remnants, can wall) is still
    rejected. Small components are held to a fixed slice bound; large components are admitted
    only when their axial extent is commensurate with their thinnest lateral dimension (see
    ``_axial_ok``). A separate hard cap, ``z_extent_max``, bounds the largest admissible
    component and is decoupled from the regime logic because it also sizes the slab halo and
    therefore the dedup-containment guarantee. The axial gate operates on the *grown* extent,
    so ``z_extent_max`` must accommodate the post-growth object.

    Args:
        src: Lazy residual provider. ``src.read(z0, z1)`` returns ``(R, sigma_R)`` for
            slices ``[z0, z1)``, where ``R`` is the background-subtracted residual
            (shape ``(z1-z0, H, W)``) and ``sigma_R`` is the residual noise level,
            broadcastable to ``R`` (typically ``(z1-z0, 1, 1)`` for a per-slice profile).
            ``src.shape`` gives the full ``(Z, H, W)`` volume dimensions. The caller is
            responsible for background subtraction, noise estimation, and zeroing metal
            regions before the residual is read.
        k_high: Per-voxel seeding threshold in units of ``sigma_R``. A voxel may seed a
            component when ``R / sigma_R >= k_high``. This is the detection threshold; the
            false-positive control is governed entirely by it.
        k_low: Per-voxel growth threshold in units of ``sigma_R``, with ``k_low < k_high``.
            Voxels above ``k_low`` and connected to a seed are absorbed into the component
            for feature measurement only. Around 2.5-3.0 is typical. Too low bridges adjacent
            particles through low-SNR ridges (a real risk in winding geometry) and distorts
            shape; too high recovers no skirt. Sweep against known particles. Defaults to 2.8.
        min_voxels: Minimum size, in voxels, of the high-threshold core (``n_seed``) required
            for detection. The value of 2 is the essential speckle suppressor: isolated
            single-voxel core excursions (the dominant false-positive family) are discarded,
            collapsing the accidental count via ~p**2 scaling. Tested on the core, NOT the
            grown blob.
        z_extent_max: Hard upper bound on a component's *grown* axial extent, in slices.
            Serves two roles: it is the absolute backstop reject for anything spanning more z
            than the largest credible particle (including blooming/skirt margin), and it sets
            the slab halo (``halo = z_extent_max``), guaranteeing any admissible component is
            fully contained in the slab that owns its centroid. Drives peak per-slab memory
            (``slices_per_chunk + 2*z_extent_max`` slices), so size it from physics — against
            grown extents, not core extents — not generously.
        small_z_bounds: Inclusive ``(min, max)`` slice bounds applied to small components
            (grown size below ``small_voxel_cutoff``), where lateral dimensions are too small
            for a meaningful shape ratio and real particles are axially short.
        small_voxel_cutoff: Grown-voxel-count boundary between the two axial-extent regimes.
            Components with ``n_voxels < small_voxel_cutoff`` use the fixed ``small_z_bounds``;
            larger components use the lateral-scaled test. If large particles routinely have a
            thinnest lateral dimension of only 2-3 voxels, the ratio test is unstable there
            and this cutoff may need raising (or the large regime gating on ``lat_min``
            directly).
        aniso_factor: Permitted ratio of axial extent to thinnest lateral dimension in the
            large-component regime: a component passes when
            ``z_extent <= aniso_factor * min(y_extent, x_extent) + z_pad``. Values above 1
            allow z to exceed the lateral cross-section, accommodating high-Z blooming, which
            smears more in z than laterally. The parameter most worth calibrating against
            known-size implanted particles or a phantom.
        z_pad: Additive slack (slices) in the large-component axial test, preventing integer
            roundoff from clipping borderline mid-size blobs.
        min_fill: Minimum fill fraction for large components. The fill fraction is defined as
            the ratio of the number of voxels in the grown component to the volume of its
            bounding box. This parameter helps to reject elongated or sparse components that
            may pass the axial extent test but are not compact enough to be considered valid
            candidates.
        slices_per_chunk: Number of z-slices each slab owns (the core region). The actual
            read range is the core padded by ``z_extent_max`` slices on each side. Reduce
            if per-slab memory is too large.
        z_offset: Offset to add to the z coordinates of the candidates. This is useful when
        the residual source is a subvolume of a larger volume and you want to report the candidate coordinates in the context of the larger volume. Defaults to 0.

    Returns:
        A list of ``Candidate`` records, one per surviving component, in slab-processing
        order. Each carries global (volume-frame) coordinates and both size measures:
        ``n_seed`` (high-threshold core, the detection basis) and ``n_voxels`` (grown blob,
        the feature basis), plus z-range, axial and lateral extents, intensity-weighted
        centroid ``(z, y, x)``, cluster-integrated SNR, and peak-shape features (``r_peak``,
        ``r_peak_ratio``, ``peak_offset``, ``snr_peak``). Components are deduplicated across
        slab boundaries by centroid ownership.

    Notes:
        - ``halo = z_extent_max`` accounts only for component containment, NOT for the
          background estimator's own window; ``src`` must internally read whatever extra
          slices its background/noise model needs and return a residual already consistent
          over ``[z0, z1)``.
        - ``sigma_R`` must be aligned to the same z-range as ``R`` (the read range including
          halo), not the core range.
        - Coordinates and ``peak_offset`` are in isotropic voxel units; rescale by voxel pitch
          per axis if the acquisition is anisotropic. The axial-extent regimes likewise
          compare raw voxel counts across axes and assume isotropic voxels.
        - ``snr_cluster`` is now computed over the grown component and is NOT comparable to
          tier edges calibrated on a single-threshold blob; the MC calibration must be redone
          on the grown statistic. ``snr_peak`` is threshold-independent (just the max voxel)
          and is the robust significance anchor in the interim. Tier assignment is
          intentionally not applied here.
    """
    Z, H, W = src.shape
    halo = z_extent_max  # hard physical ceiling sizes the slab overlap
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    z_lo, z_hi_small = small_z_bounds

    out: list[Candidate] = []
    current_candidates = 0
    for r0, r1, c0, c1 in iter_slabs(Z, slices_per_chunk, halo):
        print(
            f"    Processing chunk z=[{c0}, {c1}) with read z=[{r0}, {r1}), candidates: {len(out)}, found: {len(out) - current_candidates} in last chunk"
        )
        current_candidates = len(out)

        R, sigma = src.read(r0, r1)  # (r1-r0, H, W)
        snr = R / sigma

        seed_high = snr >= k_high  # detection seeds
        grow_low = snr >= k_low  # skirt mask (k_low < k_high)

        # hysteresis: grow seeds into connected low-threshold voxels
        grown = ndimage.binary_propagation(
            seed_high, mask=grow_low, structure=structure
        )

        labels, n = ndimage.label(grown, structure=structure)
        if n == 0:
            continue

        ids = np.arange(1, n + 1)
        objs = ndimage.find_objects(labels)

        for i, lid in enumerate(ids):
            sl = objs[i]  # bbox of grown component

            # bbox-local masks: grown blob, and its high-threshold core
            sub_grown = labels[sl] == lid
            sub_seed = sub_grown & seed_high[sl]

            # --- detection criterion: a CONNECTED high core of >= min_voxels ---
            # Count seeds per connected cluster, not by bulk sum: two isolated
            # single-voxel seeds bridged through a k_low ridge must NOT fake a core.
            seed_lbl, _ = ndimage.label(sub_seed, structure=structure)
            if seed_lbl.max() == 0:
                continue  # defensive; propagation guarantees >=1 seed voxel
            n_seed = int(np.bincount(seed_lbl.ravel())[1:].max())
            if n_seed < min_voxels:  # p^min_voxels logic intact at k_high
                continue

            n_vox = int(sub_grown.sum())  # grown size, for features

            # extents from the GROWN bbox (full physical footprint incl. skirt)
            z_min_loc, z_max_loc = sl[0].start, sl[0].stop - 1
            z_extent = z_max_loc - z_min_loc + 1
            y_extent = sl[1].stop - sl[1].start
            x_extent = sl[2].stop - sl[2].start
            fill = n_vox / (
                z_extent * y_extent * x_extent
            )  # fill fraction of the bounding box

            # axial-extent gate (hard cap + two-regime transience), on grown extent
            if z_extent > z_extent_max:
                continue
            if not _axial_ok(
                z_extent=z_extent,
                y_extent=y_extent,
                x_extent=x_extent,
                n_vox=n_vox,
                fill=fill,
                z_lo=z_lo,
                z_hi_small=z_hi_small,
                small_voxel_cutoff=small_voxel_cutoff,
                aniso_factor=aniso_factor,
                z_pad=z_pad,
                min_fill=min_fill,
            ):
                continue

            R_sub = R[sl]
            snr_sub = snr[sl]

            # --- intensity-weighted COM over the GROWN component ---
            w = np.where(sub_grown, R_sub, 0.0)
            cz_loc, cy, cx = ndimage.center_of_mass(w)
            cz_loc += sl[0].start
            cy += sl[1].start
            cx += sl[2].start
            cz_glob = cz_loc + r0

            # ownership: claim only if weighted centroid falls in this slab's core
            if not (c0 <= cz_glob < c1):
                continue

            # --- features over the grown component ---
            r_vals = R_sub[sub_grown]
            r_peak = float(r_vals.max())
            r_mean = float(r_vals.mean())
            r_peak_ratio = r_peak / r_mean

            snr_cluster = float(snr_sub[sub_grown].sum() / np.sqrt(n_vox))
            snr_peak = float(snr_sub[sub_grown].max())

            peak_local = np.unravel_index(
                np.argmax(np.where(sub_grown, R_sub, -np.inf)), R_sub.shape
            )
            pz = peak_local[0] + sl[0].start
            py = peak_local[1] + sl[1].start
            px = peak_local[2] + sl[2].start
            peak_offset = float(
                np.sqrt((pz - cz_loc) ** 2 + (py - cy) ** 2 + (px - cx) ** 2)
            )

            linearity, planarity, sphericity, axis_z, normal_z = (
                _eigen_features_with_axes(sub_grown)
            )
            edge_contrast, decay_drop = _shell_features(
                labels=labels,
                snr=snr,
                R=R,
                k_sigma=k_low,
                sl=sl,
                lid=lid,
                structure=structure,
                shell_radius=3,
            )

            # --- seed-to-grown intensity concentration ---
            # Mass in the connected core over mass in the grown blob. Distinct from
            # n_seed/n_voxels (count): keys on where the *intensity* sits, not how many
            # voxels. ->1 compact core + faint skirt (healthy); low = mass leaked to the
            # skirt (diffuse, or a k_low-bridged second core outside sub_seed).
            S_grown = float(R_sub[sub_grown].sum())
            S_seed = float(R_sub[sub_seed].sum())
            seed_grown_ratio = S_seed / S_grown  # (0, 1]; S_grown > 0 by construction

            # region DEBUGGING
            if n_vox > 30 and False:
                grown_voxels = np.argwhere(sub_grown) + np.array(
                    [sl[0].start, sl[1].start, sl[2].start]
                )
                grown_voxels = sorted(grown_voxels, key=lambda x: (x[0], x[1], x[2]))
                seed_voxels = np.argwhere(sub_seed) + np.array(
                    [sl[0].start, sl[1].start, sl[2].start]
                )
                seed_voxels = sorted(seed_voxels, key=lambda x: (x[0], x[1], x[2]))

                print(
                    f"{'=' * 40}\n"
                    f"z_min: {z_min_loc + r0 + z_offset}, z_max: {z_max_loc + r0 + z_offset}, "
                    f"z_extent: {z_extent}, "
                    f"n_seed: {n_seed}, "
                    f"n_vox: {n_vox}, "
                    f"centroid: ({cz_glob + z_offset:.1f}, {cy:.1f}, {cx:.1f}), "
                    f"{'\n'}"
                    f"Seed locations: {seed_voxels}, "
                    f"{'=' * 40}"
                    f"Grown locations: {grown_voxels}, "
                )

            # end region DEBUGGING"""

            out.append(
                Candidate(
                    n_voxels=n_vox,
                    n_seed=n_seed,
                    z_min=z_min_loc + r0 + z_offset,
                    z_max=z_max_loc + r0 + z_offset,
                    z_extent=z_extent,
                    y_extent=int(y_extent),
                    x_extent=int(x_extent),
                    fill=fill,
                    centroid=(cz_glob + z_offset, float(cy), float(cx)),
                    snr_cluster=snr_cluster,
                    r_peak=r_peak,
                    r_peak_ratio=r_peak_ratio,
                    peak_offset=peak_offset,
                    snr_peak=snr_peak,
                    linearity=linearity,
                    planarity=planarity,
                    sphericity=sphericity,
                    axis_z=axis_z,
                    normal_z=normal_z,
                    edge_contrast=edge_contrast,
                    decay_drop=decay_drop,
                    seed_grown_ratio=seed_grown_ratio,
                )
            )

    return out
