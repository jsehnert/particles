from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

FloatArray: TypeAlias = NDArray[np.float32]


class ResidualSource(Protocol):
    """Lazy provider of residual + noise for a z-range. Backed by memmap,
    HDF5, zarr, or on-the-fly background subtraction — caller's choice.

    PARALLEL CONTRACT: an instance is pickled once per slab task and re-opened
    inside each worker process. It MUST pickle to lightweight metadata only
    (filename, dtype, shape, offset) — never the array bytes. See
    ``MemmapResidualSource`` for the pattern: drop any open handle in
    ``__getstate__`` and re-open lazily. A source that pickles the array by
    value will copy the whole volume into every task and defeat the point."""

    shape: tuple[int, int, int]  # (Z, H, W)

    def read(self, z0: int, z1: int) -> tuple[FloatArray, FloatArray]:
        """Return (R, sigma_R) for slices [z0, z1).
        sigma_R shaped (z1-z0,1,1), (z1-z0,H,W), or broadcastable."""
        ...


class MemmapResidualSource:
    """Picklable, process-safe residual source backed by an on-disk memmap.

    TEMPLATE — replace ``read`` with your real background subtraction / noise
    model. The important part for parallelism is the pickling behaviour:
    ``__getstate__`` drops the open memmap so only the path + metadata cross the
    process boundary, and ``_vol`` re-opens lazily on first access in the worker.

    A ``np.memmap`` is an ndarray subclass and pickles its *data buffer* by
    default; storing one directly as an attribute and letting it pickle would
    serialize the entire volume per task. This class avoids that.
    """

    def __init__(
        self,
        path: str,
        shape: tuple[int, int, int],
        dtype: np.dtype = np.dtype(np.float32),
        sigma: float = 10.0,
    ):
        self.path = path
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.sigma = sigma
        self._vol: Optional[np.memmap] = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_vol"] = None  # never ship the open handle / buffer
        return state

    @property
    def vol(self) -> np.memmap:
        if self._vol is None:
            self._vol = np.memmap(
                self.path, dtype=self.dtype, mode="r", shape=self.shape
            )
        return self._vol

    def read(self, z0: int, z1: int) -> tuple[FloatArray, FloatArray]:
        # Replace this body with your background subtraction + noise estimation.
        # Keep the return contract: R over [z0, z1) and a broadcastable sigma_R
        # aligned to the SAME z-range (the read range incl. halo, not the core).
        R = np.asarray(self.vol[z0:z1], dtype=np.float32)
        sigma_R = np.full((z1 - z0, 1, 1), self.sigma, dtype=np.float32)
        return R, sigma_R


class PrecomputedResidualSource:
    """Stage-2 source: a trivial slicer over a precomputed residual memmap.

    This is what the parallel detector reads. It carries no volume and no stats in
    its ``__dict__`` — only a path, a shape, and a per-slice ``sigma`` vector of
    length Z — so it pickles to a few hundred bytes plus the sigma array and is
    inherently process-safe. Each worker re-opens the residual memmap read-only and
    lazily; the OS page cache shares the file across workers at ~zero per-worker
    cost. No background model runs here, so there is no recompute over slab halos —
    the halo only re-reads cheap slices.

    The residual is stored on disk as int16 (~18.5 GB for the full volume vs ~37 GB
    as float32) and cast up to float32 on read, since the detector works in float
    SNR. This is lossless: the residual is a difference of two uint8 values
    (raw voxel minus uint8 median baseline), so every value is an exact integer in
    [-255, 255], comfortably inside int16's range. That bound assumes the residual
    is the raw integer difference with metal regions zeroed — NOT scaled,
    normalized, or made fractional before storage. If a future ``src`` produces a
    residual outside +/-255 or with a fractional part, store float32 instead
    (pass ``dtype=np.float32``).
    """

    def __init__(
        self,
        residual_path: str,
        shape: tuple[int, int, int],
        sigma: NDArray[np.float32],
        dtype: np.dtype = np.dtype(np.int16),
    ):
        self.residual_path = residual_path
        self.shape = shape
        self.sigma = np.asarray(sigma, dtype=np.float32)  # (Z,), per-slice noise
        self.dtype = np.dtype(dtype)
        self._vol: np.memmap | None = None
        assert self.sigma.shape == (shape[0],), (
            f"sigma must be per-slice length Z={shape[0]}, got {self.sigma.shape}"
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_vol"] = None  # drop the open handle; re-open in the worker
        return state

    def free_volume(self) -> None:
        """Close the memmap handle to free the OS page cache. The residual is
        re-opened lazily on next access."""
        if self._vol is not None:
            del self._vol
            self._vol = None
            import gc

            gc.collect()

    @property
    def vol(self) -> np.memmap:
        if self._vol is None:
            self._vol = np.memmap(
                self.residual_path, dtype=self.dtype, mode="r", shape=self.shape
            )
        return self._vol

    def read(self, z0: int, z1: int) -> tuple[FloatArray, FloatArray]:
        # stored int16 -> float32 for the SNR division the detector performs
        R = np.asarray(self.vol[z0:z1], dtype=np.float32)
        sigma_R = self.sigma[z0:z1][:, None, None]  # (z1-z0, 1, 1)
        return R, sigma_R


def precompute_residual(
    src: ResidualSource,
    residual_path: str,
    sigma_path: str,
    chunk: int = 256,
    verbose: bool = True,
    residual_dtype: np.dtype = np.dtype(np.int16),
    flush_every: int = 0,
) -> PrecomputedResidualSource:
    """Stage 1: materialize the residual + per-slice sigma to disk, once, serially.

    Walks the analysis volume in NON-OVERLAPPING chunks and calls ``src.read`` on
    each, writing the returned residual into a memmap and the per-slice sigma into a
    companion ``.npy``. Non-overlapping is correct here precisely because the
    residual for a slice does not depend on the chunk it is computed in (the baseline
    window is drawn from the full volume inside ``src``), so every slice is computed
    exactly once — no halo, no redundant baseline passes.

    Runs in the parent process, so ``src``'s own internal threading (if any) gets the
    whole machine with no process-level oversubscription. The expensive
    background/noise work happens here and only here; stage-2 detection then reads
    cheap slices from ``residual_path``.

    The residual is stored as int16 by default (~18.5 GB full-volume vs ~37 GB
    float32), which is lossless when the residual is the integer difference of two
    uint8 volumes (see ``PrecomputedResidualSource``). Every value is then an exact
    integer in [-255, 255], comfortably inside int16 range.

    Overflow/truncation validation is applied ONLY when ``src`` returns a dtype that
    must actually be converted to fit ``residual_dtype`` (e.g. a float or wider-int
    source into an int store). When ``src`` already returns the storage dtype — the
    normal int16 path — the store is an exact copy and cannot overflow, so the
    validation (two full-chunk reductions for the range check, plus an integer-valued
    check) is skipped entirely rather than paid on every chunk. A genuinely lossy
    conversion still fails loudly rather than silently truncating; pass
    ``residual_dtype=np.float32`` to keep a fractional residual.

    Args:
        src: your residual provider (e.g. ``GlobalData``). Used only in-process; it is
            never pickled or shipped to a worker.
        residual_path: output path for the ``(Z, H, W)`` residual memmap.
        sigma_path: output path (``.npy``) for the ``(Z,)`` per-slice sigma vector.
        chunk: slices per write pass. Larger = fewer ``src.read`` calls (and, for a
            source whose masking is computed per chunk, fewer mask-boundary seams);
            bounded by the RAM for one ``(chunk, H, W)`` block.
        verbose: if True, print per-chunk progress.
        residual_dtype: on-disk residual dtype. int16 (default) for the integer
            uint8-difference residual; float32 to preserve fractional residuals.
        flush_every: memmap flush cadence, in chunks. ``0`` (default) flushes only
            once at the end (in ``finally``), letting the OS write dirty pages back in
            the background so disk I/O overlaps the next chunk's compute instead of
            blocking on a synchronous ``msync`` every iteration. Set to a positive
            ``N`` to force a flush every ``N`` chunks, bounding peak dirty-page memory
            to ~``N`` chunks' worth if the background write-back can't keep up and RAM
            pressure appears late in a long run. Durability at completion is
            guaranteed by the final flush regardless of this setting.

    Returns:
        A ready ``PrecomputedResidualSource`` over the written files, to hand to
        ``detect_candidates_parallel``.
    """
    rdt = np.dtype(residual_dtype)
    Z, H, W = src.shape
    resid = np.memmap(residual_path, dtype=rdt, mode="w+", shape=(Z, H, W))
    sigma = np.zeros(Z, dtype=np.float32)

    is_int = np.issubdtype(rdt, np.integer)
    lo, hi = (np.iinfo(rdt).min, np.iinfo(rdt).max) if is_int else (None, None)

    try:
        for k, c0 in enumerate(range(0, Z, chunk)):
            c1 = min(c0 + chunk, Z)
            R, sig = src.read(c0, c1)
            R = np.asarray(R)

            # Validate only on a real dtype conversion; an int16->int16 store is exact
            # and cannot overflow, so skip the full-chunk reductions on that path.
            if is_int and R.dtype != rdt:
                if not np.can_cast(R.dtype, rdt, casting="same_kind"):
                    if np.any(R != np.rint(R)):
                        raise ValueError(
                            f"residual has non-integer values; cannot store as {rdt} "
                            "without loss (pass residual_dtype=np.float32)"
                        )
                rmin, rmax = float(R.min()), float(R.max())
                if rmin < lo or rmax > hi:
                    raise ValueError(
                        f"residual range [{rmin}, {rmax}] exceeds {rdt} [{lo}, {hi}]"
                    )

            resid[c0:c1] = R
            sigma[c0:c1] = np.asarray(sig, dtype=np.float32).reshape(-1)

            # Background write-back overlaps the next chunk's compute; force a
            # synchronous flush only when bounding dirty-page memory is needed.
            if flush_every and (k + 1) % flush_every == 0:
                resid.flush()

            if verbose:
                print(f"    precompute z=[{c0}, {c1})  ({c1}/{Z})")
    finally:
        resid.flush()

    np.save(sigma_path, sigma)
    del resid  # close the w+ handle before anything re-opens read-only

    return PrecomputedResidualSource(
        residual_path=residual_path, shape=(Z, H, W), sigma=sigma, dtype=rdt
    )


@dataclass
class Candidate:
    n_voxels: int  # grown component size (k_low)
    n_seed: int  # high-threshold core size (k_high) — detection basis
    n_seed_regions: int  # number of connected seed regions in the grown component
    n_seed_total: int  # total number of seed voxels in the grown component
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
    fill_pca: float  # fill fraction of the PCA-aligned ellipsoid
    diag: float  # diagonal length of the PCA-aligned ellipsoid
    radial_pos: float  # radial position from the center of the volume


@dataclass(frozen=True)
class DetectParams:
    """All scalar detection parameters, bundled so one small picklable object
    crosses the process boundary per task instead of a long kwargs tail."""

    k_high: float = 4.0
    k_low: float = 2.99
    min_voxels: int = 2
    z_extent_max: int = 10
    small_z_bounds: tuple[int, int] = (1, 4)
    small_voxel_cutoff: int = 5
    aniso_factor: float = 1.0
    z_pad: int = 2
    min_fill: float = 0.15
    z_offset: int = 0


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
) -> tuple[float, float, float, float, float, float, float, float]:
    """Rotation-invariant shape triple plus orientation of a voxel component.

    Returns ``(linearity, planarity, sphericity, axis_z, normal_z, l1, l2, l3)``, all from the
    Sheppard-corrected covariance of the voxel coordinates.
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

    return linearity, planarity, sphericity, axis_z, normal_z, l1, l2, l3


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
    falloff across the grown blob's surface, both in sigma units.
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
    d2 = ndimage.binary_dilation(d1, structure, iterations=1)
    dR = d2  # ndimage.binary_dilation(comp, structure, iterations=shell_radius)
    for _ in range(shell_radius - 2):
        dR = ndimage.binary_dilation(dR, structure, iterations=1)

    inner_edge = comp & ~ndimage.binary_erosion(comp, structure)
    ring1 = (d1 & ~comp) & nonzero  # pinned isosurface ring
    ring2 = (d2 & ~d1) & nonzero
    baseline = (dR & ~d1) & nonzero  # distance 2..shell_radius

    # edge_contrast: inner-edge level minus outer-baseline level
    if baseline.any():
        edge_contrast = float(np.median(snr_p[inner_edge]) - np.median(snr_p[baseline]))
    else:
        edge_contrast = float("nan")
    # decay_drop: shape of falloff, brightness-independent
    if ring1.any() and ring2.any():
        m1 = float(np.median(snr_p[ring1]))
        m2 = float(np.median(snr_p[ring2]))
        decay_drop = float(m1 - m2)
    else:
        decay_drop = float("nan")

    return edge_contrast, decay_drop


def _process_slab(
    src: ResidualSource,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    p: DetectParams,
) -> list[Candidate]:
    """Detect + measure every owned candidate in a single slab.

    Pure function of ``(src, slab bounds, params)`` — no shared state, no cross-slab
    coupling. This is the unit of parallel work AND the body the serial driver calls,
    so serial and parallel runs are guaranteed to produce the same candidate set
    (dedup is by centroid ownership into the disjoint core ``[c0, c1)``, which is
    independent of execution order).

    ``structure`` is rebuilt here rather than passed in so nothing non-trivial has to
    pickle; ``generate_binary_structure`` is effectively free.
    """
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    z_lo, z_hi_small = p.small_z_bounds

    R, sigma = src.read(r0, r1)  # (r1-r0, H, W)
    _, H, W = src.shape
    snr = R / sigma

    seed_high = snr >= p.k_high  # detection seeds
    grow_low = snr >= p.k_low  # skirt mask (k_low < k_high)

    # hysteresis: grow seeds into connected low-threshold voxels
    # grown = ndimage.binary_propagation(seed_high, mask=grow_low, structure=structure)
    low_labels, _ = ndimage.label(grow_low, structure=structure)
    seeded = np.zeros(low_labels.max() + 1, dtype=bool)
    seeded[low_labels[seed_high]] = True
    seeded[0] = False
    grown = seeded[low_labels]  # mask of grown components

    labels, n = ndimage.label(grown, structure=structure)
    if n == 0:
        return []

    ids = np.arange(1, n + 1)
    objs = ndimage.find_objects(labels)

    out: list[Candidate] = []
    for i, lid in enumerate(ids):
        sl = objs[i]  # bbox of grown component

        # bbox-local masks: grown blob, and its high-threshold core
        sub_grown = labels[sl] == lid
        sub_seed = sub_grown & seed_high[sl]
        n_seed_total = int(sub_seed.sum())  # total seed voxels in the grown blob
        if n_seed_total < p.min_voxels:
            continue  # defensive; propagation guarantees >=1 seed voxel

        # --- detection criterion: a CONNECTED high core of >= min_voxels ---
        seed_lbl, _ = ndimage.label(sub_seed, structure=structure)
        if seed_lbl.max() == 0:
            continue  # defensive; propagation guarantees >=1 seed voxel
        n_seed = int(np.bincount(seed_lbl.ravel())[1:].max())
        if n_seed < p.min_voxels:  # p^min_voxels logic intact at k_high
            continue

        # n_seed_total = int(sub_seed.sum())  # total seed voxels in the grown blob
        n_seed_regions = int(seed_lbl.max())  # number of connected seed regions

        n_vox = int(sub_grown.sum())  # grown size, for features

        # extents from the GROWN bbox (full physical footprint incl. skirt)
        z_min_loc, z_max_loc = sl[0].start, sl[0].stop - 1
        z_extent = z_max_loc - z_min_loc + 1
        y_extent = sl[1].stop - sl[1].start
        x_extent = sl[2].stop - sl[2].start
        fill = n_vox / (z_extent * y_extent * x_extent)  # fill fraction of the bbox

        # axial-extent gate (hard cap + two-regime transience), on grown extent
        if z_extent > p.z_extent_max:
            continue
        if not _axial_ok(
            z_extent=z_extent,
            y_extent=y_extent,
            x_extent=x_extent,
            n_vox=n_vox,
            fill=fill,
            z_lo=z_lo,
            z_hi_small=z_hi_small,
            small_voxel_cutoff=p.small_voxel_cutoff,
            aniso_factor=p.aniso_factor,
            z_pad=p.z_pad,
            min_fill=p.min_fill,
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

        linearity, planarity, sphericity, axis_z, normal_z, l1, l2, l3 = (
            _eigen_features_with_axes(sub_grown)
        )

        # More features extracted from the eigenvalues
        fill_pca: float = float("nan")
        if l2 > 0 and l3 > 0:
            den = 4 * np.pi * (3**1.5) * np.sqrt(l1 * l2 * l3)
            fill_pca = n_vox / den  # PCA-based fill fraction of ellipsoid

        diag = 2 * np.sqrt(3 * l1) / max(z_extent, x_extent, y_extent)

        edge_contrast, decay_drop = _shell_features(
            labels=labels,
            snr=snr,
            R=R,
            k_sigma=p.k_low,
            sl=sl,
            lid=lid,
            structure=structure,
            shell_radius=3,
        )

        radial_pos = np.sqrt((cy - H / 2) ** 2 + (cx - W / 2) ** 2)

        # --- seed-to-grown intensity concentration ---
        S_grown = float(R_sub[sub_grown].sum())
        S_seed = float(R_sub[sub_seed].sum())
        seed_grown_ratio = S_seed / S_grown  # (0, 1]; S_grown > 0 by construction

        out.append(
            Candidate(
                n_voxels=n_vox,
                n_seed=n_seed,
                n_seed_regions=n_seed_regions,
                n_seed_total=n_seed_total,
                z_min=z_min_loc + r0 + p.z_offset,
                z_max=z_max_loc + r0 + p.z_offset,
                z_extent=z_extent,
                y_extent=int(y_extent),
                x_extent=int(x_extent),
                fill=fill,
                centroid=(cz_glob + p.z_offset, float(cy), float(cx)),
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
                fill_pca=fill_pca,
                diag=diag,
                radial_pos=radial_pos,
            )
        )

    return out


def detect_candidates_streaming(
    src: ResidualSource,
    k_high: float = 4.0,
    k_low: float = 2.99,
    min_voxels: int = 2,
    z_extent_max: int = 10,
    small_z_bounds: tuple[int, int] = (1, 4),
    small_voxel_cutoff: int = 5,
    aniso_factor: float = 1.0,
    z_pad: int = 2,
    min_fill: float = 0.15,
    slices_per_chunk: int = 128,
    z_offset: int = 0,
    verbose: bool = True,
) -> list[Candidate]:
    """Serial driver — unchanged behaviour, now a thin loop over ``_process_slab``.

    Kept as the single-worker reference and the baseline to diff the parallel run
    against (sort both by centroid first; only ordering differs).
    """
    Z, _, _ = src.shape
    halo = z_extent_max
    p = DetectParams(
        k_high=k_high,
        k_low=k_low,
        min_voxels=min_voxels,
        z_extent_max=z_extent_max,
        small_z_bounds=small_z_bounds,
        small_voxel_cutoff=small_voxel_cutoff,
        aniso_factor=aniso_factor,
        z_pad=z_pad,
        min_fill=min_fill,
        z_offset=z_offset,
    )

    out: list[Candidate] = []
    for r0, r1, c0, c1 in iter_slabs(Z, slices_per_chunk, halo):
        found = _process_slab(src, r0, r1, c0, c1, p)
        if verbose:
            print(
                f"    chunk z=[{c0}, {c1}) read z=[{r0}, {r1}): "
                f"+{len(found)} (total {len(out) + len(found)})"
            )
        out.extend(found)
    return out


def detect_candidates_parallel(
    src: ResidualSource,
    k_high: float = 4.0,
    k_low: float = 2.99,
    min_voxels: int = 2,
    z_extent_max: int = 10,
    small_z_bounds: tuple[int, int] = (1, 4),
    small_voxel_cutoff: int = 5,
    aniso_factor: float = 1.0,
    z_pad: int = 2,
    min_fill: float = 0.15,
    slices_per_chunk: int = 128,
    z_offset: int = 0,
    n_workers: int = 3,
    verbose: bool = True,
) -> list[Candidate]:
    """Process-parallel driver. Dispatches each slab to a loky worker and
    concatenates the per-slab candidate lists.

    Identical candidate set to ``detect_candidates_streaming`` for the same
    parameters — dedup is per-slab centroid ownership into disjoint cores, so the
    result does not depend on execution order. ONLY the ordering of the returned
    list differs (completion order, not slab order); sort by ``centroid`` before
    diffing against a serial baseline or a lo-invariance check.

    Memory: each worker holds one read-range slab, ~= (slices_per_chunk +
    2*z_extent_max) slices of derived arrays (R, snr float32; three bool masks;
    int32 labels; + scipy temporaries) ~= 38 MB/slice at 1425^2. The read-only
    input memmap is shared via the OS page cache and costs ~nothing per worker,
    PROVIDED ``src`` pickles to metadata only (see ResidualSource contract /
    MemmapResidualSource). At 1425^2, halo 16: chunk 128 -> ~6.4 GB/worker.

    Set ``n_workers`` <= physical cores with the parent idle. Falls back to the
    serial driver when ``n_workers <= 1``.
    """
    if n_workers <= 1:
        return detect_candidates_streaming(
            src,
            k_high=k_high,
            k_low=k_low,
            min_voxels=min_voxels,
            z_extent_max=z_extent_max,
            small_z_bounds=small_z_bounds,
            small_voxel_cutoff=small_voxel_cutoff,
            aniso_factor=aniso_factor,
            z_pad=z_pad,
            min_fill=min_fill,
            slices_per_chunk=slices_per_chunk,
            z_offset=z_offset,
            verbose=verbose,
        )

    from joblib import Parallel, delayed

    Z, _, _ = src.shape
    halo = z_extent_max
    p = DetectParams(
        k_high=k_high,
        k_low=k_low,
        min_voxels=min_voxels,
        z_extent_max=z_extent_max,
        small_z_bounds=small_z_bounds,
        small_voxel_cutoff=small_voxel_cutoff,
        aniso_factor=aniso_factor,
        z_pad=z_pad,
        min_fill=min_fill,
        z_offset=z_offset,
    )

    slabs = list(iter_slabs(Z, slices_per_chunk, halo))

    # backend="loky": process-based (the per-component loop is GIL-bound, so
    # threads would not help). Dynamic dispatch balances the ~Z/chunk tasks over
    # the workers; the last partial slab's imbalance is negligible at this count.
    results: list[list[Candidate]] = Parallel(
        n_jobs=n_workers, backend="loky", verbose=10 if verbose else 0
    )(delayed(_process_slab)(src, r0, r1, c0, c1, p) for (r0, r1, c0, c1) in slabs)

    out: list[Candidate] = []
    for chunk_result in results:
        out.extend(chunk_result)
    return out
