from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Protocol, TypeAlias

import cv2
import numpy as np
from circle_fit import taubinSVD
from numpy.typing import NDArray
from scipy import ndimage

from metal import extract_metal_mask

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

    # OPTIONAL grayscale capability (duck-typed, not required by the protocol):
    # a source MAY additionally provide
    #     read_grayscale(z0, z1) -> NDArray[np.uint8] | None
    # returning the RAW grayscale slab aligned to the same analysis-frame z-range
    # as ``read`` (shape (z1-z0, H, W)), or None when unavailable. Detection
    # computes grayscale features and the per-candidate voxel dump only when a
    # non-None grayscale slab is supplied; otherwise behaviour (and the candidate
    # set) is bit-identical to a residual-only source.


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
        grayscale_path: str | None = None,
        grayscale_shape: tuple[int, int, int] | None = None,
        grayscale_dtype: np.dtype = np.dtype(np.uint8),
        grayscale_z_offset: int = 0,
    ):
        """``grayscale_*``: optional spec of the RAW grayscale volume file (the
        original reconstruction on disk), enabling ``read_grayscale`` for
        grayscale features / the voxel dump during detection. Metadata only —
        the file is opened lazily and read-only in each worker (page-cache
        shared), exactly like the residual. ``grayscale_z_offset`` maps the
        analysis frame to the raw volume frame (i.e. ``z_min``): analysis slice
        ``n`` is raw slice ``n + grayscale_z_offset``. When ``grayscale_path``
        is None the source is residual-only and detection behaviour is
        unchanged."""
        self.residual_path = residual_path
        self.shape = shape
        self.sigma = np.asarray(sigma, dtype=np.float32)  # (Z,), per-slice noise
        self.dtype = np.dtype(dtype)
        self._vol: np.memmap | None = None
        self.grayscale_path = grayscale_path
        self.grayscale_shape = grayscale_shape
        self.grayscale_dtype = np.dtype(grayscale_dtype)
        self.grayscale_z_offset = int(grayscale_z_offset)
        self._gs: np.memmap | None = None
        assert self.sigma.shape == (shape[0],), (
            f"sigma must be per-slice length Z={shape[0]}, got {self.sigma.shape}"
        )
        if grayscale_path is not None:
            assert grayscale_shape is not None, (
                "grayscale_shape (full raw volume shape) is required with "
                "grayscale_path"
            )
            gz = grayscale_shape[0]
            assert self.grayscale_z_offset + shape[0] <= gz, (
                f"analysis range [{self.grayscale_z_offset}, "
                f"{self.grayscale_z_offset + shape[0]}) exceeds raw volume Z={gz}"
            )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_vol"] = None  # drop the open handle; re-open in the worker
        state["_gs"] = None
        return state

    def free_volume(self) -> None:
        """Close the memmap handles to free the OS page cache. Both are
        re-opened lazily on next access."""
        if self._vol is not None:
            del self._vol
            self._vol = None
        if self._gs is not None:
            del self._gs
            self._gs = None
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

    @property
    def gs_vol(self) -> np.memmap | None:
        if self.grayscale_path is None:
            return None
        if self._gs is None:
            self._gs = np.memmap(
                self.grayscale_path,
                dtype=self.grayscale_dtype,
                mode="r",
                shape=self.grayscale_shape,
            )
        return self._gs

    def read_grayscale(self, z0: int, z1: int) -> NDArray[np.uint8] | None:
        """Raw grayscale slab for analysis-frame slices [z0, z1), or None when no
        grayscale volume was configured. Aligned to the same z-range as ``read``:
        analysis slice n maps to raw slice n + grayscale_z_offset."""
        gs = self.gs_vol
        if gs is None:
            return None
        g0 = z0 + self.grayscale_z_offset
        g1 = z1 + self.grayscale_z_offset
        return np.asarray(gs[g0:g1])


def precompute_residual(
    src: ResidualSource,
    residual_path: str,
    sigma_path: str,
    chunk: int = 256,
    verbose: bool = True,
    residual_dtype: np.dtype = np.dtype(np.int16),
    flush_every: int = 0,
    grayscale_path: str | None = None,
    grayscale_shape: tuple[int, int, int] | None = None,
    grayscale_dtype: np.dtype = np.dtype(np.uint8),
    grayscale_z_offset: int = 0,
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
        grayscale_path: optional path to the RAW grayscale volume file on disk.
            When given, the returned source exposes ``read_grayscale`` so stage-2
            detection can compute grayscale features and the per-candidate voxel
            dump. Nothing is copied or stored — this is metadata pointing at the
            file that already exists; workers open it read-only and lazily.
        grayscale_shape: full (Z, H, W) of the raw volume file (required with
            ``grayscale_path``; the raw file is headerless so shape must be
            supplied, exactly as for the residual memmap).
        grayscale_dtype: raw volume dtype (default uint8).
        grayscale_z_offset: maps the analysis frame to the raw frame; pass the
            same ``z_min`` used to slice ``vol_for_analysis`` from the raw volume.

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
        residual_path=residual_path,
        shape=(Z, H, W),
        sigma=sigma,
        dtype=rdt,
        grayscale_path=grayscale_path,
        grayscale_shape=grayscale_shape,
        grayscale_dtype=grayscale_dtype,
        grayscale_z_offset=grayscale_z_offset,
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
    # worst-case (max over seed regions) distance from a seed region's own
    # intensity-weighted centroid to the grown centroid -- see the
    # computation site in _process_slab for why MAX, not an average over all
    # seed voxels combined (the latter can hide exactly the "did detection
    # merge two nearby objects" signal this exists to catch). For
    # n_seed_regions == 1, reduces to that single region's offset.
    seed_offset: float
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
    radial_pos: float  # normalized radial position (fraction of the fitted can
    # radius; 0 = jelly-roll axis, ~1 = can wall, may exceed 1 at/beyond it —
    # unclamped). Falls back to the raw volume-centre voxel distance when no can
    # circle is available.
    # --- integer peak voxel, global frame: the exact, float-free join key that
    # links this row to its voxel-dump records (with volume_name; experiment is
    # implied by the output folder, matching the candidates.csv convention) ---
    peak_z: int = -1
    peak_y: int = -1
    peak_x: int = -1
    # --- grayscale features (NaN when the source provides no grayscale) ---
    gs_median: float = float("nan")  # median raw value over the grown mask
    gs_p90: float = float("nan")  # 90th-percentile raw value over the grown mask
    gs_peak: float = float("nan")  # max raw value over the grown mask
    gs_shell_median: float = float("nan")  # median raw value of the outer shell
    gs_contrast: float = float(
        "nan"
    )  # inner-edge median minus shell median (raw units)
    # grayscale interquartile range (p75-p25) over the grown mask -- texture/
    # heterogeneity of the raw attenuation LEVEL, distinct from gs_median/
    # gs_p90/gs_peak (which only capture the level itself): a compositionally
    # uniform metal particle should read more homogeneously bright than a
    # partial-volume/reconstruction artifact. NaN under the same condition
    # gs_median is (no grayscale slab available).
    gs_iqr: float = float("nan")
    # metal-threshold surrogate used for the grayscale shell exclusion, echoed per
    # row so the feature table is self-describing for offline recomputation
    metal_threshold: float = float("nan")
    # PCA eigenvalues of the grown component's voxel coordinates (Sheppard-
    # corrected covariance; see _eigen_features_with_axes), largest to
    # smallest, in voxel^2 units -- the ABSOLUTE scale that linearity/
    # planarity/sphericity (ratios of these) discard by construction. Already
    # computed in _process_slab for fill_pca/diag; kept here since retaining
    # them costs nothing further. Needs a voxel_size**2 unit conversion before
    # use across cell formats -- see ml_classifier/features.py.
    l1: float = float("nan")
    l2: float = float("nan")
    l3: float = float("nan")
    # surface-to-volume compactness: (boundary voxel count) / n_voxels, where
    # boundary = grown voxels with at least one non-grown 6-neighbor. A
    # complementary compactness signal to fill_pca (which measures fill of the
    # best-fit ellipsoid -- a GLOBAL regularity measure): this instead catches
    # voxel-level surface roughness/jaggedness a smooth continuous ellipsoid
    # fit can miss. Dimensionless (ratio of voxel counts) -- no unit
    # conversion needed across cell formats.
    surface_ratio: float = float("nan")
    # surrogate DB key, assigned by experiment_db.ExperimentRecorder.on_slab_result
    # as each slab's candidates are persisted -- not set (-1) for callers that
    # don't wire a DB recorder in (e.g. the aniso_pilot/min_fill_pilot CSV path).
    candidate_id: int = -1


@dataclass(frozen=True)
class DetectParams:
    """All scalar detection parameters, bundled so one small picklable object
    crosses the process boundary per task instead of a long kwargs tail."""

    k_high: float = 4.0
    k_low: float = 2.99
    # Minimum size (voxels) of the largest CONNECTED above-k_high component (n_seed)
    # for a blob to be admitted. This is the noise-collapse floor, not a particle-size
    # or false-positive-budget knob: the expected pure-noise candidate count scales as
    # ~p^min_seed_voxels, so this is the smallest value that keeps the candidate set from
    # being noise-dominated. Detection is deliberately sensitivity-first; genuine FP
    # rejection is the classifier's job. (Config key + logged column: min_seed_voxels;
    # formerly the config key vol_threshold / arg min_voxels.)
    min_seed_voxels: int = 2
    z_extent_max: int = 10
    small_z_bounds: tuple[int, int] = (1, 4)
    small_voxel_cutoff: int = 5
    aniso_factor: float = 1.0
    # Additive slack (VOXELS) in the anisotropy gate: z_extent > aniso_factor*lat_min + z_pad.
    # Deliberately voxel-native, NOT a physical length — do not scale by voxel_size. The gate
    # itself is already resolution-invariant (aniso_factor is a ratio; z_extent and lat_min
    # scale together), and z_pad only softens the integer round-off at the boundary, which is
    # a +/-1-voxel effect at any resolution. Physicalizing it would shrink it to sub-voxel at
    # coarse resolution — removing the guard exactly where quantization is worst (few-voxel
    # particles). Fixed voxel count keeps the slack matched to the quantization granularity.
    z_pad: int = 2
    min_fill: float = 0.15
    # Master switch for the shape-discrimination gate ``_axial_ok`` (anisotropy,
    # min_fill, the small-regime routing, and z_lo — everything EXCEPT the two
    # load-bearing keepers, min_seed_voxels and z_extent_max). When False the whole
    # gate is skipped and detection is recall-first: every seed-validated blob within
    # the z_extent_max cap is emitted, and shape discrimination is deferred to the
    # downstream classifier (which already receives all the shape features). All the
    # shape params above remain configured but inert. Default True preserves the
    # historical enforcing behaviour for direct callers (e.g. the aniso/min_fill
    # pilots that study those gates); config sets it False for research runs.
    enforce_shape_gates: bool = True
    z_offset: int = 0
    # metal-threshold SURROGATE (volume-level scalar, e.g. GlobalData.metal_threshold).
    # Used to exclude bright metal from grayscale shell statistics via
    # ``grayscale > metal_threshold`` — an approximation of the real per-slice
    # hysteresis mask (which is not carried into stage 2), chosen deliberately so
    # live features and offline recomputation from the voxel dump apply the
    # IDENTICAL rule. NaN disables the surrogate (R != 0 exclusion still applies).
    metal_threshold: float = float("nan")
    # metal-mask parameters for the per-slab can-circle fit (radial_pos): passed
    # straight to ``extract_metal_mask`` on the slab's 9-slice max projection.
    # Match the volume's configured values (GlobalData.metal_min_area /
    # metal_grayscale_margin). Only used when a grayscale slab is available and
    # ``metal_threshold`` is finite; otherwise radial_pos falls back to the raw
    # volume-centre distance.
    metal_min_area: int = 100
    metal_grayscale_margin: int = 0
    # padding (voxels, per side) of the grown component's bbox for the voxel dump.
    # Must be >= the largest shell radius you ever want to recompute offline;
    # shells themselves are NOT stored — they are re-derived offline from the
    # in_grown geometry, so the dump stays radius/connectivity agnostic.
    storage_margin: int = 5
    # shell radius (voxels) for the edge_contrast / decay_drop / grayscale shell
    # features. Deliberately kept in DISCRETE VOXEL units, not converted to a
    # physical length: it is a morphological reach (number of dilation iterations)
    # whose meaning is tied to voxel-grid connectivity and the inter-slice
    # noise-correlation structure, so holding it constant in voxels is truer to its
    # purpose than holding it constant in microns. (Its absolute surround reach does
    # drift with resolution — a known, accepted trade-off, not an oversight.)
    # Exposed here instead of a bare function default so it is configurable and
    # travels with the other detection parameters.
    shell_radius: int = 3

    def __post_init__(self) -> None:
        # Offline shell recomputation reads from the voxel dump, so the dumped
        # bbox padding must cover every shell ring.
        if self.storage_margin < self.shell_radius:
            raise ValueError(
                f"storage_margin ({self.storage_margin}) must be >= shell_radius "
                f"({self.shell_radius}); the voxel dump must cover the shells for "
                "offline recomputation."
            )


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


def _grayscale_features(
    G: NDArray[np.uint8],
    R: FloatArray,
    labels: NDArray[np.integer],
    sl: tuple[slice, slice, slice],
    lid: int,
    structure: NDArray[np.bool_],
    metal_threshold: float,
    shell_radius: int = 3,
) -> tuple[float, float, float, float, float, float]:
    """Raw-grayscale level and lateral-surround contrast of a component.

    Returns ``(gs_median, gs_p90, gs_peak, gs_shell_median, gs_contrast, gs_iqr)``,
    all in raw grayscale units, recovering the two channels the residual discards
    by construction:

    gs_median / gs_p90 / gs_peak: absolute attenuation LEVEL over the grown mask.
        High-Z contaminants sit at high absolute values; blobs on/near the Cu
        anode foil read high-absolute even at modest residual SNR (the Cu-foil FP
        discriminator).
    gs_shell_median: absolute level of the outer surround shell (distance
        2..shell_radius), for context.
    gs_contrast: inner-edge median minus gs_shell_median. A real particle is a
        LATERAL local maximum, so gs_contrast >> 0; a cathode-winding artifact
        "surrounded by brighter pixels" produced its residual only against the
        z-median baseline and reads near zero or negative here (the cathode FP
        discriminator). The residual cannot express this because its baseline is
        axial, not lateral.
    gs_iqr: interquartile range (p75-p25) of raw grayscale over the grown mask --
        texture/heterogeneity of the attenuation LEVEL, distinct from the level
        statistics above: a compositionally uniform metal particle should read
        more homogeneously bright than a partial-volume/reconstruction artifact.

    Outer-shell exclusion: voxels with ``R == 0`` (metal-zeroed / out-of-support
    upstream, mirroring ``_shell_features``) and, when ``metal_threshold`` is
    finite, voxels with ``grayscale > metal_threshold`` (the volume-level
    SURROGATE for the per-slice hysteresis metal mask, which is not carried into
    stage 2). The surrogate is used both live and offline so the two computations
    agree exactly; it is deliberately not the real grown/hysteresis mask. The
    inner edge is grown mask, never metal, so no exclusion applies there.

    Shell values are NaN when the shells are empty after exclusion (ringed by
    metal / clipped by the slab edge) — read as 'no evidence', never a rejection.
    """
    psl = tuple(
        slice(max(0, s.start - shell_radius), min(dim, s.stop + shell_radius))
        for s, dim in zip(sl, G.shape)
    )
    comp = labels[psl] == lid
    G_p = G[psl]
    R_p = R[psl]

    grown_vals = G_p[comp]
    gs_median = float(np.median(grown_vals))
    gs_p90 = float(np.percentile(grown_vals, 90))
    gs_peak = float(grown_vals.max())
    q75, q25 = np.percentile(grown_vals, [75, 25])
    gs_iqr = float(q75 - q25)

    valid = R_p != 0.0  # exclude metal-zeroed / out-of-support, as _shell_features
    if np.isfinite(metal_threshold):
        valid &= G_p <= metal_threshold  # surrogate metal exclusion

    d1 = ndimage.binary_dilation(comp, structure, iterations=1)
    dR = d1
    for _ in range(shell_radius - 1):
        dR = ndimage.binary_dilation(dR, structure, iterations=1)

    inner_edge = comp & ~ndimage.binary_erosion(comp, structure)
    baseline = (dR & ~d1) & valid  # distance 2..shell_radius

    if baseline.any():
        gs_shell_median = float(np.median(G_p[baseline]))
        gs_contrast = float(np.median(G_p[inner_edge]) - gs_shell_median)
    else:
        gs_shell_median = float("nan")
        gs_contrast = float("nan")

    return gs_median, gs_p90, gs_peak, gs_shell_median, gs_contrast, gs_iqr


def _voxel_records(
    G: NDArray[np.uint8],
    R: FloatArray,
    labels: NDArray[np.integer],
    sl: tuple[slice, slice, slice],
    lid: int,
    margin: int,
    r0: int,
    z_offset: int,
    peak_key: tuple[int, int, int],
) -> NDArray[np.int32]:
    """Per-voxel dump of one candidate over its margin-dilated bbox.

    Returns an int32 array of shape (n_voxels_in_padded_bbox, 9) with columns
    ``(peak_z, peak_y, peak_x, z, y, x, residual, grayscale, in_grown)``:

    - peak_z/y/x: the candidate's integer peak voxel in the GLOBAL frame — the
      exact join key back to the feature table row (with volume_name).
    - z/y/x: this voxel's global coordinates (z includes z_offset, matching
      candidate coordinates).
    - residual: the int residual value (exact — the residual is integer-valued).
    - grayscale: the raw uint8 value.
    - in_grown: 1 if the voxel belongs to the grown component, else 0.

    The dump covers the bbox padded by ``margin`` on every side (clipped to the
    slab), so BOTH interior and surround statistics are reconstructable offline:
    shells are re-derived from the in_grown geometry with any radius <= margin
    and any connectivity, rather than being frozen at dump time. Offline
    exclusion rules reproduce the live ones from the stored columns alone
    (``residual != 0`` and ``grayscale > metal_threshold`` with the threshold
    echoed in the feature table).
    """
    psl = tuple(
        slice(max(0, s.start - margin), min(dim, s.stop + margin))
        for s, dim in zip(sl, G.shape)
    )
    comp = labels[psl] == lid
    nz = psl[0].stop - psl[0].start
    ny = psl[1].stop - psl[1].start
    nx = psl[2].stop - psl[2].start

    zz, yy, xx = np.indices((nz, ny, nx), dtype=np.int32)
    n = nz * ny * nx
    rec = np.empty((n, 9), dtype=np.int32)
    rec[:, 0] = peak_key[0]
    rec[:, 1] = peak_key[1]
    rec[:, 2] = peak_key[2]
    rec[:, 3] = zz.ravel() + (psl[0].start + r0 + z_offset)
    rec[:, 4] = yy.ravel() + psl[1].start
    rec[:, 5] = xx.ravel() + psl[2].start
    rec[:, 6] = np.rint(R[psl].ravel()).astype(np.int32)  # integer-valued residual
    rec[:, 7] = G[psl].ravel().astype(np.int32)
    rec[:, 8] = comp.ravel().astype(np.int32)
    return rec


def _slab_can_geometry(
    G: NDArray[np.uint8],
    metal_threshold: float,
    min_area: int,
    margin: int,
) -> tuple[float, float, float, NDArray[np.bool_]] | None:
    """Per-slab jelly-roll can geometry, in the (y=row, x=col) frame of the
    residual/grayscale arrays.

    The can is an axially-extruded cylinder, so a single fit characterises the
    whole slab. To be robust to any local dropout we max-project three slices each
    from the top, middle and bottom of the slab (nine in total), extract the metal
    mask with the volume's configured ``metal_threshold`` / ``min_area`` /
    ``margin``, and take the largest external contour. That contour gives both a
    Taubin circle fit ``(cx, cy, r)`` (for the normalized radial position) and its
    filled interior as a boolean support mask (for gating detection to inside the
    can).

    Returns ``(cx, cy, r, inside)`` where ``cx`` is the column (x/axis-2) centre,
    ``cy`` the row (y/axis-1) centre — matching the ``(cy, cx)`` centroid
    convention used downstream — and ``inside`` is an ``(H, W)`` bool mask, True
    strictly inside/on the outer contour. Returns ``None`` when no metal contour
    is found or ``metal_threshold`` is not finite (caller then skips the gate and
    falls back to the volume's geometric centre for radial_pos)."""
    if not np.isfinite(metal_threshold):
        return None
    nz = G.shape[0]
    mid = nz // 2
    # 3 slices each at top / middle / bottom of the slab (deduped for thin slabs)
    idx = sorted(
        {
            i
            for i in (0, 1, 2, mid - 1, mid, mid + 1, nz - 3, nz - 2, nz - 1)
            if 0 <= i < nz
        }
    )
    proj = G[idx].max(axis=0)
    return _can_geometry_from_projection(proj, metal_threshold, min_area, margin)


def _can_geometry_from_projection(
    proj: NDArray[np.uint8],
    metal_threshold: float,
    min_area: int,
    margin: int,
) -> tuple[float, float, float, NDArray[np.bool_]] | None:
    """Fit the can circle + interior mask from an already-computed max projection
    ``proj`` (H, W). Shared by live detection (``_slab_can_geometry``) and the
    offline radial backfill so both apply identical mask/contour/fit logic; see
    ``_slab_can_geometry`` for the return contract."""
    if not np.isfinite(metal_threshold):
        return None
    mm = extract_metal_mask(
        proj, int(metal_threshold), min_area=min_area, margin=margin
    )
    contours, _ = cv2.findContours(
        mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    outer = max(contours, key=cv2.contourArea)
    pts = outer.reshape(-1, 2)  # (N, 2) as (x, y)
    if len(pts) < 3:
        return None
    xc, yc, r, _ = taubinSVD(pts.astype(np.float64))
    inside = np.zeros(proj.shape, dtype=np.uint8)
    cv2.drawContours(inside, [outer], -1, 1, thickness=cv2.FILLED)
    return float(xc), float(yc), float(r), inside.astype(bool)


def _process_slab(
    src: ResidualSource,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    p: DetectParams,
    collect_voxels: bool = False,
) -> tuple[list[Candidate], NDArray[np.int32] | None]:
    """Detect + measure every owned candidate in a single slab.

    Pure function of ``(src, slab bounds, params)`` — no shared state, no cross-slab
    coupling. This is the unit of parallel work AND the body the serial driver calls,
    so serial and parallel runs are guaranteed to produce the same candidate set
    (dedup is by centroid ownership into the disjoint core ``[c0, c1)``, which is
    independent of execution order).

    Returns ``(candidates, voxel_records)``. Grayscale features are computed only
    when the source provides ``read_grayscale`` returning a non-None slab; the
    original candidate fields are bit-identical either way. ``voxel_records`` is a
    single int32 array of the concatenated per-candidate dumps (see
    ``_voxel_records``) when ``collect_voxels`` is True AND grayscale is
    available, else None.

    ``structure`` is rebuilt here rather than passed in so nothing non-trivial has to
    pickle; ``generate_binary_structure`` is effectively free.
    """
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    z_lo, z_hi_small = p.small_z_bounds

    R, sigma = src.read(r0, r1)  # (r1-r0, H, W)
    _, H, W = src.shape

    # optional grayscale slab, aligned to the same analysis z-range
    read_gs = getattr(src, "read_grayscale", None)
    G: NDArray[np.uint8] | None = read_gs(r0, r1) if read_gs is not None else None
    voxel_chunks: list[NDArray[np.int32]] = []

    # per-slab jelly-roll can geometry: the circle (cx, cy, r) normalizes the
    # radial position, and the filled outer contour ``inside`` gates detection to
    # the can interior. None when no grayscale slab / metal contour is available,
    # in which case the gate is skipped and radial_pos falls back to the raw
    # volume-centre distance.
    geom = (
        _slab_can_geometry(
            G, p.metal_threshold, p.metal_min_area, p.metal_grayscale_margin
        )
        if G is not None
        else None
    )
    if geom is not None:
        cxc, cyc, rc, inside = geom
        slab_circle: tuple[float, float, float] | None = (cxc, cyc, rc)
        # strict gate: no residual (hence no seeds/candidates) outside the can.
        # R is a fresh float32 array from ``read`` (not a memmap view), so this
        # in-place zeroing is safe and does not touch the stored residual.
        R[:, ~inside] = 0.0
    else:
        slab_circle = None

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
        return [], None

    ids = np.arange(1, n + 1)
    objs = ndimage.find_objects(labels)

    out: list[Candidate] = []
    for i, lid in enumerate(ids):
        sl = objs[i]  # bbox of grown component

        # bbox-local masks: grown blob, and its high-threshold core
        sub_grown = labels[sl] == lid
        sub_seed = sub_grown & seed_high[sl]
        n_seed_total = int(sub_seed.sum())  # total seed voxels in the grown blob
        if n_seed_total < p.min_seed_voxels:
            continue  # defensive; propagation guarantees >=1 seed voxel

        # --- detection criterion: a CONNECTED high core of >= min_seed_voxels ---
        seed_lbl, _ = ndimage.label(sub_seed, structure=structure)
        if seed_lbl.max() == 0:
            continue  # defensive; propagation guarantees >=1 seed voxel
        n_seed = int(np.bincount(seed_lbl.ravel())[1:].max())
        if n_seed < p.min_seed_voxels:  # p^min_seed_voxels logic intact at k_high
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

        # z_extent_max: load-bearing keeper (compute/halo envelope + streak cap),
        # always enforced regardless of enforce_shape_gates.
        if z_extent > p.z_extent_max:
            continue
        # _axial_ok bundles the shape-discrimination gates (anisotropy, min_fill,
        # small-regime routing, z_lo). Skipped entirely when enforce_shape_gates is
        # False (recall-first: defer shape discrimination to the classifier).
        if p.enforce_shape_gates and not _axial_ok(
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

        # seed_offset: worst-case (max) distance from any individual seed
        # region's own intensity-weighted centroid to the grown centroid.
        # Computed here, in the pre-offset bbox-local frame -- translation
        # cancels in a distance, so this is identical to computing it in the
        # global frame after the sl[*].start/r0 offsets below. Deliberately
        # the MAX over regions, not one centroid averaged over all seed
        # voxels combined: two seed regions symmetric about the grown
        # centroid would average back to ~0 and hide exactly the "did
        # detection merge two nearby objects" signal this feature exists to
        # catch -- max does not cancel that way. seed_lbl.max() >= 1 is
        # guaranteed here (checked defensively above), so this is always
        # well-defined, never NaN.
        seed_region_coms = ndimage.center_of_mass(
            R_sub, labels=seed_lbl, index=np.arange(1, seed_lbl.max() + 1)
        )
        seed_offset = float(
            max(
                np.sqrt((rz - cz_loc) ** 2 + (ry - cy) ** 2 + (rx - cx) ** 2)
                for rz, ry, rx in seed_region_coms
            )
        )

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

        # surface-to-volume compactness: boundary voxels (grown voxels with
        # >=1 non-grown 6-neighbor) / n_voxels. Complementary to fill_pca
        # (which is a GLOBAL best-fit-ellipsoid regularity measure) -- this
        # instead catches voxel-level surface roughness a smooth continuous
        # ellipsoid fit can miss. border_value=0 (default) is exactly right
        # here: sub_grown is already cropped to its own tight bbox, so a
        # voxel at the array edge correctly erodes away as a true surface
        # voxel of the shape, same as if it were surrounded by more padding.
        boundary = sub_grown & ~ndimage.binary_erosion(sub_grown, structure)
        surface_ratio = float(boundary.sum()) / n_vox

        edge_contrast, decay_drop = _shell_features(
            labels=labels,
            snr=snr,
            R=R,
            k_sigma=p.k_low,
            sl=sl,
            lid=lid,
            structure=structure,
            shell_radius=p.shell_radius,
        )

        # normalized radial position: fraction of the fitted can radius (0 = axis,
        # ~1 = wall). Not clamped — values >1 for features at/beyond the fitted
        # circle carry signal (see near-wall FP concentration) and the fitted
        # radius is the outer wall surface. Falls back to the raw volume-centre
        # voxel distance when no can circle could be fitted for this slab.
        if slab_circle is not None:
            cxc, cyc, rc = slab_circle
            radial_pos = (
                float(np.hypot(cy - cyc, cx - cxc) / rc)
                if rc > 0
                else float("nan")
            )
        else:
            radial_pos = float(np.sqrt((cy - H / 2) ** 2 + (cx - W / 2) ** 2))

        # --- seed-to-grown intensity concentration ---
        S_grown = float(R_sub[sub_grown].sum())
        S_seed = float(R_sub[sub_seed].sum())
        seed_grown_ratio = S_seed / S_grown  # (0, 1]; S_grown > 0 by construction

        # integer peak voxel in the GLOBAL frame — exact, float-free join key
        peak_z = int(pz + r0 + p.z_offset)
        peak_y = int(py)
        peak_x = int(px)

        # --- grayscale features (only when the source supplies a raw slab) ---
        gs_median = gs_p90 = gs_peak = gs_shell_median = gs_contrast = gs_iqr = float("nan")
        if G is not None:
            gs_median, gs_p90, gs_peak, gs_shell_median, gs_contrast, gs_iqr = (
                _grayscale_features(
                    G=G,
                    R=R,
                    labels=labels,
                    sl=sl,
                    lid=lid,
                    structure=structure,
                    metal_threshold=p.metal_threshold,
                    shell_radius=p.shell_radius,
                )
            )
            if collect_voxels:
                voxel_chunks.append(
                    _voxel_records(
                        G=G,
                        R=R,
                        labels=labels,
                        sl=sl,
                        lid=lid,
                        margin=p.storage_margin,
                        r0=r0,
                        z_offset=p.z_offset,
                        peak_key=(peak_z, peak_y, peak_x),
                    )
                )

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
                seed_offset=seed_offset,
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
                peak_z=peak_z,
                peak_y=peak_y,
                peak_x=peak_x,
                gs_median=gs_median,
                gs_p90=gs_p90,
                gs_peak=gs_peak,
                gs_shell_median=gs_shell_median,
                gs_contrast=gs_contrast,
                gs_iqr=gs_iqr,
                metal_threshold=p.metal_threshold,
                l1=l1,
                l2=l2,
                l3=l3,
                surface_ratio=surface_ratio,
            )
        )

    voxels: NDArray[np.int32] | None = (
        np.concatenate(voxel_chunks, axis=0) if voxel_chunks else None
    )
    return out, voxels


# trailing experiment-level param columns, shared between VOXEL_FIELDS (after
# in_grown) and vol_analysis.py's _CANDIDATE_FIELDS (after metal_threshold) --
# kept here as the one canonical order so both writers agree with each other
# and with the historical combined_candidates.csv / combined_voxels.parquet
# column order. New columns are APPENDED at the end (never inserted) so pre-existing
# column positions never shift: min_fill, then max_axial_extent_mm, then enforce_shape_gates.
# NOTE: z_extent_max stays here as the EFFECTIVE slice count that actually gated
# detection (run provenance); max_axial_extent_mm is the physical spec it was derived
# from (z_extent_max = ceil(max_axial_extent_mm / voxel_size)). enforce_shape_gates records
# whether the _axial_ok shape gates were applied for this run (it changes the candidate SET,
# so it is logged per row for provenance). All are logged so the row is self-describing.
EXPERIMENT_PARAM_FIELDS = [
    "slab_thickness",
    "high_threshold_scale",
    "low_threshold_scale",
    "baseline_method",
    "z_extent_max",
    "min_seed_voxels",
    "aniso_factor",
    "small_vol_cutoff",
    "z_pad",
    "small_z_bounds",
    "min_fill",
    "max_axial_extent_mm",
    "enforce_shape_gates",
]

VOXEL_FIELDS = [
    "experiment_number",
    "volume_name",
    "peak_z",
    "peak_y",
    "peak_x",
    "z",
    "y",
    "x",
    "residual",
    "grayscale",
    "in_grown",
    *EXPERIMENT_PARAM_FIELDS,
]


def write_voxel_rows(
    writer: "object",
    voxels: NDArray[np.int32] | None,
    volume_name: str,
    experiment_number: int,
    experiment_params: list[object] | None = None,
) -> int:
    """Append one CSV row per dumped voxel (columns: VOXEL_FIELDS) to an
    already-open ``csv.writer``. ``voxels`` is the int32 (n, 9) array produced by
    ``_process_slab``; ``experiment_number`` and ``volume_name`` are prepended to
    every row, and ``experiment_params`` (values matching
    ``EXPERIMENT_PARAM_FIELDS`` -- slab_thickness..min_fill, in that order) are
    appended, so every voxel row is fully self-describing and combining across
    experiments is a plain concatenation rather than a join against a separate
    params.csv. The (experiment_number, volume_name, peak_z, peak_y, peak_x)
    columns are the exact join key back to the feature table. Pass
    ``experiment_params=None`` to omit the trailing columns entirely (only do
    this if the header written to ``writer`` was also built without
    ``EXPERIMENT_PARAM_FIELDS``, e.g. a caller using a bespoke schema).
    Returns the number of rows written."""
    if voxels is None or voxels.shape[0] == 0:
        return 0
    extra = list(experiment_params) if experiment_params is not None else []
    writer.writerows(
        [experiment_number, volume_name, *row, *extra] for row in voxels.tolist()
    )
    return int(voxels.shape[0])


def detect_candidates_streaming(
    src: ResidualSource,
    k_high: float = 4.0,
    k_low: float = 2.99,
    min_seed_voxels: int = 2,
    z_extent_max: int = 10,
    small_z_bounds: tuple[int, int] = (1, 4),
    small_voxel_cutoff: int = 5,
    aniso_factor: float = 1.0,
    z_pad: int = 2,
    min_fill: float = 0.15,
    enforce_shape_gates: bool = True,
    slices_per_chunk: int = 128,
    z_offset: int = 0,
    verbose: bool = True,
    metal_threshold: float = float("nan"),
    metal_min_area: int = 100,
    metal_grayscale_margin: int = 0,
    voxel_writer: "object | None" = None,
    volume_name: str = "",
    storage_margin: int = 5,
    shell_radius: int = 3,
    experiment_number: int = -1,
    experiment_params: list[object] | None = None,
    on_slab_result: "Callable[[list[Candidate], NDArray[np.int32] | None], None] | None" = None,
) -> list[Candidate]:
    """Serial driver — unchanged behaviour, now a thin loop over ``_process_slab``.

    Kept as the single-worker reference and the baseline to diff the parallel run
    against (sort both by centroid first; only ordering differs).

    Grayscale features populate automatically when ``src`` provides
    ``read_grayscale`` (see ResidualSource); ``metal_threshold`` is the surrogate
    used for their shell exclusion. When ``voxel_writer`` (an open ``csv.writer``
    whose file already carries a VOXEL_FIELDS header) is given AND grayscale is
    available, per-candidate voxel dumps are appended after each slab completes,
    keeping only one slab's records in memory at a time. ``experiment_number``
    and ``experiment_params`` (values matching ``EXPERIMENT_PARAM_FIELDS``) are
    forwarded to ``write_voxel_rows`` so every voxel row is self-describing.

    ``on_slab_result``, if given, is called once per slab as
    ``on_slab_result(found, voxels)`` -- the same pair ``voxel_writer`` would
    otherwise consume -- instead of/alongside the CSV path. This is how
    experiment_db.ExperimentRecorder persists straight to DuckDB without this
    module knowing anything about CSVs or databases; it just hands each slab's
    result to whichever sink the caller wired up. Either or both of
    ``voxel_writer``/``on_slab_result`` may be set.
    """
    Z, _, _ = src.shape
    halo = z_extent_max
    p = DetectParams(
        k_high=k_high,
        k_low=k_low,
        min_seed_voxels=min_seed_voxels,
        z_extent_max=z_extent_max,
        small_z_bounds=small_z_bounds,
        small_voxel_cutoff=small_voxel_cutoff,
        aniso_factor=aniso_factor,
        z_pad=z_pad,
        min_fill=min_fill,
        enforce_shape_gates=enforce_shape_gates,
        z_offset=z_offset,
        metal_threshold=metal_threshold,
        metal_min_area=metal_min_area,
        metal_grayscale_margin=metal_grayscale_margin,
        storage_margin=storage_margin,
        shell_radius=shell_radius,
    )
    collect_voxels = voxel_writer is not None or on_slab_result is not None

    out: list[Candidate] = []
    for r0, r1, c0, c1 in iter_slabs(Z, slices_per_chunk, halo):
        found, voxels = _process_slab(src, r0, r1, c0, c1, p, collect_voxels)
        if voxel_writer is not None:
            write_voxel_rows(
                voxel_writer, voxels, volume_name, experiment_number, experiment_params
            )
        if on_slab_result is not None:
            on_slab_result(found, voxels)
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
    min_seed_voxels: int = 2,
    z_extent_max: int = 10,
    small_z_bounds: tuple[int, int] = (1, 4),
    small_voxel_cutoff: int = 5,
    aniso_factor: float = 1.0,
    z_pad: int = 2,
    min_fill: float = 0.15,
    enforce_shape_gates: bool = True,
    slices_per_chunk: int = 128,
    z_offset: int = 0,
    n_workers: int = 3,
    verbose: bool = True,
    metal_threshold: float = float("nan"),
    metal_min_area: int = 100,
    metal_grayscale_margin: int = 0,
    voxel_writer: "object | None" = None,
    volume_name: str = "",
    storage_margin: int = 5,
    shell_radius: int = 3,
    experiment_number: int = -1,
    experiment_params: list[object] | None = None,
    on_slab_result: "Callable[[list[Candidate], NDArray[np.int32] | None], None] | None" = None,
) -> list[Candidate]:
    """Process-parallel driver. Dispatches each slab to a loky worker and
    concatenates the per-slab candidate lists.

    Identical candidate set to ``detect_candidates_streaming`` for the same
    parameters — dedup is per-slab centroid ownership into disjoint cores, so the
    result does not depend on execution order. ONLY the ordering of the returned
    list differs (completion order, not slab order); sort by ``centroid`` before
    diffing against a serial baseline or a lo-invariance check.

    Grayscale features populate automatically when ``src`` provides
    ``read_grayscale`` (each worker lazily re-opens the raw-volume memmap
    read-only, page-cache shared — same pattern as the residual). When
    ``voxel_writer`` and/or ``on_slab_result`` is given AND grayscale is
    available, workers return their per-slab voxel arrays and the PARENT
    consumes them as slab results stream back — workers never touch the CSV or
    DB, so there is no concurrent-write hazard (the ``for found, voxels in
    results_iter:`` loop below is serial in the parent process). See
    ``detect_candidates_streaming`` for what ``on_slab_result`` receives.

    Memory: each worker holds one read-range slab, ~= (slices_per_chunk +
    2*z_extent_max) slices of derived arrays (R, snr float32; three bool masks;
    int32 labels; + scipy temporaries) ~= 38 MB/slice at 1425^2, plus the uint8
    grayscale slab (~2 MB/slice) when grayscale is enabled. The read-only input
    memmaps are shared via the OS page cache and cost ~nothing per worker,
    PROVIDED ``src`` pickles to metadata only (see ResidualSource contract).

    Set ``n_workers`` <= physical cores with the parent idle. Falls back to the
    serial driver when ``n_workers <= 1``.
    """
    if n_workers <= 1:
        return detect_candidates_streaming(
            src,
            k_high=k_high,
            k_low=k_low,
            min_seed_voxels=min_seed_voxels,
            z_extent_max=z_extent_max,
            small_z_bounds=small_z_bounds,
            small_voxel_cutoff=small_voxel_cutoff,
            aniso_factor=aniso_factor,
            z_pad=z_pad,
            min_fill=min_fill,
            enforce_shape_gates=enforce_shape_gates,
            slices_per_chunk=slices_per_chunk,
            z_offset=z_offset,
            verbose=verbose,
            metal_threshold=metal_threshold,
            metal_min_area=metal_min_area,
            metal_grayscale_margin=metal_grayscale_margin,
            voxel_writer=voxel_writer,
            volume_name=volume_name,
            storage_margin=storage_margin,
            shell_radius=shell_radius,
            experiment_number=experiment_number,
            experiment_params=experiment_params,
            on_slab_result=on_slab_result,
        )

    from joblib import Parallel, delayed

    Z, _, _ = src.shape
    halo = z_extent_max
    p = DetectParams(
        k_high=k_high,
        k_low=k_low,
        min_seed_voxels=min_seed_voxels,
        z_extent_max=z_extent_max,
        small_z_bounds=small_z_bounds,
        small_voxel_cutoff=small_voxel_cutoff,
        aniso_factor=aniso_factor,
        z_pad=z_pad,
        min_fill=min_fill,
        enforce_shape_gates=enforce_shape_gates,
        z_offset=z_offset,
        metal_threshold=metal_threshold,
        metal_min_area=metal_min_area,
        metal_grayscale_margin=metal_grayscale_margin,
        storage_margin=storage_margin,
        shell_radius=shell_radius,
    )
    collect_voxels = voxel_writer is not None or on_slab_result is not None

    slabs = list(iter_slabs(Z, slices_per_chunk, halo))

    # backend="loky": process-based (the per-component loop is GIL-bound, so
    # threads would not help). Dynamic dispatch balances the ~Z/chunk tasks over
    # the workers; the last partial slab's imbalance is negligible at this count.
    parallel = Parallel(n_jobs=n_workers, backend="loky", verbose=10 if verbose else 0)
    jobs = (
        delayed(_process_slab)(src, r0, r1, c0, c1, p, collect_voxels)
        for (r0, r1, c0, c1) in slabs
    )
    try:
        # stream results so voxel rows are written per slab (bounded memory)
        results_iter = Parallel(
            n_jobs=n_workers,
            backend="loky",
            verbose=10 if verbose else 0,
            return_as="generator",
        )(jobs)
    except TypeError:  # joblib < 1.3: no return_as; fall back to full list
        results_iter = parallel(
            delayed(_process_slab)(src, r0, r1, c0, c1, p, collect_voxels)
            for (r0, r1, c0, c1) in slabs
        )

    out: list[Candidate] = []
    for found, voxels in results_iter:
        if voxel_writer is not None:
            write_voxel_rows(
                voxel_writer, voxels, volume_name, experiment_number, experiment_params
            )
        if on_slab_result is not None:
            on_slab_result(found, voxels)
        out.extend(found)
    return out
