from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture


def _dbg_display(
    img: np.ndarray,
    exit: bool = False,
    path: str = str(Path(__file__).resolve().parent / "Graphics" / "dbg_display.png"),
) -> None:
    """Save `img` as a normalized PNG for debug review (thread-safe, unlike plt.show()).

    Args:
        img: 2D array to visualize. Rescaled so its max value maps to 255.
        exit: If True, sys.exit() right after saving, short-circuiting the caller.
        path: Output PNG path.
    """
    img = img.astype(np.float32)
    max_val = img.max()
    if max_val > 0:
        img = img / max_val * 255
    ok = cv2.imwrite(path, img.astype(np.uint8))
    if not ok:
        # cv2.imwrite fails silently (returns False) rather than raising, so check
        # explicitly -- otherwise a bad path/permission issue vanishes with no signal.
        raise RuntimeError(f"cv2.imwrite failed to write debug image to {path!r}")
    print(f"[_dbg_display] wrote {path}")

    if exit:
        import sys

        sys.exit()


def gaussian_mixture_fit(
    data: NDArray, n_components: int = 2
) -> dict[str, NDArray[np.float32]]:
    """
    Fit a Gaussian Mixture Model to the given data.

    Args:
        data: The input data as a 1D numpy array.
        n_components: The number of Gaussian components to fit.

    Returns:
        A dictionary containing the means, standard deviations, and weights of the fitted Gaussian components.
    """
    # Reshape data for fitting
    data_reshaped = data.reshape(-1, 1)

    # Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data_reshaped)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()

    results = {
        "means": means.squeeze().astype(np.float32),
        "stds": stds.squeeze().astype(np.float32),
        "weights": gmm.weights_.squeeze().astype(np.float32),
    }
    return results


def estimate_grayscale_range(
    vol: np.ndarray, slice_step: int = 200
) -> tuple[int, int, int]:
    """
    For the given volume, estimate the air gray value and the max gray value by analyzing histograms of slices.

    Args:
        vol: The input volume as a 3D numpy array.
        slice_step: The step size for selecting slices to analyze.
    Returns:
        A tuple containing the estimated air gray value, the core gray value, and the (effective) max gray value.

    Notes:
        - This code was developed for raw cylindrical battery cel volumes after Glimpse post-processing
        - The air gray value is estimated as the lowest significant peak in the histogram, which typically
          corresponds to the background (air) in CT scans.
          - Any grayvalue at or below the air gray value should be considered as air.
        - The max gray value is estimated as the overall max from the analyzed slices
    """

    # Set the slice levels for analysis. Always including the first and last slice.
    def _build_hist(vol: NDArray[np.uint8], slice_step: int = 200):
        slice_levels = np.arange(0, vol.shape[0], slice_step)
        slice_levels = np.append(slice_levels, vol.shape[0] - 1)
        histogram = np.zeros(256, dtype=float)
        for sl in slice_levels:
            img = vol[sl]
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).squeeze()
            histogram += hist
        return histogram

    def _process_hist(hist: NDArray) -> NDArray:
        hist[0] = hist[2]
        hist[1] = hist[2]
        hist[-1] = hist[-2]
        hist = gaussian_filter1d(hist, sigma=3)
        return hist

    hist = _build_hist(vol, slice_step=slice_step)
    hist = _process_hist(hist)

    # Identify the core gray value as the first peak in the inverted histogram
    neg_hist = -hist
    neg_hist = neg_hist - neg_hist.min()
    pks, _ = find_peaks(neg_hist, height=neg_hist.max() / 2, width=1)
    if len(pks) < 1:
        raise ValueError("No significant peaks found in the inverted histogram.")
    core_grayvalue = pks[0]
    air_grayvalue = 2

    # March down from the clipped saturation level to find the useable maximum value
    max_grayvalue = 254
    hist_threshold = hist[max_grayvalue] // 2
    while True:
        h_value = hist[max_grayvalue - 1]
        if h_value < hist_threshold:
            break
        max_grayvalue -= 1

    pks, _ = find_peaks(hist, height=hist.max() / 2, width=1)
    if len(pks) < 1:
        raise ValueError("No significant peaks found in the histogram.")

    max_grayvalue = max(max_grayvalue, pks[-1] + 10)
    max_grayvalue = min(max_grayvalue, 253)

    return air_grayvalue, core_grayvalue, max_grayvalue


def estimate_grayscale_range_for_Nikon_raw(
    vol: np.ndarray, slice_step: int = 200
) -> tuple[int, int, int]:
    """
    For the given volume, estimate the air gray value and the max gray value by analyzing histograms of slices.

    Args:
        vol: The input volume as a 3D numpy array.
        slice_step: The step size for selecting slices to analyze.
    Returns:
        A tuple containing the estimated air gray value and the max gray value.

    Notes:
        - This code was developed for the analysis of Nikon produced 8-bit raw volumes prior to Glimpse
          post-processing.
        - The air gray value is estimated as the lowest significant peak in the histogram, which typically
          corresponds to the background (air) in CT scans.
          - Any grayvalue at or below the air gray value should be considered as air.
        - The max gray value is estimated as the overall max from the analyzed slices
    """
    # Set the slice levels for analysis. Always including the first and last slice.
    slice_levels = np.arange(0, vol.shape[0], slice_step)
    slice_levels = np.append(slice_levels, vol.shape[0] - 1)

    max_grayvalue = -1
    hist = np.zeros(256, dtype=np.float32)  # Initialize histogram for gray values
    for sl in slice_levels:
        img = vol[sl]
        max_grayvalue = max(max_grayvalue, img.max())
        _hist = cv2.calcHist([img], [0], None, [256], [0, 256]).squeeze()
        hist += _hist

    peaks, _ = find_peaks(hist, height=hist.max() / 2, width=1)
    air_grayvalue = peaks[0] if len(peaks) > 0 else 0

    hist[: air_grayvalue + 1] = (
        0  # Ignore air gray values for max gray value estimation
    )
    hist = uniform_filter1d(hist, size=5)  # Smooth the histogram to reduce noise
    peaks, _ = find_peaks(-hist, prominence=1.0)
    core_grayvalue = peaks[0] if len(peaks) > 0 else air_grayvalue
    core_grayvalue = (
        air_grayvalue + core_grayvalue
    ) // 2  # Average of air and core gray values
    return air_grayvalue, core_grayvalue, max_grayvalue


def identify_cylindrical_support(metal_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Identifies the interior cylindrical support region from the given metal mask.

    Assumption: The metal mask is a binary image where True values indicate the presence of metal. The cylindrical support region is defined as the area enclosed by the outer contour of the metal mask.

    Args:
        metal_mask (NDArray[np.bool_]): A 2D boolean array where True values indicate the presence of metal.

    Returns:
        NDArray[np.bool_]: A 2D boolean array where True values indicate the interior cylindrical support region.
    """
    contours, _ = cv2.findContours(
        metal_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        # print("[identify_cylindrical_support] No contours found; caller stack:")
        # print("".join(traceback.format_stack(limit=3)[:-1]))
        _dbg_display(metal_mask, exit=True)
        raise ValueError("No contours found in the metal mask.")
    outer_contour = max(contours, key=cv2.contourArea)
    inside_mask = np.zeros_like(metal_mask, dtype=np.uint8)
    cv2.drawContours(inside_mask, [outer_contour], -1, 255, thickness=cv2.FILLED)
    return inside_mask.astype(bool)


#######
FloatArray = NDArray[np.float32]


class UF:
    """Index partitioning into disjoint sets using union-find (UF) data structure.
    Each element initially belongs to its own set. The `find` method returns the
    representative of the set containing a given element, and the `union` method
    merges the sets containing two given elements.
    """

    def __init__(self, k):
        self.p = list(range(k))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


@njit(cache=True, parallel=True, fastmath=False)
def _median_baseline_core(
    vol: NDArray[np.uint8],
    z0: int,
    z1: int,
    w: int,
    out: FloatArray,
) -> None:
    """Leave-in median baseline for output slices [z0, z1), 1D median filter of
    width ``w`` down axis 0, with fixed-width clamp-and-shift window edges.

    For each output slice n the window is [start, start+w) with
    start = clamp(n - w//2, 0, Z-w) — exactly the window your _compute_residual
    builds. A per-column 256-bin histogram is slid down z (one slice out, one in
    per step) and the median is tracked with a moving pointer, so cost per output
    voxel is O(1), independent of w.

    Median convention matches numpy: for odd w the middle value; for even w the
    mean of the two middle values. uint8 makes the histogram exact (256 bins), so
    the result is bit-identical to np.median(...).astype(float32).
    """
    Z = vol.shape[0]
    H = vol.shape[1]
    W = vol.shape[2]
    half = w // 2
    k = half  # 0-indexed order statistic tracked by the pointer (odd-window median)
    zmw = Z - w  # maximum legal window start

    for yy in prange(H):
        hist = np.zeros((W, 256), dtype=np.int32)
        m = np.zeros(W, dtype=np.int64)  # per-column median bin (pointer)
        lt = np.zeros(W, dtype=np.int64)  # per-column count of elements < m

        prev_start = -1
        for n in range(z0, z1):
            start = n - half
            if start < 0:
                start = 0
            elif start > zmw:
                start = zmw

            if prev_start == -1:
                # first output of this row: build the window histogram from scratch
                for s in range(start, start + w):
                    row = vol[s, yy]
                    for xx in range(W):
                        hist[xx, row[xx]] += 1
                # initialise each column's pointer to the k-th order statistic
                for xx in range(W):
                    c = 0
                    mm = 0
                    while c + hist[xx, mm] <= k:
                        c += hist[xx, mm]
                        mm += 1
                    m[xx] = mm
                    lt[xx] = c
                prev_start = start
            elif start != prev_start:
                # window advanced by one: drop slice prev_start, add slice prev_start+w
                row_out = vol[prev_start, yy]
                row_in = vol[prev_start + w, yy]
                for xx in range(W):
                    vo = row_out[xx]
                    vi = row_in[xx]
                    if vo != vi:
                        hist[xx, vo] -= 1
                        hist[xx, vi] += 1
                        mm = m[xx]
                        c = lt[xx]
                        if vo < mm:
                            c -= 1
                        if vi < mm:
                            c += 1
                        # restore invariant: lt <= k < lt + hist[m]
                        if c > k:
                            while c > k:
                                mm -= 1
                                c -= hist[xx, mm]
                        else:
                            while c + hist[xx, mm] <= k:
                                c += hist[xx, mm]
                                mm += 1
                        m[xx] = mm
                        lt[xx] = c
                prev_start = start
            # else: window unchanged (boundary run) → pointers already correct

            oi = n - z0
            # window is enforced odd, so the median is exactly m[xx] (an integer
            # bin in [0,255]). Writing it works for a float32 or a uint8 out array
            # alike — Numba specializes the store on out's dtype. uint8 is lossless.
            for xx in range(W):
                out[oi, yy, xx] = m[xx]


def median_baseline_block(
    vol: NDArray[np.uint8],
    z0: int,
    z1: int,
    window_size: int,
    out_dtype: type = np.uint8,
) -> NDArray:
    """Leave-in median baseline for output slices [z0, z1), shape (z1-z0, H, W).

    Computed for a whole slice-run at once so the sliding histogram is reused
    across slices (drop-in for the ``np.median`` step in ``_compute_residual``).
    ``vol`` must be C-contiguous uint8 with Z >= window_size, and window_size odd.

    out_dtype:
        np.uint8 — (default) LOSSLESS here: an odd-window median of uint8 values is itself a
            uint8 value, so no rounding. 4x smaller on disk/RAM (~9.3 GB vs ~37 GB
            for the full volume). See the subtraction caveat below.

    Caveat when out_dtype=np.uint8: do NOT compute the residual as uint8 minus
    uint8 — that wraps around (10 - 20 -> 246). Cast first, e.g.
        residual = vol[n].astype(np.int16) - baseline[n].astype(np.int16)
    which is exact and signed (residual range [-255, 255], fits int16), consistent
    with passing negative residuals pre-clip into detection.
    """
    if vol.dtype != np.uint8:
        raise TypeError("median_baseline_block requires a uint8 volume")
    if window_size % 2 == 0:
        raise ValueError(
            f"window_size must be odd for a centered median; got {window_size}. "
            "(An even window has no single center; a symmetric [n-w//2, n+w//2+1) "
            "span is w+1 wide, not w.)"
        )
    dt = np.dtype(out_dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.uint8)):
        raise ValueError(f"out_dtype must be float32 or uint8, got {dt}")
    Z, H, W = vol.shape
    if Z < window_size:
        raise ValueError(f"Z={Z} < window_size={window_size}")
    vol = np.ascontiguousarray(vol)
    out = np.empty((z1 - z0, H, W), dtype=dt)
    _median_baseline_core(vol, z0, z1, window_size, out)
    return out


@njit(cache=True, parallel=True, fastmath=False)
def _median_max_core(
    vol: NDArray[np.uint8],
    z0: int,
    z1: int,
    w: int,
    med_out,
    max_out,
) -> None:
    """Fused leave-in median + windowed max for output slices [z0, z1).

    Same sliding 256-bin per-column histogram as the median kernel; the max is the
    highest non-empty bin, tracked with one extra per-column pointer. Both use the
    identical clamp-and-shift window, so this is exactly the (baseline, metal-mask
    source) pair _compute_residual needs — computed in ONE pass over the volume
    instead of two. Window must be odd; uint8 makes both exact.
    """
    Z = vol.shape[0]
    H = vol.shape[1]
    W = vol.shape[2]
    half = w // 2
    k = half
    zmw = Z - w

    for yy in prange(H):
        hist = np.zeros((W, 256), dtype=np.int32)
        m = np.zeros(W, dtype=np.int64)  # median bin
        lt = np.zeros(W, dtype=np.int64)  # count < m
        hi = np.zeros(W, dtype=np.int64)  # highest non-empty bin (window max)

        prev_start = -1
        for n in range(z0, z1):
            start = n - half
            if start < 0:
                start = 0
            elif start > zmw:
                start = zmw

            if prev_start == -1:
                for s in range(start, start + w):
                    row = vol[s, yy]
                    for xx in range(W):
                        hist[xx, row[xx]] += 1
                for xx in range(W):
                    c = 0
                    mm = 0
                    while c + hist[xx, mm] <= k:
                        c += hist[xx, mm]
                        mm += 1
                    m[xx] = mm
                    lt[xx] = c
                    b = 255
                    while hist[xx, b] == 0:
                        b -= 1
                    hi[xx] = b
                prev_start = start
            elif start != prev_start:
                row_out = vol[prev_start, yy]
                row_in = vol[prev_start + w, yy]
                for xx in range(W):
                    vo = row_out[xx]
                    vi = row_in[xx]
                    if vo != vi:
                        hist[xx, vo] -= 1
                        hist[xx, vi] += 1
                        # median pointer
                        mm = m[xx]
                        c = lt[xx]
                        if vo < mm:
                            c -= 1
                        if vi < mm:
                            c += 1
                        if c > k:
                            while c > k:
                                mm -= 1
                                c -= hist[xx, mm]
                        else:
                            while c + hist[xx, mm] <= k:
                                c += hist[xx, mm]
                                mm += 1
                        m[xx] = mm
                        lt[xx] = c
                        # max pointer
                        h = hi[xx]
                        if vi > h:
                            h = vi
                        if hist[xx, h] == 0:  # only if we emptied the top bin
                            while h > 0 and hist[xx, h] == 0:
                                h -= 1
                        hi[xx] = h
                prev_start = start

            oi = n - z0
            for xx in range(W):
                med_out[oi, yy, xx] = m[xx]
                max_out[oi, yy, xx] = hi[xx]


def median_max_baseline_block(
    vol: NDArray[np.uint8],
    z0: int,
    z1: int,
    window_size: int,
    med_dtype: type = np.uint8,
    max_dtype: type = np.uint8,
):
    """Fused (median baseline, windowed max) for output slices [z0, z1).

    Returns (median, maximum), each shape (z1-z0, H, W). Both over the identical
    odd, clamp-and-shift centered window, computed in a single sliding-histogram
    pass — the max is the top non-empty bin of the same histogram used for the
    median, so it costs almost nothing beyond the median alone and avoids a second
    read of the volume. Both are exact for uint8 input (median lossless for odd w;
    max of uint8 is uint8).
    """
    if vol.dtype != np.uint8:
        raise TypeError("median_max_baseline_block requires a uint8 volume")
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd; got {window_size}")
    Z, H, W = vol.shape
    if Z < window_size:
        raise ValueError(f"Z={Z} < window_size={window_size}")
    vol = np.ascontiguousarray(vol)
    med = np.empty((z1 - z0, H, W), dtype=np.dtype(med_dtype))
    mx = np.empty((z1 - z0, H, W), dtype=np.dtype(max_dtype))
    _median_max_core(vol, z0, z1, window_size, med, mx)
    return med, mx


@njit(cache=True, parallel=True, fastmath=False)
def _max_filter_core(
    vol: NDArray[np.uint8],
    z0: int,
    z1: int,
    w: int,
    out,
) -> None:
    """Windowed max down axis 0 for output slices [z0, z1), same clamp-and-shift
    window as the median kernel. Per-column 256-bin histogram slid down z; the max
    is the highest non-empty bin, updated in O(1) amortized. Reads each slice ~twice
    (one in, one out) instead of the ~w times a per-slice np.max window costs."""
    Z = vol.shape[0]
    H = vol.shape[1]
    W = vol.shape[2]
    half = w // 2
    zmw = Z - w
    for yy in prange(H):
        hist = np.zeros((W, 256), dtype=np.int32)
        hi = np.zeros(W, dtype=np.int64)
        prev_start = -1
        for n in range(z0, z1):
            start = n - half
            if start < 0:
                start = 0
            elif start > zmw:
                start = zmw
            if prev_start == -1:
                for s in range(start, start + w):
                    row = vol[s, yy]
                    for xx in range(W):
                        hist[xx, row[xx]] += 1
                for xx in range(W):
                    b = 255
                    while hist[xx, b] == 0:
                        b -= 1
                    hi[xx] = b
                prev_start = start
            elif start != prev_start:
                row_out = vol[prev_start, yy]
                row_in = vol[prev_start + w, yy]
                for xx in range(W):
                    vo = row_out[xx]
                    vi = row_in[xx]
                    if vo != vi:
                        hist[xx, vo] -= 1
                        hist[xx, vi] += 1
                        h = hi[xx]
                        if vi > h:
                            h = vi
                        if hist[xx, h] == 0:
                            while h > 0 and hist[xx, h] == 0:
                                h -= 1
                        hi[xx] = h
                prev_start = start
            oi = n - z0
            for xx in range(W):
                out[oi, yy, xx] = hi[xx]


def max_filter_block(
    vol: NDArray[np.uint8],
    z0: int,
    z1: int,
    window_size: int,
    out_dtype: type = np.uint8,
):
    """Windowed max down z for output slices [z0, z1), shape (z1-z0, H, W).

    Single sliding pass (each input slice read ~twice) rather than the ~window_size
    re-reads a per-slice np.max window incurs; parallel over rows. Same clamp-and-shift
    centered window as median_baseline_block, so the metal-mask max lines up with the
    baseline when both use the same window_size. uint8 in/out is exact (max of uint8
    is uint8). Valid for any window_size >= 1 (max is well-defined for even windows).
    """
    if vol.dtype != np.uint8:
        raise TypeError("max_filter_block requires a uint8 volume")
    Z, H, W = vol.shape
    if Z < window_size:
        raise ValueError(f"Z={Z} < window_size={window_size}")
    vol = np.ascontiguousarray(vol)
    out = np.empty((z1 - z0, H, W), dtype=np.dtype(out_dtype))
    _max_filter_core(vol, z0, z1, window_size, out)
    return out
