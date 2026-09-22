import cv2
import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from skimage.morphology import isotropic_dilation


def estimate_metal_threshold(
    volume: NDArray[np.uint8], slice_step: int = 200, offset_percentile: int = 50
):
    """
    Generates a histogram of the volume to identify a reasonable metal threshold to be used for
    segmenting the metal regions.

    Note: This code was designed for cylindrical cells with metal can and tabs.

    Args:
        volume (NDArray[np.uint8]): Battery volume with metal
        slice_step (int, optional): Step size for sampling slices along the z-axis. Defaults to 200.
        offset_percentile (int, optional): Percentile offset from the base of the last significant peak to determine the metal threshold. Defaults to 50.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        int: the metal threshold dividing the grayscales from electrodes from metal.
    """
    vol = volume

    def _build_hist():
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

    def _first_inflection(
        hist_d2: np.ndarray, start: int, limit: int = None
    ) -> int | None:
        """
        Find the first zero-crossing of the input array to the right of the start point.

        Args:
            hist_d2 (np.ndarray): Second derivative of a cylindrical cell histogram.
            start (int): Index to start searching from.
            limit (int, optional): Maximum number of elements to search. Defaults to None.

        Returns:
            int or None: Index of the first inflection point, or None if not found.
        """
        end = (
            len(hist_d2) - 1 if limit is None else min(len(hist_d2) - 1, start + limit)
        )
        for i in range(start, end):
            if hist_d2[i] < 0 and hist_d2[i + 1] >= 0:
                return i + 1
        return None  # no inflection found in range — tail may be monotonic concave-down

    histogram = _build_hist()
    histogram = _process_hist(histogram)

    pks, _ = find_peaks(histogram, height=histogram.max() // 2, distance=10)
    if len(pks) == 0:
        raise ValueError("No significant peaks found in the histogram.")

    hist_d2 = savgol_filter(histogram, window_length=21, polyorder=3, deriv=2)

    pk_location = pks[-1]
    # If there is still a peak at the saturation end, then move back to the second peak
    if pk_location > 245:
        if len(pks) < 2:
            raise ValueError(
                f"Found only one significant peak in the histogram near the saturation end ({pk_location})."
            )
        pk_location = pks[-2]

    hist_d2 = savgol_filter(histogram, window_length=21, polyorder=3, deriv=2)
    pk_base = _first_inflection(hist_d2, pk_location)
    if pk_base is None:
        raise ValueError("No inflection point found for the last significant peak.")

    delta = 255 - pk_base
    offset = (offset_percentile * delta) // 100
    metal_threshold = pk_base + offset
    return metal_threshold


def estimate_metal_threshold_for_Nikon_raw(
    vol: NDArray[np.uint8], air_value: int, slice_step: int = 200
) -> int:
    """
    NOTE: This was developed for processing the raw Nikon volumes
    For the given volume, estimate the metal threshold by analyzing histograms of slices.

    Args:
        vol: The input volume as a 3D numpy array.
        air_value: The estimated air gray value (ignore grayscales below this).
        slice_step: The step size for selecting slices to analyze.
    Returns:
        An integer representing the estimated metal threshold.
    """

    # Build the histogram
    slice_levels = np.arange(0, vol.shape[0], slice_step)
    slice_levels = np.append(slice_levels, vol.shape[0] - 1)
    histogram = np.zeros(256, dtype=float)
    for sl in slice_levels:
        img = vol[sl]
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).squeeze()
        hist[: air_value + 1] = 0
        histogram += hist

    # Smooth the histogram to reduce noise and make peak detection more robust
    histogram = gaussian_filter1d(histogram, sigma=2)

    peaks, _ = scipy.signal.find_peaks(histogram, height=histogram.max() / 10)
    right_peak = max(peaks)

    right_tail = histogram[right_peak:]
    right_tail = gaussian_filter1d(right_tail, sigma=5)
    other_peaks, other_peak_info = scipy.signal.find_peaks(
        right_tail, prominence=10 * len(slice_levels)
    )
    if len(other_peaks) == 0:
        raise ValueError(
            "No significant peaks found in the right tail. Using right peak as threshold."
        )

    print(
        f"Right Peak: {right_peak}, Other Peaks: {other_peaks + right_peak}, Prominences: {other_peak_info['prominences']}"
    )
    best_peak = np.argmax(other_peak_info["prominences"])
    metal_threshold = right_peak + other_peak_info["left_bases"][best_peak]
    return metal_threshold


def extract_metal_mask(
    img: NDArray[np.uint8],
    metal_threshold: int,
    min_area: int,
    margin: int = 0,
    dilation_radius: int = 1,
) -> NDArray[np.bool_]:
    """
    Extract a binary mask of the metal regions from the given image using the specified metal threshold.

    Args:
        img: A 2D numpy array representing a single slice of the volume.
        metal_threshold: An integer representing the gray value threshold for identifying metal regions.
        min_area: An integer representing the minimum area of a metal region to be considered valid.
        margin: If > 0, apply hysteresis: seed regions from pixels above metal_threshold, then
            expand each seed region to include connected neighbouring pixels above
            (metal_threshold - margin). This is analogous to Canny's two-threshold approach.
    Returns:
        A binary numpy array of the same shape as img, where True values indicate metal regions and False values indicate non-metal regions.
    """
    seeds = (img > metal_threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
        seeds, 8, cv2.CV_32S, cv2.CCL_BBDT
    )
    keep = np.zeros(n, dtype=bool)
    if n > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    if margin <= 0 or not keep.any():
        return keep[labels]
    low = (img > metal_threshold - margin).astype(np.uint8)
    n_low, low_labels = cv2.connectedComponentsWithAlgorithm(
        low, 8, cv2.CV_32S, cv2.CCL_BBDT
    )
    ys, xs = np.nonzero(seeds)
    lab = labels[ys, xs]
    lo = low_labels[ys, xs]
    ok = keep[lab]
    seeded = np.zeros(n_low, dtype=bool)
    seeded[lo[ok]] = True
    seeded[0] = False

    mask = seeded[low_labels]
    mask = isotropic_dilation(mask, dilation_radius)
    return mask
