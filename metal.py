from typing import cast

import cv2
import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


def estimate_metal_threshold(
    vol: NDArray[np.uint8], air_value: int, slice_step: int = 200
) -> int:
    """
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
    img: np.ndarray, metal_threshold: int, min_area: int, margin: int = 0
) -> np.ndarray[np.bool_]:
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
    # High-threshold seeds
    seeds = (img > metal_threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    valid_seeds = set(
        int(i + 1) for i in np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0]
    )
    seed_mask = np.isin(labels, list(valid_seeds))

    if margin <= 0 or not valid_seeds:
        return cast(np.ndarray[np.bool_], seed_mask)

    # Low-threshold candidates: pixels above (metal_threshold - margin)
    low_binary = (img > metal_threshold - margin).astype(np.uint8)
    _, low_labels = cv2.connectedComponents(low_binary, connectivity=8)

    # Keep low-threshold components that contain at least one seed pixel
    seeded_components = set(low_labels[seed_mask].tolist()) - {0}
    return cast(np.ndarray[np.bool_], np.isin(low_labels, list(seeded_components)))
