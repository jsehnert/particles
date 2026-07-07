from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


def estimate_grayscale_range(vol: np.ndarray, slice_step: int = 200) -> tuple[int, int]:
    """
    For the given volume, estimate the air gray value and the max gray value by analyzing histograms of slices.

    Args:
        vol: The input volume as a 3D numpy array.
        slice_step: The step size for selecting slices to analyze.
    Returns:
        A tuple containing the estimated air gray value and the max gray value.

    Notes:
        - The air gray value is estimated as the lowest significant peak in the histogram, which typically
          corresponds to the background (air) in CT scans.
          - Any grayvalue at or below the air gray value should be considered as air.
        - The max gray value is estimated as the overall max from the analyzed slices
    """
    # Set the slice levels for analysis. Always including the first and last slice.
    slice_levels = np.arange(0, vol.shape[0], slice_step)
    slice_levels = np.append(slice_levels, vol.shape[0] - 1)

    air_grayvalue = 256
    max_grayvalue = -1

    for sl in slice_levels:
        img = vol[sl]
        max_grayvalue = max(max_grayvalue, img.max())
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).squeeze()
        peaks, _ = find_peaks(hist, height=hist.max() / 2, width=1)
        air_grayvalue = (
            min(peaks[0], air_grayvalue) if len(peaks) > 0 else air_grayvalue
        )

    return air_grayvalue, max_grayvalue


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
        raise ValueError("No contours found in the metal mask.")
    outer_contour = max(contours, key=cv2.contourArea)
    inside_mask = np.zeros_like(metal_mask, dtype=np.uint8)
    cv2.drawContours(inside_mask, [outer_contour], -1, 255, thickness=cv2.FILLED)
    return inside_mask.astype(bool)
