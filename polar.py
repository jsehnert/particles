from __future__ import annotations

from typing import TypeAlias

import cv2
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]


def build_polar_maps(
    img_shape: tuple[int, int],
    cx: float,
    cy: float,
    n_r: int,
    n_theta: int,
    r_min: float = 0.0,
    r_max: float | None = None,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Precompute Cartesian->polar sampling maps. Call once per scan geometry.

    Polar array is indexed [r_i, theta_j]; r spans [r_min, r_max].
    r_min > 0 lets you skip the unreliable central-axis annulus.
    """
    h, w = img_shape
    if r_max is None:
        corners = np.array([[0, 0], [0, w], [h, 0], [h, w]], dtype=np.float64)
        r_max = float(np.max(np.hypot(corners[:, 0] - cy, corners[:, 1] - cx)))

    r = np.linspace(r_min, r_max, n_r)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(r, theta, indexing="ij")  # (n_r, n_theta)

    map_x: FloatArray = (cx + R * np.cos(T)).astype(np.float32)  # col coords
    map_y: FloatArray = (cy + R * np.sin(T)).astype(np.float32)  # row coords
    return map_x, map_y, r_min, r_max


def cartesian_to_polar(
    img: NDArray[np.floating],
    map_x: FloatArray,
    map_y: FloatArray,
    interp: int = cv2.INTER_CUBIC,
) -> FloatArray:
    return cv2.remap(
        img,
        map_x,
        map_y,
        interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def build_inverse_maps(
    out_shape: tuple[int, int],
    cx: float,
    cy: float,
    n_r: int,
    n_theta: int,
    r_min: float,
    r_max: float,
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]]:
    """Precompute polar->Cartesian sampling maps. Call once per scan geometry.

    Maps each Cartesian pixel to fractional (r_idx, theta_idx) in the polar array.
    Theta wraps; r outside [r_min, r_max] is masked.
    """
    h, w = out_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy

    rr = np.hypot(dx, dy)
    tt = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    # physical r -> fractional row index over [r_min, r_max]
    r_idx = (rr - r_min) / (r_max - r_min) * (n_r - 1)
    # theta -> fractional col index; n_theta (not -1) because angle wraps
    t_idx = tt / (2 * np.pi) * n_theta

    valid: npt.NDArray[np.bool_] = (rr >= r_min) & (rr <= r_max)
    # remap map: map_x = column index (theta), map_y = row index (r)
    return t_idx.astype(np.float32), r_idx.astype(np.float32), valid


def polar_to_cartesian(
    polar: NDArray[np.floating],
    map_t: FloatArray,
    map_r: FloatArray,
    valid: NDArray[np.bool_],
    interp: int = cv2.INTER_CUBIC,
) -> FloatArray:
    # wrap theta by tiling one column on the angular (column) axis
    polar_w = np.concatenate([polar, polar[:, :1]], axis=1)
    out: FloatArray = cv2.remap(
        polar_w,
        map_t,
        map_r,
        interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    out[~valid] = 0
    return out
