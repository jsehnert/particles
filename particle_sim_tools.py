"""Reusable volume, noise, and particle simulation tools.

These definitions originated in Notebooks/noise_modeling.ipynb and are
shared by the notebook particle sweep and its optimized implementations.
Experiment configuration and execution remain outside this module.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy import ndimage
from scipy.special import erf

from noise import ar1_filter, ar2_filter

# region notebook cell 5

"""
## Dataclasses

`data` is indexed `[z, y, x]` throughout, shape `(voxels_z, voxels_xy, voxels_xy)`.

- **`Cylinder`** — a cylindrical inclusion added on top of the noise field.
  - `gray_value` — value added to every voxel inside the cylinder.
  - `radius`, `height` — in voxels.
  - `theta`, `phi` — alignment angles in **degrees**, both default `0.0` (a vertical, unrotated cylinder). `phi` tilts the cylinder axis away from the z-axis; `theta` is the azimuth of that tilt around the z-axis. See `Volume.add_cylinder` for how they enter the geometry and `Volume.align` for undoing them.
- **`Noise`** — parameters of the correlated Gaussian noise field (see "Correlation filters" above).
  - `std` — output standard deviation.
  - `rhoxy` — lateral (x-y) lag-1 correlation, applied isotropically to both axes.
  - `rho1`, `rho2` — z-direction lag-1/lag-2 correlation (`rho2 < rho1`).
- **`Volume`** — the voxel grid itself, built up in `create_data` when the object is constructed:
  - `add_noise` — fills `data` with the correlated noise field (`Noise`), if one is given.
  - `add_cylinder` — adds the tilted `Cylinder`, if one is given, on top of the noise.
  - `align(theta=0.0, phi=0.0, order=1)` — returns a *new* `Volume` after two successive `scipy.ndimage.rotate` calls. The angles are independent of `Volume.cylinder`; callers pass the desired corrections explicitly. Orders 1 through 5 are supported. `data` on the original `Volume` is left untouched, and the returned volume records its non-padding region in `valid_bounds`. Those bounds are calculated geometrically with an order-dependent interpolation margin, without allocating a volume-sized validity mask. An aligned volume cannot be aligned again."""


@dataclass
class ValidXBounds:
    """Half-open valid x interval for each (z, y) row."""

    x_min: NDArray[np.int32]
    x_max: NDArray[np.int32]


def backward_rotation_map(
    angle: float, axes: tuple[int, int], shape: tuple[int, int, int]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return scipy.rotate's backward affine map for reshape=False."""
    axis_a, axis_b = sorted(axes)
    radians = np.deg2rad(angle)
    cosine, sine = np.cos(radians), np.sin(radians)

    matrix = np.eye(3, dtype=np.float64)
    matrix[axis_a, axis_a] = cosine
    matrix[axis_a, axis_b] = sine
    matrix[axis_b, axis_a] = -sine
    matrix[axis_b, axis_b] = cosine

    center = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    offset = center - matrix @ center
    return matrix, offset


def forward_rotate_points_zyx(
    points_zyx: NDArray[np.float64],
    angle: float,
    axes: tuple[int, int],
    shape_zyx: tuple[int, int, int],
) -> NDArray[np.float64]:
    """Map source points to output coordinates for ``ndimage.rotate``.

    ``backward_rotation_map`` describes the output-to-input sampling map
    used by SciPy. This function applies its inverse to row-vector points.
    The volume shape is unchanged because all rotations use
    ``reshape=False``.
    """
    points_zyx = np.asarray(points_zyx, dtype=np.float64)
    if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
        raise ValueError("points_zyx must have shape (N, 3)")

    matrix, offset = backward_rotation_map(angle, axes, shape_zyx)
    return (points_zyx - offset) @ matrix


def forward_align_points_zyx(
    points_zyx: NDArray[np.float64],
    theta: float,
    phi: float,
    shape_zyx: tuple[int, int, int],
) -> NDArray[np.float64]:
    """Apply the same forward rotation sequence as ``Volume.align``."""
    transformed = np.asarray(points_zyx, dtype=np.float64)
    if theta != 0.0:
        transformed = forward_rotate_points_zyx(transformed, theta, (1, 2), shape_zyx)
    if phi != 0.0:
        transformed = forward_rotate_points_zyx(transformed, -phi, (0, 2), shape_zyx)
    return transformed


def valid_x_bounds_after_alignment(
    shape: tuple[int, int, int],
    theta: float,
    phi: float,
    margin: float = 1.0,
) -> ValidXBounds:
    """Calculate valid x bounds without allocating a 3-D mask.

    For each output (z, y), every backward-mapped coordinate is affine in x.
    Intersecting those linear constraints with the inset volume boxes therefore
    produces one valid x interval. ``x_max`` is exclusive.
    """
    z_size, y_size, x_size = shape
    if theta == 0.0 and phi == 0.0:
        return ValidXBounds(
            x_min=np.zeros((z_size, y_size), dtype=np.int32),
            x_max=np.full((z_size, y_size), x_size, dtype=np.int32),
        )

    z = np.arange(z_size, dtype=np.float64)[:, None]
    y = np.arange(y_size, dtype=np.float64)[None, :]
    lower = np.zeros((z_size, y_size), dtype=np.float64)
    upper = np.full((z_size, y_size), x_size - 1.0, dtype=np.float64)

    def intersect_inset_box(
        matrix: NDArray, offset: NDArray, rotation_axes: tuple[int, int]
    ) -> None:
        nonlocal lower, upper
        for coordinate_axis, axis_size in enumerate(shape):
            axis_margin = margin if coordinate_axis in rotation_axes else 0.0
            box_lower = axis_margin
            box_upper = axis_size - 1.0 - axis_margin
            base = (
                matrix[coordinate_axis, 0] * z
                + matrix[coordinate_axis, 1] * y
                + offset[coordinate_axis]
            )
            x_coefficient = matrix[coordinate_axis, 2]

            if abs(x_coefficient) < 1e-12:
                outside = (base < box_lower) | (base > box_upper)
                lower[outside] = 1.0
                upper[outside] = 0.0
                continue

            endpoint_a = (box_lower - base) / x_coefficient
            endpoint_b = (box_upper - base) / x_coefficient
            lower = np.maximum(lower, np.minimum(endpoint_a, endpoint_b))
            upper = np.minimum(upper, np.maximum(endpoint_a, endpoint_b))

    identity = np.eye(3, dtype=np.float64)
    zero_offset = np.zeros(3, dtype=np.float64)
    phi_matrix, phi_offset = identity, zero_offset

    if phi != 0.0:
        phi_matrix, phi_offset = backward_rotation_map(-phi, (0, 2), shape)
        # The second rotation must sample inside its intermediate input array.
        intersect_inset_box(phi_matrix, phi_offset, rotation_axes=(0, 2))

    if theta != 0.0:
        theta_matrix, theta_offset = backward_rotation_map(theta, (1, 2), shape)
        source_matrix = theta_matrix @ phi_matrix
        source_offset = theta_matrix @ phi_offset + theta_offset
        # The composed backward map must also remain inside the original data.
        intersect_inset_box(source_matrix, source_offset, rotation_axes=(1, 2))

    valid_rows = lower <= upper
    x_min = np.where(valid_rows, np.ceil(lower - 1e-9), x_size)
    x_max = np.where(valid_rows, np.floor(upper + 1e-9) + 1, 0)
    return ValidXBounds(
        x_min=np.clip(x_min, 0, x_size).astype(np.int32),
        x_max=np.clip(x_max, 0, x_size).astype(np.int32),
    )


@dataclass
class Cylinder:
    gray_value: float
    radius: int
    height: int
    theta: float = 0.0  # azimuth about the z-axis, degrees
    phi: float = 0.0  # tilt away from the z-axis, degrees


@dataclass
class Noise:
    std: float
    rhoxy: float
    rho1: float
    rho2: float


@dataclass
class Volume:
    type: DTypeLike
    voxels_z: int
    voxels_xy: int
    noise: Noise | None = None
    cylinder: Cylinder | None = None

    data: NDArray | None = None
    valid_bounds: ValidXBounds | None = None

    def __init__(
        self,
        type: DTypeLike,
        voxels_z: int,
        voxels_xy: int,
        noise: Noise | None = None,
        cylinder: Cylinder | None = None,
    ):
        self.type = type
        self.voxels_z = voxels_z
        self.voxels_xy = voxels_xy
        self.noise = noise
        self.cylinder = cylinder
        self.create_data()

    def create_data(self):
        self.valid_bounds = None
        self.data = np.zeros(
            (self.voxels_z, self.voxels_xy, self.voxels_xy), dtype=self.type
        )
        if self.noise is not None:
            self.add_noise()

        if self.cylinder is not None:
            self.add_cylinder()

    def add_noise(self):
        # Unit-variance white noise; each filter below preserves unit
        # variance, so the whole field is scaled by noise.std at the end.
        field = np.random.normal(size=self.data.shape)

        # Lateral (x-y) first-order correlation: separable AR(1) along x
        # then y (Kronecker-product covariance; see "Correlation filters").
        field = ar1_filter(field, self.noise.rhoxy, axis=2)  # x
        field = ar1_filter(field, self.noise.rhoxy, axis=1)  # y

        # z-direction first- and second-order correlation: AR(2) along z,
        # matching rho1 (lag 1) and rho2 (lag 2) via Yule-Walker.
        field = ar2_filter(field, self.noise.rho1, self.noise.rho2, axis=0)

        # Now ensure the noise is zero mean and of unit variance.
        field -= field.mean()
        field /= field.std(ddof=0)

        self.data += self.noise.std * field

    def cylinder_mask(self) -> NDArray:
        """Boolean mask of voxels inside `self.cylinder` (must not be None).

        Shared by `add_cylinder` (to place the inclusion) and
        `estimate_volume_noise` (to locate the cylinder's z-extent so the
        slices it contaminates can be excluded from the noise estimate).
        """
        cz = self.voxels_z // 2
        cxy = self.voxels_xy // 2

        rh = self.cylinder.height / 2.0
        rr = self.cylinder.radius

        theta = np.float32(np.deg2rad(self.cylinder.theta))
        phi = np.float32(np.deg2rad(self.cylinder.phi))

        dz = np.cos(phi)
        dy = np.sin(phi) * np.sin(theta)
        dx = np.sin(phi) * np.cos(theta)

        # Broadcasted coordinate arrays rather than three full 3-D grids.
        vz = np.arange(self.voxels_z, dtype=np.float32)[:, None, None] - cz
        vy = np.arange(self.voxels_xy, dtype=np.float32)[None, :, None] - cxy
        vx = np.arange(self.voxels_xy, dtype=np.float32)[None, None, :] - cxy

        t = vz * dz + vy * dy + vx * dx

        r2 = (vz - t * dz) ** 2 + (vy - t * dy) ** 2 + (vx - t * dx) ** 2

        return (np.abs(t) <= rh) & (r2 <= rr**2)

    def _cylinder_mask(self) -> NDArray:
        """Boolean mask of voxels inside `self.cylinder` (must not be None).

        Shared by `add_cylinder` (to place the inclusion) and
        `estimate_volume_noise` (to locate the cylinder's z-extent so the
        slices it contaminates can be excluded from the noise estimate).
        """
        cz, cxy = self.voxels_z // 2, self.voxels_xy // 2
        rh = self.cylinder.height / 2.0
        rr = self.cylinder.radius

        # Cylinder axis direction: phi tilts it away from the z-axis, theta
        # sets the azimuth of that tilt in the x-y plane (both degrees).
        theta = np.deg2rad(self.cylinder.theta)
        phi = np.deg2rad(self.cylinder.phi)
        dz = np.cos(phi)
        dy = np.sin(phi) * np.sin(theta)
        dx = np.sin(phi) * np.cos(theta)

        zz, yy, xx = np.indices(self.data.shape, dtype=np.float64)
        vz, vy, vx = zz - cz, yy - cxy, xx - cxy

        # Axial coordinate along the (possibly tilted) axis, and the
        # perpendicular (radial) distance from it.
        t = vz * dz + vy * dy + vx * dx
        r2 = (vz - t * dz) ** 2 + (vy - t * dy) ** 2 + (vx - t * dx) ** 2

        return (np.abs(t) <= rh) & (r2 <= rr**2)

    def add_cylinder(self, cylinder: Cylinder | None = None):
        """
        Add the cylinder as a constant gray-value inclusion.

        input:
            cylinder: Cylinder | None - If None, uses internal cylinder otherwise it overwrites the internal cylinder
        """
        if cylinder is not None:
            self.cylinder = cylinder

        self.data[self.cylinder_mask()] += self.cylinder.gray_value

    def align(
        self,
        theta: float = 0.0,
        phi: float = 0.0,
        order: int = 1,
    ) -> "Volume":
        """Return an independently stored volume rotated by theta and phi.

        The angles are independent of ``self.cylinder`` and are expressed in
        degrees, matching ``scipy.ndimage.rotate``. ``valid_bounds`` on the
        returned Volume identifies the non-padding x interval in every (z, y) row.
        """
        if not 1 <= order <= 5:
            raise ValueError(f"order must be between 1 and 5, got {order}")

        # Orders above 1 use scipy's recursive spline prefilter, whose boundary
        # transient extends beyond the nominal n + 1 sample interpolation stencil.
        interpolation_margin = 1.0 if order == 1 else float(2 * order)

        if self.valid_bounds is not None:
            raise ValueError("align() cannot be applied to an already aligned Volume")

        if np.issubdtype(self.data.dtype, np.floating) and any(
            not np.isfinite(z_slice).all() for z_slice in self.data
        ):
            raise ValueError("The source volume must contain only finite values")

        # Copy metadata without invoking __init__ and regenerating the data.
        aligned = copy.copy(self)
        rotated = self.data
        rotation_applied = False

        if theta != 0.0:
            rotated = ndimage.rotate(
                rotated,
                angle=theta,
                axes=(1, 2),
                reshape=False,
                order=order,
                mode="constant",
                cval=0.0,
                output=self.data.dtype,
                prefilter=order > 1,
            )
            rotation_applied = True

        if phi != 0.0:
            rotated = ndimage.rotate(
                rotated,
                angle=-phi,
                axes=(0, 2),
                reshape=False,
                order=order,
                mode="constant",
                cval=0.0,
                output=self.data.dtype,
                prefilter=order > 1,
            )
            rotation_applied = True

        if rotation_applied:
            # ndimage.rotate returned a newly allocated array.
            aligned.data = rotated
        else:
            # Maintain the contract that align() returns independent data.
            aligned.data = self.data.copy()

        aligned.valid_bounds = valid_x_bounds_after_alignment(
            self.data.shape, theta, phi, margin=interpolation_margin
        )

        if self.cylinder is not None:
            cylinder = copy.copy(self.cylinder)
            cylinder.theta -= theta
            cylinder.phi -= phi
            aligned.cylinder = cylinder

        return aligned


class DigitalNoiseConverter:
    """Digital converter class to convert standard noise into uint8 or uint16 format for analysis."""

    def __init__(self, target_noise_std: float, data_type: DTypeLike):
        """Initialize the digital converter."""
        _dtype = np.dtype(data_type)
        if _dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            raise ValueError("data_type must be uint8 or uint16")

        if target_noise_std <= 0:
            raise ValueError("target_noise_std must be positive")

        self.data_type = _dtype
        self.max_value = np.iinfo(_dtype).max
        self.target_noise_std = target_noise_std

        self.background_value = self.max_value / 2

    @property
    def lower_headroom_sigma(self) -> float:
        return self.background_value / self.target_noise_std

    @property
    def upper_headroom_sigma(self) -> float:
        return (self.max_value - self.background_value) / self.target_noise_std

    def convert(
        self, normalized_values: float | NDArray
    ) -> np.uint8 | np.uint16 | NDArray[np.uint8 | np.uint16]:
        """Apply the digital conversion to a given value."""
        normalized_values = np.asarray(normalized_values)
        converted = np.rint(
            self.background_value + self.target_noise_std * normalized_values
        )
        return np.clip(
            converted,
            0,
            self.max_value,
        ).astype(self.data_type)

    def convert_volume(self, noise_volume: Volume) -> Volume:
        """Apply the digital conversion to an entire noise volume."""
        source_dtype = noise_volume.data.dtype
        target_dtype = np.dtype(self.data_type)

        if source_dtype == target_dtype:
            raise ValueError(
                f"The source volume already has the target dtype {target_dtype}"
            )

        if not np.issubdtype(source_dtype, np.floating):
            raise TypeError(
                f"DigitalNoiseConverter expects floating-point input, but got {source_dtype}"
            )

        converted = copy.copy(noise_volume)
        converted.type = self.data_type
        converted.data = self.convert(noise_volume.data)

        if noise_volume.cylinder is not None:
            converted.cylinder = copy.copy(noise_volume.cylinder)
            converted.cylinder.gray_value = self.convert(
                noise_volume.cylinder.gray_value
            ).item()

        if noise_volume.noise is not None:
            converted.noise = copy.copy(noise_volume.noise)
            converted.noise.std *= self.target_noise_std

        return converted


# endregion notebook cell 5


# region notebook cell 10
"""
## Noise estimation

`estimate_volume_noise` recovers per-z noise std from the volume itself: it takes the difference of each pair of adjacent z-slices and estimates that slice's noise as `std(diff) / sqrt(2 * (1 - rho1))`.

The `sqrt(2)` corrects for the differencing doubling the variance of two slices; the extra `(1 - rho1)` factor corrects for those two slices not being independent -- adjacent z-slices are lag-1 correlated by `rho1` (see `Noise`), which shrinks `Var(diff) = 2 * sigma**2 * (1 - rho1)` below the independent case and biases the plain `std(diff) / sqrt(2)` estimator low by a factor of `sqrt(1 - rho1)`. `rho1` defaults to `volume.noise.rho1` when available, and can be overridden (e.g. `rho1=0.0` to reproduce the naive estimator, or when estimating from data without a known `Noise`).

When `volume.valid_bounds` is present, each z-difference uses the intersection of the valid x intervals from its two contributing slices, excluding padding introduced by alignment.

When `volume.cylinder` is set, four of the raw `voxels_z - 1` diffs are dropped before returning, symmetrically at both caps, since they're contaminated by the cylinder's signal step rather than pure noise: at the bottom, the diff entering the cylinder from below (background -> bottom cap) and the diff wholly inside it just above that (bottom cap -> its interior neighbour); at the top, the mirror image -- the diff wholly inside the cylinder just below its top (interior neighbour -> top cap) and the diff exiting it above (top cap -> background). The slice right at each cap is itself inconsistent with its interior neighbour, not just with the background across the boundary (see `Volume.cylinder_mask`, shared with `add_cylinder`, for how the cylinder's z-extent is located). The returned array is therefore `voxels_z - 5` long when a cylinder is present, `voxels_z - 1` otherwise -- its indices no longer line up one-to-one with z past the point of the first exclusion."""


def estimate_axis0_lag1_correlation(volume: Volume) -> NDArray[np.float64]:
    """Estimate centered lag-1 axis-0 correlation for every z slice.

    Each interior value is the mean of the Pearson correlations between the
    slice and each immediate neighbor. The two endpoints are ``NaN`` because
    they do not have neighbors on both sides. When alignment bounds are
    present, each pair uses only their shared valid samples.
    """
    data = np.asarray(volume.data, dtype=np.float32)
    correlations = np.full(data.shape[0], np.nan, dtype=np.float64)
    x_coordinates = np.arange(data.shape[2])[None, :]

    def pair_correlation(lower_z: int, upper_z: int) -> float:
        if volume.valid_bounds is None:
            lower_values = data[lower_z].ravel()
            upper_values = data[upper_z].ravel()
        else:
            x_min = np.maximum(
                volume.valid_bounds.x_min[lower_z],
                volume.valid_bounds.x_min[upper_z],
            )
            x_max = np.minimum(
                volume.valid_bounds.x_max[lower_z],
                volume.valid_bounds.x_max[upper_z],
            )
            valid = (x_coordinates >= x_min[:, None]) & (x_coordinates < x_max[:, None])
            lower_values = data[lower_z][valid]
            upper_values = data[upper_z][valid]

        if (
            lower_values.size < 2
            or lower_values.std() == 0.0
            or upper_values.std() == 0.0
        ):
            return np.nan
        return float(np.corrcoef(lower_values, upper_values)[0, 1])

    pair_correlations = np.array(
        [pair_correlation(z, z + 1) for z in range(data.shape[0] - 1)],
        dtype=np.float64,
    )
    correlations[1:-1] = 0.5 * (pair_correlations[:-1] + pair_correlations[1:])
    return correlations


def estimate_axis0_difference_noise(volume: Volume) -> NDArray[np.float64]:
    """Estimate centered per-slice noise from adjacent axis-0 differences.

    Each interior value averages the standard deviations of the differences
    with the slices immediately below and above, divided by ``sqrt(2)``. This
    is the naive independent-slice estimate: it intentionally does not correct
    for lag-1 axis-0 correlation. The two endpoints are ``NaN`` because they
    do not have neighbors on both sides.
    """
    data = np.asarray(volume.data, dtype=np.float32)
    noise = np.full(data.shape[0], np.nan, dtype=np.float64)
    x_coordinates = np.arange(data.shape[2])[None, :]

    def pair_noise(lower_z: int, upper_z: int) -> float:
        difference = data[upper_z] - data[lower_z]
        if volume.valid_bounds is None:
            values = difference.ravel()
        else:
            x_min = np.maximum(
                volume.valid_bounds.x_min[lower_z],
                volume.valid_bounds.x_min[upper_z],
            )
            x_max = np.minimum(
                volume.valid_bounds.x_max[lower_z],
                volume.valid_bounds.x_max[upper_z],
            )
            valid = (x_coordinates >= x_min[:, None]) & (x_coordinates < x_max[:, None])
            values = difference[valid]

        if values.size < 2:
            return np.nan
        return float(np.std(values, dtype=np.float64) / np.sqrt(2.0))

    pair_noise_estimates = np.array(
        [pair_noise(z, z + 1) for z in range(data.shape[0] - 1)],
        dtype=np.float64,
    )
    noise[1:-1] = 0.5 * (pair_noise_estimates[:-1] + pair_noise_estimates[1:])
    return noise


def estimate_volume_noise(volume: Volume) -> tuple[NDArray, NDArray]:
    """Estimates the volume noise and correlation coefficients from the given volume."""

    _noise = estimate_axis0_difference_noise(volume)
    _rho1 = estimate_axis0_lag1_correlation(volume)
    if len(_noise) < 3:
        raise ValueError("volume must contain at least three axis-0 slices")

    noise = _noise.astype(np.float32, copy=True)
    rho1 = _rho1.astype(np.float32, copy=True)

    # Extend the endpoint estimates from their only interior neighbor.
    noise[0] = noise[1]
    noise[-1] = noise[-2]
    rho1[0] = rho1[1]
    rho1[-1] = rho1[-2]

    # Correct the naive difference estimate for lag-1 correlation.
    noise /= np.sqrt(1.0 - rho1)
    return noise, rho1


def estimate_volume_noise_fixed_rho(
    volume: Volume, rho1: float | None = None
) -> NDArray:
    if rho1 is None:
        rho1 = volume.noise.rho1 if volume.noise is not None else 0.0

    diffs = np.subtract(
        volume.data[1:], volume.data[:-1], dtype=np.float32
    )  # np.diff(volume.data, axis=0)
    noise_scale = np.sqrt(2.0 * (1.0 - rho1))

    if volume.valid_bounds is None:
        sigma = np.std(diffs, axis=(1, 2), dtype=np.float64) / noise_scale
    else:
        sigma = np.full(diffs.shape[0], np.nan, dtype=np.float64)
        x_coordinates = np.arange(diffs.shape[2])[None, :]
        for z in range(diffs.shape[0]):
            # A difference is valid only where both adjacent slices are valid.
            x_min = np.maximum(
                volume.valid_bounds.x_min[z],
                volume.valid_bounds.x_min[z + 1],
            )
            x_max = np.minimum(
                volume.valid_bounds.x_max[z],
                volume.valid_bounds.x_max[z + 1],
            )
            valid = (x_coordinates >= x_min[:, None]) & (x_coordinates < x_max[:, None])
            if valid.any():
                sigma[z] = np.std(diffs[z][valid], dtype=np.float64) / noise_scale

    if volume.cylinder is None:
        return sigma

    # diffs[i] = data[i + 1] - data[i]. Exclude the diffs contaminated by
    # the cylinder's signal step, symmetrically at both caps: the
    # transition entering the cylinder from below (z_min - 1 -> z_min) and
    # the diff wholly inside it at the very bottom (z_min -> z_min + 1),
    # plus the mirror image at the top -- the diff wholly inside the
    # cylinder at its very top (z_max - 1 -> z_max) and the transition
    # exiting it above (z_max -> z_max + 1). The slice right at each cap is
    # itself inconsistent with its interior neighbour, not just with the
    # background on the other side of the boundary.
    if volume.cylinder.phi == 0.0:
        cz = volume.voxels_z // 2
        rh = volume.cylinder.height / 2.0

        z_min = max(0, int(np.ceil(cz - rh)))
        z_max = min(
            volume.voxels_z - 1,
            int(np.floor(cz + rh)),
        )
    else:
        in_cylinder_z = np.flatnonzero(volume.cylinder_mask().any(axis=(1, 2)))
        z_min, z_max = in_cylinder_z[0], in_cylinder_z[-1]

    exclude = {z_min - 1, z_min, z_max - 1, z_max}

    keep = np.array([i for i in range(len(sigma)) if i not in exclude])
    return sigma[keep]


# endregion notebook cell 10


# region notebook cell 12
"""
## Global noise sampler

`NoiseSampler` owns one correlated-noise realization and supplies independent working volumes for the alignment sweeps. The base realization is always generated once as `float32`, irrespective of the requested analysis dtype. This ensures that dtype comparisons begin with the same underlying noise field rather than different random samples.

### Configuration

- `v_xy`: Number of voxels along both lateral axes.
- `v_z`: Number of voxels along the z-axis. The volume shape is `(v_z, v_xy, v_xy)`.
- `noise_std`: Standard deviation of the base `float32` noise field in normalized intensity units.
- `rho_xy`: Lag-1 correlation applied along both lateral axes.
- `rho1`, `rho2`: Lag-1 and lag-2 correlations along the z-axis.
- `analysis_dtype`: Dtype returned by `get_noise_volume()`. Supported values are `np.float32`, `np.uint8`, and `np.uint16`. It defaults to `VOL_DATA_TYPE`.

### Initialized attributes

- `noise`: The `Noise` metadata describing the normalized base field.
- `noise_vol`: The shared base `Volume`. Its data is always `float32`, and it contains noise only (`cylinder=None`). Callers should treat this volume as read-only.

### Methods

- `get_noise_volume()`: Returns an independent working `Volume`. For `float32` analysis it copies the base data and `Noise` metadata. For unsigned analysis it uses `DigitalConverter` to create a new quantized array and scales `volume.noise.std` into digital-number units. Mutating the returned data or noise metadata does not alter the base realization.
- `get_workers_limit(max_gb=10)`: Estimates the maximum number of simultaneous worker volumes that fit within the requested memory budget. The estimate uses one analysis-volume array per worker and does not include interpolation temporaries or other process overhead.
- `std_for_type()`: Returns `NOISE_LEVEL` scaled by the gain associated with the global `VOL_DATA_TYPE`. This helper is intended for unsigned global analysis types.

### Typical use

```python
sampler = NoiseSampler(v_z=128, v_xy=512, analysis_dtype=np.uint8)
volume = sampler.get_noise_volume()
aligned = volume.align(theta=theta, phi=phi, order=order)
noise_profile, correlation_profile = estimate_volume_noise(aligned)
```
"""

""" This code-block should be deleted - it remains in case we need to debug
target_data_type = np.uint8

target_std = 1.0
if target_data_type != np.float32:
    target_std = 0.016 * np.iinfo(target_data_type).max
"""


@dataclass
class NoiseSampler:
    v_xy: int = 256
    v_z: int = 128

    # cyl_gray_value = 0.5

    rho_xy: float = 0.5
    rho1: float = 0.5
    rho2: float = 0.2
    target_std: float = 1.0
    noise: Noise = field(init=False)
    noise_vol: Volume = field(init=False)
    analysis_dtype: DTypeLike = np.uint8

    def __post_init__(self):
        self.noise = Noise(std=1.0, rhoxy=self.rho_xy, rho1=self.rho1, rho2=self.rho2)

        # Always create the standard noise volume during initialization
        self.noise_vol = Volume(
            type=np.float32,
            voxels_z=self.v_z,
            voxels_xy=self.v_xy,
            noise=self.noise,
            cylinder=None,
        )

    def get_workers_limit(self, max_gb: int = 10):
        volume_size = (
            self.v_z * self.v_xy * self.v_xy * np.dtype(self.analysis_dtype).itemsize
        )
        max_workers = max(1, int((max_gb * (1024**3)) / volume_size))
        return max_workers

    def get_noise_volume(self, target_std: float | None = None) -> Volume:
        """Return the standard noise volume with optional cylinder embedded"""

        if target_std is None:
            target_std = self.target_std

        # For non-floating point types, adjust for integer quantization error of 1/12 so that we get back the correct target variance
        if not np.issubdtype(np.dtype(self.analysis_dtype), np.floating):
            target_var = target_std**2
            if target_var > 1 / 12:
                target_var -= 1 / 12
            target_std = np.sqrt(target_var)

        if self.analysis_dtype == np.float32:
            # Copy the volume metatdata without invoking __init__/create_data.
            vol = copy.copy(self.noise_vol)

            # Each volume returned must own it's own data because add_cylinder mutates it
            vol.data = self.noise_vol.data.copy() * target_std
            vol.noise = copy.copy(self.noise_vol.noise)
            return vol

        dc = DigitalNoiseConverter(
            target_noise_std=target_std, data_type=self.analysis_dtype
        )
        return dc.convert_volume(self.noise_vol)


# endregion notebook cell 12


# region notebook cell 39
"""
### Simulated Particles
Simulated particles are continuous, axis-aligned Gaussian blobs whose centers may lie at integer or sub-voxel `(z, y, x)` coordinates. `spatial_sigma` may be a positive scalar for an isotropic particle or three positive values in `(z, y, x)` order. Rasterization integrates the Gaussian over each intersecting voxel, truncates it with an ellipsoid in sigma-normalized coordinates, and uses `peak_voxel` normalization so that the strongest original support voxel equals `amplitude_snr`. The floating particle volume therefore represents signal only, with no background or noise.

A newly constructed `Particle` is always unrotated: `theta == phi == 0`. These fields are not constructor arguments. `ParticleVolume.rotate()` applies exactly the same centered, `reshape=False` coordinate sequence as `Volume.align`: `theta` in the `(y, x)` plane followed by `-phi` in the `(z, x)` plane. It returns a new `ParticleVolume`; each copied particle records the requested angles and has an eagerly updated continuous `position_zyx`. This avoids requiring downstream consumers to reinterpret an original position using the volume center. The original particle and volume are unchanged.

The original `support_values` describe the analytically rasterized particle and are not valid after image interpolation. A rotated particle therefore marks these values dirty, and accessing `support_values` raises `RuntimeError`. Rotated particles are tracking/measurement objects and must not be reinjected into another volume.

For a rotated particle, `support_zyx` contains the rounded, clipped transformation of its geometric support and `measurement_support_top_k` separately contains the compact measurement support. The original support is transformed into an output-space bounding region, expanded by an interpolation-order halo, and clipped to the output shape. The strongest `top_k` locations are then selected from the actual rotated particle-only volume—not from stale analytical values and not from a noisy combined volume. This makes the selected locations follow interpolation-induced peak movement. `measurement_top_k` records the number retained.

`ParticleStateMeasurement` identifies a state with `(bit_depth, theta, phi)`. Its top-k candidate locations come from the matching particle-only state: directly from the unrotated support for an original particle, or from `measurement_support_top_k` for a rotated particle. It stores both the particle-only reference values and the values sampled from the quantized combined particle-plus-noise volume at those same locations. The operational peak samples the combined volume at the strongest particle-only location, so noise cannot change which top-k candidate supplies the peak. A top-k sum is a compact core measurement rather than total Gaussian mass, since signal outside the strongest k voxels is excluded. This particle-only attribution assumes particle search regions do not overlap; overlapping particles require separate per-particle signal layers."""


@dataclass
class Volume_wrapper(Volume):
    """
    Encapsulates a volume of float32 via the Volume object.
    """

    volume: Volume = field(init=False)

    def __init__(self, type: DTypeLike, voxels_z: int, voxels_xy: int) -> None:
        self.volume = Volume(
            type=type,
            voxels_z=voxels_z,
            voxels_xy=voxels_xy,
            noise=None,
            cylinder=None,
        )

    def get_array(self) -> NDArray:
        """Return the underlying NumPy array representing the volume."""
        if self.volume.data is None:
            raise ValueError("Volume data has not been initialized.")
        return self.volume.data

    def rotate(self, theta=0.0, phi=0.0, order: int = 1) -> Self:
        """Return an independently stored, rotated wrapper of the same type."""
        rotated = copy.copy(self)
        rotated.volume = self.volume.align(theta=theta, phi=phi, order=order)
        return rotated


@dataclass
class VolumeUint(Volume_wrapper):
    """
    Object that does the direct conversion from a noisy floating point volume to the uint type
    """

    target_std_scale: float = 0.016

    def __init__(
        self,
        vol_float: "VolumeFloat",
        type: np.uint8 | np.uint16,
        target_std_scale=0.016,
    ) -> None:
        self.target_std_scale = target_std_scale
        target_std = target_std_scale * np.iinfo(type).max
        dc = DigitalNoiseConverter(target_noise_std=target_std, data_type=type)
        self.volume = dc.convert_volume(vol_float.volume)


@dataclass
class VolumeFloat(Volume_wrapper):
    def __init__(self, voxels_z: int, voxels_xy: int) -> None:
        super().__init__(type=np.float32, voxels_z=voxels_z, voxels_xy=voxels_xy)

    def add(self, other_vol: "VolumeFloat") -> None:
        """Adds the data from another VolumeFloat to this one."""
        if isinstance(other_vol, VolumeFloat):
            if self.volume.valid_bounds != other_vol.volume.valid_bounds:
                raise ValueError("Volume bounds do not have matching bounds")
            self.volume.data += other_vol.volume.data
        else:
            raise TypeError("Unsupported type for addition")

    @classmethod
    def subtract(cls, vol1: "VolumeUint", vol2: "VolumeUint") -> "VolumeFloat":
        """Subtracts vol2 from vol1 and returns a new VolumeFloat."""
        if not isinstance(vol1, VolumeUint) or not isinstance(vol2, VolumeUint):
            raise TypeError("Both arguments must be of type VolumeUint")
        if vol1.volume.valid_bounds != vol2.volume.valid_bounds:
            raise ValueError("Volume bounds do not have matching bounds")
        data1 = vol1.volume.data
        data2 = vol2.volume.data
        if data1 is None or data2 is None:
            raise ValueError("Volume data must be initialized")
        result = cls(
            voxels_z=data1.shape[0],
            voxels_xy=data1.shape[1],
        )

        result.volume.data = np.subtract(data1, data2, dtype=np.float32)
        return result


# endregion notebook cell 39


# region notebook cell 41


def rasterize_gaussian_particle(
    position_zyx: np.ndarray,
    sigma: float | np.ndarray,
    amplitude: float,
    truncation_sigma: float = 2.5,
    volume_shape_zyx: tuple[int, int, int] | None = None,
    amplitude_mode: Literal[
        "peak_voxel",
        "total_intensity",
        "continuous_peak",
    ] = "peak_voxel",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the support indices and voxel-integrated values of an
    axis-aligned 3D Gaussian particle.

    Integer coordinates are assumed to represent voxel centers.

    Parameters
    ----------
    position_zyx
        Continuous particle center in (z, y, x) coordinates.

    sigma
        Gaussian standard deviation in voxel units. A scalar creates an
        isotropic particle. An array-like value must have shape (3,) and
        specifies sigma in (z, y, x) order.

    amplitude
        Particle amplitude. Its interpretation is controlled by
        ``amplitude_mode``.

    truncation_sigma
        Support radius as a multiple of ``sigma``.

    volume_shape_zyx
        Optional volume shape in (z, y, x) order. When provided, support
        outside the volume is removed. Values are normalized before this
        clipping, so particles at the boundary are genuinely truncated.

    amplitude_mode
        ``"peak_voxel"``:
            The largest rasterized voxel value equals ``amplitude``.

        ``"total_intensity"``:
            The sum of the unbounded support values equals ``amplitude``.

        ``"continuous_peak"``:
            The continuous Gaussian peak equals ``amplitude``.

    Returns
    -------
    support_zyx
        Integer array of shape (N, 3), containing support indices in
        (z, y, x) order.

    support_values
        Floating-point array of shape (N,), containing the signal to add
        at each support index.
    """
    position_zyx = np.asarray(position_zyx, dtype=np.float64)

    if position_zyx.shape != (3,):
        raise ValueError("position_zyx must have shape (3,)")

    if not np.all(np.isfinite(position_zyx)):
        raise ValueError("position_zyx must contain finite values")

    sigma_zyx = np.asarray(sigma, dtype=np.float64)
    if sigma_zyx.ndim == 0:
        sigma_zyx = np.full(3, sigma_zyx.item(), dtype=np.float64)
    elif sigma_zyx.shape != (3,):
        raise ValueError("sigma must be a scalar or have shape (3,) in zyx order")

    if not np.all(np.isfinite(sigma_zyx)) or np.any(sigma_zyx <= 0):
        raise ValueError("all sigma values must be finite and positive")

    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")

    if not np.isfinite(truncation_sigma) or truncation_sigma <= 0:
        raise ValueError("truncation_sigma must be finite and positive")

    radii_zyx = truncation_sigma * sigma_zyx

    # Candidate voxel cubes that intersect the truncation bounding box.
    lower_zyx = np.ceil(position_zyx - radii_zyx - 0.5).astype(np.int64)

    upper_zyx = np.floor(position_zyx + radii_zyx + 0.5).astype(np.int64)

    z_indices = np.arange(lower_zyx[0], upper_zyx[0] + 1)
    y_indices = np.arange(lower_zyx[1], upper_zyx[1] + 1)
    x_indices = np.arange(lower_zyx[2], upper_zyx[2] + 1)

    zz, yy, xx = np.meshgrid(
        z_indices,
        y_indices,
        x_indices,
        indexing="ij",
    )

    candidate_zyx = np.column_stack((zz.ravel(), yy.ravel(), xx.ravel()))

    # Minimum distance from the particle center to each voxel cube.
    # This selects every voxel whose cube intersects the ellipsoidal
    # truncation region in sigma-normalized coordinates.
    cube_distance_zyx = np.maximum(
        np.abs(candidate_zyx - position_zyx) - 0.5,
        0.0,
    )

    normalized_cube_distance = cube_distance_zyx / sigma_zyx
    intersects_ellipsoid = (
        np.sum(normalized_cube_distance**2, axis=1) <= truncation_sigma**2
    )

    support_zyx = candidate_zyx[intersects_ellipsoid]

    if support_zyx.size == 0:
        raise ValueError("Particle support is empty")

    # Integrate a unit-total Gaussian independently along each axis.
    voxel_lower_zyx = support_zyx - 0.5
    voxel_upper_zyx = support_zyx + 0.5
    scale_zyx = np.sqrt(2.0) * sigma_zyx

    axis_weights = 0.5 * (
        erf((voxel_upper_zyx - position_zyx) / scale_zyx)
        - erf((voxel_lower_zyx - position_zyx) / scale_zyx)
    )

    weights = np.prod(axis_weights, axis=1)

    if amplitude_mode == "peak_voxel":
        support_values = amplitude * weights / weights.max()

    elif amplitude_mode == "total_intensity":
        support_values = amplitude * weights / weights.sum()

    elif amplitude_mode == "continuous_peak":
        integral_scale = np.prod(np.sqrt(2.0 * np.pi) * sigma_zyx)
        support_values = amplitude * integral_scale * weights

    else:
        raise ValueError(
            "amplitude_mode must be 'peak_voxel', "
            "'total_intensity', or 'continuous_peak'"
        )

    # Clip only after normalization. This prevents a boundary-clipped
    # particle from being artificially rescaled.
    if volume_shape_zyx is not None:
        shape_zyx = np.asarray(volume_shape_zyx, dtype=np.int64)

        if shape_zyx.shape != (3,) or np.any(shape_zyx <= 0):
            raise ValueError("volume_shape_zyx must contain three positive integers")

        inside_volume = np.all(
            (support_zyx >= 0) & (support_zyx < shape_zyx),
            axis=1,
        )

        support_zyx = support_zyx[inside_volume]
        support_values = support_values[inside_volume]

    return support_zyx, support_values


# endregion notebook cell 41


# region notebook cell 42


@dataclass
class Particle:
    """
    Represents a 3D Gaussian particle with a position, amplitude, and spatial extent.

    The particle is represented in floating point format where the amplitude is expressed as a signal-to-noise
    ratio (SNR). Particles are meant to be injected into a 3D volume of type float32 representing a standard
    noise volume with mean 0 and unit variance.
    """

    position_zyx: np.ndarray  # continuous (z, y, x) particle center
    amplitude_snr: float
    spatial_sigma: np.ndarray | float  # scalar or (z, y, x)
    particle_id: int = -1

    truncation_sigma: float = field(init=False, default=3.5)
    original_position_zyx: np.ndarray = field(init=False, repr=False)
    theta: float = field(init=False, default=0.0)
    phi: float = field(init=False, default=0.0)
    measurement_top_k: int = 7
    measurement_support_top_k: np.ndarray = field(
        init=False,
        repr=False,
        default_factory=lambda: np.empty((0, 3), dtype=np.int64),
    )
    support_zyx: np.ndarray = field(init=False, repr=False)
    _support_values: np.ndarray | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.position_zyx = np.asarray(self.position_zyx, dtype=np.float64)
        if self.position_zyx.shape != (3,):
            raise ValueError("position_zyx must have shape (3,)")
        if not np.all(np.isfinite(self.position_zyx)):
            raise ValueError("position_zyx must contain finite values")
        if (
            not isinstance(self.measurement_top_k, (int, np.integer))
            or self.measurement_top_k <= 0
        ):
            raise ValueError("measurement_top_k must be a positive integer")
        self.original_position_zyx = self.position_zyx.copy()

        sigma_zyx = np.asarray(self.spatial_sigma, dtype=np.float64)
        if sigma_zyx.ndim == 0:
            sigma_zyx = np.full(3, sigma_zyx.item(), dtype=np.float64)
        elif sigma_zyx.shape != (3,):
            raise ValueError(
                "spatial_sigma must be a scalar or have shape (3,) in zyx order"
            )
        if not np.all(np.isfinite(sigma_zyx)) or np.any(sigma_zyx <= 0):
            raise ValueError("all spatial_sigma values must be finite and positive")

        self.spatial_sigma = sigma_zyx
        self._create_particle()

    @property
    def is_rotated(self) -> bool:
        """Whether this particle represents a geometrically rotated result."""
        return self.theta != 0.0 or self.phi != 0.0

    @property
    def support_values(self) -> np.ndarray:
        """Return original raster values, rejecting stale rotated values."""
        if self._support_values is None:
            raise RuntimeError(
                "support_values are unavailable after rotation because their "
                "values depend on image interpolation; sample the rotated "
                "particle-only volume at measurement_support_top_k instead"
            )
        return self._support_values

    def _create_particle(self) -> None:
        self.support_zyx, self._support_values = rasterize_gaussian_particle(
            position_zyx=self.position_zyx,
            sigma=self.spatial_sigma,
            amplitude=self.amplitude_snr,
            truncation_sigma=self.truncation_sigma,
        )

    def strongest_support_zyx(self, top_k: int) -> np.ndarray:
        """Return locations of the strongest original rasterized values."""
        if not isinstance(top_k, (int, np.integer)) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        values = self.support_values
        top_k = min(top_k, len(values))
        selected = np.argpartition(values, len(values) - top_k)[-top_k:]
        selected = selected[np.argsort(values[selected])[::-1]]
        return self.support_zyx[selected].copy()

    def measure_top_k_signal(
        self, particle_only_data: np.ndarray, top_k: int | None = None
    ) -> float:
        """Sum core signal at top-k locations in particle-only data.

        Original particles select their strongest ``top_k`` locations,
        defaulting to ``measurement_top_k`` when ``top_k`` is omitted.
        Rotated particles already store the
        locations selected from the transformed data; for them ``top_k`` may
        be omitted or must match ``measurement_top_k``.
        """
        if particle_only_data.ndim != 3:
            raise ValueError("particle_only_data must be a 3D zyx array")

        if not self.is_rotated:
            if top_k is None:
                top_k = self.measurement_top_k
            indices = self.strongest_support_zyx(top_k)
        else:
            if top_k is not None and top_k != self.measurement_top_k:
                raise ValueError(
                    f"top_k={top_k} does not match the rotated particle's "
                    f"measurement_top_k={self.measurement_top_k}"
                )
            indices = self.measurement_support_top_k

        if indices.size == 0:
            return 0.0

        shape_zyx = np.asarray(particle_only_data.shape, dtype=np.int64)
        if not np.all((indices >= 0) & (indices < shape_zyx)):
            raise IndexError("particle measurement support is outside the data shape")

        return float(
            np.sum(
                particle_only_data[indices[:, 0], indices[:, 1], indices[:, 2]],
                dtype=np.float64,
            )
        )

    def inject_into(self, volume: np.ndarray) -> None:
        """Add the in-bounds portion of the particle to a volume.

        Args:
            volume (np.ndarray): 3D volume of data into which the particle will be injected.

        Note:
            The particle is added in-place. Support entries outside the volume
            are omitted and reported with a warning; the stored particle support
            itself is not modified.
        """
        if volume.ndim != 3:
            raise ValueError("volume must be a 3D array in zyx order")

        indices = self.support_zyx
        shape_zyx = np.asarray(volume.shape, dtype=np.int64)
        valid = np.all((indices >= 0) & (indices < shape_zyx), axis=1)

        if not np.all(valid):
            invalid_indices = indices[~valid]
            violations = []
            for axis, axis_name in enumerate("zyx"):
                axis_invalid = invalid_indices[
                    (invalid_indices[:, axis] < 0)
                    | (invalid_indices[:, axis] >= shape_zyx[axis]),
                    axis,
                ]
                if axis_invalid.size:
                    violations.append(
                        f"{axis_name} axis {axis} (valid 0..{shape_zyx[axis] - 1}): "
                        f"{np.unique(axis_invalid).tolist()}"
                    )
            warnings.warn(
                f"Particle {self.particle_id} has {len(invalid_indices)} support "
                f"indices outside volume shape {volume.shape}; omitting "
                f"them. Violating indices by axis: {'; '.join(violations)}",
                RuntimeWarning,
                stacklevel=2,
            )

        indices = indices[valid]
        values = self.support_values[valid]
        if indices.size == 0:
            return

        volume[
            indices[:, 0],
            indices[:, 1],
            indices[:, 2],
        ] += values


@dataclass
class ParticleVolume(VolumeFloat):
    """
    Represents an entire volume of partcles superimposed onto a zero background
    """

    particles: list[Particle] = field(default_factory=list)

    def __init__(self, voxels_z, voxels_xy, particles: list[Particle]) -> None:
        super().__init__(voxels_z=voxels_z, voxels_xy=voxels_xy)
        self.particles = []
        self.add_particles(particles)

    def add_particles(self, particles: list[Particle]) -> None:
        for particle in particles:
            self.add_particle(particle)

    def add_particle(self, particle: Particle) -> None:
        particle.inject_into(self.get_array())
        self.particles.append(particle)

    def rotate(
        self,
        theta=0.0,
        phi=0.0,
        order: int = 1,
        top_k: int | None = None,
    ) -> Self:
        """Rotate signal data and create matching particle measurements."""
        rotated = super().rotate(theta=theta, phi=phi, order=order)
        rotated.particles = self.rotate_particles(
            particles=self.particles,
            rotated_particle_data=rotated.get_array(),
            theta=theta,
            phi=phi,
            order=order,
            top_k=top_k,
        )
        return rotated

    @staticmethod
    def rotate_particles(
        particles: list[Particle],
        rotated_particle_data: np.ndarray,
        theta: float = 0.0,
        phi: float = 0.0,
        order: int = 1,
        top_k: int | None = None,
    ) -> list[Particle]:
        """Return particle copies tracking their rotated signal locations.

        The continuous center and original support centers are mapped with the
        forward transform corresponding to ``Volume.align``. For each particle,
        a clipped bounding ROI around that transformed support is searched in
        the actual rotated particle-only data. ``measurement_support_top_k``
        on the returned particle contains the strongest ``top_k`` locations
        from that ROI, while ``support_zyx`` tracks the rounded, transformed
        geometric support. When ``top_k`` is omitted, each particle's
        configured ``measurement_top_k`` is used. An explicit override must
        match that configured value.
        Analytical support values are invalidated because interpolation changes
        them.
        """
        if rotated_particle_data.ndim != 3:
            raise ValueError("rotated_particle_data must be a 3D zyx array")
        if top_k is not None and (
            not isinstance(top_k, (int, np.integer)) or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None")
        if not 1 <= order <= 5:
            raise ValueError(f"order must be between 1 and 5, got {order}")
        if any(particle.is_rotated for particle in particles):
            raise ValueError("rotate_particles expects unrotated source particles")

        if theta == 0.0 and phi == 0.0:
            return copy.deepcopy(particles)

        shape_zyx = rotated_particle_data.shape
        shape_array = np.asarray(shape_zyx, dtype=np.int64)
        interpolation_halo = 1 if order == 1 else 2 * order
        rotated_particles = []

        for particle in particles:
            particle_top_k = particle.measurement_top_k
            if top_k is not None and top_k != particle_top_k:
                raise ValueError(
                    f"top_k={top_k} does not match particle "
                    f"{particle.particle_id} measurement_top_k={particle_top_k}"
                )
            rotated_particle = copy.deepcopy(particle)
            rotated_particle.position_zyx = forward_align_points_zyx(
                particle.position_zyx[None, :], theta, phi, shape_zyx
            )[0]
            transformed_support = forward_align_points_zyx(
                particle.support_zyx, theta, phi, shape_zyx
            )
            geometric_support = np.rint(transformed_support).astype(np.int64)
            geometric_support = geometric_support[
                np.all(
                    (geometric_support >= 0) & (geometric_support < shape_array),
                    axis=1,
                )
            ]
            rotated_particle.support_zyx = np.unique(geometric_support, axis=0)

            lower_zyx = (
                np.floor(transformed_support.min(axis=0)).astype(np.int64)
                - interpolation_halo
            )
            upper_zyx = (
                np.ceil(transformed_support.max(axis=0)).astype(np.int64)
                + interpolation_halo
            )
            lower_zyx = np.maximum(lower_zyx, 0)
            upper_zyx = np.minimum(upper_zyx, shape_array - 1)

            if np.any(lower_zyx > upper_zyx):
                warnings.warn(
                    f"Particle {particle.particle_id} rotates entirely outside "
                    f"volume shape {shape_zyx}; its measurement support is empty",
                    RuntimeWarning,
                    stacklevel=2,
                )
                rotated_particle.measurement_support_top_k = np.empty(
                    (0, 3), dtype=np.int64
                )
                rotated_particle.measurement_top_k = 0
            else:
                axes = [
                    np.arange(lower_zyx[axis], upper_zyx[axis] + 1) for axis in range(3)
                ]
                zz, yy, xx = np.meshgrid(*axes, indexing="ij")
                candidate_zyx = np.column_stack((zz.ravel(), yy.ravel(), xx.ravel()))
                candidate_values = rotated_particle_data[
                    candidate_zyx[:, 0],
                    candidate_zyx[:, 1],
                    candidate_zyx[:, 2],
                ]
                retained_k = min(particle_top_k, len(candidate_values))
                selected = np.argpartition(
                    candidate_values, len(candidate_values) - retained_k
                )[-retained_k:]
                selected = selected[np.argsort(candidate_values[selected])[::-1]]
                rotated_particle.measurement_support_top_k = candidate_zyx[selected]
                rotated_particle.measurement_top_k = retained_k

            rotated_particle.theta = float(theta)
            rotated_particle.phi = float(phi)
            rotated_particle._support_values = None
            rotated_particles.append(rotated_particle)

        return rotated_particles


@dataclass
class ParticleVolumeUint(VolumeUint):
    """Quantized particle volume that retains particle metadata."""

    particles: list[Particle] = field(default_factory=list)

    def __init__(
        self,
        source: ParticleVolume,
        type: np.uint8 | np.uint16,
        target_std_scale: float = 0.016,
    ) -> None:
        if not isinstance(source, ParticleVolume):
            raise TypeError("source must be a ParticleVolume")

        super().__init__(
            vol_float=source,
            type=type,
            target_std_scale=target_std_scale,
        )

        # Avoid sharing mutable Particle objects with the float volume.
        self.particles = copy.deepcopy(source.particles)

    def rotate(
        self,
        theta: float = 0.0,
        phi: float = 0.0,
        order: int = 1,
        top_k: int | None = None,
    ) -> Self:
        rotated = super().rotate(theta=theta, phi=phi, order=order)

        rotated.particles = ParticleVolume.rotate_particles(
            particles=self.particles,
            rotated_particle_data=rotated.get_array(),
            theta=theta,
            phi=phi,
            order=order,
            top_k=top_k,
        )
        return rotated


# endregion notebook cell 42


# region notebook cell 43


def create_uniform_particle_volume(
    voxels_z: int,
    voxels_xy: int,
    sigma: float | tuple[float],
    amplitude: float,
    z_start_rel,
    z_end_rel,
    delta_z_rel,
    delta_angle_deg: float = 15.0,
    stop_angle_deg: float = 360.0,
    measurement_top_k: int = 7,
) -> ParticleVolume:
    """Create a uniform particle volume.

    Args:
        voxels_z (int): Number of voxels along the z-axis.
        voxels_xy (int): Number of voxels along the x and y axes.
        sigma (float | tuple[float]): Standard deviation of the particle distribution.
        amplitude (float): The amplitude of the particles.
        z_start_rel (float): The relative starting position along the z-axis.
        z_end_rel (float): The relative ending position along the z-axis.
        delta_z_rel (float): The relative spacing between consecutive z-planes.
        delta_angle_deg (float): The angular spacing between points on each circle, in degrees.
        stop_angle_deg (float): The maximum angular extent of the polar grid, in degrees.
        measurement_top_k (int): The number of the highest particle measurements to consider.

    Returns:
        ParticleVolume: The created uniform particle volume.

    Notes:
    - The relative z-positions are in uints [0,1]
    """

    def _create_polar_grid(delta_radius: float) -> list[tuple[float, float]]:
        """Create a polar grid of points in Cartesian coordinates.

        The grid is defined by concentric circles with radii increasing by `delta_radius`
        and points along each circle separated by `delta_angle_deg` degrees.

        The polar grid is then replicated along the z-axis to create a 3D cylindrical grid.


        Args:
            radius (float): The maximum radius of the polar grid.
            delta_radius (float): The radial spacing between consecutive circles.
            delta_angle_deg (float): The angular spacing between points on each circle, in degrees.

        Returns:
            list[tuple[float, float]]: A list of points in Cartesian coordinates (x, y).
        """
        radii = []
        radius = voxels_xy // 2
        r = delta_radius
        while r < radius:
            radii.append(r)
            r += delta_radius

        pts = []

        for angle in np.arange(
            0,
            np.deg2rad(stop_angle_deg) - np.deg2rad(delta_angle_deg) + 1e-6,
            np.deg2rad(delta_angle_deg),
        ):
            for r in radii:
                pts.append((r * np.cos(angle), r * np.sin(angle)))

        return pts

    def _create_z_grid() -> np.ndarray:
        """Create a grid of z positions within the volume."""
        z_start = round(z_start_rel * voxels_z)
        z_end = round(z_end_rel * voxels_z)
        delta_z = round(delta_z_rel * voxels_z)
        return np.arange(z_start, z_end, delta_z)

    def _create_uniform_cylindrical_grid() -> NDArray:
        """Here we copy the uniform polar grid into a 3D volume by adding z positions."""
        positions_yx = [
            (y + voxels_xy // 2, x + voxels_xy // 2)
            for x, y in _create_polar_grid(delta_radius=voxels_xy // 8)
        ]

        # Sample z in steps of 1/8 the total height
        positions_z = _create_z_grid()

        n_z = len(positions_z)
        n_yx = len(positions_yx)
        positions_zyx = np.empty((n_z * n_yx, 3), dtype=np.float32)
        positions_zyx[:, 0] = np.repeat(positions_z, n_yx)
        positions_zyx[:, 1:] = np.tile(positions_yx, (n_z, 1))
        return positions_zyx

    positions_zyx = _create_uniform_cylindrical_grid()
    particles = []
    for n, pos in enumerate(positions_zyx):
        particles.append(
            Particle(
                position_zyx=pos,
                amplitude_snr=amplitude,
                spatial_sigma=sigma,
                particle_id=n,
                measurement_top_k=measurement_top_k,
            )
        )

    particle_vol = ParticleVolume(
        voxels_z=voxels_z, voxels_xy=voxels_xy, particles=particles
    )
    return particle_vol


# endregion notebook cell 43


# region notebook cell 45
"""
### Simulated Noise
Here we create a convenience noise class that wraps the NoiseSampler.
"""


@dataclass
class StandardNoiseVolume:
    """
    Creates a standard noise volume with mean zero and unit variance and provides a getter that returns a noise volume copy of the noise.
    Represents an entire volume of standard noise with mean zero and unit variance with a getter that returns a copy of the
    noise after
    """

    sampler: NoiseSampler = field(init=False, default_factory=NoiseSampler)

    def __init__(
        self,
        voxels_z: int = 128,
        voxels_xy: int = 256,
        rho1_z: float = 0.5,
        rho1_xy: float = 0.5,
    ) -> None:
        self.sampler = NoiseSampler(
            v_z=voxels_z, v_xy=voxels_xy, rho1=rho1_z, rho_xy=rho1_xy
        )

    def get_array(self) -> NDArray[np.float32]:
        return self.sampler.noise_vol.data

    def get_copy(self) -> NDArray[np.float32]:
        return self.sampler.noise_vol.data.copy()

    def get_quantized_noise(
        self,
        target_std: float,
        uint_type: type = np.uint8,
    ) -> Volume:
        orig_type = self.sampler.analysis_dtype
        self.sampler.analysis_dtype = uint_type

        quantized_noise = self.sampler.get_noise_volume(target_std=target_std)
        self.sampler.analysis_dtype = orig_type

        return quantized_noise


# endregion notebook cell 45


# region notebook cell 47
"""
### Particle Metrics
A `ParticleStateMeasurement` is an immutable record of one particle in one acquisition state. The state key is `(bit_depth, theta, phi)`, where `bit_depth` is exactly `np.float32`, `np.uint16`, or `np.uint8`. This makes bit depth and geometric orientation explicit parts of the measurement identity.

For an unrotated state, the top-k locations are selected from the particle's unrotated particle-only volume. For a rotated state, the locations previously selected from the actual rotated particle-only volume are used. The particle-only values provide the signal-based ranking; the combined `particle + noise` volume is sampled at those same locations. Consequently, each state retains both the reference signal values and the observed, noise-contaminated values without assuming that quantization or transformation is additive.

The factory accepts the established noise mean and the profile returned by `estimate_volume_noise()`. Because that function estimates noise from adjacent z-slice differences, the two neighboring variance estimates are averaged for each interior x-y plane and the square root is taken; an endpoint uses its sole neighbor. The peak CNR is `(combined_peak - noise_mean) / plane_noise_std` at the z plane of the strongest particle-only top-k location. The combined volume is sampled at that signal-defined location, avoiding positive selection bias from maximizing noisy candidates.

Integrated contrast is the combined top-k sum minus `top_k * noise_mean`. Its noise standard deviation uses the practical independence approximation `sqrt(sum(sigma_z**2))`, which reduces to `sqrt(k) * sigma` when all selected voxels share one plane estimate. It does not model spatial covariance and should not be interpreted as the exact standard deviation of correlated integrated noise. `ParticleTransformationMeasurement` reports before/after retention for peak contrast, integrated contrast, peak CNR, and integrated CNR.
"""


@dataclass(frozen=True, eq=False)
class ParticleStateMeasurement:
    """Immutable signal and observed samples for one particle state.

    Top-k locations are defined by the particle-only volume for this state.
    Values from the particle-only and combined volumes are then sampled at
    exactly those locations. Plane noise estimates provide practical peak
    and integrated CNR metrics for the observed combined-volume values.
    """

    particle_id: int
    state: tuple[DTypeLike, float, float]
    position_zyx: np.ndarray
    top_k: int  # Number of locations used for the integrated measurement.
    top_k_zyx: np.ndarray  # Particle-only reference locations.
    particle_only_top_k_values: np.ndarray  # Reference signal values.
    combined_top_k_values: np.ndarray  # Observed particle-plus-noise values.
    noise_mean: float  # Established mean/background level for this state.
    noise_std_by_z: np.ndarray  # Estimated noise std for each x-y plane.

    def __post_init__(self) -> None:
        if not isinstance(self.particle_id, (int, np.integer)):
            raise TypeError("particle_id must be an integer")
        if not isinstance(self.state, tuple) or len(self.state) != 3:
            raise TypeError("state must be a (bit_depth, theta, phi) tuple")
        bit_depth = np.dtype(self.state[0]).type
        allowed_bit_depths = (np.float32, np.uint16, np.uint8)
        if bit_depth not in allowed_bit_depths:
            raise ValueError(
                "state bit_depth must be np.float32, np.uint16, or np.uint8"
            )
        theta = float(self.state[1])
        phi = float(self.state[2])
        if not np.isfinite(theta) or not np.isfinite(phi):
            raise ValueError("state theta and phi must be finite")
        if not isinstance(self.top_k, (int, np.integer)) or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")

        position_zyx = np.asarray(self.position_zyx, dtype=np.float64).copy()
        top_k_zyx = np.asarray(self.top_k_zyx, dtype=np.int64).copy()
        particle_only_values = np.asarray(
            self.particle_only_top_k_values, dtype=np.float64
        ).copy()
        combined_values = np.asarray(
            self.combined_top_k_values, dtype=np.float64
        ).copy()
        noise_std_by_z = np.asarray(self.noise_std_by_z, dtype=np.float64).copy()

        if position_zyx.shape != (3,) or not np.all(np.isfinite(position_zyx)):
            raise ValueError("position_zyx must have shape (3,) and be finite")
        if top_k_zyx.shape != (self.top_k, 3):
            raise ValueError("top_k_zyx must have shape (top_k, 3)")
        if particle_only_values.shape != (self.top_k,):
            raise ValueError("particle_only_top_k_values must have shape (top_k,)")
        if combined_values.shape != (self.top_k,):
            raise ValueError("combined_top_k_values must have shape (top_k,)")
        if not np.all(np.isfinite(particle_only_values)):
            raise ValueError("particle_only_top_k_values must be finite")
        if not np.all(np.isfinite(combined_values)):
            raise ValueError("combined_top_k_values must be finite")
        if not np.isfinite(self.noise_mean):
            raise ValueError("noise_mean must be finite")
        if noise_std_by_z.ndim != 1 or len(noise_std_by_z) == 0:
            raise ValueError("noise_std_by_z must be a nonempty 1D array")
        valid_noise_std = np.isnan(noise_std_by_z) | (
            np.isfinite(noise_std_by_z) & (noise_std_by_z > 0.0)
        )
        if not np.all(valid_noise_std):
            raise ValueError("noise_std_by_z values must be positive or NaN")
        if self.top_k and (
            np.any(top_k_zyx[:, 0] < 0)
            or np.any(top_k_zyx[:, 0] >= len(noise_std_by_z))
        ):
            raise ValueError("top-k z indices must index noise_std_by_z")

        descending = np.argsort(particle_only_values)[::-1]
        top_k_zyx = top_k_zyx[descending]
        particle_only_values = particle_only_values[descending]
        combined_values = combined_values[descending]
        position_zyx.flags.writeable = False
        top_k_zyx.flags.writeable = False
        particle_only_values.flags.writeable = False
        combined_values.flags.writeable = False
        noise_std_by_z.flags.writeable = False
        object.__setattr__(self, "particle_id", int(self.particle_id))
        object.__setattr__(self, "state", (bit_depth, theta, phi))
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "position_zyx", position_zyx)
        object.__setattr__(self, "top_k_zyx", top_k_zyx)
        object.__setattr__(self, "particle_only_top_k_values", particle_only_values)
        object.__setattr__(self, "combined_top_k_values", combined_values)
        object.__setattr__(self, "noise_mean", float(self.noise_mean))
        object.__setattr__(self, "noise_std_by_z", noise_std_by_z)

    @staticmethod
    def _to_plane_noise_std(noise_std_profile: np.ndarray, voxels_z: int) -> np.ndarray:
        """Convert adjacent-slice estimates to one noise std per z plane.

        ``estimate_volume_noise`` returns one corrected value per z plane.
        Legacy adjacent-slice profiles with ``voxels_z - 1`` values are also
        accepted and mapped onto the planes.
        """
        profile = np.asarray(noise_std_profile, dtype=np.float64)
        if profile.ndim != 1:
            raise ValueError("noise_std_profile must be a 1D array")
        if profile.shape == (voxels_z,):
            return profile.copy()
        if voxels_z < 2 or profile.shape != (voxels_z - 1,):
            raise ValueError(
                "noise_std_profile must have voxels_z - 1 values from "
                "estimate_volume_noise, or one value per z plane"
            )

        plane_std = np.empty(voxels_z, dtype=np.float64)
        plane_std[0] = profile[0]
        plane_std[-1] = profile[-1]
        if voxels_z > 2:
            neighbors = np.stack((profile[:-1], profile[1:]))
            finite = np.isfinite(neighbors)
            count = finite.sum(axis=0)
            variance_sum = np.where(finite, neighbors**2, 0.0).sum(axis=0)
            plane_std[1:-1] = np.sqrt(
                np.divide(
                    variance_sum,
                    count,
                    out=np.full_like(variance_sum, np.nan),
                    where=count > 0,
                )
            )
        return plane_std

    @classmethod
    def from_particle(
        cls,
        particle: Particle,
        particle_only_data: np.ndarray,
        combined_data: np.ndarray,
        state: tuple[DTypeLike, float, float],
        noise_mean: float,
        noise_std_profile: np.ndarray,
        top_k: int | None = None,
    ) -> Self:
        """Sample particle-only and combined data at signal-defined locations."""
        if particle_only_data.ndim != 3 or combined_data.ndim != 3:
            raise ValueError("particle_only_data and combined_data must be 3D")
        if particle_only_data.shape != combined_data.shape:
            raise ValueError("particle_only_data and combined_data shapes must match")
        if not isinstance(state, tuple) or len(state) != 3:
            raise TypeError("state must be a (bit_depth, theta, phi) tuple")
        bit_depth = np.dtype(state[0]).type
        if particle_only_data.dtype.type is not bit_depth:
            raise ValueError("particle_only_data dtype must match state bit_depth")
        if combined_data.dtype.type is not bit_depth:
            raise ValueError("combined_data dtype must match state bit_depth")
        if not np.isclose(float(state[1]), particle.theta) or not np.isclose(
            float(state[2]), particle.phi
        ):
            raise ValueError("state angles must match the particle angles")

        shape_zyx = np.asarray(particle_only_data.shape, dtype=np.int64)
        noise_std_by_z = cls._to_plane_noise_std(
            noise_std_profile, particle_only_data.shape[0]
        )
        if not particle.is_rotated:
            if top_k is None:
                top_k = particle.measurement_top_k
            if not isinstance(top_k, (int, np.integer)) or top_k <= 0:
                raise ValueError("top_k must be a positive integer")
            valid = np.all(
                (particle.support_zyx >= 0) & (particle.support_zyx < shape_zyx),
                axis=1,
            )
            candidate_zyx = np.unique(particle.support_zyx[valid], axis=0)
            candidate_values = particle_only_data[
                candidate_zyx[:, 0], candidate_zyx[:, 1], candidate_zyx[:, 2]
            ]
            retained_k = min(int(top_k), len(candidate_values))
            if retained_k == 0:
                top_k_zyx = np.empty((0, 3), dtype=np.int64)
            else:
                selected = np.argpartition(
                    candidate_values, len(candidate_values) - retained_k
                )[-retained_k:]
                selected = selected[np.argsort(candidate_values[selected])[::-1]]
                top_k_zyx = candidate_zyx[selected]
        else:
            if top_k is not None and top_k != particle.measurement_top_k:
                raise ValueError(
                    f"top_k={top_k} does not match measurement_top_k="
                    f"{particle.measurement_top_k}"
                )
            retained_k = particle.measurement_top_k
            top_k_zyx = particle.measurement_support_top_k

        if retained_k:
            particle_only_values = particle_only_data[
                top_k_zyx[:, 0], top_k_zyx[:, 1], top_k_zyx[:, 2]
            ]
            combined_values = combined_data[
                top_k_zyx[:, 0], top_k_zyx[:, 1], top_k_zyx[:, 2]
            ]
        else:
            particle_only_values = np.empty(0, dtype=np.float64)
            combined_values = np.empty(0, dtype=np.float64)

        return cls(
            particle_id=particle.particle_id,
            state=state,
            position_zyx=particle.position_zyx,
            top_k=retained_k,
            top_k_zyx=top_k_zyx,
            particle_only_top_k_values=particle_only_values,
            combined_top_k_values=combined_values,
            noise_mean=noise_mean,
            noise_std_by_z=noise_std_by_z,
        )

    @property
    def bit_depth(self) -> type[np.generic]:
        return self.state[0]

    @property
    def theta(self) -> float:
        return self.state[1]

    @property
    def phi(self) -> float:
        return self.state[2]

    @property
    def particle_only_peak(self) -> float:
        return 0.0 if self.top_k == 0 else float(self.particle_only_top_k_values[0])

    @property
    def combined_peak_index(self) -> int | None:
        """Index of the strongest particle-only location within the top-k."""
        return None if self.top_k == 0 else 0

    @property
    def combined_peak_zyx(self) -> np.ndarray | None:
        index = self.combined_peak_index
        return None if index is None else self.top_k_zyx[index]

    @property
    def combined_peak_within_top_k(self) -> float:
        index = self.combined_peak_index
        return np.nan if index is None else float(self.combined_top_k_values[index])

    @property
    def peak_contrast(self) -> float:
        """Observed combined-volume peak above the noise mean."""
        return self.combined_peak_within_top_k - self.noise_mean

    @property
    def peak_noise_std(self) -> float:
        peak_zyx = self.combined_peak_zyx
        return np.nan if peak_zyx is None else float(self.noise_std_by_z[peak_zyx[0]])

    @property
    def peak_cnr(self) -> float:
        sigma = self.peak_noise_std
        return (
            np.nan
            if not np.isfinite(sigma) or sigma <= 0.0
            else float(self.peak_contrast / sigma)
        )

    @property
    def combined_top_k_sum(self) -> float:
        return float(np.sum(self.combined_top_k_values, dtype=np.float64))

    @property
    def integrated_contrast(self) -> float:
        """Observed top-k sum above its expected background sum."""
        return self.combined_top_k_sum - self.top_k * self.noise_mean

    @property
    def integrated_noise_std(self) -> float:
        """RSS approximation treating the selected voxels as independent."""
        if self.top_k == 0:
            return np.nan
        selected_std = self.noise_std_by_z[self.top_k_zyx[:, 0]]
        return float(np.sqrt(np.sum(selected_std**2, dtype=np.float64)))

    @property
    def integrated_cnr(self) -> float:
        sigma = self.integrated_noise_std
        return (
            np.nan
            if not np.isfinite(sigma) or sigma <= 0.0
            else float(self.integrated_contrast / sigma)
        )


@dataclass(frozen=True)
class ParticleTransformationMeasurement:
    """Validated before/after state pair for the same particle."""

    before: ParticleStateMeasurement
    after: ParticleStateMeasurement

    def __post_init__(self) -> None:
        if self.before.particle_id != self.after.particle_id:
            raise ValueError("before and after must measure the same particle_id")
        if self.before.top_k != self.after.top_k:
            raise ValueError("before and after must use the same top_k")
        if self.before.bit_depth is not self.after.bit_depth:
            raise ValueError("before and after must use the same bit_depth")

    @staticmethod
    def _ratio(after: float, before: float) -> float:
        return (
            np.nan
            if not np.isfinite(before) or before == 0.0
            else float(after / before)
        )

    @property
    def peak_contrast_retention(self) -> float:
        return self._ratio(self.after.peak_contrast, self.before.peak_contrast)

    @property
    def peak_contrast_drop(self) -> float:
        return 1.0 - self.peak_contrast_retention

    @property
    def integrated_contrast_retention(self) -> float:
        return self._ratio(
            self.after.integrated_contrast, self.before.integrated_contrast
        )

    @property
    def integrated_contrast_drop(self) -> float:
        return 1.0 - self.integrated_contrast_retention

    @property
    def peak_cnr_retention(self) -> float:
        return self._ratio(self.after.peak_cnr, self.before.peak_cnr)

    @property
    def integrated_cnr_retention(self) -> float:
        return self._ratio(self.after.integrated_cnr, self.before.integrated_cnr)


# endregion notebook cell 47
