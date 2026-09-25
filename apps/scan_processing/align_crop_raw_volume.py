#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy~=2.4.1",
#     "scipy~=1.17.1",
#     "scikit-image~=0.26.0",
#     "typer~=0.15.1",
# ]
# ///
"""Standalone postprocessing of a cylindrical-cell CT volume: .raw in, .raw out.

Edit the CONFIGURATION block below, then run the script with the name of the volume file to
process (assumed to live in VOLUME_DIR): `./align_crop_raw_volume.py scan.raw`
(or `uv run align_crop_raw_volume.py scan.raw`).

The algorithms here are adapted directly from the `scan_processing` package
(`postprocessing/{cylindrical,alignment,segmentation,cropping_radial,cropping_principal_axis}.py`
and `arraycompat/rotation.py`), stripped of GPU support, S3/API I/O, ScanInfo, and loguru so that
the script runs anywhere with CPU numpy/scipy/scikit-image.

Steps, in order (each can be switched off in the CONFIGURATION block):
    1. align       — rotate so the cylinder axis is parallel to z
    2. crop_radial — crop x/y to the can plus padding, centred on the fitted can circle
    3. mask_radial — zero voxels inside the image but outside the can plus padding
    4. crop_axial  — crop z to the can plus padding, using 1D edge detection along z

Deviations from the package, all deliberate:
    * `upper_crop_distance_voxels` is `can_height + axial_padding + 1`. The package takes the max of
      that and two slice-position-derived options, which only matter when the requested slice
      positions extend past the nominal cell; override with UPPER_CROP_DISTANCE_MM if needed.
    * Validation failures raise `ValidationError` instead of uploading debug slices.
    * `mask_radial` clips the disk to the slice shape, so it is safe to run with RUN_CROP_RADIAL off.
    * `fit_can_to_circle` checks that the circle fit converged.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from time import time

import numpy as np
import scipy.ndimage
import skimage.draw
import skimage.feature
import skimage.filters
import skimage.util
import typer
from numpy.typing import DTypeLike, NDArray
from skimage.measure import CircleModel

# ==================================================================================================
# CONFIGURATION — edit everything in this block, nothing below it
# ==================================================================================================

# --- Files ----------------------------------------------------------------------------------------
# The volume file to process is given on the command line as a filename within this directory.
# Input is a headerless .raw volume stored in (z, y, x) order; the output is written the same way.
VOLUME_DIR = Path("./Data/volumes")

# The output goes next to the input, with this appended to the file name:
# e.g. scan.raw -> scan_processed.raw
OUTPUT_NAME_SUFFIX = "_aligned"

# Overwrite the output file if it already exists
OVERWRITE_OUTPUT = False

# --- Input volume dimensions ----------------------------------------------------------------------
# Required: a .raw file has no header, so the voxel counts cannot be inferred. The dtype (uint8,
# uint16, or float32) is inferred from the file size and downcast to uint8 if needed.
VOXELS_X = 1425
VOXELS_Y = 1425
VOXELS_Z = None
VOXELS_Z_UPPERLIMIT = 6000  # Upper limit for the z dimension

# --- Cell geometry --------------------------------------------------------------------------------
VOXEL_SIZE_MM = 0.0164
CAN_DIAMETER_MM = 21.0  # Nominal can diameter
CAN_HEIGHT_MM = 70.0  # Nominal can height

# --- Steps to run ---------------------------------------------------------------------------------
RUN_ALIGN = True
RUN_CROP_RADIAL = True
RUN_MASK_RADIAL = True
RUN_CROP_AXIAL = True

# --- Alignment ------------------------------------------------------------------------------------
DOWNSAMPLING_FACTOR = 10  # Downsampling factor for the alignment segmentation
SIGMA_ALIGNMENT = 3.0  # Blurring extent in the Gaussian filter, in voxels
MAX_ROTATION_ANGLE_DEGREES = 30.0  # Applies to both theta and phi

# --- Radial cropping and masking ------------------------------------------------------------------
RADIAL_PADDING_MM = 0.5  # Padding beyond the nominal can radius
SIGMA_RADIAL_CROPPING = 3.0  # Blurring extent in the Canny filter, in voxels
RELATIVE_DETECTED_DIAMETER_TOLERANCE = (
    0.05  # Allowed error between detected and nominal diameter
)

# --- Axial cropping -------------------------------------------------------------------------------
AXIAL_PADDING_MM = 0.5  # Padding below the detected cell edge
# Peak offset is an adjustment to the detected edge position, used to move the cell up or down in the
# image. Positive values move the cell lower in the image; use a negative offset to avoid cropping
# past the end of the volume.
PEAK_OFFSET_MM = 0.0
EDGE_DETECTION_SIGMA_VOXELS = (
    1.0  # Blurring extent for edge detection's Gaussian filter
)
# Distance from the detected edge to the top of the crop. None uses can height + axial padding.
UPPER_CROP_DISTANCE_MM: float | None = None

# --- Runtime --------------------------------------------------------------------------------------
# Memory budget per rotation chunk, in bytes. The in-place rotations are applied chunk by chunk along
# the axis orthogonal to the rotation plane so that peak memory stays bounded for large volumes.
ROTATION_CHUNK_BYTES = 2 * 1024**3
VERBOSE = False  # Emit debug logging

# ==================================================================================================
# END OF CONFIGURATION
# ==================================================================================================

logger = logging.getLogger("process_raw_volume")


class ValidationError(RuntimeError):
    """Raised when a processing step detects a volume it cannot handle."""


@dataclass(frozen=True)
class Params:
    """Processing parameters, in voxels where the algorithms work in voxels.

    Mirrors the subset of `ScanInfoCylindrical` used by the four steps below.
    """

    voxel_size_mm: float

    # Cell dimensions
    can_diameter_mm: float
    can_height_mm: float

    # Alignment
    downsampling_factor: int
    sigma_alignment: float
    max_rotation_angle_degrees: float

    # Radial cropping and masking
    sigma_radial_cropping: float
    relative_detected_diameter_tolerance: float
    radius_plus_padding_voxels: int

    # Axial cropping
    axial_padding_voxels: int
    upper_crop_distance_voxels: int
    peak_offset_voxels: int
    edge_detection_sigma_voxels: float

    @classmethod
    def from_config(cls) -> Params:
        """Derive voxel quantities from the millimetre settings above, as ScanInfo does."""
        can_radius_voxels = round((CAN_DIAMETER_MM / 2) / VOXEL_SIZE_MM)
        radial_padding_voxels = round(RADIAL_PADDING_MM / VOXEL_SIZE_MM)
        axial_padding_voxels = round(AXIAL_PADDING_MM / VOXEL_SIZE_MM)
        can_height_voxels = round(CAN_HEIGHT_MM / VOXEL_SIZE_MM)

        # Distance from the detected edge to the upper edge of the cropped volume.
        # The +1 makes the upper slice inclusive, matching the package.
        if UPPER_CROP_DISTANCE_MM is not None:
            upper_crop_distance_voxels = round(UPPER_CROP_DISTANCE_MM / VOXEL_SIZE_MM)
        else:
            upper_crop_distance_voxels = can_height_voxels + axial_padding_voxels + 1

        return cls(
            voxel_size_mm=VOXEL_SIZE_MM,
            can_diameter_mm=CAN_DIAMETER_MM,
            can_height_mm=CAN_HEIGHT_MM,
            downsampling_factor=DOWNSAMPLING_FACTOR,
            sigma_alignment=SIGMA_ALIGNMENT,
            max_rotation_angle_degrees=MAX_ROTATION_ANGLE_DEGREES,
            sigma_radial_cropping=SIGMA_RADIAL_CROPPING,
            relative_detected_diameter_tolerance=RELATIVE_DETECTED_DIAMETER_TOLERANCE,
            radius_plus_padding_voxels=can_radius_voxels + radial_padding_voxels,
            axial_padding_voxels=axial_padding_voxels,
            upper_crop_distance_voxels=upper_crop_distance_voxels,
            peak_offset_voxels=round(PEAK_OFFSET_MM / VOXEL_SIZE_MM),
            edge_detection_sigma_voxels=EDGE_DETECTION_SIGMA_VOXELS,
        )


# --------------------------------------------------------------------------------------------------
# I/O (adapted from scan_processing/io/load_volume.py)
# --------------------------------------------------------------------------------------------------


def _infer_dtype(file_size_bytes: int, n_voxels: int) -> DTypeLike:
    """Infer the numpy dtype of a raw volume from its file size and voxel count."""
    bytes_per_voxel, remainder = divmod(file_size_bytes, n_voxels)
    if remainder != 0:
        msg = (
            f"File size ({file_size_bytes} bytes) is not cleanly divisible by the number of voxels "
            f"({n_voxels} voxels). Remainder: {remainder}. Check the VOXELS_X/Y/Z settings."
        )
        raise ValidationError(msg)

    dtype_by_bytes_per_voxel: dict[int, DTypeLike] = {
        1: np.uint8,
        2: np.uint16,
        4: np.float32,
    }
    dtype = dtype_by_bytes_per_voxel.get(bytes_per_voxel)
    if dtype is None:
        msg = (
            f"{bytes_per_voxel=} is not supported. We expect 1, 2, or 4 bytes per voxel for uint8, "
            f"uint16, or float32 volumes, respectively. {file_size_bytes=}, {n_voxels=}."
        )
        raise ValidationError(msg)
    return dtype


def _downcast_to_uint8(data: NDArray, chunks: int = 5) -> NDArray[np.uint8]:
    """Linearly rescale an array to uint8 using its global min and max, one slab at a time."""
    full_array_min = float(data.min())
    full_array_max = float(data.max())
    if full_array_max == full_array_min:
        msg = f"Volume is constant ({full_array_min}); nothing to process."
        raise ValidationError(msg)

    downcast_data = np.zeros(shape=data.shape, dtype=np.uint8)
    chunk_size = max(1, data.shape[0] // chunks)
    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start : start + chunk_size].astype(np.float32)
        chunk -= full_array_min
        chunk /= full_array_max - full_array_min
        chunk *= 255
        downcast_data[start : start + chunk_size] = chunk.astype(np.uint8)
    return downcast_data


def load_raw_volume_as_uint8(
    file_path: Path, voxels_x: int, voxels_y: int, voxels_z: int
) -> NDArray[np.uint8]:
    """Load a .raw volume from disk as uint8 with shape (x, y, z).

    Raw volumes are stored slice-major (z, y, x), so the array is transposed after loading; this
    matches `scan_processing.io.load_volume._load_raw_or_vol`.
    """
    volume_dims_zyx = (voxels_z, voxels_y, voxels_x)
    n_voxels = voxels_x * voxels_y * voxels_z
    dtype = _infer_dtype(file_path.stat().st_size, n_voxels)

    start_time = time()
    data = np.fromfile(file_path, dtype=dtype).reshape(volume_dims_zyx)
    logger.info(
        f"Loaded {dtype} volume {volume_dims_zyx} (z, y, x) in {time() - start_time:.2f}s"
    )

    if dtype != np.uint8:
        start_time = time()
        data = _downcast_to_uint8(data)
        logger.info(f"Downcast {dtype} to uint8 in {time() - start_time:.2f}s")

    # np.transpose returns a view; copy so downstream in-place rotations operate on contiguous memory
    return np.ascontiguousarray(np.transpose(data))


def save_raw_volume(file_path: Path, volume: NDArray[np.uint8]) -> None:
    """Write an (x, y, z) volume to a headerless .raw file in (z, y, x) order, matching the input."""
    start_time = time()
    np.ascontiguousarray(np.transpose(volume)).tofile(file_path)
    logger.info(
        f"Wrote {file_path} ({volume.nbytes / 1024**3:.2f} GiB) in {time() - start_time:.2f}s"
    )


# --------------------------------------------------------------------------------------------------
# Rotation (adapted from scan_processing/arraycompat/rotation.py, CPU path only)
# --------------------------------------------------------------------------------------------------


def _determine_chunk_size(
    shape: tuple[int, ...], itemsize: int, chunk_dim: int, budget_bytes: int
) -> int:
    """Pick the largest chunk size along chunk_dim that fits the memory budget.

    The factor of 3 mirrors the package, which sizes chunks for the copy plus scipy's working buffers.
    """
    bytes_per_index = itemsize * 3
    for dim, size in enumerate(shape):
        if dim != chunk_dim:
            bytes_per_index *= size
    return max(1, min(shape[chunk_dim], budget_bytes // bytes_per_index))


def rotate_volume_inplace(
    volume: NDArray[np.uint8],
    angle_degrees: float,
    axes: tuple[int, int],
    order: int = 1,
) -> None:
    """Rotate a volume in place within the plane given by `axes`, chunked along the third axis."""
    chunk_dim = ({0, 1, 2} - set(axes)).pop()
    chunk_size = _determine_chunk_size(
        volume.shape, volume.itemsize, chunk_dim, ROTATION_CHUNK_BYTES
    )
    num_chunks = (volume.shape[chunk_dim] + chunk_size - 1) // chunk_size
    logger.debug(
        f"Rotating {angle_degrees:.3f}° in plane {axes} in {num_chunks} chunk(s) of {chunk_size}"
    )

    for n in range(num_chunks):
        slice_obj: list[slice] = [slice(None)] * volume.ndim
        slice_obj[chunk_dim] = slice(
            n * chunk_size, min((n + 1) * chunk_size, volume.shape[chunk_dim])
        )
        index = tuple(slice_obj)
        chunk = volume[index].copy()
        volume[index] = scipy.ndimage.rotate(
            chunk, angle_degrees, axes=axes, reshape=False, order=order
        )


# --------------------------------------------------------------------------------------------------
# Step 1: alignment
# (adapted from postprocessing/cylindrical.py:align, alignment.py, segmentation.py)
# --------------------------------------------------------------------------------------------------


def segment_cell_within_volume(
    volume: NDArray[np.uint8], downsampling_factor: int, sigma: float
) -> NDArray[np.bool_]:
    """Segment the cell within the scan volume with an Otsu threshold on a downsampled copy."""
    # Downsample volume to reduce computation time.
    # As long as the downsampling factor is consistent, it won't affect the alignment.
    volume = volume[::downsampling_factor, ::downsampling_factor, ::downsampling_factor]

    # Apply Gaussian filter to volume to blur the cell
    if sigma != 0:
        volume = skimage.util.img_as_ubyte(
            skimage.filters.gaussian(volume, sigma=sigma)
        )

    # Compute the Otsu threshold of a slice in the middle of the blurred volume
    middle_slice = volume[:, :, volume.shape[2] // 2]
    threshold = skimage.filters.threshold_otsu(middle_slice)
    logger.debug(f"Alignment grayscale threshold: {threshold}")

    return volume > threshold


def find_rotation_angles_cylindrical(
    segmented_volume: NDArray[np.bool_], is_z_axis_largest_eigenvalue: bool
) -> tuple[float, float]:
    """Find the x-axis and y-axis rotation angles that align the cylinder axis with z.

    Note that this may fail for large angles (>45°); most scanned cylinders are not that misaligned.
    """
    # Compute the center of mass of the cylinder
    com = np.array(scipy.ndimage.center_of_mass(segmented_volume))

    # Compute the covariance matrix of the voxel coordinates relative to the center of mass
    coords = np.argwhere(segmented_volume)
    coords_centered = coords - com
    cov_matrix = np.cov(coords_centered.T)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Find the index of the appropriate eigenvalue. Below, H = cell height and D = cell diameter:
    # - If H >= sqrt(3)/2*D (e.g., cylindrical cell), take the eigenvector with the largest eigenvalue
    # - If H < sqrt(3)/2*D (e.g., coin cell), take the eigenvector with the smallest eigenvalue
    eigenvalue_index = -1 if is_z_axis_largest_eigenvalue else 0
    idx = eigenvalues.argsort()[eigenvalue_index]

    # This eigenvector represents the axis of symmetry of the cylinder; normalize it
    v = eigenvectors[:, idx]
    v = v / np.linalg.norm(v)

    # Flip if necessary to ensure that the vector points in the positive z direction
    if v[2] < 0:
        v = -v

    # Compute the angle of the first rotation around the x-axis, then around the y-axis
    theta = float(np.rad2deg(np.arctan2(v[1], v[2])))
    phi = float(np.rad2deg(np.arctan2(v[0], np.sqrt(v[1] ** 2 + v[2] ** 2))))
    return theta, phi


def calculate_min_rotation_angle_degrees(volume_height_voxels: int) -> float:
    """Return the rotation angle that would shift the volume by a single voxel.

    Rotations smaller than this are skipped. For typical 3000-4000 voxel-tall cylindrical volumes
    this is around 0.01°, so in practice both rotations run.
    """
    return float(np.degrees(np.arctan(1 / volume_height_voxels)))


def convert_angle_to_first_quadrant_equivalent(angle: float) -> float:
    """Convert an angle to its first-quadrant equivalent.

    The eigenvector analysis above can return angles in the second, third, or fourth quadrants.
    """
    angle_float = float(angle) % 180
    return min(angle_float, 180 - angle_float)


def align(volume: NDArray[np.uint8], params: Params) -> NDArray[np.uint8]:
    """Align the scan so that the principal axis of the cylinder is parallel to the z axis."""
    # Segment the cylindrical cell within the volume
    volume_segmented_cell = segment_cell_within_volume(
        volume, params.downsampling_factor, params.sigma_alignment
    )

    # Determine if the z axis corresponds to the largest eigenvalue.
    # The factor of sqrt(3)/2 determines the crossover point where the z axis is the largest eigenvalue:
    # if the can height exceeds sqrt(3)/2 times the diameter, the z axis is the largest eigenvalue.
    # Most cylindrical cells have the z axis as the largest eigenvalue, most coin cells do not.
    is_z_axis_largest_eigenvalue = (
        params.can_height_mm >= (math.sqrt(3) / 2) * params.can_diameter_mm
    )

    # Find rotation angles for the cylindrical cell
    theta, phi = find_rotation_angles_cylindrical(
        volume_segmented_cell, is_z_axis_largest_eigenvalue
    )
    logger.info(f"Rotation angles: theta={theta:.3f}°, phi={phi:.3f}°")

    # Confirm the measured rotation angles are not too large (the cell shouldn't need much rotation)
    theta_validation_degrees = convert_angle_to_first_quadrant_equivalent(theta)
    phi_validation_degrees = convert_angle_to_first_quadrant_equivalent(phi)
    if (
        theta_validation_degrees > params.max_rotation_angle_degrees
        or phi_validation_degrees > params.max_rotation_angle_degrees
    ):
        msg = (
            "Validation error: At least one rotation angle exceeds the threshold during alignment "
            f"(theta={theta_validation_degrees:.3g}°, phi={phi_validation_degrees:.3f}°)."
        )
        raise ValidationError(msg)

    # Calculate the min rotation angle threshold
    min_rotation_angle_degrees = calculate_min_rotation_angle_degrees(volume.shape[2])

    # Rotate the volume (first x axis, then y axis)
    if abs(theta) > min_rotation_angle_degrees:
        rotate_volume_inplace(volume, theta, axes=(1, 2))
    else:
        logger.debug(
            f"Skipping theta rotation ({theta=:.3f}°, {min_rotation_angle_degrees=:.3f}°)"
        )

    if abs(phi) > min_rotation_angle_degrees:
        rotate_volume_inplace(volume, phi, axes=(0, 2))
    else:
        logger.debug(
            f"Skipping phi rotation ({phi=:.3f}°, {min_rotation_angle_degrees=:.3f}°)"
        )

    return volume


# --------------------------------------------------------------------------------------------------
# Step 2: radial cropping
# (adapted from postprocessing/cylindrical.py:crop_radial and postprocessing/cropping_radial.py)
# --------------------------------------------------------------------------------------------------


def find_edge_labels(labeled_edges: NDArray[np.int_], axis: int) -> tuple[int, int]:
    """Return the first and last non-background labels along a central linecut on `axis`."""
    centerline_idx = labeled_edges.shape[axis] // 2
    centerline = (
        labeled_edges[centerline_idx, :]
        if axis == 0
        else labeled_edges[:, centerline_idx]
    )

    # Find indices of non-background labels (background = 0)
    non_background_indices = np.where(centerline != 0)[0]
    if len(non_background_indices) == 0:
        msg = (
            "Validation error: No non-background labels were found along the central linecut of "
            f"{axis=} during radial cropping."
        )
        raise ValidationError(msg)

    return int(centerline[non_background_indices[0]]), int(
        centerline[non_background_indices[-1]]
    )


def get_most_common_edge_label(labeled_edges: NDArray[np.int_]) -> int:
    """Return the most common of the four labels closest to the image edges (two per axis)."""
    edge_labels = [
        *find_edge_labels(labeled_edges, axis=0),
        *find_edge_labels(labeled_edges, axis=1),
    ]
    unique_labels, counts = np.unique(edge_labels, return_counts=True)
    return int(unique_labels[np.argmax(counts)])


def fit_can_to_circle(
    volume: NDArray[np.uint8], sigma: float
) -> tuple[float, float, float]:
    """Fit a circle to the outer contour of the can and return (x_center, y_center, radius).

    Algorithm:
    1. Get a nominal middle slice
    2. Find the edges via a canny filter
    3. Label the edges
    4. Get the label corresponding to the can outer diameter
    5. Get the coordinates of the can outer diameter contour
    6. Fit the coordinates to a circle
    """
    # Get a nominal middle slice
    middle_slice = volume[:, :, volume.shape[2] // 2]

    # Find the edges via a canny filter
    edges = skimage.feature.canny(middle_slice, sigma=sigma)
    if not np.any(edges):
        msg = "Validation error: No edges were detected by the Canny filter during radial cropping."
        raise ValidationError(msg)

    # Label the edges, using a np.ones((3, 3)) structuring element to ensure connectivity
    labeled_edges, _ = scipy.ndimage.label(edges, structure=np.ones((3, 3)))

    # Get the coordinates of the can outer diameter contour
    can_outer_diameter_label = get_most_common_edge_label(labeled_edges)
    coordinates = np.argwhere(labeled_edges == can_outer_diameter_label)

    # Fit the coordinates to a circle
    model = CircleModel.from_estimate(coordinates)
    if not model:
        msg = (
            "Validation error: Could not fit a circle to the detected can contour during radial "
            f"cropping ({len(coordinates)} contour points). Consider adjusting SIGMA_RADIAL_CROPPING."
        )
        raise ValidationError(msg)

    x_center, y_center = model.center
    return float(x_center), float(y_center), float(model.radius)


def crop_radial(volume: NDArray[np.uint8], params: Params) -> NDArray[np.uint8]:
    """Crop the scan radially so that the volume is only the cylinder, plus some padding."""
    # Detect the contour of the can outer diameter
    x_center, y_center, detected_radius_voxels = fit_can_to_circle(
        volume, sigma=params.sigma_radial_cropping
    )

    # Calculate the detected diameter and round the center points
    detected_diameter_mm = 2 * detected_radius_voxels * params.voxel_size_mm
    x_center, y_center = round(x_center), round(y_center)
    logger.info(f"Detected diameter from crop_radial is {detected_diameter_mm:.2f}mm")
    logger.info(f"Detected center from crop_radial is {x_center=}, {y_center=}")

    # Check that the detected diameter is close to the nominal cell diameter
    relative_error = (
        abs(detected_diameter_mm - params.can_diameter_mm) / params.can_diameter_mm
    )
    if relative_error > params.relative_detected_diameter_tolerance:
        msg = (
            f"Validation error: The detected diameter ({detected_diameter_mm:.2f}mm) differs from the "
            f"nominal can diameter ({params.can_diameter_mm:.2f}mm) by {relative_error:.1%}, which "
            f"exceeds the tolerance of {params.relative_detected_diameter_tolerance:.1%}."
        )
        raise ValidationError(msg)

    # Define cropping limits in x and y
    radius_plus_padding = params.radius_plus_padding_voxels
    min_x_limit, max_x_limit = (
        x_center - radius_plus_padding,
        x_center + radius_plus_padding,
    )
    min_y_limit, max_y_limit = (
        y_center - radius_plus_padding,
        y_center + radius_plus_padding,
    )
    logger.info(
        f"Radial cropping limits: {min_x_limit=}, {max_x_limit=}, {min_y_limit=}, {max_y_limit=}"
    )

    # Confirm that we don't crop out of the bounds of the image
    # Note that x = axis0 and y = axis1 (standard convention, not image convention)
    checks = [
        (min_x_limit < 0, f"{min_x_limit=}, which is < 0."),
        (min_y_limit < 0, f"{min_y_limit=}, which is < 0."),
        (
            max_x_limit > volume.shape[0],
            f"{max_x_limit=}, which is > volume.shape[0]={volume.shape[0]}.",
        ),
        (
            max_y_limit > volume.shape[1],
            f"{max_y_limit=}, which is > volume.shape[1]={volume.shape[1]}.",
        ),
    ]
    for condition, error_details in checks:
        if condition:
            msg = (
                f"Validation error: {error_details} center=[{x_center}, {y_center}], "
                f"{radius_plus_padding=}. First confirm the detected center is roughly in the center "
                "of the image. Then consider decreasing RADIAL_PADDING_MM."
            )
            raise ValidationError(msg)

    return volume[min_x_limit:max_x_limit, min_y_limit:max_y_limit, :]


# --------------------------------------------------------------------------------------------------
# Step 3: radial masking (adapted from postprocessing/cylindrical.py:mask_radial)
# --------------------------------------------------------------------------------------------------


def mask_radial(volume: NDArray[np.uint8], params: Params) -> NDArray[np.uint8]:
    """Zero the voxels inside the image boundary but outside the cylindrical can plus padding."""
    # Get coordinates to mask
    center_coordinates_xy = tuple(round(dim / 2) for dim in volume.shape[:2])

    # Create the disk, clipped to the slice shape so this is safe without a preceding radial crop
    rr, cc = skimage.draw.disk(
        center_coordinates_xy, params.radius_plus_padding_voxels, shape=volume.shape[:2]
    )

    # Create the 2d mask
    mask = np.zeros(volume.shape[:2], dtype=np.uint8)
    mask[rr, cc] = 1

    # Mask the volume by reshaping the mask into (x, y, 1) so that it broadcasts across the z axis
    return volume * mask.reshape((*mask.shape, 1))


# --------------------------------------------------------------------------------------------------
# Step 4: axial cropping
# (adapted from postprocessing/cylindrical.py:crop_axial and cropping_principal_axis.py)
# --------------------------------------------------------------------------------------------------


def detect_strongest_edge(
    volume: NDArray[np.uint8], axis: int, sigma_voxels: float
) -> int:
    """Detect the strongest edge in the volume along `axis`.

    1D edge detection applied to a 3D volume: average the volume over the other two axes, keep the
    first half of the resulting linecut, smooth it, and take the argmax of its first derivative.
    This works because cells often (but not always!) have strong edges aligned with the principal axes.
    """
    # Compute the mean over the other two axes, leaving a single linecut along `axis`
    axes = tuple(ax for ax in (0, 1, 2) if ax != axis)
    linecut = np.mean(volume, axis=axes)

    # Only look at the first half of the linecut
    linecut = linecut[: len(linecut) // 2]

    # Compute the gradient of the smoothed linecut
    gradient = np.gradient(scipy.ndimage.gaussian_filter1d(linecut, sigma=sigma_voxels))

    # need to skip the first part of the array where there is a false edge introduced by alignment
    skip_first = 5
    edge_position_voxels = int(np.argmax(gradient[skip_first:])) + skip_first

    # TESTING -
    if False:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 12))
        plt.subplot(2, 1, 1)
        plt.plot(linecut)
        plt.title("Axial Line Cut")
        plt.subplot(2, 1, 2)
        plt.plot(gradient)
        plt.axvline(edge_position_voxels, color="g", linestyle="--")
        plt.title("Gradient of Axial Line Cut")
        plt.show()
    # END TESTING

    logger.info(f"Edge position for {axis=} = {edge_position_voxels}")
    return edge_position_voxels


def crop_along_principal_axis(
    volume: NDArray[np.uint8], params: Params, axis: int
) -> NDArray[np.uint8]:
    """Crop the scan in one dimension to remove extra space, keeping some padding."""
    # Detect the strongest edge in the volume along this axis
    first_edge_index = detect_strongest_edge(
        volume, axis, params.edge_detection_sigma_voxels
    )

    logger.info(
        f"    crop_along_principal_axis() first edge index for {axis=}: {first_edge_index}"
    )

    # Define the start of the object (object edge) as the strongest edge + the peak offset.
    # A positive peak offset moves the object lower in the image, a negative one moves it higher.
    starting_position_voxels = first_edge_index + params.peak_offset_voxels

    # Define the min and max indices for cropping. We only use the object edge for positioning, then
    # crop based on the nominal cell dimension so that slice indices match the cell volume.
    index_min = starting_position_voxels - params.axial_padding_voxels
    index_max = starting_position_voxels + params.upper_crop_distance_voxels
    logger.info(f"Cropping limits for {axis=}: {index_min=}, {index_max=}")

    # Check that index_min is positive
    if index_min < 0:
        msg = (
            f"Validation error: {index_min=} is negative for {axis=} ({first_edge_index=}, "
            f"peak_offset_voxels={params.peak_offset_voxels}, "
            f"padding_voxels={params.axial_padding_voxels}). "
            "Consider using a more negative PEAK_OFFSET_MM or a smaller AXIAL_PADDING_MM."
        )
        raise ValidationError(msg)

    # Check that index_max does not exceed the volume size along this axis
    volume_len_along_axis = volume.shape[axis]
    if index_max > volume_len_along_axis:
        msg = (
            f"Validation error: {index_max=} exceeds the volume size ({volume_len_along_axis}) for "
            f"{axis=} ({first_edge_index=}, peak_offset_voxels={params.peak_offset_voxels}, "
            f"upper_crop_distance_voxels={params.upper_crop_distance_voxels}). "
            "Consider using a more negative PEAK_OFFSET_MM."
        )
        raise ValidationError(msg)

    return np.take(volume, np.arange(index_min, index_max), axis=axis)


def crop_axial(volume: NDArray[np.uint8], params: Params) -> NDArray[np.uint8]:
    """Crop the scan axially so that the volume is only the cylinder, plus some padding."""
    return crop_along_principal_axis(volume, params, axis=2)


def _find_z_dimension(volume_path: Path) -> int:
    """
    Find the z dimension of the raw volume file.

    Assumptions:
      - raw volume is 1 byte per voxels (uint8)

    The assumption is weakly validated to ensure that the height of the volume is reasonable (<= VOXELS_Z_UPPERLIMIT).
    """
    size_vol = volume_path.stat().st_size

    # We assume type uint8 which will be validated via reasonable limits of the z dimension
    z_size = size_vol // (VOXELS_X * VOXELS_Y)

    if z_size > VOXELS_Z_UPPERLIMIT:
        raise ValidationError(
            f"Validation error: estimated z dimension {z_size} exceeds the upper limit ({VOXELS_Z_UPPERLIMIT})."
        )
    return z_size


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------


def main(
    filename: str = typer.Argument(
        ..., help="Name of the raw volume file to process, within VOLUME_DIR."
    ),
) -> None:
    """Load the raw volume, run the postprocessing steps, and write the result."""
    logging.basicConfig(
        level=logging.DEBUG if VERBOSE else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    if not VOLUME_DIR.exists():
        logger.error(f"Volume directory does not exist: {VOLUME_DIR}")
        raise typer.Exit(code=1)

    input_path = VOLUME_DIR / filename
    if not input_path.is_file():
        logger.error(f"Input file does not exist: {input_path}")
        raise typer.Exit(code=1)

    # Set the volume height from the file
    VOXELS_Z = _find_z_dimension(input_path)

    # The output sits next to the input, with OUTPUT_NAME_SUFFIX appended to the stem
    output_path = input_path.with_name(
        input_path.stem + OUTPUT_NAME_SUFFIX + input_path.suffix
    )
    logger.info(f"Output path: {output_path}")
    if output_path.resolve() == input_path.resolve():
        logger.error(
            "Refusing to overwrite the input volume; set a non-empty OUTPUT_NAME_SUFFIX."
        )
        raise typer.Exit(code=1)
    if output_path.exists() and not OVERWRITE_OUTPUT:
        logger.error(
            f"Output file already exists: {output_path}. Set OVERWRITE_OUTPUT = True to replace it."
        )
        raise typer.Exit(code=1)

    params = Params.from_config()
    logger.debug(f"Parameters: {params}")

    try:
        volume = load_raw_volume_as_uint8(input_path, VOXELS_X, VOXELS_Y, VOXELS_Z)
        logger.info(f"Volume shape (x, y, z): {volume.shape}")

        steps = (
            ("align", RUN_ALIGN, align),
            ("crop_radial", RUN_CROP_RADIAL, crop_radial),
            ("mask_radial", RUN_MASK_RADIAL, mask_radial),
            ("crop_axial", RUN_CROP_AXIAL, crop_axial),
        )
        for name, enabled, step in steps:
            if not enabled:
                logger.info(f"Skipping {name}")
                continue
            start_time = time()
            volume = step(volume, params)
            logger.info(
                f"{name} finished in {time() - start_time:.2f}s; shape (x, y, z): {volume.shape}"
            )

        save_raw_volume(output_path, volume)
    except ValidationError as error:
        logger.error(str(error))
        raise typer.Exit(code=1) from error

    x, y, z = volume.shape
    logger.info(
        f"Done. Output is uint8 with dims x={x}, y={y}, z={z}, written as (z, y, x)."
    )


if __name__ == "__main__":
    typer.run(main)
