#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy~=2.4.1",
#     "scipy~=1.17.1",
#     "scikit-image~=0.26.0",
# ]
# ///
"""Standalone postprocessing of a cylindrical-cell CT volume: .raw in, .raw out.

Edit the CONFIGURATION block below, then run the script with one or more filenames (within
VOLUME_DIR): `./process_raw_volume.py scan1.raw scan2.raw` (or `uv run process_raw_volume.py ...`).

The algorithms here are adapted directly from the `scan_processing` package
(`postprocessing/{cylindrical,exposure,alignment,segmentation,cropping_radial,cropping_principal_axis}.py`
and `arraycompat/rotation.py`), stripped of GPU support, S3/API I/O, ScanInfo, and loguru so that
the script runs anywhere with CPU numpy/scipy/scikit-image.

Volumes are loaded at their native bit depth: uint8 volumes stay uint8; uint16 volumes stay uint16;
float32 volumes are downcast to uint16. adjust_contrast and align both preserve that dtype, then the
volume is downcast to uint8 (a no-op if it's already uint8) before the remaining, uint8-only steps.

Steps, in order (each can be switched off in the CONFIGURATION block, except downcast_to_uint8, which
always runs since the remaining steps assume uint8):
    1. adjust_contrast   — contrast-stretch the intensity histogram into [1, dtype max]
    2. align             — rotate so the cylinder axis is parallel to z
    3. downcast_to_uint8 — downcast to uint8 if the volume is still uint16
    4. crop_radial       — crop x/y to the can plus padding, centred on the fitted can circle
    5. mask_radial       — zero voxels inside the image but outside the can plus padding
    6. crop_axial        — crop z to the can plus padding, using 1D edge detection along z
    7. flip_y            — reverse the volume's order along the y-axis

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
import csv
import json
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
# VOLUME_DIR = Path("/Volumes/SSK SSD")

# The output goes next to the input, with this appended to the file name:
# e.g. scan.raw -> scan_processed.raw
OUTPUT_NAME_SUFFIX = "_processed"

# Overwrite the output file if it already exists
OVERWRITE_OUTPUT = False

# Filenames whose stem ends with one of these are treated as already-processed volumes and are
# skipped rather than reprocessed. This lets a glob like `Data/volumes/*.raw` be passed straight
# to the script without it trying to run outputs (its own or from earlier pipeline steps) back
# through processing. OUTPUT_NAME_SUFFIX is included automatically.
SKIP_NAME_SUFFIXES = ("_processed",)

# --- Input volume dimensions ----------------------------------------------------------------------
# Required: a .raw file has no header, so the voxel counts cannot be inferred. The dtype (uint8,
# uint16, or float32) is inferred from the file size. uint8 and uint16 volumes are loaded as-is;
# float32 volumes are downcast to uint16. The volume is downcast to uint8 later, after adjust_contrast
# and align have run at the loaded dtype (see downcast_to_uint8 below).
VOXELS_X = 1425
VOXELS_Y = 1425
VOXELS_Z = None
# The z dimension (and with it, the dtype) is inferred from the file size assuming 1 byte/voxel
# (uint8): that naive z scales linearly with the true bytes-per-voxel, so its magnitude tells us
# which dtype it actually is. Naive z <= VOXELS_Z_UPPERLIMIT_UINT8 means uint8 (naive z is the true
# z); naive z <= VOXELS_Z_UPPERLIMIT_UINT16 means uint16 (true z is naive z / 2); otherwise float32
# (true z is naive z / 4). See _find_z_dimension.
VOXELS_Z_UPPERLIMIT_UINT8 = 6000  # Upper limit for the z dimension of a uint8 volume
VOXELS_Z_UPPERLIMIT_UINT16 = 2 * VOXELS_Z_UPPERLIMIT_UINT8  # ...and of a uint16 volume

# --- Cell geometry --------------------------------------------------------------------------------
VOXEL_SIZE_MM = 0.0164
CAN_DIAMETER_MM = 21.0  # Nominal can diameter
CAN_HEIGHT_MM = 70.0  # Nominal can height

# --- Steps to run ---------------------------------------------------------------------------------
RUN_ADJUST_CONTRAST = True
RUN_ALIGN = True
RUN_CROP_RADIAL = True
RUN_MASK_RADIAL = True
RUN_CROP_AXIAL = True
RUN_FLIP_Y = True

# --- Contrast adjustment --------------------------------------------------------------------------
# Percentile of a middle slice used as the maximum input intensity. Cylindrical cells use a relatively
# low percentile because much of the can can be set to the max value without losing detail.
MAX_INPUT_PERCENTILE = 98.5

# --- Alignment ------------------------------------------------------------------------------------
DOWNSAMPLING_FACTOR = 10  # Downsampling factor for the alignment segmentation
SIGMA_ALIGNMENT = 3.0  # Blurring extent in the Gaussian filter, in voxels
MAX_ROTATION_ANGLE_DEGREES = 30.0  # Applies to both theta and phi
ORDER_ALIGNMENT_ROTATION = 3
ALIGN_COMBINED = (
    False  # Perform the rotational alignment with a single affine transformation
)

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

# One row is appended for every successfully processed input volume.
PROCESS_LOG_FILE = Path("process_raw_volume_log.csv")

# ==================================================================================================
# END OF CONFIGURATION
# ==================================================================================================

logger = logging.getLogger("process_raw_volume")

# adjust_contrast and align run at whichever of these two dtypes the volume was loaded as; every
# later step runs on uint8 only, once downcast_to_uint8 has converted it.
Volume = NDArray[np.uint8] | NDArray[np.uint16]


class ValidationError(RuntimeError):
    """Raised when a processing step detects a volume it cannot handle."""


@dataclass(frozen=True)
class Params:
    """Processing parameters, in voxels where the algorithms work in voxels.

    Mirrors the subset of `ScanInfoCylindrical` used by the processing steps below.
    """

    voxel_size_mm: float

    # Cell dimensions
    can_diameter_mm: float
    can_height_mm: float

    # Contrast adjustment
    max_input_percentile: float

    # Alignment
    downsampling_factor: int
    sigma_alignment: float
    max_rotation_angle_degrees: float
    rotation_interpolation_order: int

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
            max_input_percentile=MAX_INPUT_PERCENTILE,
            downsampling_factor=DOWNSAMPLING_FACTOR,
            sigma_alignment=SIGMA_ALIGNMENT,
            max_rotation_angle_degrees=MAX_ROTATION_ANGLE_DEGREES,
            rotation_interpolation_order=ORDER_ALIGNMENT_ROTATION,
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


def _rescale_to_dtype(data: NDArray, dtype: DTypeLike, chunks: int = 5) -> NDArray:
    """Linearly rescale an array to `dtype` using its global min and max, one slab at a time."""
    full_array_min = float(data.min())
    full_array_max = float(data.max())
    if full_array_max == full_array_min:
        msg = f"Volume is constant ({full_array_min}); nothing to process."
        raise ValidationError(msg)

    dtype_max = np.iinfo(dtype).max
    rescaled_data = np.zeros(shape=data.shape, dtype=dtype)
    chunk_size = max(1, data.shape[0] // chunks)
    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start : start + chunk_size].astype(np.float32)
        chunk -= full_array_min
        chunk /= full_array_max - full_array_min
        chunk *= dtype_max
        rescaled_data[start : start + chunk_size] = chunk.astype(dtype)
    return rescaled_data


def load_raw_volume(
    file_path: Path, voxels_x: int, voxels_y: int, voxels_z: int
) -> Volume:
    """Load a .raw volume from disk with shape (x, y, z), at its native bit depth.

    uint8 and uint16 volumes are loaded as-is; float32 volumes are downcast to uint16 (the volume is
    downcast further to uint8 later, by `downcast_to_uint8`, after adjust_contrast and align have run).
    Raw volumes are stored slice-major (z, y, x), so the array is transposed after loading; this
    matches `scan_processing.io.load_volume._load_raw_or_vol`.

    The file is memory-mapped rather than read in one `np.fromfile` call: for a full-FOV volume near
    VOXELS_Z_UPPERLIMIT_UINT8/UINT16, a float32 source can be tens of GiB, and `_rescale_to_dtype`
    below needs to hold a same-shape uint16 output alongside it -- reading the whole thing into RAM
    first makes that peak (input + output, fully resident) exceed physical memory on real machines.
    Memory-mapping keeps the raw bytes paged in from disk on demand instead.
    """
    volume_dims_zyx = (voxels_z, voxels_y, voxels_x)
    n_voxels = voxels_x * voxels_y * voxels_z
    dtype = _infer_dtype(file_path.stat().st_size, n_voxels)

    start_time = time()
    data = np.memmap(file_path, dtype=dtype, mode="r", shape=volume_dims_zyx)
    logger.info(
        f"Memory-mapped {dtype} volume {volume_dims_zyx} (z, y, x) in {time() - start_time:.2f}s"
    )

    if dtype == np.float32:
        start_time = time()
        data = _rescale_to_dtype(data, np.uint16)
        logger.info(f"Downcast {dtype} to uint16 in {time() - start_time:.2f}s")

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
    volume: Volume,
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
# Step 1: contrast adjustment
# (adapted from postprocessing/cylindrical.py:adjust_image_contrast and postprocessing/exposure.py)
# --------------------------------------------------------------------------------------------------


def validate_rescale_intensity_inputs(
    array: Volume, in_range: tuple[int, int], out_range: tuple[int, int]
) -> None:
    """Validate the input and output intensity ranges for `rescale_intensity`."""
    min_input, max_input = in_range
    min_output, max_output = out_range

    error_msg: str | None = None
    if array.dtype not in (np.uint8, np.uint16):
        error_msg = f"Expected a uint8 or uint16 array, got {array.dtype}."
    elif min_input >= max_input:
        error_msg = f"Input range is empty or inverted: min ({min_input}) must be less than max ({max_input})."
    elif min_output >= max_output:
        error_msg = f"Output range is empty or inverted: min ({min_output}) must be less than max ({max_output})."
    else:
        dtype_max = np.iinfo(array.dtype).max
        if min_input < 0 or max_input > dtype_max:
            error_msg = f"Input range [{min_input}, {max_input}] is outside the valid {array.dtype} range [0, {dtype_max}]."
        elif min_output < 0 or max_output > dtype_max:
            error_msg = f"Output range [{min_output}, {max_output}] is outside the valid {array.dtype} range [0, {dtype_max}]."

    if error_msg is not None:
        raise ValidationError(f"Validation error: {error_msg}")


def rescale_intensity(
    array: Volume, in_range: tuple[int, int], out_range: tuple[int, int]
) -> Volume:
    """Rescale intensity using contrast stretching.

    A replacement for `skimage.exposure.rescale_intensity` that is roughly 30x faster and avoids
    skimage's memory blow-up (https://github.com/scikit-image/scikit-image/issues/7199), by mapping
    the array's values through a lookup table sized to the full range of its dtype (uint8 or uint16).
    """
    min_input, max_input = in_range
    min_output, max_output = out_range
    dtype_max = np.iinfo(array.dtype).max

    # Pre-compute the scaling factor
    shifted_max = max_input - min_input
    scaling_factor = (max_output - min_output) / (max_input - min_input)

    # Create the rescaling array (a lookup table)
    rescale = np.arange(dtype_max + 1, dtype=np.float64)
    rescale = rescale - np.fmin(rescale, min_input)
    rescale = np.fmin(rescale, shifted_max)
    rescale = rescale * scaling_factor
    rescale += min_output
    rescale = np.round(rescale)
    rescale = rescale.astype(array.dtype)

    # Use the lookup table to convert the array
    return rescale[array]


def adjust_contrast(volume: Volume, params: Params, axis_for_in_max: int = 2) -> Volume:
    """Rescale the intensity histogram of the scan to improve contrast.

    By default the contrast of a CT scan is not optimized: the available grayscale values do not cover
    the full grayscale range, and the tails of the distribution limit the contrast in the "bulk".
    Both are addressed by contrast stretching, which does not distort the physical meaning of the
    grayscale values (see https://github.com/glimpse-engineering/analysis/issues/28).

    The clipping points (see https://github.com/glimpse-engineering/analysis/issues/115):
    - Low end: the constant-valued background of the reconstruction, taken as the mode of a middle
      slice. IMPORTANT: this assumption may vary by scanner vendor and reconstruction method.
    - High end: a high percentile of a middle slice, relatively low for cylindrical cells because much
      of the can can be set to the max value without losing fine detail.

    We scale over [1, dtype max] rather than [0, dtype max], reserving 0 for the background so that
    it stays transparent and is easy to ignore in downstream analysis. `dtype max` is 255 or 65535
    depending on whether the volume is uint8 or uint16 (see the module docstring).

    `axis_for_in_max` defaults to 2, the value the package uses for cylindrical cells.
    """
    # Grab the middle slices that we'll need
    middle_slice_in_max_axis = np.take(
        volume, volume.shape[axis_for_in_max] // 2, axis=axis_for_in_max
    )
    middle_slice_axis_2 = volume[:, :, volume.shape[2] // 2]

    # The minimum input intensity is the background: the mode of a middle slice on axis=2
    in_min = np.bincount(middle_slice_axis_2.flatten()).argmax()

    # The maximum input intensity is a percentile of a middle slice on axis_for_in_max
    in_max = np.percentile(
        middle_slice_in_max_axis.flatten(), params.max_input_percentile
    )

    # The out_range is (1, dtype max); see the docstring for context
    in_range = (int(in_min), int(in_max))
    out_range = (1, int(np.iinfo(volume.dtype).max))
    logger.info(f"Contrast adjustment: {in_range=} -> {out_range=}")

    validate_rescale_intensity_inputs(volume, in_range, out_range)

    return rescale_intensity(volume, in_range, out_range)


# --------------------------------------------------------------------------------------------------
# Step 2: alignment
# (adapted from postprocessing/cylindrical.py:align, alignment.py, segmentation.py)
# --------------------------------------------------------------------------------------------------


def segment_cell_within_volume(
    volume: Volume, downsampling_factor: int, sigma: float
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


def align(volume: Volume, params: Params, rotation_log: dict[str, float] | None = None) -> Volume:
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
    if rotation_log is not None:
        rotation_log.update(theta_degrees=float(theta), phi_degrees=float(phi))
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
        rotation_start = time()
        rotate_volume_inplace(
            volume, theta, axes=(1, 2), order=params.rotation_interpolation_order
        )
        theta_rotation_seconds = time() - rotation_start
    else:
        theta_rotation_seconds = 0.0
        logger.debug(
            f"Skipping theta rotation ({theta=:.3f}°, {min_rotation_angle_degrees=:.3f}°)"
        )
    if rotation_log is not None:
        rotation_log["theta_rotation_seconds"] = theta_rotation_seconds

    if abs(phi) > min_rotation_angle_degrees:
        rotation_start = time()
        rotate_volume_inplace(
            volume, phi, axes=(0, 2), order=params.rotation_interpolation_order
        )
        phi_rotation_seconds = time() - rotation_start
    else:
        phi_rotation_seconds = 0.0
        logger.debug(
            f"Skipping phi rotation ({phi=:.3f}°, {min_rotation_angle_degrees=:.3f}°)"
        )
    if rotation_log is not None:
        rotation_log["phi_rotation_seconds"] = phi_rotation_seconds
        logger.info(
            f"Theta rotation: {theta_rotation_seconds:.2f}s; "
            f"phi rotation: {phi_rotation_seconds:.2f}s"
        )

    return volume


# --------------------------------------------------------------------------------------------------
# Step 3: downcast to uint8
#
# adjust_contrast and align (above) run at whichever dtype the volume was loaded as -- uint8 as-is,
# uint16 as-is or downcast from float32 (see Volume and load_raw_volume) -- so that both operate at
# full precision when the source data is 16-bit. Every step from here on assumes uint8, so the volume
# is downcast now if it isn't already.
# --------------------------------------------------------------------------------------------------


def downcast_to_uint8(volume: Volume, params: Params) -> NDArray[np.uint8]:
    """Downcast a uint16 volume to uint8; a uint8 volume passes through unchanged."""
    if volume.dtype == np.uint8:
        return volume

    start_time = time()
    downcast = _rescale_to_dtype(volume, np.uint8)
    logger.info(f"Downcast {volume.dtype} to uint8 in {time() - start_time:.2f}s")
    return downcast


# --------------------------------------------------------------------------------------------------
# Step 4: radial cropping
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
# Step 5: radial masking (adapted from postprocessing/cylindrical.py:mask_radial)
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
# Step 6: axial cropping
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

    # if not performing the grayscale adjustment, we need to skip the gradient from the
    # rotation fill value.
    skip_first = 0
    if not RUN_ADJUST_CONTRAST:
        skip_first = 5
    edge_position_voxels = int(np.argmax(gradient[skip_first:])) + skip_first
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
    Find the z dimension of the raw volume file, inferring the dtype along the way.

    The dtype isn't known yet at this point (that's `_infer_dtype`'s job, downstream, once z is
    known), so this starts from the z you'd get by naively assuming 1 byte/voxel (uint8): that naive
    z scales linearly with the true bytes-per-voxel (2x for uint16, 4x for float32), so its magnitude
    tells us which one it actually is -- see VOXELS_Z_UPPERLIMIT_UINT8/UINT16 above. `_infer_dtype`
    then independently confirms the same dtype from the file size, using the true z returned here.
    """
    size_vol = volume_path.stat().st_size
    naive_z = size_vol // (VOXELS_X * VOXELS_Y)

    if naive_z <= VOXELS_Z_UPPERLIMIT_UINT8:
        z_size = naive_z  # uint8: naive z is already the true z
    elif naive_z <= VOXELS_Z_UPPERLIMIT_UINT16:
        z_size = naive_z // 2  # uint16: naive z is 2x the true z
    else:
        z_size = naive_z // 4  # float32: naive z is 4x the true z

    return z_size


# --------------------------------------------------------------------------------------------------
# Step 7: flip y-axis
# --------------------------------------------------------------------------------------------------


def flip_y(volume: NDArray[np.uint8], params: Params) -> NDArray[np.uint8]:
    """Reverse the volume's order along the y-axis.

    The on-disk volume is (z, y, x); internally it is loaded transposed to (x, y, z), so y is axis 1.
    """
    return np.ascontiguousarray(np.flip(volume, axis=1))


# --------------------------------------------------------------------------------------------------
# Alternate Step 2: single-pass combined-rotation alignment
#
# `align()` above applies theta and phi as two independent, cleanly-chunked passes: each rotation
# plane leaves the third axis completely untouched, so slicing along it gives exact, zero-overlap
# chunks. The functions below combine both rotations into a single 3x3 affine and apply it in one pass
# instead of two: half the data movement, and one interpolation instead of two cascaded ones (each
# `order >= 1` resample blurs a bit; cascading two compounds it, most visibly right at sharp edges).
#
# The cost: a general 3-axis rotation has no axis left invariant, so chunking it needs a small "halo"
# of extra input per chunk. That halo is computed exactly below (by back-projecting each output
# chunk's bounding box through the affine map), not approximated, and validated to match an unchunked
# call to floating-point precision (~1e-12) and to agree with two sequential `scipy.ndimage.rotate`
# calls to within normal single- vs double-interpolation differences (sub-voxel on point sources;
# a handful of grey levels right at sharp edges, nowhere else).
#
# Not wired into `process_one`'s `steps` tuple. `align_combined` is a drop-in replacement for `align`
# (same signature, same angle detection) if it turns out to be worth using instead.
# --------------------------------------------------------------------------------------------------


def _embed_plane_rotation(
    angle_degrees: float, axis_a: int, axis_b: int
) -> NDArray[np.float64]:
    """3x3 backward-mapping matrix for a rotation in the (axis_a, axis_b) plane, axis_a < axis_b.

    Matches the `rot_matrix = [[c, s], [-s, c]]` that `scipy.ndimage.rotate` builds internally for its
    own per-plane rotation (see its source), embedded into a 3x3 identity so the third axis is
    untouched. `affine_transform` uses this matrix as a backward map: `output[o] = input[matrix @ o +
    offset]`.
    """
    assert axis_a < axis_b, f"expected axis_a < axis_b, got {axis_a=}, {axis_b=}"
    c, s = math.cos(math.radians(angle_degrees)), math.sin(math.radians(angle_degrees))
    matrix = np.eye(3)
    matrix[axis_a, axis_a] = c
    matrix[axis_a, axis_b] = s
    matrix[axis_b, axis_a] = -s
    matrix[axis_b, axis_b] = c
    return matrix


def combined_rotation_matrix_and_offset(
    theta_degrees: float, phi_degrees: float, volume_shape: tuple[int, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Single (matrix, offset) pair equivalent to align()'s theta-then-phi two-pass rotation.

    theta is the y-z plane rotation (`axes=(1, 2)`, as in `rotate_volume_inplace`), phi the x-z plane
    rotation (`axes=(0, 2)`), applied theta-first and both about the volume's center -- exactly what
    `align()` does via two `rotate_volume_inplace` calls.

    Returns the (matrix, offset) pair for `scipy.ndimage.affine_transform`'s backward mapping:
    `output[o] = input[matrix @ o + offset]`.
    """
    b_theta = _embed_plane_rotation(theta_degrees, 1, 2)
    b_phi = _embed_plane_rotation(phi_degrees, 0, 2)
    # theta's backward map composed after phi's: applying phi's map to an output coordinate first,
    # then theta's, undoes "theta forward, then phi forward" -- the order align() applies them in.
    matrix = b_theta @ b_phi

    center = (np.array(volume_shape, dtype=np.float64) - 1) / 2.0
    offset = center - matrix @ center
    return matrix, offset


def _input_range_for_output_chunk(
    matrix: NDArray[np.float64],
    offset: NDArray[np.float64],
    output_start: int,
    output_stop: int,
    chunk_axis: int,
    volume_shape: tuple[int, ...],
    pad: int = 2,
) -> tuple[int, int]:
    """Exact input index range along `chunk_axis` needed to fill output[chunk_axis] in [start, stop).

    Back-projects the 8 corners of the output chunk's bounding box (full extent on the other two axes)
    through `matrix`/`offset` -- the same backward mapping `affine_transform` itself uses -- then pads
    by `pad` voxels for interpolation support and clips to the volume bounds.
    """
    other_axes = [ax for ax in range(3) if ax != chunk_axis]
    chunk_axis_extremes = (output_start, output_stop - 1)
    other_extremes = [(0, volume_shape[ax] - 1) for ax in other_axes]

    corners = []
    for chunk_val in chunk_axis_extremes:
        for a_val in other_extremes[0]:
            for b_val in other_extremes[1]:
                corner = [0.0, 0.0, 0.0]
                corner[chunk_axis] = chunk_val
                corner[other_axes[0]] = a_val
                corner[other_axes[1]] = b_val
                corners.append(corner)

    input_coords = np.array(corners) @ matrix.T + offset
    lo = input_coords[:, chunk_axis].min()
    hi = input_coords[:, chunk_axis].max()
    input_start = max(0, int(np.floor(lo)) - pad)
    input_stop = min(volume_shape[chunk_axis], int(np.ceil(hi)) + 1 + pad)
    return input_start, input_stop


def affine_transform_volume_combined(
    volume: Volume,
    matrix: NDArray[np.float64],
    offset: NDArray[np.float64],
    order: int,
    axis_to_chunk: int = 2,
) -> Volume:
    """Apply a combined 3-axis rotation to `volume` in one pass, chunked along `axis_to_chunk`.

    Unlike `rotate_volume_inplace`, this cannot safely overwrite `volume` chunk by chunk: because the
    rotation has no invariant axis, an output chunk's halo can reach back into input indices an earlier
    chunk already overwrote. So this allocates one fresh output array (as `mask_radial`/`flip_y`
    already do elsewhere in this file) and only ever reads from the untouched `volume`.

    `axis_to_chunk` defaults to 2 (z): for volumes shaped like these scans (x, y ~1500, z up to 6000),
    chunking along z keeps each chunk's cross-section (x * y) small, whereas chunking along x or y
    would keep the much larger x/y * z cross-section resident per chunk.

    `order <= 1` (nearest/linear) is exact under chunking: `affine_transform` only spline-prefilters
    its input when `order > 1`, and for `order <= 1` there is no prefilter step, so a chunked call
    reproduces an unchunked one to floating-point precision (~1e-12).

    `order` in `{2, 3}` is chunk-safe in practice but not bit-exact: prefiltering is a whole-array IIR
    filter, so prefiltering each padded chunk separately doesn't exactly reproduce prefiltering the
    full volume in one call. The filter's influence decays fast, though, and `_input_range_for_output_
    chunk`'s default `pad=2` already absorbs it -- validated by forcing pathologically small chunks
    (a handful of voxels) at `order=3` and comparing against an unchunked call: max error 1 uint8 level,
    on effectively zero voxels. Real chunk sizes (hundreds+ voxels under `ROTATION_CHUNK_BYTES`) will
    be smaller still. Separately, `order >= 2` can ring (overshoot/undershoot) at sharp edges like the
    can wall -- scipy clips this safely into uint8 range, so it's a mild edge artifact, not a
    correctness bug, but it's unrelated to chunking and worth knowing about if `crop_radial`'s Canny or
    `crop_axial`'s gradient edge detection get more sensitive to it than expected.

    `order > 3` (quartic/quintic) is untested here: the prefilter's poles decay more slowly as spline
    order increases, so `pad` may need to be larger than the default to stay chunk-safe.
    """
    if order > 3:
        logger.warning(
            f"affine_transform_volume_combined called with {order=} > 3; only order <= 3 has been "
            "validated as chunk-safe with the default pad in _input_range_for_output_chunk. Consider "
            "increasing that pad, or verifying against an unchunked affine_transform call."
        )

    shape = volume.shape
    output = np.empty_like(volume)
    chunk_size = _determine_chunk_size(
        shape, volume.itemsize, axis_to_chunk, ROTATION_CHUNK_BYTES
    )
    num_chunks = (shape[axis_to_chunk] + chunk_size - 1) // chunk_size
    logger.debug(
        f"Applying combined rotation in {num_chunks} chunk(s) of {chunk_size} along axis {axis_to_chunk}"
    )

    for n in range(num_chunks):
        output_start = n * chunk_size
        output_stop = min((n + 1) * chunk_size, shape[axis_to_chunk])

        input_start, input_stop = _input_range_for_output_chunk(
            matrix, offset, output_start, output_stop, axis_to_chunk, shape
        )

        input_index: list[slice] = [slice(None)] * 3
        input_index[axis_to_chunk] = slice(input_start, input_stop)
        input_chunk = volume[tuple(input_index)]

        # Shift both origins to zero for this call: matrix @ output_origin + offset - input_origin.
        output_origin = np.zeros(3)
        output_origin[axis_to_chunk] = output_start
        input_origin = np.zeros(3)
        input_origin[axis_to_chunk] = input_start
        local_offset = matrix @ output_origin + offset - input_origin

        output_shape = list(shape)
        output_shape[axis_to_chunk] = output_stop - output_start

        output_index: list[slice] = [slice(None)] * 3
        output_index[axis_to_chunk] = slice(output_start, output_stop)
        output[tuple(output_index)] = scipy.ndimage.affine_transform(
            input_chunk,
            matrix,
            offset=local_offset,
            output_shape=tuple(output_shape),
            order=order,
            mode="constant",
            cval=0.0,
        )

    return output


def align_combined(
    volume: Volume, params: Params, rotation_log: dict[str, float] | None = None
) -> Volume:
    """Drop-in alternative to `align()`: identical angle detection, single-pass combined rotation.

    Angle detection is identical to `align()` (same segmentation, same eigenvector analysis, same
    validation and minimum-angle skip). The only difference is how the rotation is applied: one
    `affine_transform_volume_combined` call instead of two `rotate_volume_inplace` calls.
    """
    logger.info("Running alignment in combined mode with a single rotation.")
    volume_segmented_cell = segment_cell_within_volume(
        volume, params.downsampling_factor, params.sigma_alignment
    )
    is_z_axis_largest_eigenvalue = (
        params.can_height_mm >= (math.sqrt(3) / 2) * params.can_diameter_mm
    )
    theta, phi = find_rotation_angles_cylindrical(
        volume_segmented_cell, is_z_axis_largest_eigenvalue
    )
    if rotation_log is not None:
        rotation_log.update(theta_degrees=float(theta), phi_degrees=float(phi))
    logger.info(f"Rotation angles: theta={theta:.3f}°, phi={phi:.3f}°")

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

    min_rotation_angle_degrees = calculate_min_rotation_angle_degrees(volume.shape[2])
    if (
        abs(theta) <= min_rotation_angle_degrees
        and abs(phi) <= min_rotation_angle_degrees
    ):
        logger.debug(
            f"Skipping combined rotation ({theta=:.3f}°, {phi=:.3f}°, "
            f"{min_rotation_angle_degrees=:.3f}°)"
        )
        return volume

    matrix, offset = combined_rotation_matrix_and_offset(theta, phi, volume.shape)
    return affine_transform_volume_combined(
        volume, matrix, offset, params.rotation_interpolation_order
    )


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------


def _is_already_processed(filename: str) -> bool:
    """True if this looks like a previously-generated volume, not a fresh scan to process."""
    stem = Path(filename).stem
    skip_suffixes = SKIP_NAME_SUFFIXES + (OUTPUT_NAME_SUFFIX,)
    return any(stem.endswith(suffix) for suffix in skip_suffixes)


def _config_for_log() -> dict[str, str | int | float | bool | None]:
    """Return the user-editable, capitalized configuration values in CSV-safe form."""
    config = {}
    for name, value in globals().items():
        if name.isupper() and name != "__BUILTINS__":
            if isinstance(value, Path):
                value = str(value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                value = json.dumps(value, default=str)
            config[name] = value
    return config


def _write_process_log(
    log_path: Path,
    input_path: Path,
    output_path: Path,
    rotation_log: dict[str, float],
    stage_timings: dict[str, float],
) -> None:
    """Append one processing run to a CSV file, creating its header if needed."""
    row: dict[str, object] = {
        "input_file_name": input_path.name,
        "output_file_name": output_path.name,
        "input_file_path": str(input_path),
        "output_file_path": str(output_path),
        **_config_for_log(),
        **rotation_log,
        **{
            f"time_seconds_{name}": stage_timings.get(name)
            for name in (
                "load_raw_volume",
                "adjust_contrast",
                "align",
                "downcast_to_uint8",
                "crop_radial",
                "mask_radial",
                "crop_axial",
                "flip_y",
                "save_raw_volume",
            )
        },
        "time_seconds_total": stage_timings["total_processing"],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Appended processing log: {log_path}")


def process_one(filename: str) -> int:
    """Load one raw volume, run the postprocessing steps, and write the result."""
    input_path = VOLUME_DIR / Path(filename)
    if not input_path.is_file():
        logger.error(f"Input file does not exist: {input_path}")
        return 1

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
        return 1
    if output_path.exists() and not OVERWRITE_OUTPUT:
        logger.error(
            f"Output file already exists: {output_path}. Set OVERWRITE_OUTPUT = True to replace it."
        )
        return 1

    params = Params.from_config()
    logger.debug(f"Parameters: {params}")
    rotation_log: dict[str, float] = {}
    stage_timings: dict[str, float] = {}
    processing_start = time()

    try:
        stage_start = time()
        volume = load_raw_volume(input_path, VOXELS_X, VOXELS_Y, VOXELS_Z)
        stage_timings["load_raw_volume"] = time() - stage_start
        logger.info(f"Volume shape (x, y, z): {volume.shape}, dtype: {volume.dtype}")

        steps = (
            ("adjust_contrast", RUN_ADJUST_CONTRAST, adjust_contrast),
            (
                "align",
                RUN_ALIGN,
                lambda volume, params: (
                    align if not ALIGN_COMBINED else align_combined
                )(volume, params, rotation_log),
            ),
            # Always runs (not config-gated): every step below assumes uint8, see downcast_to_uint8.
            ("downcast_to_uint8", True, downcast_to_uint8),
            ("crop_radial", RUN_CROP_RADIAL, crop_radial),
            ("mask_radial", RUN_MASK_RADIAL, mask_radial),
            ("crop_axial", RUN_CROP_AXIAL, crop_axial),
            ("flip_y", RUN_FLIP_Y, flip_y),
        )
        for name, enabled, step in steps:
            if not enabled:
                logger.info(f"Skipping {name}")
                continue
            start_time = time()
            volume = step(volume, params)
            stage_timings[name] = time() - start_time
            logger.info(
                f"{name} finished in {stage_timings[name]:.2f}s; shape (x, y, z): {volume.shape}"
            )

        stage_start = time()
        save_raw_volume(output_path, volume)
        stage_timings["save_raw_volume"] = time() - stage_start
        total_processing_time = time() - processing_start
        stage_timings["total_processing"] = total_processing_time
        logger.info(f"Total processing time: {total_processing_time:.2f}s")
        _write_process_log(
            input_path.parent / PROCESS_LOG_FILE.name,
            input_path,
            output_path,
            rotation_log,
            stage_timings,
        )
    except ValidationError as error:
        logger.error(str(error))
        return 1

    x, y, z = volume.shape
    logger.info(
        f"Done. Output is uint8 with dims x={x}, y={y}, z={z}, written as (z, y, x)."
    )
    return 0


def _find_project_root(start_path: Path) -> Path:
    for dir in [start_path, *start_path.parents]:
        if (dir / ".git").exists() or (dir / "pyproject.toml").exists():
            return dir
    raise FileNotFoundError("Project root with .git directory not found.")


def main(
    filenames: list[str] = typer.Argument(
        ..., help="Names of the raw volume files to process, within VOLUME_DIR."
    ),
) -> int:
    """Process one or more raw volumes in sequence, in a single invocation."""
    logging.basicConfig(
        level=logging.DEBUG if VERBOSE else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    global VOLUME_DIR
    if not VOLUME_DIR.exists():
        project_root = _find_project_root(Path(__file__).resolve().parent)
        VOLUME_DIR = project_root / VOLUME_DIR
        if not VOLUME_DIR.exists():
            logger.error(f"Volume directory does not exist: {VOLUME_DIR}")
            raise typer.Exit(code=1)

    to_process = [f for f in filenames if not _is_already_processed(f)]
    skipped = [f for f in filenames if _is_already_processed(f)]
    if skipped:
        logger.info(f"Skipping {len(skipped)} already-processed file(s): {skipped}")

    failures = []
    for i, filename in enumerate(to_process, start=1):
        logger.info(f"[{i}/{len(to_process)}] Processing {filename}")
        if process_one(filename) != 0:
            failures.append(filename)

    if failures:
        logger.error(f"Failed on {len(failures)}/{len(to_process)} file(s): {failures}")
        return 1
    return 0


if __name__ == "__main__":
    typer.run(main)
