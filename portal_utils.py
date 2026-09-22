from __future__ import annotations

from numpy.typing import NDArray


def rawyx_to_portalyx(
    y_raw: int, x_raw: int, volume_shape: tuple[int, int, int]
) -> tuple[int, int]:
    """
    Convert y,x coordinates from the raw volume to portal coordinates.

    Note: We don't transform the z-coordinates because in the portal the z value is a range
          within the raw volume

    Args:
        y_raw (int): The raw y-coordinate.
        x_raw (int): The raw x-coordinate.
        volume_shape (tuple[int, int, int]): The shape of the raw volume.

    Returns:
        tuple[int, int]: The portal coordinates (y, x).
    """
    # Example conversion, replace with actual logic
    portal_y = volume_shape[1] - 1 - y_raw
    portal_x = x_raw
    return portal_y, portal_x


def portal_pos_to_slab_range(
    portal_z: int | float,
    slab_thickness: float = 1.0,
    voxel_size_mm: float = 0.0164,
    padding_mm: float = -0.5,
) -> tuple[int, int]:
    """
    Convert a portal position to a slab range in an all-voxels volume.

    Args:
        portal_z (int | float): The z-coordinate of the portal position.
        slab_thickness (float, optional): The thickness of the slab in millimeters. Defaults to 1.0.
        voxel_size_mm (float, optional): The size of a voxel in millimeters. Defaults to 0.0164.
        padding_mm (float, optional): The padding to apply to the portal position in millimeters. Defaults to -0.5.

    Returns:
        tuple[float, float]: The minimum and maximum voxel indices for the slab.
    """
    z_pos = portal_z - padding_mm
    z_min = z_pos / voxel_size_mm
    z_max = z_min + slab_thickness / voxel_size_mm

    return round(z_min), round(z_max)


def portal_slab(
    volume: NDArray,
    portal_z: int | float,
    slab_thickness: float = 1.0,
    voxel_size_mm: float = 0.0164,
    padding_mm: float = -0.5,
) -> NDArray:
    """
    Identifies the sub-volume of the volume that the portal uses for 'slabbing'. In standard practice, the
    slab is averaged over z with a low-pass filter, such as a centered triangle. In special cases, like particle
    surfacing a max projection filter can be used.

    Args:
        volume (NDArray): The input 3D volume.
        portal_z (int | float): The z-coordinate of the portal position.
        slab_thickness (float, optional): The thickness of the slab in millimeters. Defaults to 1.0.
        voxel_size_mm (float, optional): The size of a voxel in millimeters. Defaults to 0.0164.
        padding_mm (float, optional): The padding to apply to the portal position in millimeters. Defaults to -0.5.

    Returns:
        NDArray: The sub-volume corresponding to the slab.
    """
    z_range = portal_pos_to_slab_range(
        portal_z,
        slab_thickness=slab_thickness,
        voxel_size_mm=voxel_size_mm,
        padding_mm=padding_mm,
    )
    return volume[z_range[0] : z_range[1], :, :]
