from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage import morphology


# ------------------------ Particle Extraction & Enhancement -------------------------------
# Tools for segmenting a particle from a volume at a know location and then
# enhancing its visibility within a single projection slice.
@dataclass
class ParticleExtractor:
    """
    Particle extractor class for analyzing and enhancing local particle regions within a 3D volume.

    initialization and configuration of the particle extractor.
    1. On initialization set the local analysis volume and neighborhood size.
    2. Set the voxel location with a search window to find the local maximum intensity.
    3. Call project_particle to get the 2D projection of the particle.

    Internally, this is the particle extraction steps:
    1. Estimate the background of the local volume as the median intensity of the projection along the z-axis.
    2. Subtract the estimated background from the local volume.
    3. Use a MAD estimate to assess the noise level in the local volume.
    4. Seed a particle mask using the snr_high threshold and retain only the connected region corresponding to the given location.
    5. Grow the connected region to include neighboring voxels that meet the snr_low threshold.

    The 2D projected particle is obtained by a weighted sum of the particle intensities along the z-axis. The weights are
    confidences from the particle snr values.
    """

    volume_data: NDArray[np.uint8]  # The underlying 3D volume with coordinates (z,y,x)

    neighborhood_size: tuple[int, int, int] = (15, 15, 15)
    snr_high: float = 4.0
    snr_low: float = 2.5

    # The (z, y, x) coordinates of the voxel to be analyzed. Must be set explicitly using set_voxel_location.
    point_zyx: tuple[int, int, int] | None = field(default=None, init=False)
    _particle_region: NDArray[np.bool_] | None = field(default=None, init=False)
    _particle: NDArray | None = field(default=None, init=False)
    _noise_estimate: float | None = field(default=None, init=False)

    def __init__(
        self,
        vol_data: NDArray[np.uint8],
        neighborhood_size: tuple | int = 15,
        snr_high: float = 4.0,
        snr_low: float = 2.5,
    ) -> None:
        # Ensure that the neighborhood size is odd in all dimensions
        if isinstance(neighborhood_size, int):
            if neighborhood_size % 2 == 0:
                neighborhood_size += 1
            neighborhood_size = (
                neighborhood_size,
                neighborhood_size,
                neighborhood_size,
            )
        else:
            if any(s % 2 == 0 for s in neighborhood_size):
                neighborhood_size = tuple(
                    s + 1 if s % 2 == 0 else s for s in neighborhood_size
                )

        self.volume_data = vol_data
        self.neighborhood_size = neighborhood_size
        self.snr_high = snr_high
        self.snr_low = snr_low
        self.point_zyx = None

    def set_voxel_location(
        self,
        point_zyx: tuple[int, int, int],
        search_window: tuple[int, int, int] | int = 3,
    ) -> None:
        """
        Set the voxel location within a search window where the max intensity is located.

        The search window is used to identify the local maximum intensity around the specified voxel location.
        Args:
            point_zyx (tuple[int, int, int]): The (z, y, x) coordinates of the center voxel to search for the max response.
            search_window (tuple[int, int, int], optional): The size of the search window in (z, y, x) dimensions. Defaults to (3, 3, 3).
        """

        """Return the global (z, y, x) location of the maximum in the search window."""
        if self.volume_data is None:
            raise ValueError("Volume data is not set.")

        z, y, x = point_zyx
        depth, height, width = self.volume_data.shape

        if isinstance(search_window, int):
            search_window = (search_window, search_window, search_window)

        sz, sy, sx = search_window
        z1, z2 = max(0, z - sz // 2), min(depth, z + sz // 2 + 1)
        y1, y2 = max(0, y - sy // 2), min(height, y + sy // 2 + 1)
        x1, x2 = max(0, x - sx // 2), min(width, x + sx // 2 + 1)

        neighborhood = self.volume_data[z1:z2, y1:y2, x1:x2]

        local_zyx = np.unravel_index(
            np.argmax(neighborhood),
            neighborhood.shape,
        )

        self.point_zyx = (
            z1 + int(local_zyx[0]),
            y1 + int(local_zyx[1]),
            x1 + int(local_zyx[2]),
        )

    def get_neighborhood_volume(self) -> NDArray:
        if self.volume_data is None:
            raise ValueError("Volume data is not set.")
        if self.point_zyx is None:
            raise ValueError("Voxel location not set.")

        z, y, x = self.point_zyx
        nz, ny, nx = self.neighborhood_size
        z1 = max(z - nz // 2, 0)
        y1 = max(y - ny // 2, 0)
        x1 = max(x - nx // 2, 0)
        z2 = min(z + nz // 2 + 1, self.volume_data.shape[0])
        y2 = min(y + ny // 2 + 1, self.volume_data.shape[1])
        x2 = min(x + nx // 2 + 1, self.volume_data.shape[2])
        return self.volume_data[z1:z2, y1:y2, x1:x2]

    def get_baseline_trend(self) -> NDArray:
        # Returns the baseline trend for the local neighborhood
        if self.point_zyx is None:
            raise ValueError("Voxel location not set.")
        neighborhood = self.get_neighborhood_volume()
        return np.median(neighborhood, axis=0).astype(np.float32)

    def get_residual_volume(self) -> NDArray:
        """Get the residual volume for the local neighborhood.

        Raises:
            ValueError: If the voxel location has not been set.

        Returns:
            NDArray: The residual volume obtained by subtracting the baseline trend from the neighborhood volume.

        Raises:
            ValueError: if the particle's location has not been set.

        Returns:
            NDArray: The residual volume obtained by subtracting the baseline trend from the neighborhood volume.
        """
        # Get the local residual by subtracting the baseline trend from the neighborhood volume
        if self.point_zyx is None:
            raise ValueError("Voxel location not set.")
        neighborhood = self.get_neighborhood_volume()
        baseline = self.get_baseline_trend()
        return np.subtract(neighborhood, baseline, dtype=np.float32)

    @property
    def noise_estimate(self) -> float:
        # Estimate the noise level in the local neighborhood using an iterative clipping method
        if self._noise_estimate is not None:
            return self._noise_estimate

        if self.point_zyx is None:
            raise ValueError("Voxel location not set.")

        residual = self.get_residual_volume()
        mask = np.ones_like(residual, dtype=bool)
        local_point_zyx = tuple(s // 2 for s in residual.shape)
        mask[local_point_zyx] = False

        med = float(np.median(residual[mask]))
        mad = float(np.median(np.abs(residual[mask] - med)))
        mask = np.abs(residual - med) < 3.5 * mad
        med = float(np.median(residual[mask]))
        mad = float(np.median(np.abs(residual[mask] - med)))
        self._noise_estimate = (
            mad * 1.4826
        )  # Convert MAD to an estimate of the standard deviation
        return self._noise_estimate

    @property
    def particle_region(self) -> NDArray[np.bool_]:
        """
        From the center location find the connected component from the center location with values above the noise floor.

        Region extraction results in a connected component from the center of the particle region.
        1. estimate the noise level in the local neighborhood.
        2. create a high-threshold mask based on the noise level.
        3. identify the connected component from the center voxel within the high-threshold mask.
        4. grow the connected component into a lower-threshold mask to capture the skirt region.

        Returns:
            NDArray[bool]: connected particle region
        """
        if self._particle_region is not None:
            return self._particle_region

        if self.point_zyx is None:
            raise ValueError("Voxel location not set.")
        residual = self.get_residual_volume()
        noise_level = self.noise_estimate

        # Find the connected component from the center location with values above the noise floor
        mask_hi = residual > self.snr_high * noise_level

        center = tuple(s // 2 for s in residual.shape)
        if not mask_hi[center]:
            return np.zeros_like(mask_hi, dtype=bool)

        # From the center voxel grow the seeded into the high-threshold mask using full connectivity (26)
        seed = np.zeros_like(mask_hi, dtype=bool)
        seed[center] = True
        structure = ndimage.generate_binary_structure(3, 3)
        mask_connected = ndimage.binary_propagation(
            seed, structure=structure, mask=mask_hi
        )

        if mask_connected.sum() == 0:
            return mask_connected

        # Now capture the skirt region with a lower threshold, ie, grow mask_conncted into mask_lo
        mask_lo = residual > self.snr_low * noise_level
        grown_mask = ndimage.binary_propagation(
            mask_connected & mask_lo, structure=structure, mask=mask_lo
        )

        # Clean up the mask by filling holes and closing.
        cleaned_mask = ndimage.binary_fill_holes(grown_mask, structure=structure)
        cleaned_mask = morphology.binary_closing(cleaned_mask, morphology.ball(1))
        self._particle_region = cleaned_mask | grown_mask
        return self._particle_region

    @property
    def particle(self) -> NDArray:
        """
        Extract the particle from the residual volume. This is accomplished by applying the particle region mask obtained from `extract_particle_region` to the residual volume.

        Returns:
            NDArray: The extracted particle region, with the same shape as the residual volume.

        """
        if self._particle is None:
            self._particle = self.particle_region * self.get_residual_volume()
        return self._particle

    def get_particle_confidence_map(self) -> NDArray:
        """
        Compute the confidence of the extracted particle based on the signal-to-noise ratio (SNR) within the particle region and the clipping thresholds defined by `snr_low` and `snr_high`.

        Think of this as a fuzzy segmentation of the particle region.

        Args:
            particle (NDArray): The 3D particle volume.

        Returns:
            NDArray: The 3D confidence map of the particle.
        """
        particle = self.particle
        particle_region = particle > 0
        if particle_region.sum() == 0:
            return np.zeros_like(particle, dtype=np.float32)

        noise_level = self.noise_estimate
        snr_map = (particle / noise_level).astype(np.float32)

        # Formulate a weighed map in the range [0,1] based on the SNR values of the particles
        _half = np.float32(0.5)
        snr_0 = np.clip(self.snr_low - _half, a_min=0, a_max=None)
        snr_1 = np.clip(
            self.snr_high + _half, a_min=snr_0 + np.finfo(np.float32).eps, a_max=None
        )
        confidence_map = np.clip((snr_map - snr_0) / (snr_1 - snr_0), a_min=0, a_max=1)
        return confidence_map

    def project_particle(self) -> NDArray:
        # Project along z using confidence as an additive weight.
        # Do not normalize by summed confidence: additional supporting
        # voxels along a z-profile should increase projected strength.
        confidence = self.get_particle_confidence_map()
        particle = self.particle
        # A low-confidence voxel contributes proportionally less, while
        # each additional voxel contributes additional evidence.
        confidence_projected_particle = (particle * confidence).sum(
            axis=0, dtype=np.float32
        )

        particle_max = max(particle.max(), 0.0)
        projected_max = max(confidence_projected_particle.max(), 0.0)
        if projected_max > particle_max:
            confidence_projected_particle *= np.float32(particle_max / projected_max)
        return confidence_projected_particle


def blend_particle(
    image: NDArray[np.uint8],
    particle: NDArray[np.floating],
    particle_center: tuple[int, int],
    blur: float = 1.0,
    truncate_filter: float = 2.0,
    gain: float = 0.75,
) -> NDArray[np.uint8]:
    """Blend a projected particle into an image.

    The particle is embedded at ``particle_center``, optionally blurred
    with a Gaussian filter, scaled by ``gain``, and added to the image.
    ``truncate_filter`` controls the Gaussian kernel support in units of
    the blur standard deviation: the filter radius is approximately
    ``blur * truncate_filter`` pixels on each side of the kernel.

    Args:
        image: Two-dimensional uint8 image to which the particle is added.
        particle: Two-dimensional projected particle intensity array.
        particle_center: ``(y, x)`` location of the particle center in
            ``image`` coordinates.
        blur: Gaussian-filter standard deviation in pixels. A value of
            zero disables blurring.
        truncate_filter: Gaussian-filter truncation radius in units of
            ``blur``. Smaller values produce shorter filter tails.
        gain: Multiplicative intensity applied to the blurred particle
            before it is added to the image.

    Returns:
        The blended image clipped to ``[0, 255]`` and converted to
        ``np.uint8``.

    Raises:
        ValueError: If the inputs are not two-dimensional, ``blur`` is
            negative, or the requested Gaussian kernel is too narrow.
    """
    if image.ndim != 2 or particle.ndim != 2:
        raise ValueError("image and particle must both be 2D arrays")
    if blur < 0:
        raise ValueError("blur must be non-negative")
    if blur * truncate_filter < 1.5:
        raise ValueError("blur * truncate_filter must be at least 1.5")

    def _embed_particle(
        particle: NDArray[np.floating],
        shape: tuple[int, int],
        location: tuple[int, int],
    ) -> NDArray[np.floating]:
        """
        Embed a particle image into an array of the given shape at the specified location.
        Args:
            particle: Two-dimensional projected particle intensity array.
            shape: Shape of the output array.
            location: ``(y, x)`` location of the particle center in the output array.

        Returns:
            The embedded particle image as a floating-point array of the specified shape.
        """
        embedded_particle = np.zeros(shape, dtype=np.float32)
        y0, x0 = location
        y_start = y0 - particle.shape[0] // 2
        x_start = x0 - particle.shape[1] // 2
        y_stop = y_start + particle.shape[0]
        x_stop = x_start + particle.shape[1]

        iy0, iy1 = max(y_start, 0), min(y_stop, shape[0])
        ix0, ix1 = max(x_start, 0), min(x_stop, shape[1])
        if iy0 < iy1 and ix0 < ix1:
            py0, px0 = iy0 - y_start, ix0 - x_start
            py1, px1 = py0 + (iy1 - iy0), px0 + (ix1 - ix0)
            embedded_particle[iy0:iy1, ix0:ix1] = particle[py0:py1, px0:px1]

        return embedded_particle

    embedded_particle = _embed_particle(particle, image.shape, particle_center)
    # embedded_particle = np.zeros(image.shape, dtype=np.float32)
    # y0, x0 = particle_center
    # y_start = y0 - particle.shape[0] // 2
    # x_start = x0 - particle.shape[1] // 2
    # y_stop = y_start + particle.shape[0]
    # x_stop = x_start + particle.shape[1]

    # # Clip the insertion window so edge-adjacent particles are supported.
    # iy0, iy1 = max(y_start, 0), min(y_stop, image.shape[0])
    # ix0, ix1 = max(x_start, 0), min(x_stop, image.shape[1])
    # if iy0 < iy1 and ix0 < ix1:
    #     py0, px0 = iy0 - y_start, ix0 - x_start
    #     py1, px1 = py0 + (iy1 - iy0), px0 + (ix1 - ix0)
    #     embedded_particle[iy0:iy1, ix0:ix1] = particle[py0:py1, px0:px1]

    if blur > 0 and embedded_particle.max() > 0:
        original_max = embedded_particle.max()
        embedded_particle = ndimage.gaussian_filter(
            embedded_particle, sigma=blur, truncate=truncate_filter
        )
        blurred_max = embedded_particle.max()
        if blurred_max > 0:
            embedded_particle *= original_max / blurred_max

    particle_added = image + gain * embedded_particle
    return np.clip(np.rint(particle_added), 0, 255).astype(np.uint8)


#
# -----------------------------------------------------------------------------
