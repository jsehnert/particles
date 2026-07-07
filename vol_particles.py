from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Protocol, TypeAlias
from scipy import ndimage
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]


class ResidualSource(Protocol):
    """Lazy provider of residual + noise for a z-range. Backed by memmap,
    HDF5, zarr, or on-the-fly background subtraction — caller's choice."""
    shape: tuple[int, int, int]   # (Z, H, W)

    def read(self, z0: int, z1: int) -> tuple[FloatArray, FloatArray]:
        """Return (R, sigma_R) for slices [z0, z1).
        sigma_R shaped (z1-z0,1,1), (z1-z0,H,W), or broadcastable."""
        ...

class MyResidualSource:
    def __init__(self, vol: NDArray[np.uint8]):
        self.vol: NDArray[np.uint8] = vol
        self.shape: tuple[int, int, int] = vol.shape


    def read(self, z0: int, z1: int) -> tuple[FloatArray, FloatArray]:
        # Placeholder implementation: return the raw volume as residual and a constant noise level.
        R = self.vol[z0:z1].astype(np.float32)
        sigma_R = np.full((z1 - z0, 1, 1), 10.0, dtype=np.float32)  # Example noise level
        return R, sigma_R

@dataclass
class Candidate:
    n_voxels: int
    z_min: int
    z_max: int
    z_extent: int
    centroid: tuple[float, float, float]   # global (z, y, x)
    snr_cluster: float


def iter_slabs(
    Z: int, core: int, halo: int
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (read_z0, read_z1, core_z0, core_z1).
    Core is the ownership region; [read_z0,read_z1) includes the halo."""
    for c0 in range(0, Z, core):
        c1 = min(c0 + core, Z)
        r0 = max(0, c0 - halo)
        r1 = min(Z, c1 + halo)
        yield r0, r1, c0, c1


def detect_streaming(
    src: ResidualSource,
    k_sigma: float = 4.7,
    min_size: int = 2,
    z_extent_bounds: tuple[int, int] = (1, 4),
    core_slices: int = 512,
) -> list[Candidate]:
    Z, H, W = src.shape
    z_lo, z_hi = z_extent_bounds
    halo = z_hi                      # halo ≥ max physical z-extent
    structure = ndimage.generate_binary_structure(3, 1)

    out: list[Candidate] = []

    for r0, r1, c0, c1 in iter_slabs(Z, core_slices, halo):
        R, sigma = src.read(r0, r1)          # (r1-r0, H, W)
        snr = R / sigma
        seed = snr >= k_sigma

        labels, n = ndimage.label(seed, structure=structure)
        if n == 0:
            continue

        ids = np.arange(1, n + 1)
        sizes = ndimage.sum_labels(
            np.ones_like(labels, dtype=np.int64), labels, ids)
        objs = ndimage.find_objects(labels)

        for i, lid in enumerate(ids):
            n_vox = int(sizes[i])
            if n_vox < min_size:
                continue
            sl = objs[i]
            z_min_loc, z_max_loc = sl[0].start, sl[0].stop - 1
            z_extent = z_max_loc - z_min_loc + 1
            if not (z_lo <= z_extent <= z_hi):
                continue

            comp = labels == lid
            cz_loc, cy, cx = ndimage.center_of_mass(comp)
            cz_glob = cz_loc + r0

            # --- ownership: claim only if centroid falls in this slab's core ---
            if not (c0 <= cz_glob < c1):
                continue

            snr_cluster = float(snr[comp].sum() / np.sqrt(n_vox))
            out.append(Candidate(
                n_voxels=n_vox,
                z_min=z_min_loc + r0, z_max=z_max_loc + r0,
                z_extent=z_extent,
                centroid=(cz_glob, float(cy), float(cx)),
                snr_cluster=snr_cluster,
            ))

    return out
