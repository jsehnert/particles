from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def parse_metadata(meta_path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with meta_path.open() as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, _, value = line.partition("=")
                metadata[key.strip()] = value.strip()
    return metadata


def load_volume(volume_path: Path) -> NDArray[np.uint8] | NDArray[np.float32]:
    meta_path = volume_path.with_suffix(".xtekhelixct")
    metadata = parse_metadata(meta_path)

    x = int(metadata["VoxelsX"])
    y = int(metadata["VoxelsY"])
    z = int(metadata["VoxelsZ"])

    dtype: type[np.uint8] | type[np.float32] = (
        np.float32 if volume_path.suffix == ".vol" else np.uint8
    )

    return np.memmap(volume_path, dtype=dtype, mode="r", shape=(z, y, x))


def voxel_size_mm(volume_path: Path) -> tuple[float, float, float]:
    # meta_path = volume_path.with_suffix(".xtekhelixct")
    # metadata = parse_metadata(meta_path)
    return float(0.01640435), float(0.01640435), float(0.01640435)
