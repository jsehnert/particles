import numpy as np
from numpy.typing import NDArray


def softmax_projection(
    slab: NDArray[np.uint8],
    beta: float,
    window: int = 3,
) -> NDArray[np.float32]:
    """
    Project a z-slab to one scalar per (x, y) via softmax over a z run-sum.

    This is a soft version of max projection that rewards contiguous bright
    clusters along z, while suppressing isolated hot voxels.

    For each column ``v(:, x, y)`` along the first (z) axis this computes a
    sliding-window running mean of length ``window``, then a
    temperature-controlled soft-max over the resulting sequence::

        softmax_beta(s) = (1 / beta) * log( sum_j exp(beta * s_j) )

    A genuine contiguous bright cluster of length >= ``window`` fills the whole
    window and is rewarded; an isolated hot voxel is diluted by ~1/window.

    Parameters
    ----------
    slab
        Slab of shape ``(N, H, W)`` with z along axis 0. dtype ``uint8``.
    beta
        Soft-max temperature (inverse scale), in run-sum output units.
        Larger -> closer to a hard max; -> 0 -> mean. Positive, finite.
    window
        Sliding-window width along z, ``1 <= window <= N``. Sets the minimum
        cluster size rewarded; do not exceed the smallest anomaly of interest.

    Returns
    -------
    NDArray[np.float32]
        Array of shape ``(H, W)`` with the soft-max response per column.
    """
    volume = slab
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D (N, H, W); got shape {volume.shape!r}")
    if volume.dtype != np.uint8:
        raise ValueError(f"volume must be uint8; got {volume.dtype!r}")

    n_z = volume.shape[0]
    if not 1 <= window <= n_z:
        raise ValueError(f"window must be in [1, N={n_z}]; got {window}")
    if not (np.isfinite(beta) and beta > 0.0):
        raise ValueError(f"beta must be a positive finite float; got {beta!r}")

    # Sliding-window run-sum along z via cumulative-sum differences: O(N*H*W).
    # uint16 is safe: max csum value = N*255 <= 61*255 = 15,555 < 65,535.
    csum = np.zeros((n_z + 1, *volume.shape[1:]), dtype=np.uint16)
    np.cumsum(volume, axis=0, out=csum[1:])
    run_mean: NDArray[np.float32] = (csum[window:] - csum[:-window]).astype(np.float32)
    run_mean /= window

    # Numerically stable soft-max along z (axis 0).
    s_max = run_mean.max(axis=0)
    shifted = run_mean - s_max
    shifted *= beta
    np.exp(shifted, out=shifted)
    acc = shifted.sum(axis=0)
    result: NDArray[np.float32] = s_max + np.log(acc) / beta
    return result


def contiguity_projection(
    slab: NDArray[np.uint8],
    beta: float,
    window: int = 3,
    *,
    tau: float = 64.0,
    kappa: float = 0.2,
) -> NDArray[np.float32]:
    """
    Project a z-slab to one scalar per (x, y), rewarding *contiguous* bright
    runs along z over the same brightness arranged with gaps.

    Unlike a run-mean (which is permutation-invariant and scores [h,l,h] and
    [h,h,l] identically), this scores a window by a sum of adjacent-pair
    products over soft brightness weights, so consecutive bright voxels are
    rewarded and split bright voxels are penalised. Works for any window>=2.

    For each z-column the per-voxel brightness weight is
    ``w = sigmoid(kappa * (value - tau))`` in [0, 1]. The per-pair score is
    ``p_j = w_j * w_{j+1}`` (large only when *both* neighbours are bright).
    A length-``window`` voxel window contains ``window-1`` such pairs; their
    mean is the window's contiguity score, and a temperature-``beta`` soft-max
    over all window positions along z gives the per-column response.

    Parameters
    ----------
    slab
        Slab of shape ``(N, H, W)`` with z along axis 0, dtype ``uint8``.
    beta
        Soft-max temperature over window positions (inverse scale), in
        pair-score units (the score is in [0, 1]). Larger -> harder max.
    window
        Window width along z, ``2 <= window <= N``. Number of pairs summed
        is ``window - 1``.
    tau
        Brightness mid-point (in voxel units) for the soft threshold.
    kappa
        Soft-threshold steepness (per voxel unit). Larger -> sharper
        bright/dark gate; -> 0 -> all weights ~0.5 (loses selectivity).

    Returns
    -------
    NDArray[np.float32]
        Array of shape ``(H, W)`` with the contiguity response per column.
    """
    volume = slab
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D (N, H, W); got shape {volume.shape!r}")
    if volume.dtype != np.uint8:
        raise ValueError(f"volume must be uint8; got {volume.dtype!r}")

    n_z = volume.shape[0]
    if not 2 <= window <= n_z:
        raise ValueError(f"window must be in [2, N={n_z}]; got {window}")
    if not (np.isfinite(beta) and beta > 0.0):
        raise ValueError(f"beta must be a positive finite float; got {beta!r}")
    if not (np.isfinite(kappa) and kappa > 0.0):
        raise ValueError(f"kappa must be a positive finite float; got {kappa!r}")

    vol = volume.astype(np.float32, copy=False)

    # Soft brightness gate in [0, 1], computed stably.
    w = 1.0 / (1.0 + np.exp(-kappa * (vol - tau), dtype=np.float64))

    # Adjacent-pair products along z: shape (n_z - 1, H, W).
    # p_j large only when BOTH w_j and w_{j+1} are bright -> rewards adjacency.
    pair = w[:-1] * w[1:]
    n_pair = pair.shape[0]  # == n_z - 1
    n_win_pairs = window - 1  # pairs spanned by one window

    # Sliding sum of (window-1) consecutive pair scores via cumsum differences.
    csum = np.empty((n_pair + 1, *pair.shape[1:]), dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(pair, axis=0, out=csum[1:])
    win_score = (
        csum[n_win_pairs:] - csum[:-n_win_pairs]
    )  # (n_pair - n_win_pairs + 1, H, W)
    win_score /= n_win_pairs  # mean pair-score in [0, 1]

    # Numerically stable soft-max over window positions (axis 0).
    s_max = win_score.max(axis=0)
    shifted = win_score - s_max
    shifted *= beta
    np.exp(shifted, out=shifted)
    acc = shifted.sum(axis=0)
    result = (s_max + np.log(acc) / beta).astype(np.float32, copy=False)
    return result
