from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from metal import extract_metal_mask
from utils import identify_cylindrical_support

# ----------------------------------------------------------------------------
# Noise estimation from EM-maximization fit to a 2-component mixture distribution
# of the pixel differences distr = w * delta(x) + (1-w) * N(x; mu, sigma).
# ----------------------------------------------------------------------------


def _build_diff_histogram(
    data: NDArray[np.int16],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build a compact histogram for int16 pixel-difference data bounded to [-255, 255].

    Returns (hist, bin_centers) where hist.shape == (511,) and bin_centers[i] == i - 255.
    Using a 511-bin histogram instead of the full raw array reduces each EM iteration
    from O(N) to O(511), eliminating the large float64 intermediate allocation.
    """
    hist = np.bincount(data.ravel().astype(np.int32) + 255, minlength=511).astype(
        np.float64
    )
    bin_centers = np.arange(-255, 256, dtype=np.float64)
    return hist, bin_centers


def _em_fit_from_histogram(
    hist: NDArray[np.float64],
    bin_centers: NDArray[np.float64],
    trunc_k: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[float, float, float]:
    """
    EM fit of p(x) = w·delta(x=0) + (1-w)·N(x; mu, sigma) on a pre-built histogram.
    All operations are O(511) regardless of the original data size.
    """

    def truncation_factor(k: float) -> float:
        phi = np.exp(-0.5 * k * k) / np.sqrt(2.0 * np.pi)
        Phi = 0.5 * (1.0 + math.erf(k / np.sqrt(2.0)))
        return 1.0 - (2.0 * k * phi) / (2.0 * Phi - 1.0)

    k = trunc_k
    c_k_2_inv = 1.0 / np.sqrt(truncation_factor(k))

    n = hist.sum()
    n_zeros = hist[255]  # bin for value == 0
    zero_mask = np.zeros(511, dtype=bool)
    zero_mask[255] = True

    # Initialise from non-zero bins
    nz_hist = hist.copy()
    nz_hist[255] = 0.0
    nz_total = nz_hist.sum()
    w = float(np.clip(n_zeros / n, 1e-6, 1.0 - 1e-6))
    if nz_total > 0:
        mu = float((nz_hist * bin_centers).sum() / nz_total)
        sigma = float(np.sqrt((nz_hist * (bin_centers - mu) ** 2).sum() / nz_total))
    else:
        mu = 0.0
        sigma = 1.0
    sigma = max(sigma, 1e-6)

    _sqrt2pi_inv = 1.0 / np.sqrt(2.0 * np.pi)

    for _ in range(max_iter):
        # E-step: responsibility of delta for x=0 bins
        gauss0 = _sqrt2pi_inv / sigma * np.exp(-0.5 * (mu / sigma) ** 2)
        denom = w + (1.0 - w) * gauss0
        r0 = w / denom if denom > 1e-8 else 1.0  # P(delta | x=0)

        # M-step: exclude outliers beyond ±3σ from the Gaussian fit
        inlier = np.abs(bin_centers - mu) <= k * sigma
        w_new = float(np.clip(n_zeros * r0 / n, 1e-6, 1.0 - 1e-6))

        gw = np.where(zero_mask, 1.0 - r0, 1.0) * hist * inlier
        total_gw = gw.sum()
        if total_gw > 0.0:
            mu_new = float((gw * bin_centers).sum() / total_gw)
            sigma_new = max(
                float(np.sqrt((gw * (bin_centers - mu_new) ** 2).sum() / total_gw))
                * c_k_2_inv,
                1e-6,
            )
        else:
            mu_new, sigma_new = mu, sigma * c_k_2_inv

        if (
            abs(w_new - w) < tol
            and abs(mu_new - mu) < tol
            and abs(sigma_new - sigma) < tol
        ):
            w, mu, sigma = w_new, mu_new, sigma_new
            break

        w, mu, sigma = w_new, mu_new, sigma_new

    return w, mu, sigma


def fit_mixture_distribution(
    data: NDArray[np.int16],
    clip_scale: float = 2.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[float, float, float]:
    """
    For a given set of residual data, fit a 2-component mixture model using EM (expectation-maximization) to estimate the parameters of the underlying Gaussian distribution and the weight of the delta mass at zero.

    EM fit of p(x) = w·delta(x=0) + (1-w)·N(x; mu, sigma).
    Returns (w, mu, sigma).

    args
    ----
    data
        1D array of residual values to fit the mixture model to.
    max_iter
        Maximum number of EM iterations to perform.
    tol
        Convergence threshold for parameter changes; EM stops when all parameters change by less than tol.

    returns
    -------
    w
        Estimated weight of the delta mass at zero (0 < w < 1).
    mu
        Estimated mean of the Gaussian component.
    sigma
        Estimated standard deviation of the Gaussian component.
    """
    hist, bin_centers = _build_diff_histogram(data)
    return _em_fit_from_histogram(hist, bin_centers, clip_scale, max_iter, tol)


# -----------------------------------------------------------------------------
# Scaled noise estimate when using a median of the slab as baseline (instead
# of the mean)
# -----------------------------------------------------------------------------

###############################################################################
# Coefficients for the pentadiagonal leave-in median residual noise model.
# sigma_R/sigma = 1 - f(rho1,rho2)/N - g(rho1,rho2)/N^2
# Fit to Monte Carlo over a pentadiagonal (lag-1, lag-2) Gaussian covariance,
# odd N in [5, 61], rho1 in [0.30, 0.60], rho2 in [0.08, 0.12].

_F_COEF = (0.48417, -0.04204, 0.16959, -0.14318, 3.06188)  # const, r1, r1^2, r2, r1*r2
_F_COEF = (0.50373, 0.00559, 0.11647, -0.00362, 1.98957)  # const, r1, r1^2, r2, r1*r2
_G_COEF = (-1.06974, 8.88342, 5.72699)  # const, r1, r2
_G_COEF = (-1.45079, 9.46226, 7.31325)  # const, r1, r2

_N_MIN, _N_MAX = 5, 61
_RHO1_MIN, _RHO1_MAX = 0.30, 0.70
_RHO2_MIN, _RHO2_MAX = 0.08, 0.18


def _pentadiagonal_min_eig(n: int, rho1: float, rho2: float) -> float:
    """Smallest eigenvalue of the unit-variance pentadiagonal Toeplitz
    correlation matrix (lag-1=rho1, lag-2=rho2). <= 0 means the covariance is
    not a valid (positive-definite) noise model."""
    c = np.eye(n)
    i1 = np.arange(n - 1)
    c[i1, i1 + 1] = rho1
    c[i1 + 1, i1] = rho1
    i2 = np.arange(n - 2)
    c[i2, i2 + 2] = rho2
    c[i2 + 2, i2] = rho2
    return float(np.linalg.eigvalsh(c).min())


def leavein_median_noise_scale_penta(
    n: int,
    rho1: float,
    rho2: float,
    check_pd: bool = True,
) -> float:
    """Ratio sigma_R / sigma for a leave-in axial-median background residual,
    with lag-1 AND lag-2 inter-slice noise correlation.

    R = I_center - median(window of n slices, including center). Returns the
    factor scaling per-voxel noise std to residual-image noise std:
    sigma_R = scale * sigma. This is a ratio of STANDARD DEVIATIONS, not
    variances (square it for the variance ratio).

    Empirical fit to Monte Carlo over a pentadiagonal (lag-1, lag-2) Gaussian
    covariance; lag>=3 confirmed negligible (<0.7% impact). Model:

        sigma_R / sigma = 1 - f(rho1,rho2)/n - g(rho1,rho2)/n**2
        f = 0.48417 - 0.04204*rho1 + 0.16959*rho1**2
              - 0.14318*rho2 + 3.06188*rho1*rho2
        g = -1.06974 + 8.88342*rho1 + 5.72699*rho2

    Valid for odd n in [5, 61], rho1 in [0.30, 0.60], rho2 in [0.08, 0.12].
    Not anchored at rho->0; do not extrapolate outside the fitted box. Some
    high-rho1/low-rho2 combinations are not positive-definite (e.g. rho1=0.60
    needs rho2 >= ~0.10); these are rejected when check_pd is True.

    Parameters
    ----------
    n : int
        Axial window size (number of slices), odd, in [5, 61].
    rho1 : float
        Lag-1 inter-slice noise correlation, in [0.30, 0.60].
    rho2 : float
        Lag-2 inter-slice noise correlation, in [0.08, 0.12].
    check_pd : bool
        If True, verify the (rho1, rho2) covariance is positive-definite at
        this n and raise if not.

    Returns
    -------
    float
        sigma_R / sigma (std ratio), in (0, 1].
    """
    try:
        if not (_N_MIN <= n <= _N_MAX):
            raise ValueError(f"n={n} outside fitted range [{_N_MIN}, {_N_MAX}]")
        if n % 2 == 0:
            raise ValueError(f"n={n} must be odd (window centered on a slice)")
        if not (_RHO1_MIN <= rho1 <= _RHO1_MAX):
            _orig = rho1
            rho1 = float(np.clip(rho1, _RHO1_MIN, _RHO1_MAX))
            raise ValueError(
                f"rho1={_orig} outside fitted range [{_RHO1_MIN}, {_RHO1_MAX}]"
            )

        if not (_RHO2_MIN <= rho2 <= _RHO2_MAX):
            _orig = rho2
            rho2 = float(np.clip(rho2, _RHO2_MIN, _RHO2_MAX))
            raise ValueError(
                f"rho2={_orig} outside fitted range [{_RHO2_MIN}, {_RHO2_MAX}]"
            )
        if check_pd and _pentadiagonal_min_eig(n, rho1, rho2) <= 1e-9:
            raise ValueError(
                f"(rho1={rho1}, rho2={rho2}) is not positive-definite at n={n}; "
                f"this correlation pair is unphysical (lag-2 too small for this lag-1)"
            )
    except ValueError as e:
        print(
            f"Invalid parameters: {e}. Clipped values will be used for the calculation, but the result may be inaccurate."
        )

    f0, f1, f2, f3, f4 = _F_COEF
    g0, g1, g2 = _G_COEF
    f = f0 + f1 * rho1 + f2 * rho1 * rho1 + f3 * rho2 + f4 * rho1 * rho2
    g = g0 + g1 * rho1 + g2 * rho2
    return 1.0 - f / n - g / (n * n)


# ----------------------------------------------------------------------------
# Noise and correlation estimation from a block of 5 slices
# ----------------------------------------------------------------------------


def estimate_slice_stats(
    block: NDArray[np.uint8],
    valid_mask: NDArray[np.bool_] | None = None,
    trunc_k: float = 2.0,
) -> dict[str, object]:
    """Estimate per-voxel noise and inter-slice correlation from a 5-slice window.

    Combines two complementary estimators on one slab:
      * Correlation structure (rho1, rho2) and a moment-based sigma come from
        the UNGATED second-difference solve -- unbiased, since voxel selection
        induces spurious correlation.
      * The shoulder-free Gaussian core width (sigma_diff) comes from the
        delta+Gaussian mixture fit on the first differences (trunc_k controls
        its truncated-core window for estimating the Gaussian variance).
      * sigma_corrected combines them: the noise std implied by the clean core
        width and the measured lag-1 correlation.

    Differences from 5 slices S0..S4:
        D1 = S0-S1,  D2 = S1-S2                      (first differences)
        Da = S0-2*S1+S2,  Db = S2-2*S3+S4            (second differences)

    Solve (stationary noise, lag>=3 = 0) for (sigma^2, rho1, rho2):
        Var(D1)     = sigma^2 (2 - 2 rho1)
        Var(Da)     = sigma^2 (6 - 8 rho1 + 2 rho2)
        Cov(Da, Db) = sigma^2 (1 - 4 rho1 + 6 rho2)

    Then:
        sigma_diff      = pooled Gaussian-core std of D1, D2 (mixture fit)
        sigma_corrected = sigma_diff / sqrt(2 (1 - rho1))

    Parameters
    ----------
    block : NDArray[np.uint8], shape (5, H, W)
        Five consecutive transaxial slices (the window for one z location).
    valid_mask : NDArray[np.bool_], shape (H, W), optional
        Boolean mask where True indicates a valid voxel to use. None = use all voxels.
    trunc_k : float
        Truncated-core window (in sigma) for the mixture fit of sigma_diff.

    Returns
    -------
    dict
        Primary:
          'sigma_corrected' : float   per-voxel noise std (sigma_diff + rho1 correction)
          'sigma_diff'      : float   pooled Gaussian-core std of the first differences
          'rho1'            : float   lag-1 inter-slice correlation
          'rho2'            : float   lag-2 inter-slice correlation
        Secondary / diagnostic:
          'sigma_solve'     : float   per-voxel std from the moment solve (incl. shoulders)
          'var_d1'          : float   pooled first-difference variance (survivor)
          'var_d2sq'        : float   pooled second-difference variance (survivor)
          'cov_dd'          : float   once-removed second-difference covariance (survivor)
          'cov_dd_null'     : float   Cov(Da,Db) expected if rho2==0 (given solved rho1)
          'delta_w'         : (float, float)  mixture delta weights (D1, D2)
          'n_survivor'      : int     voxels used in the solve
          'valid'           : bool    False if solve gave sigma^2<=0 or |rho|>=1
    """
    if block.shape[0] != 5:
        raise ValueError(f"expected block of shape (5, H, W), got {block.shape}")

    s = block.astype(np.int16)
    d1 = s[0] - s[1]
    d2 = s[1] - s[2]
    da = s[0] - 2 * s[1] + s[2]
    db = s[2] - 2 * s[3] + s[4]

    if valid_mask is not None:
        if valid_mask.shape != block.shape[1:]:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} does not match slice shape {block.shape[1:]}"
            )
        valid = valid_mask
    else:
        valid = np.ones(block.shape[1:], dtype=bool)

    d1v, d2v = d1[valid].ravel(), d2[valid].ravel()
    dav, dbv = da[valid].ravel(), db[valid].ravel()

    # exact-zero (delta) exclusion, joint across the fields used in the solve
    keep = (d1v != 0) & (d2v != 0) & (dav != 0) & (dbv != 0)
    n_survivor = int(keep.sum())
    if n_survivor < 2:
        raise ValueError("too few survivor voxels for estimation")
    d1g, d2g, dag, dbg = d1v[keep], d2v[keep], dav[keep], dbv[keep]

    # --- ungated moments -> (sigma^2, rho1, rho2) ---
    var_d1 = float(0.5 * (d1g.var(ddof=1) + d2g.var(ddof=1)))
    var_d2sq = float(0.5 * (dag.var(ddof=1) + dbg.var(ddof=1)))
    cov_dd = float(np.cov(dag, dbg, ddof=1)[0, 1])

    M = np.array(
        [
            [2.0, -2.0, 0.0],
            [6.0, -8.0, 2.0],
            [1.0, -4.0, 6.0],
        ]
    )
    a, b, c = np.linalg.solve(M, np.array([var_d1, var_d2sq, cov_dd]))

    valid_solve = bool(a > 0.0)
    sigma2 = float(a)
    rho1 = float(b / a) if valid_solve else float("nan")
    rho2 = float(c / a) if valid_solve else float("nan")
    if valid_solve and (abs(rho1) >= 1.0 or abs(rho2) >= 1.0):
        valid_solve = False
    sigma_solve = float(np.sqrt(sigma2)) if valid_solve else float("nan")
    cov_dd_null = float(sigma2 * (1.0 - 4.0 * rho1)) if valid_solve else float("nan")

    # --- mixture-fit shoulder-free core width of the first differences ---
    w1, _, sig1 = fit_mixture_distribution(
        d1[valid].ravel().astype(np.int16), clip_scale=trunc_k
    )
    w2, _, sig2 = fit_mixture_distribution(
        d2[valid].ravel().astype(np.int16), clip_scale=trunc_k
    )
    sigma_diff = float(np.sqrt(0.5 * (sig1**2 + sig2**2)))

    # --- corrected per-voxel noise std ---
    if valid_solve and rho1 < 1.0:
        sigma_corrected = float(sigma_diff / np.sqrt(2.0 * (1.0 - rho1)))
    else:
        sigma_corrected = float("nan")

    return {
        # primary
        "sigma_corrected": sigma_corrected,
        "sigma_diff": sigma_diff,
        "rho1": rho1,
        "rho2": rho2,
        # secondary / diagnostic
        "sigma_solve": sigma_solve,
        "var_d1": var_d1,
        "var_d2sq": var_d2sq,
        "cov_dd": cov_dd,
        "cov_dd_null": cov_dd_null,
        "delta_w": (float(w1), float(w2)),
        "n_survivor": n_survivor,
        "valid": valid_solve,
    }


def _estimate_volume_slice_stat(
    volume: NDArray[np.uint8],
    slice_level: int,
    trunc_k: float,
    metal_threshold: int | None,
    min_area: int,
    grayscale_metal_margin: int,
) -> tuple[int, dict[str, object]]:
    block = volume[slice_level - 2 : slice_level + 3, :, :]
    if metal_threshold is not None:
        metal_mask = cast(
            NDArray,
            extract_metal_mask(
                block.max(axis=0),
                metal_threshold,
                min_area=min_area,
                margin=grayscale_metal_margin,
            ),
        )
        support_mask = identify_cylindrical_support(metal_mask)
        valid_mask = ~metal_mask & support_mask
    else:
        valid_mask = None

    return slice_level, estimate_slice_stats(
        block, valid_mask=valid_mask, trunc_k=trunc_k
    )


def estimate_volume_slice_stats(
    volume: NDArray[np.uint8],
    slice_levels: Sequence[int],
    metal_threshold: int,
    min_area: int,
    grayscale_metal_margin: int,
    trunc_k: float = 2.5,
    use_parallel: bool = True,
) -> dict[str, Any]:
    """
    Estimate the slice statistics over the volume at the given slice levels. Each slice statistics are computed over a 5-slice block centered at the slice level, using the estimate_slice_stats function. The metal mask is computed for each block using the maximum intensity projection and the given metal threshold.

    args:
    volume
        3D array of the CT volume, with shape (n_slices, H, W).
    slice_levels
        Sequence of slice levels to evaluate.
    trunc_k
        Truncation factor for robust statistics.
    metal_threshold
        Threshold for metal mask extraction.
    min_area
        Minimum area for metal mask extraction.
    grayscale_metal_margin
        Margin for grayscale metal mask extraction.
    use_parallel
        If True, compute per-slice statistics concurrently. If False, use the
        direct serial route.

    Returns:
        dict[str, Any]: a dictionary of statistics for each slice level and summary values
    """
    z_min, z_max = 2, volume.shape[0] - 3

    slice_indices: list[int] = []
    slices_seen: set[int] = set()
    for s in slice_levels:
        s_ = int(np.clip(s, z_min, z_max))
        if s_ in slices_seen:
            continue
        slice_indices.append(s_)
        slices_seen.add(s_)

    if use_parallel and len(slice_indices) > 1:
        with ThreadPoolExecutor() as executor:
            stats_items = list(
                executor.map(
                    lambda s_: _estimate_volume_slice_stat(
                        volume,
                        slice_level=s_,
                        trunc_k=trunc_k,
                        metal_threshold=metal_threshold,
                        min_area=min_area,
                        grayscale_metal_margin=grayscale_metal_margin,
                    ),
                    slice_indices,
                )
            )
    else:
        stats_items = [
            _estimate_volume_slice_stat(
                volume,
                slice_level=s_,
                trunc_k=trunc_k,
                metal_threshold=metal_threshold,
                min_area=min_area,
                grayscale_metal_margin=grayscale_metal_margin,
            )
            for s_ in slice_indices
        ]

    slice_stats: dict[int, dict[str, object]] = dict(stats_items)

    results: dict[str, Any] = {}
    results["slice_stats"] = slice_stats

    sigma_corrected_vals: list[float] = [
        v["sigma_corrected"] for v in slice_stats.values() if v["valid"]
    ]
    results["sigma_corrected_min"] = (
        float(np.min(sigma_corrected_vals)) if sigma_corrected_vals else float("nan")
    )
    results["sigma_corrected_max"] = (
        float(np.max(sigma_corrected_vals)) if sigma_corrected_vals else float("nan")
    )
    results["sigma_corrected_median"] = (
        float(np.median(sigma_corrected_vals)) if sigma_corrected_vals else float("nan")
    )

    sigma_diff_vals: list[float] = [
        v["sigma_diff"] for v in slice_stats.values() if v["valid"]
    ]
    results["sigma_diff_min"] = (
        float(np.min(sigma_diff_vals)) if sigma_diff_vals else float("nan")
    )
    results["sigma_diff_max"] = (
        float(np.max(sigma_diff_vals)) if sigma_diff_vals else float("nan")
    )
    results["sigma_diff_median"] = (
        float(np.median(sigma_diff_vals)) if sigma_diff_vals else float("nan")
    )

    rho1_vals: list[float] = [
        float(v["rho1"]) for v in slice_stats.values() if v["valid"]
    ]
    results["rho1_min"] = float(np.min(rho1_vals)) if rho1_vals else float("nan")
    results["rho1_max"] = float(np.max(rho1_vals)) if rho1_vals else float("nan")
    results["rho1_median"] = float(np.median(rho1_vals)) if rho1_vals else float("nan")

    rho2_vals: list[float] = [
        float(v["rho2"]) for v in slice_stats.values() if v["valid"]
    ]
    results["rho2_min"] = float(np.min(rho2_vals)) if rho2_vals else float("nan")
    results["rho2_max"] = float(np.max(rho2_vals)) if rho2_vals else float("nan")
    results["rho2_median"] = float(np.median(rho2_vals)) if rho2_vals else float("nan")

    return results


# -----------------------------------------------------------------------------
# Noise Modeling
# -----------------------------------------------------------------------------

"""
## Noise Correlation filters

Each filter below turns unit-variance white noise into a unit-variance stationary process with the requested lag correlation(s), applied along one axis at a time (`axis=2`/`1`/`0` for x/y/z). Applying separate 1-D filters along each axis makes the resulting 3-D covariance a Kronecker product of the three 1-D covariances — exact along each axis individually (diagonal neighbours in x-y come out to `rhoxy**2`, not `rhoxy`), which is the sense in which this is a "first order approximation" to a fully isotropic field.

- **`ar1_filter`** — AR(1) recursion, used for the lateral (x, y) correlation `rhoxy`. Lag-*k* correlation decays as `rhoxy**k`.
- **`ar2_filter`** — AR(2) recursion, used for the z-direction correlation. The two AR coefficients are solved from the requested lag-1/lag-2 correlations `rho1`/`rho2` via the Yule-Walker equations, so both are matched exactly (rather than only the lag-1 correlation with lag-2 falling out as `rho1**2`).

Both filters seed their recursion (via `zi`) from the process's own stationary distribution, rather than the default zero initial state `scipy.signal.lfilter` would use — otherwise the first one or two slices along each axis would start with a variance deficit and only reach the target variance/correlation after a short transient."""


def ar1_filter(x: NDArray, rho: float, axis: int) -> NDArray:
    """First-order (AR(1)) recursive filter along `axis`.

    Turns unit-variance white noise into a unit-variance AR(1) series with
    lag-1 correlation `rho`, seeded from the process's own stationary
    distribution so there is no start-up transient at the edges.
    """
    if rho == 0.0:
        return x

    b = [np.sqrt(1.0 - rho**2)]
    a = [1.0, -rho]

    seed_shape = list(x.shape)
    seed_shape[axis] = 1
    y_m1 = np.random.normal(size=seed_shape)  # y[-1] ~ N(0, 1)
    zi = rho * y_m1

    y, _ = lfilter(b, a, x, axis=axis, zi=zi)
    return y


def ar2_filter(x: NDArray, rho1: float, rho2: float, axis: int) -> NDArray:
    """Second-order (AR(2)) recursive filter along `axis`.

    Turns unit-variance white noise into a unit-variance AR(2) series with
    lag-1/lag-2 correlations `rho1`/`rho2`, via the Yule-Walker equations.
    Seeded the same way as `ar1_filter`, from the joint stationary
    distribution of (y[-1], y[-2]), so there is no start-up transient.
    """
    if rho1 == 0.0 and rho2 == 0.0:
        return x

    # Yule-Walker: solve for the AR coefficients that reproduce rho1, rho2.
    phi2 = (rho2 - rho1**2) / (1.0 - rho1**2)
    phi1 = rho1 * (1.0 - phi2)

    # Driving-noise variance that keeps the output at unit variance.
    sigma_e2 = 1.0 - phi1 * rho1 - phi2 * rho2
    if sigma_e2 <= 0.0:
        raise ValueError(
            f"rho1={rho1}, rho2={rho2} is not a valid AR(2) autocorrelation pair"
        )

    b = [np.sqrt(sigma_e2)]
    a = [1.0, -phi1, -phi2]

    seed_shape = list(x.shape)
    seed_shape[axis] = 1
    y_m2 = np.random.normal(size=seed_shape)  # y[-2] ~ N(0, 1)
    y_m1 = rho1 * y_m2 + np.sqrt(1.0 - rho1**2) * np.random.normal(
        size=seed_shape
    )  # y[-1]
    z0 = phi1 * y_m1 + phi2 * y_m2
    z1 = phi2 * y_m1
    zi = np.concatenate([z0, z1], axis=axis)

    y, _ = lfilter(b, a, x, axis=axis, zi=zi)
    return y
