#!/usr/bin/env python3
import random
import sys
import tomllib
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pandas as pd
import typer
from numpy.typing import NDArray
from screeninfo import get_monitors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
sys.path.insert(0, str(PROJECT_ROOT))

import experiment_db as db  # noqa: E402
from cfg_data import AnalysisBase  # noqa: E402


class GlobalData(AnalysisBase):
    def __init__(self, config_path: Path, vol_index: int):
        super().__init__(config_path, vol_index=vol_index, exp_index=0)

    slab_thickness: int = 11


global_data: GlobalData

app = typer.Typer()

INTERP = cv2.INTER_LINEAR  # interpolation method for zooming

SIDE_VIEW_Z_STRIDE = 4  # sample every Nth z-slice for the side view, trading depth resolution for load speed
RADIAL_PLANE_OFFSETS: tuple[int, ...] = (-1, 0, 1)
LABEL_RGB_COLORS = {
    -1: (255, 215, 0),
    0: (240, 0, 0),
    1: (20, 251, 0),
    2: (100, 240, 84),
    3: (156, 225, 136),
    4: (255, 170, 0),
    10: (215, 0, 215),
    11: (0, 210, 192),
}


def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return b, g, r


def _circle_color_for_label(label: int) -> tuple[int, int, int]:
    return _rgb_to_bgr(LABEL_RGB_COLORS.get(label, LABEL_RGB_COLORS[-1]))


def _read_cells(cfg_path: Path) -> list[dict]:
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg["cell"]


PROPAGATE_LABELS: frozenset[int] = frozenset({0, 1, 2, 3})
DEFAULT_OVERWRITE_LABELS: frozenset[int] = frozenset({-1, 4, 10})


def _set_label_and_propagate(
    con: duckdb.DuckDBPyConnection,
    full_df: pd.DataFrame,
    candidate_id: int,
    particle_label: int,
    force_overwrite_reviewed: bool = False,
) -> tuple[bool, int]:
    """Set one candidate's label (in memory AND in the DB) and propagate within
    its live-linked physical cluster (db.linked_candidate_ids -- shared grown
    voxels, computed on demand; no precomputed clusters.csv/instance_to_cluster.csv
    anymore, see experiment_db module docstring).

    Only definitive labels (0/1/2/3) are propagated. By default, propagation
    overwrites only {-1,4,10}; set force_overwrite_reviewed=True to overwrite
    all labels in the cluster.
    """
    previous_label = int(full_df.at[candidate_id, "particle"])
    if previous_label == particle_label:
        return False, 0

    full_df.at[candidate_id, "particle"] = particle_label
    db.set_particle_label(con, candidate_id, particle_label)

    if particle_label not in PROPAGATE_LABELS:
        return True, 0

    other_ids = [
        cid for cid in db.linked_candidate_ids(con, candidate_id) if cid != candidate_id
    ]
    if not other_ids:
        return True, 0

    if force_overwrite_reviewed:
        target_ids = other_ids
    else:
        target_ids = [
            cid
            for cid in other_ids
            if int(full_df.at[cid, "particle"]) in DEFAULT_OVERWRITE_LABELS
        ]

    if target_ids:
        full_df.loc[target_ids, "particle"] = particle_label
        db.set_particle_labels(con, target_ids, particle_label)

    return True, len(target_ids)


def _render_info_panel(row: pd.Series) -> NDArray:
    line_h = 22
    height = 20 + len(row) * line_h + 10
    panel = np.full((height, 380, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    y = 20
    for col, val in row.items():
        if isinstance(val, float):
            text = f"{col}: {val:.3f}"
        else:
            text = f"{col}: {val}"
        cv2.putText(
            panel, text, (10, y), font, scale, (200, 200, 200), thick, cv2.LINE_AA
        )
        y += line_h
    return panel


def _render_message_panel(lines: list[str]) -> NDArray:
    line_h = 30
    height = 40 + len(lines) * line_h
    panel = np.full((height, 640, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 40
    for line in lines:
        cv2.putText(panel, line, (20, y), font, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
        y += line_h
    return panel


def _normalize_image(img: NDArray) -> NDArray:
    """Normalize a 2D frame to 8-bit grayscale based on the grayscale range of the volume."""
    gray_landmarks = global_data.grayscale_landmarks
    air_grayvalue = int(gray_landmarks["air_grayvalue"])
    max_grayvalue = int(gray_landmarks["max_grayvalue"])
    metal_grayvalue = int(global_data.metal_threshold)
    upper_limit = max_grayvalue
    lower_limit = air_grayvalue
    contrast = 1.2
    brightness = 10
    img = 255 * (
        contrast * (img.astype(float) - lower_limit) / (upper_limit - lower_limit)
        + brightness / 255.0
    )
    img = img.clip(0, 255).astype("uint8")
    return img


def _create_mip(
    row: pd.Series,
    type: str = "xz",
    circle_color: tuple[int, int, int] = (0, 255, 0),
    mag_factor: float = 1.0,
) -> NDArray:
    """Create a maximum intensity projection (MIP) of the volume data."""
    if global_data.vol is None:
        raise ValueError("Volume data is not loaded.")
    _extent = 50
    y_extent = max(1, round(row["y_extent"]))
    x_extent = max(1, round(row["x_extent"]))
    x_pad = max(_extent, x_extent + 10)
    y_pad = max(_extent, y_extent + 10)
    z_pad = max(_extent, int(row["z_extent"]) + 10)
    cx = round(row["peak_x"])
    cy = round(row["peak_y"])
    cz = round(row["peak_z"])
    z_lo = max(0, int(cz) - z_pad)
    z_hi = min(global_data.vol.shape[0], int(cz) + z_pad)
    x_lo = max(0, int(cx - x_pad))
    x_hi = min(global_data.vol.shape[2], int(cx + x_pad))
    y_lo = max(0, int(cy - y_pad))
    y_hi = min(global_data.vol.shape[1], int(cy + y_pad))

    y_extent, x_extent = 2, 2
    mip = None
    if type == "xz":
        mip = global_data.vol[
            z_lo:z_hi, cy - (y_extent) // 2 : cy + (y_extent) // 2 + 1, x_lo:x_hi
        ].max(axis=1)
        mip = cv2.cvtColor(_normalize_image(mip), cv2.COLOR_GRAY2BGR)
        cv2.circle(mip, (cx - x_lo, cz - z_lo), 10, circle_color, thickness=1)
        if mag_factor != 1.0:
            mip = cv2.resize(
                mip,
                None,
                fx=mag_factor,
                fy=mag_factor,
                interpolation=INTERP,
            )
        return mip
    elif type == "yz":
        mip = global_data.vol[
            z_lo:z_hi, y_lo:y_hi, cx - (x_extent) // 2 : cx + (x_extent) // 2 + 1
        ].max(axis=2)
        mip = cv2.cvtColor(_normalize_image(mip), cv2.COLOR_GRAY2BGR)
        cv2.circle(mip, (cy - y_lo, cz - z_lo), 10, circle_color, thickness=1)
        if mag_factor != 1.0:
            mip = cv2.resize(
                mip,
                None,
                fx=mag_factor,
                fy=mag_factor,
                interpolation=INTERP,
            )
        return mip
    elif type == "rz":
        # Radial cut-plane through the particle peak and image center, then max
        # over three parallel neighboring planes for a more stable cross-section.
        z_block = global_data.vol[z_lo:z_hi, :, :]
        h, w = global_data.vol.shape[1], global_data.vol.shape[2]
        mid_x = (w - 1) / 2.0
        mid_y = (h - 1) / 2.0

        vx = float(cx) - mid_x
        vy = float(cy) - mid_y
        norm = float(np.hypot(vx, vy))
        if norm < 1e-6:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = vx / norm, vy / norm

        # Perpendicular unit vector for neighboring parallel planes.
        px, py = -uy, ux

        t_pad = max(x_pad, y_pad)
        t = np.arange(-t_pad, t_pad + 1, dtype=np.float64)

        y_idx_all: list[np.ndarray] = []
        x_idx_all: list[np.ndarray] = []
        for off in RADIAL_PLANE_OFFSETS:
            x_coords = cx + ux * t + px * off
            y_coords = cy + uy * t + py * off
            x_idx_all.append(np.clip(np.rint(x_coords).astype(np.int32), 0, w - 1))
            y_idx_all.append(np.clip(np.rint(y_coords).astype(np.int32), 0, h - 1))

        x_idx = np.stack(x_idx_all, axis=0)  # (n_offsets, n_t)
        y_idx = np.stack(y_idx_all, axis=0)  # (n_offsets, n_t)

        sampled = z_block[:, y_idx, x_idx]  # (n_z, n_offsets, n_t)
        mip = sampled.max(axis=1)  # (n_z, n_t)
        mip = cv2.cvtColor(_normalize_image(mip), cv2.COLOR_GRAY2BGR)

        peak_t = t_pad
        peak_z = cz - z_lo
        if 0 <= peak_z < mip.shape[0]:
            cv2.circle(mip, (peak_t, peak_z), 10, circle_color, thickness=1)
        if mag_factor != 1.0:
            mip = cv2.resize(
                mip,
                None,
                fx=mag_factor,
                fy=mag_factor,
                interpolation=INTERP,
            )
        return mip
    else:
        raise ValueError(f"Unsupported MIP type: {type}")


def _create_named_window(win_namesP: list[str]):
    for win_name in win_namesP:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


def _create_side_frame_base(cache_dir: Path, stem: str, img_h: int):
    if global_data.vol is None:
        raise ValueError("Volume data is not loaded.")
    z_depth = global_data.vol.shape[0]
    mid_x = global_data.vol.shape[2] // 2
    z_stride = SIDE_VIEW_Z_STRIDE
    sampled_z = (z_depth + z_stride - 1) // z_stride

    # A full-height, full-depth column touches nearly every page of the
    # multi-GB file regardless of read order (row size < page size, so every
    # row must be paged in) — reading it cold is disk-bound at ~13s for a
    # 9GB volume no matter how the reads are ordered. Only reading every
    # z_stride-th slice actually cuts the bytes touched (and thus the time)
    # by roughly that factor; cache the result so repeat visits to the same
    # cell skip the scan entirely.
    side_cache_path = cache_dir / f"{stem}_side_x{mid_x}_s{z_stride}.npy"
    side_slice = None
    if side_cache_path.exists():
        cached = np.load(side_cache_path)
        if cached.shape == (sampled_z, img_h) and cached.dtype == global_data.vol.dtype:
            side_slice = cached

    if side_slice is None:
        side_slice = np.empty((sampled_z, img_h), dtype=global_data.vol.dtype)
        for i, z in enumerate(range(0, z_depth, z_stride)):
            side_slice[i] = np.asarray(global_data.vol[z, :, mid_x])
        np.save(side_cache_path, side_slice)

    side_frame_base = cv2.cvtColor(_normalize_image(side_slice), cv2.COLOR_GRAY2BGR)
    return side_frame_base


def _display_loop(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    cache_dir: Path,
    stem: str,
    experiment: int,
    img_h: int,
    img_w: int,
    auto_zoom: bool = False,
    force_overwrite_reviewed: bool = False,
) -> bool:
    """Review one cell. Returns True when the cell is complete, False on user quit."""
    if global_data.vol is None:
        raise ValueError("Volume data is not loaded.")
    monitors = get_monitors()
    win_offset1 = 900
    win_offset2 = 1200
    if len(monitors) > 1:
        win_offset1 = monitors[0].width // 2
        win_offset2 = monitors[1].x + 100

    win = "vol_review"
    win_mip = "vol_review_mip_xy"
    win_mip_xz = "vol_review_mip_xz"
    win_mip_yz = "vol_review_mip_yz"
    win_mip_rz = "vol_review_mip_rz"
    win_info = "candidate_info"
    win_side = "vol_review_side"
    _create_named_window(
        [win, win_mip, win_info, win_side, win_mip_xz, win_mip_yz, win_mip_rz]
    )
    z_depth = global_data.vol.shape[0]
    z_stride = SIDE_VIEW_Z_STRIDE

    side_frame_base = _create_side_frame_base(cache_dir, stem, img_h)

    default_zoom = 9 if auto_zoom else 1.0
    idx = 0
    zoom = default_zoom
    cell_complete = False
    user_quit = False
    first_time = True
    use_peak = True
    while True:
        row = df.iloc[idx]
        if use_peak:
            z0 = round(row["peak_z"])
            cx, cy = round(row["peak_x"]), round(row["peak_y"])
        else:
            z0 = round(row["centroid_z"])
            cx, cy = round(row["centroid_x"]), round(row["centroid_y"])
        dist = (
            (row["centroid_x"] - img_w / 2) ** 2 + (row["centroid_y"] - img_h / 2) ** 2
        ) ** 0.5

        circle_color = _circle_color_for_label(int(row["particle"]))

        frame = np.max(global_data.vol[z0 - 1 : z0 + 2, :, :], axis=0)
        frame = cv2.cvtColor(_normalize_image(frame), cv2.COLOR_GRAY2BGR)
        cv2.circle(frame, (cx, cy), 10, circle_color, thickness=1)

        h, w = frame.shape[:2]
        if zoom > 1.0:
            crop_h = max(1, int(h / zoom))
            crop_w = max(1, int(w / zoom))
            x1 = max(0, min(cx - crop_w // 2, w - crop_w))
            y1 = max(0, min(cy - crop_h // 2, h - crop_h))

            def _apply_zoom(img: NDArray) -> NDArray:
                return cv2.resize(
                    img[y1 : y1 + crop_h, x1 : x1 + crop_w],
                    (w, h),
                    interpolation=INTERP,
                )

            frame = _apply_zoom(frame)

        def _fmt_prob(value: object) -> str:
            if pd.isna(value):
                return "nan"
            if isinstance(value, (int, float, np.integer, np.floating)):
                return f"{float(value):.3f}"
            if isinstance(value, str):
                try:
                    return f"{float(value):.3f}"
                except ValueError:
                    return "nan"
            return "nan"

        ml_prob = row.get("ml_prob", np.nan)
        ml_prob_particle_max = row.get("ml_prob_particle_max", np.nan)
        prob_title = f"P:{_fmt_prob(ml_prob)}/{_fmt_prob(ml_prob_particle_max)}"
        title_base = f"[{idx + 1}/{len(df)}]  z={z0}  ({cx}, {cy})  zoom={zoom:.1f}x"
        title_base_with_prob = f"{title_base}  {prob_title}"

        z_pad = global_data.slab_thickness // 2
        z_lo = max(0, int(z0 - z_pad))
        z_hi = min(z_depth, int(z0) + z_pad + 1)
        if global_data.vol is None:
            raise ValueError("Volume data is not loaded.")

        # mip = global_data.vol[z_lo:z_hi, :, :].max(axis=0)
        med_ = np.median(global_data.vol[z_lo:z_hi, :, :], axis=0)
        dif_ = global_data.vol[int(row["peak_z"]), :, :].astype(np.int16) - med_.astype(
            np.int16
        )

        def _create_residual_proxy(mip: NDArray, area_min=1) -> NDArray:
            sigma_threshold_high = 4.0
            sigma_threshold_low = 2.8

            pk_y, pk_x = int(row["peak_y"]), int(row["peak_x"])
            _sliced = mip[pk_y - 1 : pk_y + 2, pk_x - 1 : pk_x + 2]
            peak_val = np.partition(_sliced.ravel(), -1)[-2:]
            peak_val = float(peak_val.mean())

            # Formulate the upper limit to the noise sigma
            sigma_proxy = peak_val / 4.2

            # print(f"\npeak_val: {peak_val}, sigma_proxy: {sigma_proxy}\n\n")
            sigma_proxy = min(2.7, sigma_proxy)

            seed_mask = mip > sigma_threshold_high * sigma_proxy
            grow_mask = mip > sigma_threshold_low * sigma_proxy
            T = area_min
            _, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(
                seed_mask.astype(np.uint8), connectivity=8
            )
            large_seeds = set(
                int(i + 1) for i in np.where(seed_stats[1:, cv2.CC_STAT_AREA] >= T)[0]
            )
            large_seed_mask = np.isin(seed_labels, list(large_seeds))

            _, grow_labels = cv2.connectedComponents(
                grow_mask.astype(np.uint8), connectivity=8
            )
            seeded_components = set(grow_labels[large_seed_mask].tolist()) - {0}
            mask = np.isin(grow_labels, list(seeded_components))

            mip = mip * mask
            return mip

        mip = dif_
        # mip = _create_residual_proxy(dif_)

        c = 1
        _min, max_ = mip.min() - c, mip.max() + c
        mip = (mip - _min) / (max_ - _min + 1e-8) * 255
        mip = mip.astype(np.uint8)
        mip_frame = cv2.cvtColor((mip), cv2.COLOR_GRAY2BGR)
        # mip_frame = cv2.cvtColor(_normalize_image(mip), cv2.COLOR_GRAY2BGR)
        cv2.circle(mip_frame, (cx, cy), 10, circle_color, thickness=1)

        cv2.line(
            mip_frame,
            (max(0, cx), max(0, cy - 50)),
            (min(w - 1, cx), min(h - 1, cy - 10)),
            (200, 64, 64),
            thickness=1,
        )
        cv2.line(
            mip_frame,
            (max(0, cx), max(0, cy + 10)),
            (min(w - 1, cx), min(h - 1, cy + 50)),
            (200, 64, 64),
            thickness=1,
        )

        cv2.line(
            mip_frame,
            (max(0, cx - 50), max(0, cy)),
            (min(w - 1, cx - 10), min(h - 1, cy)),
            (200, 64, 64),
            thickness=1,
        )
        cv2.line(
            mip_frame,
            (max(0, cx + 10), max(0, cy)),
            (min(w - 1, cx + 50), min(h - 1, cy)),
            (200, 64, 64),
            thickness=1,
        )

        if zoom > 1.0:
            mip_frame = _apply_zoom(mip_frame)
        cv2.setWindowTitle(win_mip, f"max projection  {stem}  {title_base_with_prob}")
        cv2.imshow(win_mip, mip_frame)
        if first_time:
            cv2.moveWindow(
                win_mip, win_offset1, 10
            )  # move to top-left so it doesn't cover the main view

        side_frame = side_frame_base.copy()
        side_w = side_frame.shape[1]
        side_row = round(z0 / z_stride)
        cv2.line(
            side_frame, (0, side_row), (side_w, side_row), (230, 216, 173), 3
        )  # light blue
        cv2.setWindowTitle(win_side, f"side view  {stem}  z={z0}")
        cv2.imshow(win_side, side_frame)
        if first_time:
            cv2.moveWindow(
                win_side, win_offset2, 10
            )  # move to top-left so it doesn't cover the main view

        info_row = pd.concat([row, pd.Series({"distance_to_center": dist})])
        cv2.imshow(win_info, _render_info_panel(info_row))
        cv2.setWindowTitle(win, f"{stem}  {title_base_with_prob}")
        cv2.imshow(win, frame)

        _mag = 6
        mip_xz_frame = _create_mip(
            row, type="xz", circle_color=circle_color, mag_factor=_mag
        )
        mip_yz_frame = _create_mip(
            row, type="yz", circle_color=circle_color, mag_factor=_mag
        )
        mip_rz_frame = _create_mip(
            row, type="rz", circle_color=circle_color, mag_factor=_mag
        )
        cv2.setWindowTitle(win_mip_xz, f"max projection XZ  {stem}  {title_base}")
        cv2.imshow(win_mip_xz, mip_xz_frame)
        if first_time:
            cv2.moveWindow(
                win_mip_xz, win_offset1, 10
            )  # move to top-left so it doesn't cover the main view
        cv2.setWindowTitle(win_mip_yz, f"max projection YZ  {stem}  {title_base}")
        cv2.imshow(win_mip_yz, mip_yz_frame)
        if first_time:
            cv2.moveWindow(
                win_mip_yz, win_offset1, 660
            )  # move to top-left so it doesn't cover the main view

        cv2.setWindowTitle(
            win_mip_rz, f"max projection radial(Z)  {stem}  {title_base}"
        )
        cv2.imshow(win_mip_rz, mip_rz_frame)
        first_time = False

        key = cv2.waitKey(0)
        if key == 3:  # right arrow (macOS)
            if idx == len(df) - 1:
                cell_complete = True
                break
            idx += 1
            zoom = default_zoom
        elif key == 2:  # left arrow (macOS)
            idx = max(idx - 1, 0)
            zoom = default_zoom
        elif key in (ord("+"), ord("=")):
            zoom *= 1.5
        elif key == ord("-"):
            zoom = max(1.0, zoom / 1.5)
        elif ord("0") <= key <= ord("4") or key == ord("u"):
            particle_label = key - ord("0") if key != ord("u") else -1
            label_changed, n_propagated = _set_label_and_propagate(
                con,
                full_df,
                row.name,
                particle_label,
                force_overwrite_reviewed=force_overwrite_reviewed,
            )
            if label_changed:
                affected_indices = df.index.intersection(full_df.index)
                df.loc[affected_indices, "particle"] = full_df.loc[
                    affected_indices, "particle"
                ]
                typer.echo(
                    f"  [{idx + 1}] labeled particle={particle_label} "
                    f"and propagated to {n_propagated} nearby candidates"
                )
            else:
                typer.echo(f"  [{idx + 1}] particle already labeled {particle_label}")
        elif key == ord("n"):
            typer.echo(f"  [{idx + 1}] moving to next volume")
            cell_complete = True
            break
        elif key == ord("q") or key == 27:  # q or Esc
            user_quit = True
            break

    cv2.destroyAllWindows()

    return cell_complete and not user_quit


def _wait_for_final_q() -> None:
    win = "vol_review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(
        win,
        _render_message_panel(
            [
                "Finished reviewing all cells.",
                "Press q to close.",
            ]
        ),
    )
    while True:
        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyAllWindows()


@app.command()
def main(
    cell_index: int | None = typer.Argument(
        None,
        help="Index into the config's [[cell]] array to start reviewing; "
        "if omitted, cells are presented in random order",
    ),
    experiment: int = typer.Option(
        ..., help="Experiment number (config's [[experiment]] experiment_number)"
    ),
    min_voxels: int = typer.Option(3, help="Minimum n_voxels threshold (inclusive)"),
    max_voxels: int | None = typer.Option(
        None, help="Maximum n_voxels threshold (inclusive); no upper limit if unset"
    ),
    min_seed: int | None = typer.Option(
        None, help="Minimum n_seed threshold (inclusive); no lower limit if unset"
    ),
    min_snr_peak: float | None = typer.Option(
        None, help="Minimum snr_peak threshold (inclusive); no lower limit if unset"
    ),
    az: bool = typer.Option(False, "--az", help="Auto-zoom to centroid on navigation"),
    skip_classified: bool = typer.Option(
        False,
        "--skip-classified",
        help="Skip candidates that already have a particle classification (particle != -1)",
    ),
    show_positives: bool = typer.Option(
        False,
        "--sp",
        help="Show only positive particle features (particle > 0)",
    ),
    show_only_labeled: bool = typer.Option(
        False,
        "--sol",
        help="Show all labeled candidates (particle != -1), ignoring other filters",
    ),
    show_class: list[int] | None = typer.Option(
        None,
        "--show-class",
        help="Show only candidates with particle label in this set "
        "(repeatable, e.g. --show-class 1 --show-class 2 --show-class 3)",
    ),
    sort_by_score: bool = typer.Option(
        False,
        "--sort-by-score",
        help="Sort candidates by ml_prob descending (highest model-confidence "
        "first) instead of centroid_z. Candidates with no score (outside "
        "slab_thickness 21/23/25, the model's training window) sort last.",
    ),
    min_score: float | None = typer.Option(
        None, help="Minimum ml_prob threshold (inclusive); excludes unscored candidates"
    ),
    max_score: float | None = typer.Option(
        None, help="Maximum ml_prob threshold (inclusive); excludes unscored candidates"
    ),
    force_overwrite_reviewed: bool = typer.Option(
        False,
        "--force-overwrite-reviewed",
        help="When propagating definitive labels (0/1/2/3), overwrite existing "
        "reviewed labels too. Default only overwrites -1/4/10.",
    ),
) -> None:
    global global_data

    cells = _read_cells(CONFIG_PATH)
    if cell_index is not None:
        if not 0 <= cell_index < len(cells):
            typer.echo(f"Index {cell_index} out of range (0..{len(cells) - 1})")
            raise typer.Exit(code=1)
        cell_order = list(range(cell_index, len(cells)))
    else:
        cell_order = list(range(len(cells)))
        random.shuffle(cell_order)
        typer.echo(
            f"No cell index given; reviewing {len(cell_order)} cells in random order"
        )

    cache_dir = PROJECT_ROOT / "Data" / "experiments"
    con = db.connect()
    typer.echo(f"Candidates: {db.DB_PATH}")

    full_df = db.load_candidates_dataframe(con)
    if full_df.empty:
        typer.echo(f"No candidates recorded in {db.DB_PATH} yet.")
        raise typer.Exit(code=1)
    print(
        f"Total candidates loaded: {len(full_df)}, total classified: {len(full_df[full_df['particle'].isin([0, 1, 2, 3])])}, total TPs: {len(full_df[full_df['particle'].isin([1, 2, 3])])}"
    )
    voxel_range = (
        f"{min_voxels} <= n_voxels <= {max_voxels}"
        if max_voxels is not None
        else f"n_voxels >= {min_voxels}"
    )
    seed_range = f" and n_seed >= {min_seed}" if min_seed is not None else ""
    snr_peak_range = (
        f" and snr_peak >= {min_snr_peak}" if min_snr_peak is not None else ""
    )

    for current_cell_index in cell_order:
        cell = cells[current_cell_index]
        cell_name = cell["name"]
        z_min = float(cell["z_min"])

        global_data = GlobalData(CONFIG_PATH, vol_index=current_cell_index)

        if global_data.vol is not None:
            _, img_h, img_w = global_data.vol.shape
        else:
            raise ValueError("Volume data is not loaded.")

        typer.echo(
            f"\nVolume : {cell_name}.raw  "
            f"(cell {current_cell_index}/{len(cells) - 1}, shape: {global_data.vol.shape})"
        )
        mask = (
            full_df["volume_name"].eq(cell_name)
            & full_df["experiment_number"].eq(experiment)
            & (full_df["centroid_z"] >= z_min)
        )

        # region    ----- TESTING -----
        # Temp value while we re-classify
        # mask &= full_df["peak_z"].ge(1400)
        # mask &= full_df["peak_z"].le(1425)
        # mask &= full_df["volume_name"].eq("M50L-03")
        # endregion ----- TESTING -----

        if show_class:
            mask &= full_df["particle"].isin(show_class)
        elif show_only_labeled:
            mask &= full_df["particle"].ne(-1)
        elif show_positives:
            mask &= full_df["particle"].isin([1, 2, 3])
        elif skip_classified:
            mask &= full_df["particle"].isin([-1, 10])
            mask &= full_df["n_voxels"] >= min_voxels

        # additional morphological filters
        if max_voxels is not None:
            mask &= full_df["n_voxels"] <= max_voxels
        if min_seed is not None:
            mask &= full_df["n_seed"] >= min_seed

        # Additional signal features
        if min_snr_peak is not None:
            mask &= full_df["snr_peak"] >= min_snr_peak

        # Model-score filters (ml_prob only populated for slab_thickness
        # 21/23/25 -- see ml_classifier/score_backlog.py)
        if min_score is not None:
            mask &= full_df["ml_prob"] >= min_score
        if max_score is not None:
            mask &= full_df["ml_prob"] <= max_score

        if sort_by_score:
            df = full_df[mask].sort_values(
                "ml_prob", ascending=False, na_position="last"
            )
        else:
            df = full_df[mask].sort_values("centroid_z")
        if df.empty:
            continue
        global_data.slab_thickness = df["slab_thickness"].iloc[0]
        if show_only_labeled:
            typer.echo(f"{len(df)} labeled candidates with centroid_z >= {z_min}")
        else:
            typer.echo(
                f"{len(df)} candidates with {voxel_range}{seed_range}{snr_peak_range} "
                f"and centroid_z >= {z_min}"
            )

        if df.empty:
            typer.echo("No candidates left to review; moving to next cell.")
            continue

        cell_complete = _display_loop(
            con,
            df,
            full_df,
            cache_dir,
            cell_name,
            experiment,
            img_h,
            img_w,
            auto_zoom=az,
            force_overwrite_reviewed=force_overwrite_reviewed,
        )
        if not cell_complete:
            con.close()
            return

    con.close()
    _wait_for_final_q()


if __name__ == "__main__":
    app()
