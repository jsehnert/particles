#!/usr/bin/env python3
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import typer
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cfg_data import AnalysisBase


class GlobalData(AnalysisBase):
    def __init__(self, config_path: Path, vol_file_name: str):
        super().__init__(config_path, vol_file_name=vol_file_name)


global_data: GlobalData

app = typer.Typer()


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


def _normalize_image(img: NDArray) -> NDArray:
    """Normalize a 2D frame to 8-bit grayscale based on the grayscale range of the volume."""
    gray_landmarks = global_data.grayscale_landmarks
    air_grayvalue = gray_landmarks["air_grayvalue"]
    metal_threshold = gray_landmarks["metal_threshold"]
    max_grayvalue = gray_landmarks["max_grayvalue"]
    upper_limit = max_grayvalue
    img = (img.astype(float) - air_grayvalue) / (upper_limit - air_grayvalue)
    img *= 255.0
    img = img.clip(0, 255).astype("uint8")
    return img


def _display_loop(
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    labels_path: Path,
    stem: str,
    img_h: int,
    img_w: int,
    auto_zoom: bool = False,
) -> None:
    win = "vol_review"
    win_mip = "vol_review_mip"
    win_info = "candidate_info"
    win_side = "vol_review_side"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_mip, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_info, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_side, cv2.WINDOW_NORMAL)
    z_depth = global_data.vol.shape[0]
    side_slice = global_data.vol[:, :, img_w // 2]  # shape (z_depth, img_h)

    default_zoom = 5.0 if auto_zoom else 1.0
    idx = 0
    zoom = default_zoom
    while True:
        row = df.iloc[idx]
        z0 = round(row["centroid_z"])
        cx, cy = round(row["centroid_x"]), round(row["centroid_y"])
        dist = (
            (row["centroid_x"] - img_w / 2) ** 2 + (row["centroid_y"] - img_h / 2) ** 2
        ) ** 0.5

        label_match = labels_df[
            labels_df["stem"].eq(stem)
            & labels_df["centroid_z"].eq(row["centroid_z"])
            & labels_df["centroid_x"].eq(row["centroid_x"])
            & labels_df["centroid_y"].eq(row["centroid_y"])
        ]
        if label_match.empty:
            circle_color = (0, 255, 255)  # yellow — unlabeled
        elif label_match.iloc[0]["Particle"] == 0:
            circle_color = (0, 0, 255)  # red — Particle=0
        else:
            circle_color = (0, 255, 0)  # green — Particle=1

        c_z = row["centroid_z"]
        z1, z2 = np.floor(c_z), np.ceil(c_z)
        if z1 != z2:
            w2 = c_z - z1
            frame = (1.0 - w2) * global_data.vol[int(z1), :, :] + w2 * global_data.vol[
                int(z2), :, :
            ]
        else:
            frame = global_data.vol[z0, :, :]
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
                    interpolation=cv2.INTER_LINEAR,
                )

            frame = _apply_zoom(frame)

        title_base = f"[{idx + 1}/{len(df)}]  z={z0}  ({cx}, {cy})  zoom={zoom:.1f}x"

        z_lo = max(0, z0 - 5)
        z_hi = min(z_depth, z0 + 6)
        mip = global_data.vol[z_lo:z_hi, :, :].max(axis=0)
        mip_frame = cv2.cvtColor(_normalize_image(mip), cv2.COLOR_GRAY2BGR)
        cv2.circle(mip_frame, (cx, cy), 10, circle_color, thickness=1)
        if zoom > 1.0:
            mip_frame = _apply_zoom(mip_frame)
        cv2.setWindowTitle(win_mip, f"max projection  {stem}  {title_base}")
        cv2.imshow(win_mip, mip_frame)

        side_frame = cv2.cvtColor(_normalize_image(side_slice), cv2.COLOR_GRAY2BGR)
        side_w = side_frame.shape[1]
        cv2.line(side_frame, (0, z0), (side_w, z0), (230, 216, 173), 3)  # light blue
        cv2.setWindowTitle(win_side, f"side view  {stem}  z={z0}")
        cv2.imshow(win_side, side_frame)

        info_row = pd.concat([row, pd.Series({"distance_to_center": dist})])
        cv2.imshow(win_info, _render_info_panel(info_row))

        cv2.setWindowTitle(win, f"{stem}  {title_base}")
        cv2.imshow(win, frame)

        key = cv2.waitKey(0)
        if key == 3:  # right arrow (macOS)
            idx = min(idx + 1, len(df) - 1)
            zoom = default_zoom
        elif key == 2:  # left arrow (macOS)
            idx = max(idx - 1, 0)
            zoom = default_zoom
        elif key in (ord("+"), ord("=")):
            zoom *= 1.5
        elif key == ord("-"):
            zoom = max(1.0, zoom / 1.5)
        elif key in (ord("f"), ord("t")):
            particle = 0 if key == ord("f") else 1
            dup_mask = (
                labels_df["stem"].eq(stem)
                & labels_df["centroid_z"].eq(row["centroid_z"])
                & labels_df["centroid_x"].eq(row["centroid_x"])
                & labels_df["centroid_y"].eq(row["centroid_y"])
            )
            if dup_mask.any():
                labels_df.loc[dup_mask, "Particle"] = particle
                typer.echo(f"  [{idx + 1}] updated to Particle={particle}")
            else:
                new_row = {
                    "stem": stem,
                    "Particle": particle,
                    **row.to_dict(),
                    "distance_to_center": dist,
                }
                labels_df = pd.concat(
                    [labels_df, pd.DataFrame([new_row])], ignore_index=True
                )
                typer.echo(f"  [{idx + 1}] labeled Particle={particle}")
            labels_df.to_csv(labels_path, index=False)
        elif key == ord("q") or key == 27:  # q or Esc
            break

    cv2.destroyAllWindows()


@app.command()
def main(
    vol_file_name: str = typer.Argument(
        ..., help="Volume file name (e.g. M50L-04.raw)"
    ),
    data_path: Path = typer.Argument(
        PROJECT_ROOT / "Data" / "experiments",
        help="Directory containing the candidates CSV",
    ),
    min_voxels: int = typer.Option(10, help="Minimum n_voxels threshold (inclusive)"),
    z_min: float = typer.Option(0.0, help="Minimum centroid_z value (inclusive)"),
    az: bool = typer.Option(False, "--az", help="Auto-zoom to centroid on navigation"),
    experiment: str = typer.Option(..., help="Experiment name"),
) -> None:
    global global_data
    stem = Path(vol_file_name).stem
    global_data = GlobalData(PROJECT_ROOT / "config.toml", vol_file_name=stem + ".raw")

    experiment_path = data_path / Path(experiment)
    if not experiment_path.exists():
        typer.echo(f"Experiment path does not exist: {experiment_path}")
        raise typer.Exit(code=1)
    candidates_path = experiment_path / (stem + "_candidates.csv")

    typer.echo(f"Volume : {vol_file_name}  (shape: {global_data.vol.shape})")
    typer.echo(f"Candidates: {candidates_path}")

    df = pd.read_csv(candidates_path)
    df = (
        df[(df["n_voxels"] >= min_voxels) & (df["centroid_z"] >= z_min)]
        .sort_values("centroid_z")
        .reset_index(drop=True)
    )
    typer.echo(f"\n{len(df)} candidates with n_voxels >= {min_voxels} and centroid_z >= {z_min}:\n")
    typer.echo(df.columns)

    labels_path = experiment_path / "data_labels.csv"
    label_cols = ["stem", "Particle"] + list(df.columns) + ["distance_to_center"]
    if not labels_path.exists():
        labels_df = pd.DataFrame(columns=label_cols)
        labels_df.to_csv(labels_path, index=False)
    else:
        labels_df = pd.read_csv(labels_path)

    _, img_h, img_w = global_data.vol.shape
    _display_loop(df, labels_df, labels_path, stem, img_h, img_w, auto_zoom=az)


if __name__ == "__main__":
    app()
