#!/usr/bin/env python3
"""Review and adjust the z_min/z_max extents for a cell defined in config.toml."""

import re
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np
import typer
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
sys.path.insert(0, str(PROJECT_ROOT))

from cfg_data import AnalysisBase  # noqa: E402

app = typer.Typer()

WIN = "z extents review"
LINE_COLOR = (230, 216, 173)  # light blue (BGR)
LINE_THICKNESS = 1
CROP_ROWS_MIN = 500
CROP_ROWS_MAX = 800

KEYS_UP = {82, 0, 63232}
KEYS_DOWN = {84, 1, 63233}
KEYS_RIGHT = {83, 3, 63235}
KEYS_QUIT = {27, ord("q")}
KEY_SAVE = ord("s")


class GlobalData(AnalysisBase):
    def __init__(self, cfg_path: Path, vol_index: int) -> None:
        super().__init__(cfg_path, vol_index=vol_index, exp_index=0)


def _read_cells(cfg_path: Path) -> list[dict]:
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg["cell"]


def _write_cell_field(cfg_path: Path, index: int, field: str, new_value: int) -> None:
    """Rewrite the given field for the index-th [[cell]] block, leaving the
    rest of the config file (including comments) untouched."""
    lines = cfg_path.read_text().splitlines(keepends=True)
    cell_count = -1
    in_target = False
    for i, line in enumerate(lines):
        if line.strip() == "[[cell]]":
            cell_count += 1
            in_target = cell_count == index
            continue
        if in_target and re.match(rf"^\s*{field}\s*=", line):
            lines[i] = re.sub(r"=\s*-?\d+", f"= {new_value}", line, count=1)
            break
    cfg_path.write_text("".join(lines))


def _normalize_image(img: NDArray) -> NDArray:
    """Normalize a 2D frame to 8-bit grayscale based on the grayscale range of the volume."""
    global global_data

    try:
        gray_landmarks = global_data.grayscale_landmarks
        air_grayvalue = gray_landmarks["air_grayvalue"]
        max_grayvalue = gray_landmarks["max_grayvalue"]
        upper_limit = max_grayvalue
        img = (img.astype(float) - air_grayvalue) / (upper_limit - air_grayvalue)
    except Exception:
        img = (img - img.min()) / (img.max() - img.min())
        print("Global data is not properly initialized with grayscale landmarks.")

    img *= 255.0
    img = img.clip(0, 255).astype("uint8")
    return img


global_data: GlobalData


@app.command()
def main(
    max_mode: bool = typer.Option(
        False, "--max", help="Edit the cell's z_max entry instead of z_min"
    ),
) -> None:
    global global_data
    field = "z_max" if max_mode else "z_min"
    n_cells = len(_read_cells(CONFIG_PATH))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    state: dict = {}

    def load_cell(index: int) -> None:
        global global_data
        cell = _read_cells(CONFIG_PATH)[index]
        name = cell["name"]
        z_value = int(cell[field])

        typer.echo(f"Cell[{index}] = {name}  {field} = {z_value}")
        typer.echo(f"Loading volume for {name}...")
        global_data = GlobalData(CONFIG_PATH, vol_index=index)
        vol = global_data.vol
        if vol is None:
            raise RuntimeError(f"Failed to load volume for {name}.")

        typer.echo(f"Volume shape: {vol.shape}")

        z_depth = vol.shape[0]
        if max_mode:
            crop_rows = CROP_ROWS_MAX
            crop_start = max(0, z_depth - crop_rows)
        else:
            crop_rows = CROP_ROWS_MIN
            crop_start = 0
        crop_end = min(crop_start + crop_rows, z_depth)

        # Pull the needed z-range into memory as one contiguous (sequential)
        # read, then extract the axial column in-memory. Indexing the fixed-x
        # column directly off the memmap for the full volume is a maximally
        # strided access pattern that ends up faulting in nearly the whole
        # multi-GB file just to build a single column.
        sub_vol = np.asarray(vol[crop_start:crop_end, :, :])
        mid_x = sub_vol.shape[2] // 2
        mid_y = sub_vol.shape[1] // 2
        display_slice = _normalize_image(sub_vol[:, mid_y, :])

        state["index"] = index
        state["name"] = name
        state["z_value"] = z_value
        state["z_bound"] = z_depth - 1
        state["crop_start"] = crop_start
        state["display_slice"] = display_slice

    def render() -> None:
        img = state["display_slice"]
        print(f"Image shape: {img.shape}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        w = img.shape[1]
        line_y = state["z_value"] - state["crop_start"]
        cv2.line(img, (0, line_y), (w - 1, line_y), LINE_COLOR, LINE_THICKNESS)
        cv2.setWindowTitle(WIN, f"{state['name']}  {field}={state['z_value']}")
        cv2.imshow(WIN, img)

    load_cell(0)
    render()
    typer.echo(
        f"Up/Down: move {field}   Right: next cell   s: save to config.toml   q: quit"
    )

    while True:
        key = cv2.waitKey(20)
        if key in KEYS_QUIT:
            break
        elif key in KEYS_UP:
            state["z_value"] = max(state["z_value"] - 1, 0)
            render()
        elif key in KEYS_DOWN:
            state["z_value"] = min(state["z_value"] + 1, state["z_bound"])
            render()
        elif key in KEYS_RIGHT:
            next_index = state["index"] + 1
            if next_index < n_cells:
                load_cell(next_index)
                render()
            else:
                typer.echo("Already at the last cell.")
        elif key == KEY_SAVE:
            _write_cell_field(CONFIG_PATH, state["index"], field, state["z_value"])
            typer.echo(
                f"Saved {field}={state['z_value']} for cell[{state['index']}] "
                f"({state['name']}) to {CONFIG_PATH}"
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
