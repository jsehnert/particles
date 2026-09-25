# Slice viewer

Install dependencies and run with the cell index from config.toml:

```sh
uv sync
uv run python apps/slice_analysis/slice_analysis.py 0
```

PySide6 manages three independent, resizable image windows (Slice, Max
Projection, Enhancement) and an Info/controls window. Move or maximize the
windows on any monitor. Closing a viewer window exits the application.
Images retain their source resolution and aspect ratio.

- Left click: place a green point at the exact image pixel and update the
  pending maximum from its 3×3×3 neighborhood.
- Left drag: draw an ROI. Drag distance is measured in screen pixels.
- Right click: recenter the shared viewport.
- Wheel, +/−: zoom all three views. 0 restores the full image.
- A/D or left/right arrows: change z. The Info window also has a slice
  slider and numeric entry.
- R or Record: append the pending refined voxel to Data/particles_found.csv.
  The row includes the estimated noise level and particle volume in voxels.
  Existing volume/coordinate records are skipped. Changing z clears the pending
  selection.
- Next particle: scan forward one slice at a time; press Stop search to stop.
- Q or Escape: exit.

Displayed centers use (row, col); voxel locations use (z, row, col).
CSV coordinates are ordered z, y, x, where y=row and x=column.

Processing still executes on the GUI thread. A particularly expensive slice
can briefly delay input; the search returns to the Qt event loop between slices.

Run the synthetic interaction checks without opening desktop windows:

```sh
QT_QPA_PLATFORM=offscreen uv run python -m unittest scripts.test_slice_viewer_qt
```
