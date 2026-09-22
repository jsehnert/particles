#!/usr/bin/env python3
"""Delete recorded rows for one experiment_number from the DuckDB experiments
database -- either the whole experiment, or (with --volume-name) just one
volume within it.

vol_analysis.py's main() skips any (experiment_number, volume_name) pair
already recorded in `runs`, so re-running the same command is normally
harmless -- it just picks up whatever volumes haven't been done yet. This
script is for the case where you deliberately want a volume (or a whole
experiment) redone: clear it first, since main() never overwrites recorded
data in place.

Usage:
    uv run scripts/clear_experiment.py <experiment_number>
    uv run scripts/clear_experiment.py <experiment_number> --volume-name M50L-01_processed
"""

import sys
from pathlib import Path

import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import experiment_db as db  # noqa: E402


def main(
    experiment_number: int,
    volume_name: str | None = typer.Option(
        None,
        help="Restrict the clear to one volume within the experiment, leaving "
        "the experiment and its other volumes untouched. Defaults to clearing "
        "the whole experiment.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    con = db.connect()
    counts = db.counts_for_experiment(con, experiment_number, volume_name=volume_name)
    if counts is None:
        scope = f"volume_name={volume_name!r} of " if volume_name else ""
        typer.echo(
            f"No {scope}experiment_number={experiment_number} in {db.DB_PATH}; "
            "nothing to clear."
        )
        raise typer.Exit(0)

    n_candidates, n_voxels = counts
    scope_desc = f"volume {volume_name!r} of experiment_number={experiment_number}" if (
        volume_name
    ) else f"experiment_number={experiment_number} (all volumes)"
    typer.echo(
        f"{scope_desc}: {n_candidates} candidates, {n_voxels} voxels will be "
        f"permanently deleted from {db.DB_PATH}."
    )
    if not yes and not typer.confirm("Proceed?"):
        typer.echo("Aborted; nothing was deleted.")
        raise typer.Exit(1)

    cleared = db.clear_experiment(con, experiment_number, volume_name=volume_name)
    if cleared:
        typer.echo(f"Cleared {scope_desc}.")
    else:
        typer.echo(f"Nothing matched {scope_desc}; nothing was deleted.")


if __name__ == "__main__":
    typer.run(main)
