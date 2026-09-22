"""DuckDB-backed persistence for vol_analysis.py experiment runs.

Replaces the old per-experiment candidates.csv / voxels.csv / params.csv files
with four linked tables in a single DuckDB database:

    experiments  -- one row per experiment_number (the run's config)
    candidates   -- one row per detected candidate (identity/location only)
    features     -- one row per candidate (classification-relevant measurements)
    voxels       -- one row per dumped voxel, linked to its candidate

experiment_id links candidates -> experiments; candidate_id links
features/voxels -> candidates. All keys are surrogate (sequence-generated)
integers, so joins never depend on float coordinates or the old
(experiment_number, volume_name, peak_z, peak_y, peak_x) composite key.

A fifth table, `runs`, tracks completion at (experiment_id, volume_name)
granularity -- one experiment_number can be run incrementally, one volume (or
batch of volumes) at a time, and main() skips any volume already recorded for
that experiment rather than gating on the experiment as a whole. `runs` is the
source of truth for "was this volume already done", NOT "does `candidates`
have rows for it" -- a cell can legitimately produce zero candidates, which
would otherwise look indistinguishable from "never ran".

Candidate linking (correspondence between detections of the same physical
particle across different experiments on the same volume, formerly
link_candidates.py's batch union-find over `combined_voxels.parquet` into a
stored `clusters.csv`/`instance_to_cluster.csv`) is NOT materialized anywhere
here. It's answered on demand, two ways:
  - `linked_candidate_ids()` -- a recursive query for ONE seed candidate's
    component, for interactive use (vol_data_review.py's label propagation).
  - `voxel_overlap_edges()` + `assign_components()` -- direct edges plus a
    scipy connected-components pass, for grouping MANY candidates into
    physical particles at once (ml_classifier/score_candidates.py,
    candidate_linking/find_tp_contradictions.py, find_tp_nontp_links.py).
    Component ids from `assign_components()` are scipy-assigned integers,
    stable only within one call -- never persisted or compared across runs
    (unlike the old `cluster_root`, a stringified copy of the composite key).
Both are strictly more correct than a stored clustering: they can never go
stale relative to new experiment runs.

NOTE on z_extent_max: of the historical EXPERIMENT_PARAM_FIELDS, every field
except z_extent_max is constant for a whole experiment_number, so it lives on
`experiments`. z_extent_max is the EFFECTIVE axial slice ceiling for one
cell's voxel_size_mm (see vol_analysis._experiment_param_values) -- it can
differ across cells/volumes within the same experiment, so it is not
experiment-level and is echoed per-candidate on `features` instead (same
place the old code kept it, just no longer dragging the other twelve
constant fields along with it).

Write-path note: `voxels` intentionally has NO foreign-key constraint (it is
plain BIGINT `candidate_id`). It is the ~10-100x larger table, and constraint
checks on every append are not worth the cost when candidate_id is always
minted by this module before use, never user-supplied. `experiments`,
`candidates`, and `features` are small enough that real FK/PK constraints are
free and worth having as a correctness backstop.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from extract_candidates_3d import Candidate

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "Data" / "experiments" / "particles.duckdb"

# The 12 EXPERIMENT_PARAM_FIELDS that are truly experiment-level (constant
# across every cell/volume run under one experiment_number). Order matches the
# `experiments` table's column order after (experiment_id, experiment_number).
EXPERIMENT_FIELDS = [
    "slab_thickness",
    "high_threshold_scale",
    "low_threshold_scale",
    "baseline_method",
    "min_seed_voxels",
    "aniso_factor",
    "small_vol_cutoff",
    "z_pad",
    "small_z_bounds",
    "min_fill",
    "max_axial_extent_mm",
    "enforce_shape_gates",
]

# Classification-relevant measurements, one row per candidate. Order matches
# the `features` table's column order after (candidate_id,).
FEATURE_FIELDS = [
    "n_voxels",
    "n_seed",
    "n_seed_regions",
    "n_seed_total",
    "z_min",
    "z_max",
    "z_extent",
    "y_extent",
    "x_extent",
    "fill",
    "snr_cluster",
    "r_peak",
    "r_peak_ratio",
    "peak_offset",
    "seed_offset",
    "snr_peak",
    "linearity",
    "planarity",
    "sphericity",
    "axis_z",
    "normal_z",
    "edge_contrast",
    "decay_drop",
    "seed_grown_ratio",
    "fill_pca",
    "diag",
    "radial_pos",
    "gs_median",
    "gs_p90",
    "gs_peak",
    "gs_shell_median",
    "gs_contrast",
    "gs_iqr",
    "metal_threshold",
    "l1",
    "l2",
    "l3",
    "surface_ratio",
    "z_extent_max",  # see module docstring -- cell-dependent, echoed here
]


def connect(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the experiments DB and ensure its schema exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    _create_schema(con)
    return con


def _create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SEQUENCE IF NOT EXISTS experiment_id_seq START 1")
    con.execute("CREATE SEQUENCE IF NOT EXISTS candidate_id_seq START 1")

    con.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id BIGINT PRIMARY KEY,
            experiment_number BIGINT NOT NULL UNIQUE,
            slab_thickness BIGINT,
            high_threshold_scale DOUBLE,
            low_threshold_scale DOUBLE,
            baseline_method VARCHAR,
            min_seed_voxels BIGINT,
            aniso_factor DOUBLE,
            small_vol_cutoff BIGINT,
            z_pad BIGINT,
            small_z_bounds BIGINT[],
            min_fill DOUBLE,
            max_axial_extent_mm DOUBLE,
            enforce_shape_gates BOOLEAN
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id BIGINT PRIMARY KEY,
            experiment_id BIGINT NOT NULL REFERENCES experiments(experiment_id),
            volume_name VARCHAR,
            particle INTEGER,
            peak_z INTEGER,
            peak_y INTEGER,
            peak_x INTEGER,
            centroid_z DOUBLE,
            centroid_y DOUBLE,
            centroid_x DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS features (
            candidate_id BIGINT PRIMARY KEY REFERENCES candidates(candidate_id),
            n_voxels BIGINT,
            n_seed BIGINT,
            n_seed_regions BIGINT,
            n_seed_total BIGINT,
            z_min INTEGER,
            z_max INTEGER,
            z_extent INTEGER,
            y_extent INTEGER,
            x_extent INTEGER,
            fill DOUBLE,
            snr_cluster DOUBLE,
            r_peak DOUBLE,
            r_peak_ratio DOUBLE,
            peak_offset DOUBLE,
            seed_offset DOUBLE,
            snr_peak DOUBLE,
            linearity DOUBLE,
            planarity DOUBLE,
            sphericity DOUBLE,
            axis_z DOUBLE,
            normal_z DOUBLE,
            edge_contrast DOUBLE,
            decay_drop DOUBLE,
            seed_grown_ratio DOUBLE,
            fill_pca DOUBLE,
            diag DOUBLE,
            radial_pos DOUBLE,
            gs_median DOUBLE,
            gs_p90 DOUBLE,
            gs_peak DOUBLE,
            gs_shell_median DOUBLE,
            gs_contrast DOUBLE,
            gs_iqr DOUBLE,
            metal_threshold DOUBLE,
            l1 DOUBLE,
            l2 DOUBLE,
            l3 DOUBLE,
            surface_ratio DOUBLE,
            z_extent_max INTEGER
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS voxels (
            candidate_id BIGINT NOT NULL,
            z INTEGER,
            y INTEGER,
            x INTEGER,
            residual INTEGER,
            grayscale INTEGER,
            in_grown INTEGER
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            experiment_id BIGINT NOT NULL REFERENCES experiments(experiment_id),
            volume_name VARCHAR NOT NULL,
            n_candidates BIGINT,
            n_voxels BIGINT,
            PRIMARY KEY (experiment_id, volume_name)
        )
    """)

    # Model scores -- one row per scored candidate, entirely replaced on each
    # ml_classifier/score_candidates.py run (see that module). A candidate
    # outside the scoring slab_thickness window simply has no row here (same
    # meaning as NaN in the old combined_candidates.csv). Small table, so a
    # real FK is cheap -- unlike `voxels`.
    con.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            candidate_id BIGINT PRIMARY KEY REFERENCES candidates(candidate_id),
            ml_prob DOUBLE,
            ml_prob_particle_max DOUBLE,
            ml_prob_max_source_candidate_id BIGINT,
            cluster_ml_label INTEGER,
            in_sample BOOLEAN
        )
    """)


def get_experiment_id(con: duckdb.DuckDBPyConnection, experiment_number: int) -> int | None:
    """Return the experiment_id already recorded for `experiment_number`, or
    None if it hasn't been run at all yet."""
    row = con.execute(
        "SELECT experiment_id FROM experiments WHERE experiment_number = ?",
        [experiment_number],
    ).fetchone()
    return row[0] if row is not None else None


def get_experiment_row(
    con: duckdb.DuckDBPyConnection, experiment_number: int
) -> dict[str, object] | None:
    """Return the stored EXPERIMENT_FIELDS values for `experiment_number` (as a
    dict, same shape vol_analysis._experiment_row_values produces), or None if
    it has no row yet. Used to detect config.toml drift when resuming an
    experiment that already has some volumes recorded -- see
    vol_analysis.main."""
    row = con.execute(
        f"SELECT {', '.join(EXPERIMENT_FIELDS)} FROM experiments WHERE experiment_number = ?",
        [experiment_number],
    ).fetchone()
    if row is None:
        return None
    return dict(zip(EXPERIMENT_FIELDS, row))


def recorded_volumes(con: duckdb.DuckDBPyConnection, experiment_id: int) -> set[str]:
    """Return every volume_name already completed (has a `runs` row) for this
    experiment_id."""
    rows = con.execute(
        "SELECT volume_name FROM runs WHERE experiment_id = ?", [experiment_id]
    ).fetchall()
    return {r[0] for r in rows}


def insert_experiment(
    con: duckdb.DuckDBPyConnection, experiment_number: int, values: dict[str, object]
) -> int:
    """Insert one `experiments` row and return its new experiment_id.

    `values` must have an entry for every name in EXPERIMENT_FIELDS (extra keys,
    e.g. z_extent_max, are ignored -- see module docstring).
    """
    experiment_id = con.execute("SELECT nextval('experiment_id_seq')").fetchone()[0]
    row = [experiment_id, experiment_number] + [values[f] for f in EXPERIMENT_FIELDS]
    placeholders = ",".join("?" for _ in row)
    con.execute(f"INSERT INTO experiments VALUES ({placeholders})", row)
    return experiment_id


def clear_experiment(
    con: duckdb.DuckDBPyConnection,
    experiment_number: int,
    volume_name: str | None = None,
) -> bool:
    """Delete an experiment's recorded data. With `volume_name=None` (default),
    deletes everything under `experiment_number` including the `experiments`
    row itself. With a `volume_name`, scopes the delete to just that one
    volume's candidates/features/voxels/runs row, leaving the experiment and
    its other volumes untouched -- the complement to main()'s per-volume skip:
    this is how you deliberately force one already-recorded volume to be
    redone. Returns False if there was nothing matching to delete.

    Deliberately NOT one explicit transaction: DuckDB's FK enforcement does
    not see a child delete and its parent delete as ordered within the same
    explicit BEGIN/COMMIT (deleting the parent raises a spurious "still
    referenced" ConstraintException even though the child rows are already
    gone) -- see the DuckDB foreign-key-limitations docs. Each DELETE below
    runs as its own autocommitted statement instead, which does not hit this;
    the small risk is that a crash between statements can leave a partial
    delete, cleaned up by simply re-running this (idempotent per stage).
    """
    experiment_id = get_experiment_id(con, experiment_number)
    if experiment_id is None:
        return False

    if volume_name is not None:
        if experiment_id not in {
            r[0]
            for r in con.execute(
                "SELECT experiment_id FROM runs WHERE experiment_id = ? AND volume_name = ?",
                [experiment_id, volume_name],
            ).fetchall()
        }:
            return False
        cand_subquery = (
            "SELECT candidate_id FROM candidates "
            "WHERE experiment_id = ? AND volume_name = ?"
        )
        params = [experiment_id, volume_name]
        con.execute(f"DELETE FROM voxels WHERE candidate_id IN ({cand_subquery})", params)
        con.execute(f"DELETE FROM features WHERE candidate_id IN ({cand_subquery})", params)
        con.execute(
            "DELETE FROM candidates WHERE experiment_id = ? AND volume_name = ?", params
        )
        con.execute(
            "DELETE FROM runs WHERE experiment_id = ? AND volume_name = ?", params
        )
        return True

    con.execute(
        "DELETE FROM voxels WHERE candidate_id IN "
        "(SELECT candidate_id FROM candidates WHERE experiment_id = ?)",
        [experiment_id],
    )
    con.execute(
        "DELETE FROM features WHERE candidate_id IN "
        "(SELECT candidate_id FROM candidates WHERE experiment_id = ?)",
        [experiment_id],
    )
    con.execute("DELETE FROM candidates WHERE experiment_id = ?", [experiment_id])
    con.execute("DELETE FROM runs WHERE experiment_id = ?", [experiment_id])
    con.execute("DELETE FROM experiments WHERE experiment_id = ?", [experiment_id])
    return True


def counts_for_experiment(
    con: duckdb.DuckDBPyConnection, experiment_number: int, volume_name: str | None = None
) -> tuple[int, int] | None:
    """Return (n_candidates, n_voxels) for `experiment_number` (optionally
    scoped to one `volume_name`), or None if the experiment has no row."""
    experiment_id = get_experiment_id(con, experiment_number)
    if experiment_id is None:
        return None
    if volume_name is not None:
        cand_filter = "experiment_id = ? AND volume_name = ?"
        params = [experiment_id, volume_name]
    else:
        cand_filter = "experiment_id = ?"
        params = [experiment_id]
    n_candidates, n_voxels = con.execute(
        f"""
        SELECT
            (SELECT count(*) FROM candidates WHERE {cand_filter}),
            (SELECT count(*) FROM voxels WHERE candidate_id IN
                (SELECT candidate_id FROM candidates WHERE {cand_filter}))
        """,
        params + params,
    ).fetchone()
    return n_candidates, n_voxels


class ExperimentRecorder:
    """Persists one experiment's candidates/features/voxels as they stream out
    of detect_candidates_parallel/streaming.

    Bind one instance per experiment_number (per call into the exp loop in
    vol_analysis.main), and pass `.on_slab_result` as the `on_slab_result=`
    callback threaded through to `_process_slab`. Since results are consumed
    serially in the parent process (see detect_candidates_parallel's
    `for found, voxels in results_iter:` loop), no locking is needed here even
    though detection itself runs across worker processes.

    `volume_name` and `z_extent_max` vary per cell within one experiment, so
    call `begin_volume(volume_name, z_extent_max)` before running detection
    on each cell, and `finish_volume()` after it completes successfully --
    this writes the `runs` row that marks (experiment_id, volume_name) done,
    which is what lets main() skip it on a later invocation. Only call
    finish_volume() on success; if detection raises, skip it so a retry
    picks the volume back up.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, experiment_id: int) -> None:
        self.con = con
        self.experiment_id = experiment_id
        self.volume_name: str = ""
        self.z_extent_max: int = -1
        self.n_candidates = 0
        self.n_voxels = 0
        self._volume_candidates = 0
        self._volume_voxels = 0

    def begin_volume(self, volume_name: str, z_extent_max: int) -> None:
        self.volume_name = volume_name
        self.z_extent_max = z_extent_max
        self._volume_candidates = 0
        self._volume_voxels = 0

    def finish_volume(self) -> None:
        self.con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?)",
            [self.experiment_id, self.volume_name, self._volume_candidates, self._volume_voxels],
        )

    def on_slab_result(
        self, candidates: list[Candidate], voxels: NDArray[np.int32] | None
    ) -> None:
        if not candidates:
            return

        n = len(candidates)
        ids = [
            row[0]
            for row in self.con.execute(
                f"SELECT nextval('candidate_id_seq') FROM range({n})"
            ).fetchall()
        ]

        peak_to_id: dict[tuple[int, int, int], int] = {}
        cand_rows = []
        feat_rows = []
        for c, cid in zip(candidates, ids):
            c.candidate_id = cid
            peak_to_id[(c.peak_z, c.peak_y, c.peak_x)] = cid
            cz, cy, cx = c.centroid
            cand_rows.append(
                (
                    cid,
                    self.experiment_id,
                    self.volume_name,
                    None,  # particle: unlabeled until a manual labeling pass
                    c.peak_z,
                    c.peak_y,
                    c.peak_x,
                    cz,
                    cy,
                    cx,
                )
            )
            feat_rows.append((cid, *(getattr(c, f) for f in FEATURE_FIELDS[:-1]), self.z_extent_max))

        self.con.executemany(
            f"INSERT INTO candidates VALUES ({','.join(['?'] * len(cand_rows[0]))})",
            cand_rows,
        )
        self.con.executemany(
            f"INSERT INTO features VALUES ({','.join(['?'] * len(feat_rows[0]))})",
            feat_rows,
        )
        self.n_candidates += n
        self._volume_candidates += n

        if voxels is not None and voxels.shape[0] > 0:
            cid_col = np.fromiter(
                (peak_to_id[(int(r0), int(r1), int(r2))] for r0, r1, r2 in voxels[:, 0:3]),
                dtype=np.int64,
                count=voxels.shape[0],
            )
            df = pd.DataFrame(
                {
                    "candidate_id": cid_col,
                    "z": voxels[:, 3],
                    "y": voxels[:, 4],
                    "x": voxels[:, 5],
                    "residual": voxels[:, 6],
                    "grayscale": voxels[:, 7],
                    "in_grown": voxels[:, 8],
                }
            )
            self.con.append("voxels", df)
            self.n_voxels += len(df)
            self._volume_voxels += len(df)


def linked_candidate_ids(con: duckdb.DuckDBPyConnection, candidate_id: int) -> list[int]:
    """Return every candidate_id transitively linked to `candidate_id` by
    shared grown (in_grown=1) voxels -- the live replacement for
    link_candidates.py's stored union-find clustering (see module docstring).

    Always includes `candidate_id` itself: a candidate that shares no voxel
    with anything is its own singleton component, same as before when
    cluster_root was NaN. Scoped to the seed's volume_name -- components never
    cross volumes (voxel coordinates are only ever comparable within one
    physical cell), so this also bounds the query to one cell's voxels rather
    than scanning the whole table.
    """
    rows = con.execute(
        """
        WITH RECURSIVE
        edges AS (
            SELECT v1.candidate_id AS a, v2.candidate_id AS b
            FROM voxels v1
            JOIN voxels v2
              ON v1.z = v2.z AND v1.y = v2.y AND v1.x = v2.x
              AND v1.candidate_id <> v2.candidate_id
            JOIN candidates c1 ON c1.candidate_id = v1.candidate_id
            JOIN candidates c2 ON c2.candidate_id = v2.candidate_id
              AND c2.volume_name = c1.volume_name
            WHERE v1.in_grown = 1 AND v2.in_grown = 1
              AND c1.volume_name = (
                  SELECT volume_name FROM candidates WHERE candidate_id = ?
              )
        ),
        component(candidate_id) AS (
            SELECT ?::BIGINT
            UNION
            SELECT e.b FROM component cmp JOIN edges e ON e.a = cmp.candidate_id
        )
        SELECT candidate_id FROM component
        """,
        [candidate_id, candidate_id],
    ).fetchall()
    return [r[0] for r in rows]


def set_particle_label(con: duckdb.DuckDBPyConnection, candidate_id: int, label: int) -> None:
    """Set one candidate's `particle` label in place. `label` is stored
    literally (including -1, the tool's explicit "unsure/unclassify" value) --
    only a candidate that has never been reviewed at all stores SQL NULL."""
    con.execute(
        "UPDATE candidates SET particle = ? WHERE candidate_id = ?", [label, candidate_id]
    )


def set_particle_labels(
    con: duckdb.DuckDBPyConnection, candidate_ids: list[int], label: int
) -> None:
    """Batch form of set_particle_label, for propagating one label to every
    other member of a linked_candidate_ids() component."""
    if not candidate_ids:
        return
    placeholders = ",".join("?" for _ in candidate_ids)
    con.execute(
        f"UPDATE candidates SET particle = ? WHERE candidate_id IN ({placeholders})",
        [label, *candidate_ids],
    )


def load_candidates_dataframe(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Flat, denormalized view (candidates x experiments x features), indexed
    by candidate_id -- the DB-backed replacement for
    `pd.read_csv(combined_candidates.csv)`, shaped to match its old columns so
    downstream code (vol_data_review.py, eventually df_wrapper.py) needs
    minimal changes beyond dropping the file path.

    `particle` is NULL-coalesced to -1 (the tool's "never reviewed" sentinel)
    -- only genuinely untouched candidates get this; once a label is set
    (including explicitly back to -1 via the "u" key) it's stored literally,
    see set_particle_label.

    `ml_prob`/`ml_prob_particle_max`/`ml_prob_max_source_candidate_id`/
    `cluster_ml_label`/`in_sample` come from a LEFT JOIN against `scores` --
    NaN/NULL for any candidate outside the scoring slab_thickness window (see
    ml_classifier/score_candidates.py), same meaning as in the old
    combined_candidates.csv.
    """
    df = con.execute(
        """
        SELECT
            c.candidate_id,
            e.experiment_number,
            c.volume_name,
            COALESCE(c.particle, -1) AS particle,
            c.peak_z, c.peak_y, c.peak_x,
            c.centroid_z, c.centroid_y, c.centroid_x,
            f.n_voxels, f.n_seed, f.n_seed_regions, f.n_seed_total,
            f.z_min, f.z_max, f.z_extent, f.y_extent, f.x_extent, f.fill,
            f.snr_cluster, f.r_peak, f.r_peak_ratio, f.peak_offset, f.seed_offset, f.snr_peak,
            f.linearity, f.planarity, f.sphericity, f.axis_z, f.normal_z,
            f.edge_contrast, f.decay_drop, f.seed_grown_ratio, f.fill_pca, f.diag,
            f.radial_pos, f.gs_median, f.gs_p90, f.gs_peak, f.gs_shell_median,
            f.gs_contrast, f.gs_iqr, f.metal_threshold,
            f.l1, f.l2, f.l3, f.surface_ratio, f.z_extent_max,
            e.slab_thickness, e.high_threshold_scale, e.low_threshold_scale,
            e.baseline_method, e.min_seed_voxels, e.aniso_factor,
            e.small_vol_cutoff, e.z_pad, e.small_z_bounds, e.min_fill,
            e.max_axial_extent_mm, e.enforce_shape_gates,
            s.ml_prob, s.ml_prob_particle_max, s.ml_prob_max_source_candidate_id,
            s.cluster_ml_label, s.in_sample
        FROM candidates c
        JOIN experiments e ON e.experiment_id = c.experiment_id
        JOIN features f ON f.candidate_id = c.candidate_id
        LEFT JOIN scores s ON s.candidate_id = c.candidate_id
        """
    ).df()
    return df.set_index("candidate_id", drop=True)


def voxel_overlap_edges(
    con: duckdb.DuckDBPyConnection, volume_names: list[str] | None = None
) -> pd.DataFrame:
    """Direct (non-transitive) voxel-overlap edges between candidates: pairs
    sharing a grown (in_grown=1) voxel, restricted to the same volume_name
    (components never cross volumes -- see linked_candidate_ids). Returns a
    DataFrame with columns (a, b), one row per edge.

    This is the batch counterpart to linked_candidate_ids(): feed the result
    to scipy.sparse.csgraph.connected_components to get the transitive closure
    over many candidates in one shot (used by
    ml_classifier/score_candidates.py to group candidates into physical
    particles for score/label aggregation). linked_candidate_ids() is the
    single-seed equivalent, for vol_data_review.py's interactive propagation.

    Pass `volume_names` to scope the underlying self-join to just those cells
    (recommended -- avoids scanning voxel rows for volumes you don't care
    about); omit to compute over every volume in the DB.
    """
    conditions = ["v1.in_grown = 1", "v2.in_grown = 1"]
    params: list = []
    if volume_names is not None:
        placeholders = ",".join("?" for _ in volume_names)
        conditions.append(f"c1.volume_name IN ({placeholders})")
        params = list(volume_names)
    where_clause = " AND ".join(conditions)
    return con.execute(
        f"""
        SELECT DISTINCT v1.candidate_id AS a, v2.candidate_id AS b
        FROM voxels v1
        JOIN voxels v2
          ON v1.z = v2.z AND v1.y = v2.y AND v1.x = v2.x
          AND v1.candidate_id < v2.candidate_id
        JOIN candidates c1 ON c1.candidate_id = v1.candidate_id
        JOIN candidates c2 ON c2.candidate_id = v2.candidate_id
          AND c2.volume_name = c1.volume_name
        WHERE {where_clause}
        """,
        params,
    ).df()


def assign_components(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.Series:
    """Component id per row of `df` (must be indexed by candidate_id, with a
    `volume_name` column) -- groups candidates into physical particles via
    shared grown voxels, spanning EVERY candidate in `df` regardless of which
    experiment detected them. Built on voxel_overlap_edges() (direct edges)
    + scipy connected_components (transitive closure) rather than N calls to
    linked_candidate_ids(), so it scales to grouping a whole candidate pool
    at once instead of one seed at a time.

    Component ids are arbitrary integers assigned by scipy, stable only
    within this one call -- never persist or compare them across calls (see
    module docstring). A candidate that shares no voxel with anything in
    `df` gets its own singleton component, same as linked_candidate_ids().
    """
    volume_names = df["volume_name"].unique().tolist()
    edges = voxel_overlap_edges(con, volume_names=volume_names)
    ids = df.index.to_numpy()
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    n = len(ids)
    if len(edges):
        rows = edges["a"].map(id_to_idx).to_numpy()
        cols = edges["b"].map(id_to_idx).to_numpy()
        adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, labels = connected_components(adj, directed=False)
    else:
        labels = np.arange(n)
    return pd.Series(labels, index=ids, name="component")


def replace_scores(con: duckdb.DuckDBPyConnection, scores_df: pd.DataFrame) -> None:
    """Replace the entire `scores` table with fresh rows. `scores_df` must
    have columns (candidate_id, ml_prob, ml_prob_particle_max,
    ml_prob_max_source_candidate_id, cluster_ml_label, in_sample) -- extra
    columns are ignored. A candidate with no row here (e.g. outside the
    scoring slab_thickness window) reads back as NaN/NULL via
    load_candidates_dataframe, same meaning as in the old master CSV.

    Always a full replace, not a merge: scores are only ever "recompute from
    the current model + current data", never edited by hand, so there's
    nothing to preserve from the previous contents.
    """
    con.execute("DELETE FROM scores")
    if len(scores_df):
        con.append(
            "scores",
            scores_df[
                [
                    "candidate_id",
                    "ml_prob",
                    "ml_prob_particle_max",
                    "ml_prob_max_source_candidate_id",
                    "cluster_ml_label",
                    "in_sample",
                ]
            ],
        )
