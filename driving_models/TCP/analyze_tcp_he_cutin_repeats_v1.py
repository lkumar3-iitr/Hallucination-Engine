"""
analyze_tcp_he_cutin_repeats_v1.py

Repeated CARLA-vs-HE analysis for tcp_cutin_001.

The important distinction for a cut-in is:

    pre-existing closed-loop divergence
        vs
    additional divergence during / after lane entry.

Metrics:
    - cut-in start
    - actor-center lane-boundary crossing
    - ego-lane-center arrival
    - gap at each event
    - minimum gap after lane entry
    - sustained braking
    - final gap
    - collision
    - speed/control/gap MAE after lane entry

Also saves:
    - aggregate time-series CSV
    - JSON summary
"""

import csv
import json
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
TCP_DIR = THIS_FILE.parent

SCENARIO_ID = "tcp_cutin_001"

ROOT = (
    TCP_DIR
    / "outputs"
    / "tcp_he_repeat_v1"
    / SCENARIO_ID
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
)


# ------------------------------------------------------------
# Cut-in event definitions
# ------------------------------------------------------------

# Actor starts at y=+3.5 m.
#
# +y = left in ScenarioGenerator.
#
# It moves toward y=0 during the cut-in.

CUTIN_MOVE_THRESHOLD_M = 0.10

# Approximate center-line boundary between:
#
# adjacent lane center = +3.5 m
# ego lane center      =  0.0 m
#
# Therefore midpoint = 1.75 m.
LANE_BOUNDARY_Y_M = 1.75

# Actor considered approximately centered in ego lane.
LANE_CENTER_THRESHOLD_M = 0.20


BRAKE_THRESHOLD = 0.5
BRAKE_CONFIRM_FRAMES = 3


# ============================================================
# Helpers
# ============================================================

def as_float(row, key):

    value = str(
        row.get(key, "")
    ).strip()

    if value == "":
        return float("nan")

    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_csv(path):

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as fp:

        return list(
            csv.DictReader(fp)
        )


def find_csv(run_dir, condition):

    files = sorted(
        (
            run_dir
            / condition
        ).glob("*.csv")
    )

    files = [
        p
        for p in files
        if "comparison" not in p.name
    ]

    if len(files) != 1:

        raise RuntimeError(
            f"Expected exactly one CSV in "
            f"{run_dir / condition}; found {files}"
        )

    return files[0]


def finite_mean_std(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            float("nan"),
            float("nan"),
        )

    mean = float(
        np.mean(values)
    )

    std = (
        float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else 0.0
    )

    return mean, std


def print_stat(
    label,
    values,
    unit="",
):

    mean, std = (
        finite_mean_std(
            values
        )
    )

    print(
        f"{label:36s}: "
        f"{mean:9.3f} +/- "
        f"{std:8.3f} {unit}"
    )


# ============================================================
# Event detection
# ============================================================

def find_cutin_start(rows):

    initial_y = as_float(
        rows[0],
        "actor_y_sg_m",
    )

    for i, row in enumerate(
        rows
    ):

        y = as_float(
            row,
            "actor_y_sg_m",
        )

        if not np.isfinite(y):
            continue

        if (
            abs(
                y - initial_y
            )
            >= CUTIN_MOVE_THRESHOLD_M
        ):
            return i

    return None


def find_lane_boundary_crossing(
    rows
):

    for i, row in enumerate(
        rows
    ):

        y = as_float(
            row,
            "actor_y_sg_m",
        )

        if (
            np.isfinite(y)
            and
            y <= LANE_BOUNDARY_Y_M
        ):
            return i

    return None


def find_lane_center_arrival(
    rows
):

    for i, row in enumerate(
        rows
    ):

        y = as_float(
            row,
            "actor_y_sg_m",
        )

        if (
            np.isfinite(y)
            and
            abs(y)
            <= LANE_CENTER_THRESHOLD_M
        ):
            return i

    return None


def first_sustained_brake(
    rows,
    start_idx,
):

    if start_idx is None:
        return None

    for i in range(
        start_idx,
        len(rows)
        - BRAKE_CONFIRM_FRAMES
        + 1,
    ):

        window = rows[
            i:
            i + BRAKE_CONFIRM_FRAMES
        ]

        brakes = [
            as_float(
                row,
                "tcp_brake",
            )
            for row in window
        ]

        if all(
            np.isfinite(v)
            and
            v >= BRAKE_THRESHOLD
            for v in brakes
        ):
            return i

    return None


# ============================================================
# Per-condition metrics
# ============================================================

def compute_metrics(rows):

    cutin_idx = (
        find_cutin_start(
            rows
        )
    )

    boundary_idx = (
        find_lane_boundary_crossing(
            rows
        )
    )

    center_idx = (
        find_lane_center_arrival(
            rows
        )
    )

    sustained_idx = (
        first_sustained_brake(
            rows,
            cutin_idx,
        )
    )

    def event_value(
        idx,
        key,
    ):

        if idx is None:
            return float("nan")

        return as_float(
            rows[idx],
            key,
        )

    gaps = np.array(
        [
            as_float(
                row,
                "bumper_gap_m",
            )
            for row in rows
        ],
        dtype=float,
    )

    collision = any(
        str(
            row.get(
                "virtual_collision",
                "",
            )
        ).strip().lower()
        in (
            "true",
            "1",
            "yes",
        )
        for row in rows
    )

    if boundary_idx is not None:

        post_boundary_gaps = (
            gaps[
                boundary_idx:
            ]
        )

        post_boundary_gaps = (
            post_boundary_gaps[
                np.isfinite(
                    post_boundary_gaps
                )
            ]
        )

    else:

        post_boundary_gaps = (
            np.array(
                [],
                dtype=float,
            )
        )

    finite_gaps = gaps[
        np.isfinite(gaps)
    ]

    return {

        "rows":
            rows,

        "cutin_idx":
            cutin_idx,

        "boundary_idx":
            boundary_idx,

        "center_idx":
            center_idx,

        "cutin_start_s":
            event_value(
                cutin_idx,
                "t_s",
            ),

        "cutin_gap_m":
            event_value(
                cutin_idx,
                "bumper_gap_m",
            ),

        "boundary_time_s":
            event_value(
                boundary_idx,
                "t_s",
            ),

        "boundary_gap_m":
            event_value(
                boundary_idx,
                "bumper_gap_m",
            ),

        "boundary_ego_speed_mps":
            event_value(
                boundary_idx,
                "ego_speed_mps",
            ),

        "center_time_s":
            event_value(
                center_idx,
                "t_s",
            ),

        "center_gap_m":
            event_value(
                center_idx,
                "bumper_gap_m",
            ),

        "sustained_brake_s":
            event_value(
                sustained_idx,
                "t_s",
            ),

        "minimum_gap_after_boundary_m":
            (
                float(
                    np.min(
                        post_boundary_gaps
                    )
                )
                if len(
                    post_boundary_gaps
                )
                else float("nan")
            ),

        "final_gap_m":
            (
                float(
                    finite_gaps[-1]
                )
                if len(
                    finite_gaps
                )
                else float("nan")
            ),

        "collision":
            bool(
                collision
            ),
    }


# ============================================================
# Pair-wise trajectory differences
# ============================================================

def aligned_pair_metrics(
    carla,
    he,
):

    c_rows = carla["rows"]
    h_rows = he["rows"]

    n = min(
        len(c_rows),
        len(h_rows),
    )

    # Same physical scenario means actor-frame events
    # should occur at the same probe index.
    boundary_idx = (
        carla[
            "boundary_idx"
        ]
    )

    if boundary_idx is None:

        boundary_idx = 0

    boundary_idx = min(
        boundary_idx,
        n - 1,
    )

    def mae_after(
        key
    ):

        diffs = []

        for i in range(
            boundary_idx,
            n,
        ):

            c = as_float(
                c_rows[i],
                key,
            )

            h = as_float(
                h_rows[i],
                key,
            )

            if (
                np.isfinite(c)
                and
                np.isfinite(h)
            ):

                diffs.append(
                    abs(
                        h - c
                    )
                )

        if not diffs:
            return float("nan")

        return float(
            np.mean(
                diffs
            )
        )

    return {

        "post_boundary_speed_mae_mps":
            mae_after(
                "ego_speed_mps"
            ),

        "post_boundary_gap_mae_m":
            mae_after(
                "bumper_gap_m"
            ),

        "post_boundary_throttle_mae":
            mae_after(
                "tcp_throttle"
            ),

        "post_boundary_brake_mae":
            mae_after(
                "tcp_brake"
            ),

        "post_boundary_steer_mae":
            mae_after(
                "tcp_steer"
            ),
    }


# ============================================================
# Aggregate time series for future plots
# ============================================================

def build_aggregate_timeseries(
    results
):

    n = min(
        min(
            len(
                result[
                    "carla"
                ]["rows"]
            ),
            len(
                result[
                    "he"
                ]["rows"]
            ),
        )
        for result in results
    )

    fields = [
        "ego_speed_mps",
        "bumper_gap_m",
        "lateral_distance_m",
        "tcp_throttle",
        "tcp_brake",
        "tcp_steer",
    ]

    output = []

    for i in range(n):

        row = {

            "probe_idx":
                i,

            "t_s":
                as_float(
                    results[0][
                        "carla"
                    ]["rows"][i],
                    "t_s",
                ),

            "actor_y_sg_m":
                as_float(
                    results[0][
                        "carla"
                    ]["rows"][i],
                    "actor_y_sg_m",
                ),
        }

        for condition in (
            "carla",
            "he",
        ):

            for field in fields:

                values = [
                    as_float(
                        result[
                            condition
                        ]["rows"][i],
                        field,
                    )
                    for result in results
                ]

                mean, std = (
                    finite_mean_std(
                        values
                    )
                )

                row[
                    f"{condition}_{field}_mean"
                ] = mean

                row[
                    f"{condition}_{field}_std"
                ] = std

        row[
            "gap_delta_mean_m"
        ] = (
            row[
                "he_bumper_gap_m_mean"
            ]
            -
            row[
                "carla_bumper_gap_m_mean"
            ]
        )

        row[
            "speed_delta_mean_mps"
        ] = (
            row[
                "he_ego_speed_mps_mean"
            ]
            -
            row[
                "carla_ego_speed_mps_mean"
            ]
        )

        output.append(
            row
        )

    return output


# ============================================================
# Main
# ============================================================

def main():

    if not ROOT.exists():

        raise FileNotFoundError(
            ROOT
        )

    run_dirs = sorted(
        [
            p
            for p in ROOT.glob(
                "run_*"
            )
            if p.is_dir()
        ]
    )

    if not run_dirs:

        raise RuntimeError(
            f"No repeat runs in {ROOT}"
        )

    results = []

    for run_dir in run_dirs:

        run_number = int(
            run_dir.name.split(
                "_"
            )[-1]
        )

        carla_rows = load_csv(
            find_csv(
                run_dir,
                "carla",
            )
        )

        he_rows = load_csv(
            find_csv(
                run_dir,
                "he",
            )
        )

        carla = compute_metrics(
            carla_rows
        )

        he = compute_metrics(
            he_rows
        )

        pair = aligned_pair_metrics(
            carla,
            he,
        )

        results.append({

            "run":
                run_number,

            "carla":
                carla,

            "he":
                he,

            "pair":
                pair,
        })

    print()
    print("=" * 108)
    print("TCP CARLA vs HE CUT-IN REPEATED-TRIAL ANALYSIS")
    print("=" * 108)

    print()
    print("SCENARIO EVENTS")
    print("-" * 108)

    # Physical trajectory should be identical in both.
    example = results[0][
        "carla"
    ]

    print(
        "cut-in begins          :",
        f"{example['cutin_start_s']:.2f} s",
    )

    print(
        "lane-boundary crossing :",
        f"{example['boundary_time_s']:.2f} s",
        f"(actor center y <= {LANE_BOUNDARY_Y_M:.2f} m)",
    )

    print(
        "ego-lane center arrival:",
        f"{example['center_time_s']:.2f} s",
    )

    # --------------------------------------------------------
    # Per-run
    # --------------------------------------------------------

    print()
    print("PER-RUN OUTCOMES")
    print("-" * 108)

    print(
        "run | "
        "C gap(bound) H gap(bound) delta | "
        "C min-post H min-post delta | "
        "C final H final delta | collision"
    )

    print("-" * 108)

    for r in results:

        c = r["carla"]
        h = r["he"]

        print(
            f"{r['run']:3d} | "
            f"{c['boundary_gap_m']:11.3f} "
            f"{h['boundary_gap_m']:11.3f} "
            f"{h['boundary_gap_m'] - c['boundary_gap_m']:+7.3f} | "
            f"{c['minimum_gap_after_boundary_m']:10.3f} "
            f"{h['minimum_gap_after_boundary_m']:10.3f} "
            f"{h['minimum_gap_after_boundary_m'] - c['minimum_gap_after_boundary_m']:+7.3f} | "
            f"{c['final_gap_m']:7.3f} "
            f"{h['final_gap_m']:7.3f} "
            f"{h['final_gap_m'] - c['final_gap_m']:+7.3f} | "
            f"C={int(c['collision'])} "
            f"H={int(h['collision'])}"
        )

    # --------------------------------------------------------
    # Condition distributions
    # --------------------------------------------------------

    for condition, title in (
        (
            "carla",
            "CARLA DISTRIBUTION",
        ),
        (
            "he",
            "HE DISTRIBUTION",
        ),
    ):

        print()
        print(title)
        print("-" * 108)

        print_stat(
            "gap when cut-in begins",
            [
                r[condition][
                    "cutin_gap_m"
                ]
                for r in results
            ],
            "m",
        )

        print_stat(
            "gap at lane boundary",
            [
                r[condition][
                    "boundary_gap_m"
                ]
                for r in results
            ],
            "m",
        )

        print_stat(
            "ego speed at lane boundary",
            [
                r[condition][
                    "boundary_ego_speed_mps"
                ]
                for r in results
            ],
            "m/s",
        )

        print_stat(
            "gap at lane center",
            [
                r[condition][
                    "center_gap_m"
                ]
                for r in results
            ],
            "m",
        )

        print_stat(
            "minimum gap after boundary",
            [
                r[condition][
                    "minimum_gap_after_boundary_m"
                ]
                for r in results
            ],
            "m",
        )

        print_stat(
            "final gap",
            [
                r[condition][
                    "final_gap_m"
                ]
                for r in results
            ],
            "m",
        )

        print_stat(
            "sustained brake onset",
            [
                r[condition][
                    "sustained_brake_s"
                ]
                for r in results
            ],
            "s",
        )

    # --------------------------------------------------------
    # Paired differences
    # --------------------------------------------------------

    print()
    print("PAIRED HE - CARLA DIFFERENCES")
    print("-" * 108)

    paired_metrics = {

        "gap at cut-in start":
            [
                r["he"]["cutin_gap_m"]
                -
                r["carla"]["cutin_gap_m"]
                for r in results
            ],

        "gap at lane boundary":
            [
                r["he"]["boundary_gap_m"]
                -
                r["carla"]["boundary_gap_m"]
                for r in results
            ],

        "gap at lane center":
            [
                r["he"]["center_gap_m"]
                -
                r["carla"]["center_gap_m"]
                for r in results
            ],

        "minimum post-boundary gap":
            [
                r["he"][
                    "minimum_gap_after_boundary_m"
                ]
                -
                r["carla"][
                    "minimum_gap_after_boundary_m"
                ]
                for r in results
            ],

        "final gap":
            [
                r["he"]["final_gap_m"]
                -
                r["carla"]["final_gap_m"]
                for r in results
            ],
    }

    print_stat(
        "delta gap at cut-in start",
        paired_metrics[
            "gap at cut-in start"
        ],
        "m",
    )

    print_stat(
        "delta gap at lane boundary",
        paired_metrics[
            "gap at lane boundary"
        ],
        "m",
    )

    print_stat(
        "delta gap at lane center",
        paired_metrics[
            "gap at lane center"
        ],
        "m",
    )

    print_stat(
        "delta minimum post-boundary gap",
        paired_metrics[
            "minimum post-boundary gap"
        ],
        "m",
    )

    print_stat(
        "delta final gap",
        paired_metrics[
            "final gap"
        ],
        "m",
    )

    # --------------------------------------------------------
    # Important: additional divergence caused after cut-in starts
    # --------------------------------------------------------

    additional_from_start = [
        (
            r["he"]["final_gap_m"]
            -
            r["carla"]["final_gap_m"]
        )
        -
        (
            r["he"]["cutin_gap_m"]
            -
            r["carla"]["cutin_gap_m"]
        )
        for r in results
    ]

    additional_from_boundary = [
        (
            r["he"]["final_gap_m"]
            -
            r["carla"]["final_gap_m"]
        )
        -
        (
            r["he"]["boundary_gap_m"]
            -
            r["carla"]["boundary_gap_m"]
        )
        for r in results
    ]

    print()
    print("DIVERGENCE ATTRIBUTION")
    print("-" * 108)

    print_stat(
        "extra gap divergence after cut-in start",
        additional_from_start,
        "m",
    )

    print_stat(
        "extra gap divergence after lane boundary",
        additional_from_boundary,
        "m",
    )

    # --------------------------------------------------------
    # Post-boundary behavior
    # --------------------------------------------------------

    print()
    print("POST-LANE-ENTRY CARLA-vs-HE TRAJECTORY DIFFERENCE")
    print("-" * 108)

    print_stat(
        "speed MAE",
        [
            r["pair"][
                "post_boundary_speed_mae_mps"
            ]
            for r in results
        ],
        "m/s",
    )

    print_stat(
        "gap MAE",
        [
            r["pair"][
                "post_boundary_gap_mae_m"
            ]
            for r in results
        ],
        "m",
    )

    print_stat(
        "throttle MAE",
        [
            r["pair"][
                "post_boundary_throttle_mae"
            ]
            for r in results
        ],
        "",
    )

    print_stat(
        "brake MAE",
        [
            r["pair"][
                "post_boundary_brake_mae"
            ]
            for r in results
        ],
        "",
    )

    print_stat(
        "steer MAE",
        [
            r["pair"][
                "post_boundary_steer_mae"
            ]
            for r in results
        ],
        "",
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    carla_collisions = sum(
        int(
            r["carla"][
                "collision"
            ]
        )
        for r in results
    )

    he_collisions = sum(
        int(
            r["he"][
                "collision"
            ]
        )
        for r in results
    )

    print()
    print("SAFETY OUTCOME")
    print("-" * 108)

    print(
        f"CARLA collisions: "
        f"{carla_collisions}/{len(results)}"
    )

    print(
        f"HE collisions   : "
        f"{he_collisions}/{len(results)}"
    )

    # --------------------------------------------------------
    # Save future-paper data
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate = (
        build_aggregate_timeseries(
            results
        )
    )

    aggregate_path = (
        OUTPUT_DIR
        / "tcp_cutin_001_aggregate_timeseries.csv"
    )

    with aggregate_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                aggregate[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            aggregate
        )

    summary = {

        "scenario_id":
            SCENARIO_ID,

        "num_pairs":
            len(results),

        "event_definitions": {

            "cutin_move_threshold_m":
                CUTIN_MOVE_THRESHOLD_M,

            "lane_boundary_y_m":
                LANE_BOUNDARY_Y_M,

            "lane_center_threshold_m":
                LANE_CENTER_THRESHOLD_M,
        },

        "additional_gap_divergence_after_cutin_start_m":
            additional_from_start,

        "additional_gap_divergence_after_lane_boundary_m":
            additional_from_boundary,

        "carla_collisions":
            carla_collisions,

        "he_collisions":
            he_collisions,
    }

    summary_path = (
        OUTPUT_DIR
        / "tcp_cutin_001_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            summary,
            fp,
            indent=2,
        )

    print()
    print("SAVED")
    print("-" * 108)

    print(
        "aggregate CSV:",
        aggregate_path,
    )

    print(
        "summary JSON :",
        summary_path,
    )

    print()
    print("=" * 108)


if __name__ == "__main__":
    main()