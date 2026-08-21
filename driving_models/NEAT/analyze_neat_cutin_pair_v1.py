"""
analyze_neat_cutin_pair_v1.py

Single-pair CARLA-vs-HE analysis for NEAT on tcp_cutin_001.

Event definitions are intentionally kept consistent with the
frozen TCP cut-in evaluator:

    cut-in start:
        actor lateral displacement >= 0.10 m

    lane-boundary crossing:
        actor_y_sg_m <= 1.75 m

    ego-lane-center arrival:
        abs(actor_y_sg_m) <= 0.20 m

    sustained brake:
        model_brake >= 0.5 for 3 consecutive frames

The scenario geometry is shared across CARLA and HE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]

SCENARIO_ID = "tcp_cutin_001"

DEFAULT_ROOT = (
    HE_ROOT
    / "driving_models"
    / "NEAT"
    / "outputs"
    / "neat_he_pair_v1"
    / SCENARIO_ID
)

DEFAULT_CARLA = (
    DEFAULT_ROOT
    / "carla"
    / f"{SCENARIO_ID}_carla.csv"
)

DEFAULT_HE = (
    DEFAULT_ROOT
    / "he"
    / f"{SCENARIO_ID}_he.csv"
)

DEFAULT_OUTPUT = (
    DEFAULT_ROOT
    / "analysis"
    / f"{SCENARIO_ID}_pair_summary.json"
)


# ============================================================
# Frozen cut-in definitions
# ============================================================

CUTIN_MOVE_THRESHOLD_M = 0.10

LANE_BOUNDARY_Y_M = 1.75

LANE_CENTER_THRESHOLD_M = 0.20

BRAKE_THRESHOLD = 0.5

BRAKE_CONFIRM_FRAMES = 3


# ============================================================
# Helpers
# ============================================================

def load_csv(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as fp:

        return list(
            csv.DictReader(fp)
        )


def as_float(
    row,
    key,
    default=float("nan"),
):

    value = str(
        row.get(
            key,
            "",
        )
    ).strip()

    if value == "":

        return default

    try:

        return float(value)

    except ValueError:

        return default


def as_bool(
    row,
    key,
):

    value = str(
        row.get(
            key,
            "",
        )
    ).strip().lower()

    return value in (
        "1",
        "true",
        "yes",
    )


def fmt(
    value,
    digits=3,
):

    if (
        value is None
        or
        not math.isfinite(
            float(value)
        )
    ):

        return "none"

    return (
        f"{float(value):.{digits}f}"
    )


# ============================================================
# Scenario event detection
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
            >=
            CUTIN_MOVE_THRESHOLD_M
        ):

            return i

    return None


def find_lane_boundary_crossing(rows):

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


def find_lane_center_arrival(rows):

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
            <=
            LANE_CENTER_THRESHOLD_M
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
                "model_brake",
                0.0,
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
# Condition metrics
# ============================================================

def compute_metrics(rows):

    cutin_idx = (
        find_cutin_start(rows)
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

    brake_idx = (
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

    # --------------------------------------------------------
    # Gaps
    # --------------------------------------------------------

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

    if boundary_idx is not None:

        post_boundary = gaps[
            boundary_idx:
        ]

        post_boundary = post_boundary[
            np.isfinite(
                post_boundary
            )
        ]

    else:

        post_boundary = np.array(
            [],
            dtype=float,
        )

    finite_gaps = gaps[
        np.isfinite(
            gaps
        )
    ]

    # --------------------------------------------------------
    # Collision
    # --------------------------------------------------------

    collision = any(
        as_bool(
            row,
            "virtual_collision",
        )
        for row in rows
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {

        "rows":
            rows,

        "cutin_idx":
            cutin_idx,

        "boundary_idx":
            boundary_idx,

        "center_idx":
            center_idx,

        "brake_idx":
            brake_idx,

        # Cut-in starts moving
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

        "cutin_ego_speed_mps":
            event_value(
                cutin_idx,
                "ego_speed_mps",
            ),

        # Lane boundary
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

        # Ego lane center
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

        "center_ego_speed_mps":
            event_value(
                center_idx,
                "ego_speed_mps",
            ),

        # Sustained braking
        "sustained_brake_s":
            event_value(
                brake_idx,
                "t_s",
            ),

        "sustained_brake_gap_m":
            event_value(
                brake_idx,
                "bumper_gap_m",
            ),

        # Outcome
        "minimum_gap_after_boundary_m":
            (
                float(
                    np.min(
                        post_boundary
                    )
                )
                if len(
                    post_boundary
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

        "final_ego_speed_mps":
            as_float(
                rows[-1],
                "ego_speed_mps",
            ),

        "collision":
            bool(
                collision
            ),
    }


# ============================================================
# Paired CARLA ↔ HE differences
# ============================================================

def aligned_pair_metrics(
    carla,
    he,
):

    c_rows = carla[
        "rows"
    ]

    h_rows = he[
        "rows"
    ]

    n = min(
        len(c_rows),
        len(h_rows),
    )

    # Actor trajectory is shared, so use the
    # physical boundary-crossing index as alignment point.
    boundary_idx = carla[
        "boundary_idx"
    ]

    if boundary_idx is None:

        boundary_idx = 0

    boundary_idx = min(
        boundary_idx,
        n - 1,
    )

    def mae_after(key):

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

    # --------------------------------------------------------
    # Binary brake disagreement
    # --------------------------------------------------------

    brake_disagreement = []

    for i in range(
        boundary_idx,
        n,
    ):

        c_brake = (
            as_float(
                c_rows[i],
                "model_brake",
                0.0,
            )
            >=
            BRAKE_THRESHOLD
        )

        h_brake = (
            as_float(
                h_rows[i],
                "model_brake",
                0.0,
            )
            >=
            BRAKE_THRESHOLD
        )

        brake_disagreement.append(
            c_brake
            !=
            h_brake
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
                "model_throttle"
            ),

        "post_boundary_brake_mae":
            mae_after(
                "model_brake"
            ),

        "post_boundary_steer_mae":
            mae_after(
                "model_steer"
            ),

        "post_boundary_brake_disagreement_fraction":
            (
                float(
                    np.mean(
                        brake_disagreement
                    )
                )
                if brake_disagreement
                else float("nan")
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--carla",
        default=str(
            DEFAULT_CARLA
        ),
    )

    parser.add_argument(
        "--he",
        default=str(
            DEFAULT_HE
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    args = parser.parse_args()

    carla_rows = load_csv(
        Path(
            args.carla
        )
    )

    he_rows = load_csv(
        Path(
            args.he
        )
    )

    carla = compute_metrics(
        carla_rows
    )

    he = compute_metrics(
        he_rows
    )

    paired = aligned_pair_metrics(
        carla,
        he,
    )

    # ========================================================
    # Pair deltas
    # ========================================================

    def delta(key):

        c = carla[
            key
        ]

        h = he[
            key
        ]

        if (
            not np.isfinite(c)
            or
            not np.isfinite(h)
        ):

            return float("nan")

        return float(
            h - c
        )

    delta_keys = [
        "cutin_gap_m",
        "cutin_ego_speed_mps",
        "boundary_gap_m",
        "boundary_ego_speed_mps",
        "center_gap_m",
        "center_ego_speed_mps",
        "sustained_brake_s",
        "sustained_brake_gap_m",
        "minimum_gap_after_boundary_m",
        "final_gap_m",
        "final_ego_speed_mps",
    ]

    deltas = {
        key:
            delta(key)
        for key in delta_keys
    }

    # ========================================================
    # Save JSON
    # ========================================================

    def clean_condition(data):

        return {
            key: value
            for key, value
            in data.items()
            if key != "rows"
        }

    summary = {

        "scenario_id":
            SCENARIO_ID,

        "definitions": {

            "cutin_move_threshold_m":
                CUTIN_MOVE_THRESHOLD_M,

            "lane_boundary_y_m":
                LANE_BOUNDARY_Y_M,

            "lane_center_threshold_m":
                LANE_CENTER_THRESHOLD_M,

            "brake_threshold":
                BRAKE_THRESHOLD,

            "brake_confirm_frames":
                BRAKE_CONFIRM_FRAMES,
        },

        "carla":
            clean_condition(
                carla
            ),

        "he":
            clean_condition(
                he
            ),

        "delta_he_minus_carla":
            deltas,

        "paired_time_series":
            paired,
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            summary,
            fp,
            indent=2,
        )

    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 94)
    print(
        "NEAT CUT-IN CARLA <-> HE"
    )
    print("=" * 94)

    print(
        f"Cut-in start threshold : "
        f"{CUTIN_MOVE_THRESHOLD_M:.2f} m"
    )

    print(
        f"Lane boundary          : "
        f"y <= {LANE_BOUNDARY_Y_M:.2f} m"
    )

    print(
        f"Lane center            : "
        f"|y| <= {LANE_CENTER_THRESHOLD_M:.2f} m"
    )

    print(
        f"Sustained brake        : "
        f">= {BRAKE_THRESHOLD:.2f} "
        f"for {BRAKE_CONFIRM_FRAMES} frames"
    )

    print()

    print(
        "%-32s %12s %12s %12s"
        %
        (
            "Metric",
            "CARLA",
            "HE",
            "HE-CARLA",
        )
    )

    print("-" * 94)

    def print_row(
        label,
        key,
    ):

        print(
            "%-32s %12s %12s %12s"
            %
            (
                label,

                fmt(
                    carla[key]
                ),

                fmt(
                    he[key]
                ),

                fmt(
                    deltas.get(
                        key,
                        float("nan"),
                    )
                ),
            )
        )

    print_row(
        "Gap at cut-in start (m)",
        "cutin_gap_m",
    )

    print_row(
        "Ego speed at cut-in start",
        "cutin_ego_speed_mps",
    )

    print_row(
        "Gap at lane boundary (m)",
        "boundary_gap_m",
    )

    print_row(
        "Ego speed at boundary",
        "boundary_ego_speed_mps",
    )

    print_row(
        "Gap at lane center (m)",
        "center_gap_m",
    )

    print_row(
        "Ego speed at lane center",
        "center_ego_speed_mps",
    )

    print_row(
        "Sustained brake time (s)",
        "sustained_brake_s",
    )

    print_row(
        "Gap at sustained brake (m)",
        "sustained_brake_gap_m",
    )

    print_row(
        "Minimum gap after boundary",
        "minimum_gap_after_boundary_m",
    )

    print_row(
        "Final gap (m)",
        "final_gap_m",
    )

    print_row(
        "Final ego speed (m/s)",
        "final_ego_speed_mps",
    )

    print()
    print(
        "Physical scenario event times"
    )
    print("-" * 94)

    print(
        "Cut-in start:",
        "CARLA",
        fmt(
            carla[
                "cutin_start_s"
            ]
        ),
        "HE",
        fmt(
            he[
                "cutin_start_s"
            ]
        ),
    )

    print(
        "Boundary crossing:",
        "CARLA",
        fmt(
            carla[
                "boundary_time_s"
            ]
        ),
        "HE",
        fmt(
            he[
                "boundary_time_s"
            ]
        ),
    )

    print(
        "Lane center arrival:",
        "CARLA",
        fmt(
            carla[
                "center_time_s"
            ]
        ),
        "HE",
        fmt(
            he[
                "center_time_s"
            ]
        ),
    )

    print()
    print(
        "Post-boundary CARLA <-> HE differences"
    )
    print("-" * 94)

    print(
        "Speed MAE:",
        f"{paired['post_boundary_speed_mae_mps']:.3f} m/s",
    )

    print(
        "Gap MAE:",
        f"{paired['post_boundary_gap_mae_m']:.3f} m",
    )

    print(
        "Throttle MAE:",
        f"{paired['post_boundary_throttle_mae']:.4f}",
    )

    print(
        "Brake MAE:",
        f"{paired['post_boundary_brake_mae']:.4f}",
    )

    print(
        "Steer MAE:",
        f"{paired['post_boundary_steer_mae']:.4f}",
    )

    print(
        "Brake disagreement:",
        f"{100.0 * paired['post_boundary_brake_disagreement_fraction']:.2f}%",
    )

    print()
    print(
        "Collision CARLA:",
        carla[
            "collision"
        ],
    )

    print(
        "Collision HE:",
        he[
            "collision"
        ],
    )

    print()

    print(
        "Summary:",
        output_path,
    )

    print("=" * 94)


if __name__ == "__main__":

    main()