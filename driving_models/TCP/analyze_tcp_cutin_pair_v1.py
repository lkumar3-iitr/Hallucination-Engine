"""
analyze_tcp_cutin_pair_v1.py

CARLA <-> HE paired behavioral analysis for TCP cut-in scenarios.

This intentionally PRESERVES the earlier HE/NEAT evaluation metrics
and adds additional paired-difference metrics.

The analysis separates:

    1. scenario-event equivalence
    2. control/action equivalence
    3. closed-loop state equivalence
    4. safety/outcome equivalence
    5. short-horizon agreement before closed-loop divergence dominates

Important:
    Image-level PSNR / SSIM / LPIPS are NOT computed here.
    They belong in a same-state paired-rendering evaluator rather than
    independently evolving closed-loop trajectories.
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
    / "TCP"
    / "outputs"
    / "tcp_4320_pair_v1"
    / SCENARIO_ID
)

DEFAULT_CARLA = (
    DEFAULT_ROOT
    / f"{SCENARIO_ID}_carla.csv"
)

DEFAULT_HE = (
    DEFAULT_ROOT
    / f"{SCENARIO_ID}_he.csv"
)

DEFAULT_OUTPUT_DIR = (
    DEFAULT_ROOT
    / "analysis"
)


# ============================================================
# Frozen event definitions
# ============================================================

CUTIN_MOVE_THRESHOLD_M = 0.10

LANE_BOUNDARY_Y_M = 1.75

LANE_CENTER_THRESHOLD_M = 0.20

BRAKE_ANY_THRESHOLD = 1e-3

BRAKE_THRESHOLD = 0.5

BRAKE_CONFIRM_FRAMES = 3


# ============================================================
# Basic helpers
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

    except (
        ValueError,
        TypeError,
    ):

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


def finite_or_none(value):

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(value):
        return None

    return value


def fmt(
    value,
    digits=4,
):

    if value is None:
        return "none"

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return "none"

    if not math.isfinite(value):
        return "none"

    return f"{value:.{digits}f}"


# ============================================================
# Scenario-event detection
# ============================================================

def find_cutin_start(rows):

    initial_y = as_float(
        rows[0],
        "actor_y_sg_m",
    )

    for i, row in enumerate(rows):

        y = as_float(
            row,
            "actor_y_sg_m",
        )

        if (
            np.isfinite(y)
            and
            abs(
                y
                -
                initial_y
            )
            >=
            CUTIN_MOVE_THRESHOLD_M
        ):

            return i

    return None


def find_lane_boundary_crossing(rows):

    for i, row in enumerate(rows):

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

    for i, row in enumerate(rows):

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


def first_brake(
    rows,
    start_idx,
    threshold,
):

    if start_idx is None:
        return None

    for i in range(
        start_idx,
        len(rows),
    ):

        value = as_float(
            rows[i],
            "tcp_brake",
            0.0,
        )

        if (
            np.isfinite(value)
            and
            value >= threshold
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
        -
        BRAKE_CONFIRM_FRAMES
        + 1,
    ):

        window = rows[
            i:
            i + BRAKE_CONFIRM_FRAMES
        ]

        values = [
            as_float(
                row,
                "tcp_brake",
                0.0,
            )
            for row
            in window
        ]

        if all(
            np.isfinite(v)
            and
            v >= BRAKE_THRESHOLD
            for v
            in values
        ):

            return i

    return None


# ============================================================
# Condition-level metrics
# ============================================================

def compute_condition_metrics(rows):

    cutin_idx = find_cutin_start(
        rows
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

    first_any_brake_idx = (
        first_brake(
            rows,
            cutin_idx,
            BRAKE_ANY_THRESHOLD,
        )
    )

    first_strong_brake_idx = (
        first_brake(
            rows,
            cutin_idx,
            BRAKE_THRESHOLD,
        )
    )

    sustained_brake_idx = (
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

    gaps = np.asarray(
        [
            as_float(
                row,
                "bumper_gap_m",
            )
            for row
            in rows
        ],
        dtype=np.float64,
    )

    ttc = np.asarray(
        [
            as_float(
                row,
                "ttc_s",
            )
            for row
            in rows
        ],
        dtype=np.float64,
    )

    speeds = np.asarray(
        [
            as_float(
                row,
                "ego_speed_mps",
            )
            for row
            in rows
        ],
        dtype=np.float64,
    )

    route_deviation = np.asarray(
        [
            as_float(
                row,
                "route_deviation_m",
            )
            for row
            in rows
        ],
        dtype=np.float64,
    )

    finite_gaps = gaps[
        np.isfinite(gaps)
    ]

    positive_finite_ttc = ttc[
        np.isfinite(ttc)
        &
        (ttc >= 0.0)
    ]

    finite_route_dev = (
        route_deviation[
            np.isfinite(
                route_deviation
            )
        ]
    )

    collision = any(
        as_bool(
            row,
            "virtual_collision",
        )
        for row
        in rows
    )

    if boundary_idx is not None:

        post_boundary_gap = gaps[
            boundary_idx:
        ]

        post_boundary_gap = (
            post_boundary_gap[
                np.isfinite(
                    post_boundary_gap
                )
            ]
        )

    else:

        post_boundary_gap = (
            np.asarray(
                [],
                dtype=np.float64,
            )
        )

    return {

        "frame_count":
            len(rows),

        "cutin_idx":
            cutin_idx,

        "boundary_idx":
            boundary_idx,

        "center_idx":
            center_idx,

        "first_any_brake_idx":
            first_any_brake_idx,

        "first_strong_brake_idx":
            first_strong_brake_idx,

        "sustained_brake_idx":
            sustained_brake_idx,

        # ----------------------------------------------------
        # Cut-in start
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Lane boundary
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Lane center
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Brake events
        # ----------------------------------------------------

        "first_any_brake_s":
            event_value(
                first_any_brake_idx,
                "t_s",
            ),

        "first_any_brake_gap_m":
            event_value(
                first_any_brake_idx,
                "bumper_gap_m",
            ),

        "first_strong_brake_s":
            event_value(
                first_strong_brake_idx,
                "t_s",
            ),

        "first_strong_brake_gap_m":
            event_value(
                first_strong_brake_idx,
                "bumper_gap_m",
            ),

        "sustained_brake_s":
            event_value(
                sustained_brake_idx,
                "t_s",
            ),

        "sustained_brake_gap_m":
            event_value(
                sustained_brake_idx,
                "bumper_gap_m",
            ),

        # ----------------------------------------------------
        # Safety / outcome
        # ----------------------------------------------------

        "minimum_gap_m":
            (
                float(
                    np.min(
                        finite_gaps
                    )
                )
                if len(
                    finite_gaps
                )
                else float("nan")
            ),

        "minimum_gap_after_boundary_m":
            (
                float(
                    np.min(
                        post_boundary_gap
                    )
                )
                if len(
                    post_boundary_gap
                )
                else float("nan")
            ),

        "minimum_ttc_s":
            (
                float(
                    np.min(
                        positive_finite_ttc
                    )
                )
                if len(
                    positive_finite_ttc
                )
                else float("nan")
            ),

        "maximum_route_deviation_m":
            (
                float(
                    np.max(
                        finite_route_dev
                    )
                )
                if len(
                    finite_route_dev
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
            (
                float(
                    speeds[-1]
                )
                if len(
                    speeds
                )
                else float("nan")
            ),

        "collision":
            bool(
                collision
            ),
    }


# ============================================================
# Numerical paired metric
# ============================================================

def paired_numeric_metric(
    carla_rows,
    he_rows,
    key,
    start_idx=0,
    end_idx=None,
):

    n = min(
        len(carla_rows),
        len(he_rows),
    )

    start_idx = max(
        0,
        int(start_idx),
    )

    if end_idx is None:

        end_idx = n

    end_idx = min(
        n,
        int(end_idx),
    )

    carla_values = []
    he_values = []

    for i in range(
        start_idx,
        end_idx,
    ):

        c = as_float(
            carla_rows[i],
            key,
        )

        h = as_float(
            he_rows[i],
            key,
        )

        if (
            np.isfinite(c)
            and
            np.isfinite(h)
        ):

            carla_values.append(c)
            he_values.append(h)

    if not carla_values:

        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "p95_abs": None,
            "max_abs": None,
            "bias_he_minus_carla": None,
            "correlation": None,
        }

    c = np.asarray(
        carla_values,
        dtype=np.float64,
    )

    h = np.asarray(
        he_values,
        dtype=np.float64,
    )

    diff = h - c

    abs_diff = np.abs(
        diff
    )

    if (
        len(c) >= 2
        and
        np.std(c) > 1e-12
        and
        np.std(h) > 1e-12
    ):

        corr = float(
            np.corrcoef(
                c,
                h,
            )[0, 1]
        )

    else:

        corr = float("nan")

    return {

        "n":
            int(
                len(diff)
            ),

        "mae":
            float(
                np.mean(
                    abs_diff
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        diff ** 2
                    )
                )
            ),

        "p95_abs":
            float(
                np.percentile(
                    abs_diff,
                    95,
                )
            ),

        "max_abs":
            float(
                np.max(
                    abs_diff
                )
            ),

        "bias_he_minus_carla":
            float(
                np.mean(
                    diff
                )
            ),

        "correlation":
            corr,
    }


# ============================================================
# Brake-decision agreement
# ============================================================

def brake_agreement(
    carla_rows,
    he_rows,
    start_idx=0,
    end_idx=None,
):

    n = min(
        len(carla_rows),
        len(he_rows),
    )

    if end_idx is None:
        end_idx = n

    end_idx = min(
        int(end_idx),
        n,
    )

    agree = 0
    disagree = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(
        int(start_idx),
        end_idx,
    ):

        c = (
            as_float(
                carla_rows[i],
                "tcp_brake",
                0.0,
            )
            >=
            BRAKE_THRESHOLD
        )

        h = (
            as_float(
                he_rows[i],
                "tcp_brake",
                0.0,
            )
            >=
            BRAKE_THRESHOLD
        )

        if c == h:
            agree += 1
        else:
            disagree += 1

        if c and h:
            tp += 1

        elif (
            not c
            and
            not h
        ):
            tn += 1

        elif (
            not c
            and
            h
        ):
            fp += 1

        else:
            fn += 1

    total = (
        agree
        +
        disagree
    )

    return {

        "threshold":
            BRAKE_THRESHOLD,

        "n":
            total,

        "agreement_fraction":
            (
                agree / total
                if total
                else None
            ),

        "disagreement_fraction":
            (
                disagree / total
                if total
                else None
            ),

        "both_brake":
            tp,

        "both_no_brake":
            tn,

        "he_only_brake":
            fp,

        "carla_only_brake":
            fn,
    }


# ============================================================
# Ego trajectory divergence
# ============================================================

def trajectory_divergence(
    carla_rows,
    he_rows,
    start_idx=0,
    end_idx=None,
):

    n = min(
        len(carla_rows),
        len(he_rows),
    )

    if end_idx is None:
        end_idx = n

    end_idx = min(
        n,
        int(end_idx),
    )

    values = []

    for i in range(
        int(start_idx),
        end_idx,
    ):

        cx = as_float(
            carla_rows[i],
            "ego_x",
        )

        cy = as_float(
            carla_rows[i],
            "ego_y",
        )

        hx = as_float(
            he_rows[i],
            "ego_x",
        )

        hy = as_float(
            he_rows[i],
            "ego_y",
        )

        if all(
            np.isfinite(v)
            for v
            in (
                cx,
                cy,
                hx,
                hy,
            )
        ):

            values.append(
                math.hypot(
                    hx - cx,
                    hy - cy,
                )
            )

    if not values:

        return {
            "n": 0,
            "mean_m": None,
            "p95_m": None,
            "max_m": None,
            "final_m": None,
        }

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {

        "n":
            int(
                len(values)
            ),

        "mean_m":
            float(
                np.mean(
                    values
                )
            ),

        "p95_m":
            float(
                np.percentile(
                    values,
                    95,
                )
            ),

        "max_m":
            float(
                np.max(
                    values
                )
            ),

        "final_m":
            float(
                values[-1]
            ),
    }


# ============================================================
# Analysis windows
# ============================================================

def compute_window(
    carla_rows,
    he_rows,
    start_idx,
    end_idx,
):

    keys = [
        "tcp_steer",
        "tcp_throttle",
        "tcp_brake",
        "ego_speed_mps",
        "bumper_gap_m",
        "route_deviation_m",
        "desired_speed",
        "pred_speed",
    ]

    metrics = {}

    for key in keys:

        metrics[key] = (
            paired_numeric_metric(
                carla_rows,
                he_rows,
                key,
                start_idx,
                end_idx,
            )
        )

    metrics[
        "brake_decision"
    ] = (
        brake_agreement(
            carla_rows,
            he_rows,
            start_idx,
            end_idx,
        )
    )

    metrics[
        "ego_xy_trajectory"
    ] = (
        trajectory_divergence(
            carla_rows,
            he_rows,
            start_idx,
            end_idx,
        )
    )

    return metrics


# ============================================================
# JSON cleanup
# ============================================================

def json_safe(value):

    if isinstance(
        value,
        dict,
    ):

        return {
            key:
                json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            json_safe(item)
            for item
            in value
        ]

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(
        value,
        np.integer,
    ):

        return int(value)

    return value


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
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    args = parser.parse_args()

    carla_path = Path(
        args.carla
    ).resolve()

    he_path = Path(
        args.he
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    carla_rows = load_csv(
        carla_path
    )

    he_rows = load_csv(
        he_path
    )

    n = min(
        len(carla_rows),
        len(he_rows),
    )

    carla = (
        compute_condition_metrics(
            carla_rows
        )
    )

    he = (
        compute_condition_metrics(
            he_rows
        )
    )

    # Shared scenario trajectory means the event
    # indices should be identical.
    cutin_idx = (
        carla[
            "cutin_idx"
        ]
        if carla[
            "cutin_idx"
        ]
        is not None
        else 0
    )

    boundary_idx = (
        carla[
            "boundary_idx"
        ]
        if carla[
            "boundary_idx"
        ]
        is not None
        else cutin_idx
    )

    windows = {

        "frames_0_20":
            compute_window(
                carla_rows,
                he_rows,
                0,
                min(
                    21,
                    n,
                ),
            ),

        "frames_0_40":
            compute_window(
                carla_rows,
                he_rows,
                0,
                min(
                    41,
                    n,
                ),
            ),

        "frames_0_80":
            compute_window(
                carla_rows,
                he_rows,
                0,
                min(
                    81,
                    n,
                ),
            ),

        "full":
            compute_window(
                carla_rows,
                he_rows,
                0,
                n,
            ),

        "post_cutin":
            compute_window(
                carla_rows,
                he_rows,
                cutin_idx,
                n,
            ),

        "post_boundary":
            compute_window(
                carla_rows,
                he_rows,
                boundary_idx,
                n,
            ),
    }

    delta_keys = [
        "cutin_start_s",
        "cutin_gap_m",
        "cutin_ego_speed_mps",

        "boundary_time_s",
        "boundary_gap_m",
        "boundary_ego_speed_mps",

        "center_time_s",
        "center_gap_m",
        "center_ego_speed_mps",

        "first_any_brake_s",
        "first_any_brake_gap_m",

        "first_strong_brake_s",
        "first_strong_brake_gap_m",

        "sustained_brake_s",
        "sustained_brake_gap_m",

        "minimum_gap_m",
        "minimum_gap_after_boundary_m",
        "minimum_ttc_s",
        "maximum_route_deviation_m",

        "final_gap_m",
        "final_ego_speed_mps",
    ]

    deltas = {}

    for key in delta_keys:

        c = finite_or_none(
            carla.get(
                key
            )
        )

        h = finite_or_none(
            he.get(
                key
            )
        )

        if (
            c is None
            or
            h is None
        ):

            deltas[key] = None

        else:

            deltas[key] = (
                h - c
            )

    summary = {

        "schema":
            "tcp_carla_he_pair_analysis_v1",

        "scenario_id":
            SCENARIO_ID,

        "carla_csv":
            str(
                carla_path
            ),

        "he_csv":
            str(
                he_path
            ),

        "frames_compared":
            n,

        "definitions": {

            "cutin_move_threshold_m":
                CUTIN_MOVE_THRESHOLD_M,

            "lane_boundary_y_m":
                LANE_BOUNDARY_Y_M,

            "lane_center_threshold_m":
                LANE_CENTER_THRESHOLD_M,

            "brake_any_threshold":
                BRAKE_ANY_THRESHOLD,

            "brake_threshold":
                BRAKE_THRESHOLD,

            "brake_confirm_frames":
                BRAKE_CONFIRM_FRAMES,
        },

        "carla":
            carla,

        "he":
            he,

        "he_minus_carla":
            deltas,

        "collision_outcome_agreement":
            (
                carla[
                    "collision"
                ]
                ==
                he[
                    "collision"
                ]
            ),

        "paired_windows":
            windows,
    }

    summary = json_safe(
        summary
    )

    output_path = (
        output_dir
        / "tcp_cutin_pair_summary.json"
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
    # Console report
    # ========================================================

    print()
    print("=" * 88)
    print(
        "TCP CARLA <-> HE CUT-IN PAIR ANALYSIS V1"
    )
    print("=" * 88)

    print(
        "Frames compared:",
        n,
    )

    print()

    print("Scenario events")
    print(
        "  cut-in start        :",
        fmt(
            carla[
                "cutin_start_s"
            ]
        ),
        "/",
        fmt(
            he[
                "cutin_start_s"
            ]
        ),
        "s",
    )

    print(
        "  lane boundary       :",
        fmt(
            carla[
                "boundary_time_s"
            ]
        ),
        "/",
        fmt(
            he[
                "boundary_time_s"
            ]
        ),
        "s",
    )

    print(
        "  lane center         :",
        fmt(
            carla[
                "center_time_s"
            ]
        ),
        "/",
        fmt(
            he[
                "center_time_s"
            ]
        ),
        "s",
    )

    print()
    print("Brake response")

    print(
        "  first any brake     :",
        fmt(
            carla[
                "first_any_brake_s"
            ]
        ),
        "/",
        fmt(
            he[
                "first_any_brake_s"
            ]
        ),
        "s",
    )

    print(
        "  first strong brake  :",
        fmt(
            carla[
                "first_strong_brake_s"
            ]
        ),
        "/",
        fmt(
            he[
                "first_strong_brake_s"
            ]
        ),
        "s",
    )

    print(
        "  sustained brake     :",
        fmt(
            carla[
                "sustained_brake_s"
            ]
        ),
        "/",
        fmt(
            he[
                "sustained_brake_s"
            ]
        ),
        "s",
    )

    print()
    print("Safety / outcome")

    print(
        "  collision           :",
        carla[
            "collision"
        ],
        "/",
        he[
            "collision"
        ],
    )

    print(
        "  collision agreement :",
        summary[
            "collision_outcome_agreement"
        ],
    )

    print(
        "  minimum gap         :",
        fmt(
            carla[
                "minimum_gap_m"
            ]
        ),
        "/",
        fmt(
            he[
                "minimum_gap_m"
            ]
        ),
        "m",
    )

    print(
        "  min-gap delta HE-C  :",
        fmt(
            deltas[
                "minimum_gap_m"
            ]
        ),
        "m",
    )

    print(
        "  final gap           :",
        fmt(
            carla[
                "final_gap_m"
            ]
        ),
        "/",
        fmt(
            he[
                "final_gap_m"
            ]
        ),
        "m",
    )

    print()

    for name in (
        "frames_0_20",
        "frames_0_40",
        "frames_0_80",
        "full",
        "post_boundary",
    ):

        window = windows[
            name
        ]

        print(
            name
        )

        print(
            "  steer MAE/P95       :",
            fmt(
                window[
                    "tcp_steer"
                ][
                    "mae"
                ]
            ),
            "/",
            fmt(
                window[
                    "tcp_steer"
                ][
                    "p95_abs"
                ]
            ),
        )

        print(
            "  throttle MAE/P95    :",
            fmt(
                window[
                    "tcp_throttle"
                ][
                    "mae"
                ]
            ),
            "/",
            fmt(
                window[
                    "tcp_throttle"
                ][
                    "p95_abs"
                ]
            ),
        )

        print(
            "  brake MAE/P95       :",
            fmt(
                window[
                    "tcp_brake"
                ][
                    "mae"
                ]
            ),
            "/",
            fmt(
                window[
                    "tcp_brake"
                ][
                    "p95_abs"
                ]
            ),
        )

        print(
            "  speed MAE/P95       :",
            fmt(
                window[
                    "ego_speed_mps"
                ][
                    "mae"
                ]
            ),
            "/",
            fmt(
                window[
                    "ego_speed_mps"
                ][
                    "p95_abs"
                ]
            ),
            "m/s",
        )

        print(
            "  gap MAE/P95         :",
            fmt(
                window[
                    "bumper_gap_m"
                ][
                    "mae"
                ]
            ),
            "/",
            fmt(
                window[
                    "bumper_gap_m"
                ][
                    "p95_abs"
                ]
            ),
            "m",
        )

        print(
            "  brake agreement     :",
            fmt(
                window[
                    "brake_decision"
                ][
                    "agreement_fraction"
                ]
            ),
        )

        print(
            "  ego XY mean/P95     :",
            fmt(
                window[
                    "ego_xy_trajectory"
                ][
                    "mean_m"
                ]
            ),
            "/",
            fmt(
                window[
                    "ego_xy_trajectory"
                ][
                    "p95_m"
                ]
            ),
            "m",
        )

        print()

    print(
        "Saved:",
        output_path,
    )

    print("=" * 88)


if __name__ == "__main__":
    main()