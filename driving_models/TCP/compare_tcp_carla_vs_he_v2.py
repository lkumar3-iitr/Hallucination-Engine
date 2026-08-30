"""
compare_tcp_carla_vs_he_v2.py

Frame-aligned behavioral comparison:

    TCP + CARLA-rendered adversary
                vs
    TCP + HE-rendered adversary

This does NOT compare image similarity.

It compares:
    - TCP steer / throttle / brake
    - brake onset
    - ego speed
    - ego trajectory divergence
    - bumper gap
    - TTC
    - stopping time
    - minimum/final gap
    - virtual collision
    - scenario trajectory consistency

Outputs:
    - console summary
    - summary JSON
    - per-frame comparison CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PAIR_DIR = (
    ROOT
    / "driving_models"
    / "TCP"
    / "outputs"
    / "tcp_he_pair_v1"
    / "tcp_lead_brake_001"
)

CARLA_CSV = (
    PAIR_DIR
    / "tcp_lead_brake_001_carla.csv"
)

HE_CSV = (
    PAIR_DIR
    / "tcp_lead_brake_001_he.csv"
)

OUTPUT_CSV = (
    PAIR_DIR
    / "tcp_lead_brake_001_comparison.csv"
)

OUTPUT_JSON = (
    PAIR_DIR
    / "tcp_lead_brake_001_summary.json"
)

EVENT_START_S = 4.0

BRAKE_THRESHOLD = 0.5

STOP_SPEED_MPS = 0.10

STOP_CONFIRM_FRAMES = 5
BRAKE_CONFIRM_FRAMES = 3

# ============================================================
# Loading
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing CSV:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def as_float(row, key, default=float("nan")):
    value = row.get(
        key,
        "",
    )

    if value is None:
        return default

    value = str(value).strip()

    if value == "":
        return default

    try:
        return float(value)
    except ValueError:
        return default


def as_bool(row, key):
    value = str(
        row.get(
            key,
            "0",
        )
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
    }


def preferred_float(
    row,
    preferred_key,
    fallback_key,
):
    """
    Prefer a newer/map-aware metric when available.

    Fall back to the legacy metric for older experiments.
    """

    value = as_float(
        row,
        preferred_key,
    )

    if np.isfinite(
        value
    ):
        return value

    return as_float(
        row,
        fallback_key,
    )


def safety_gap_m(row):
    return preferred_float(
        row,
        "route_bumper_gap_m",
        "bumper_gap_m",
    )


def safety_ttc_s(row):
    return preferred_float(
        row,
        "route_ttc_s",
        "ttc_s",
    )


def safety_collision(row):
    route_value = str(
        row.get(
            "route_virtual_collision",
            "",
        )
    ).strip()

    if route_value != "":
        return as_bool(
            row,
            "route_virtual_collision",
        )

    return as_bool(
        row,
        "virtual_collision",
    )


def index_rows(rows):
    return {
        int(
            row["probe_idx"]
        ): row
        for row in rows
    }


# ============================================================
# Metrics
# ============================================================

def mae(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    if not np.any(valid):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                a[valid]
                - b[valid]
            )
        )
    )


def rmse(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    if not np.any(valid):
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                (
                    a[valid]
                    - b[valid]
                ) ** 2
            )
        )
    )


def max_abs(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    if not np.any(valid):
        return float("nan")

    return float(
        np.max(
            np.abs(
                a[valid]
                - b[valid]
            )
        )
    )


def first_brake(rows):
    for row in rows:

        t = as_float(
            row,
            "t_s",
        )

        if t < EVENT_START_S:
            continue

        brake = as_float(
            row,
            "tcp_brake",
        )

        if brake >= BRAKE_THRESHOLD:
            return {
                "frame":
                    int(
                        row[
                            "probe_idx"
                        ]
                    ),

                "t_s":
                    t,

                "gap_m":
                    safety_gap_m(
                        row
                    ),

                "ego_speed_mps":
                    as_float(
                        row,
                        "ego_speed_mps",
                    ),

                "brake":
                    brake,
            }

    return None

def first_sustained_brake(rows):
    """
    First strong-brake onset sustained for at least
    BRAKE_CONFIRM_FRAMES consecutive frames.
    """

    for i in range(
        len(rows)
        - BRAKE_CONFIRM_FRAMES
        + 1
    ):

        t = as_float(
            rows[i],
            "t_s",
        )

        if t < EVENT_START_S:
            continue

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
            np.isfinite(b)
            and b >= BRAKE_THRESHOLD
            for b in brakes
        ):
            row = rows[i]

            return {
                "frame":
                    int(
                        row["probe_idx"]
                    ),

                "t_s":
                    t,

                "gap_m":
                    safety_gap_m(
                        row
                    ),

                "ego_speed_mps":
                    as_float(
                        row,
                        "ego_speed_mps",
                    ),

                "brake":
                    as_float(
                        row,
                        "tcp_brake",
                    ),
            }

    return None

def first_stop(rows):
    """
    First time after event at which ego speed stays <= 0.1 m/s
    for STOP_CONFIRM_FRAMES consecutive samples.
    """

    for i in range(
        len(rows)
        - STOP_CONFIRM_FRAMES
        + 1
    ):

        t = as_float(
            rows[i],
            "t_s",
        )

        if t < EVENT_START_S:
            continue

        window = (
            rows[
                i:
                i + STOP_CONFIRM_FRAMES
            ]
        )

        speeds = [
            as_float(
                row,
                "ego_speed_mps",
            )
            for row in window
        ]

        if all(
            np.isfinite(v)
            and
            v <= STOP_SPEED_MPS
            for v in speeds
        ):
            row = rows[i]

            return {
                "frame":
                    int(
                        row[
                            "probe_idx"
                        ]
                    ),

                "t_s":
                    t,

                "gap_m":
                    safety_gap_m(
                        row
                    ),
            }

    return None


def min_finite(rows, key):
    values = [
        as_float(
            row,
            key,
        )
        for row in rows
    ]

    values = [
        value
        for value in values
        if np.isfinite(value)
    ]

    if not values:
        return float("nan")

    return float(
        min(values)
    )


def final_finite(rows, key):
    for row in reversed(rows):

        value = as_float(
            row,
            key,
        )

        if np.isfinite(value):
            return value

    return float("nan")


def any_collision(rows):
    return any(
        safety_collision(
            row
        )
        for row in rows
    )

def min_safety_gap(rows):

    values = [
        safety_gap_m(
            row
        )
        for row in rows
    ]

    values = [
        value
        for value in values
        if np.isfinite(
            value
        )
    ]

    if not values:
        return float(
            "nan"
        )

    return float(
        min(
            values
        )
    )


def final_safety_gap(rows):

    for row in reversed(
        rows
    ):

        value = safety_gap_m(
            row
        )

        if np.isfinite(
            value
        ):
            return float(
                value
            )

    return float(
        "nan"
    )


def min_safety_ttc(rows):

    values = [
        safety_ttc_s(
            row
        )
        for row in rows
    ]

    values = [
        value
        for value in values
        if np.isfinite(
            value
        )
    ]

    if not values:
        return float(
            "nan"
        )

    return float(
        min(
            values
        )
    )


def safe_delta(a, b):
    if (
        not np.isfinite(a)
        or not np.isfinite(b)
    ):
        return float("nan")

    return float(
        b - a
    )


# ============================================================
# Main
# ============================================================

def main():

    global EVENT_START_S

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--carla-csv",
        required=True,
    )

    parser.add_argument(
        "--he-csv",
        required=True,
    )

    parser.add_argument(
        "--event-start-s",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional output prefix without extension. "
            "Defaults beside the CARLA CSV."
        ),
    )

    args = parser.parse_args()

    carla_csv = Path(
        args.carla_csv
    ).resolve()

    he_csv = Path(
        args.he_csv
    ).resolve()

    EVENT_START_S = float(
        args.event_start_s
    )

    if args.output_prefix is None:

        carla_stem = (
            carla_csv.stem
        )

        if carla_stem.endswith(
            "_carla"
        ):
            carla_stem = (
                carla_stem[
                    :-len("_carla")
                ]
            )

        output_prefix = (
            carla_csv.parent
            /
            (
                carla_stem
                + "_carla_vs_he_v2"
            )
        )

    else:

        output_prefix = Path(
            args.output_prefix
        ).resolve()

    output_csv = Path(
        str(output_prefix)
        + ".csv"
    )

    output_json = Path(
        str(output_prefix)
        + ".json"
    )

    carla_rows_all = (
        load_csv(
            carla_csv
        )
    )

    he_rows_all = (
        load_csv(
            he_csv
        )
    )

    # ========================================================
    # Frame alignment
    # ========================================================

    carla_index = (
        index_rows(
            carla_rows_all
        )
    )

    he_index = (
        index_rows(
            he_rows_all
        )
    )

    common_frames = sorted(
        set(
            carla_index
        )
        &
        set(
            he_index
        )
    )

    if not common_frames:
        raise RuntimeError(
            "CARLA and HE CSVs have "
            "no matching probe_idx."
        )

    carla_rows = [
        carla_index[i]
        for i in common_frames
    ]

    he_rows = [
        he_index[i]
        for i in common_frames
    ]

    print()
    print("=" * 78)
    print(
        "TCP CARLA vs HE BEHAVIORAL COMPARISON V2"
    )
    print("=" * 78)

    print(
        "CARLA frames :",
        len(
            carla_rows_all
        ),
    )

    print(
        "HE frames    :",
        len(
            he_rows_all
        ),
    )

    print(
        "Matched      :",
        len(
            common_frames
        ),
    )

    print(
        "Event start  :",
        f"{EVENT_START_S:.2f} s",
    )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    def values(rows, key):
        return np.array(
            [
                as_float(
                    row,
                    key,
                )
                for row in rows
            ],
            dtype=np.float64,
        )

    time_c = values(
        carla_rows,
        "t_s",
    )

    time_h = values(
        he_rows,
        "t_s",
    )

    steer_c = values(
        carla_rows,
        "tcp_steer",
    )

    steer_h = values(
        he_rows,
        "tcp_steer",
    )

    throttle_c = values(
        carla_rows,
        "tcp_throttle",
    )

    throttle_h = values(
        he_rows,
        "tcp_throttle",
    )

    brake_c = values(
        carla_rows,
        "tcp_brake",
    )

    brake_h = values(
        he_rows,
        "tcp_brake",
    )

    speed_c = values(
        carla_rows,
        "ego_speed_mps",
    )

    speed_h = values(
        he_rows,
        "ego_speed_mps",
    )

    gap_c = np.array(
        [
            safety_gap_m(
                row
            )
            for row in carla_rows
        ],
        dtype=np.float64,
    )

    gap_h = np.array(
        [
            safety_gap_m(
                row
            )
            for row in he_rows
        ],
        dtype=np.float64,
    )

    ego_x_c = values(
        carla_rows,
        "ego_x",
    )

    ego_x_h = values(
        he_rows,
        "ego_x",
    )

    ego_y_c = values(
        carla_rows,
        "ego_y",
    )

    ego_y_h = values(
        he_rows,
        "ego_y",
    )

    position_error = np.sqrt(
        (
            ego_x_h
            - ego_x_c
        ) ** 2
        +
        (
            ego_y_h
            - ego_y_c
        ) ** 2
    )

    # --------------------------------------------------------
    # Verify scenario physical truth is identical
    # --------------------------------------------------------

    actor_x_c = values(
        carla_rows,
        "actor_x_sg_m",
    )

    actor_x_h = values(
        he_rows,
        "actor_x_sg_m",
    )

    actor_y_c = values(
        carla_rows,
        "actor_y_sg_m",
    )

    actor_y_h = values(
        he_rows,
        "actor_y_sg_m",
    )

    actor_speed_c = values(
        carla_rows,
        "actor_speed_mps",
    )

    actor_speed_h = values(
        he_rows,
        "actor_speed_mps",
    )

    scenario_position_error = (
        np.sqrt(
            (
                actor_x_h
                - actor_x_c
            ) ** 2
            +
            (
                actor_y_h
                - actor_y_c
            ) ** 2
        )
    )

    max_actor_position_error = float(
        np.nanmax(
            scenario_position_error
        )
    )

    max_actor_speed_error = (
        max_abs(
            actor_speed_c,
            actor_speed_h,
        )
    )

    max_time_error = (
        max_abs(
            time_c,
            time_h,
        )
    )

    # --------------------------------------------------------
    # Event/post-event mask
    # --------------------------------------------------------

    post_event = (
        time_c
        >= EVENT_START_S
    )

    # --------------------------------------------------------
    # Event metrics
    # --------------------------------------------------------

    carla_brake = (
        first_brake(
            carla_rows
        )
    )

    he_brake = (
        first_brake(
            he_rows
        )
    )

    carla_sustained_brake = (
        first_sustained_brake(
            carla_rows
        )
    )

    he_sustained_brake = (
        first_sustained_brake(
            he_rows
        )
    )

    carla_stop = (
        first_stop(
            carla_rows
        )
    )

    he_stop = (
        first_stop(
            he_rows
        )
    )

    # --------------------------------------------------------
    # Brake disagreement
    # --------------------------------------------------------

    brake_state_c = (
        brake_c
        >= BRAKE_THRESHOLD
    )

    brake_state_h = (
        brake_h
        >= BRAKE_THRESHOLD
    )

    if np.any(
        post_event
    ):
        brake_disagreement_fraction = float(
            np.mean(
                brake_state_c[
                    post_event
                ]
                !=
                brake_state_h[
                    post_event
                ]
            )
        )
    else:
        brake_disagreement_fraction = (
            float("nan")
        )

    # --------------------------------------------------------
    # First substantial behavioral divergence
    # --------------------------------------------------------

    first_divergence = None

    for j, frame_idx in enumerate(
        common_frames
    ):

        if time_c[j] < EVENT_START_S:
            continue

        brake_diff = abs(
            brake_h[j]
            - brake_c[j]
        )

        speed_diff = abs(
            speed_h[j]
            - speed_c[j]
        )

        pos_diff = (
            position_error[j]
        )

        if (
            brake_diff >= 0.30
            or
            speed_diff >= 0.50
            or
            pos_diff >= 0.50
        ):
            first_divergence = {
                "frame":
                    int(
                        frame_idx
                    ),

                "t_s":
                    float(
                        time_c[j]
                    ),

                "brake_abs_diff":
                    float(
                        brake_diff
                    ),

                "speed_abs_diff_mps":
                    float(
                        speed_diff
                    ),

                "ego_position_error_m":
                    float(
                        pos_diff
                    ),
            }

            break

    # --------------------------------------------------------
    # HE rendering coverage
    # --------------------------------------------------------

    he_rendered = np.array(
        [
            as_bool(
                row,
                "he_rendered",
            )
            for row in he_rows
        ],
        dtype=np.bool_,
    )

    he_render_fraction = float(
        np.mean(
            he_rendered
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "matched_frames":
            len(
                common_frames
            ),

        "event_start_s":
            EVENT_START_S,

        "scenario_consistency": {
            "max_time_error_s":
                max_time_error,

            "max_actor_position_error_m":
                max_actor_position_error,

            "max_actor_speed_error_mps":
                max_actor_speed_error,
        },

        "carla": {
            "first_brake":
                carla_brake,

            "first_stop":
                carla_stop,

            "min_gap_m":
                min_safety_gap(
                    carla_rows
                ),

            "final_gap_m":
                final_safety_gap(
                    carla_rows
                ),

            "min_ttc_s":
                min_safety_ttc(
                    carla_rows
                ),

            "virtual_collision":
                any_collision(
                    carla_rows
                ),
        },

        "he": {
            "first_brake":
                he_brake,

            "first_stop":
                he_stop,

            "min_gap_m":
                min_safety_gap(
                    he_rows
                ),

            "final_gap_m":
                final_safety_gap(
                    he_rows
                ),

            "min_ttc_s":
                min_safety_ttc(
                    he_rows
                ),

            "virtual_collision":
                any_collision(
                    he_rows
                ),

            "rendered_fraction":
                he_render_fraction,
        },

        "difference_he_minus_carla": {
            "first_brake_pulse_delta_s":
            (
                safe_delta(
                    carla_brake["t_s"],
                    he_brake["t_s"],
                )
                if
                carla_brake is not None
                and
                he_brake is not None
                else None
            ),

        "sustained_brake_onset_delta_s":
            (
                safe_delta(
                    carla_sustained_brake["t_s"],
                    he_sustained_brake["t_s"],
                )
                if
                carla_sustained_brake is not None
                and
                he_sustained_brake is not None
                else None
            ),

            "min_gap_delta_m":
                (
                    min_safety_gap(
                        he_rows
                    )
                    -
                    min_safety_gap(
                        carla_rows
                    )
                ),

            "final_gap_delta_m":
                (
                    final_safety_gap(
                        he_rows
                    )
                    -
                    final_safety_gap(
                        carla_rows
                    )
                ),

            "steer_mae":
                mae(
                    steer_c,
                    steer_h,
                ),

            "throttle_mae":
                mae(
                    throttle_c,
                    throttle_h,
                ),

            "brake_mae":
                mae(
                    brake_c,
                    brake_h,
                ),

            "speed_mae_mps":
                mae(
                    speed_c,
                    speed_h,
                ),

            "speed_rmse_mps":
                rmse(
                    speed_c,
                    speed_h,
                ),

            "gap_mae_m":
                mae(
                    gap_c,
                    gap_h,
                ),

            "gap_rmse_m":
                rmse(
                    gap_c,
                    gap_h,
                ),

            "max_ego_position_error_m":
                float(
                    np.nanmax(
                        position_error
                    )
                ),

            "final_ego_position_error_m":
                float(
                    position_error[
                        -1
                    ]
                ),

            "post_event_steer_mae":
                mae(
                    steer_c[
                        post_event
                    ],
                    steer_h[
                        post_event
                    ],
                ),

            "post_event_throttle_mae":
                mae(
                    throttle_c[
                        post_event
                    ],
                    throttle_h[
                        post_event
                    ],
                ),

            "post_event_brake_mae":
                mae(
                    brake_c[
                        post_event
                    ],
                    brake_h[
                        post_event
                    ],
                ),

            "brake_state_disagreement_fraction":
                brake_disagreement_fraction,

            "first_substantial_divergence":
                first_divergence,
        },
    }

    # --------------------------------------------------------
    # Per-frame output
    # --------------------------------------------------------

    fields = [
        "probe_idx",
        "t_s",

        "carla_speed_mps",
        "he_speed_mps",
        "delta_speed_mps",

        "carla_gap_m",
        "he_gap_m",
        "delta_gap_m",

        "carla_steer",
        "he_steer",
        "delta_steer",

        "carla_throttle",
        "he_throttle",
        "delta_throttle",

        "carla_brake",
        "he_brake",
        "delta_brake",

        "ego_position_error_m",

        "carla_virtual_collision",
        "he_virtual_collision",

        "he_rendered",
    ]

    with output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for j, frame_idx in enumerate(
            common_frames
        ):

            writer.writerow({
                "probe_idx":
                    frame_idx,

                "t_s":
                    time_c[j],

                "carla_speed_mps":
                    speed_c[j],

                "he_speed_mps":
                    speed_h[j],

                "delta_speed_mps":
                    speed_h[j]
                    - speed_c[j],

                "carla_gap_m":
                    gap_c[j],

                "he_gap_m":
                    gap_h[j],

                "delta_gap_m":
                    gap_h[j]
                    - gap_c[j],

                "carla_steer":
                    steer_c[j],

                "he_steer":
                    steer_h[j],

                "delta_steer":
                    steer_h[j]
                    - steer_c[j],

                "carla_throttle":
                    throttle_c[j],

                "he_throttle":
                    throttle_h[j],

                "delta_throttle":
                    throttle_h[j]
                    - throttle_c[j],

                "carla_brake":
                    brake_c[j],

                "he_brake":
                    brake_h[j],

                "delta_brake":
                    brake_h[j]
                    - brake_c[j],

                "ego_position_error_m":
                    position_error[j],

                "carla_virtual_collision":
                    int(
                        safety_collision(
                            carla_rows[j]
                        )
                    ),

                "he_virtual_collision":
                    int(
                        safety_collision(
                            he_rows[j]
                        )
                    ),

                "he_rendered":
                    int(
                        he_rendered[j]
                    ),
            })

    with output_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=True,
        )

    # ========================================================
    # Human-readable summary
    # ========================================================

    print()
    print(
        "SCENARIO CONSISTENCY"
    )
    print("-" * 78)

    print(
        "max time mismatch       :",
        f"{max_time_error:.6f} s",
    )

    print(
        "max actor position diff :",
        f"{max_actor_position_error:.6f} m",
    )

    print(
        "max actor speed diff    :",
        f"{max_actor_speed_error:.6f} m/s",
    )

    print()
    print(
        "EVENT OUTCOMES"
    )
    print("-" * 78)

    def print_condition(
        label,
        rows,
        brake_event,
        sustained_brake_event,
        stop_event,
    ):

        print()
        print(
            label
        )

        if brake_event:
            print(
                "  first brake >=0.5 : "
                f"{brake_event['t_s']:.2f} s "
                f"(frame {brake_event['frame']}, "
                f"gap {brake_event['gap_m']:.2f} m)"
            )
        else:
            print(
                "  first brake >=0.5 : none"
            )
        if sustained_brake_event:
            print(
                "  sustained brake    : "
                f"{sustained_brake_event['t_s']:.2f} s "
                f"(frame {sustained_brake_event['frame']}, "
                f"gap {sustained_brake_event['gap_m']:.2f} m)"
            )
        else:
            print(
                "  sustained brake    : none"
            )

        if stop_event:
            print(
                "  confirmed stop    : "
                f"{stop_event['t_s']:.2f} s "
                f"(gap {stop_event['gap_m']:.2f} m)"
            )
        else:
            print(
                "  confirmed stop    : none"
            )

        print(
            "  minimum safety gap: "
            f"{min_safety_gap(rows):.3f} m"
        )

        print(
            "  final safety gap  : "
            f"{final_safety_gap(rows):.3f} m"
        )

        print(
            "  virtual collision :",
            any_collision(
                rows
            ),
        )

    print_condition(
        "CARLA",
        carla_rows,
        carla_brake,
        carla_sustained_brake,
        carla_stop,
    )

    print_condition(
        "HE",
        he_rows,
        he_brake,
        he_sustained_brake,
        he_stop,
    )

    diff = summary[
        "difference_he_minus_carla"
    ]

    print()
    print(
        "HE - CARLA DIFFERENCE"
    )
    print("-" * 78)

    print(
        "first brake pulse delta:",
        diff[
            "first_brake_pulse_delta_s"
        ],
        "s",
    )

    print(
        "sustained brake delta  :",
        diff[
            "sustained_brake_onset_delta_s"
        ],
        "s",
    )

    print(
        "minimum gap delta      :",
        f"{diff['min_gap_delta_m']:+.3f} m",
    )

    print(
        "final gap delta        :",
        f"{diff['final_gap_delta_m']:+.3f} m",
    )

    print()
    print(
        "CONTROL EQUIVALENCE"
    )
    print("-" * 78)

    print(
        "steer MAE              :",
        f"{diff['steer_mae']:.4f}",
    )

    print(
        "throttle MAE           :",
        f"{diff['throttle_mae']:.4f}",
    )

    print(
        "brake MAE              :",
        f"{diff['brake_mae']:.4f}",
    )

    print(
        "post-event brake MAE   :",
        f"{diff['post_event_brake_mae']:.4f}",
    )

    print(
        "brake-state disagreement:",
        f"{100.0 * diff['brake_state_disagreement_fraction']:.2f} %",
    )

    print()
    print(
        "CLOSED-LOOP RESPONSE"
    )
    print("-" * 78)

    print(
        "speed MAE              :",
        f"{diff['speed_mae_mps']:.3f} m/s",
    )

    print(
        "gap MAE                :",
        f"{diff['gap_mae_m']:.3f} m",
    )

    print(
        "max ego position error :",
        f"{diff['max_ego_position_error_m']:.3f} m",
    )

    print(
        "final ego position error:",
        f"{diff['final_ego_position_error_m']:.3f} m",
    )

    divergence = diff[
        "first_substantial_divergence"
    ]

    if divergence:
        print(
            "first substantial divergence:",
            f"t={divergence['t_s']:.2f}s "
            f"frame={divergence['frame']}"
        )
    else:
        print(
            "first substantial divergence: none"
        )

    print()
    print(
        "HE rendered frames      :",
        f"{100.0 * he_render_fraction:.2f} %",
    )

    print()
    print(
        "comparison CSV:",
        output_csv,
    )

    print(
        "summary JSON  :",
        output_json,
    )

    print("=" * 78)


if __name__ == "__main__":
    main()