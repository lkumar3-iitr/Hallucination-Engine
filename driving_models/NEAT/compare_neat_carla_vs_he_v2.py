"""
compare_neat_carla_vs_he_v2.py

Route-aware frame-aligned behavioral comparison:

    NEAT + CARLA-rendered scenario actor
                vs
    NEAT + Hallucination Engine actor

This script compares behavioral/safety equivalence,
NOT image similarity.

Primary safety metrics:
    route_bumper_gap_m
    route_virtual_collision

Legacy fallback:
    bumper_gap_m
    virtual_collision

Outputs:
    console summary
    per-frame comparison CSV
    summary JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


# ============================================================
# CSV helpers
# ============================================================

def load_csv(path: Path):

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


def number(
    row,
    key,
    default=float("nan"),
):

    value = row.get(
        key,
        "",
    )

    if (
        value is None
        or
        str(value).strip() == ""
    ):

        return default

    try:

        return float(
            value
        )

    except ValueError:

        return default


def boolean(
    row,
    key,
    default=False,
):

    value = row.get(
        key,
        "",
    )

    if (
        value is None
        or
        str(value).strip() == ""
    ):

        return bool(
            default
        )

    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "y",
        }
    )


def index_rows(
    rows,
):

    result = {}

    for row in rows:

        idx = int(
            number(
                row,
                "probe_idx",
            )
        )

        result[
            idx
        ] = row

    return result


# ============================================================
# Route-aware safety helpers
# ============================================================

def preferred_gap(
    row,
):
    """
    Use route-aware bumper gap whenever available.

    The legacy bumper_gap_m is invalid after large route
    curvature / turns because it is based on the original
    scenario longitudinal axis.
    """

    route_gap = number(
        row,
        "route_bumper_gap_m",
    )

    if math.isfinite(
        route_gap
    ):

        return route_gap

    return number(
        row,
        "bumper_gap_m",
    )


def preferred_collision(
    row,
):

    route_value = row.get(
        "route_virtual_collision",
        "",
    )

    if (
        route_value is not None
        and
        str(
            route_value
        ).strip() != ""
    ):

        return boolean(
            row,
            "route_virtual_collision",
        )

    return boolean(
        row,
        "virtual_collision",
    )


# ============================================================
# Numeric helpers
# ============================================================

def mae(
    a,
    b,
):

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
        &
        np.isfinite(b)
    )

    if not np.any(
        valid
    ):

        return float(
            "nan"
        )

    return float(
        np.mean(
            np.abs(
                a[valid]
                -
                b[valid]
            )
        )
    )


def rmse(
    a,
    b,
):

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
        &
        np.isfinite(b)
    )

    if not np.any(
        valid
    ):

        return float(
            "nan"
        )

    return float(
        np.sqrt(
            np.mean(
                (
                    a[valid]
                    -
                    b[valid]
                )
                ** 2
            )
        )
    )


def max_abs(
    a,
    b,
):

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
        &
        np.isfinite(b)
    )

    if not np.any(
        valid
    ):

        return float(
            "nan"
        )

    return float(
        np.max(
            np.abs(
                a[valid]
                -
                b[valid]
            )
        )
    )


def finite_min(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = values[
        np.isfinite(
            values
        )
    ]

    if valid.size == 0:

        return float(
            "nan"
        )

    return float(
        np.min(
            valid
        )
    )


def finite_mean(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = values[
        np.isfinite(
            values
        )
    ]

    if valid.size == 0:

        return float(
            "nan"
        )

    return float(
        np.mean(
            valid
        )
    )


def finite_last(
    values,
):

    for value in reversed(
        list(
            values
        )
    ):

        value = float(
            value
        )

        if math.isfinite(
            value
        ):

            return value

    return float(
        "nan"
    )


def safe_delta(
    he_value,
    carla_value,
):

    if (
        he_value is None
        or
        carla_value is None
    ):

        return None

    try:

        he_value = float(
            he_value
        )

        carla_value = float(
            carla_value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if (
        not math.isfinite(
            he_value
        )
        or
        not math.isfinite(
            carla_value
        )
    ):

        return None

    return (
        he_value
        -
        carla_value
    )


# ============================================================
# Sustained events
# ============================================================

def first_sustained(
    rows,
    predicate,
    consecutive,
    start_t,
):

    count = 0
    start_idx = None

    for idx, row in enumerate(
        rows
    ):

        t_s = number(
            row,
            "t_s",
        )

        if t_s < start_t:

            count = 0
            start_idx = None

            continue

        if predicate(
            row
        ):

            if count == 0:

                start_idx = idx

            count += 1

            if count >= consecutive:

                return (
                    start_idx,
                    rows[
                        start_idx
                    ],
                )

        else:

            count = 0
            start_idx = None

    return (
        None,
        None,
    )


def first_brake(
    rows,
    event_start_s,
    brake_threshold,
):

    for row in rows:

        t_s = number(
            row,
            "t_s",
        )

        if t_s < event_start_s:

            continue

        brake = number(
            row,
            "model_brake",
            0.0,
        )

        if brake >= brake_threshold:

            return {
                "frame_idx":
                    int(
                        number(
                            row,
                            "probe_idx",
                        )
                    ),

                "t_s":
                    t_s,

                "gap_m":
                    preferred_gap(
                        row
                    ),

                "ego_speed_mps":
                    number(
                        row,
                        "ego_speed_mps",
                    ),

                "brake":
                    brake,
            }

    return None


def first_sustained_brake(
    rows,
    event_start_s,
    brake_threshold,
    brake_frames,
):

    _, row = first_sustained(
        rows,

        predicate=lambda r:
            number(
                r,
                "model_brake",
                0.0,
            )
            >=
            brake_threshold,

        consecutive=
            brake_frames,

        start_t=
            event_start_s,
    )

    if row is None:

        return None

    return {
        "frame_idx":
            int(
                number(
                    row,
                    "probe_idx",
                )
            ),

        "t_s":
            number(
                row,
                "t_s",
            ),

        "gap_m":
            preferred_gap(
                row
            ),

        "ego_speed_mps":
            number(
                row,
                "ego_speed_mps",
            ),

        "brake":
            number(
                row,
                "model_brake",
            ),
    }


def first_stop(
    rows,
    event_start_s,
    stop_speed_mps,
    stop_frames,
):

    _, row = first_sustained(
        rows,

        predicate=lambda r:
            number(
                r,
                "ego_speed_mps",
                float("inf"),
            )
            <=
            stop_speed_mps,

        consecutive=
            stop_frames,

        start_t=
            event_start_s,
    )

    if row is None:

        return None

    return {
        "frame_idx":
            int(
                number(
                    row,
                    "probe_idx",
                )
            ),

        "t_s":
            number(
                row,
                "t_s",
            ),

        "gap_m":
            preferred_gap(
                row
            ),
    }


def first_red_light_positive(
    rows,
):

    for row in rows:

        value = int(
            number(
                row,
                "red_light_occ",
                0.0,
            )
        )

        if value != 0:

            return {
                "frame_idx":
                    int(
                        number(
                            row,
                            "probe_idx",
                        )
                    ),

                "t_s":
                    number(
                        row,
                        "t_s",
                    ),

                "value":
                    value,
            }

    return None


# ============================================================
# Individual-condition summary
# ============================================================

def summarize_condition(
    rows,
    event_start_s,
    brake_threshold,
    brake_frames,
    stop_speed_mps,
    stop_frames,
):

    times = np.array(
        [
            number(
                row,
                "t_s",
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    gaps = np.array(
        [
            preferred_gap(
                row
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    post_mask = (
        times
        >=
        event_start_s
    )

    red_light = np.array(
        [
            int(
                number(
                    row,
                    "red_light_occ",
                    0.0,
                )
            )
            for row in rows
        ],
        dtype=np.int64,
    )

    return {
        "frames":
            len(
                rows
            ),

        "first_brake":
            first_brake(
                rows,
                event_start_s,
                brake_threshold,
            ),

        "sustained_brake":
            first_sustained_brake(
                rows,
                event_start_s,
                brake_threshold,
                brake_frames,
            ),

        "confirmed_stop":
            first_stop(
                rows,
                event_start_s,
                stop_speed_mps,
                stop_frames,
            ),

        "minimum_route_gap_m":
            finite_min(
                gaps
            ),

        "minimum_route_gap_post_event_m":
            finite_min(
                gaps[
                    post_mask
                ]
            ),

        "final_route_gap_m":
            finite_last(
                gaps
            ),

        "route_virtual_collision":
            bool(
                any(
                    preferred_collision(
                        row
                    )
                    for row in rows
                )
            ),

        "final_speed_mps":
            number(
                rows[-1],
                "ego_speed_mps",
            ),

        "red_light_positive_fraction":
            float(
                np.mean(
                    red_light != 0
                )
            ),

        "first_red_light_positive":
            first_red_light_positive(
                rows
            ),
    }


# ============================================================
# Pair comparison
# ============================================================

def compare_pair(
    carla_rows,
    he_rows,
    event_start_s,
    brake_threshold,
):

    carla_index = index_rows(
        carla_rows
    )

    he_index = index_rows(
        he_rows
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

    carla = [
        carla_index[
            idx
        ]
        for idx in common_frames
    ]

    he = [
        he_index[
            idx
        ]
        for idx in common_frames
    ]

    def values(
        rows,
        key,
    ):

        return np.array(
            [
                number(
                    row,
                    key,
                )
                for row in rows
            ],
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    time_c = values(
        carla,
        "t_s",
    )

    time_h = values(
        he,
        "t_s",
    )

    post_event = (
        time_c
        >=
        event_start_s
    )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    steer_c = values(
        carla,
        "model_steer",
    )

    steer_h = values(
        he,
        "model_steer",
    )

    throttle_c = values(
        carla,
        "model_throttle",
    )

    throttle_h = values(
        he,
        "model_throttle",
    )

    brake_c = values(
        carla,
        "model_brake",
    )

    brake_h = values(
        he,
        "model_brake",
    )

    # --------------------------------------------------------
    # Ego response
    # --------------------------------------------------------

    speed_c = values(
        carla,
        "ego_speed_mps",
    )

    speed_h = values(
        he,
        "ego_speed_mps",
    )

    gap_c = np.array(
        [
            preferred_gap(
                row
            )
            for row in carla
        ],
        dtype=np.float64,
    )

    gap_h = np.array(
        [
            preferred_gap(
                row
            )
            for row in he
        ],
        dtype=np.float64,
    )

    ego_x_c = values(
        carla,
        "ego_x",
    )

    ego_x_h = values(
        he,
        "ego_x",
    )

    ego_y_c = values(
        carla,
        "ego_y",
    )

    ego_y_h = values(
        he,
        "ego_y",
    )

    ego_xy_error = np.sqrt(
        (
            ego_x_h
            -
            ego_x_c
        )
        ** 2
        +
        (
            ego_y_h
            -
            ego_y_c
        )
        ** 2
    )

    # --------------------------------------------------------
    # Scenario physical truth
    # --------------------------------------------------------

    actor_x_c = values(
        carla,
        "actor_world_x",
    )

    actor_x_h = values(
        he,
        "actor_world_x",
    )

    actor_y_c = values(
        carla,
        "actor_world_y",
    )

    actor_y_h = values(
        he,
        "actor_world_y",
    )

    actor_xy_error = np.sqrt(
        (
            actor_x_h
            -
            actor_x_c
        )
        ** 2
        +
        (
            actor_y_h
            -
            actor_y_c
        )
        ** 2
    )

    actor_speed_c = values(
        carla,
        "actor_speed_mps",
    )

    actor_speed_h = values(
        he,
        "actor_speed_mps",
    )

    # --------------------------------------------------------
    # Binary model states
    # --------------------------------------------------------

    brake_state_c = (
        brake_c
        >=
        brake_threshold
    )

    brake_state_h = (
        brake_h
        >=
        brake_threshold
    )

    brake_disagreement = (
        brake_state_c
        !=
        brake_state_h
    )

    red_light_c = np.array(
        [
            int(
                number(
                    row,
                    "red_light_occ",
                    0.0,
                )
            )
            for row in carla
        ],
        dtype=np.int64,
    )

    red_light_h = np.array(
        [
            int(
                number(
                    row,
                    "red_light_occ",
                    0.0,
                )
            )
            for row in he
        ],
        dtype=np.int64,
    )

    red_light_disagreement = (
        red_light_c
        !=
        red_light_h
    )

    # --------------------------------------------------------
    # First substantial divergence
    #
    # This is intentionally searched over the whole run,
    # rather than only after the event.
    # --------------------------------------------------------

    first_divergence = None

    for j, frame_idx in enumerate(
        common_frames
    ):

        brake_diff = abs(
            brake_h[j]
            -
            brake_c[j]
        )

        speed_diff = abs(
            speed_h[j]
            -
            speed_c[j]
        )

        position_diff = float(
            ego_xy_error[j]
        )

        if (
            brake_diff >= 0.30
            or
            speed_diff >= 0.50
            or
            position_diff >= 0.50
        ):

            first_divergence = {
                "frame_idx":
                    int(
                        frame_idx
                    ),

                "t_s":
                    float(
                        time_c[j]
                    ),

                "post_event":
                    bool(
                        time_c[j]
                        >=
                        event_start_s
                    ),

                "brake_abs_diff":
                    float(
                        brake_diff
                    ),

                "speed_abs_diff_mps":
                    float(
                        speed_diff
                    ),

                "ego_xy_error_m":
                    position_diff,
            }

            break

    # --------------------------------------------------------
    # HE rendering coverage
    # --------------------------------------------------------

    he_view_coverage = {}

    for view in (
        "front",
        "left",
        "right",
    ):

        key = (
            f"he_{view}_rendered"
        )

        rendered = np.array(
            [
                boolean(
                    row,
                    key,
                )
                for row in he
            ],
            dtype=np.bool_,
        )

        he_view_coverage[
            view
        ] = float(
            np.mean(
                rendered
            )
        )

    # --------------------------------------------------------
    # Per-frame comparison rows
    # --------------------------------------------------------

    comparison_rows = []

    for j, frame_idx in enumerate(
        common_frames
    ):

        comparison_rows.append({
            "probe_idx":
                int(
                    frame_idx
                ),

            "t_s":
                float(
                    time_c[j]
                ),

            "post_event":
                int(
                    post_event[j]
                ),

            "carla_steer":
                float(
                    steer_c[j]
                ),

            "he_steer":
                float(
                    steer_h[j]
                ),

            "steer_abs_diff":
                float(
                    abs(
                        steer_h[j]
                        -
                        steer_c[j]
                    )
                ),

            "carla_throttle":
                float(
                    throttle_c[j]
                ),

            "he_throttle":
                float(
                    throttle_h[j]
                ),

            "throttle_abs_diff":
                float(
                    abs(
                        throttle_h[j]
                        -
                        throttle_c[j]
                    )
                ),

            "carla_brake":
                float(
                    brake_c[j]
                ),

            "he_brake":
                float(
                    brake_h[j]
                ),

            "brake_abs_diff":
                float(
                    abs(
                        brake_h[j]
                        -
                        brake_c[j]
                    )
                ),

            "brake_state_disagreement":
                int(
                    brake_disagreement[j]
                ),

            "carla_speed_mps":
                float(
                    speed_c[j]
                ),

            "he_speed_mps":
                float(
                    speed_h[j]
                ),

            "speed_abs_diff_mps":
                float(
                    abs(
                        speed_h[j]
                        -
                        speed_c[j]
                    )
                ),

            "carla_route_gap_m":
                float(
                    gap_c[j]
                ),

            "he_route_gap_m":
                float(
                    gap_h[j]
                ),

            "route_gap_abs_diff_m":
                float(
                    abs(
                        gap_h[j]
                        -
                        gap_c[j]
                    )
                ),

            "ego_xy_error_m":
                float(
                    ego_xy_error[j]
                ),

            "actor_xy_error_m":
                float(
                    actor_xy_error[j]
                ),

            "carla_red_light_occ":
                int(
                    red_light_c[j]
                ),

            "he_red_light_occ":
                int(
                    red_light_h[j]
                ),

            "red_light_disagreement":
                int(
                    red_light_disagreement[j]
                ),
        })

    metrics = {
        "matched_frames":
            len(
                common_frames
            ),

        # Exact scenario consistency.
        "max_time_mismatch_s":
            max_abs(
                time_c,
                time_h,
            ),

        "max_actor_xy_mismatch_m":
            float(
                np.nanmax(
                    actor_xy_error
                )
            ),

        "max_actor_speed_mismatch_mps":
            max_abs(
                actor_speed_c,
                actor_speed_h,
            ),

        # Controls.
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

        "post_event_brake_mae":
            mae(
                brake_c[
                    post_event
                ],
                brake_h[
                    post_event
                ],
            ),

        # Closed-loop response.
        "speed_mae_mps":
            mae(
                speed_c,
                speed_h,
            ),

        "post_event_speed_mae_mps":
            mae(
                speed_c[
                    post_event
                ],
                speed_h[
                    post_event
                ],
            ),

        "route_gap_mae_m":
            mae(
                gap_c,
                gap_h,
            ),

        "post_event_route_gap_mae_m":
            mae(
                gap_c[
                    post_event
                ],
                gap_h[
                    post_event
                ],
            ),

        "route_gap_rmse_m":
            rmse(
                gap_c,
                gap_h,
            ),

        "mean_ego_xy_error_m":
            finite_mean(
                ego_xy_error
            ),

        "max_ego_xy_error_m":
            float(
                np.nanmax(
                    ego_xy_error
                )
            ),

        "final_ego_xy_error_m":
            finite_last(
                ego_xy_error
            ),

        # Binary behavior.
        "brake_state_disagreement_fraction":
            float(
                np.mean(
                    brake_disagreement
                )
            ),

        "post_event_brake_state_disagreement_fraction":
            (
                float(
                    np.mean(
                        brake_disagreement[
                            post_event
                        ]
                    )
                )
                if np.any(
                    post_event
                )
                else float(
                    "nan"
                )
            ),

        "red_light_disagreement_fraction":
            float(
                np.mean(
                    red_light_disagreement
                )
            ),

        "first_substantial_divergence":
            first_divergence,

        "he_view_render_fraction":
            he_view_coverage,
    }

    return (
        metrics,
        comparison_rows,
    )


# ============================================================
# Pretty printing
# ============================================================

def print_event(
    name,
    carla_event,
    he_event,
):

    print(
        name
    )

    if carla_event is None:

        print(
            "  CARLA : none"
        )

    else:

        print(
            "  CARLA : "
            f"frame={carla_event['frame_idx']} "
            f"t={carla_event['t_s']:.2f}s "
            f"gap={carla_event['gap_m']:.3f}m"
        )

    if he_event is None:

        print(
            "  HE    : none"
        )

    else:

        print(
            "  HE    : "
            f"frame={he_event['frame_idx']} "
            f"t={he_event['t_s']:.2f}s "
            f"gap={he_event['gap_m']:.3f}m"
        )

    if (
        carla_event is not None
        and
        he_event is not None
    ):

        print(
            "  delta : "
            f"dt="
            f"{safe_delta(he_event['t_s'], carla_event['t_s']):+.3f}s "
            f"dgap="
            f"{safe_delta(he_event['gap_m'], carla_event['gap_m']):+.3f}m"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--carla",
        required=True,
    )

    parser.add_argument(
        "--he",
        required=True,
    )

    parser.add_argument(
        "--event-start-s",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--brake-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--brake-frames",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--stop-speed-mps",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--stop-frames",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    carla_path = Path(
        args.carla
    ).resolve()

    he_path = Path(
        args.he
    ).resolve()

    if args.output_dir is None:

        output_dir = (
            he_path
            .parent
            .parent
            /
            "analysis"
        )

    else:

        output_dir = Path(
            args.output_dir
        ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_csv = (
        output_dir
        /
        "neat_carla_vs_he_frames.csv"
    )

    summary_json = (
        output_dir
        /
        "neat_carla_vs_he_summary.json"
    )

    # ========================================================
    # Load
    # ========================================================

    carla_rows = load_csv(
        carla_path
    )

    he_rows = load_csv(
        he_path
    )

    # ========================================================
    # Individual summaries
    # ========================================================

    carla_summary = summarize_condition(
        carla_rows,

        event_start_s=
            args.event_start_s,

        brake_threshold=
            args.brake_threshold,

        brake_frames=
            args.brake_frames,

        stop_speed_mps=
            args.stop_speed_mps,

        stop_frames=
            args.stop_frames,
    )

    he_summary = summarize_condition(
        he_rows,

        event_start_s=
            args.event_start_s,

        brake_threshold=
            args.brake_threshold,

        brake_frames=
            args.brake_frames,

        stop_speed_mps=
            args.stop_speed_mps,

        stop_frames=
            args.stop_frames,
    )

    # ========================================================
    # Pair comparison
    # ========================================================

    (
        pair_metrics,
        comparison_rows,
    ) = compare_pair(
        carla_rows,
        he_rows,

        event_start_s=
            args.event_start_s,

        brake_threshold=
            args.brake_threshold,
    )

    # ========================================================
    # Save per-frame CSV
    # ========================================================

    with comparison_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                comparison_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            comparison_rows
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "carla_csv":
            str(
                carla_path
            ),

        "he_csv":
            str(
                he_path
            ),

        "configuration": {
            "event_start_s":
                args.event_start_s,

            "brake_threshold":
                args.brake_threshold,

            "brake_confirm_frames":
                args.brake_frames,

            "stop_speed_mps":
                args.stop_speed_mps,

            "stop_confirm_frames":
                args.stop_frames,

            "gap_metric":
                (
                    "route_bumper_gap_m "
                    "with bumper_gap_m fallback"
                ),

            "collision_metric":
                (
                    "route_virtual_collision "
                    "with virtual_collision fallback"
                ),
        },

        "carla":
            carla_summary,

        "he":
            he_summary,

        "deltas": {
            "first_brake_time_s":
                safe_delta(
                    (
                        he_summary[
                            "first_brake"
                        ]
                        or {}
                    ).get(
                        "t_s"
                    ),
                    (
                        carla_summary[
                            "first_brake"
                        ]
                        or {}
                    ).get(
                        "t_s"
                    ),
                ),

            "sustained_brake_time_s":
                safe_delta(
                    (
                        he_summary[
                            "sustained_brake"
                        ]
                        or {}
                    ).get(
                        "t_s"
                    ),
                    (
                        carla_summary[
                            "sustained_brake"
                        ]
                        or {}
                    ).get(
                        "t_s"
                    ),
                ),

            "minimum_route_gap_m":
                safe_delta(
                    he_summary[
                        "minimum_route_gap_m"
                    ],
                    carla_summary[
                        "minimum_route_gap_m"
                    ],
                ),

            "final_route_gap_m":
                safe_delta(
                    he_summary[
                        "final_route_gap_m"
                    ],
                    carla_summary[
                        "final_route_gap_m"
                    ],
                ),
        },

        "pairwise":
            pair_metrics,
    }

    with summary_json.open(
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
    # Console
    # ========================================================

    print()
    print(
        "=" * 78
    )

    print(
        "NEAT CARLA vs HE ROUTE-AWARE COMPARISON"
    )

    print(
        "=" * 78
    )

    print(
        "CARLA frames :",
        len(
            carla_rows
        ),
    )

    print(
        "HE frames    :",
        len(
            he_rows
        ),
    )

    print(
        "Matched      :",
        pair_metrics[
            "matched_frames"
        ],
    )

    print()
    print(
        "SCENARIO PHYSICAL TRUTH"
    )

    print(
        "max time mismatch       : "
        f"{pair_metrics['max_time_mismatch_s']:.6f} s"
    )

    print(
        "max actor XY mismatch   : "
        f"{pair_metrics['max_actor_xy_mismatch_m']:.6f} m"
    )

    print(
        "max actor speed mismatch: "
        f"{pair_metrics['max_actor_speed_mismatch_mps']:.6f} m/s"
    )

    print()
    print(
        "EVENT OUTCOMES"
    )

    print_event(
        "first brake >= threshold",
        carla_summary[
            "first_brake"
        ],
        he_summary[
            "first_brake"
        ],
    )

    print_event(
        "sustained brake",
        carla_summary[
            "sustained_brake"
        ],
        he_summary[
            "sustained_brake"
        ],
    )

    print_event(
        "confirmed stop",
        carla_summary[
            "confirmed_stop"
        ],
        he_summary[
            "confirmed_stop"
        ],
    )

    print(
        "minimum route gap"
    )

    print(
        "  CARLA : "
        f"{carla_summary['minimum_route_gap_m']:.3f} m"
    )

    print(
        "  HE    : "
        f"{he_summary['minimum_route_gap_m']:.3f} m"
    )

    print(
        "  delta : "
        f"{summary['deltas']['minimum_route_gap_m']:+.3f} m"
    )

    print(
        "route collision"
    )

    print(
        "  CARLA :",
        carla_summary[
            "route_virtual_collision"
        ],
    )

    print(
        "  HE    :",
        he_summary[
            "route_virtual_collision"
        ],
    )

    print()
    print(
        "CONTROL EQUIVALENCE"
    )

    print(
        "steer MAE                : "
        f"{pair_metrics['steer_mae']:.4f}"
    )

    print(
        "throttle MAE             : "
        f"{pair_metrics['throttle_mae']:.4f}"
    )

    print(
        "brake MAE                : "
        f"{pair_metrics['brake_mae']:.4f}"
    )

    print(
        "post-event brake MAE     : "
        f"{pair_metrics['post_event_brake_mae']:.4f}"
    )

    print(
        "brake disagreement       : "
        f"{100.0 * pair_metrics['brake_state_disagreement_fraction']:.2f}%"
    )

    print(
        "post brake disagreement  : "
        f"{100.0 * pair_metrics['post_event_brake_state_disagreement_fraction']:.2f}%"
    )

    print()
    print(
        "CLOSED-LOOP RESPONSE"
    )

    print(
        "speed MAE                : "
        f"{pair_metrics['speed_mae_mps']:.3f} m/s"
    )

    print(
        "post-event speed MAE     : "
        f"{pair_metrics['post_event_speed_mae_mps']:.3f} m/s"
    )

    print(
        "route-gap MAE            : "
        f"{pair_metrics['route_gap_mae_m']:.3f} m"
    )

    print(
        "post-event route-gap MAE : "
        f"{pair_metrics['post_event_route_gap_mae_m']:.3f} m"
    )

    print(
        "route-gap RMSE           : "
        f"{pair_metrics['route_gap_rmse_m']:.3f} m"
    )

    print(
        "mean ego XY error        : "
        f"{pair_metrics['mean_ego_xy_error_m']:.3f} m"
    )

    print(
        "max ego XY error         : "
        f"{pair_metrics['max_ego_xy_error_m']:.3f} m"
    )

    print(
        "final ego XY error       : "
        f"{pair_metrics['final_ego_xy_error_m']:.3f} m"
    )

    print()
    print(
        "NEAT-SPECIFIC"
    )

    print(
        "red-light disagreement   : "
        f"{100.0 * pair_metrics['red_light_disagreement_fraction']:.2f}%"
    )

    print(
        "CARLA red-light positive : "
        f"{100.0 * carla_summary['red_light_positive_fraction']:.2f}%"
    )

    print(
        "HE red-light positive    : "
        f"{100.0 * he_summary['red_light_positive_fraction']:.2f}%"
    )

    print(
        "HE view render fraction  : "
        f"front="
        f"{100.0 * pair_metrics['he_view_render_fraction']['front']:.1f}% "
        f"left="
        f"{100.0 * pair_metrics['he_view_render_fraction']['left']:.1f}% "
        f"right="
        f"{100.0 * pair_metrics['he_view_render_fraction']['right']:.1f}%"
    )

    divergence = pair_metrics[
        "first_substantial_divergence"
    ]

    if divergence is None:

        print(
            "first substantial divergence: none"
        )

    else:

        print(
            "first substantial divergence: "
            f"frame={divergence['frame_idx']} "
            f"t={divergence['t_s']:.2f}s "
            f"post_event={divergence['post_event']} "
            f"dB={divergence['brake_abs_diff']:.3f} "
            f"dV={divergence['speed_abs_diff_mps']:.3f}m/s "
            f"dXY={divergence['ego_xy_error_m']:.3f}m"
        )

    print()
    print(
        "comparison CSV:",
        comparison_csv,
    )

    print(
        "summary JSON :",
        summary_json,
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()