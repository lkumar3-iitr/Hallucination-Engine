"""
analyze_neat_lead_brake_pair_v1.py

Single-pair analysis for the NEAT lead-brake experiment.

Compares:

    physical CARLA adversary
        vs
    Hallucination Engine adversary

using the exact same resolved scenario and NEAT model.

Event definitions
-----------------
Brake onset:
    model_brake >= 0.5 for 3 consecutive frames.

Stop onset:
    ego_speed_mps <= 0.15 m/s for 5 consecutive frames.

These sustained definitions avoid treating one-frame controller
oscillations as behavioral events.
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


DEFAULT_ROOT = (
    HE_ROOT
    / "driving_models"
    / "NEAT"
    / "outputs"
    / "neat_he_pair_v1"
    / "tcp_lead_brake_001"
)


DEFAULT_CARLA = (
    DEFAULT_ROOT
    / "carla"
    / "tcp_lead_brake_001_carla.csv"
)


DEFAULT_HE = (
    DEFAULT_ROOT
    / "he"
    / "tcp_lead_brake_001_he.csv"
)


DEFAULT_OUTPUT = (
    DEFAULT_ROOT
    / "analysis"
    / "tcp_lead_brake_001_pair_summary.json"
)


# ============================================================
# CSV
# ============================================================

def load_csv(path):

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
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
        ""
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


# ============================================================
# Sustained events
# ============================================================

def first_sustained(
    rows,
    predicate,
    consecutive,
    start_t=0.0,
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


# ============================================================
# Individual condition
# ============================================================

def analyze_condition(
    rows,
    event_start_s,
    brake_threshold,
    brake_frames,
    stop_speed,
    stop_frames,
):

    if not rows:

        raise RuntimeError(
            "Empty CSV."
        )

    # --------------------------------------------------------
    # Sustained brake onset
    # --------------------------------------------------------

    (
        brake_idx,
        brake_row,
    ) = first_sustained(
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

    # --------------------------------------------------------
    # Sustained complete stop
    # --------------------------------------------------------

    (
        stop_idx,
        stop_row,
    ) = first_sustained(
        rows,

        predicate=lambda r:
            number(
                r,
                "ego_speed_mps",
                float("inf"),
            )
            <=
            stop_speed,

        consecutive=
            stop_frames,

        start_t=
            event_start_s,
    )

    # --------------------------------------------------------
    # Event region
    # --------------------------------------------------------

    post_event = [
        row
        for row in rows
        if number(
            row,
            "t_s",
        )
        >=
        event_start_s
    ]

    gaps = np.array(
        [
            number(
                r,
                "bumper_gap_m",
            )
            for r in post_event
        ],
        dtype=np.float64,
    )

    speeds = np.array(
        [
            number(
                r,
                "ego_speed_mps",
            )
            for r in post_event
        ],
        dtype=np.float64,
    )

    brakes = np.array(
        [
            number(
                r,
                "model_brake",
                0.0,
            )
            for r in post_event
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Ego progress / post-event travel
    # --------------------------------------------------------

    event_row = post_event[
        0
    ]

    final_row = rows[
        -1
    ]

    event_progress = number(
        event_row,
        "ego_x",
    )

    final_progress = number(
        final_row,
        "ego_x",
    )

    # More robust than raw ego_x:
    #
    # center_distance =
    # actor_progress - ego_progress
    #
    # therefore:
    #
    # ego_progress =
    # actor_progress - center_distance

    event_ego_progress = (
        number(
            event_row,
            "actor_x_sg_m",
        )
        -
        number(
            event_row,
            "center_distance_m",
        )
    )

    final_ego_progress = (
        number(
            final_row,
            "actor_x_sg_m",
        )
        -
        number(
            final_row,
            "center_distance_m",
        )
    )

    post_event_travel = (
        final_ego_progress
        -
        event_ego_progress
    )

    # --------------------------------------------------------
    # Collision
    # --------------------------------------------------------

    collision = any(
        int(
            number(
                r,
                "virtual_collision",
                0,
            )
        )
        != 0
        for r in rows
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    result = {
        "frames":
            len(
                rows
            ),

        "event_start_s":
            float(
                event_start_s
            ),

        "brake_onset": None,

        "stop_onset": None,

        "minimum_gap_post_event_m":
            float(
                np.nanmin(
                    gaps
                )
            ),

        "final_gap_m":
            number(
                final_row,
                "bumper_gap_m",
            ),

        "final_speed_mps":
            number(
                final_row,
                "ego_speed_mps",
            ),

        "post_event_travel_m":
            float(
                post_event_travel
            ),

        "mean_speed_post_event_mps":
            float(
                np.nanmean(
                    speeds
                )
            ),

        "brake_fraction_post_event":
            float(
                np.mean(
                    brakes
                    >=
                    brake_threshold
                )
            ),

        "collision":
            bool(
                collision
            ),
    }

    if brake_row is not None:

        result[
            "brake_onset"
        ] = {
            "frame_idx":
                int(
                    number(
                        brake_row,
                        "probe_idx",
                    )
                ),

            "t_s":
                number(
                    brake_row,
                    "t_s",
                ),

            "gap_m":
                number(
                    brake_row,
                    "bumper_gap_m",
                ),

            "speed_mps":
                number(
                    brake_row,
                    "ego_speed_mps",
                ),
        }

    if stop_row is not None:

        result[
            "stop_onset"
        ] = {
            "frame_idx":
                int(
                    number(
                        stop_row,
                        "probe_idx",
                    )
                ),

            "t_s":
                number(
                    stop_row,
                    "t_s",
                ),

            "gap_m":
                number(
                    stop_row,
                    "bumper_gap_m",
                ),
        }

    return result


# ============================================================
# Pairwise time-series comparison
# ============================================================

def analyze_pair(
    carla_rows,
    he_rows,
    event_start_s,
    brake_threshold,
):

    n = min(
        len(
            carla_rows
        ),
        len(
            he_rows
        ),
    )

    carla_rows = (
        carla_rows[
            :n
        ]
    )

    he_rows = (
        he_rows[
            :n
        ]
    )

    indices = [
        i
        for i in range(
            n
        )
        if (
            number(
                carla_rows[i],
                "t_s",
            )
            >=
            event_start_s
        )
    ]

    def mae(
        key,
    ):

        values = []

        for i in indices:

            a = number(
                carla_rows[i],
                key,
            )

            b = number(
                he_rows[i],
                key,
            )

            if (
                math.isfinite(
                    a
                )
                and
                math.isfinite(
                    b
                )
            ):

                values.append(
                    abs(
                        b - a
                    )
                )

        if not values:

            return float(
                "nan"
            )

        return float(
            np.mean(
                values
            )
        )

    # Binary brake disagreement.

    brake_disagreement = []

    for i in indices:

        carla_brake = (
            number(
                carla_rows[i],
                "model_brake",
                0.0,
            )
            >=
            brake_threshold
        )

        he_brake = (
            number(
                he_rows[i],
                "model_brake",
                0.0,
            )
            >=
            brake_threshold
        )

        brake_disagreement.append(
            carla_brake
            !=
            he_brake
        )

    return {
        "paired_frames":
            n,

        "post_event_frames":
            len(
                indices
            ),

        "speed_mae_post_event_mps":
            mae(
                "ego_speed_mps"
            ),

        "gap_mae_post_event_m":
            mae(
                "bumper_gap_m"
            ),

        "steer_mae_post_event":
            mae(
                "model_steer"
            ),

        "throttle_mae_post_event":
            mae(
                "model_throttle"
            ),

        "brake_mae_post_event":
            mae(
                "model_brake"
            ),

        "brake_disagreement_fraction":
            float(
                np.mean(
                    brake_disagreement
                )
            )
            if brake_disagreement
            else float(
                "nan"
            ),
    }


# ============================================================
# Delta
# ============================================================

def delta(
    he_value,
    carla_value,
):

    if (
        he_value is None
        or
        carla_value is None
    ):

        return None

    return float(
        he_value
        -
        carla_value
    )


# ============================================================
# Pretty printing
# ============================================================

def format_value(
    value,
    digits=3,
):

    if value is None:

        return "none"

    if not math.isfinite(
        float(
            value
        )
    ):

        return "nan"

    return (
        f"{float(value):.{digits}f}"
    )


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
        "--event-start-s",
        type=float,
        default=4.0,
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
        "--stop-speed",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--stop-frames",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    args = parser.parse_args()

    carla_path = Path(
        args.carla
    )

    he_path = Path(
        args.he
    )

    carla_rows = load_csv(
        carla_path
    )

    he_rows = load_csv(
        he_path
    )

    carla = analyze_condition(
        carla_rows,
        event_start_s=
            args.event_start_s,
        brake_threshold=
            args.brake_threshold,
        brake_frames=
            args.brake_frames,
        stop_speed=
            args.stop_speed,
        stop_frames=
            args.stop_frames,
    )

    he = analyze_condition(
        he_rows,
        event_start_s=
            args.event_start_s,
        brake_threshold=
            args.brake_threshold,
        brake_frames=
            args.brake_frames,
        stop_speed=
            args.stop_speed,
        stop_frames=
            args.stop_frames,
    )

    paired = analyze_pair(
        carla_rows,
        he_rows,
        event_start_s=
            args.event_start_s,
        brake_threshold=
            args.brake_threshold,
    )

    carla_brake_t = (
        carla[
            "brake_onset"
        ][
            "t_s"
        ]
        if
        carla[
            "brake_onset"
        ]
        is not None
        else None
    )

    he_brake_t = (
        he[
            "brake_onset"
        ][
            "t_s"
        ]
        if
        he[
            "brake_onset"
        ]
        is not None
        else None
    )

    carla_brake_gap = (
        carla[
            "brake_onset"
        ][
            "gap_m"
        ]
        if
        carla[
            "brake_onset"
        ]
        is not None
        else None
    )

    he_brake_gap = (
        he[
            "brake_onset"
        ][
            "gap_m"
        ]
        if
        he[
            "brake_onset"
        ]
        is not None
        else None
    )

    carla_stop_t = (
        carla[
            "stop_onset"
        ][
            "t_s"
        ]
        if
        carla[
            "stop_onset"
        ]
        is not None
        else None
    )

    he_stop_t = (
        he[
            "stop_onset"
        ][
            "t_s"
        ]
        if
        he[
            "stop_onset"
        ]
        is not None
        else None
    )

    carla_stop_gap = (
        carla[
            "stop_onset"
        ][
            "gap_m"
        ]
        if
        carla[
            "stop_onset"
        ]
        is not None
        else None
    )

    he_stop_gap = (
        he[
            "stop_onset"
        ][
            "gap_m"
        ]
        if
        he[
            "stop_onset"
        ]
        is not None
        else None
    )

    deltas = {
        "brake_onset_s":
            delta(
                he_brake_t,
                carla_brake_t,
            ),

        "brake_onset_gap_m":
            delta(
                he_brake_gap,
                carla_brake_gap,
            ),

        "stop_onset_s":
            delta(
                he_stop_t,
                carla_stop_t,
            ),

        "stop_gap_m":
            delta(
                he_stop_gap,
                carla_stop_gap,
            ),

        "minimum_gap_post_event_m":
            delta(
                he[
                    "minimum_gap_post_event_m"
                ],
                carla[
                    "minimum_gap_post_event_m"
                ],
            ),

        "final_gap_m":
            delta(
                he[
                    "final_gap_m"
                ],
                carla[
                    "final_gap_m"
                ],
            ),

        "post_event_travel_m":
            delta(
                he[
                    "post_event_travel_m"
                ],
                carla[
                    "post_event_travel_m"
                ],
            ),
    }

    summary = {
        "definitions": {
            "event_start_s":
                args.event_start_s,

            "brake_threshold":
                args.brake_threshold,

            "brake_sustained_frames":
                args.brake_frames,

            "stop_speed_threshold_mps":
                args.stop_speed,

            "stop_sustained_frames":
                args.stop_frames,
        },

        "carla":
            carla,

        "he":
            he,

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
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 78)
    print(
        "NEAT LEAD-BRAKE CARLA <-> HE"
    )
    print("=" * 78)

    print(
        "Brake definition:",
        f">={args.brake_threshold:.2f}",
        "for",
        args.brake_frames,
        "frames",
    )

    print(
        "Stop definition:",
        f"<={args.stop_speed:.2f} m/s",
        "for",
        args.stop_frames,
        "frames",
    )

    print()

    print(
        "%-30s %12s %12s %12s"
        %
        (
            "Metric",
            "CARLA",
            "HE",
            "HE-CARLA",
        )
    )

    print("-" * 78)

    def row(
        name,
        carla_value,
        he_value,
        delta_value,
    ):

        print(
            "%-30s %12s %12s %12s"
            %
            (
                name,

                format_value(
                    carla_value
                ),

                format_value(
                    he_value
                ),

                format_value(
                    delta_value
                ),
            )
        )

    row(
        "Brake onset (s)",
        carla_brake_t,
        he_brake_t,
        deltas[
            "brake_onset_s"
        ],
    )

    row(
        "Gap at brake onset (m)",
        carla_brake_gap,
        he_brake_gap,
        deltas[
            "brake_onset_gap_m"
        ],
    )

    row(
        "Stop onset (s)",
        carla_stop_t,
        he_stop_t,
        deltas[
            "stop_onset_s"
        ],
    )

    row(
        "Gap at stop (m)",
        carla_stop_gap,
        he_stop_gap,
        deltas[
            "stop_gap_m"
        ],
    )

    row(
        "Minimum gap (m)",
        carla[
            "minimum_gap_post_event_m"
        ],
        he[
            "minimum_gap_post_event_m"
        ],
        deltas[
            "minimum_gap_post_event_m"
        ],
    )

    row(
        "Final gap (m)",
        carla[
            "final_gap_m"
        ],
        he[
            "final_gap_m"
        ],
        deltas[
            "final_gap_m"
        ],
    )

    row(
        "Post-event travel (m)",
        carla[
            "post_event_travel_m"
        ],
        he[
            "post_event_travel_m"
        ],
        deltas[
            "post_event_travel_m"
        ],
    )

    print()
    print(
        "Paired post-event trajectory/control differences"
    )
    print("-" * 78)

    print(
        "Speed MAE:",
        f"{paired['speed_mae_post_event_mps']:.3f} m/s"
    )

    print(
        "Gap MAE:",
        f"{paired['gap_mae_post_event_m']:.3f} m"
    )

    print(
        "Steer MAE:",
        f"{paired['steer_mae_post_event']:.4f}"
    )

    print(
        "Throttle MAE:",
        f"{paired['throttle_mae_post_event']:.4f}"
    )

    print(
        "Brake MAE:",
        f"{paired['brake_mae_post_event']:.4f}"
    )

    print(
        "Brake disagreement:",
        f"{100.0 * paired['brake_disagreement_fraction']:.2f}%"
    )

    print()

    print(
        "Collision CARLA:",
        carla[
            "collision"
        ]
    )

    print(
        "Collision HE:",
        he[
            "collision"
        ]
    )

    print()

    print(
        "Summary:",
        output_path,
    )

    print("=" * 78)


if __name__ == "__main__":

    main()