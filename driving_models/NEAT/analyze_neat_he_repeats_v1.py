"""
analyze_neat_he_repeats_v1.py

Aggregate repeated NEAT CARLA <-> HE lead-brake experiments.

Reads:

    outputs/neat_he_repeat_v1/
        pair_01/...
        pair_02/...
        ...
        pair_05/...

and reports:

    - individual pair metrics
    - CARLA mean +/- sample std
    - HE mean +/- sample std
    - paired HE-CARLA mean +/- sample std
    - directional consistency
    - collision counts
    - paired trajectory/control differences
"""

from __future__ import annotations

import argparse
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
    / "neat_he_repeat_v1"
)


DEFAULT_SCENARIO = (
    "tcp_lead_brake_001"
)


# ============================================================
# Helpers
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def nested_get(
    data,
    keys,
    default=float("nan"),
):

    current = data

    for key in keys:

        if (
            current is None
            or
            key not in current
        ):

            return default

        current = current[
            key
        ]

    if current is None:

        return default

    try:

        return float(
            current
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def mean_std(
    values,
):

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    if len(
        array
    ) == 0:

        return (
            float("nan"),
            float("nan"),
        )

    mean = float(
        np.mean(
            array
        )
    )

    if len(
        array
    ) >= 2:

        std = float(
            np.std(
                array,
                ddof=1,
            )
        )

    else:

        std = 0.0

    return (
        mean,
        std,
    )


def fmt(
    value,
    digits=3,
):

    if not math.isfinite(
        float(
            value
        )
    ):

        return "nan"

    return (
        f"{float(value):.{digits}f}"
    )


def fmt_mean_std(
    values,
    digits=3,
):

    mean, std = mean_std(
        values
    )

    return (
        f"{mean:.{digits}f} "
        f"+/- "
        f"{std:.{digits}f}"
    )


# ============================================================
# Metric definitions
# ============================================================

CONDITION_METRICS = {

    "brake_onset_s":
        (
            "brake_onset",
            "t_s",
        ),

    "gap_at_brake_m":
        (
            "brake_onset",
            "gap_m",
        ),

    "stop_onset_s":
        (
            "stop_onset",
            "t_s",
        ),

    "gap_at_stop_m":
        (
            "stop_onset",
            "gap_m",
        ),

    "minimum_gap_m":
        (
            "minimum_gap_post_event_m",
        ),

    "final_gap_m":
        (
            "final_gap_m",
        ),

    "post_event_travel_m":
        (
            "post_event_travel_m",
        ),

    "mean_speed_post_event_mps":
        (
            "mean_speed_post_event_mps",
        ),

    "brake_fraction_post_event":
        (
            "brake_fraction_post_event",
        ),
}


PAIRED_METRICS = {

    "speed_mae_post_event_mps":
        (
            "speed_mae_post_event_mps",
        ),

    "gap_mae_post_event_m":
        (
            "gap_mae_post_event_m",
        ),

    "steer_mae_post_event":
        (
            "steer_mae_post_event",
        ),

    "throttle_mae_post_event":
        (
            "throttle_mae_post_event",
        ),

    "brake_mae_post_event":
        (
            "brake_mae_post_event",
        ),

    "brake_disagreement_fraction":
        (
            "brake_disagreement_fraction",
        ),
}


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=str(
            DEFAULT_ROOT
        ),
    )

    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).resolve()

    summaries = []

    # ========================================================
    # Load pair summaries
    # ========================================================

    for pair_idx in range(
        1,
        args.repeats + 1,
    ):

        pair_name = (
            f"pair_{pair_idx:02d}"
        )

        path = (
            root
            / pair_name
            / args.scenario
            / "analysis"
            / (
                args.scenario
                + "_pair_summary.json"
            )
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing pair summary:\n{path}"
            )

        summary = load_json(
            path
        )

        summaries.append(
            {
                "pair_idx":
                    pair_idx,

                "pair_name":
                    pair_name,

                "path":
                    str(
                        path
                    ),

                "summary":
                    summary,
            }
        )

    # ========================================================
    # Collect condition metrics
    # ========================================================

    carla_values = {
        metric: []
        for metric
        in CONDITION_METRICS
    }

    he_values = {
        metric: []
        for metric
        in CONDITION_METRICS
    }

    delta_values = {
        metric: []
        for metric
        in CONDITION_METRICS
    }

    paired_values = {
        metric: []
        for metric
        in PAIRED_METRICS
    }

    carla_collisions = 0
    he_collisions = 0

    pair_rows = []

    for item in summaries:

        pair_idx = item[
            "pair_idx"
        ]

        summary = item[
            "summary"
        ]

        carla = summary[
            "carla"
        ]

        he = summary[
            "he"
        ]

        paired = summary[
            "paired_time_series"
        ]

        row = {
            "pair":
                pair_idx,
        }

        # ----------------------------------------------------
        # CARLA / HE / deltas
        # ----------------------------------------------------

        for (
            metric_name,
            keys,
        ) in CONDITION_METRICS.items():

            carla_value = nested_get(
                carla,
                keys,
            )

            he_value = nested_get(
                he,
                keys,
            )

            if (
                math.isfinite(
                    carla_value
                )
                and
                math.isfinite(
                    he_value
                )
            ):

                delta_value = (
                    he_value
                    -
                    carla_value
                )

            else:

                delta_value = (
                    float("nan")
                )

            carla_values[
                metric_name
            ].append(
                carla_value
            )

            he_values[
                metric_name
            ].append(
                he_value
            )

            delta_values[
                metric_name
            ].append(
                delta_value
            )

            row[
                "carla_"
                + metric_name
            ] = (
                carla_value
            )

            row[
                "he_"
                + metric_name
            ] = (
                he_value
            )

            row[
                "delta_"
                + metric_name
            ] = (
                delta_value
            )

        # ----------------------------------------------------
        # Paired time-series metrics
        # ----------------------------------------------------

        for (
            metric_name,
            keys,
        ) in PAIRED_METRICS.items():

            value = nested_get(
                paired,
                keys,
            )

            paired_values[
                metric_name
            ].append(
                value
            )

        # ----------------------------------------------------
        # Collisions
        # ----------------------------------------------------

        carla_collision = bool(
            carla.get(
                "collision",
                False,
            )
        )

        he_collision = bool(
            he.get(
                "collision",
                False,
            )
        )

        carla_collisions += int(
            carla_collision
        )

        he_collisions += int(
            he_collision
        )

        row[
            "carla_collision"
        ] = (
            carla_collision
        )

        row[
            "he_collision"
        ] = (
            he_collision
        )

        pair_rows.append(
            row
        )

    # ========================================================
    # Individual pair table
    # ========================================================

    print()
    print("=" * 100)
    print(
        "NEAT LEAD-BRAKE REPEATED CARLA <-> HE"
    )
    print("=" * 100)

    print()
    print(
        "%-6s %10s %10s %10s "
        "%10s %10s %10s "
        "%10s %10s %10s"
        %
        (
            "Pair",

            "C brake",
            "H brake",
            "Delta",

            "C stop",
            "H stop",
            "Delta",

            "C minGap",
            "H minGap",
            "Delta",
        )
    )

    print("-" * 100)

    for row in pair_rows:

        print(
            "%-6d %10s %10s %10s "
            "%10s %10s %10s "
            "%10s %10s %10s"
            %
            (
                row[
                    "pair"
                ],

                fmt(
                    row[
                        "carla_brake_onset_s"
                    ]
                ),

                fmt(
                    row[
                        "he_brake_onset_s"
                    ]
                ),

                fmt(
                    row[
                        "delta_brake_onset_s"
                    ]
                ),

                fmt(
                    row[
                        "carla_stop_onset_s"
                    ]
                ),

                fmt(
                    row[
                        "he_stop_onset_s"
                    ]
                ),

                fmt(
                    row[
                        "delta_stop_onset_s"
                    ]
                ),

                fmt(
                    row[
                        "carla_minimum_gap_m"
                    ]
                ),

                fmt(
                    row[
                        "he_minimum_gap_m"
                    ]
                ),

                fmt(
                    row[
                        "delta_minimum_gap_m"
                    ]
                ),
            )
        )

    # ========================================================
    # Aggregate table
    # ========================================================

    print()
    print("=" * 100)
    print(
        "AGGREGATE: mean +/- sample std"
    )
    print("=" * 100)

    pretty_names = {

        "brake_onset_s":
            "Brake onset (s)",

        "gap_at_brake_m":
            "Gap at brake onset (m)",

        "stop_onset_s":
            "Stop onset (s)",

        "gap_at_stop_m":
            "Gap at stop (m)",

        "minimum_gap_m":
            "Minimum gap (m)",

        "final_gap_m":
            "Final gap (m)",

        "post_event_travel_m":
            "Post-event travel (m)",

        "mean_speed_post_event_mps":
            "Mean post-event speed",

        "brake_fraction_post_event":
            "Brake fraction",
    }

    print()
    print(
        "%-30s %-22s %-22s %-22s"
        %
        (
            "Metric",
            "CARLA",
            "HE",
            "HE-CARLA",
        )
    )

    print("-" * 100)

    for metric_name in CONDITION_METRICS:

        print(
            "%-30s %-22s %-22s %-22s"
            %
            (
                pretty_names[
                    metric_name
                ],

                fmt_mean_std(
                    carla_values[
                        metric_name
                    ]
                ),

                fmt_mean_std(
                    he_values[
                        metric_name
                    ]
                ),

                fmt_mean_std(
                    delta_values[
                        metric_name
                    ]
                ),
            )
        )

    # ========================================================
    # Directional consistency
    # ========================================================

    brake_later = sum(
        value > 0.0
        for value
        in delta_values[
            "brake_onset_s"
        ]
        if math.isfinite(
            value
        )
    )

    stop_closer = sum(
        value < 0.0
        for value
        in delta_values[
            "gap_at_stop_m"
        ]
        if math.isfinite(
            value
        )
    )

    min_gap_smaller = sum(
        value < 0.0
        for value
        in delta_values[
            "minimum_gap_m"
        ]
        if math.isfinite(
            value
        )
    )

    final_gap_smaller = sum(
        value < 0.0
        for value
        in delta_values[
            "final_gap_m"
        ]
        if math.isfinite(
            value
        )
    )

    travel_farther = sum(
        value > 0.0
        for value
        in delta_values[
            "post_event_travel_m"
        ]
        if math.isfinite(
            value
        )
    )

    print()
    print("=" * 100)
    print(
        "DIRECTIONAL CONSISTENCY"
    )
    print("=" * 100)

    print(
        "HE brakes later:",
        f"{brake_later}/{args.repeats}",
    )

    print(
        "HE stops closer:",
        f"{stop_closer}/{args.repeats}",
    )

    print(
        "HE has smaller minimum gap:",
        f"{min_gap_smaller}/{args.repeats}",
    )

    print(
        "HE has smaller final gap:",
        f"{final_gap_smaller}/{args.repeats}",
    )

    print(
        "HE travels farther post-event:",
        f"{travel_farther}/{args.repeats}",
    )

    print()

    print(
        "CARLA collisions:",
        f"{carla_collisions}/{args.repeats}",
    )

    print(
        "HE collisions:",
        f"{he_collisions}/{args.repeats}",
    )

    # ========================================================
    # Paired trajectory/control differences
    # ========================================================

    print()
    print("=" * 100)
    print(
        "PAIRED POST-EVENT DIFFERENCES"
    )
    print("=" * 100)

    paired_pretty = {

        "speed_mae_post_event_mps":
            "Speed MAE (m/s)",

        "gap_mae_post_event_m":
            "Gap MAE (m)",

        "steer_mae_post_event":
            "Steer MAE",

        "throttle_mae_post_event":
            "Throttle MAE",

        "brake_mae_post_event":
            "Brake MAE",

        "brake_disagreement_fraction":
            "Brake disagreement",
    }

    for metric_name in PAIRED_METRICS:

        mean, std = mean_std(
            paired_values[
                metric_name
            ]
        )

        if (
            metric_name
            ==
            "brake_disagreement_fraction"
        ):

            print(
                f"{paired_pretty[metric_name]:28s}: "
                f"{100.0 * mean:.2f}% "
                f"+/- "
                f"{100.0 * std:.2f}%"
            )

        else:

            print(
                f"{paired_pretty[metric_name]:28s}: "
                f"{mean:.4f} "
                f"+/- "
                f"{std:.4f}"
            )

    # ========================================================
    # Save aggregate JSON
    # ========================================================

    aggregate = {
        "scenario_id":
            args.scenario,

        "num_pairs":
            args.repeats,

        "pairs":
            pair_rows,

        "aggregate": {},

        "directional_consistency": {
            "he_brakes_later":
                brake_later,

            "he_stops_closer":
                stop_closer,

            "he_smaller_minimum_gap":
                min_gap_smaller,

            "he_smaller_final_gap":
                final_gap_smaller,

            "he_farther_post_event_travel":
                travel_farther,

            "carla_collisions":
                carla_collisions,

            "he_collisions":
                he_collisions,
        },

        "paired_time_series": {},
    }

    for metric_name in CONDITION_METRICS:

        carla_mean, carla_std = (
            mean_std(
                carla_values[
                    metric_name
                ]
            )
        )

        he_mean, he_std = (
            mean_std(
                he_values[
                    metric_name
                ]
            )
        )

        delta_mean, delta_std = (
            mean_std(
                delta_values[
                    metric_name
                ]
            )
        )

        aggregate[
            "aggregate"
        ][
            metric_name
        ] = {
            "carla_mean":
                carla_mean,

            "carla_std":
                carla_std,

            "he_mean":
                he_mean,

            "he_std":
                he_std,

            "delta_mean":
                delta_mean,

            "delta_std":
                delta_std,
        }

    for metric_name in PAIRED_METRICS:

        mean, std = mean_std(
            paired_values[
                metric_name
            ]
        )

        aggregate[
            "paired_time_series"
        ][
            metric_name
        ] = {
            "mean":
                mean,

            "std":
                std,
        }

    analysis_dir = (
        root
        / "analysis"
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        analysis_dir
        /
        (
            args.scenario
            + "_repeat_summary.json"
        )
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            aggregate,
            f,
            indent=2,
        )

    print()
    print(
        "Aggregate summary:",
        output_path,
    )

    print("=" * 100)


if __name__ == "__main__":

    main()