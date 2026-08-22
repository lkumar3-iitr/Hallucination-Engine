"""
analyze_tcp_he_repeats_v1.py

Aggregate repeated TCP CARLA <-> HE cut-in experiments.

Nothing from the single-pair evaluator is removed.

Reports:

    CARLA mean +/- sample std
    HE mean +/- sample std
    paired HE-CARLA mean +/- sample std

for scenario events, braking, safety outcomes and paired
control/state divergence.

Also aggregates short-horizon and full closed-loop metrics.
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
    / "TCP"
    / "outputs"
    / "tcp_4320_repeats_v1"
)


DEFAULT_SCENARIO = (
    "tcp_cutin_001"
)


# ============================================================
# Condition metrics retained from pair analyzer
# ============================================================

CONDITION_METRICS = [

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


WINDOWS = [

    "frames_0_20",
    "frames_0_40",
    "frames_0_80",

    "full",

    "post_cutin",
    "post_boundary",
]


NUMERIC_SERIES = [

    "tcp_steer",
    "tcp_throttle",
    "tcp_brake",

    "ego_speed_mps",
    "bumper_gap_m",

    "route_deviation_m",

    "desired_speed",
    "pred_speed",
]


# ============================================================
# Helpers
# ============================================================

def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as fp:

        return json.load(
            fp
        )


def nested_get(
    data,
    keys,
):

    current = data

    for key in keys:

        if (
            not isinstance(
                current,
                dict,
            )
            or
            key not in current
        ):

            return float("nan")

        current = current[
            key
        ]

    if current is None:

        return float("nan")

    try:

        value = float(
            current
        )

    except (
        TypeError,
        ValueError,
    ):

        return float("nan")

    return value


def finite_values(
    values,
):

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return array[
        np.isfinite(
            array
        )
    ]


def summarize(
    values,
):

    values = finite_values(
        values
    )

    if len(values) == 0:

        return {
            "n": 0,
            "mean": None,
            "sample_std": None,
            "median": None,
            "min": None,
            "max": None,
        }

    if len(values) >= 2:

        sample_std = float(
            np.std(
                values,
                ddof=1,
            )
        )

    else:

        sample_std = 0.0

    return {

        "n":
            int(
                len(values)
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "sample_std":
            sample_std,

        "median":
            float(
                np.median(
                    values
                )
            ),

        "min":
            float(
                np.min(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),
    }


def fmt_mean_std(
    summary,
    digits=4,
):

    mean = summary.get(
        "mean"
    )

    std = summary.get(
        "sample_std"
    )

    if (
        mean is None
        or
        std is None
    ):

        return "none"

    return (
        f"{mean:.{digits}f}"
        f" +/- "
        f"{std:.{digits}f}"
    )


def json_safe(
    value,
):

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
            for item in value
        ]

    if isinstance(
        value,
        (
            np.floating,
            float,
        ),
    ):

        value = float(
            value
        )

        if not math.isfinite(
            value
        ):

            return None

        return value

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    return value


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

        pair_root = (
            root
            / pair_name
        )

        summary_path = (
            pair_root
            / args.scenario
            / "analysis"
            / "tcp_cutin_pair_summary.json"
        )

        if not summary_path.exists():

            raise FileNotFoundError(
                "Missing pair summary:\n"
                f"{summary_path}"
            )

        summary = load_json(
            summary_path
        )

        pair_manifest_path = (
            pair_root
            / "pair_manifest.json"
        )

        if pair_manifest_path.exists():

            pair_manifest = (
                load_json(
                    pair_manifest_path
                )
            )

            order = (
                " -> ".join(
                    pair_manifest.get(
                        "condition_order",
                        [],
                    )
                )
            )

        else:

            order = ""

        summaries.append({

            "pair_idx":
                pair_idx,

            "pair_name":
                pair_name,

            "order":
                order,

            "summary":
                summary,

            "summary_path":
                str(
                    summary_path
                ),
        })

    # ========================================================
    # Aggregate condition metrics
    # ========================================================

    condition_stats = {}

    pair_table = []

    for metric in CONDITION_METRICS:

        carla_values = []

        he_values = []

        delta_values = []

        for item in summaries:

            summary = item[
                "summary"
            ]

            c = nested_get(
                summary,
                [
                    "carla",
                    metric,
                ],
            )

            h = nested_get(
                summary,
                [
                    "he",
                    metric,
                ],
            )

            d = nested_get(
                summary,
                [
                    "he_minus_carla",
                    metric,
                ],
            )

            carla_values.append(
                c
            )

            he_values.append(
                h
            )

            delta_values.append(
                d
            )

        condition_stats[
            metric
        ] = {

            "carla":
                summarize(
                    carla_values
                ),

            "he":
                summarize(
                    he_values
                ),

            "he_minus_carla":
                summarize(
                    delta_values
                ),
        }

    # ========================================================
    # Aggregate paired time-series metrics
    # ========================================================

    paired_window_stats = {}

    for window_name in WINDOWS:

        window_result = {}

        for series_name in NUMERIC_SERIES:

            metric_result = {}

            for statistic in (
                "mae",
                "rmse",
                "p95_abs",
                "max_abs",
                "bias_he_minus_carla",
                "correlation",
            ):

                values = []

                for item in summaries:

                    value = nested_get(
                        item[
                            "summary"
                        ],
                        [
                            "paired_windows",
                            window_name,
                            series_name,
                            statistic,
                        ],
                    )

                    values.append(
                        value
                    )

                metric_result[
                    statistic
                ] = summarize(
                    values
                )

            window_result[
                series_name
            ] = metric_result

        # ----------------------------------------------------
        # Brake agreement
        # ----------------------------------------------------

        brake_result = {}

        for statistic in (
            "agreement_fraction",
            "disagreement_fraction",
        ):

            values = []

            for item in summaries:

                value = nested_get(
                    item[
                        "summary"
                    ],
                    [
                        "paired_windows",
                        window_name,
                        "brake_decision",
                        statistic,
                    ],
                )

                values.append(
                    value
                )

            brake_result[
                statistic
            ] = summarize(
                values
            )

        window_result[
            "brake_decision"
        ] = brake_result

        # ----------------------------------------------------
        # Ego XY divergence
        # ----------------------------------------------------

        trajectory_result = {}

        for statistic in (
            "mean_m",
            "p95_m",
            "max_m",
            "final_m",
        ):

            values = []

            for item in summaries:

                value = nested_get(
                    item[
                        "summary"
                    ],
                    [
                        "paired_windows",
                        window_name,
                        "ego_xy_trajectory",
                        statistic,
                    ],
                )

                values.append(
                    value
                )

            trajectory_result[
                statistic
            ] = summarize(
                values
            )

        window_result[
            "ego_xy_trajectory"
        ] = trajectory_result

        paired_window_stats[
            window_name
        ] = window_result

    # ========================================================
    # Collision statistics
    # ========================================================

    carla_collision_count = 0

    he_collision_count = 0

    outcome_agreement_count = 0

    for item in summaries:

        summary = item[
            "summary"
        ]

        if summary[
            "carla"
        ][
            "collision"
        ]:

            carla_collision_count += 1

        if summary[
            "he"
        ][
            "collision"
        ]:

            he_collision_count += 1

        if summary.get(
            "collision_outcome_agreement",
            False,
        ):

            outcome_agreement_count += 1

    # ========================================================
    # Pair table
    # ========================================================

    for item in summaries:

        summary = item[
            "summary"
        ]

        pair_table.append({

            "pair":
                item[
                    "pair_idx"
                ],

            "order":
                item[
                    "order"
                ],

            "carla_sustained_brake_s":
                nested_get(
                    summary,
                    [
                        "carla",
                        "sustained_brake_s",
                    ],
                ),

            "he_sustained_brake_s":
                nested_get(
                    summary,
                    [
                        "he",
                        "sustained_brake_s",
                    ],
                ),

            "delta_sustained_brake_s":
                nested_get(
                    summary,
                    [
                        "he_minus_carla",
                        "sustained_brake_s",
                    ],
                ),

            "carla_min_gap_m":
                nested_get(
                    summary,
                    [
                        "carla",
                        "minimum_gap_m",
                    ],
                ),

            "he_min_gap_m":
                nested_get(
                    summary,
                    [
                        "he",
                        "minimum_gap_m",
                    ],
                ),

            "delta_min_gap_m":
                nested_get(
                    summary,
                    [
                        "he_minus_carla",
                        "minimum_gap_m",
                    ],
                ),

            "full_steer_mae":
                nested_get(
                    summary,
                    [
                        "paired_windows",
                        "full",
                        "tcp_steer",
                        "mae",
                    ],
                ),

            "full_throttle_mae":
                nested_get(
                    summary,
                    [
                        "paired_windows",
                        "full",
                        "tcp_throttle",
                        "mae",
                    ],
                ),

            "full_brake_mae":
                nested_get(
                    summary,
                    [
                        "paired_windows",
                        "full",
                        "tcp_brake",
                        "mae",
                    ],
                ),

            "full_speed_mae_mps":
                nested_get(
                    summary,
                    [
                        "paired_windows",
                        "full",
                        "ego_speed_mps",
                        "mae",
                    ],
                ),

            "full_gap_mae_m":
                nested_get(
                    summary,
                    [
                        "paired_windows",
                        "full",
                        "bumper_gap_m",
                        "mae",
                    ],
                ),

            "full_brake_agreement":
                nested_get(
                    summary,
                    [
                        "paired_windows",
                        "full",
                        "brake_decision",
                        "agreement_fraction",
                    ],
                ),

            "full_ego_xy_mean_m":
                nested_get(
                    summary,
                    [
                        "paired_windows",
                        "full",
                        "ego_xy_trajectory",
                        "mean_m",
                    ],
                ),

            "collision_agreement":
                bool(
                    summary.get(
                        "collision_outcome_agreement",
                        False,
                    )
                ),
        })

    # ========================================================
    # Final summary
    # ========================================================

    aggregate = {

        "schema":
            "tcp_he_repeat_analysis_v1",

        "scenario_id":
            args.scenario,

        "repeat_count":
            args.repeats,

        "condition_statistics":
            condition_stats,

        "paired_window_statistics":
            paired_window_stats,

        "collision_statistics": {

            "carla_collision_count":
                carla_collision_count,

            "he_collision_count":
                he_collision_count,

            "outcome_agreement_count":
                outcome_agreement_count,

            "outcome_agreement_fraction":
                (
                    outcome_agreement_count
                    /
                    args.repeats
                ),
        },

        "pairs":
            pair_table,
    }

    aggregate = json_safe(
        aggregate
    )

    output_json = (
        root
        / "tcp_repeat_summary.json"
    )

    with output_json.open(
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            aggregate,
            fp,
            indent=2,
        )

    # ========================================================
    # Pair-level CSV
    # ========================================================

    output_csv = (
        root
        / "tcp_repeat_pairs.csv"
    )

    if pair_table:

        with output_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as fp:

            writer = csv.DictWriter(
                fp,
                fieldnames=list(
                    pair_table[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()

            for row in pair_table:

                writer.writerow(
                    row
                )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 92)
    print(
        "TCP CARLA <-> HE REPEATABILITY ANALYSIS V1"
    )
    print("=" * 92)

    print(
        "Scenario:",
        args.scenario,
    )

    print(
        "Pairs:",
        args.repeats,
    )

    print()

    print(
        "Sustained brake time [s]"
    )

    print(
        "  CARLA:",
        fmt_mean_std(
            condition_stats[
                "sustained_brake_s"
            ][
                "carla"
            ]
        ),
    )

    print(
        "  HE   :",
        fmt_mean_std(
            condition_stats[
                "sustained_brake_s"
            ][
                "he"
            ]
        ),
    )

    print(
        "  HE-C :",
        fmt_mean_std(
            condition_stats[
                "sustained_brake_s"
            ][
                "he_minus_carla"
            ]
        ),
    )

    print()

    print(
        "Minimum bumper gap [m]"
    )

    print(
        "  CARLA:",
        fmt_mean_std(
            condition_stats[
                "minimum_gap_m"
            ][
                "carla"
            ]
        ),
    )

    print(
        "  HE   :",
        fmt_mean_std(
            condition_stats[
                "minimum_gap_m"
            ][
                "he"
            ]
        ),
    )

    print(
        "  HE-C :",
        fmt_mean_std(
            condition_stats[
                "minimum_gap_m"
            ][
                "he_minus_carla"
            ]
        ),
    )

    print()

    full = paired_window_stats[
        "full"
    ]

    early = paired_window_stats[
        "frames_0_20"
    ]

    print(
        "Early first-20-frame agreement"
    )

    print(
        "  steer MAE:",
        fmt_mean_std(
            early[
                "tcp_steer"
            ][
                "mae"
            ]
        ),
    )

    print(
        "  speed MAE:",
        fmt_mean_std(
            early[
                "ego_speed_mps"
            ][
                "mae"
            ]
        ),
        "m/s",
    )

    print(
        "  gap MAE:",
        fmt_mean_std(
            early[
                "bumper_gap_m"
            ][
                "mae"
            ]
        ),
        "m",
    )

    print(
        "  brake agreement:",
        fmt_mean_std(
            early[
                "brake_decision"
            ][
                "agreement_fraction"
            ]
        ),
    )

    print()

    print(
        "Full closed-loop pair differences"
    )

    print(
        "  steer MAE:",
        fmt_mean_std(
            full[
                "tcp_steer"
            ][
                "mae"
            ]
        ),
    )

    print(
        "  throttle MAE:",
        fmt_mean_std(
            full[
                "tcp_throttle"
            ][
                "mae"
            ]
        ),
    )

    print(
        "  brake MAE:",
        fmt_mean_std(
            full[
                "tcp_brake"
            ][
                "mae"
            ]
        ),
    )

    print(
        "  speed MAE:",
        fmt_mean_std(
            full[
                "ego_speed_mps"
            ][
                "mae"
            ]
        ),
        "m/s",
    )

    print(
        "  gap MAE:",
        fmt_mean_std(
            full[
                "bumper_gap_m"
            ][
                "mae"
            ]
        ),
        "m",
    )

    print(
        "  brake agreement:",
        fmt_mean_std(
            full[
                "brake_decision"
            ][
                "agreement_fraction"
            ]
        ),
    )

    print(
        "  ego XY mean divergence:",
        fmt_mean_std(
            full[
                "ego_xy_trajectory"
            ][
                "mean_m"
            ]
        ),
        "m",
    )

    print()

    print(
        "Collisions CARLA:",
        carla_collision_count,
        "/",
        args.repeats,
    )

    print(
        "Collisions HE:",
        he_collision_count,
        "/",
        args.repeats,
    )

    print(
        "Outcome agreement:",
        outcome_agreement_count,
        "/",
        args.repeats,
    )

    print()

    print(
        "JSON:",
        output_json,
    )

    print(
        "CSV :",
        output_csv,
    )

    print("=" * 92)


if __name__ == "__main__":

    main()