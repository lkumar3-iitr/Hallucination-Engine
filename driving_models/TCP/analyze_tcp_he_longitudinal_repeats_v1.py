"""
analyze_tcp_he_longitudinal_repeats_v1.py

Longitudinal-response analysis for repeated CARLA-vs-HE TCP trials.

Purpose
-------
The repeated experiment showed:

    sustained-brake timing ~= similar
    stop time             ~= similar
    HE final gap          ~= 1.1 m smaller

This script diagnoses WHERE that distance difference accumulates.

It computes:
    - throttle-release time
    - sustained-brake onset
    - peak 0.15 s deceleration
    - post-event distance travelled
    - distance travelled after sustained braking
    - throttle command integral
    - brake command integral
    - strong-brake duration
    - paired gap evolution over time

It also saves an aggregate time-series CSV that we can use later
for paper plots without rerunning the experiments.
"""

import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# Paths / experiment constants
# ============================================================

THIS_FILE = Path(__file__).resolve()
TCP_DIR = THIS_FILE.parent

SCENARIO_ID = "tcp_lead_brake_001"

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

EVENT_START_S = 4.0

BRAKE_THRESHOLD = 0.5
BRAKE_CONFIRM_FRAMES = 3

THROTTLE_RELEASE_THRESHOLD = 0.10
THROTTLE_CONFIRM_FRAMES = 3

STOP_SPEED_MPS = 0.10
STOP_CONFIRM_FRAMES = 5

# 20 Hz -> 3 frames = 0.15 s
DECEL_WINDOW_FRAMES = 3

REPORT_TIMES_S = [
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
]


# ============================================================
# CSV helpers
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
            f"{run_dir / condition}, found {files}"
        )

    return files[0]


# ============================================================
# Basic arrays
# ============================================================

def arrays_from_rows(rows):

    return {
        "frame":
            np.array(
                [
                    int(
                        row["probe_idx"]
                    )
                    for row in rows
                ],
                dtype=int,
            ),

        "t":
            np.array(
                [
                    as_float(
                        row,
                        "t_s",
                    )
                    for row in rows
                ],
                dtype=float,
            ),

        "speed":
            np.array(
                [
                    as_float(
                        row,
                        "ego_speed_mps",
                    )
                    for row in rows
                ],
                dtype=float,
            ),

        "gap":
            np.array(
                [
                    as_float(
                        row,
                        "bumper_gap_m",
                    )
                    for row in rows
                ],
                dtype=float,
            ),

        "throttle":
            np.array(
                [
                    as_float(
                        row,
                        "tcp_throttle",
                    )
                    for row in rows
                ],
                dtype=float,
            ),

        "brake":
            np.array(
                [
                    as_float(
                        row,
                        "tcp_brake",
                    )
                    for row in rows
                ],
                dtype=float,
            ),

        "steer":
            np.array(
                [
                    as_float(
                        row,
                        "tcp_steer",
                    )
                    for row in rows
                ],
                dtype=float,
            ),
    }


# ============================================================
# Event detectors
# ============================================================

def first_sustained_brake(a):

    brake = a["brake"]
    t = a["t"]

    for i in range(
        len(t)
        - BRAKE_CONFIRM_FRAMES
        + 1
    ):

        if t[i] < EVENT_START_S:
            continue

        window = brake[
            i:
            i + BRAKE_CONFIRM_FRAMES
        ]

        if all(
            np.isfinite(v)
            and
            v >= BRAKE_THRESHOLD
            for v in window
        ):
            return i

    return None


def first_sustained_throttle_release(a):

    throttle = a["throttle"]
    t = a["t"]

    for i in range(
        len(t)
        - THROTTLE_CONFIRM_FRAMES
        + 1
    ):

        if t[i] < EVENT_START_S:
            continue

        window = throttle[
            i:
            i + THROTTLE_CONFIRM_FRAMES
        ]

        if all(
            np.isfinite(v)
            and
            v <= THROTTLE_RELEASE_THRESHOLD
            for v in window
        ):
            return i

    return None


def first_confirmed_stop(a):

    speed = a["speed"]
    t = a["t"]

    for i in range(
        len(t)
        - STOP_CONFIRM_FRAMES
        + 1
    ):

        window = speed[
            i:
            i + STOP_CONFIRM_FRAMES
        ]

        if all(
            np.isfinite(v)
            and
            v <= STOP_SPEED_MPS
            for v in window
        ):
            return i

    return None


# ============================================================
# Numerical integration
# ============================================================

def integrate_signal(
    t,
    values,
    start_time=None,
    end_time=None,
):

    mask = (
        np.isfinite(t)
        &
        np.isfinite(values)
    )

    if start_time is not None:
        mask &= (
            t >= float(start_time)
        )

    if end_time is not None:
        mask &= (
            t <= float(end_time)
        )

    tt = t[mask]
    vv = values[mask]

    if len(tt) < 2:
        return float("nan")

    return float(
        np.trapezoid(
            vv,
            tt,
        )
    )


# ============================================================
# Deceleration metric
# ============================================================

def peak_window_deceleration(a):
    """
    Maximum speed reduction over a fixed 3-frame window.

    At 20 Hz this corresponds to approximately 0.15 s.

    Positive result means deceleration magnitude.
    """

    t = a["t"]
    speed = a["speed"]

    best = float("nan")

    for i in range(
        len(t)
        - DECEL_WINDOW_FRAMES
    ):

        j = (
            i
            + DECEL_WINDOW_FRAMES
        )

        if t[i] < EVENT_START_S:
            continue

        if not (
            np.isfinite(speed[i])
            and
            np.isfinite(speed[j])
            and
            np.isfinite(t[i])
            and
            np.isfinite(t[j])
        ):
            continue

        dt = (
            t[j]
            -
            t[i]
        )

        if dt <= 0:
            continue

        decel = (
            speed[i]
            -
            speed[j]
        ) / dt

        if (
            not np.isfinite(best)
            or
            decel > best
        ):
            best = float(
                decel
            )

    return best


# ============================================================
# Per-run metrics
# ============================================================

def compute_longitudinal_metrics(rows):

    a = arrays_from_rows(
        rows
    )

    brake_idx = (
        first_sustained_brake(
            a
        )
    )

    throttle_release_idx = (
        first_sustained_throttle_release(
            a
        )
    )

    stop_idx = (
        first_confirmed_stop(
            a
        )
    )

    if brake_idx is not None:
        brake_time = float(
            a["t"][brake_idx]
        )
    else:
        brake_time = float(
            "nan"
        )

    if throttle_release_idx is not None:
        throttle_release_time = float(
            a["t"][
                throttle_release_idx
            ]
        )
    else:
        throttle_release_time = float(
            "nan"
        )

    if stop_idx is not None:
        stop_time = float(
            a["t"][stop_idx]
        )
    else:
        stop_time = float(
            "nan"
        )

    # Integral of speed ~= distance travelled.
    post_event_distance = (
        integrate_signal(
            a["t"],
            a["speed"],
            start_time=
                EVENT_START_S,
        )
    )

    if np.isfinite(
        brake_time
    ):
        distance_after_brake = (
            integrate_signal(
                a["t"],
                a["speed"],
                start_time=
                    brake_time,
            )
        )
    else:
        distance_after_brake = (
            float("nan")
        )

    throttle_integral = (
        integrate_signal(
            a["t"],
            a["throttle"],
            start_time=
                EVENT_START_S,
        )
    )

    brake_integral = (
        integrate_signal(
            a["t"],
            a["brake"],
            start_time=
                EVENT_START_S,
        )
    )

    strong_brake_mask = (
        (a["t"] >= EVENT_START_S)
        &
        np.isfinite(
            a["brake"]
        )
        &
        (
            a["brake"]
            >= BRAKE_THRESHOLD
        )
    )

    if np.sum(
        strong_brake_mask
    ) > 0:
        # Each logged frame represents ~0.05 s.
        dt_values = np.diff(
            a["t"]
        )

        valid_dt = dt_values[
            np.isfinite(
                dt_values
            )
            &
            (
                dt_values > 0
            )
        ]

        if len(valid_dt):
            nominal_dt = float(
                np.median(
                    valid_dt
                )
            )
        else:
            nominal_dt = 0.05

        strong_brake_duration = (
            float(
                np.sum(
                    strong_brake_mask
                )
            )
            * nominal_dt
        )
    else:
        strong_brake_duration = 0.0

    finite_gap = a["gap"][
        np.isfinite(
            a["gap"]
        )
    ]

    return {
        "arrays":
            a,

        "throttle_release_s":
            throttle_release_time,

        "sustained_brake_s":
            brake_time,

        "stop_time_s":
            stop_time,

        "peak_decel_mps2":
            peak_window_deceleration(
                a
            ),

        "post_event_distance_m":
            post_event_distance,

        "distance_after_brake_m":
            distance_after_brake,

        "throttle_integral":
            throttle_integral,

        "brake_integral":
            brake_integral,

        "strong_brake_duration_s":
            strong_brake_duration,

        "final_gap_m":
            (
                float(
                    finite_gap[-1]
                )
                if len(finite_gap)
                else float("nan")
            ),
    }


# ============================================================
# Statistics helpers
# ============================================================

def finite_array(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return values[
        np.isfinite(
            values
        )
    ]


def mean_std(values):

    values = finite_array(
        values
    )

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    mean = float(
        np.mean(
            values
        )
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

    return (
        mean,
        std,
    )


def print_stat(
    label,
    values,
    unit="",
):

    mean, std = mean_std(
        values
    )

    print(
        f"{label:32s}: "
        f"{mean:9.3f} +/- "
        f"{std:8.3f} {unit}"
    )


# ============================================================
# Time lookup
# ============================================================

def nearest_value(
    arrays,
    key,
    target_t,
):

    t = arrays["t"]

    valid = (
        np.isfinite(t)
        &
        np.isfinite(
            arrays[key]
        )
    )

    if not np.any(valid):
        return float("nan")

    indices = np.where(
        valid
    )[0]

    idx = indices[
        np.argmin(
            np.abs(
                t[indices]
                -
                target_t
            )
        )
    ]

    return float(
        arrays[key][idx]
    )


# ============================================================
# Aggregate time-series
# ============================================================

def build_aggregate_timeseries(
    results,
):

    # All runs should have identical probe-frame indexing.
    frames = results[0][
        "carla"
    ]["arrays"]["frame"]

    rows = []

    keys = [
        "speed",
        "gap",
        "throttle",
        "brake",
        "steer",
    ]

    for idx, frame in enumerate(
        frames
    ):

        row = {
            "probe_idx":
                int(frame),

            "t_s":
                float(
                    results[0][
                        "carla"
                    ]["arrays"]["t"][idx]
                ),
        }

        for condition in (
            "carla",
            "he",
        ):

            for key in keys:

                values = []

                for result in results:

                    a = result[
                        condition
                    ]["arrays"]

                    if idx >= len(
                        a[key]
                    ):
                        continue

                    value = (
                        a[key][idx]
                    )

                    if np.isfinite(
                        value
                    ):
                        values.append(
                            float(value)
                        )

                mean, std = (
                    mean_std(
                        values
                    )
                )

                row[
                    f"{condition}_{key}_mean"
                ] = mean

                row[
                    f"{condition}_{key}_std"
                ] = std

        # Useful directly for later plotting.
        row[
            "he_minus_carla_gap_mean"
        ] = (
            row[
                "he_gap_mean"
            ]
            -
            row[
                "carla_gap_mean"
            ]
        )

        row[
            "he_minus_carla_speed_mean"
        ] = (
            row[
                "he_speed_mean"
            ]
            -
            row[
                "carla_speed_mean"
            ]
        )

        rows.append(
            row
        )

    return rows


# ============================================================
# Main
# ============================================================

def main():

    if not ROOT.exists():
        raise FileNotFoundError(
            ROOT
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
            f"No repeated runs found in {ROOT}"
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

        results.append(
            {
                "run":
                    run_number,

                "carla":
                    compute_longitudinal_metrics(
                        carla_rows
                    ),

                "he":
                    compute_longitudinal_metrics(
                        he_rows
                    ),
            }
        )

    print()
    print("=" * 104)
    print("TCP CARLA vs HE LONGITUDINAL RESPONSE ANALYSIS")
    print("=" * 104)

    print()
    print("PER-RUN LONGITUDINAL RESPONSE")
    print("-" * 104)

    print(
        "run | "
        "C brake  H brake | "
        "C dist(post) H dist(post) delta | "
        "C brakeInt H brakeInt | "
        "final-gap delta"
    )

    print("-" * 104)

    for result in results:

        c = result["carla"]
        h = result["he"]

        print(
            f"{result['run']:3d} | "
            f"{c['sustained_brake_s']:7.2f} "
            f"{h['sustained_brake_s']:7.2f} | "
            f"{c['post_event_distance_m']:11.3f} "
            f"{h['post_event_distance_m']:11.3f} "
            f"{h['post_event_distance_m'] - c['post_event_distance_m']:+7.3f} | "
            f"{c['brake_integral']:10.3f} "
            f"{h['brake_integral']:10.3f} | "
            f"{h['final_gap_m'] - c['final_gap_m']:+8.3f}"
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
        print("-" * 104)

        print_stat(
            "throttle release",
            [
                r[condition][
                    "throttle_release_s"
                ]
                for r in results
            ],
            "s",
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

        print_stat(
            "confirmed stop",
            [
                r[condition][
                    "stop_time_s"
                ]
                for r in results
            ],
            "s",
        )

        print_stat(
            "peak 0.15s deceleration",
            [
                r[condition][
                    "peak_decel_mps2"
                ]
                for r in results
            ],
            "m/s^2",
        )

        print_stat(
            "distance after event",
            [
                r[condition][
                    "post_event_distance_m"
                ]
                for r in results
            ],
            "m",
        )

        print_stat(
            "distance after brake onset",
            [
                r[condition][
                    "distance_after_brake_m"
                ]
                for r in results
            ],
            "m",
        )

        print_stat(
            "throttle integral",
            [
                r[condition][
                    "throttle_integral"
                ]
                for r in results
            ],
            "cmd*s",
        )

        print_stat(
            "brake integral",
            [
                r[condition][
                    "brake_integral"
                ]
                for r in results
            ],
            "cmd*s",
        )

        print_stat(
            "strong brake duration",
            [
                r[condition][
                    "strong_brake_duration_s"
                ]
                for r in results
            ],
            "s",
        )

    # --------------------------------------------------------
    # Paired differences
    # --------------------------------------------------------

    metrics = [
        (
            "throttle release",
            "throttle_release_s",
            "s",
        ),
        (
            "sustained brake",
            "sustained_brake_s",
            "s",
        ),
        (
            "confirmed stop",
            "stop_time_s",
            "s",
        ),
        (
            "peak deceleration",
            "peak_decel_mps2",
            "m/s^2",
        ),
        (
            "distance after event",
            "post_event_distance_m",
            "m",
        ),
        (
            "distance after brake onset",
            "distance_after_brake_m",
            "m",
        ),
        (
            "throttle integral",
            "throttle_integral",
            "cmd*s",
        ),
        (
            "brake integral",
            "brake_integral",
            "cmd*s",
        ),
        (
            "strong brake duration",
            "strong_brake_duration_s",
            "s",
        ),
        (
            "final gap",
            "final_gap_m",
            "m",
        ),
    ]

    print()
    print("PAIRED HE - CARLA DIFFERENCES")
    print("-" * 104)

    paired_summary = {}

    for label, key, unit in metrics:

        values = [
            r["he"][key]
            -
            r["carla"][key]
            for r in results
        ]

        print_stat(
            f"delta {label}",
            values,
            unit,
        )

        mean, std = mean_std(
            values
        )

        paired_summary[key] = {
            "mean":
                mean,

            "std":
                std,

            "values":
                [
                    float(v)
                    for v in values
                ],
        }

    # --------------------------------------------------------
    # Where does the gap bias accumulate?
    # --------------------------------------------------------

    print()
    print("WHERE DOES THE GAP DIFFERENCE ACCUMULATE?")
    print("-" * 104)

    print(
        "time | mean HE-CARLA gap | std | "
        "change relative to t=4.0 s"
    )

    print("-" * 104)

    baseline_deltas = []

    for result in results:

        c_gap = nearest_value(
            result[
                "carla"
            ]["arrays"],
            "gap",
            EVENT_START_S,
        )

        h_gap = nearest_value(
            result[
                "he"
            ]["arrays"],
            "gap",
            EVENT_START_S,
        )

        baseline_deltas.append(
            h_gap
            -
            c_gap
        )

    baseline_mean, _ = mean_std(
        baseline_deltas
    )

    temporal_gap_summary = []

    for target_t in REPORT_TIMES_S:

        deltas = []

        for result in results:

            c_gap = nearest_value(
                result[
                    "carla"
                ]["arrays"],
                "gap",
                target_t,
            )

            h_gap = nearest_value(
                result[
                    "he"
                ]["arrays"],
                "gap",
                target_t,
            )

            deltas.append(
                h_gap
                -
                c_gap
            )

        mean, std = mean_std(
            deltas
        )

        accumulated = (
            mean
            -
            baseline_mean
        )

        print(
            f"{target_t:4.1f} | "
            f"{mean:+18.3f} m | "
            f"{std:6.3f} | "
            f"{accumulated:+10.3f} m"
        )

        temporal_gap_summary.append(
            {
                "t_s":
                    target_t,

                "gap_delta_mean_m":
                    mean,

                "gap_delta_std_m":
                    std,

                "change_from_event_start_m":
                    accumulated,
            }
        )

    # --------------------------------------------------------
    # Save aggregate time series for future plotting
    # --------------------------------------------------------

    aggregate_rows = (
        build_aggregate_timeseries(
            results
        )
    )

    aggregate_csv = (
        OUTPUT_DIR
        / "tcp_lead_brake_001_aggregate_timeseries.csv"
    )

    with aggregate_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                aggregate_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            aggregate_rows
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "scenario_id":
            SCENARIO_ID,

        "num_pairs":
            len(results),

        "event_start_s":
            EVENT_START_S,

        "paired_differences":
            paired_summary,

        "temporal_gap_difference":
            temporal_gap_summary,
    }

    summary_json = (
        OUTPUT_DIR
        / "tcp_lead_brake_001_longitudinal_summary.json"
    )

    with summary_json.open(
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
    print("-" * 104)

    print(
        "aggregate CSV :",
        aggregate_csv,
    )

    print(
        "summary JSON  :",
        summary_json,
    )

    print()
    print("=" * 104)


if __name__ == "__main__":
    main()