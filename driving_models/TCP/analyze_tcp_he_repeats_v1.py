"""
analyze_tcp_he_repeats_v1.py

Analyze repeated TCP closed-loop CARLA-vs-HE trials.

Expected structure:

outputs/tcp_he_repeat_v1/tcp_lead_brake_001/
    run_01/
        carla/*.csv
        he/*.csv
    ...
    run_05/
        carla/*.csv
        he/*.csv
"""

import csv
import math
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()

TCP_DIR = THIS_FILE.parent

SCENARIO_ID = "tcp_lead_brake_001"

ROOT = (
    TCP_DIR
    / "outputs"
    / "tcp_he_repeat_v1"
    / SCENARIO_ID
)


EVENT_START_S = 4.0

BRAKE_THRESHOLD = 0.5
BRAKE_CONFIRM_FRAMES = 3

STOP_SPEED_MPS = 0.10
STOP_CONFIRM_FRAMES = 5


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

    # Avoid accidentally using a comparison CSV.
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
# Event detection
# ============================================================

def first_brake_pulse(rows):

    for row in rows:

        t = as_float(
            row,
            "t_s",
        )

        brake = as_float(
            row,
            "tcp_brake",
        )

        if (
            t >= EVENT_START_S
            and
            np.isfinite(brake)
            and
            brake >= BRAKE_THRESHOLD
        ):
            return {
                "t_s":
                    t,

                "frame":
                    int(
                        row["probe_idx"]
                    ),

                "gap_m":
                    as_float(
                        row,
                        "bumper_gap_m",
                    ),
            }

    return None


def first_sustained_brake(rows):

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
            and
            b >= BRAKE_THRESHOLD
            for b in brakes
        ):
            row = rows[i]

            return {
                "t_s":
                    t,

                "frame":
                    int(
                        row["probe_idx"]
                    ),

                "gap_m":
                    as_float(
                        row,
                        "bumper_gap_m",
                    ),
            }

    return None


def confirmed_stop(rows):

    for i in range(
        len(rows)
        - STOP_CONFIRM_FRAMES
        + 1
    ):

        window = rows[
            i:
            i + STOP_CONFIRM_FRAMES
        ]

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
                "t_s":
                    as_float(
                        row,
                        "t_s",
                    ),

                "frame":
                    int(
                        row["probe_idx"]
                    ),

                "gap_m":
                    as_float(
                        row,
                        "bumper_gap_m",
                    ),
            }

    return None


# ============================================================
# Per-run metrics
# ============================================================

def compute_metrics(rows):

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

    finite_gaps = gaps[
        np.isfinite(gaps)
    ]

    first_brake = (
        first_brake_pulse(
            rows
        )
    )

    sustained = (
        first_sustained_brake(
            rows
        )
    )

    stop = (
        confirmed_stop(
            rows
        )
    )

    collision = False

    for row in rows:
        value = str(
            row.get(
                "virtual_collision",
                "",
            )
        ).strip().lower()

        if value in (
            "true",
            "1",
            "yes",
        ):
            collision = True
            break

    return {
        "first_brake_s":
            (
                first_brake["t_s"]
                if first_brake
                else float("nan")
            ),

        "sustained_brake_s":
            (
                sustained["t_s"]
                if sustained
                else float("nan")
            ),

        "sustained_brake_gap_m":
            (
                sustained["gap_m"]
                if sustained
                else float("nan")
            ),

        "stop_time_s":
            (
                stop["t_s"]
                if stop
                else float("nan")
            ),

        "stop_gap_m":
            (
                stop["gap_m"]
                if stop
                else float("nan")
            ),

        "minimum_gap_m":
            (
                float(
                    np.min(
                        finite_gaps
                    )
                )
                if len(finite_gaps)
                else float("nan")
            ),

        "final_gap_m":
            (
                float(
                    finite_gaps[-1]
                )
                if len(finite_gaps)
                else float("nan")
            ),

        "collision":
            collision,
    }


# ============================================================
# Statistics
# ============================================================

def mean_std(values):

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

    if len(values) > 1:
        std = float(
            np.std(
                values,
                ddof=1,
            )
        )
    else:
        std = 0.0

    return mean, std


def print_stat(label, values, unit):

    mean, std = mean_std(
        values
    )

    print(
        f"{label:27s}: "
        f"{mean:8.3f} +/- {std:7.3f} {unit}"
    )


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
            for p in ROOT.glob("run_*")
            if p.is_dir()
        ]
    )

    if not run_dirs:
        raise RuntimeError(
            f"No run directories found in {ROOT}"
        )

    results = []

    for run_dir in run_dirs:

        run_number = int(
            run_dir.name.split("_")[-1]
        )

        carla_csv = find_csv(
            run_dir,
            "carla",
        )

        he_csv = find_csv(
            run_dir,
            "he",
        )

        carla = compute_metrics(
            load_csv(
                carla_csv
            )
        )

        he = compute_metrics(
            load_csv(
                he_csv
            )
        )

        results.append(
            {
                "run":
                    run_number,

                "carla":
                    carla,

                "he":
                    he,
            }
        )

    print()
    print("=" * 100)
    print("TCP CARLA vs HE REPEATED-TRIAL ANALYSIS")
    print("=" * 100)

    print()
    print("PER-RUN OUTCOMES")
    print("-" * 100)

    print(
        "run | "
        "CARLA brake | HE brake | delta | "
        "CARLA final gap | HE final gap | delta | "
        "collision"
    )

    print("-" * 100)

    for result in results:

        c = result["carla"]
        h = result["he"]

        brake_delta = (
            h["sustained_brake_s"]
            -
            c["sustained_brake_s"]
        )

        gap_delta = (
            h["final_gap_m"]
            -
            c["final_gap_m"]
        )

        print(
            f"{result['run']:3d} | "
            f"{c['sustained_brake_s']:11.2f} | "
            f"{h['sustained_brake_s']:8.2f} | "
            f"{brake_delta:+6.2f} | "
            f"{c['final_gap_m']:15.3f} | "
            f"{h['final_gap_m']:12.3f} | "
            f"{gap_delta:+7.3f} | "
            f"C={int(c['collision'])} "
            f"H={int(h['collision'])}"
        )

    # --------------------------------------------------------
    # Condition distributions
    # --------------------------------------------------------

    print()
    print("CARLA DISTRIBUTION")
    print("-" * 100)

    print_stat(
        "sustained brake onset",
        [
            r["carla"][
                "sustained_brake_s"
            ]
            for r in results
        ],
        "s",
    )

    print_stat(
        "confirmed stop time",
        [
            r["carla"][
                "stop_time_s"
            ]
            for r in results
        ],
        "s",
    )

    print_stat(
        "gap at confirmed stop",
        [
            r["carla"][
                "stop_gap_m"
            ]
            for r in results
        ],
        "m",
    )

    print_stat(
        "minimum gap",
        [
            r["carla"][
                "minimum_gap_m"
            ]
            for r in results
        ],
        "m",
    )

    print_stat(
        "final gap",
        [
            r["carla"][
                "final_gap_m"
            ]
            for r in results
        ],
        "m",
    )


    print()
    print("HE DISTRIBUTION")
    print("-" * 100)

    print_stat(
        "sustained brake onset",
        [
            r["he"][
                "sustained_brake_s"
            ]
            for r in results
        ],
        "s",
    )

    print_stat(
        "confirmed stop time",
        [
            r["he"][
                "stop_time_s"
            ]
            for r in results
        ],
        "s",
    )

    print_stat(
        "gap at confirmed stop",
        [
            r["he"][
                "stop_gap_m"
            ]
            for r in results
        ],
        "m",
    )

    print_stat(
        "minimum gap",
        [
            r["he"][
                "minimum_gap_m"
            ]
            for r in results
        ],
        "m",
    )

    print_stat(
        "final gap",
        [
            r["he"][
                "final_gap_m"
            ]
            for r in results
        ],
        "m",
    )


    # --------------------------------------------------------
    # Paired differences
    # --------------------------------------------------------

    print()
    print("PAIRED HE - CARLA DIFFERENCES")
    print("-" * 100)

    brake_deltas = np.array(
        [
            r["he"][
                "sustained_brake_s"
            ]
            -
            r["carla"][
                "sustained_brake_s"
            ]
            for r in results
        ],
        dtype=float,
    )

    stop_deltas = np.array(
        [
            r["he"][
                "stop_time_s"
            ]
            -
            r["carla"][
                "stop_time_s"
            ]
            for r in results
        ],
        dtype=float,
    )

    stop_gap_deltas = np.array(
        [
            r["he"][
                "stop_gap_m"
            ]
            -
            r["carla"][
                "stop_gap_m"
            ]
            for r in results
        ],
        dtype=float,
    )

    min_gap_deltas = np.array(
        [
            r["he"][
                "minimum_gap_m"
            ]
            -
            r["carla"][
                "minimum_gap_m"
            ]
            for r in results
        ],
        dtype=float,
    )

    final_gap_deltas = np.array(
        [
            r["he"][
                "final_gap_m"
            ]
            -
            r["carla"][
                "final_gap_m"
            ]
            for r in results
        ],
        dtype=float,
    )

    print_stat(
        "delta sustained brake",
        brake_deltas,
        "s",
    )

    print_stat(
        "delta confirmed stop",
        stop_deltas,
        "s",
    )

    print_stat(
        "delta stop gap",
        stop_gap_deltas,
        "m",
    )

    print_stat(
        "delta minimum gap",
        min_gap_deltas,
        "m",
    )

    print_stat(
        "delta final gap",
        final_gap_deltas,
        "m",
    )


    # --------------------------------------------------------
    # Direction consistency
    # --------------------------------------------------------

    valid_gap_deltas = (
        final_gap_deltas[
            np.isfinite(
                final_gap_deltas
            )
        ]
    )

    lower_count = int(
        np.sum(
            valid_gap_deltas
            <
            0.0
        )
    )

    higher_count = int(
        np.sum(
            valid_gap_deltas
            >
            0.0
        )
    )

    print()
    print("DIRECTION CONSISTENCY")
    print("-" * 100)

    print(
        "HE stopped closer than CARLA :",
        f"{lower_count}/{len(valid_gap_deltas)} pairs",
    )

    print(
        "HE stopped farther than CARLA:",
        f"{higher_count}/{len(valid_gap_deltas)} pairs",
    )


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
    print("-" * 100)

    print(
        f"CARLA collisions: "
        f"{carla_collisions}/{len(results)}"
    )

    print(
        f"HE collisions   : "
        f"{he_collisions}/{len(results)}"
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()