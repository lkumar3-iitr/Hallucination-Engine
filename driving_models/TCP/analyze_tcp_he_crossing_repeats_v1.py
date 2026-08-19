"""
analyze_tcp_he_crossing_repeats_v1.py

Crossing-vehicle CARLA vs HE repeated-trial analysis.

Main behavioral sequence:

    approach
      ->
    brake
      ->
    stop before conflict point
      ->
    wait
      ->
    crossing actor clears
      ->
    restart
      ->
    recover speed

Important:
The old bumper_gap_m is not used as the crossing safety metric.

Instead we compute the ego front-bumper distance to the fixed
crossing conflict point.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

THIS_FILE = Path(__file__).resolve()
TCP_DIR = THIS_FILE.parent
HE_ROOT = TCP_DIR.parent.parent

SCENARIO_ID = "tcp_crossing_001"
ACTOR_ID = "adv_crossing"

ROOT = (
    TCP_DIR
    / "outputs"
    / "tcp_he_repeat_v1"
    / SCENARIO_ID
)

RESOLVED_PATH = (
    HE_ROOT
    / "ScenarioGenerator"
    / "outputs"
    / "v2_resolved"
    / f"{SCENARIO_ID}.resolved_v2.json"
)

OUTPUT_DIR = ROOT / "analysis"


BRAKE_THRESHOLD = 0.5
BRAKE_CONFIRM_FRAMES = 3

STOP_SPEED_MPS = 0.10
STOP_CONFIRM_FRAMES = 3

RESTART_SPEED_MPS = 0.50
RESTART_CONFIRM_FRAMES = 3

RECOVERY_SPEED_2_MPS = 2.0
RECOVERY_SPEED_4_MPS = 4.0

EVENT_START_S = 3.5


# ============================================================
# Utilities
# ============================================================

def as_float(row, key):

    value = str(
        row.get(key, "")
    ).strip()

    if not value:
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

    folder = (
        run_dir
        / condition
    )

    files = sorted(
        folder.glob("*.csv")
    )

    if len(files) != 1:

        raise RuntimeError(
            f"Expected exactly one CSV in "
            f"{folder}; found {files}"
        )

    return files[0]


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

    mean, std = mean_std(
        values
    )

    print(
        f"{label:42s}: "
        f"{mean:9.3f} +/- "
        f"{std:8.3f} {unit}"
    )


def first_sustained_condition(
    rows,
    start_idx,
    predicate,
    confirm_frames,
):

    if start_idx is None:
        return None

    end = (
        len(rows)
        - confirm_frames
        + 1
    )

    for i in range(
        start_idx,
        end,
    ):

        ok = True

        for j in range(
            confirm_frames
        ):

            if not predicate(
                rows[i + j]
            ):

                ok = False
                break

        if ok:
            return i

    return None


def time_at(rows, idx):

    if idx is None:
        return float("nan")

    return as_float(
        rows[idx],
        "t_s",
    )


# ============================================================
# Scenario geometry
# ============================================================

def load_actor_dimensions():

    data = json.loads(
        RESOLVED_PATH.read_text(
            encoding="utf-8",
        )
    )

    actor = next(
        actor
        for actor in data[
            "actors"
        ]
        if actor["actor_id"]
        == ACTOR_ID
    )

    dims = actor.get(
        "dimensions_m",
        {},
    )

    return {

        "length_m":
            float(
                dims.get(
                    "length_m",
                    4.2,
                )
            ),

        "width_m":
            float(
                dims.get(
                    "width_m",
                    1.8,
                )
            ),

        "height_m":
            float(
                dims.get(
                    "height_m",
                    1.5,
                )
            ),
    }


def find_conflict_point(rows):

    """
    The conflict point is where actor SG y is closest to zero.

    Because the actor moves perpendicular to the ego road,
    that world location is the center of the crossing.
    """

    valid = []

    for row in rows:

        y = as_float(
            row,
            "actor_y_sg_m",
        )

        wx = as_float(
            row,
            "actor_world_x",
        )

        wy = as_float(
            row,
            "actor_world_y",
        )

        if (
            np.isfinite(y)
            and np.isfinite(wx)
            and np.isfinite(wy)
        ):

            valid.append(
                (
                    abs(y),
                    wx,
                    wy,
                )
            )

    if not valid:

        raise RuntimeError(
            "Could not determine crossing "
            "conflict point."
        )

    valid.sort(
        key=lambda x: x[0]
    )

    return (
        float(valid[0][1]),
        float(valid[0][2]),
    )


def initial_forward(rows):

    yaw_deg = as_float(
        rows[0],
        "ego_yaw",
    )

    yaw = math.radians(
        yaw_deg
    )

    return np.array(
        [
            math.cos(yaw),
            math.sin(yaw),
        ],
        dtype=float,
    )


def infer_ego_half_length(
    rows,
    actor_length_m,
):

    """
    Existing runner computes:

        bumper_gap =
            center_distance
            - ego_half_length
            - actor_half_length

    Therefore we can recover ego bbox half-length.

    We only use this for ego front-bumper position,
    NOT as a crossing gap metric.
    """

    estimates = []

    actor_half = (
        actor_length_m / 2.0
    )

    for row in rows:

        center = as_float(
            row,
            "center_distance_m",
        )

        gap = as_float(
            row,
            "bumper_gap_m",
        )

        if (
            np.isfinite(center)
            and np.isfinite(gap)
        ):

            ego_half = (
                center
                - gap
                - actor_half
            )

            if (
                ego_half > 0.5
                and
                ego_half < 5.0
            ):

                estimates.append(
                    ego_half
                )

    if not estimates:

        raise RuntimeError(
            "Could not infer ego half length."
        )

    return float(
        np.median(
            estimates
        )
    )


def conflict_margin_series(
    rows,
    conflict_point,
    ego_half_length,
):

    """
    Positive:
        ego front bumper remains before conflict point.

    Zero:
        ego front reaches conflict point.

    Negative:
        ego front has passed conflict point.
    """

    forward = initial_forward(
        rows
    )

    conflict = np.array(
        conflict_point,
        dtype=float,
    )

    margins = []

    for row in rows:

        ego = np.array(
            [
                as_float(
                    row,
                    "ego_x",
                ),
                as_float(
                    row,
                    "ego_y",
                ),
            ],
            dtype=float,
        )

        center_distance = float(
            np.dot(
                conflict - ego,
                forward,
            )
        )

        front_margin = (
            center_distance
            - ego_half_length
        )

        margins.append(
            front_margin
        )

    return np.asarray(
        margins,
        dtype=float,
    )


# ============================================================
# Crossing events
# ============================================================

def event_start_idx(rows):

    for i, row in enumerate(
        rows
    ):

        if (
            as_float(
                row,
                "t_s",
            )
            >= EVENT_START_S
        ):

            return i

    return 0


def actor_conflict_enter_idx(
    rows,
    actor_half_length,
):

    """
    Actor moves from +y toward -y.

    Entry:
        leading end of crossing vehicle
        reaches ego road centerline.

    actor_center_y <= +half_length
    """

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
            y <= actor_half_length
        ):

            return i

    return None


def actor_center_cross_idx(rows):

    best = None
    best_abs = float("inf")

    for i, row in enumerate(
        rows
    ):

        y = as_float(
            row,
            "actor_y_sg_m",
        )

        if not np.isfinite(y):
            continue

        if abs(y) < best_abs:

            best_abs = abs(y)
            best = i

    return best


def actor_conflict_clear_idx(
    rows,
    actor_half_length,
):

    """
    Clear:
        trailing end of crossing vehicle
        has passed ego road centerline.

    actor_center_y <= -half_length
    """

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
            y <= -actor_half_length
        ):

            return i

    return None


# ============================================================
# Behavioral events
# ============================================================

def sustained_brake_idx(
    rows,
    start_idx,
):

    return first_sustained_condition(

        rows,

        start_idx,

        lambda row:
            as_float(
                row,
                "tcp_brake",
            )
            >= BRAKE_THRESHOLD,

        BRAKE_CONFIRM_FRAMES,
    )


def full_stop_idx(
    rows,
    start_idx,
):

    return first_sustained_condition(

        rows,

        start_idx,

        lambda row:
            as_float(
                row,
                "ego_speed_mps",
            )
            <= STOP_SPEED_MPS,

        STOP_CONFIRM_FRAMES,
    )


def restart_idx(
    rows,
    start_idx,
):

    return first_sustained_condition(

        rows,

        start_idx,

        lambda row:
            as_float(
                row,
                "ego_speed_mps",
            )
            >= RESTART_SPEED_MPS,

        RESTART_CONFIRM_FRAMES,
    )


def recovery_idx(
    rows,
    start_idx,
    threshold,
):

    return first_sustained_condition(

        rows,

        start_idx,

        lambda row:
            as_float(
                row,
                "ego_speed_mps",
            )
            >= threshold,

        3,
    )


# ============================================================
# Per-condition analysis
# ============================================================

def analyze_condition(
    rows,
    dimensions,
):

    actor_length = (
        dimensions[
            "length_m"
        ]
    )

    actor_half = (
        actor_length / 2.0
    )

    conflict_point = (
        find_conflict_point(
            rows
        )
    )

    ego_half_length = (
        infer_ego_half_length(
            rows,
            actor_length,
        )
    )

    margins = (
        conflict_margin_series(
            rows,
            conflict_point,
            ego_half_length,
        )
    )

    event_idx = (
        event_start_idx(
            rows
        )
    )

    enter_idx = (
        actor_conflict_enter_idx(
            rows,
            actor_half,
        )
    )

    center_idx = (
        actor_center_cross_idx(
            rows
        )
    )

    clear_idx = (
        actor_conflict_clear_idx(
            rows,
            actor_half,
        )
    )

    brake_idx = (
        sustained_brake_idx(
            rows,
            event_idx,
        )
    )

    stop_idx = (
        full_stop_idx(
            rows,
            event_idx,
        )
    )

    restart_search_idx = (
        stop_idx
        if stop_idx is not None
        else event_idx
    )

    restart_i = (
        restart_idx(
            rows,
            restart_search_idx,
        )
    )

    recovery2_idx = (
        recovery_idx(
            rows,
            restart_i,
            RECOVERY_SPEED_2_MPS,
        )
        if restart_i is not None
        else None
    )

    recovery4_idx = (
        recovery_idx(
            rows,
            restart_i,
            RECOVERY_SPEED_4_MPS,
        )
        if restart_i is not None
        else None
    )

    occupied_indices = []

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
            -actor_half
            <= y
            <= actor_half
        ):

            occupied_indices.append(i)

    if occupied_indices:

        occupied_margins = (
            margins[
                occupied_indices
            ]
        )

        min_margin_occupied = (
            float(
                np.min(
                    occupied_margins
                )
            )
        )

        conflict_violation = bool(
            np.any(
                occupied_margins
                <= 0.0
            )
        )

    else:

        min_margin_occupied = (
            float("nan")
        )

        conflict_violation = False

    stop_margin = (
        float(
            margins[
                stop_idx
            ]
        )
        if stop_idx is not None
        else float("nan")
    )

    clear_time = time_at(
        rows,
        clear_idx,
    )

    restart_time = time_at(
        rows,
        restart_i,
    )

    stop_time = time_at(
        rows,
        stop_idx,
    )

    wait_duration = (
        restart_time
        - stop_time
        if (
            np.isfinite(
                restart_time
            )
            and
            np.isfinite(
                stop_time
            )
        )
        else float("nan")
    )

    restart_delay_after_clear = (
        restart_time
        - clear_time
        if (
            np.isfinite(
                restart_time
            )
            and
            np.isfinite(
                clear_time
            )
        )
        else float("nan")
    )

    return {

        "rows":
            rows,

        "margins":
            margins,

        "ego_half_length_m":
            ego_half_length,

        "conflict_x_world":
            conflict_point[0],

        "conflict_y_world":
            conflict_point[1],

        "actor_enter_time_s":
            time_at(
                rows,
                enter_idx,
            ),

        "actor_center_time_s":
            time_at(
                rows,
                center_idx,
            ),

        "actor_clear_time_s":
            clear_time,

        "brake_time_s":
            time_at(
                rows,
                brake_idx,
            ),

        "stop_time_s":
            stop_time,

        "stop_margin_m":
            stop_margin,

        "minimum_margin_while_occupied_m":
            min_margin_occupied,

        "conflict_violation":
            conflict_violation,

        "restart_time_s":
            restart_time,

        "wait_duration_s":
            wait_duration,

        "restart_delay_after_clear_s":
            restart_delay_after_clear,

        "recovery_2mps_time_s":
            time_at(
                rows,
                recovery2_idx,
            ),

        "recovery_4mps_time_s":
            time_at(
                rows,
                recovery4_idx,
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    dimensions = (
        load_actor_dimensions()
    )

    run_dirs = sorted(
        p
        for p in ROOT.glob(
            "run_*"
        )
        if p.is_dir()
    )

    if not run_dirs:

        raise RuntimeError(
            f"No runs found in {ROOT}"
        )

    results = []

    for run_dir in run_dirs:

        run_number = int(
            run_dir.name.split(
                "_"
            )[-1]
        )

        carla = analyze_condition(
            load_csv(
                find_csv(
                    run_dir,
                    "carla",
                )
            ),
            dimensions,
        )

        he = analyze_condition(
            load_csv(
                find_csv(
                    run_dir,
                    "he",
                )
            ),
            dimensions,
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
    print("=" * 118)
    print(
        "TCP CARLA vs HE CROSSING-VEHICLE "
        "REPEATED-TRIAL ANALYSIS"
    )
    print("=" * 118)

    print()
    print("CROSSING GEOMETRY")
    print("-" * 118)

    print(
        f"actor length: "
        f"{dimensions['length_m']:.3f} m"
    )

    example = (
        results[0][
            "carla"
        ]
    )

    print(
        f"actor conflict entry : "
        f"{example['actor_enter_time_s']:.2f} s"
    )

    print(
        f"actor center crossing: "
        f"{example['actor_center_time_s']:.2f} s"
    )

    print(
        f"actor conflict clear : "
        f"{example['actor_clear_time_s']:.2f} s"
    )

    print()
    print("PER-RUN OUTCOMES")
    print("-" * 118)

    print(
        "run | "
        "C stop  H stop  dT | "
        "C margin H margin delta | "
        "C restart H restart dT | "
        "C clear-delay H clear-delay | violation"
    )

    print("-" * 118)

    for r in results:

        c = r[
            "carla"
        ]

        h = r[
            "he"
        ]

        print(
            f"{r['run']:3d} | "
            f"{c['stop_time_s']:6.2f} "
            f"{h['stop_time_s']:6.2f} "
            f"{h['stop_time_s'] - c['stop_time_s']:+5.2f} | "
            f"{c['stop_margin_m']:8.3f} "
            f"{h['stop_margin_m']:8.3f} "
            f"{h['stop_margin_m'] - c['stop_margin_m']:+7.3f} | "
            f"{c['restart_time_s']:9.2f} "
            f"{h['restart_time_s']:9.2f} "
            f"{h['restart_time_s'] - c['restart_time_s']:+5.2f} | "
            f"{c['restart_delay_after_clear_s']:+8.2f} "
            f"{h['restart_delay_after_clear_s']:+8.2f} | "
            f"C={int(c['conflict_violation'])} "
            f"H={int(h['conflict_violation'])}"
        )

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
        print("-" * 118)

        print_stat(
            "sustained brake onset",
            [
                r[condition][
                    "brake_time_s"
                ]
                for r in results
            ],
            "s",
        )

        print_stat(
            "full stop time",
            [
                r[condition][
                    "stop_time_s"
                ]
                for r in results
            ],
            "s",
        )

        print_stat(
            "front margin at full stop",
            [
                r[condition][
                    "stop_margin_m"
                ]
                for r in results
            ],
            "m",
        )

        print_stat(
            "minimum margin while actor occupies path",
            [
                r[condition][
                    "minimum_margin_while_occupied_m"
                ]
                for r in results
            ],
            "m",
        )

        print_stat(
            "restart time",
            [
                r[condition][
                    "restart_time_s"
                ]
                for r in results
            ],
            "s",
        )

        print_stat(
            "wait duration",
            [
                r[condition][
                    "wait_duration_s"
                ]
                for r in results
            ],
            "s",
        )

        print_stat(
            "restart delay after actor clears",
            [
                r[condition][
                    "restart_delay_after_clear_s"
                ]
                for r in results
            ],
            "s",
        )

        print_stat(
            "time to recover >= 2 m/s",
            [
                r[condition][
                    "recovery_2mps_time_s"
                ]
                for r in results
            ],
            "s",
        )

        print_stat(
            "time to recover >= 4 m/s",
            [
                r[condition][
                    "recovery_4mps_time_s"
                ]
                for r in results
            ],
            "s",
        )

    print()
    print("PAIRED HE - CARLA DIFFERENCES")
    print("-" * 118)

    print_stat(
        "delta sustained brake onset",
        [
            r["he"]["brake_time_s"]
            -
            r["carla"]["brake_time_s"]
            for r in results
        ],
        "s",
    )

    print_stat(
        "delta full-stop time",
        [
            r["he"]["stop_time_s"]
            -
            r["carla"]["stop_time_s"]
            for r in results
        ],
        "s",
    )

    print_stat(
        "delta front stop margin",
        [
            r["he"]["stop_margin_m"]
            -
            r["carla"]["stop_margin_m"]
            for r in results
        ],
        "m",
    )

    print_stat(
        "delta minimum occupied-path margin",
        [
            r["he"][
                "minimum_margin_while_occupied_m"
            ]
            -
            r["carla"][
                "minimum_margin_while_occupied_m"
            ]
            for r in results
        ],
        "m",
    )

    print_stat(
        "delta restart time",
        [
            r["he"]["restart_time_s"]
            -
            r["carla"]["restart_time_s"]
            for r in results
        ],
        "s",
    )

    print_stat(
        "delta wait duration",
        [
            r["he"]["wait_duration_s"]
            -
            r["carla"]["wait_duration_s"]
            for r in results
        ],
        "s",
    )

    print_stat(
        "delta restart delay after clear",
        [
            r["he"][
                "restart_delay_after_clear_s"
            ]
            -
            r["carla"][
                "restart_delay_after_clear_s"
            ]
            for r in results
        ],
        "s",
    )

    print_stat(
        "delta 2 m/s recovery time",
        [
            r["he"][
                "recovery_2mps_time_s"
            ]
            -
            r["carla"][
                "recovery_2mps_time_s"
            ]
            for r in results
        ],
        "s",
    )

    print_stat(
        "delta 4 m/s recovery time",
        [
            r["he"][
                "recovery_4mps_time_s"
            ]
            -
            r["carla"][
                "recovery_4mps_time_s"
            ]
            for r in results
        ],
        "s",
    )

    carla_violations = sum(
        int(
            r["carla"][
                "conflict_violation"
            ]
        )
        for r in results
    )

    he_violations = sum(
        int(
            r["he"][
                "conflict_violation"
            ]
        )
        for r in results
    )

    print()
    print("SAFETY / CONFLICT-ZONE OUTCOME")
    print("-" * 118)

    print(
        f"CARLA conflict-zone violations: "
        f"{carla_violations}/{len(results)}"
    )

    print(
        f"HE conflict-zone violations   : "
        f"{he_violations}/{len(results)}"
    )

    # --------------------------------------------------------
    # Save compact summary
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {

        "scenario_id":
            SCENARIO_ID,

        "num_pairs":
            len(results),

        "actor_dimensions_m":
            dimensions,

        "event_definition": {
            "event_start_s":
                EVENT_START_S,

            "stop_speed_mps":
                STOP_SPEED_MPS,

            "restart_speed_mps":
                RESTART_SPEED_MPS,

            "brake_threshold":
                BRAKE_THRESHOLD,

            "conflict_occupancy":
                (
                    "actor body intersects "
                    "ego-path centerline"
                ),
        },

        "carla_conflict_violations":
            carla_violations,

        "he_conflict_violations":
            he_violations,

        "runs":
            [],
    }

    for r in results:

        row = {
            "run":
                r["run"]
        }

        for condition in (
            "carla",
            "he",
        ):

            src = r[
                condition
            ]

            row[
                condition
            ] = {
                k: v
                for k, v
                in src.items()
                if k
                not in (
                    "rows",
                    "margins",
                )
            }

        summary[
            "runs"
        ].append(
            row
        )

    summary_path = (
        OUTPUT_DIR
        / "tcp_crossing_001_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("SAVED")
    print("-" * 118)

    print(
        "summary:",
        summary_path,
    )

    print()
    print("=" * 118)


if __name__ == "__main__":
    main()