"""
run_benchmark_execution_v1.py

Milestone 4D.

Execute one or more already-validated M4C benchmark cases.

Pipeline:

    M4C invariant PASS
        ↓
    runtime-policy consistency
        ↓
    CARLA real adversary
        ↓
    CARLA background
        ↓
    ego synchronization check
        ↓
    HE compositor
        ↓
    CARLA-vs-HE bbox comparison
        ↓
    execution report

This script DOES NOT:
    - generate scenarios
    - resolve trajectories
    - modify backend plans
    - alter CARLA/HE adapters

It executes already-frozen M4C artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from datetime import datetime
from pathlib import Path


# ============================================================
# Repository paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

REPO_ROOT = (
    PROJECT_ROOT.parent
)

HE_ROOT = (
    REPO_ROOT
    / "HE_v_0.1"
)


# ============================================================
# IO
# ============================================================

def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            allow_nan=False,
        )

        f.write(
            "\n"
        )


def load_jsonl(
    path: Path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(
                    line
                )
            )

    return rows


# ============================================================
# Logged subprocess execution
# ============================================================

def run_logged(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
):
    """
    Run command while simultaneously printing output and
    saving it to a log file.
    """

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "$",
        subprocess.list2cmdline(
            command
        ),
    )

    print(
        "cwd:",
        cwd,
    )

    print()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        process = subprocess.Popen(
            command,

            cwd=str(
                cwd
            ),

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            bufsize=1,
        )

        assert (
            process.stdout
            is not None
        )

        for line in process.stdout:

            print(
                line,
                end="",
            )

            log_file.write(
                line
            )

            log_file.flush()

        return_code = (
            process.wait()
        )

    if return_code != 0:

        raise RuntimeError(
            "Command failed with "
            f"exit code {return_code}: "
            f"{subprocess.list2cmdline(command)}"
        )


# ============================================================
# M4C invariant validation
# ============================================================

def load_invariant_case_status(
    invariant_summary_path: Path,
):
    data = load_json(
        invariant_summary_path
    )

    output = {}

    for case in data.get(
        "cases",
        [],
    ):

        output[
            str(
                case[
                    "case_id"
                ]
            )
        ] = str(
            case[
                "status"
            ]
        )

    return (
        data,
        output,
    )


# ============================================================
# Runtime-policy consistency
# ============================================================

def check_visibility_policy(
    *,
    capability_path: Path,
    he_scenario_path: Path,
    tolerance: float = 1e-9,
):
    """
    M4B capability analysis and actual HE runtime configuration
    must agree on near/behind-camera culling.

    Otherwise the preflight ODD classification would not describe
    the actual compositor behavior.
    """

    capability = load_json(
        capability_path
    )

    he_scenario = load_json(
        he_scenario_path
    )

    preflight_value = float(
        capability[
            "visibility_policy"
        ][
            "min_render_depth_m"
        ]
    )

    runtime_value = float(
        he_scenario.get(
            "visibility",
            {},
        ).get(
            "min_render_depth_m",
            2.0,
        )
    )

    difference = abs(
        preflight_value
        -
        runtime_value
    )

    if difference > tolerance:

        raise ValueError(
            "M4B/HE runtime visibility-policy mismatch. "
            f"M4B min_render_depth_m="
            f"{preflight_value}, "
            f"HE scenario min_render_depth_m="
            f"{runtime_value}. "
            "Align these before benchmark execution."
        )

    return {
        "m4b_min_render_depth_m":
            preflight_value,

        "he_runtime_min_render_depth_m":
            runtime_value,

        "difference":
            difference,

        "status":
            "PASS",
    }


# ============================================================
# Ego-pose synchronization
# ============================================================

def frame_index(
    row: dict,
):
    if (
        "recorded_frame_idx"
        in row
    ):

        return int(
            row[
                "recorded_frame_idx"
            ]
        )

    if "frame_idx" in row:

        return int(
            row[
                "frame_idx"
            ]
        )

    raise KeyError(
        "Pose row contains neither "
        "recorded_frame_idx nor frame_idx."
    )


def extract_ego_transform(
    row: dict,
):
    if (
        "ego_transform"
        not in row
    ):

        raise KeyError(
            "Pose row does not contain "
            "'ego_transform'."
        )

    tf = row[
        "ego_transform"
    ]

    return {
        "x":
            float(
                tf["x"]
            ),

        "y":
            float(
                tf["y"]
            ),

        "z":
            float(
                tf["z"]
            ),

        "pitch":
            float(
                tf["pitch"]
            ),

        "yaw":
            float(
                tf["yaw"]
            ),

        "roll":
            float(
                tf["roll"]
            ),
    }


def compare_ego_pose_files(
    *,
    real_path: Path,
    background_path: Path,
    tolerance: float = 1e-6,
):
    real_rows = load_jsonl(
        real_path
    )

    bg_rows = load_jsonl(
        background_path
    )

    real = {
        frame_index(row):
            extract_ego_transform(
                row
            )

        for row
        in real_rows
    }

    background = {
        frame_index(row):
            extract_ego_transform(
                row
            )

        for row
        in bg_rows
    }

    real_frames = set(
        real.keys()
    )

    bg_frames = set(
        background.keys()
    )

    if real_frames != bg_frames:

        raise ValueError(
            "Real/background ego frame sets differ."
        )

    fields = [
        "x",
        "y",
        "z",
        "pitch",
        "yaw",
        "roll",
    ]

    max_abs_error = {
        name: 0.0
        for name
        in fields
    }

    first_mismatch = None

    for frame_idx in sorted(
        real_frames
    ):

        for field in fields:

            error = abs(
                real[
                    frame_idx
                ][
                    field
                ]
                -
                background[
                    frame_idx
                ][
                    field
                ]
            )

            max_abs_error[
                field
            ] = max(
                max_abs_error[
                    field
                ],
                error,
            )

            if (
                first_mismatch
                is None
                and
                error > tolerance
            ):

                first_mismatch = {
                    "frame_idx":
                        frame_idx,

                    "field":
                        field,

                    "real":
                        real[
                            frame_idx
                        ][
                            field
                        ],

                    "background":
                        background[
                            frame_idx
                        ][
                            field
                        ],

                    "abs_error":
                        error,
                }

    if first_mismatch is not None:

        raise ValueError(
            "Real/background ego synchronization "
            "failed: "
            f"{first_mismatch}"
        )

    return {
        "frames":
            len(
                real_frames
            ),

        "tolerance":
            tolerance,

        "max_abs_error":
            max_abs_error,

        "status":
            "PASS",
    }


# ============================================================
# Required file checks
# ============================================================

def require_file(
    path: Path,
    description: str,
):
    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found: "
            f"{path}"
        )


# ============================================================
# Execute one case
# ============================================================

def execute_case(
    *,
    case_id: str,

    m4c_root: Path,
    m4b_root: Path,

    invariant_status: dict,

    output_root: Path,

    carla_python: str,
    he_python: str,

    town: str,
    spawn_index: int,

    min_real_depth: float,
):
    started_at = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )

    case_output_dir = (
        output_root
        / case_id
    )

    logs_dir = (
        case_output_dir
        / "logs"
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # M4C artifact locations
    # ========================================================

    m4c_case_dir = (
        m4c_root
        / case_id
    )

    carla_plan_path = (
        m4c_case_dir
        / "carla_plan_v2.json"
    )

    he_scenario_path = (
        m4c_case_dir
        / "he_scenario_v2.json"
    )

    capability_path = (
        m4b_root
        / case_id
        / "he_capability.json"
    )

    require_file(
        carla_plan_path,
        "CARLA Plan V2",
    )

    require_file(
        he_scenario_path,
        "HE Scenario V2",
    )

    require_file(
        capability_path,
        "M4B capability report",
    )

    # ========================================================
    # Hard invariant gate
    # ========================================================

    status = invariant_status.get(
        case_id
    )

    if status != "PASS":

        raise RuntimeError(
            f"{case_id}: M4C invariant "
            f"status is {status!r}; "
            "execution is blocked."
        )

    # ========================================================
    # Runtime-policy gate
    # ========================================================

    visibility_check = (
        check_visibility_policy(
            capability_path=
                capability_path,

            he_scenario_path=
                he_scenario_path,
        )
    )

    print(
        "[PASS] visibility policy:",
        visibility_check,
    )

    # ========================================================
    # HE runtime scripts
    # ========================================================

    carla_script = (
        HE_ROOT
        / "record_carla_from_plan_v2.py"
    )

    compositor_script = (
        HE_ROOT
        / "run_he_temporal_compositor_v2.py"
    )

    comparator_script = (
        HE_ROOT
        / "compare_real_vs_he_bbox_v2.py"
    )

    require_file(
        carla_script,
        "CARLA V2 executor",
    )

    require_file(
        compositor_script,
        "HE V2 compositor",
    )

    require_file(
        comparator_script,
        "V2 bbox comparator",
    )

    # ========================================================
    # Runtime output paths
    # ========================================================

    pair_dir = (
        HE_ROOT
        / "recordings"
        / "he_pairs"
        / case_id
    )

    real_gt_path = (
        pair_dir
        / "real_ground_truth_v2.jsonl"
    )

    real_ego_path = (
        pair_dir
        / "real_ego_pose.jsonl"
    )

    background_ego_path = (
        pair_dir
        / "background_ego_pose.jsonl"
    )

    background_video_path = (
        pair_dir
        / "background.mp4"
    )

    he_output_dir = (
        HE_ROOT
        / "he_outputs"
        / f"{case_id}_he"
    )

    he_metadata_path = (
        he_output_dir
        / "metadata.json"
    )

    # ========================================================
    # 1. CARLA real
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        f"{case_id}: CARLA REAL"
    )

    print(
        "=" * 100
    )

    run_logged(

        command=[
            carla_python,
            str(
                carla_script
            ),

            "--plan",
            str(
                carla_plan_path.resolve()
            ),

            "--mode",
            "real_adversary",

            "--pair-name",
            case_id,

            "--output-root",
            "recordings/he_pairs",

            "--town",
            town,

            "--spawn-index",
            str(
                spawn_index
            ),
        ],

        cwd=
            HE_ROOT,

        log_path=
            logs_dir
            / "01_carla_real.txt",
    )

    require_file(
        real_gt_path,
        "real_ground_truth_v2.jsonl",
    )

    require_file(
        real_ego_path,
        "real_ego_pose.jsonl",
    )

    # ========================================================
    # 2. CARLA background
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        f"{case_id}: CARLA BACKGROUND"
    )

    print(
        "=" * 100
    )

    run_logged(

        command=[
            carla_python,
            str(
                carla_script
            ),

            "--plan",
            str(
                carla_plan_path.resolve()
            ),

            "--mode",
            "background",

            "--pair-name",
            case_id,

            "--output-root",
            "recordings/he_pairs",

            "--town",
            town,

            "--spawn-index",
            str(
                spawn_index
            ),
        ],

        cwd=
            HE_ROOT,

        log_path=
            logs_dir
            / "02_carla_background.txt",
    )

    require_file(
        background_ego_path,
        "background_ego_pose.jsonl",
    )

    require_file(
        background_video_path,
        "background.mp4",
    )

    # ========================================================
    # 3. Ego synchronization
    # ========================================================

    sync_report = (
        compare_ego_pose_files(
            real_path=
                real_ego_path,

            background_path=
                background_ego_path,
        )
    )

    save_json(
        case_output_dir
        / "ego_sync_report.json",

        sync_report,
    )

    print()
    print(
        "[PASS] real/background ego sync"
    )

    print(
        "       frames:",
        sync_report[
            "frames"
        ],
    )

    print(
        "       max error:",
        sync_report[
            "max_abs_error"
        ],
    )

    # ========================================================
    # 4. HE compositor
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        f"{case_id}: HE RENDER"
    )

    print(
        "=" * 100
    )

    run_logged(

        command=[
            he_python,
            str(
                compositor_script
            ),

            "--scenario",
            str(
                he_scenario_path.resolve()
            ),

            "--overwrite",
        ],

        cwd=
            HE_ROOT,

        log_path=
            logs_dir
            / "03_he_render.txt",
    )

    require_file(
        he_metadata_path,
        "HE metadata.json",
    )

    # ========================================================
    # 5. Geometric comparison
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        f"{case_id}: GEOMETRIC COMPARISON"
    )

    print(
        "=" * 100
    )

    run_logged(

        command=[
            he_python,
            str(
                comparator_script
            ),

            "--real-ground-truth",
            str(
                real_gt_path.resolve()
            ),

            "--he-metadata",
            str(
                he_metadata_path.resolve()
            ),

            "--min-real-depth",
            str(
                min_real_depth
            ),
        ],

        cwd=
            HE_ROOT,

        log_path=
            logs_dir
            / (
                "04_compare_depth_"
                f"{min_real_depth:g}.txt"
            ),
    )

    # ========================================================
    # Execution report
    # ========================================================

    finished_at = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )

    report = {

        "case_id":
            case_id,

        "status":
            "PASS",

        "started_at":
            started_at,

        "finished_at":
            finished_at,

        "m4c_invariant":
            "PASS",

        "visibility_policy":
            visibility_check,

        "ego_sync":
            sync_report,

        "comparison_min_real_depth_m":
            min_real_depth,

        "artifacts": {

            "carla_plan":
                str(
                    carla_plan_path
                ),

            "he_scenario":
                str(
                    he_scenario_path
                ),

            "real_ground_truth":
                str(
                    real_gt_path
                ),

            "real_ego_pose":
                str(
                    real_ego_path
                ),

            "background_ego_pose":
                str(
                    background_ego_path
                ),

            "background_video":
                str(
                    background_video_path
                ),

            "he_metadata":
                str(
                    he_metadata_path
                ),
        },

        "logs": {

            "carla_real":
                str(
                    logs_dir
                    / "01_carla_real.txt"
                ),

            "carla_background":
                str(
                    logs_dir
                    / "02_carla_background.txt"
                ),

            "he_render":
                str(
                    logs_dir
                    / "03_he_render.txt"
                ),

            "comparison":
                str(
                    logs_dir
                    / (
                        "04_compare_depth_"
                        f"{min_real_depth:g}.txt"
                    )
                ),
        },
    }

    save_json(
        case_output_dir
        / "execution_report.json",

        report,
    )

    return report


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Execute M4D controlled benchmark cases "
            "from validated M4C outputs."
        )
    )

    parser.add_argument(
        "--m4c-summary",
        required=True,
    )

    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help=(
            "Case to execute. May be supplied multiple times. "
            "If omitted, all exported cases are selected."
        ),
    )

    parser.add_argument(
        "--carla-python",
        default=sys.executable,
    )

    parser.add_argument(
        "--he-python",
        default=sys.executable,
    )

    parser.add_argument(
        "--town",
        default="Town10HD_Opt",
    )

    parser.add_argument(
        "--spawn-index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-real-depth",
        type=float,
        default=15.0,
    )

    args = parser.parse_args()

    # ========================================================
    # M4C structure
    # ========================================================

    m4c_summary_path = Path(
        args.m4c_summary
    )

    m4c_summary = load_json(
        m4c_summary_path
    )

    m4c_root = (
        m4c_summary_path.parent
    )

    benchmark_root = (
        m4c_root.parent
    )

    m4b_root = (
        benchmark_root
        / "m4b"
    )

    invariant_summary_path = (
        m4c_root
        / "m4c_invariant_summary.json"
    )

    require_file(
        invariant_summary_path,
        "M4C invariant summary",
    )

    (
        invariant_summary,
        invariant_status,
    ) = load_invariant_case_status(
        invariant_summary_path
    )

    if not bool(
        invariant_summary.get(
            "all_passed",
            False,
        )
    ):

        raise RuntimeError(
            "M4C invariant summary is not all_passed. "
            "M4D execution is blocked."
        )

    benchmark_id = str(
        m4c_summary[
            "benchmark_id"
        ]
    )

    exported_cases = [
        str(
            case[
                "case_id"
            ]
        )

        for case
        in m4c_summary.get(
            "cases",
            []
        )
    ]

    # ========================================================
    # Case selection
    # ========================================================

    if args.case_id:

        selected = list(
            args.case_id
        )

        unknown = sorted(
            set(
                selected
            )
            -
            set(
                exported_cases
            )
        )

        if unknown:

            raise ValueError(
                "Requested case(s) are not present "
                "in M4C export: "
                f"{unknown}"
            )

    else:

        selected = (
            exported_cases
        )

    output_root = (
        benchmark_root
        / "m4d"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "M4D CONTROLLED BENCHMARK EXECUTION"
    )

    print(
        "=" * 100
    )

    print(
        "benchmark:",
        benchmark_id,
    )

    print(
        "cases selected:",
        len(
            selected
        ),
    )

    for case_id in selected:

        print(
            "  -",
            case_id,
        )

    print(
        "CARLA python:",
        args.carla_python,
    )

    print(
        "HE python:",
        args.he_python,
    )

    print(
        "town:",
        args.town,
    )

    print(
        "spawn index:",
        args.spawn_index,
    )

    print(
        "comparison min depth:",
        args.min_real_depth,
    )

    print()

    # ========================================================
    # Execute
    # ========================================================

    reports = []

    failed = []

    for case_id in selected:

        print()
        print(
            "#" * 100
        )

        print(
            "EXECUTING:",
            case_id,
        )

        print(
            "#" * 100
        )

        try:

            report = execute_case(

                case_id=
                    case_id,

                m4c_root=
                    m4c_root,

                m4b_root=
                    m4b_root,

                invariant_status=
                    invariant_status,

                output_root=
                    output_root,

                carla_python=
                    args.carla_python,

                he_python=
                    args.he_python,

                town=
                    args.town,

                spawn_index=
                    args.spawn_index,

                min_real_depth=
                    args.min_real_depth,
            )

            reports.append(
                report
            )

            print()
            print(
                "[PASS]",
                case_id,
            )

        except Exception as exc:

            error_text = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failed.append(
                {
                    "case_id":
                        case_id,

                    "error":
                        error_text,
                }
            )

            print()
            print(
                "[FAIL]",
                case_id,
            )

            print(
                "       ",
                error_text,
            )

            # First version intentionally stops.
            # We do not want a broken pilot to continue
            # consuming CARLA time.
            break

    # ========================================================
    # Aggregate manifest
    # ========================================================

    aggregate = {

        "benchmark_id":
            benchmark_id,

        "cases_selected":
            selected,

        "cases_passed":
            len(
                reports
            ),

        "cases_failed":
            len(
                failed
            ),

        "all_passed":
            (
                len(
                    failed
                )
                == 0
                and
                len(
                    reports
                )
                ==
                len(
                    selected
                )
            ),

        "reports":
            reports,

        "failures":
            failed,
    }

    aggregate_path = (
        output_root
        / "m4d_execution_summary.json"
    )

    save_json(
        aggregate_path,
        aggregate,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "M4D SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        "Selected:",
        len(
            selected
        ),
    )

    print(
        "Passed:  ",
        len(
            reports
        ),
    )

    print(
        "Failed:  ",
        len(
            failed
        ),
    )

    print(
        "Summary: ",
        aggregate_path,
    )

    print(
        "=" * 100
    )

    if failed:

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()