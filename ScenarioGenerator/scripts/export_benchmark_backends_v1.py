"""
export_benchmark_backends_v1.py

Milestone 4C1.

Consumes the output of M4B.

For each HE_SUPPORTED benchmark case:

    ResolvedScenarioV2
          │
          ├──> CARLA Adapter V2
          │       └── carla_plan_v2.json
          │
          └──> HE Adapter V2
                  └── he_scenario_v2.json

No CARLA simulation is executed.
No HE rendering is executed.
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from pathlib import PureWindowsPath


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================
# Existing validated architecture
# ============================================================

from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedScenarioV2,
)

from scenario_generator.backends.carla_backend.carla_adapter_v2 import (
    resolved_v2_to_carla_plan,
)

from scenario_generator.backends.he_backend.current_he_adapter_v2 import (
    resolved_to_current_he_json,
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


# ============================================================
# HE runtime paths
# ============================================================

def make_he_runtime_paths(
    case_id: str,
):
    """
    Paths are intentionally relative to HE_v_0.1 because the
    compositor is normally executed from that directory.
    """

    pair_root = PureWindowsPath(
        "recordings",
        "he_pairs",
        case_id,
    )

    he_output_root = PureWindowsPath(
        "he_outputs",
        f"{case_id}_he",
    )

    return {

        "background_video_path":
            str(
                pair_root
                / "background.mp4"
            ),

        "ego_pose_path":
            str(
                pair_root
                / "background_ego_pose.jsonl"
            ),

        "output_dir":
            str(
                he_output_root
            ),

        "output_video_name":
            "he_output.mp4",
    }


# ============================================================
# Export one case
# ============================================================

def export_case(
    *,
    case_id: str,
    resolved_path: Path,
    output_dir: Path,
    he_template: dict,
):
    # --------------------------------------------------------
    # Load exact physical truth from M4B
    # --------------------------------------------------------

    resolved = (
        ResolvedScenarioV2
        .model_validate(
            load_json(
                resolved_path
            )
        )
    )

    if (
        resolved.scenario_id
        != case_id
    ):
        raise ValueError(
            "Resolved scenario ID mismatch: "
            f"expected {case_id!r}, "
            f"found {resolved.scenario_id!r}"
        )

    # ========================================================
    # CARLA export
    # ========================================================

    carla_plan = (
        resolved_v2_to_carla_plan(
            resolved
        )
    )

    carla_path = (
        output_dir
        / "carla_plan_v2.json"
    )

    save_json(
        carla_path,
        carla_plan,
    )

    # ========================================================
    # HE export
    # ========================================================

    runtime_paths = (
        make_he_runtime_paths(
            case_id
        )
    )

    he_scenario = (
        resolved_to_current_he_json(

            resolved=resolved,

            template_json=
                he_template,

            output_dir=
                runtime_paths[
                    "output_dir"
                ],

            output_video_name=
                runtime_paths[
                    "output_video_name"
                ],

            use_keyframes=True,

            background_video_path=
                runtime_paths[
                    "background_video_path"
                ],

            ego_pose_path=
                runtime_paths[
                    "ego_pose_path"
                ],
        )
    )

    he_path = (
        output_dir
        / "he_scenario_v2.json"
    )

    save_json(
        he_path,
        he_scenario,
    )

    # ========================================================
    # Export manifest
    # ========================================================

    actor_frame_counts = {}

    for actor in resolved.actors:

        actor_frame_counts[
            actor.actor_id
        ] = sum(
            1
            for frame
            in resolved.actor_frames
            if (
                frame.actor_id
                ==
                actor.actor_id
            )
        )

    manifest = {

        "case_id":
            case_id,

        "scenario_id":
            resolved.scenario_id,

        "duration_s":
            float(
                resolved.duration_s
            ),

        "fps":
            int(
                resolved.fps
            ),

        "ego_frames":
            len(
                resolved.ego_frames
            ),

        "actor_frame_counts":
            actor_frame_counts,

        "carla_plan":
            str(
                carla_path
            ),

        "he_scenario":
            str(
                he_path
            ),

        "he_runtime_paths":
            runtime_paths,

        "execution_status":
            "NOT_EXECUTED",
    }

    save_json(
        output_dir
        / "backend_export_manifest.json",

        manifest,
    )

    return manifest


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Export M4B HE-supported benchmark cases "
            "to CARLA Plan V2 and HE Scenario V2."
        )
    )

    parser.add_argument(
        "--m4b-summary",
        required=True,
    )

    parser.add_argument(
        "--he-template",
        default=(
            "examples/he_templates/"
            "oncoming_vehicle_001.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--case-id",
        default=None,
        help=(
            "Optional single benchmark case to export."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Inputs
    # ========================================================

    m4b_summary_path = Path(
        args.m4b_summary
    )

    m4b_summary = load_json(
        m4b_summary_path
    )

    m4b_root = (
        m4b_summary_path.parent
    )

    benchmark_id = str(
        m4b_summary[
            "benchmark_id"
        ]
    )

    template_path = Path(
        args.he_template
    )

    if not template_path.is_absolute():

        template_path = (
            PROJECT_ROOT
            / template_path
        )

    if not template_path.exists():

        raise FileNotFoundError(
            "HE template not found: "
            f"{template_path}"
        )

    he_template = load_json(
        template_path
    )

    # ========================================================
    # Output root
    # ========================================================

    if args.output_dir:

        output_root = Path(
            args.output_dir
        )

    else:

        output_root = (
            m4b_root.parent
            / "m4c"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Select cases
    # ========================================================

    supported_cases = [

        case

        for case
        in m4b_summary.get(
            "cases",
            [],
        )

        if (
            case.get(
                "status"
            )
            ==
            "HE_SUPPORTED"
        )
    ]

    if args.case_id is not None:

        supported_cases = [

            case

            for case
            in supported_cases

            if (
                case.get(
                    "case_id"
                )
                ==
                args.case_id
            )
        ]

        if not supported_cases:

            raise ValueError(
                "Requested case was not found "
                "among HE_SUPPORTED cases: "
                f"{args.case_id}"
            )

    # ========================================================
    # Export
    # ========================================================

    print()
    print(
        "=" * 96
    )

    print(
        "M4C1 BENCHMARK BACKEND EXPORT"
    )

    print(
        "=" * 96
    )

    print(
        "benchmark:",
        benchmark_id,
    )

    print(
        "HE template:",
        template_path,
    )

    print(
        "supported cases selected:",
        len(
            supported_cases
        ),
    )

    print(
        "output:",
        output_root,
    )

    print()

    exported = []
    errors = []

    for case in supported_cases:

        case_id = str(
            case[
                "case_id"
            ]
        )

        resolved_path = (
            m4b_root
            / case_id
            / "resolved_scenario_v2.json"
        )

        case_output_dir = (
            output_root
            / case_id
        )

        try:

            manifest = export_case(

                case_id=
                    case_id,

                resolved_path=
                    resolved_path,

                output_dir=
                    case_output_dir,

                he_template=
                    he_template,
            )

            exported.append(
                manifest
            )

            actor_text = ", ".join(
                f"{actor_id}={count}"
                for (
                    actor_id,
                    count,
                )
                in manifest[
                    "actor_frame_counts"
                ].items()
            )

            print(
                f"[OK] {case_id}"
            )

            print(
                f"     ego frames: "
                f"{manifest['ego_frames']}"
            )

            print(
                f"     actor frames: "
                f"{actor_text}"
            )

            print(
                f"     CARLA: "
                f"{case_output_dir / 'carla_plan_v2.json'}"
            )

            print(
                f"     HE:    "
                f"{case_output_dir / 'he_scenario_v2.json'}"
            )

        except Exception as exc:

            error_text = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            errors.append(
                {
                    "case_id":
                        case_id,

                    "error":
                        error_text,
                }
            )

            print(
                f"[ERROR] "
                f"{case_id}: "
                f"{error_text}"
            )

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "benchmark_id":
            benchmark_id,

        "selected_supported_cases":
            len(
                supported_cases
            ),

        "exported_cases":
            len(
                exported
            ),

        "errors":
            len(
                errors
            ),

        "cases":
            exported,

        "error_cases":
            errors,
    }

    summary_path = (
        output_root
        / "m4c_export_summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    print()
    print(
        "-" * 96
    )

    print(
        "Selected: ",
        len(
            supported_cases
        ),
    )

    print(
        "Exported: ",
        len(
            exported
        ),
    )

    print(
        "Errors:   ",
        len(
            errors
        ),
    )

    print(
        "Summary:  ",
        summary_path,
    )

    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()