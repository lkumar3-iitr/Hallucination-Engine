from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
HE_ROOT = REPO_ROOT / "HE_v_0.1"

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ScenarioGenerator imports
# ============================================================
from scenario_generator.planner.scenario_request import (
    ScenarioRequest,
    scenario_request_to_spec,
)
from scenario_generator.backends.carla_backend.carla_adapter_placeholder import (
    resolved_to_carla_plan,
)

from scenario_generator.backends.he_backend.current_he_adapter import (
    load_he_template,
    resolved_to_current_he_json,
    save_he_json,
)

from scenario_generator.io.json_io import (
    load_model,
    save_model,
)

from scenario_generator.planner.manual_planner import (
    make_cut_in_vehicle_demo,
    make_oncoming_vehicle_demo,
    make_oncoming_vehicle_with_360_rig_demo,
    make_static_vehicle_demo,
    make_cut_in_from_left_demo,
    make_cut_in_from_right_demo,
    make_fast_oncoming_demo,
    make_slow_static_near_demo,
)

from scenario_generator.schema import (
    ScenarioSpec,
)

from scenario_generator.trajectory.rule_based.rule_based_generator import (
    RuleBasedTrajectoryGenerator,
)

from scenario_generator.validator.scenario_validator import (
    ScenarioValidator,
)


# ============================================================
# Demo registry
# ============================================================

DEMO_BUILDERS = {
    "oncoming": make_oncoming_vehicle_demo,
    "static": make_static_vehicle_demo,
    "cut_in": make_cut_in_vehicle_demo,
    "oncoming_360": make_oncoming_vehicle_with_360_rig_demo,
    "cut_in_right": make_cut_in_from_right_demo,
    "cut_in_left": make_cut_in_from_left_demo,
    "fast_oncoming": make_fast_oncoming_demo,
    "static_near": make_slow_static_near_demo,
}


# ============================================================
# JSON helper
# ============================================================

def save_json(
    path: Path,
    data: dict,
) -> None:
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
        )

        f.write("\n")


# ============================================================
# Build ScenarioSpec
# ============================================================

def build_scenario_spec(
    args,
) -> tuple[ScenarioSpec, ScenarioRequest | None]:
    """
    Obtain a ScenarioSpec from one of four sources:

        1. legacy/manual demo
        2. ScenarioSpec JSON
        3. ScenarioRequest JSON
        4. direct semantic CLI parameters
    """

    # --------------------------------------------------------
    # Existing full ScenarioSpec JSON
    # --------------------------------------------------------

    if args.input_json is not None:
        scenario = load_model(
            args.input_json,
            ScenarioSpec,
        )

        return scenario, None

    # --------------------------------------------------------
    # Semantic ScenarioRequest JSON
    # --------------------------------------------------------

    if args.request_json is not None:
        request = load_model(
            args.request_json,
            ScenarioRequest,
        )

        scenario = scenario_request_to_spec(
            request
        )

        return scenario, request

    # --------------------------------------------------------
    # Direct semantic parameters
    # --------------------------------------------------------

    if args.scenario_type is not None:

        request = ScenarioRequest(
            scenario_type=(
                args.scenario_type
            ),

            scenario_id=(
                args.output_name
            ),

            side=(
                args.side
            ),

            start_distance_m=(
                args.start_distance
            ),

            lane_y_m=(
                args.lane_y
            ),

            target_lane_y_m=(
                args.target_lane_y
            ),

            lane_width_m=(
                args.lane_width
            ),

            actor_speed_mps=(
                args.actor_speed
            ),

            ego_speed_mps=(
                args.ego_speed
            ),

            duration_s=(
                args.duration
            ),

            fps=(
                args.fps
            ),

            cut_start_s=(
                args.cut_start
            ),

            cut_duration_s=(
                args.cut_duration
            ),
        )

        scenario = scenario_request_to_spec(
            request
        )

        return scenario, request

    # --------------------------------------------------------
    # Existing demo
    # --------------------------------------------------------

    if args.demo is not None:
        builder = DEMO_BUILDERS.get(
            args.demo
        )

        if builder is None:
            raise ValueError(
                f"Unknown demo: {args.demo}"
            )

        return builder(), None

    raise RuntimeError(
        "No scenario source selected."
    )

# ============================================================
# Validation
# ============================================================

def validate_scenario_spec(
    scenario: ScenarioSpec,
    validator: ScenarioValidator,
) -> None:
    result = validator.validate_structured(
        scenario
    )

    for issue in result.issues:
        print(
            f"[{issue.severity.upper()}] "
            f"{issue.message}"
        )

    if not result.ok:
        raise RuntimeError(
            "Structured ScenarioSpec validation failed."
        )


def validate_resolved_scenario(
    resolved,
    validator: ScenarioValidator,
) -> None:
    result = validator.validate_resolved(
        resolved
    )

    for issue in result.issues:
        print(
            f"[{issue.severity.upper()}] "
            f"{issue.message}"
        )

    if not result.ok:
        raise RuntimeError(
            "ResolvedScenario validation failed."
        )


# ============================================================
# Main pipeline
# ============================================================

def build_pipeline(args) -> None:

    # --------------------------------------------------------
    # 1. ScenarioSpec
    # --------------------------------------------------------

    scenario, scenario_request = (
        build_scenario_spec(
            args
        )
    )

    # Give each built experiment its own identity.
    if args.output_name is not None:
        scenario.scenario_id = (
            args.output_name
        )

    experiment_name = (
        args.output_name
        or scenario.scenario_id
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_root = Path(
        args.output_root
    )

    experiment_dir = (
        output_root
        / experiment_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 2. Validate structured scenario
    # --------------------------------------------------------

    validator = ScenarioValidator()

    validate_scenario_spec(
        scenario,
        validator,
    )

    # --------------------------------------------------------
    # Save ScenarioSpec
    # --------------------------------------------------------

    scenario_spec_path = (
        experiment_dir
        / "scenario_spec.json"
    )

    save_model(
        scenario_spec_path,
        scenario,
    )
    # --------------------------------------------------------
    # Save semantic request when this scenario came from the
    # parameter/request layer.
    # --------------------------------------------------------

    scenario_request_path = None

    if scenario_request is not None:
        scenario_request_path = (
            experiment_dir
            / "scenario_request.json"
        )

        save_model(
            scenario_request_path,
            scenario_request,
        )
    # --------------------------------------------------------
    # 3. Resolve physical trajectory ONCE
    # --------------------------------------------------------

    generator = (
        RuleBasedTrajectoryGenerator()
    )

    resolved = generator.resolve(
        scenario
    )

    validate_resolved_scenario(
        resolved,
        validator,
    )

    resolved_path = (
        experiment_dir
        / "resolved_scenario.json"
    )

    save_model(
        resolved_path,
        resolved,
    )

    # --------------------------------------------------------
    # 4. CARLA execution plan
    # --------------------------------------------------------

    carla_plan = (
        resolved_to_carla_plan(
            resolved
        )
    )

    carla_plan_path = (
        experiment_dir
        / "carla_plan.json"
    )

    save_json(
        carla_plan_path,
        carla_plan,
    )

    # --------------------------------------------------------
    # 5. HE execution scenario
    # --------------------------------------------------------

    he_template_path = Path(
        args.he_template
    )

    if not he_template_path.is_absolute():
        he_template_path = (
            PROJECT_ROOT
            / he_template_path
        ).resolve()

    if not he_template_path.exists():
        raise FileNotFoundError(
            f"HE template not found: "
            f"{he_template_path}"
        )

    he_template = load_he_template(
        he_template_path
    )

    # These paths are interpreted when
    # run_he_temporal_compositor_v1.py
    # is run from HE_v_0.1.
    pair_name = experiment_name

    background_video_path = (
        rf"recordings\he_pairs"
        rf"\{pair_name}"
        rf"\background.mp4"
    )

    background_ego_pose_path = (
        rf"recordings\he_pairs"
        rf"\{pair_name}"
        rf"\background_ego_pose.jsonl"
    )

    he_output_dir = (
        rf"he_outputs"
        rf"\{pair_name}_he"
    )

    he_output_video_name = (
        f"{pair_name}_he.mp4"
    )

    he_json = (
        resolved_to_current_he_json(
            resolved=resolved,
            template_json=he_template,

            background_video_path=(
                background_video_path
            ),

            ego_pose_path=(
                background_ego_pose_path
            ),

            output_dir=(
                he_output_dir
            ),

            output_video_name=(
                he_output_video_name
            ),
        )
    )

    he_scenario_path = (
        experiment_dir
        / "he_scenario.json"
    )

    save_he_json(
        he_scenario_path,
        he_json,
    )

    # --------------------------------------------------------
    # 6. Manifest
    # --------------------------------------------------------

    manifest = {
        "experiment_name":
            experiment_name,

        "scenario_id":
            scenario.scenario_id,

        "source": {
            "type": (
                "request_json"
                if args.request_json
                else (
                    "parameters"
                    if args.scenario_type
                    else (
                        "input_json"
                        if args.input_json
                        else "demo"
                    )
                )
            ),

            "demo":
                args.demo,

            "input_json":
                args.input_json,

            "request_json":
                args.request_json,

            "scenario_type":
                args.scenario_type,
        },

        "artifacts": {
            "scenario_spec":
                str(
                    scenario_spec_path.resolve()
                ),

            "resolved_scenario":
                str(
                    resolved_path.resolve()
                ),

            "carla_plan":
                str(
                    carla_plan_path.resolve()
                ),
            "scenario_request": (
                str(
                    scenario_request_path.resolve()
                )
                if scenario_request_path
                is not None
                else None
            ),

            "he_scenario":
                str(
                    he_scenario_path.resolve()
                ),
        },

        "expected_runtime_outputs": {
            "pair_directory":
                str(
                    (
                        HE_ROOT
                        / "recordings"
                        / "he_pairs"
                        / pair_name
                    ).resolve()
                ),

            "carla_background_video":
                background_video_path,

            "carla_background_ego_pose":
                background_ego_pose_path,

            "he_output_dir":
                he_output_dir,

            "he_output_video":
                he_output_video_name,
        },

        "execution": {
            "fps":
                int(resolved.fps),

            "duration_s":
                float(
                    resolved.duration_s
                ),

            "ego_frames":
                len(
                    resolved.ego_frames
                ),

            "actor_frames":
                len(
                    resolved.frames
                ),

            "actors":
                len(
                    resolved.actors
                ),
        },

        "architecture": {
            "trajectory_source":
                "ResolvedScenario",

            "carla_uses_same_resolved_trajectory":
                True,

            "he_uses_same_resolved_trajectory":
                True,

            "he_sprite_selection":
                "viewpoint",
        },
    }

    manifest_path = (
        experiment_dir
        / "manifest.json"
    )

    save_json(
        manifest_path,
        manifest,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    if scenario_request_path is not None:
        print(
            f"ScenarioRequest:  "
            f"{scenario_request_path}"
        )
    print(
        "=" * 64
    )

    print(
        "Scenario build complete"
    )

    print(
        "=" * 64
    )

    print(
        f"Experiment:       "
        f"{experiment_name}"
    )

    print(
        f"FPS:              "
        f"{resolved.fps}"
    )

    print(
        f"Duration:         "
        f"{resolved.duration_s:.3f} s"
    )

    print(
        f"Ego frames:       "
        f"{len(resolved.ego_frames)}"
    )

    print(
        f"Actor frames:     "
        f"{len(resolved.frames)}"
    )

    print()

    print(
        f"ScenarioSpec:     "
        f"{scenario_spec_path}"
    )

    print(
        f"ResolvedScenario: "
        f"{resolved_path}"
    )

    print(
        f"CARLA plan:       "
        f"{carla_plan_path}"
    )

    print(
        f"HE scenario:      "
        f"{he_scenario_path}"
    )

    print(
        f"Manifest:         "
        f"{manifest_path}"
    )

    print(
        "=" * 64
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build one scenario into the shared "
            "ResolvedScenario -> CARLA / HE pipeline."
        )
    )

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--demo",
        choices=sorted(
            DEMO_BUILDERS.keys()
        ),
    )

    source.add_argument(
        "--input-json",
        type=str,
        help="Existing full ScenarioSpec JSON.",
    )

    source.add_argument(
        "--request-json",
        type=str,
        help="Semantic ScenarioRequest JSON.",
    )

    source.add_argument(
        "--scenario-type",
        choices=[
            "static",
            "oncoming",
            "following",
            "cut_in",
            "crossing",
        ],
        help=(
            "Build a scenario directly from "
            "semantic command-line parameters."
        ),
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Experiment/scenario name. "
            "Defaults to ScenarioSpec.scenario_id."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/scenarios",
    )

    parser.add_argument(
        "--he-template",
        type=str,
        default=(
            r"..\HE_v_0.1"
            r"\configs\scenarios"
            r"\pair_oncoming_straight_audi_x3_001_he.json"
        ),
        help=(
            "Validated HE template used for placement, "
            "sprite-bank and renderer configuration."
        ),
    )
        # --------------------------------------------------------
    # Parameterized scenario options
    # --------------------------------------------------------

    parser.add_argument(
        "--side",
        choices=[
            "left",
            "right",
        ],
        default=None,
    )

    parser.add_argument(
        "--start-distance",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--lane-y",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--target-lane-y",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--lane-width",
        type=float,
        default=3.5,
    )

    parser.add_argument(
        "--actor-speed",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--ego-speed",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--cut-start",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--cut-duration",
        type=float,
        default=3.0,
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_pipeline(
        parse_args()
    )