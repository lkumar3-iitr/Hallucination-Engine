from __future__ import annotations

from pathlib import Path
import sys
import json
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scenario_generator.backends.he_backend.current_he_adapter import (
    load_he_template,
    resolved_to_current_he_json,
    save_he_json,
)
from scenario_generator.io.json_io import save_model
from scenario_generator.planner.manual_planner import (
    make_oncoming_vehicle_demo,
)
from scenario_generator.trajectory.rule_based.rule_based_generator import (
    RuleBasedTrajectoryGenerator,
)
from scenario_generator.backends.carla_backend.carla_adapter_placeholder import (
    resolved_to_carla_plan,
)


def main() -> None:
    # --------------------------------------------------------
    # Build ScenarioSpec
    # --------------------------------------------------------

    scenario = make_oncoming_vehicle_demo()

    scenario.scenario_id = (
        "validation_oncoming_straight_audi_x3_001"
    )

    scenario.description = (
        "Validation scenario matching the previously validated "
        "CARLA/HE oncoming pair."
    )

    # Existing validated pair:
    # 150 frames at 30 FPS => frames 0 ... 149
    scenario.fps = 30
    scenario.duration_s = 149.0 / 30.0

    # --------------------------------------------------------
    # Canonical camera
    # --------------------------------------------------------

    scenario.camera.image_width = 1280
    scenario.camera.image_height = 720
    scenario.camera.fov = 90.0

    scenario.camera.camera_x = 1.5
    scenario.camera.camera_y = 0.0
    scenario.camera.camera_z = 1.6

    scenario.camera.pitch = 0.0
    scenario.camera.yaw = 0.0
    scenario.camera.roll = 0.0

    # --------------------------------------------------------
    # Ego
    #
    # ScenarioGenerator:
    #   +x = forward
    #   +y = left
    # --------------------------------------------------------

    scenario.ego.initial_x_m = 0.0
    scenario.ego.initial_y_m = 0.0
    scenario.ego.initial_yaw_deg = 0.0
    scenario.ego.speed_mps = 5.0

    # Default ego motion is straight in our cleaned schema.

    # --------------------------------------------------------
    # Adversary
    #
    # Validated HE pair:
    #   HE x = -3 m
    #   HE z = 60 m
    #
    # Therefore ScenarioGenerator:
    #   SG y = +3 m
    #   SG x = 60 m
    # --------------------------------------------------------

    actor = scenario.actors[0]

    actor.actor_id = "adv_001"
    actor.blueprint = "vehicle.audi.tt"

    actor.initial_x_m = 60.0
    actor.initial_y_m = 3.0
    actor.initial_yaw_deg = 180.0
    actor.initial_speed_mps = 5.0

    actor.trajectory.params["speed_mps"] = 5.0
    actor.trajectory.params["target_lane_y_m"] = 3.0

    # --------------------------------------------------------
    # Resolve once
    # --------------------------------------------------------

    generator = RuleBasedTrajectoryGenerator()
    resolved = generator.resolve(scenario)

    out_dir = (
        PROJECT_ROOT
        / "examples"
        / "validation"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_path = (
        out_dir
        / f"{scenario.scenario_id}.resolved.json"
    )

    save_model(
        resolved_path,
        resolved,
    )
        # --------------------------------------------------------
    # Convert SAME resolved scenario into CARLA execution plan
    # --------------------------------------------------------

    carla_plan = resolved_to_carla_plan(
        resolved
    )

    carla_plan_path = (
        out_dir
        / f"{scenario.scenario_id}.carla_plan.json"
    )

    with carla_plan_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            carla_plan,
            f,
            indent=2,
        )
        f.write("\n")
        
    # --------------------------------------------------------
    # Convert SAME resolved scenario into current HE format
    # --------------------------------------------------------

    he_root = (
        PROJECT_ROOT.parent
        / "HE_v_0.1"
    )

    template_path = (
        he_root
        / "configs"
        / "scenarios"
        / "pair_oncoming_straight_audi_x3_001_he.json"
    )

    template = load_he_template(
        template_path
    )

    he_json = resolved_to_current_he_json(
        resolved=resolved,
        template_json=template,

        # These paths are interpreted when the compositor runs
        # from HE_v_0.1.
        background_video_path=(
            r"recordings\he_pairs"
            r"\pair_oncoming_straight_audi_x3_001"
            r"\background.mp4"
        ),

        ego_pose_path=(
            r"recordings\he_pairs"
            r"\pair_oncoming_straight_audi_x3_001"
            r"\background_ego_pose.jsonl"
        ),

        output_dir=(
            r"he_outputs"
            r"\sg_validation_oncoming_001"
        ),

        output_video_name=(
            "sg_validation_oncoming_001.mp4"
        ),
    )

    he_output_path = (
        out_dir
        / f"{scenario.scenario_id}.current_he.json"
    )

    save_he_json(
        he_output_path,
        he_json,
    )

    print()
    print("Validation scenario generated")
    print("-----------------------------")
    print(f"Resolved: {resolved_path}")
    print(f"HE JSON:  {he_output_path}")
    print(f"Frames:   {len(resolved.frames)}")
    print(f"FPS:      {resolved.fps}")
    print()


if __name__ == "__main__":
    main()