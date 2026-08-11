from __future__ import annotations

from scenario_generator.schema import (
    ActorRole,
    ActorSpec,
    ActorType,
    CameraRigConfig,
    ManeuverType,
    ScenarioSpec,
    TrajectoryMode,
    TrajectorySpec,
)


def make_oncoming_vehicle_demo() -> ScenarioSpec:
    """Temporary manual planner.

    Later this is replaced by:
    text/user intent -> LLM scenario planner -> ScenarioSpec.
    """

    return ScenarioSpec(
        scenario_id="demo_oncoming_vehicle_v1",
        description="One oncoming vehicle approaches ego in the opposite lane.",
        duration_s=8.0,
        fps=10,
        actors=[
            ActorSpec(
                actor_id="adv_001",
                role=ActorRole.ADVERSARY,
                actor_type=ActorType.VEHICLE,
                blueprint="vehicle.sedan.generic",
                initial_x_m=55.0,
                initial_y_m=-3.5,
                initial_yaw_deg=180.0,
                initial_speed_mps=8.0,
                trajectory=TrajectorySpec(
                    mode=TrajectoryMode.RULE_BASED,
                    maneuver=ManeuverType.ONCOMING,
                    params={
                        "speed_mps": 8.0,
                        "target_lane_y_m": -3.5,
                    },
                ),
            )
        ],
    )


def make_static_vehicle_demo() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="demo_static_vehicle_v1",
        description="One static vehicle ahead of ego.",
        duration_s=8.0,
        fps=10,
        actors=[
            ActorSpec(
                actor_id="adv_001",
                role=ActorRole.ADVERSARY,
                actor_type=ActorType.VEHICLE,
                blueprint="vehicle.sedan.generic",
                initial_x_m=35.0,
                initial_y_m=0.0,
                initial_yaw_deg=0.0,
                initial_speed_mps=0.0,
                trajectory=TrajectorySpec(
                    mode=TrajectoryMode.RULE_BASED,
                    maneuver=ManeuverType.STATIC,
                    params={},
                ),
            )
        ],
    )


def make_cut_in_vehicle_demo() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="demo_cut_in_vehicle_v1",
        description="One vehicle starts in the adjacent lane and cuts into ego lane.",
        duration_s=8.0,
        fps=10,
        actors=[
            ActorSpec(
                actor_id="adv_001",
                role=ActorRole.ADVERSARY,
                actor_type=ActorType.VEHICLE,
                blueprint="vehicle.sedan.generic",
                initial_x_m=28.0,
                initial_y_m=3.5,
                initial_yaw_deg=0.0,
                initial_speed_mps=4.0,
                trajectory=TrajectorySpec(
                    mode=TrajectoryMode.RULE_BASED,
                    maneuver=ManeuverType.CUT_IN,
                    params={
                        "speed_mps": 4.0,
                        "target_y_m": 0.0,
                        "cut_start_s": 1.5,
                        "cut_duration_s": 3.0,
                    },
                ),
            )
        ],
    )

def make_oncoming_vehicle_with_360_rig_demo() -> ScenarioSpec:
    """Oncoming demo with future 360 camera-rig metadata.

    The actor passes the ego and remains in the resolved state even after
    it is no longer visible in the front camera. This is important for BEV
    memory and future multi-camera / 360-view rendering.
    """

    scenario = make_oncoming_vehicle_demo()
    scenario.scenario_id = "demo_oncoming_vehicle_360_v1"
    scenario.description = (
        "One oncoming vehicle approaches, passes ego, and remains in BEV memory. "
        "A four-view 360 camera rig is included for visibility debugging."
    )
    scenario.camera_rig = CameraRigConfig.default_four_view_360()
    return scenario

def make_cut_in_from_right_demo() -> ScenarioSpec:
    """Receding vehicle cuts in from right side to ego lane."""

    scenario = make_cut_in_vehicle_demo()
    scenario.scenario_id = "demo_cut_in_from_right_v1"
    scenario.description = "Receding vehicle cuts in from the right side into ego lane."

    actor = scenario.actors[0]
    actor.initial_x_m = 28.0
    actor.initial_y_m = 3.5
    actor.initial_yaw_deg = 0.0
    actor.initial_speed_mps = 4.0
    actor.trajectory.params["target_y_m"] = 0.0
    actor.trajectory.params["speed_mps"] = 4.0

    return scenario


def make_cut_in_from_left_demo() -> ScenarioSpec:
    """Receding vehicle cuts in from left side to ego lane."""

    scenario = make_cut_in_vehicle_demo()
    scenario.scenario_id = "demo_cut_in_from_left_v1"
    scenario.description = "Receding vehicle cuts in from the left side into ego lane."

    actor = scenario.actors[0]
    actor.initial_x_m = 28.0
    actor.initial_y_m = -3.5
    actor.initial_yaw_deg = 0.0
    actor.initial_speed_mps = 4.0
    actor.trajectory.params["target_y_m"] = 0.0
    actor.trajectory.params["speed_mps"] = 4.0

    return scenario


def make_fast_oncoming_demo() -> ScenarioSpec:
    scenario = make_oncoming_vehicle_demo()
    scenario.scenario_id = "demo_fast_oncoming_vehicle_v1"
    scenario.description = "Fast oncoming vehicle approaches ego in opposite lane."

    actor = scenario.actors[0]
    actor.initial_x_m = 70.0
    actor.initial_y_m = -3.5
    actor.initial_speed_mps = 12.0
    actor.trajectory.params["speed_mps"] = 12.0

    return scenario


def make_slow_static_near_demo() -> ScenarioSpec:
    scenario = make_static_vehicle_demo()
    scenario.scenario_id = "demo_static_near_vehicle_v1"
    scenario.description = "Static vehicle stopped near ego in ego lane."

    actor = scenario.actors[0]
    actor.initial_x_m = 22.0
    actor.initial_y_m = 0.0

    return scenario