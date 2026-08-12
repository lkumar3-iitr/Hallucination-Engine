from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from scenario_generator.schema import (
    ActorRole,
    ActorSpec,
    ActorType,
    ManeuverType,
    RoadConfig,
    ScenarioSpec,
    TrajectoryMode,
    TrajectorySpec,
)


ScenarioType = Literal[
    "static",
    "oncoming",
    "following",
    "cut_in",
    "crossing",
]

SideType = Literal["left", "right"]


def _safe_float_tag(value: float) -> str:
    """Convert float to filename/id-safe tag."""
    s = f"{float(value):.2f}"
    s = s.rstrip("0").rstrip(".")
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def _make_scenario_id(
    scenario_type: str,
    side: Optional[str],
    start_distance_m: float,
    speed_mps: float,
) -> str:
    parts = [scenario_type]

    if side is not None:
        parts.append(side)

    parts.append(f"d{_safe_float_tag(start_distance_m)}")
    parts.append(f"v{_safe_float_tag(speed_mps)}")
    parts.append("v1")

    return "_".join(parts)


def side_to_lane_y(
    side: SideType,
    lane_width_m: float = 3.5,
) -> float:
    """
    Convert semantic side to ScenarioGenerator BEV coordinates.

    ScenarioGenerator convention:
        +y = left
        -y = right

    Backend-specific coordinate conversion must NOT happen here.
    """

    if side == "left":
        return +float(lane_width_m)

    if side == "right":
        return -float(lane_width_m)

    raise ValueError(f"Unsupported side: {side}")

def make_static_scenario(
    start_distance_m: float = 30.0,
    lane_y_m: float = 0.0,
    duration_s: float = 8.0,
    fps: int = 30,
    ego_speed_mps: float = 5.0,
    actor_id: str = "adv_001",
    scenario_id: Optional[str] = None,
) -> ScenarioSpec:
    if scenario_id is None:
        scenario_id = _make_scenario_id(
            scenario_type="static",
            side=None,
            start_distance_m=start_distance_m,
            speed_mps=0.0,
        )

    scenario = ScenarioSpec(
        scenario_id=scenario_id,
        description=(
            f"Static vehicle at {start_distance_m:.1f} m, "
            f"lane_y={lane_y_m:.1f} m."
        ),
        duration_s=duration_s,
        fps=fps,
        actors=[
            ActorSpec(
                actor_id=actor_id,
                role=ActorRole.ADVERSARY,
                actor_type=ActorType.VEHICLE,
                blueprint="vehicle.sedan.generic",
                initial_x_m=start_distance_m,
                initial_y_m=lane_y_m,
                initial_yaw_deg=0.0,
                initial_speed_mps=0.0,
                dimensions_m=(4.5, 1.8, 1.6),
                trajectory=TrajectorySpec(
                    mode=TrajectoryMode.RULE_BASED,
                    maneuver=ManeuverType.STATIC,
                    params={},
                ),
            )
        ],
    )

    scenario.ego.speed_mps = ego_speed_mps
    return scenario


def make_oncoming_scenario(
    start_distance_m: float = 55.0,
    lane_y_m: float = -3.5,
    speed_mps: float = 8.0,
    duration_s: float = 8.0,
    fps: int = 30,
    ego_speed_mps: float = 5.0,
    actor_id: str = "adv_001",
    scenario_id: Optional[str] = None,
) -> ScenarioSpec:
    if scenario_id is None:
        scenario_id = _make_scenario_id(
            scenario_type="oncoming",
            side=None,
            start_distance_m=start_distance_m,
            speed_mps=speed_mps,
        )

    scenario = ScenarioSpec(
        scenario_id=scenario_id,
        description=(
            f"Oncoming vehicle starts at {start_distance_m:.1f} m, "
            f"lane_y={lane_y_m:.1f} m, speed={speed_mps:.1f} m/s."
        ),
        duration_s=duration_s,
        fps=fps,
        actors=[
            ActorSpec(
                actor_id=actor_id,
                role=ActorRole.ADVERSARY,
                actor_type=ActorType.VEHICLE,
                blueprint="vehicle.sedan.generic",
                initial_x_m=start_distance_m,
                initial_y_m=lane_y_m,
                initial_yaw_deg=180.0,
                initial_speed_mps=speed_mps,
                dimensions_m=(4.5, 1.8, 1.6),
                trajectory=TrajectorySpec(
                    mode=TrajectoryMode.RULE_BASED,
                    maneuver=ManeuverType.ONCOMING,
                    params={
                        "speed_mps": speed_mps,
                        "target_lane_y_m": lane_y_m,
                    },
                ),
            )
        ],
    )

    scenario.ego.speed_mps = ego_speed_mps
    return scenario


def make_following_scenario(
    start_distance_m: float = 25.0,
    lane_y_m: float = 0.0,
    speed_mps: float = 4.0,
    duration_s: float = 8.0,
    fps: int = 30,
    ego_speed_mps: float = 5.0,
    actor_id: str = "adv_001",
    scenario_id: Optional[str] = None,
) -> ScenarioSpec:
    if scenario_id is None:
        scenario_id = _make_scenario_id(
            scenario_type="following",
            side=None,
            start_distance_m=start_distance_m,
            speed_mps=speed_mps,
        )

    scenario = ScenarioSpec(
        scenario_id=scenario_id,
        description=(
            f"Vehicle ahead moving in same direction from {start_distance_m:.1f} m, "
            f"lane_y={lane_y_m:.1f} m, speed={speed_mps:.1f} m/s."
        ),
        duration_s=duration_s,
        fps=fps,
        actors=[
            ActorSpec(
                actor_id=actor_id,
                role=ActorRole.ADVERSARY,
                actor_type=ActorType.VEHICLE,
                blueprint="vehicle.sedan.generic",
                initial_x_m=start_distance_m,
                initial_y_m=lane_y_m,
                initial_yaw_deg=0.0,
                initial_speed_mps=speed_mps,
                dimensions_m=(4.5, 1.8, 1.6),
                trajectory=TrajectorySpec(
                    mode=TrajectoryMode.RULE_BASED,
                    maneuver=ManeuverType.FOLLOWING,
                    params={
                        "speed_mps": speed_mps,
                    },
                ),
            )
        ],
    )

    scenario.ego.speed_mps = ego_speed_mps
    return scenario


def make_cut_in_scenario(
    side: SideType = "right",
    start_distance_m: float = 28.0,
    speed_mps: float = 4.0,
    target_lane_y_m: float = 0.0,
    lane_width_m: float = 3.5,
    cut_start_s: float = 1.5,
    cut_duration_s: float = 3.0,
    duration_s: float = 8.0,
    fps: int = 30,
    ego_speed_mps: float = 5.0,
    actor_id: str = "adv_001",
    scenario_id: Optional[str] = None,
) -> ScenarioSpec:
    start_lane_y_m = side_to_lane_y(
        side=side,
        lane_width_m=lane_width_m,
    )

    if scenario_id is None:
        scenario_id = _make_scenario_id(
            scenario_type="cut_in",
            side=side,
            start_distance_m=start_distance_m,
            speed_mps=speed_mps,
        )

    road = RoadConfig(
        lane_width_m=lane_width_m,
        num_lanes_same_direction=2,
        num_lanes_opposite_direction=1,
        ego_lane_index=0,
        road_length_m=120.0,
    )

    scenario = ScenarioSpec(
        scenario_id=scenario_id,
        description=(
            f"Vehicle cuts in from {side} side. "
            f"start_distance={start_distance_m:.1f} m, "
            f"start_y={start_lane_y_m:.1f} m, target_y={target_lane_y_m:.1f} m, "
            f"speed={speed_mps:.1f} m/s."
        ),
        duration_s=duration_s,
        fps=fps,
        road=road,
        actors=[
            ActorSpec(
                actor_id=actor_id,
                role=ActorRole.ADVERSARY,
                actor_type=ActorType.VEHICLE,
                blueprint="vehicle.sedan.generic",
                initial_x_m=start_distance_m,
                initial_y_m=start_lane_y_m,
                initial_yaw_deg=0.0,
                initial_speed_mps=speed_mps,
                dimensions_m=(4.5, 1.8, 1.6),
                trajectory=TrajectorySpec(
                    mode=TrajectoryMode.RULE_BASED,
                    maneuver=ManeuverType.CUT_IN,
                    params={
                        "speed_mps": speed_mps,
                        "target_y_m": target_lane_y_m,
                        "cut_start_s": cut_start_s,
                        "cut_duration_s": cut_duration_s,
                    },
                ),
            )
        ],
    )

    scenario.ego.speed_mps = ego_speed_mps
    return scenario


def make_crossing_scenario(
    side: SideType = "right",
    crossing_x_m: float = 25.0,
    start_lane_y_m: Optional[float] = None,
    speed_mps: float = 3.0,
    lane_width_m: float = 3.5,
    duration_s: float = 8.0,
    fps: int = 30,
    ego_speed_mps: float = 5.0,
    actor_id: str = "adv_001",
    scenario_id: Optional[str] = None,
) -> ScenarioSpec:
    """Make a lateral crossing scenario.

    For current v1:
      side="right" means the actor starts on the right visual side and moves left.
      side="left" means the actor starts on the left visual side and moves right.
    """

    if start_lane_y_m is None:
        start_lane_y_m = side_to_lane_y(
            side=side,
            lane_width_m=lane_width_m,
        ) * 2.0

    if side == "right":
        direction = -1.0
    elif side == "left":
        direction = +1.0
    else:
        raise ValueError(f"Unsupported side: {side}")

    if scenario_id is None:
        scenario_id = _make_scenario_id(
            scenario_type="crossing",
            side=side,
            start_distance_m=crossing_x_m,
            speed_mps=speed_mps,
        )

    scenario = ScenarioSpec(
        scenario_id=scenario_id,
        description=(
            f"Crossing actor from {side} side at x={crossing_x_m:.1f} m, "
            f"speed={speed_mps:.1f} m/s."
        ),
        duration_s=duration_s,
        fps=fps,
        actors=[
            ActorSpec(
                actor_id=actor_id,
                role=ActorRole.ADVERSARY,
                actor_type=ActorType.VEHICLE,
                blueprint="vehicle.sedan.generic",
                initial_x_m=crossing_x_m,
                initial_y_m=start_lane_y_m,
                initial_yaw_deg=270.0 if side == "right" else 90.0,
                initial_speed_mps=speed_mps,
                dimensions_m=(4.5, 1.8, 1.6),
                trajectory=TrajectorySpec(
                    mode=TrajectoryMode.RULE_BASED,
                    maneuver=ManeuverType.CROSSING,
                    params={
                        "speed_mps": speed_mps,
                        "target_x_m": crossing_x_m,
                        "direction": direction,
                    },
                ),
            )
        ],
    )

    scenario.ego.speed_mps = ego_speed_mps
    return scenario


def make_scenario_from_params(
    scenario_type: ScenarioType,
    side: Optional[SideType] = None,
    start_distance_m: float = 30.0,
    lane_y_m: float = 0.0,
    target_lane_y_m: float = 0.0,
    speed_mps: float = 4.0,
    duration_s: float = 8.0,
    fps: int = 30,
    ego_speed_mps: float = 5.0,
    lane_width_m: float = 3.5,
    cut_start_s: float = 1.5,
    cut_duration_s: float = 3.0,
    scenario_id: Optional[str] = None,
) -> ScenarioSpec:
    """Main factory entry point.

    This is the function the future LLM planner should call after converting
    text intent into structured scenario parameters.
    """

    if scenario_type == "static":
        return make_static_scenario(
            start_distance_m=start_distance_m,
            lane_y_m=lane_y_m,
            duration_s=duration_s,
            fps=fps,
            ego_speed_mps=ego_speed_mps,
            scenario_id=scenario_id,
        )

    if scenario_type == "oncoming":
        return make_oncoming_scenario(
            start_distance_m=start_distance_m,
            lane_y_m=lane_y_m,
            speed_mps=speed_mps,
            duration_s=duration_s,
            fps=fps,
            ego_speed_mps=ego_speed_mps,
            scenario_id=scenario_id,
        )

    if scenario_type == "following":
        return make_following_scenario(
            start_distance_m=start_distance_m,
            lane_y_m=lane_y_m,
            speed_mps=speed_mps,
            duration_s=duration_s,
            fps=fps,
            ego_speed_mps=ego_speed_mps,
            scenario_id=scenario_id,
        )

    if scenario_type == "cut_in":
        if side is None:
            side = "right"

        return make_cut_in_scenario(
            side=side,
            start_distance_m=start_distance_m,
            speed_mps=speed_mps,
            target_lane_y_m=target_lane_y_m,
            lane_width_m=lane_width_m,
            cut_start_s=cut_start_s,
            cut_duration_s=cut_duration_s,
            duration_s=duration_s,
            fps=fps,
            ego_speed_mps=ego_speed_mps,
            scenario_id=scenario_id,
        )

    if scenario_type == "crossing":
        if side is None:
            side = "right"

        return make_crossing_scenario(
            side=side,
            crossing_x_m=start_distance_m,
            speed_mps=speed_mps,
            lane_width_m=lane_width_m,
            duration_s=duration_s,
            fps=fps,
            ego_speed_mps=ego_speed_mps,
            scenario_id=scenario_id,
        )

    raise ValueError(f"Unsupported scenario_type: {scenario_type}")

