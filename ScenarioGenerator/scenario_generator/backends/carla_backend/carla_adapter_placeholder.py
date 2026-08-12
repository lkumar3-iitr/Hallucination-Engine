from __future__ import annotations

from collections import defaultdict
from typing import Any

from scenario_generator.schema import ResolvedScenario


def sg_state_to_carla_local_state(
    x_m: float,
    y_m: float,
    yaw_deg: float,
) -> dict[str, float]:
    """
    Convert ScenarioGenerator coordinates to the local coordinate
    convention used by our CARLA execution layer.

    ScenarioGenerator:
        x = forward
        y = left
        +yaw = left

    CARLA local execution convention:
        local_z = forward
        local_x = right
        +yaw = right

    Therefore:

        CARLA local_z   =  SG x
        CARLA local_x   = -SG y
        CARLA local_yaw = -SG yaw

    This deliberately matches the local convention already validated
    by the HE/CARLA pair recorder.
    """

    return {
        "local_x_m": -float(y_m),
        "local_y_m": 0.0,
        "local_z_m": float(x_m),
        "local_yaw_deg": -float(yaw_deg),
    }


def resolved_to_carla_plan(
    scenario: ResolvedScenario,
) -> dict[str, Any]:
    """
    Convert one backend-independent ResolvedScenario into a
    deterministic CARLA execution plan.

    This function does NOT import CARLA and does NOT spawn anything.

    The executor will later:
        1. choose the CARLA initial ego transform,
        2. settle the ego vehicle,
        3. use that transform as the ego-initial origin,
        4. convert every local state here to a CARLA world transform,
        5. synchronously apply the exact resolved frame states.
    """

    # ------------------------------------------------------------
    # Ego frames
    # ------------------------------------------------------------

    ego_frames = []

    for frame in sorted(
        scenario.ego_frames,
        key=lambda f: f.frame_idx,
    ):
        local = sg_state_to_carla_local_state(
            x_m=frame.x_m,
            y_m=frame.y_m,
            yaw_deg=frame.yaw_deg,
        )

        ego_frames.append(
            {
                "frame_idx": int(frame.frame_idx),
                "t_s": float(frame.t_s),

                "local_x_m": local["local_x_m"],
                "local_y_m": local["local_y_m"],
                "local_z_m": local["local_z_m"],
                "local_yaw_deg": local["local_yaw_deg"],

                "speed_mps": float(frame.speed_mps),
            }
        )

    # ------------------------------------------------------------
    # Group actor frames
    # ------------------------------------------------------------

    frames_by_actor: dict[str, list] = defaultdict(list)

    for frame in scenario.frames:
        frames_by_actor[frame.actor_id].append(frame)

    actor_info = {
        actor.actor_id: actor
        for actor in scenario.actors
    }

    actors = []

    for actor_id, frames in frames_by_actor.items():
        frames.sort(key=lambda f: f.frame_idx)

        info = actor_info[actor_id]

        carla_frames = []

        for frame in frames:
            local = sg_state_to_carla_local_state(
                x_m=frame.x_m,
                y_m=frame.y_m,
                yaw_deg=frame.yaw_deg,
            )

            carla_frames.append(
                {
                    "frame_idx": int(frame.frame_idx),
                    "t_s": float(frame.t_s),

                    "local_x_m": local["local_x_m"],
                    "local_y_m": local["local_y_m"],
                    "local_z_m": local["local_z_m"],
                    "local_yaw_deg": local["local_yaw_deg"],

                    "speed_mps": float(frame.speed_mps),
                }
            )

        actors.append(
            {
                "actor_id": actor_id,
                "role": info.role.value,
                "actor_type": info.actor_type.value,
                "blueprint": info.blueprint,
                "dimensions_m": list(info.dimensions_m),
                "frames": carla_frames,
            }
        )

    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.source_description,

        "fps": int(scenario.fps),
        "duration_s": float(scenario.duration_s),

        "coordinate_frame": "ego_initial",

        "camera": scenario.camera.model_dump(mode="json"),

        "road": scenario.road.model_dump(mode="json"),

        "ego_frames": ego_frames,
        "actors": actors,

        "backend_notes": {
            "type": "resolved_carla_execution_plan",

            "scenario_generator_coordinates": {
                "x": "forward",
                "y": "left",
                "yaw_positive": "left",
            },

            "carla_local_coordinates": {
                "local_x": "right",
                "local_y": "vertical",
                "local_z": "forward",
                "yaw_positive": "right",
            },

            "coordinate_conversion": {
                "carla_local_x": "-scenario_y",
                "carla_local_z": "scenario_x",
                "carla_local_yaw": "-scenario_yaw",
            },
        },
    }