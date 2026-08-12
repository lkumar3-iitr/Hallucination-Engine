from __future__ import annotations

from collections import defaultdict
from typing import Any

from scenario_generator.schema import ResolvedScenario


def _actor_info_by_id(scenario: ResolvedScenario):
    return {
        actor.actor_id: actor
        for actor in scenario.actors
    }


def resolved_to_he_compositor_json(
    scenario: ResolvedScenario,
) -> dict[str, Any]:
    """
    Convert the backend-independent ResolvedScenario into a simple
    HE-oriented trajectory representation.

    IMPORTANT:
    This adapter does NOT perform:
        - image placement
        - visibility prediction
        - bbox generation
        - sprite selection

    Those belong to the actual HE runtime/backend.

    ScenarioGenerator coordinate convention:
        x = forward
        y = lateral, positive LEFT
        yaw positive = LEFT turn

    Current HE convention:
        z = forward/depth
        x = lateral, positive RIGHT
        y = vertical
        yaw positive = RIGHT turn

    Therefore:

        HE z   =  SG x
        HE x   = -SG y
        HE yaw = -SG yaw

        HE vz  =  SG vx
        HE vx  = -SG vy
    """

    actor_info = _actor_info_by_id(scenario)

    # ------------------------------------------------------------
    # Ego trajectory
    # ------------------------------------------------------------

    ego_frames = []

    for frame in scenario.ego_frames:
        ego_frames.append(
            {
                "frame_idx": frame.frame_idx,
                "t_s": frame.t_s,

                # Keep ScenarioGenerator physical coordinates here.
                # CARLA and HE backends can both consume this same
                # resolved physical trajectory.
                "scenario_state": {
                    "x_m": frame.x_m,
                    "y_m": frame.y_m,
                    "yaw_deg": frame.yaw_deg,
                    "speed_mps": frame.speed_mps,
                    "vx_mps": frame.vx_mps,
                    "vy_mps": frame.vy_mps,
                },

                # Explicit HE-coordinate equivalent.
                "he_state": {
                    "x_m": -frame.y_m,
                    "y_m": 0.0,
                    "z_m": frame.x_m,
                    "yaw_deg": -frame.yaw_deg,
                    "speed_mps": frame.speed_mps,
                    "vx_mps": -frame.vy_mps,
                    "vy_mps": 0.0,
                    "vz_mps": frame.vx_mps,
                },
            }
        )

    # ------------------------------------------------------------
    # Actor trajectories
    # ------------------------------------------------------------

    actor_frames: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for frame in scenario.frames:
        actor_frames[frame.actor_id].append(
            {
                "frame_idx": frame.frame_idx,
                "t_s": frame.t_s,

                # Original backend-independent physical state.
                "scenario_state": {
                    "x_m": frame.x_m,
                    "y_m": frame.y_m,
                    "yaw_deg": frame.yaw_deg,
                    "speed_mps": frame.speed_mps,
                    "vx_mps": frame.vx_mps,
                    "vy_mps": frame.vy_mps,
                },

                # Same state expressed in current HE coordinates.
                "he_state": {
                    "x_m": -frame.y_m,
                    "y_m": 0.0,
                    "z_m": frame.x_m,
                    "yaw_deg": -frame.yaw_deg,
                    "speed_mps": frame.speed_mps,
                    "vx_mps": -frame.vy_mps,
                    "vy_mps": 0.0,
                    "vz_mps": frame.vx_mps,
                },
            }
        )

    actors = []

    for actor_id, frames in actor_frames.items():
        info = actor_info[actor_id]

        actors.append(
            {
                "actor_id": actor_id,
                "role": info.role.value,
                "actor_type": info.actor_type.value,
                "blueprint": info.blueprint,
                "dimensions_m": list(info.dimensions_m),
                "frames": frames,
            }
        )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.source_description,

        "duration_s": scenario.duration_s,
        "fps": scenario.fps,

        "coordinate_frame": scenario.coordinate_frame,

        "camera": scenario.camera.model_dump(mode="json"),

        "camera_rig": (
            scenario.camera_rig.model_dump(mode="json")
            if scenario.camera_rig is not None
            else None
        ),

        "road": scenario.road.model_dump(mode="json"),

        "ego_frames": ego_frames,

        "actors": actors,

        # Explicitly document what this file is.
        "backend_notes": {
            "type": "resolved_he_trajectory_export",
            "placement_applied": False,
            "visibility_applied": False,
            "sprite_selection_applied": False,
            "scenario_generator_coordinates": {
                "x": "forward",
                "y": "left",
                "yaw_positive": "left",
            },
            "he_coordinates": {
                "x": "right",
                "y": "vertical",
                "z": "forward",
                "yaw_positive": "right",
            },
        },
    }