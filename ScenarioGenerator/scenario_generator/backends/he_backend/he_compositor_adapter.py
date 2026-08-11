from __future__ import annotations

from collections import defaultdict
from typing import Any

from scenario_generator.schema import ResolvedScenario
from scenario_generator.backends.he_backend.visibility import (
    check_front_camera_visibility,
    check_yaw_camera_visibility,
)


def resolved_to_he_compositor_json(scenario: ResolvedScenario) -> dict[str, Any]:
    """Convert ResolvedScenario to a simple HE compositor input JSON.

    This is intentionally conservative because the exact current compositor JSON may
    differ in your repo. Use this file as the adapter layer only; do not let
    compositor-specific fields leak into the core schema.

    v1 output:
    - camera: fixed HEPlacementModel v2 camera convention
    - ego_motion: simple constant speed metadata
    - actors: per-actor frame trajectory in local metric coordinates

    Later:
    - call HEPlacementModel v2 to convert x/y/yaw/camera into
      center_x, bottom_y, box_width, box_height, visible
    - or pass these states to your existing learned-placement adapter.
    """

    actor_frames: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for frame in scenario.frames:
        visibility = check_front_camera_visibility(
            x_m=frame.x_m,
            y_m=frame.y_m,
            fov_deg=scenario.camera.fov,
            min_distance_m=0.5,
            max_distance_m=120.0,
        )
        multi_camera_visibility = {}

        if scenario.camera_rig is not None:
            for view in scenario.camera_rig.views:
                view_visibility = check_yaw_camera_visibility(
                    x_m=frame.x_m,
                    y_m=frame.y_m,
                    camera_yaw_deg=view.yaw_deg,
                    fov_deg=view.fov,
                    min_distance_m=0.5,
                    max_distance_m=120.0,
                )
                multi_camera_visibility[view.camera_id] = {
                    "visible": view_visibility.visible,
                    "reason": view_visibility.reason,
                    "bearing_deg": view_visibility.bearing_deg,
                    "distance_m": view_visibility.distance_m,
                }        

        actor_frames[frame.actor_id].append(
            {
                "frame_idx": frame.frame_idx,
                "t_s": frame.t_s,

                # Full actor state is always kept.
                # Even if the actor is behind the front camera, it remains part
                # of the resolved scenario / BEV state memory.
                "relative_state": {
                    "x_m": frame.x_m,
                    "y_m": frame.y_m,
                    "yaw_deg": frame.yaw_deg,
                    "speed_mps": frame.speed_mps,
                },

                # Camera-specific visibility.
                # Current HE compositor is front-camera only.
                # Future 360/multi-camera backend can compute this per camera.
                "visibility": {
                    "front_camera_visible": visibility.visible,
                    "reason": visibility.reason,
                    "bearing_deg": visibility.bearing_deg,
                    "distance_m": visibility.distance_m,
                    "multi_camera": multi_camera_visibility,
                },

                # Placeholder for HEPlacementModel v2 / learned placement.
                # These fields should only be filled when front_camera_visible=True.
                "placement": {
                    "visible": visibility.visible,
                    "center_x": frame.center_x,
                    "bottom_y": frame.bottom_y,
                    "box_width": frame.box_width,
                    "box_height": frame.box_height,
                },
            }
        )

    return {
        "scenario_id": scenario.scenario_id,
        "duration_s": scenario.duration_s,
        "fps": scenario.fps,
        "camera": scenario.camera.model_dump(mode="json"),
        "camera_rig": (
            scenario.camera_rig.model_dump(mode="json")
            if scenario.camera_rig is not None
            else None
        ),
        "ego_motion": {
            "speed_mps": scenario.ego.speed_mps,
            "initial_x_m": scenario.ego.initial_x_m,
            "initial_y_m": scenario.ego.initial_y_m,
            "initial_yaw_deg": scenario.ego.initial_yaw_deg,
        },
        "actors": [
            {
                "actor_id": actor_id,
                "frames": frames,
            }
            for actor_id, frames in actor_frames.items()
        ],
    }
