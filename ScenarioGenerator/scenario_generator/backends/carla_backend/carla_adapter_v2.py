"""
carla_adapter_v2.py

Pure backend adapter:

    ResolvedScenarioV2
            ↓
       CARLA plan

This file performs coordinate conversion and CARLA asset selection.

It does NOT:
    - connect to CARLA
    - spawn actors
    - tick the simulator
    - calculate trajectories

ScenarioGenerator physical frame:
    +x = forward
    +y = left
    +yaw = left / counter-clockwise

CARLA local execution frame used by our validated recorder:
    local_x = lateral right
    local_y = vertical
    local_z = forward
    local_yaw = right / clockwise

Therefore:

    CARLA local_x   = -SG y
    CARLA local_y   = 0
    CARLA local_z   =  SG x
    CARLA local_yaw = -SG yaw
"""

from __future__ import annotations

from collections import defaultdict

from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedActorFrameV2,
    ResolvedScenarioV2,
)


# ============================================================
# Backend-specific asset mapping
# ============================================================

CARLA_ASSET_MAP = {
    "sedan.generic": "vehicle.audi.tt",
}


def resolve_carla_asset(
    asset_key: str | None,
) -> str:
    """
    Map backend-independent asset key to a concrete CARLA blueprint.

    Later this map can contain:
        sedan.generic
        suv.generic
        hatchback.generic
        truck.generic
        pedestrian.generic
        ...
    """

    if asset_key is None:
        return "vehicle.audi.tt"

    if asset_key in CARLA_ASSET_MAP:
        return CARLA_ASSET_MAP[
            asset_key
        ]

    # Allow an explicitly supplied CARLA blueprint as an escape hatch.
    if asset_key.startswith(
        "vehicle."
    ):
        return asset_key

    raise ValueError(
        "No CARLA asset mapping for "
        f"asset_key={asset_key!r}"
    )


# ============================================================
# Coordinate conversion
# ============================================================

def sg_pose_to_carla_local(
    x_m: float,
    y_m: float,
    yaw_deg: float,
):
    return {
        "local_x_m": -float(y_m),
        "local_y_m": 0.0,
        "local_z_m": float(x_m),
        "local_yaw_deg": -float(
            yaw_deg
        ),
    }


# ============================================================
# Frame conversion
# ============================================================

def convert_ego_frame(
    frame,
):
    local = sg_pose_to_carla_local(
        frame.x_m,
        frame.y_m,
        frame.yaw_deg,
    )

    return {
        "frame_idx":
            frame.frame_idx,

        "t_s":
            frame.t_s,

        **local,

        "speed_mps":
            frame.speed_mps,

        "vx_sg_mps":
            frame.vx_mps,

        "vy_sg_mps":
            frame.vy_mps,
    }


def convert_actor_frame(
    frame: ResolvedActorFrameV2,
):
    local = sg_pose_to_carla_local(
        frame.x_m,
        frame.y_m,
        frame.yaw_deg,
    )

    return {
        "frame_idx":
            frame.frame_idx,

        "t_s":
            frame.t_s,

        **local,

        "speed_mps":
            frame.speed_mps,

        # Preserve ScenarioGenerator physical velocity for
        # traceability/debugging.
        "vx_sg_mps":
            frame.vx_mps,

        "vy_sg_mps":
            frame.vy_mps,
    }


# ============================================================
# Main adapter
# ============================================================

def resolved_v2_to_carla_plan(
    resolved: ResolvedScenarioV2,
):
    """
    Convert one frame-resolved physical scenario into a deterministic
    CARLA execution plan.

    Actor lifecycle is represented naturally:
        an actor only has frame states while it exists.
    """

    # --------------------------------------------------------
    # Group physical actor frames
    # --------------------------------------------------------

    grouped_frames = defaultdict(
        list
    )

    for frame in resolved.actor_frames:
        grouped_frames[
            frame.actor_id
        ].append(
            frame
        )

    # --------------------------------------------------------
    # Actors
    # --------------------------------------------------------

    actors = []

    for actor in resolved.actors:

        frames = sorted(
            grouped_frames.get(
                actor.actor_id,
                [],
            ),
            key=lambda row: row.frame_idx,
        )

        actors.append(
            {
                "actor_id":
                    actor.actor_id,

                "actor_type":
                    actor.actor_type.value,

                "role":
                    actor.role.value,

                "asset_key":
                    actor.asset_key,

                "blueprint":
                    resolve_carla_asset(
                        actor.asset_key
                    ),

                "dimensions_m": (
                    actor.dimensions_m.model_dump(
                        mode="json"
                    )
                    if actor.dimensions_m
                    is not None
                    else None
                ),

                "lifecycle": {
                    "spawn_time_s":
                        actor.spawn_time_s,

                    "despawn_time_s":
                        actor.despawn_time_s,

                    "first_frame_idx": (
                        frames[0].frame_idx
                        if frames
                        else None
                    ),

                    "last_frame_idx": (
                        frames[-1].frame_idx
                        if frames
                        else None
                    ),
                },

                "frames": [
                    convert_actor_frame(
                        frame
                    )
                    for frame in frames
                ],
            }
        )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = resolved.camera

    camera_json = {
        "image_width":
            camera.image_width,

        "image_height":
            camera.image_height,

        "fov":
            camera.fov_deg,

        "camera_x":
            camera.x_m,

        "camera_y":
            camera.y_m,

        "camera_z":
            camera.z_m,

        "pitch":
            camera.pitch_deg,

        "yaw":
            camera.yaw_deg,

        "roll":
            camera.roll_deg,
    }

    # --------------------------------------------------------
    # Final execution plan
    # --------------------------------------------------------

    return {
        "plan_version":
            "2.0",

        "scenario_id":
            resolved.scenario_id,

        "source_schema_version":
            resolved.source_schema_version,

        "coordinate_frame":
            "ego_initial",

        "fps":
            resolved.fps,

        "duration_s":
            resolved.duration_s,

        "seed":
            resolved.seed,

        "camera":
            camera_json,

        "ego": {
            "actor_id":
                "ego",

            "frames": [
                convert_ego_frame(
                    frame
                )
                for frame
                in resolved.ego_frames
            ],
        },

        "actors":
            actors,

        "traceability": {
            "trajectory_source":
                "ResolvedScenarioV2",

            "backend":
                "CARLA",

            "coordinate_conversion": {
                "local_x":
                    "-scenario_y",

                "local_y":
                    "0",

                "local_z":
                    "scenario_x",

                "local_yaw":
                    "-scenario_yaw",
            },
        },
    }