"""
resolved_schema_v2.py

Backend-independent, frame-resolved physical scenario.

This is the physical truth consumed later by execution backends.

Coordinate convention:
    +x = forward
    +y = left
    +yaw = counter-clockwise / left

There are deliberately NO:
    - CARLA transforms
    - HE x/z coordinates
    - sprite angles
    - image-space bounding boxes
    - visibility decisions
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from scenario_generator.schema.scenario_schema_v2 import (
    ActorRoleV2,
    ActorTypeV2,
    CameraSpecV2,
    DimensionsV2,
)


class ResolvedEgoFrameV2(BaseModel):
    frame_idx: int = Field(ge=0)
    t_s: float = Field(ge=0.0)

    x_m: float
    y_m: float
    yaw_deg: float

    speed_mps: float = Field(ge=0.0)
    vx_mps: float
    vy_mps: float


class ResolvedActorInfoV2(BaseModel):
    actor_id: str

    actor_type: ActorTypeV2
    role: ActorRoleV2

    asset_key: Optional[str] = None

    dimensions_m: Optional[
        DimensionsV2
    ] = None

    spawn_time_s: float
    despawn_time_s: Optional[float]


class ResolvedActorFrameV2(BaseModel):
    frame_idx: int = Field(ge=0)
    t_s: float = Field(ge=0.0)

    actor_id: str

    x_m: float
    y_m: float
    yaw_deg: float

    speed_mps: float = Field(ge=0.0)
    vx_mps: float
    vy_mps: float


class ResolvedScenarioV2(BaseModel):
    schema_version: Literal[
        "2.0-resolved"
    ] = "2.0-resolved"

    source_schema_version: Literal[
        "2.0"
    ] = "2.0"

    scenario_id: str
    source_description: Optional[str] = None

    duration_s: float = Field(gt=0.0)
    fps: int = Field(gt=0)
    seed: int = 0

    coordinate_frame: Literal[
        "ego_initial"
    ] = "ego_initial"

    camera: CameraSpecV2

    actors: list[
        ResolvedActorInfoV2
    ]

    ego_frames: list[
        ResolvedEgoFrameV2
    ]

    # Flattened deliberately:
    #
    # actor_id + frame_idx uniquely identify one actor state.
    #
    # This makes lifecycle easy because an actor simply has no
    # frame before spawning or after despawning.
    actor_frames: list[
        ResolvedActorFrameV2
    ]