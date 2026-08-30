"""
scenario_schema_v2.py

ScenarioSchema v2.

Purpose
-------
This schema describes WHAT physical scenario should happen.

It must remain independent of:
    - CARLA transforms / blueprint IDs
    - HE sprite IDs
    - image-space bounding boxes
    - visibility decisions
    - renderer-specific placement

Coordinate convention
---------------------
ScenarioGenerator physical frame:

    +x = forward
    +y = left
    +yaw = counter-clockwise / left turn

All positions below are expressed in the ego-initial physical frame
unless another frame is explicitly introduced in a future version.

Architecture
------------
ScenarioSpecV2
      ↓
Trajectory Resolver
      ↓
ResolvedScenario
      ↓
CARLA / HE
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


# ============================================================
# Basic physical types
# ============================================================

class ActorTypeV2(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    UNKNOWN = "unknown"


class ActorRoleV2(str, Enum):
    ADVERSARY = "adversary"
    TRAFFIC = "traffic"
    BACKGROUND = "background"


class DimensionsV2(BaseModel):
    """
    Physical actor dimensions.

    These are physical dimensions, not image-space dimensions.
    """

    length_m: float = Field(gt=0.0)
    width_m: float = Field(gt=0.0)
    height_m: float = Field(gt=0.0)


class Pose2DV2(BaseModel):
    """
    Physical pose in the ScenarioGenerator coordinate frame.
    """

    x_m: float
    y_m: float
    yaw_deg: float = 0.0


# ============================================================
# Actor lifecycle
# ============================================================

class ActorLifecycleV2(BaseModel):
    """
    Determines when an actor exists in the scenario.

    Times are absolute scenario times.

    spawn_time_s:
        Actor becomes active at this time.

    despawn_time_s:
        Actor disappears after this time.
        None means actor remains until scenario end.
    """

    spawn_time_s: float = Field(
        default=0.0,
        ge=0.0,
    )

    despawn_time_s: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    @model_validator(mode="after")
    def validate_lifecycle(
        self,
    ) -> "ActorLifecycleV2":

        if (
            self.despawn_time_s is not None
            and self.despawn_time_s
            <= self.spawn_time_s
        ):
            raise ValueError(
                "despawn_time_s must be greater "
                "than spawn_time_s"
            )

        return self


# ============================================================
# Motion mode 1: maneuver
# ============================================================

class ManeuverTypeV2(str, Enum):
    STATIC = "static"
    STRAIGHT = "straight"
    FOLLOWING = "following"
    ONCOMING = "oncoming"
    CUT_IN = "cut_in"
    CROSSING = "crossing"
    LEFT_TURN = "left_turn"
    RIGHT_TURN = "right_turn"


class ManeuverMotionV2(BaseModel):
    """
    High-level maneuver specification.

    This preserves the convenient interface we already have.

    The trajectory resolver converts the maneuver into exact
    per-frame physical states.
    """

    mode: Literal["maneuver"] = "maneuver"

    maneuver: ManeuverTypeV2

    speed_mps: float = Field(
        default=0.0,
        ge=0.0,
    )

    target_y_m: Optional[float] = None

    start_time_s: float = Field(
        default=0.0,
        ge=0.0,
    )

    duration_s: Optional[float] = Field(
        default=None,
        gt=0.0,
    )


# ============================================================
# Motion mode 2: geometric path
# ============================================================

class PathWaypointV2(BaseModel):
    """
    Geometric path point.

    No timestamp is required.

    The trajectory resolver decides when the actor reaches each
    waypoint using target_speed_mps or later motion constraints.
    """

    x_m: float
    y_m: float


class PathMotionV2(BaseModel):
    """
    Follow a geometric path.

    The actor begins at ActorSpecV2.spawn and then follows the
    supplied waypoints.

    Yaw is derived from the direction of travel by the resolver.
    """

    mode: Literal["path"] = "path"

    waypoints: list[PathWaypointV2] = Field(
        min_length=1,
    )

    target_speed_mps: float = Field(
        gt=0.0,
    )

    interpolation: Literal[
        "linear",
        "smooth",
    ] = "smooth"


# ============================================================
# Motion mode 3: timed keyframes
# ============================================================

class TimedKeyframeV2(BaseModel):
    """
    Exact time-constrained physical state.

    t_s is absolute scenario time.
    """

    t_s: float = Field(
        ge=0.0,
    )

    x_m: float
    y_m: float
    yaw_deg: float


class KeyframeMotionV2(BaseModel):
    """
    Explicit timed trajectory constraints.

    This is useful when exact event timing is more important than
    geometric path generation.

    The resolver interpolates between consecutive keyframes.
    """

    mode: Literal["keyframes"] = "keyframes"

    keyframes: list[TimedKeyframeV2] = Field(
        min_length=1,
    )

    interpolation: Literal[
        "linear",
        "smooth",
    ] = "linear"

    @model_validator(mode="after")
    def validate_keyframes(
        self,
    ) -> "KeyframeMotionV2":

        times = [
            k.t_s
            for k in self.keyframes
        ]

        if times != sorted(times):
            raise ValueError(
                "keyframes must be ordered by t_s"
            )

        if len(times) != len(set(times)):
            raise ValueError(
                "keyframe timestamps must be unique"
            )

        return self

# ============================================================
# Motion mode 4: physical behavior sequence
# ============================================================

class CruiseStepV2(BaseModel):
    step: Literal["cruise"] = "cruise"
    duration_s: float = Field(gt=0.0)
    speed_mps: Optional[float] = Field(
        default=None,
        ge=0.0,
    )


class LaneChangeStepV2(BaseModel):
    step: Literal["lane_change"] = "lane_change"

    # ScenarioGenerator convention:
    # positive = left, negative = right.
    lateral_delta_m: float

    duration_s: float = Field(
        gt=0.0,
    )


class AccelerateStepV2(BaseModel):
    step: Literal["accelerate"] = "accelerate"

    target_speed_mps: float = Field(
        ge=0.0,
    )

    acceleration_mps2: float = Field(
        gt=0.0,
    )


class BrakeStepV2(BaseModel):
    step: Literal["brake"] = "brake"

    target_speed_mps: float = Field(
        default=0.0,
        ge=0.0,
    )

    deceleration_mps2: float = Field(
        gt=0.0,
    )


class HoldStepV2(BaseModel):
    step: Literal["hold"] = "hold"

    duration_s: float = Field(
        gt=0.0,
    )


SequenceStepV2 = Annotated[
    Union[
        CruiseStepV2,
        LaneChangeStepV2,
        AccelerateStepV2,
        BrakeStepV2,
        HoldStepV2,
    ],
    Field(discriminator="step"),
]


class SequenceMotionV2(BaseModel):
    """
    Ordered deterministic physical behavior.

    This is still backend-independent.  It describes actor motion,
    not CARLA controls or HE rendering.

    Each step begins from the exact physical state produced by the
    previous step.
    """

    mode: Literal["sequence"] = "sequence"

    initial_speed_mps: float = Field(
        default=0.0,
        ge=0.0,
    )

    steps: list[SequenceStepV2] = Field(
        min_length=1,
    )

    end_behavior: Literal[
        "hold",
        "continue",
    ] = "hold"
# ============================================================
# Discriminated motion union
# ============================================================

ActorMotionV2 = Annotated[
    Union[
        ManeuverMotionV2,
        PathMotionV2,
        KeyframeMotionV2,
        SequenceMotionV2,
    ],
    Field(discriminator="mode"),
]


# ============================================================
# Actor specification
# ============================================================

class ActorSpecV2(BaseModel):
    """
    Generic physical actor.

    asset_key is intentionally simulator-independent.

    Example:
        sedan.generic

    CARLA may later map:
        sedan.generic -> vehicle.audi.tt

    HE may independently map:
        sedan.generic -> corresponding sprite bank
    """

    actor_id: str = Field(
        min_length=1,
    )

    actor_type: ActorTypeV2 = (
        ActorTypeV2.VEHICLE
    )

    role: ActorRoleV2 = (
        ActorRoleV2.ADVERSARY
    )

    asset_key: Optional[str] = None

    dimensions_m: Optional[
        DimensionsV2
    ] = None

    spawn: Pose2DV2

    lifecycle: ActorLifecycleV2 = Field(
        default_factory=ActorLifecycleV2
    )

    motion: ActorMotionV2


# ============================================================
# Ego specification
# ============================================================

class EgoSpecV2(BaseModel):
    """
    Ego physical specification.

    Paper 1 uses scripted ego motion.

    A future version can add:
        controlled
        reactive
        externally_driven
    without changing actor trajectory representation.
    """

    actor_id: str = "ego"

    spawn: Pose2DV2 = Field(
        default_factory=lambda: Pose2DV2(
            x_m=0.0,
            y_m=0.0,
            yaw_deg=0.0,
        )
    )

    motion: ActorMotionV2


# ============================================================
# Camera
# ============================================================

class CameraSpecV2(BaseModel):
    image_width: int = Field(
        default=1280,
        gt=0,
    )

    image_height: int = Field(
        default=720,
        gt=0,
    )

    fov_deg: float = Field(
        default=90.0,
        gt=0.0,
        lt=180.0,
    )

    x_m: float = 1.5
    y_m: float = 0.0
    z_m: float = 1.6

    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0


# ============================================================
# Scenario
# ============================================================

class ScenarioSpecV2(BaseModel):
    """
    Backend-independent scenario description.

    This is NOT the final frame-resolved representation.

    The resolver converts this into ResolvedScenario.
    """

    schema_version: Literal["2.0"] = "2.0"

    scenario_id: str = Field(
        min_length=1,
    )

    description: Optional[str] = None

    duration_s: float = Field(
        gt=0.0,
    )

    fps: int = Field(
        gt=0,
    )

    seed: int = 0

    ego: EgoSpecV2

    actors: list[ActorSpecV2] = Field(
        default_factory=list,
    )

    camera: CameraSpecV2 = Field(
        default_factory=CameraSpecV2
    )

    @model_validator(mode="after")
    def validate_scenario(
        self,
    ) -> "ScenarioSpecV2":

        # ----------------------------------------------------
        # Unique actor IDs
        # ----------------------------------------------------

        ids = [
            self.ego.actor_id
        ] + [
            actor.actor_id
            for actor in self.actors
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "All actor_id values must be unique."
            )

        # ----------------------------------------------------
        # Lifecycle must fit inside scenario
        # ----------------------------------------------------

        for actor in self.actors:

            lifecycle = actor.lifecycle

            if (
                lifecycle.spawn_time_s
                > self.duration_s
            ):
                raise ValueError(
                    f"{actor.actor_id}: "
                    "spawn_time_s exceeds "
                    "scenario duration"
                )

            if (
                lifecycle.despawn_time_s
                is not None
                and lifecycle.despawn_time_s
                > self.duration_s
            ):
                raise ValueError(
                    f"{actor.actor_id}: "
                    "despawn_time_s exceeds "
                    "scenario duration"
                )

            # -----------------------------------------------
            # Timed keyframes must lie inside actor lifetime
            # -----------------------------------------------

            if isinstance(
                actor.motion,
                KeyframeMotionV2,
            ):
                for keyframe in (
                    actor.motion.keyframes
                ):
                    if (
                        keyframe.t_s
                        < lifecycle.spawn_time_s
                    ):
                        raise ValueError(
                            f"{actor.actor_id}: "
                            f"keyframe at "
                            f"{keyframe.t_s}s occurs "
                            "before actor spawn"
                        )

                    actor_end = (
                        lifecycle.despawn_time_s
                        if lifecycle.despawn_time_s
                        is not None
                        else self.duration_s
                    )

                    if (
                        keyframe.t_s
                        > actor_end
                    ):
                        raise ValueError(
                            f"{actor.actor_id}: "
                            f"keyframe at "
                            f"{keyframe.t_s}s occurs "
                            "after actor lifetime"
                        )

        return self