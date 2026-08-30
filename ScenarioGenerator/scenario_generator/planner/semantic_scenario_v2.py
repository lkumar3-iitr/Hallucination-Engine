from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from scenario_generator.planner.behavior_compiler_v2 import (
    SemanticBehaviorV2,
    compile_semantic_behavior,
    lane_relation_to_offset_m,
)

from scenario_generator.schema.scenario_schema_v2 import (
    ActorLifecycleV2,
    ActorRoleV2,
    ActorSpecV2,
    ActorTypeV2,
    CameraSpecV2,
    DimensionsV2,
    EgoSpecV2,
    Pose2DV2,
    ScenarioSpecV2,
)


# ============================================================
# Semantic actor
# ============================================================

class SemanticActorSpecV2(BaseModel):
    """
    Human/LLM-facing actor description.

    Position is specified relative to the initial ego rather than
    with raw ScenarioGenerator x/y coordinates.
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

    # Positive = ahead of initial ego.
    # Negative = behind initial ego.
    longitudinal_offset_m: float = 0.0

    behavior: SemanticBehaviorV2

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
    ) -> "SemanticActorSpecV2":

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
# Semantic scenario
# ============================================================

class SemanticScenarioV2(BaseModel):
    """
    High-level backend-independent scenario description.

    This is intended to be the deterministic target produced by
    CLI parameters, dataset templates, or an LLM planner.
    """

    scenario_id: str = Field(
        min_length=1,
    )

    description: Optional[str] = None

    duration_s: float = Field(
        gt=0.0,
    )

    fps: int = Field(
        default=20,
        gt=0,
    )

    seed: int = 0

    lane_width_m: float = Field(
        default=3.5,
        gt=0.0,
    )

    ego_speed_mps: float = Field(
        default=5.0,
        ge=0.0,
    )

    actors: list[
        SemanticActorSpecV2
    ] = Field(
        default_factory=list,
    )

    camera: CameraSpecV2 = Field(
        default_factory=CameraSpecV2,
    )

    @model_validator(mode="after")
    def validate_scenario(
        self,
    ) -> "SemanticScenarioV2":

        actor_ids = [
            actor.actor_id
            for actor in self.actors
        ]

        if len(actor_ids) != len(
            set(actor_ids)
        ):
            raise ValueError(
                "Semantic actor IDs must be unique."
            )

        if "ego" in actor_ids:
            raise ValueError(
                "Semantic actor_id 'ego' is reserved."
            )

        for actor in self.actors:

            if (
                actor.spawn_time_s
                > self.duration_s
            ):
                raise ValueError(
                    f"Actor {actor.actor_id!r} "
                    "spawns after scenario end."
                )

            if (
                actor.despawn_time_s is not None
                and actor.despawn_time_s
                > self.duration_s
            ):
                raise ValueError(
                    f"Actor {actor.actor_id!r} "
                    "despawns after scenario end."
                )

        return self


# ============================================================
# Compiler
# ============================================================

def compile_semantic_scenario(
    semantic: SemanticScenarioV2,
    resolved_hold_durations_s: Optional[
        dict[str, dict[int, float]]
    ] = None,
    resolved_cruise_durations_s: Optional[
        dict[str, dict[int, float]]
    ] = None,
) -> ScenarioSpecV2:
    """
    Compile a high-level scenario into ScenarioSpecV2.

    No CARLA or HE knowledge belongs here.
    """

    # --------------------------------------------------------
    # Reference/scripted ego
    # --------------------------------------------------------

    ego_behavior = SemanticBehaviorV2.model_validate({
        "initial_lane": "ego_lane",
        "initial_speed_mps":
            semantic.ego_speed_mps,

        "steps": [
            {
                "action": "cruise",
                "duration_s":
                    semantic.duration_s,

                "speed_mps":
                    semantic.ego_speed_mps,
            }
        ],

        "end_behavior": "continue",
    })

    ego_motion = compile_semantic_behavior(
        ego_behavior,
        lane_width_m=semantic.lane_width_m,
    )

    ego = EgoSpecV2(
        actor_id="ego",

        spawn=Pose2DV2(
            x_m=0.0,
            y_m=0.0,
            yaw_deg=0.0,
        ),

        motion=ego_motion,
    )

    # --------------------------------------------------------
    # Non-ego actors
    # --------------------------------------------------------

    compiled_actors = []

    for actor in semantic.actors:

        initial_y_m = (
            lane_relation_to_offset_m(
                actor.behavior.initial_lane,
                semantic.lane_width_m,
            )
        )

        actor_hold_durations = None
        actor_cruise_durations = None

        if resolved_cruise_durations_s is not None:
            actor_cruise_durations = (
                resolved_cruise_durations_s.get(
                    actor.actor_id,
                    {},
                )
            )
        if resolved_hold_durations_s is not None:
            actor_hold_durations = (
                resolved_hold_durations_s.get(
                    actor.actor_id,
                    {},
                )
            )

        motion = compile_semantic_behavior(
            actor.behavior,
            lane_width_m=semantic.lane_width_m,
            resolved_hold_durations_s=(
                actor_hold_durations
            ),
            resolved_cruise_durations_s=(
                actor_cruise_durations
            ),
        )

        compiled_actors.append(
            ActorSpecV2(
                actor_id=actor.actor_id,

                actor_type=actor.actor_type,

                role=actor.role,

                asset_key=actor.asset_key,

                dimensions_m=actor.dimensions_m,

                spawn=Pose2DV2(
                    x_m=(
                        actor.longitudinal_offset_m
                    ),
                    y_m=initial_y_m,
                    yaw_deg=0.0,
                ),

                lifecycle=ActorLifecycleV2(
                    spawn_time_s=(
                        actor.spawn_time_s
                    ),

                    despawn_time_s=(
                        actor.despawn_time_s
                    ),
                ),

                motion=motion,
            )
        )

    return ScenarioSpecV2(
        scenario_id=semantic.scenario_id,

        description=semantic.description,

        duration_s=semantic.duration_s,

        fps=semantic.fps,

        seed=semantic.seed,

        ego=ego,

        actors=compiled_actors,

        camera=semantic.camera,
    )