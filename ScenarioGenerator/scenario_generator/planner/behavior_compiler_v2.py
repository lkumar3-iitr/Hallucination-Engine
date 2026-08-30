from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from scenario_generator.schema.scenario_schema_v2 import (
    SequenceMotionV2,
)
from scenario_generator.planner.event_conditions_v2 import (
    EventConditionV2,
)

# ============================================================
# Semantic lane relations
# ============================================================

LaneRelationV2 = Literal[
    "ego_lane",
    "left_adjacent",
    "right_adjacent",
]


# ============================================================
# Semantic behavior steps
# ============================================================

class SemanticCruiseStepV2(BaseModel):
    action: Literal["cruise"] = "cruise"

    duration_s: float = Field(
        gt=0.0,
    )

    speed_mps: Optional[float] = Field(
        default=None,
        ge=0.0,
    )
class SemanticCruiseUntilStepV2(BaseModel):
    """
    Cruise at the current or requested speed until an offline
    scenario condition becomes true.
    """

    action: Literal["cruise_until"] = "cruise_until"

    condition: EventConditionV2

    speed_mps: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    min_duration_s: float = Field(
        default=0.0,
        ge=0.0,
    )

class SemanticLaneChangeStepV2(BaseModel):
    action: Literal["lane_change"] = "lane_change"

    target_lane: LaneRelationV2

    duration_s: float = Field(
        gt=0.0,
    )


class SemanticBrakeToStepV2(BaseModel):
    action: Literal["brake_to"] = "brake_to"

    target_speed_mps: float = Field(
        default=0.0,
        ge=0.0,
    )

    deceleration_mps2: float = Field(
        gt=0.0,
    )


class SemanticAccelerateToStepV2(BaseModel):
    action: Literal["accelerate_to"] = "accelerate_to"

    target_speed_mps: float = Field(
        ge=0.0,
    )

    acceleration_mps2: float = Field(
        gt=0.0,
    )


class SemanticHoldStepV2(BaseModel):
    action: Literal["hold"] = "hold"

    duration_s: float = Field(
        gt=0.0,
    )
class SemanticHoldUntilStepV2(BaseModel):
    """
    Hold the actor stationary until an offline-resolved
    scenario condition becomes true.
    """

    action: Literal["hold_until"] = "hold_until"

    condition: EventConditionV2

    min_duration_s: float = Field(
        default=0.0,
        ge=0.0,
    )

SemanticBehaviorStepV2 = Annotated[
    Union[
        SemanticCruiseStepV2,
        SemanticCruiseUntilStepV2,
        SemanticLaneChangeStepV2,
        SemanticBrakeToStepV2,
        SemanticAccelerateToStepV2,
        SemanticHoldStepV2,
        SemanticHoldUntilStepV2,
    ],
    Field(discriminator="action"),
]


class SemanticBehaviorV2(BaseModel):
    """
    Human/LLM-facing description of one actor's behavior.

    It intentionally contains semantic lane relations rather than
    ScenarioGenerator lateral coordinates.
    """

    initial_lane: LaneRelationV2 = "ego_lane"

    initial_speed_mps: float = Field(
        default=0.0,
        ge=0.0,
    )

    steps: list[SemanticBehaviorStepV2] = Field(
        min_length=1,
    )

    end_behavior: Literal[
        "hold",
        "continue",
    ] = "hold"


# ============================================================
# Compiler
# ============================================================

def lane_relation_to_offset_m(
    lane: LaneRelationV2,
    lane_width_m: float,
) -> float:

    if lane == "ego_lane":
        return 0.0

    if lane == "left_adjacent":
        return lane_width_m

    if lane == "right_adjacent":
        return -lane_width_m

    raise ValueError(
        f"Unsupported lane relation: {lane}"
    )


def compile_semantic_behavior(
    behavior: SemanticBehaviorV2,
    lane_width_m: float,
    resolved_hold_durations_s: Optional[
        dict[int, float]
    ] = None,
    resolved_cruise_durations_s: Optional[
        dict[int, float]
    ] = None,
) -> SequenceMotionV2:
    """
    Compile semantic actor behavior into deterministic
    SequenceMotionV2.

    The compiler tracks the actor's current semantic lane so a lane
    change becomes a relative physical lateral displacement.
    """

    if lane_width_m <= 0.0:
        raise ValueError(
            "lane_width_m must be > 0."
        )

    current_lane_offset = (
        lane_relation_to_offset_m(
            behavior.initial_lane,
            lane_width_m,
        )
    )

    compiled_steps = []

    for step_index, step in enumerate(
        behavior.steps
    ):

        if isinstance(
            step,
            SemanticCruiseStepV2,
        ):
            compiled_steps.append({
                "step": "cruise",
                "duration_s": step.duration_s,
                "speed_mps": step.speed_mps,
            })

            continue

        if isinstance(
            step,
            SemanticCruiseUntilStepV2,
        ):
            if (
                resolved_cruise_durations_s is None
                or step_index
                not in resolved_cruise_durations_s
            ):
                raise ValueError(
                    "cruise_until step "
                    f"{step_index} has not been "
                    "resolved to a deterministic "
                    "duration yet."
                )

            duration_s = max(
                step.min_duration_s,
                resolved_cruise_durations_s[
                    step_index
                ],
            )

            if duration_s > 1e-9:
                compiled_steps.append({
                    "step": "cruise",
                    "duration_s": duration_s,
                    "speed_mps": step.speed_mps,
                })

            continue

        if isinstance(
            step,
            SemanticHoldStepV2,
        ):
            compiled_steps.append({
                "step": "hold",
                "duration_s": step.duration_s,
            })

            continue

        if isinstance(
            step,
            SemanticHoldUntilStepV2,
        ):
            if (
                resolved_hold_durations_s is None
                or step_index
                not in resolved_hold_durations_s
            ):
                raise ValueError(
                    "hold_until step "
                    f"{step_index} has not been "
                    "resolved to a deterministic "
                    "duration yet."
                )

            duration_s = max(
                step.min_duration_s,
                resolved_hold_durations_s[
                    step_index
                ],
            )

            if duration_s > 1e-9:
                compiled_steps.append({
                    "step": "hold",
                    "duration_s": duration_s,
                })

            continue

        if isinstance(
            step,
            SemanticBrakeToStepV2,
        ):
            compiled_steps.append({
                "step": "brake",
                "target_speed_mps":
                    step.target_speed_mps,
                "deceleration_mps2":
                    step.deceleration_mps2,
            })

            continue

        if isinstance(
            step,
            SemanticAccelerateToStepV2,
        ):
            compiled_steps.append({
                "step": "accelerate",
                "target_speed_mps":
                    step.target_speed_mps,
                "acceleration_mps2":
                    step.acceleration_mps2,
            })

            continue

        if isinstance(
            step,
            SemanticLaneChangeStepV2,
        ):
            target_lane_offset = (
                lane_relation_to_offset_m(
                    step.target_lane,
                    lane_width_m,
                )
            )

            lateral_delta = (
                target_lane_offset
                - current_lane_offset
            )

            compiled_steps.append({
                "step": "lane_change",
                "lateral_delta_m":
                    lateral_delta,
                "duration_s":
                    step.duration_s,
            })

            current_lane_offset = (
                target_lane_offset
            )

            continue

        raise TypeError(
            f"Unknown semantic step: {type(step)}"
        )

    return SequenceMotionV2.model_validate({
        "mode": "sequence",

        "initial_speed_mps":
            behavior.initial_speed_mps,

        "steps":
            compiled_steps,

        "end_behavior":
            behavior.end_behavior,
    })