from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedScenarioV2,
)


# ============================================================
# Event conditions
# ============================================================

class TimeConditionV2(BaseModel):
    kind: Literal["time"] = "time"

    t_s: float = Field(
        ge=0.0,
    )


class ActorStoppedConditionV2(BaseModel):
    kind: Literal["actor_stopped"] = "actor_stopped"

    actor_id: str = Field(
        min_length=1,
    )

    speed_threshold_mps: float = Field(
        default=0.05,
        ge=0.0,
    )

    min_hold_s: float = Field(
        default=0.0,
        ge=0.0,
    )


class RelativeLongitudinalConditionV2(BaseModel):
    """
    Relative position along ScenarioGenerator +x.

    Examples:

        ego passed actor:
            subject_id = "ego"
            reference_id = "car"
            relation = "ahead_by"

        car remains behind ego:
            subject_id = "car"
            reference_id = "ego"
            relation = "behind_by"
    """

    kind: Literal[
        "relative_longitudinal"
    ] = "relative_longitudinal"

    subject_id: str = Field(
        min_length=1,
    )

    reference_id: str = Field(
        min_length=1,
    )

    relation: Literal[
        "ahead_by",
        "behind_by",
    ]

    margin_m: float = Field(
        default=0.0,
        ge=0.0,
    )


class DistanceBelowConditionV2(BaseModel):
    kind: Literal[
        "distance_below"
    ] = "distance_below"

    actor_a_id: str = Field(
        min_length=1,
    )

    actor_b_id: str = Field(
        min_length=1,
    )

    distance_m: float = Field(
        gt=0.0,
    )


EventConditionV2 = Annotated[
    Union[
        TimeConditionV2,
        ActorStoppedConditionV2,
        RelativeLongitudinalConditionV2,
        DistanceBelowConditionV2,
    ],
    Field(discriminator="kind"),
]


# ============================================================
# Resolved-state helpers
# ============================================================

def _build_state_lookup(
    resolved: ResolvedScenarioV2,
):
    ego_by_frame = {
        frame.frame_idx: frame
        for frame in resolved.ego_frames
    }

    actor_by_id_and_frame = {}

    for frame in resolved.actor_frames:

        actor_by_id_and_frame[
            (
                frame.actor_id,
                frame.frame_idx,
            )
        ] = frame

    return (
        ego_by_frame,
        actor_by_id_and_frame,
    )


def _state_at(
    actor_id: str,
    frame_idx: int,
    ego_by_frame,
    actor_by_id_and_frame,
):

    if actor_id == "ego":
        return ego_by_frame.get(
            frame_idx
        )

    return actor_by_id_and_frame.get(
        (
            actor_id,
            frame_idx,
        )
    )


# ============================================================
# Condition resolver
# ============================================================

def resolve_condition_time(
    resolved: ResolvedScenarioV2,
    condition: EventConditionV2,
    not_before_s: float = 0.0,
) -> float:
    """
    Resolve one semantic event condition against an already resolved
    canonical physical scenario.

    This function is OFFLINE.

    CARLA and HE must never independently evaluate these conditions.
    They receive only the final resolved trajectory.
    """

    if not_before_s < 0.0:
        raise ValueError(
            "not_before_s must be >= 0."
        )

    (
        ego_by_frame,
        actor_by_id_and_frame,
    ) = _build_state_lookup(
        resolved
    )

    # --------------------------------------------------------
    # Explicit time
    # --------------------------------------------------------

    if isinstance(
        condition,
        TimeConditionV2,
    ):
        target = max(
            condition.t_s,
            not_before_s,
        )

        for ego_frame in resolved.ego_frames:

            if ego_frame.t_s >= target:
                return ego_frame.t_s

        raise ValueError(
            "Time condition occurs after "
            "scenario end."
        )

    # --------------------------------------------------------
    # Actor stopped
    # --------------------------------------------------------

    if isinstance(
        condition,
        ActorStoppedConditionV2,
    ):

        required_frames = max(
            1,
            int(
                math.ceil(
                    condition.min_hold_s
                    * resolved.fps
                )
            ),
        )

        stopped_run = []

        for ego_frame in resolved.ego_frames:

            if ego_frame.t_s < not_before_s:
                continue

            state = _state_at(
                condition.actor_id,
                ego_frame.frame_idx,
                ego_by_frame,
                actor_by_id_and_frame,
            )

            if state is None:
                stopped_run = []
                continue

            if (
                state.speed_mps
                <= condition.speed_threshold_mps
            ):
                stopped_run.append(
                    state
                )

                if (
                    len(stopped_run)
                    >= required_frames
                ):
                    first = stopped_run[
                        -required_frames
                    ]

                    return first.t_s

            else:
                stopped_run = []

        raise ValueError(
            f"Actor {condition.actor_id!r} "
            "never satisfied actor_stopped."
        )

    # --------------------------------------------------------
    # Relative longitudinal condition
    # --------------------------------------------------------

    if isinstance(
        condition,
        RelativeLongitudinalConditionV2,
    ):

        for ego_frame in resolved.ego_frames:

            if ego_frame.t_s < not_before_s:
                continue

            subject = _state_at(
                condition.subject_id,
                ego_frame.frame_idx,
                ego_by_frame,
                actor_by_id_and_frame,
            )

            reference = _state_at(
                condition.reference_id,
                ego_frame.frame_idx,
                ego_by_frame,
                actor_by_id_and_frame,
            )

            if (
                subject is None
                or reference is None
            ):
                continue

            delta_x = (
                subject.x_m
                - reference.x_m
            )

            if (
                condition.relation
                == "ahead_by"
            ):
                satisfied = (
                    delta_x
                    >= condition.margin_m
                )

            else:
                satisfied = (
                    delta_x
                    <= -condition.margin_m
                )

            if satisfied:
                return ego_frame.t_s

        raise ValueError(
            "Relative longitudinal condition "
            "was never satisfied."
        )

    # --------------------------------------------------------
    # Euclidean distance condition
    # --------------------------------------------------------

    if isinstance(
        condition,
        DistanceBelowConditionV2,
    ):

        for ego_frame in resolved.ego_frames:

            if ego_frame.t_s < not_before_s:
                continue

            a = _state_at(
                condition.actor_a_id,
                ego_frame.frame_idx,
                ego_by_frame,
                actor_by_id_and_frame,
            )

            b = _state_at(
                condition.actor_b_id,
                ego_frame.frame_idx,
                ego_by_frame,
                actor_by_id_and_frame,
            )

            if a is None or b is None:
                continue

            distance = math.hypot(
                a.x_m - b.x_m,
                a.y_m - b.y_m,
            )

            if (
                distance
                <= condition.distance_m
            ):
                return ego_frame.t_s

        raise ValueError(
            "Distance condition was never satisfied."
        )

    raise TypeError(
        f"Unknown condition type: "
        f"{type(condition)}"
    )