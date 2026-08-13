from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from scenario_generator.schema import ScenarioSpec

from scenario_generator.planner.scenario_factory import (
    make_scenario_from_params,
)


ScenarioType = Literal[
    "static",
    "oncoming",
    "following",
    "cut_in",
    "crossing",
]

SideType = Literal[
    "left",
    "right",
]


class ScenarioRequest(BaseModel):
    """
    Backend-independent semantic request for one driving scenario.

    This is intentionally higher-level than ScenarioSpec.

    Current deterministic sources:
        - command-line parameters
        - JSON request

    Future source:
        - natural-language / LLM planner

    Coordinate convention:
        +x = forward
        +y = left

    Speeds are positive magnitudes. Maneuver type determines
    movement direction where appropriate (for example oncoming).
    """

    scenario_type: ScenarioType

    scenario_id: Optional[str] = None

    # --------------------------------------------------------
    # Road / lateral semantics
    # --------------------------------------------------------

    side: Optional[SideType] = None

    start_distance_m: float = Field(
        default=30.0,
        gt=0.0,
    )

    # Used directly for static/oncoming/following.
    # cut_in/crossing generally use semantic `side`.
    lane_y_m: float = 0.0

    target_lane_y_m: float = 0.0

    lane_width_m: float = Field(
        default=3.5,
        gt=0.0,
    )

    # --------------------------------------------------------
    # Speeds
    # --------------------------------------------------------

    actor_speed_mps: float = Field(
        default=4.0,
        ge=0.0,
    )

    ego_speed_mps: float = Field(
        default=5.0,
        ge=0.0,
    )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    duration_s: float = Field(
        default=8.0,
        gt=0.0,
    )

    fps: int = Field(
        default=30,
        gt=0,
    )

    # Cut-in timing.
    cut_start_s: float = Field(
        default=1.5,
        ge=0.0,
    )

    cut_duration_s: float = Field(
        default=3.0,
        gt=0.0,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    @model_validator(mode="after")
    def validate_semantics(self) -> "ScenarioRequest":

        if (
            self.scenario_type
            in ("cut_in", "crossing")
            and self.side is None
        ):
            raise ValueError(
                f"side is required for "
                f"scenario_type={self.scenario_type}"
            )

        if self.scenario_type == "cut_in":
            cut_end_s = (
                self.cut_start_s
                + self.cut_duration_s
            )

            if cut_end_s > self.duration_s:
                raise ValueError(
                    "cut_start_s + cut_duration_s "
                    "must not exceed duration_s"
                )

        return self


def scenario_request_to_spec(
    request: ScenarioRequest,
) -> ScenarioSpec:
    """
    Convert semantic request into the existing ScenarioSpec.

    Backend coordinate conversions must NOT happen here.
    """

    return make_scenario_from_params(
        scenario_type=request.scenario_type,

        side=request.side,

        start_distance_m=(
            request.start_distance_m
        ),

        lane_y_m=(
            request.lane_y_m
        ),

        target_lane_y_m=(
            request.target_lane_y_m
        ),

        speed_mps=(
            request.actor_speed_mps
        ),

        duration_s=(
            request.duration_s
        ),

        fps=(
            request.fps
        ),

        ego_speed_mps=(
            request.ego_speed_mps
        ),

        lane_width_m=(
            request.lane_width_m
        ),

        cut_start_s=(
            request.cut_start_s
        ),

        cut_duration_s=(
            request.cut_duration_s
        ),

        scenario_id=(
            request.scenario_id
        ),
    )