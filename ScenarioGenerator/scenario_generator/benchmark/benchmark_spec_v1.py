"""
benchmark_spec_v1.py

Backend-independent specification for controlled Hallucination Engine
benchmark sweeps.

M4A responsibility:

    BenchmarkSpecV1
            ↓
    deterministic Cartesian expansion
            ↓
    BenchmarkCaseV1[]

This module intentionally knows nothing about:

    - CARLA
    - HE
    - placement lookup
    - image-space bounding boxes
    - rendering
    - trajectory resolution

Those belong to later stages.
"""

from __future__ import annotations

from itertools import product
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


# ============================================================
# Basic types
# ============================================================

ScenarioTypeV1 = Literal[
    "static",
    "oncoming",
    "following",
    "cut_in",
    "crossing",
]

SideV1 = Literal[
    "left",
    "right",
]


# ============================================================
# Base semantic parameters
# ============================================================

class BenchmarkRequestBaseV1(BaseModel):
    """
    Semantic parameters from which an individual scenario will
    eventually be generated.

    These names deliberately follow the semantic request layer we
    already developed.

    They are still simulator-independent.
    """

    side: SideV1 | None = None

    # Actor initial longitudinal distance from ego.
    start_distance_m: float = 30.0

    # Generic lateral coordinates.
    #
    # ScenarioGenerator convention:
    #
    #     +y = LEFT
    #     -y = RIGHT
    #
    lane_y_m: float = 0.0
    target_lane_y_m: float = 0.0

    lane_width_m: float = 3.5

    actor_speed_mps: float = 5.0
    ego_speed_mps: float = 5.0

    duration_s: float = 8.0
    fps: int = 30

    # Used by temporally constrained maneuvers such as cut-in.
    cut_start_s: float = 1.5
    cut_duration_s: float = 2.0

    @model_validator(mode="after")
    def validate_values(self):

        if self.start_distance_m < 0.0:
            raise ValueError(
                "start_distance_m must be >= 0."
            )

        if self.lane_width_m <= 0.0:
            raise ValueError(
                "lane_width_m must be > 0."
            )

        if self.actor_speed_mps < 0.0:
            raise ValueError(
                "actor_speed_mps must be >= 0."
            )

        if self.ego_speed_mps < 0.0:
            raise ValueError(
                "ego_speed_mps must be >= 0."
            )

        if self.duration_s <= 0.0:
            raise ValueError(
                "duration_s must be > 0."
            )

        if self.fps <= 0:
            raise ValueError(
                "fps must be > 0."
            )

        if self.cut_start_s < 0.0:
            raise ValueError(
                "cut_start_s must be >= 0."
            )

        if self.cut_duration_s <= 0.0:
            raise ValueError(
                "cut_duration_s must be > 0."
            )

        return self


# ============================================================
# Sweep factors
# ============================================================

class BenchmarkFactorsV1(BaseModel):
    """
    Optional values to sweep.

    Every non-None field participates in a Cartesian product.

    Example:

        start_distance_m = [20, 30]
        actor_speed_mps  = [4, 6]
        side             = ["left", "right"]

    produces:

        2 * 2 * 2 = 8 cases.
    """

    side: list[SideV1] | None = None

    start_distance_m: list[float] | None = None

    lane_y_m: list[float] | None = None

    target_lane_y_m: list[float] | None = None

    lane_width_m: list[float] | None = None

    actor_speed_mps: list[float] | None = None

    ego_speed_mps: list[float] | None = None

    duration_s: list[float] | None = None

    fps: list[int] | None = None

    cut_start_s: list[float] | None = None

    cut_duration_s: list[float] | None = None

    @model_validator(mode="after")
    def validate_factor_lists(self):

        data = self.model_dump()

        for name, values in data.items():

            if values is None:
                continue

            if len(values) == 0:
                raise ValueError(
                    f"Benchmark factor {name!r} "
                    "must not be an empty list."
                )

        return self


# ============================================================
# Benchmark specification
# ============================================================

class BenchmarkSpecV1(BaseModel):
    """
    Description of an experiment sweep.

    This says WHAT should vary, not HOW any backend executes it.
    """

    schema_version: Literal[
        "benchmark-1.0"
    ] = "benchmark-1.0"

    benchmark_id: str = Field(
        min_length=1
    )

    description: str = ""

    scenario_type: ScenarioTypeV1

    base: BenchmarkRequestBaseV1 = Field(
        default_factory=BenchmarkRequestBaseV1
    )

    factors: BenchmarkFactorsV1 = Field(
        default_factory=BenchmarkFactorsV1
    )

    @model_validator(mode="after")
    def validate_benchmark(self):

        # ----------------------------------------------------
        # cut_in and crossing currently have directional
        # semantics.
        #
        # A side can be supplied either as a constant base
        # parameter or as a swept factor.
        # ----------------------------------------------------

        if self.scenario_type in {
            "cut_in",
            "crossing",
        }:

            has_base_side = (
                self.base.side
                is not None
            )

            has_swept_side = (
                self.factors.side
                is not None
                and
                len(
                    self.factors.side
                ) > 0
            )

            if not (
                has_base_side
                or
                has_swept_side
            ):

                raise ValueError(
                    f"scenario_type="
                    f"{self.scenario_type!r} "
                    "requires side='left'/'right' "
                    "either in base or factors."
                )

        return self


# ============================================================
# Expanded case
# ============================================================

class BenchmarkCaseV1(BaseModel):
    """
    One concrete semantic benchmark case.

    After expansion there are no lists/ranges left.
    """

    schema_version: Literal[
        "benchmark-case-1.0"
    ] = "benchmark-case-1.0"

    benchmark_id: str

    case_index: int

    case_id: str

    scenario_type: ScenarioTypeV1

    parameters: BenchmarkRequestBaseV1

    swept_values: dict[
        str,
        str | int | float,
    ] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_case(self):

        p = self.parameters

        # ----------------------------------------------------
        # Directional scenarios
        # ----------------------------------------------------

        if self.scenario_type in {
            "cut_in",
            "crossing",
        }:

            if p.side is None:
                raise ValueError(
                    f"{self.case_id}: "
                    f"{self.scenario_type} "
                    "requires side."
                )

        # ----------------------------------------------------
        # Cut-in temporal consistency
        # ----------------------------------------------------

        if self.scenario_type == "cut_in":

            cut_end_s = (
                p.cut_start_s
                +
                p.cut_duration_s
            )

            if cut_end_s > p.duration_s:

                raise ValueError(
                    f"{self.case_id}: "
                    "cut maneuver ends after "
                    "scenario duration. "
                    f"cut_end={cut_end_s:.3f}s, "
                    f"duration={p.duration_s:.3f}s"
                )

        return self


# ============================================================
# Deterministic factor ordering
# ============================================================

FACTOR_ORDER = [
    "side",
    "start_distance_m",
    "lane_y_m",
    "target_lane_y_m",
    "lane_width_m",
    "actor_speed_mps",
    "ego_speed_mps",
    "duration_s",
    "fps",
    "cut_start_s",
    "cut_duration_s",
]


# ============================================================
# Expansion
# ============================================================

def expand_benchmark(
    benchmark: BenchmarkSpecV1,
) -> list[BenchmarkCaseV1]:
    """
    Deterministically expand BenchmarkSpecV1 into concrete cases.

    No trajectory or backend work occurs here.
    """

    factor_data = (
        benchmark
        .factors
        .model_dump()
    )

    active_factor_names = [

        name

        for name
        in FACTOR_ORDER

        if (
            factor_data.get(
                name
            )
            is not None
        )
    ]

    # --------------------------------------------------------
    # No factors -> benchmark still represents one case.
    # --------------------------------------------------------

    if not active_factor_names:

        combinations = [
            ()
        ]

    else:

        factor_value_lists = [

            factor_data[
                name
            ]

            for name
            in active_factor_names
        ]

        combinations = product(
            *factor_value_lists
        )

    cases = []

    base_parameters = (
        benchmark
        .base
        .model_dump()
    )

    for case_index, combination in enumerate(
        combinations,
        start=1,
    ):

        parameters_dict = dict(
            base_parameters
        )

        swept_values = {}

        for (
            factor_name,
            factor_value,
        ) in zip(
            active_factor_names,
            combination,
        ):

            parameters_dict[
                factor_name
            ] = factor_value

            swept_values[
                factor_name
            ] = factor_value

        case_id = (
            f"{benchmark.benchmark_id}"
            f"_case_{case_index:04d}"
        )

        parameters = (
            BenchmarkRequestBaseV1
            .model_validate(
                parameters_dict
            )
        )

        case = (
            BenchmarkCaseV1(
                benchmark_id=
                    benchmark.benchmark_id,

                case_index=
                    case_index,

                case_id=
                    case_id,

                scenario_type=
                    benchmark.scenario_type,

                parameters=
                    parameters,

                swept_values=
                    swept_values,
            )
        )

        cases.append(
            case
        )

    return cases