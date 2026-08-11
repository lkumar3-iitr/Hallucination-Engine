from __future__ import annotations

from dataclasses import dataclass
from typing import List

from scenario_generator.schema import RoadConfig


@dataclass
class LaneCenter:
    lane_id: str
    direction: str  # "same" or "opposite"
    y_m: float


@dataclass
class BEVLocalMap:
    lane_width_m: float
    road_length_m: float
    lane_centers: List[LaneCenter]


def build_bev_local_map(road: RoadConfig) -> BEVLocalMap:
    """Create a simple straight-road BEV map.

    v1 assumptions:
    - Straight road only.
    - Ego lane center is y=0.
    - Same-direction lanes are positive y for left lanes.
    - Opposite-direction lane closest to ego is at y=-lane_width.
    """

    lane_centers: list[LaneCenter] = []

    for i in range(road.num_lanes_same_direction):
        lane_centers.append(
            LaneCenter(
                lane_id=f"same_{i}",
                direction="same",
                y_m=i * road.lane_width_m,
            )
        )

    for i in range(road.num_lanes_opposite_direction):
        lane_centers.append(
            LaneCenter(
                lane_id=f"opposite_{i}",
                direction="opposite",
                y_m=-(i + 1) * road.lane_width_m,
            )
        )

    return BEVLocalMap(
        lane_width_m=road.lane_width_m,
        road_length_m=road.road_length_m,
        lane_centers=lane_centers,
    )
