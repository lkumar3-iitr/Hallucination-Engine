from dataclasses import dataclass

from oriented_box_metrics_v1 import (
    nearest_actor_footprint_metrics,
    rectangle_clearance,
    rectangle_corners,
)


@dataclass
class Dimensions:
    length_m: float
    width_m: float


@dataclass
class Actor:
    actor_id: str
    world_x_m: float
    world_y_m: float
    world_yaw_deg: float
    physical_dimensions: Dimensions


def test_rectangle_clearance_and_overlap():
    first = rectangle_corners(0.0, 0.0, 0.0, 4.0, 2.0)
    separated = rectangle_corners(6.0, 0.0, 0.0, 4.0, 2.0)
    overlapping = rectangle_corners(3.0, 0.0, 0.0, 4.0, 2.0)

    assert rectangle_clearance(first, separated) == (2.0, False)
    assert rectangle_clearance(first, overlapping) == (0.0, True)


def test_nearest_actor_uses_footprint_clearance():
    actors = [
        Actor("far", 20.0, 0.0, 0.0, Dimensions(4.0, 2.0)),
        Actor("near", 7.0, 0.0, 0.0, Dimensions(4.0, 2.0)),
    ]
    result = nearest_actor_footprint_metrics(
        0.0, 0.0, 0.0, 4.0, 2.0, actors
    )

    assert result["actor_id"] == "near"
    assert result["clearance_m"] == 3.0
    assert result["overlaps"] is False
