"""Condition-neutral 2D oriented-footprint safety metrics."""

from __future__ import annotations

import math


def rectangle_corners(x_m, y_m, yaw_deg, length_m, width_m):
    yaw = math.radians(float(yaw_deg))
    forward = (math.cos(yaw), math.sin(yaw))
    left = (-forward[1], forward[0])
    half_length = 0.5 * float(length_m)
    half_width = 0.5 * float(width_m)
    return [
        (
            float(x_m) + sx * half_length * forward[0]
            + sy * half_width * left[0],
            float(y_m) + sx * half_length * forward[1]
            + sy * half_width * left[1],
        )
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))
    ]


def _axes(corners):
    axes = []
    for index in (0, 1):
        start = corners[index]
        end = corners[(index + 1) % 4]
        edge_x = end[0] - start[0]
        edge_y = end[1] - start[1]
        norm = math.hypot(edge_x, edge_y)
        axes.append((-edge_y / norm, edge_x / norm))
    return axes


def _interval(corners, axis):
    values = [x * axis[0] + y * axis[1] for x, y in corners]
    return min(values), max(values)


def rectangles_overlap(first, second, tolerance=1e-9):
    for axis in _axes(first) + _axes(second):
        first_min, first_max = _interval(first, axis)
        second_min, second_max = _interval(second, axis)
        if first_max < second_min - tolerance or second_max < first_min - tolerance:
            return False
    return True


def _point_segment_distance(point, start, end):
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared <= 1e-18:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * segment_x
        + (point[1] - start[1]) * segment_y
    ) / length_squared
    projection = max(0.0, min(1.0, projection))
    nearest = (
        start[0] + projection * segment_x,
        start[1] + projection * segment_y,
    )
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def rectangle_clearance(first, second):
    if rectangles_overlap(first, second):
        return 0.0, True
    first_edges = [
        (first[index], first[(index + 1) % 4]) for index in range(4)
    ]
    second_edges = [
        (second[index], second[(index + 1) % 4]) for index in range(4)
    ]
    distances = []
    for point in first:
        distances.extend(
            _point_segment_distance(point, start, end)
            for start, end in second_edges
        )
    for point in second:
        distances.extend(
            _point_segment_distance(point, start, end)
            for start, end in first_edges
        )
    return min(distances), False


def nearest_actor_footprint_metrics(
    ego_x_m,
    ego_y_m,
    ego_yaw_deg,
    ego_length_m,
    ego_width_m,
    actors,
):
    ego = rectangle_corners(
        ego_x_m, ego_y_m, ego_yaw_deg, ego_length_m, ego_width_m
    )
    nearest = None
    for actor in actors:
        dimensions = actor.physical_dimensions
        if dimensions is None:
            continue
        footprint = rectangle_corners(
            actor.world_x_m,
            actor.world_y_m,
            actor.world_yaw_deg,
            dimensions.length_m,
            dimensions.width_m,
        )
        clearance_m, overlaps = rectangle_clearance(ego, footprint)
        sample = {
            "actor_id": str(actor.actor_id),
            "center_distance_m": math.hypot(
                float(actor.world_x_m) - float(ego_x_m),
                float(actor.world_y_m) - float(ego_y_m),
            ),
            "clearance_m": float(clearance_m),
            "overlaps": bool(overlaps),
        }
        if nearest is None or (
            sample["clearance_m"], sample["center_distance_m"], sample["actor_id"]
        ) < (
            nearest["clearance_m"],
            nearest["center_distance_m"],
            nearest["actor_id"],
        ):
            nearest = sample
    return nearest
