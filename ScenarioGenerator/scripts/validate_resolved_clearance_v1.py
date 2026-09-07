from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
COMMON_DIR = REPO_ROOT / "driving_models" / "common"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(COMMON_DIR))

from he_asset_registry_v1 import HEAssetRegistry
from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedScenarioV2,
)


def rectangle_corners(x_m, y_m, yaw_deg, length_m, width_m):
    yaw = math.radians(float(yaw_deg))
    forward = (math.cos(yaw), math.sin(yaw))
    left = (-math.sin(yaw), math.cos(yaw))
    half_length = float(length_m) / 2.0
    half_width = float(width_m) / 2.0

    return [
        (
            float(x_m) + sx * half_length * forward[0]
            + sy * half_width * left[0],
            float(y_m) + sx * half_length * forward[1]
            + sy * half_width * left[1],
        )
        for sx, sy in (
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
        )
    ]


def rectangle_axes(corners):
    axes = []
    for index in (0, 1):
        x1, y1 = corners[index]
        x2, y2 = corners[(index + 1) % 4]
        edge_x = x2 - x1
        edge_y = y2 - y1
        norm = math.hypot(edge_x, edge_y)
        axes.append((-edge_y / norm, edge_x / norm))
    return axes


def projection_interval(corners, axis):
    values = [
        point[0] * axis[0] + point[1] * axis[1]
        for point in corners
    ]
    return min(values), max(values)


def rectangles_overlap(first, second, tolerance=1e-9):
    for axis in rectangle_axes(first) + rectangle_axes(second):
        first_min, first_max = projection_interval(first, axis)
        second_min, second_max = projection_interval(second, axis)
        if (
            first_max < second_min - tolerance
            or second_max < first_min - tolerance
        ):
            return False
    return True


def point_segment_distance(point, start, end):
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared <= 1e-18:
        return math.hypot(
            point[0] - start[0],
            point[1] - start[1],
        )

    projection = (
        (point[0] - start[0]) * segment_x
        + (point[1] - start[1]) * segment_y
    ) / length_squared
    projection = max(0.0, min(1.0, projection))
    nearest = (
        start[0] + projection * segment_x,
        start[1] + projection * segment_y,
    )
    return math.hypot(
        point[0] - nearest[0],
        point[1] - nearest[1],
    )


def rectangle_clearance(first, second):
    if rectangles_overlap(first, second):
        return 0.0, True

    distances = []
    first_edges = [
        (first[index], first[(index + 1) % 4])
        for index in range(4)
    ]
    second_edges = [
        (second[index], second[(index + 1) % 4])
        for index in range(4)
    ]

    for point in first:
        distances.extend(
            point_segment_distance(point, start, end)
            for start, end in second_edges
        )
    for point in second:
        distances.extend(
            point_segment_distance(point, start, end)
            for start, end in first_edges
        )

    return min(distances), False


def physical_dimensions(actor_info, registry):
    if actor_info.dimensions_m is not None:
        return {
            "length_m": float(actor_info.dimensions_m.length_m),
            "width_m": float(actor_info.dimensions_m.width_m),
            "height_m": float(actor_info.dimensions_m.height_m),
            "source": "resolved_scenario",
        }

    asset = registry.resolve(actor_info.asset_key)
    if asset.physical_bbox is None:
        raise ValueError(
            f"Asset {actor_info.asset_key!r} has no physical bbox."
        )
    return {
        "length_m": float(asset.physical_bbox.length_m),
        "width_m": float(asset.physical_bbox.width_m),
        "height_m": float(asset.physical_bbox.height_m),
        "source": "asset_metadata",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate collision-free physical rectangle clearance in a "
            "ResolvedScenarioV2 reference trajectory."
        )
    )
    parser.add_argument("resolved_json")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--actor-id", default=None)
    parser.add_argument(
        "--ego-asset-key",
        default="vehicle.passenger_01",
    )
    parser.add_argument(
        "--minimum-terminal-clearance-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-clearance-m",
        type=float,
        default=0.0,
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    resolved_path = Path(args.resolved_json)
    with resolved_path.open("r", encoding="utf-8") as handle:
        scenario = ResolvedScenarioV2.model_validate(
            json.load(handle)
        )

    if args.actor_id is None:
        if len(scenario.actors) != 1:
            raise ValueError(
                "--actor-id is required for multi-actor scenarios."
            )
        actor_info = scenario.actors[0]
    else:
        matches = [
            actor
            for actor in scenario.actors
            if actor.actor_id == args.actor_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Actor {args.actor_id!r} not found exactly once."
            )
        actor_info = matches[0]

    registry = HEAssetRegistry(args.asset_root)
    actor_dimensions = physical_dimensions(actor_info, registry)
    ego_asset = registry.resolve(args.ego_asset_key)
    if ego_asset.physical_bbox is None:
        raise ValueError(
            f"Ego asset {args.ego_asset_key!r} has no physical bbox."
        )
    ego_dimensions = {
        "length_m": float(ego_asset.physical_bbox.length_m),
        "width_m": float(ego_asset.physical_bbox.width_m),
        "height_m": float(ego_asset.physical_bbox.height_m),
        "source": "asset_metadata",
    }

    ego_frames = {
        frame.frame_idx: frame
        for frame in scenario.ego_frames
    }
    actor_frames = {
        frame.frame_idx: frame
        for frame in scenario.actor_frames
        if frame.actor_id == actor_info.actor_id
    }
    common_frames = sorted(
        set(ego_frames) & set(actor_frames)
    )
    if not common_frames:
        raise ValueError("No common ego/actor frames to validate.")

    samples = []
    collision_frames = []
    for frame_idx in common_frames:
        ego = ego_frames[frame_idx]
        actor = actor_frames[frame_idx]
        ego_rectangle = rectangle_corners(
            ego.x_m,
            ego.y_m,
            ego.yaw_deg,
            ego_dimensions["length_m"],
            ego_dimensions["width_m"],
        )
        actor_rectangle = rectangle_corners(
            actor.x_m,
            actor.y_m,
            actor.yaw_deg,
            actor_dimensions["length_m"],
            actor_dimensions["width_m"],
        )
        clearance_m, overlaps = rectangle_clearance(
            ego_rectangle,
            actor_rectangle,
        )
        sample = {
            "frame_idx": int(frame_idx),
            "t_s": float(ego.t_s),
            "clearance_m": float(clearance_m),
            "overlaps": bool(overlaps),
            "center_distance_m": float(
                math.hypot(
                    actor.x_m - ego.x_m,
                    actor.y_m - ego.y_m,
                )
            ),
        }
        samples.append(sample)
        if overlaps:
            collision_frames.append(int(frame_idx))

    minimum_sample = min(
        samples,
        key=lambda sample: sample["clearance_m"],
    )
    terminal_sample = samples[-1]
    terminal_threshold = float(
        args.minimum_terminal_clearance_m
    )
    global_threshold = float(
        args.minimum_clearance_m
    )
    passed = (
        not collision_frames
        and minimum_sample["clearance_m"] + 1e-9 >= global_threshold
        and terminal_sample["clearance_m"] + 1e-9 >= terminal_threshold
    )

    report = {
        "schema": "resolved_clearance_validation_v1",
        "status": "PASS" if passed else "FAIL",
        "scenario_id": scenario.scenario_id,
        "resolved_path": str(resolved_path.resolve()),
        "actor_id": actor_info.actor_id,
        "ego_asset_key": args.ego_asset_key,
        "actor_dimensions": actor_dimensions,
        "ego_dimensions": ego_dimensions,
        "common_frames": len(common_frames),
        "minimum_clearance": minimum_sample,
        "terminal_clearance": terminal_sample,
        "minimum_clearance_required_m": global_threshold,
        "minimum_terminal_clearance_required_m": terminal_threshold,
        "collision_frame_count": len(collision_frames),
        "first_collision_frame": (
            collision_frames[0]
            if collision_frames
            else None
        ),
    }

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        report["output"] = str(output_path.resolve())

    print(json.dumps(report, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
