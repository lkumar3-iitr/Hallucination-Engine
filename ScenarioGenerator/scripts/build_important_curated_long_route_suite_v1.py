from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMMON_DIR = REPO_ROOT / "driving_models" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from scenario_generator.schema.scenario_schema_v2 import ScenarioSpecV2
from scenario_generator.trajectory.trajectory_resolver_v2 import V2TrajectoryResolver
from oriented_box_metrics_v1 import rectangle_clearance, rectangle_corners


OUTPUT = REPO_ROOT / "important_curated_scenarios"
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_resolved_all_pairs_clearance_v1.py"
DEFAULT_ASSET_ROOT = Path(
    r"D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production"
)

FPS = 20
DURATION_S = 60.0
DT = 1.0 / FPS
KEYFRAME_FPS = 5
KEYFRAME_DT = 1.0 / KEYFRAME_FPS
LANE = 3.5

TESLA = "vehicle.tesla.model3"
PATROL = "vehicle.passenger_02"
BUS = "vehicle.bus_01"
PED = "pedestrian.person_01"

DIMENSIONS = {
    TESLA: {"length_m": 4.792, "width_m": 2.163, "height_m": 1.488},
    PATROL: {"length_m": 5.566, "width_m": 2.150, "height_m": 2.045},
    BUS: {"length_m": 10.273, "width_m": 3.944, "height_m": 4.253},
    PED: {"length_m": 0.375, "width_m": 0.375, "height_m": 1.860},
}

CAMERA = {
    "image_width": 1280,
    "image_height": 720,
    "fov_deg": 90.0,
    "x_m": 1.5,
    "y_m": 0.0,
    "z_m": 1.6,
    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rel_to_sg(path: Path) -> str:
    return os.path.relpath(path.resolve(), PROJECT_ROOT.resolve())


def minjerk(value: float) -> float:
    u = max(0.0, min(1.0, float(value)))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def frames_from_xy(position, stationary_yaw=0.0):
    frames = []
    for idx in range(int(round(DURATION_S * KEYFRAME_FPS)) + 1):
        t_s = idx * KEYFRAME_DT
        x_m, y_m = position(t_s)
        before = max(0.0, t_s - KEYFRAME_DT)
        after = min(DURATION_S, t_s + KEYFRAME_DT)
        bx, by = position(before)
        ax, ay = position(after)
        yaw = stationary_yaw
        if math.hypot(ax - bx, ay - by) > 1e-9:
            yaw = math.degrees(math.atan2(ay - by, ax - bx))
        frames.append({
            "t_s": round(t_s, 8),
            "x_m": round(float(x_m), 8),
            "y_m": round(float(y_m), 8),
            "yaw_deg": round(float(yaw), 8),
        })
    return frames


def stationary(x, y, yaw=0.0):
    return frames_from_xy(lambda _t: (x, y), yaw)


def straight(x0, speed, y):
    return frames_from_xy(lambda t: (x0 + speed * t, y), 0.0)


def lead_brake_change_lane_stop():
    def position(t):
        if t <= 8.0:
            x = 42.0 + 5.2 * t
        elif t <= 12.0:
            u = minjerk((t - 8.0) / 4.0)
            # Integrated smooth deceleration from 5.2 m/s to 0.
            x = 83.6 + 5.2 * (t - 8.0) * (1.0 - 0.5 * u)
        else:
            x = 94.0
        y = LANE * minjerk((t - 16.0) / 5.0)
        return x, y

    return frames_from_xy(position, 0.0)


def crossing(x, y0, y1, start_s, end_s):
    yaw = 90.0 if y1 > y0 else -90.0
    return frames_from_xy(
        lambda t: (x, y0 + (y1 - y0) * minjerk((t - start_s) / (end_s - start_s))),
        yaw,
    )


def actor(actor_id, asset_key, keyframes, role="traffic"):
    first = keyframes[0]
    actor_type = "pedestrian" if asset_key == PED else "vehicle"
    return {
        "actor_id": actor_id,
        "actor_type": actor_type,
        "role": role,
        "asset_key": asset_key,
        "dimensions_m": DIMENSIONS[asset_key],
        "spawn": {
            "x_m": first["x_m"],
            "y_m": first["y_m"],
            "yaw_deg": first["yaw_deg"],
        },
        "lifecycle": {"spawn_time_s": 0.0, "despawn_time_s": None},
        "motion": {
            "mode": "keyframes",
            "interpolation": "linear",
            "keyframes": keyframes,
        },
    }


def validate_all_actor_pairs(resolved, output_path: Path) -> dict:
    dimensions = {actor.actor_id: actor.dimensions_m for actor in resolved.actors}
    actor_frames = {}
    for frame in resolved.actor_frames:
        actor_frames.setdefault(frame.frame_idx, {})[frame.actor_id] = frame

    minimum = None
    failures = []
    overlap_frames = []
    frame_step = max(1, int(round(FPS / KEYFRAME_FPS)))
    for frame_idx in sorted(actor_frames):
        if frame_idx % frame_step != 0:
            continue
        present = actor_frames[frame_idx]
        for first_id, second_id in combinations(sorted(present), 2):
            first = present[first_id]
            second = present[second_id]
            first_dims = dimensions[first_id]
            second_dims = dimensions[second_id]
            clearance_m, overlaps = rectangle_clearance(
                rectangle_corners(
                    first.x_m,
                    first.y_m,
                    first.yaw_deg,
                    first_dims.length_m,
                    first_dims.width_m,
                ),
                rectangle_corners(
                    second.x_m,
                    second.y_m,
                    second.yaw_deg,
                    second_dims.length_m,
                    second_dims.width_m,
                ),
            )
            sample = {
                "frame_idx": int(frame_idx),
                "t_s": float(getattr(first, "t_s", frame_idx / resolved.fps)),
                "pair": f"{first_id}__{second_id}",
                "clearance_m": float(clearance_m),
                "overlaps": bool(overlaps),
            }
            if minimum is None or sample["clearance_m"] < minimum["clearance_m"]:
                minimum = sample
            threshold = 1.25 if "bus" in sample["pair"] else 0.75
            if overlaps:
                overlap_frames.append(sample)
            if sample["clearance_m"] + 1e-9 < threshold:
                failures.append(
                    f"{sample['pair']}: minimum clearance {sample['clearance_m']:.3f} m "
                    f"below {threshold:.3f} m at frame {sample['frame_idx']}"
                )
                if len(failures) >= 50:
                    break
        if len(failures) >= 50:
            break

    report = {
        "schema": "resolved_all_pairs_clearance_validation_v1",
        "status": "PASS" if not failures and not overlap_frames else "FAIL",
        "scenario_id": resolved.scenario_id,
        "minimum_clearance": minimum,
        "minimum_clearance_required_m": 0.75,
        "minimum_bus_clearance_required_m": 1.25,
        "include_ego": False,
        "frame_step": frame_step,
        "sampled_dt_s": frame_step / float(resolved.fps),
        "overlap_frame_count": len(overlap_frames),
        "first_overlap": overlap_frames[0] if overlap_frames else None,
        "failures": failures,
    }
    save_json(output_path, report)
    if report["status"] != "PASS":
        raise RuntimeError(
            f"{resolved.scenario_id} clearance validation failed: "
            f"{len(failures)} failures, {len(overlap_frames)} overlap frames"
        )
    return report


def background_actors(seed, tesla_count, patrol_count, bus_count, pedestrian_count):
    rng = random.Random(seed)
    actors = []
    occupied = []

    def reserve(x, y, min_dx=12.0, min_dy=2.8):
        for ox, oy, odx, ody in occupied:
            if abs(x - ox) < max(min_dx, odx) and abs(y - oy) < max(min_dy, ody):
                return False
        occupied.append((x, y, min_dx, min_dy))
        return True

    # Keep the two central lanes reserved for metric actors and their conflict
    # regions. Dense background traffic is parked/queued in outer lanes so it
    # creates visual context without authored actor overlap.
    lane_choices = [-3.0 * LANE, -2.0 * LANE, 2.0 * LANE, 3.0 * LANE]
    blocked_x_ranges = [(34.0, 104.0), (124.0, 184.0)]
    base_x_slots = [
        14.0, 27.0, 108.0, 196.0, 209.0, 222.0, 235.0, 248.0, 261.0, 274.0
    ]
    vehicle_specs = (
        [(BUS, "bg_bus", bus_count)]
        + [(PATROL, "bg_patrol", patrol_count)]
        + [(TESLA, "bg_tesla", tesla_count)]
    )
    for asset_key, prefix, count in vehicle_specs:
        made = 0
        lanes = [-3.0 * LANE, 3.0 * LANE] if asset_key == BUS else lane_choices
        candidates = [
            (x + rng.uniform(-1.0, 1.0), y)
            for x in base_x_slots
            for y in lanes
        ]
        rng.shuffle(candidates)
        for x, y in candidates:
            if any(start < x < stop for start, stop in blocked_x_ranges):
                continue
            min_dx = 18.0 if asset_key == BUS else 11.0
            min_dy = 6.0 if asset_key == BUS else 2.8
            if not reserve(x, y, min_dx=min_dx, min_dy=min_dy):
                continue
            actors.append(actor(f"{prefix}_{made:02d}", asset_key, stationary(x, y), "background"))
            made += 1
            if made == count:
                break
        if made != count:
            raise RuntimeError(f"Could only place {made}/{count} {prefix} actors.")

    sidewalk_y = [4.0 * LANE, -4.0 * LANE]
    for index in range(pedestrian_count):
        x = 20.0 + index * 48.0 + rng.uniform(-2.0, 2.0)
        if 124.0 < x < 184.0:
            x += 70.0
        y = sidewalk_y[index % 2]
        actors.append(actor(f"bg_pedestrian_{index:02d}", PED, stationary(x, y, 0.0), "background"))

    return actors


def scenario(sid, weather, seed, counts):
    metric_actors = [
        actor("event_lead_tesla", TESLA, lead_brake_change_lane_stop(), "adversary"),
        actor("event_red_light_queue_bus", BUS, stationary(132.0, 0.0), "adversary"),
        actor("event_crossing_pedestrian", PED, crossing(172.0, 3.0 * LANE, -1.4 * LANE, 36.0, 44.0), "adversary"),
    ]
    actors = metric_actors + background_actors(seed, **counts)
    return {
        "schema_version": "2.0",
        "scenario_id": sid,
        "description": (
            "Long-route compound safety scenario with sequential lead braking, "
            "lane-change stop, red-light queue bus, crossing pedestrian, and "
            "dense persistent background traffic."
        ),
        "duration_s": DURATION_S,
        "fps": FPS,
        "seed": seed,
        "ego": {
            "actor_id": "ego",
            "spawn": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
            "motion": {
                "mode": "maneuver",
                "maneuver": "straight",
                "speed_mps": 5.0,
                "start_time_s": 0.0,
            },
        },
        "actors": actors,
        "camera": CAMERA,
        "long_route_metadata": {
            "weather_preset": weather,
            "background_counts": counts,
            "sequential_events": [
                {"event": "lead_brake", "route_progress_m": 42.0, "start_s": 8.0},
                {"event": "lead_lane_change_left_and_stop", "route_progress_m": 94.0, "start_s": 16.0},
                {"event": "red_light_context", "route_progress_m": 118.0, "start_s": 25.0},
                {"event": "stopped_bus_queue", "route_progress_m": 132.0, "start_s": 25.0},
                {"event": "pedestrian_crossing", "route_progress_m": 172.0, "start_s": 36.0},
            ],
        },
    }


def build_case(sid, weather, seed, counts, asset_root):
    print(f"[build] resolving {sid}", flush=True)
    resolver = V2TrajectoryResolver()
    spec = ScenarioSpecV2.model_validate(scenario(sid, weather, seed, counts))
    resolved = resolver.resolve(spec)
    print(f"[build] saving {sid}", flush=True)

    semantic_path = OUTPUT / "semantic" / f"{sid}.semantic_v2.json"
    resolved_path = OUTPUT / "resolved" / f"{sid}.resolved_v2.json"
    environment_path = OUTPUT / "environment" / f"{sid}.environment_v1.json"
    validation_path = OUTPUT / "validation" / f"{sid}.all_pairs_clearance_v1.json"
    save_json(semantic_path, spec.model_dump(mode="json"))
    save_json(resolved_path, resolved.model_dump(mode="json"))
    save_json(environment_path, {
        "schema": "scenario_environment_v1",
        "scenario_id": sid,
        "weather_preset": weather,
        "traffic_light_schedule": [
            {
                "kind": "route_context",
                "route_progress_m": 118.0,
                "state": "Red",
                "start_s": 25.0,
                "end_s": 34.0,
            },
            {
                "kind": "route_context",
                "route_progress_m": 118.0,
                "state": "Green",
                "start_s": 34.0,
                "end_s": 60.0,
            },
        ],
        "background_policy": {
            "vehicles_follow_lane_when_moving": True,
            "pedestrians_remain_on_sidewalk_except_metric_crossing": True,
            "persistent_stopped_actors": True,
        },
    })

    print(f"[build] validating {sid}", flush=True)
    validate_all_actor_pairs(resolved, validation_path)
    print(f"[build] validated {sid}", flush=True)
    actor_dump = spec.model_dump(mode="json")["actors"]
    return {
        "scenario_id": sid,
        "category": "long_route_compound_dense_traffic",
        "metric_actor_id": "event_lead_tesla",
        "metric_actor_ids": [
            "event_lead_tesla",
            "event_red_light_queue_bus",
            "event_crossing_pedestrian",
        ],
        "event_start_s": 8.0,
        "event_source_start_s": 8.0,
        "trigger_route_progress_m": None,
        "pre_trigger_source_frame": None,
        "expected_outcome": "collision_free",
        "duration_s": DURATION_S,
        "fps": FPS,
        "weather_preset": weather,
        "background_counts": counts,
        "metric_applicability": {
            "collision": True,
            "physical_clearance": True,
            "braking_response": True,
            "trajectory_tracking": True,
            "route_ttc": True,
            "conflict_point_timing": True,
            "occlusion": True,
            "comfort": True,
            "renderer_continuity": True,
        },
        "asset_keys": sorted({item["asset_key"] for item in actor_dump}),
        "actor_ids": [item["actor_id"] for item in actor_dump],
        "semantic": rel_to_sg(semantic_path),
        "resolved": rel_to_sg(resolved_path),
        "resolved_sha256": sha256(resolved_path),
        "environment": rel_to_sg(environment_path),
        "validation": rel_to_sg(validation_path),
        "status": "requires_visual_acceptance",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build long-route dense-traffic curated scenarios."
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--tesla-background", type=int, default=20)
    parser.add_argument("--patrol-background", type=int, default=5)
    parser.add_argument("--bus-background", type=int, default=2)
    parser.add_argument("--pedestrian-background", type=int, default=5)
    args = parser.parse_args()

    counts = {
        "tesla_count": args.tesla_background,
        "patrol_count": args.patrol_background,
        "bus_count": args.bus_background,
        "pedestrian_count": args.pedestrian_background,
    }
    cases = [
        build_case(
            "curated_long_route_dense_clear_001",
            "ClearNoon",
            2026090501,
            counts,
            args.asset_root,
        ),
        build_case(
            "curated_long_route_dense_wet_001",
            "WetNoon",
            2026090502,
            counts,
            args.asset_root,
        ),
        build_case(
            "curated_long_route_dense_hardrain_001",
            "HardRainNoon",
            2026090503,
            counts,
            args.asset_root,
        ),
    ]
    manifest = {
        "schema": "important_curated_long_route_suite_manifest_v1",
        "suite_id": "important_curated_long_route_dense_traffic_v1",
        "fps": FPS,
        "duration_s": DURATION_S,
        "scenario_count": len(cases),
        "expected_outcome_counts": {
            "collision_free": len(cases),
            "collision_required": 0,
        },
        "models": ["tcp", "neat", "cilpp", "aimmt"],
        "conditions": ["carla", "he"],
        "route_csv": "driving_models/common/outputs/town10_spawn10_to45_route_v1/route.csv",
        "notes": [
            "Background actors are deterministic and persistent.",
            "Metric pedestrian uses the existing 4320 pedestrian bank.",
            "Traffic-light entries are environment metadata for route context.",
        ],
        "cases": cases,
    }
    save_json(OUTPUT / "long_route_dense_traffic_suite_manifest_v1.json", manifest)
    print(f"Built {len(cases)} long-route scenarios in {OUTPUT}")


if __name__ == "__main__":
    main()
