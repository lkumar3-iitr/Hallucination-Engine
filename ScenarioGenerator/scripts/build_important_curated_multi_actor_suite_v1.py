from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_generator.schema.scenario_schema_v2 import ScenarioSpecV2
from scenario_generator.trajectory.trajectory_resolver_v2 import V2TrajectoryResolver


OUTPUT = REPO_ROOT / "important_curated_scenarios"
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_resolved_all_pairs_clearance_v1.py"
ASSET_ROOT = Path(
    r"D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production"
)
FPS = 20
DURATION_S = 20.0
DT = 1.0 / FPS
LANE = 3.5
TESLA = "vehicle.tesla.model3"
NISSAN = "vehicle.passenger_02"
BUS = "vehicle.bus_01"
PED = "pedestrian.person_01"
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
DIMENSIONS = {
    TESLA: {"length_m": 4.792, "width_m": 2.163, "height_m": 1.488},
    NISSAN: {"length_m": 5.566, "width_m": 2.150, "height_m": 2.045},
    BUS: {"length_m": 10.273, "width_m": 3.944, "height_m": 4.253},
    PED: {"length_m": 0.375, "width_m": 0.375, "height_m": 1.860},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rel_to_sg(path: Path) -> str:
    return os.path.relpath(path.resolve(), PROJECT_ROOT.resolve())


def smoothstep(value: float) -> float:
    u = max(0.0, min(1.0, float(value)))
    return u * u * (3.0 - 2.0 * u)


def minjerk(value: float) -> float:
    u = max(0.0, min(1.0, float(value)))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def frames_from_xy(position, stationary_yaw=0.0):
    frames = []
    previous = None
    for idx in range(int(round(DURATION_S * FPS)) + 1):
        t_s = idx * DT
        x_m, y_m = position(t_s)
        before = max(0.0, t_s - DT)
        after = min(DURATION_S, t_s + DT)
        bx, by = position(before)
        ax, ay = position(after)
        yaw = stationary_yaw
        if math.hypot(ax - bx, ay - by) > 1e-9:
            yaw = math.degrees(math.atan2(ay - by, ax - bx))
        if previous is None:
            vx = vy = speed = 0.0
        else:
            vx = (x_m - previous[0]) / DT
            vy = (y_m - previous[1]) / DT
            speed = math.hypot(vx, vy)
        frames.append({
            "t_s": round(t_s, 8),
            "x_m": round(float(x_m), 8),
            "y_m": round(float(y_m), 8),
            "yaw_deg": round(float(yaw), 8),
        })
        previous = (x_m, y_m, speed)
    return frames


def straight(x0, speed, y):
    return frames_from_xy(lambda t: (x0 + speed * t, y), 0.0)


def stationary(x, y, yaw=0.0):
    return frames_from_xy(lambda _t: (x, y), yaw)


def speed_profile(x0, y, speed_at_t):
    samples = [(0.0, float(x0))]
    x = float(x0)
    previous_speed = float(speed_at_t(0.0))
    for idx in range(1, int(round(DURATION_S * FPS)) + 1):
        t = idx * DT
        speed = float(speed_at_t(t))
        x += 0.5 * (previous_speed + speed) * DT
        samples.append((t, x))
        previous_speed = speed

    def pos(t):
        idx = max(0, min(len(samples) - 1, int(round(t * FPS))))
        return samples[idx][1], y

    return frames_from_xy(pos, 0.0)


def braking_speed(initial_speed, start_s, duration_s, final_speed=0.0):
    def speed(t):
        u = minjerk((t - start_s) / duration_s)
        return initial_speed + (final_speed - initial_speed) * u
    return speed


def lane_change(x0, speed, y0, y1, start_s, end_s):
    return frames_from_xy(
        lambda t: (
            x0 + speed * t,
            y0 + (y1 - y0) * minjerk((t - start_s) / (end_s - start_s)),
        )
    )


def crossing(x, y0, y1, start_s, end_s):
    yaw = 90.0 if y1 > y0 else -90.0
    return frames_from_xy(
        lambda t: (
            x,
            y0 + (y1 - y0) * minjerk((t - start_s) / (end_s - start_s)),
        ),
        yaw,
    )


def actor(actor_id, asset_key, keyframes, role="adversary"):
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


def scenario(sid, description, ego_speed, actors):
    return {
        "schema_version": "2.0",
        "scenario_id": sid,
        "description": description,
        "duration_s": DURATION_S,
        "fps": FPS,
        "seed": 20260905,
        "ego": {
            "actor_id": "ego",
            "spawn": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
            "motion": {
                "mode": "maneuver",
                "maneuver": "straight",
                "speed_mps": ego_speed,
                "start_time_s": 0.0,
            },
        },
        "actors": actors,
        "camera": CAMERA,
    }


def cases():
    return [
        {
            "category": "lead_brake_adjacent_overtaker",
            "metric_actor_id": "lead_tesla",
            "event_start_s": 5.5,
            "spec": scenario(
                "curated_multi_01_lead_brake_adjacent_overtaker_001",
                "Lead Tesla brakes smoothly while a Patrol remains in the adjacent left lane.",
                5.0,
                [
                    actor("lead_tesla", TESLA, speed_profile(82.0, 0.0, braking_speed(4.8, 7.0, 3.0))),
                    actor("left_patrol", NISSAN, straight(26.0, 6.0, LANE), "traffic"),
                ],
            ),
        },
        {
            "category": "cutout_reveal_stopped_hazard",
            "metric_actor_id": "stopped_tesla",
            "event_start_s": 6.0,
            "spec": scenario(
                "curated_multi_02_cutout_reveal_stopped_tesla_001",
                "Patrol cuts out to the left and reveals a stopped Tesla in the ego lane.",
                5.0,
                [
                    actor("cutout_patrol", NISSAN, lane_change(24.0, 4.5, 0.0, LANE, 6.0, 9.0), "traffic"),
                    actor("stopped_tesla", TESLA, stationary(66.0, 0.0)),
                ],
            ),
        },
        {
            "category": "crossing_vehicle_parked_bus_occluder",
            "metric_actor_id": "crossing_tesla",
            "event_start_s": 7.5,
            "spec": scenario(
                "curated_multi_03_crossing_tesla_parked_bus_001",
                "A bus parked on the left partly occludes a Tesla crossing ahead.",
                4.8,
                [
                    actor("parked_bus", BUS, stationary(48.0, LANE + 1.8), "traffic"),
                    actor("crossing_tesla", TESLA, crossing(62.0, LANE + 6.0, -LANE - 2.0, 7.5, 12.5)),
                ],
            ),
        },
        {
            "category": "offset_stopped_queue",
            "metric_actor_id": "queue_tail_tesla",
            "event_start_s": 0.0,
            "spec": scenario(
                "curated_multi_04_offset_stopped_queue_001",
                "Stopped Tesla queue tail with a bus farther ahead and offset to the left.",
                4.8,
                [
                    actor("queue_tail_tesla", TESLA, stationary(54.0, 0.0)),
                    actor("queue_front_bus", BUS, stationary(74.0, LANE + 1.7), "traffic"),
                ],
            ),
        },
        {
            "category": "occluded_pedestrian_reveal",
            "metric_actor_id": "crossing_pedestrian",
            "event_start_s": 8.0,
            "spec": scenario(
                "curated_multi_05_pedestrian_reveal_by_bus_001",
                "A pedestrian using the 4320 bank emerges from behind a stopped bus.",
                4.5,
                [
                    actor("stopped_bus", BUS, stationary(50.0, LANE + 1.9), "traffic"),
                    actor("crossing_pedestrian", PED, crossing(61.0, LANE + 4.8, -LANE - 0.8, 8.0, 12.0)),
                ],
            ),
        },
        {
            "category": "compound_cutin_with_lead",
            "metric_actor_id": "lead_patrol",
            "event_start_s": 5.5,
            "spec": scenario(
                "curated_multi_06_lead_brake_left_cutin_001",
                "A lead Patrol brakes while a Tesla cuts in from the left behind it.",
                5.0,
                [
                    actor("lead_patrol", NISSAN, speed_profile(96.0, 0.0, braking_speed(5.0, 8.0, 3.0))),
                    actor("left_cutin_tesla", TESLA, lane_change(24.0, 5.2, LANE, 0.0, 5.5, 9.5), "traffic"),
                ],
            ),
        },
    ]


def main():
    resolver = V2TrajectoryResolver()
    manifest_cases = []
    for item in cases():
        spec = ScenarioSpecV2.model_validate(item["spec"])
        resolved = resolver.resolve(spec)
        sid = spec.scenario_id
        semantic_path = OUTPUT / "semantic" / f"{sid}.semantic_v2.json"
        resolved_path = OUTPUT / "resolved" / f"{sid}.resolved_v2.json"
        environment_path = OUTPUT / "environment" / f"{sid}.environment_v1.json"
        validation_path = OUTPUT / "validation" / f"{sid}.all_pairs_clearance_v1.json"
        save_json(semantic_path, spec.model_dump(mode="json"))
        save_json(resolved_path, resolved.model_dump(mode="json"))
        save_json(environment_path, {
            "schema": "scenario_environment_v1",
            "scenario_id": sid,
            "weather_preset": "ClearNoon",
        })
        command = [
            sys.executable,
            str(VALIDATOR),
            str(resolved_path),
            "--asset-root", str(ASSET_ROOT),
            "--minimum-clearance-m", "1.0",
            "--minimum-bus-clearance-m", "1.5",
            "--output", str(validation_path),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        manifest_cases.append({
            "scenario_id": sid,
            "category": item["category"],
            "metric_actor_id": item["metric_actor_id"],
            "event_start_s": float(item["event_start_s"]),
            "event_source_start_s": float(item["event_start_s"]),
            "trigger_route_progress_m": None,
            "pre_trigger_source_frame": None,
            "expected_outcome": "collision_free",
            "duration_s": DURATION_S,
            "fps": FPS,
            "metric_applicability": {
                "collision": True,
                "physical_clearance": True,
                "braking_response": True,
                "trajectory_tracking": True,
                "route_ttc": item["category"] not in {
                    "crossing_vehicle_parked_bus_occluder",
                    "occluded_pedestrian_reveal",
                },
                "conflict_point_timing": item["category"] in {
                    "crossing_vehicle_parked_bus_occluder",
                    "occluded_pedestrian_reveal",
                },
                "occlusion": item["category"] in {
                    "cutout_reveal_stopped_hazard",
                    "crossing_vehicle_parked_bus_occluder",
                    "occluded_pedestrian_reveal",
                },
                "comfort": True,
                "renderer_continuity": True,
            },
            "asset_keys": sorted({actor["asset_key"] for actor in spec.model_dump(mode="json")["actors"]}),
            "actor_ids": [actor.actor_id for actor in spec.actors],
            "semantic": str(semantic_path.relative_to(OUTPUT)),
            "resolved": rel_to_sg(resolved_path),
            "resolved_sha256": sha256(resolved_path),
            "environment": rel_to_sg(environment_path),
            "validation": rel_to_sg(validation_path),
            "status": "requires_visual_acceptance",
        })
    manifest = {
        "schema": "important_curated_multi_actor_suite_manifest_v1",
        "suite_id": "important_curated_multi_actor_safety_critical_v1",
        "fps": FPS,
        "duration_s": DURATION_S,
        "scenario_count": len(manifest_cases),
        "expected_outcome_counts": {
            "collision_free": len(manifest_cases),
            "collision_required": 0,
        },
        "models": ["tcp", "neat", "cilpp", "aimmt"],
        "conditions": ["carla", "he"],
        "route_csv": "driving_models/common/outputs/town10_spawn10_to45_route_v1/route.csv",
        "cases": manifest_cases,
    }
    save_json(OUTPUT / "multi_actor_safety_critical_suite_manifest_v1.json", manifest)
    print(f"Built {len(manifest_cases)} curated multi-actor scenarios in {OUTPUT}")


if __name__ == "__main__":
    main()
