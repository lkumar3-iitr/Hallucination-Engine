"""Build ten long-horizon, near-field safety-critical scenarios."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from build_signalized_lead_follow_v1 import (
    load_csv,
    parse_route_rows,
    route_state_at_s,
    world_to_sg,
)


ROOT = Path(__file__).resolve().parents[2]
SG_ROOT = ROOT / "ScenarioGenerator"
OUTPUT = SG_ROOT / "outputs" / "long_safety_critical_suite_v2"
ROUTE_CSV = (
    ROOT / "driving_models" / "common" / "outputs"
    / "town10_spawn10_to45_route_v1" / "route.csv"
)
BASE_ENV = (
    SG_ROOT / "outputs" / "v2_environment"
    / "signalized_lead_follow_001.environment_v1.json"
)

FPS = 20
DURATION_S = 40.0
DT = 1.0 / FPS

ASSETS = {
    "vehicle.passenger_01": ("vehicle", 4.792, 2.163, 1.488),
    "vehicle.passenger_02": ("vehicle", 5.566, 2.150, 2.045),
    "vehicle.bus_01": ("vehicle", 10.273, 3.944, 4.253),
    "pedestrian.person_01": ("pedestrian", 0.375, 0.375, 1.860),
}


def minimum_jerk(u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class RouteFrame:
    def __init__(self, route):
        self.route = route
        origin = route_state_at_s(route, 0.0)
        self.origin = (origin["x"], origin["y"], origin["yaw_deg"])

    def pose(self, s_m: float, lateral_left_m: float = 0.0, yaw_offset_deg: float = 0.0):
        state = route_state_at_s(self.route, s_m)
        yaw_rad = math.radians(state["yaw_deg"])
        world_x = state["x"] - math.sin(yaw_rad) * lateral_left_m
        world_y = state["y"] + math.cos(yaw_rad) * lateral_left_m
        return world_to_sg(
            world_x, world_y, state["yaw_deg"] + yaw_offset_deg,
            self.origin[0], self.origin[1], self.origin[2],
        )


def actor_definition(actor_id: str, asset_key: str, spawn_s: float = 0.0):
    actor_type, length, width, height = ASSETS[asset_key]
    return {
        "actor_id": actor_id,
        "actor_type": actor_type,
        "role": "traffic",
        "asset_key": asset_key,
        "dimensions_m": {"length_m": length, "width_m": width, "height_m": height},
        "spawn_time_s": float(spawn_s),
        "despawn_time_s": None,
    }


def states_to_frames(actor_id: str, states):
    frames = []
    previous = None
    for frame_idx, (t_s, x_m, y_m, yaw_deg) in enumerate(states):
        if previous is None:
            vx = vy = speed = 0.0
        else:
            vx = (x_m - previous[0]) / DT
            vy = (y_m - previous[1]) / DT
            speed = math.hypot(vx, vy)
        frames.append({
            "frame_idx": frame_idx,
            "t_s": t_s,
            "actor_id": actor_id,
            "x_m": x_m,
            "y_m": y_m,
            "yaw_deg": yaw_deg,
            "speed_mps": speed,
            "vx_mps": vx,
            "vy_mps": vy,
        })
        previous = (x_m, y_m)
    return frames


def route_motion(rf, actor_id, initial_s, speed_fn, lateral_fn=lambda _t: 0.0):
    states = []
    s_m = float(initial_s)
    for frame_idx in range(int(DURATION_S * FPS) + 1):
        t_s = frame_idx * DT
        if frame_idx:
            s_m += max(0.0, float(speed_fn(t_s - DT))) * DT
        x, y, yaw = rf.pose(s_m, lateral_fn(t_s))
        states.append((t_s, x, y, yaw))
    return states_to_frames(actor_id, tangent_oriented_states(states))


def tangent_oriented_states(states):
    """Orient moving actors along the sampled SG trajectory tangent."""
    oriented = []
    last_yaw = float(states[0][3])
    for i, (t_s, x_m, y_m, _yaw_deg) in enumerate(states):
        before = states[max(0, i - 1)]
        after = states[min(len(states) - 1, i + 1)]
        dx = float(after[1]) - float(before[1])
        dy = float(after[2]) - float(before[2])
        if math.hypot(dx, dy) > 1e-9:
            # SG velocity and yaw use the same sampled-path tangent sign.
            # The runtime's left-handed CARLA mapping then produces a world
            # heading aligned with the transformed world velocity.
            last_yaw = math.degrees(math.atan2(dy, dx))
        oriented.append((t_s, x_m, y_m, last_yaw))
    return oriented


def static_route(rf, actor_id, s_m, lateral=0.0, yaw_offset=0.0):
    pose = rf.pose(s_m, lateral, yaw_offset)
    states = [(i * DT, *pose) for i in range(int(DURATION_S * FPS) + 1)]
    return states_to_frames(actor_id, states)


def crossing(rf, actor_id, s_m, start_t, end_t, start_left, end_left, yaw_offset):
    states = []
    for i in range(int(DURATION_S * FPS) + 1):
        t_s = i * DT
        u = minimum_jerk((t_s - start_t) / (end_t - start_t))
        lateral = start_left + (end_left - start_left) * u
        states.append((t_s, *rf.pose(s_m, lateral, yaw_offset)))
    return states_to_frames(actor_id, states)


def cruise(v):
    return lambda _t: float(v)


def brake_profile(v, start_t, duration, hold=True):
    def speed(t):
        if t <= start_t:
            return v
        if t >= start_t + duration:
            return 0.0 if hold else v
        return v * (1.0 - minimum_jerk((t - start_t) / duration))
    return speed


def signal_stop_go_profile(v, brake_t=23.75, brake_duration=4.0, green_t=30.0, accel_duration=2.5):
    def speed(t):
        if t <= brake_t:
            return v
        if t < brake_t + brake_duration:
            return v * (1.0 - minimum_jerk((t - brake_t) / brake_duration))
        if t <= green_t:
            return 0.0
        if t < green_t + accel_duration:
            return v * minimum_jerk((t - green_t) / accel_duration)
        return v
    return speed


def cutin_lateral(start_t, end_t, start_left=-3.5, end_left=0.0):
    return lambda t: start_left + (end_left - start_left) * minimum_jerk(
        (t - start_t) / (end_t - start_t)
    )


def build_cases(rf):
    cases = []

    def add(
        sid, category, metric_actor, event_t, actors, frames, description,
        trigger_route_progress_m=None,
    ):
        cases.append({
            "scenario_id": sid,
            "category": category,
            "metric_actor_id": metric_actor,
            "event_start_s": event_t,
            "event_source_start_s": event_t,
            "trigger_route_progress_m": trigger_route_progress_m,
            "actors": actors,
            "frames": frames,
            "description": description,
        })

    add(
        "long_01_signalized_lead_turn_001", "signalized_lead_follow", "lead_tesla", 23.75,
        [actor_definition("lead_tesla", "vehicle.passenger_01")],
        [route_motion(rf, "lead_tesla", 12.0, signal_stop_go_profile(5.5))],
        "Lead approaches a red signal, stops, and provides long-horizon following context.",
    )
    add(
        "long_02_close_lead_emergency_brake_001", "lead_emergency_brake", "lead_nissan", 11.0,
        [actor_definition("lead_nissan", "vehicle.passenger_02")],
        [route_motion(rf, "lead_nissan", 10.0, brake_profile(6.0, 11.0, 2.2))],
        "A close lead vehicle performs a smooth but severe stop after normal following.",
        60.0,
    )
    add(
        "long_03_close_cutin_brake_001", "cutin_then_brake", "cutin_tesla", 10.0,
        [actor_definition("cutin_tesla", "vehicle.passenger_01")],
        [route_motion(rf, "cutin_tesla", 8.0, brake_profile(6.0, 13.0, 2.5), cutin_lateral(10.0, 12.5))],
        "Adjacent Tesla cuts in at close range and brakes after settling into ego lane.",
        55.0,
    )
    add(
        "long_04_bus_cutout_reveal_stop_001", "partial_observability_cutout", "hidden_nissan", 11.0,
        [actor_definition("cutout_bus", "vehicle.bus_01"), actor_definition("hidden_nissan", "vehicle.passenger_02")],
        [route_motion(rf, "cutout_bus", 9.0, cruise(5.8), lambda t: -4.0 * minimum_jerk((t - 11.0) / 3.0)),
         static_route(rf, "hidden_nissan", 90.0)],
        "A bus moves aside and reveals a stopped Nissan in the ego lane.",
        50.0,
    )
    add(
        "long_05_crossing_nissan_001", "crossing_vehicle", "crossing_nissan", 11.0,
        [actor_definition("crossing_nissan", "vehicle.passenger_02")],
        [crossing(rf, "crossing_nissan", 72.0, 11.0, 14.5, -7.0, 7.0, 90.0)],
        "A staged Nissan enters on an ego-progress trigger and reaches the conflict center 1.75 s later.",
        59.0,
    )
    add(
        "long_06_crossing_bus_001", "crossing_bus", "crossing_bus", 12.0,
        [actor_definition("crossing_bus", "vehicle.bus_01")],
        [crossing(rf, "crossing_bus", 80.0, 12.0, 17.0, -10.0, 10.0, 90.0)],
        "A staged bus enters on an ego-progress trigger and reaches the conflict center 2.50 s later.",
        63.0,
    )
    add(
        "long_07_pedestrian_emergence_001", "pedestrian_crossing", "crossing_pedestrian", 11.5,
        [actor_definition("crossing_pedestrian", "pedestrian.person_01")],
        [crossing(rf, "crossing_pedestrian", 70.0, 11.5, 15.0, -4.5, 4.5, 90.0)],
        "A staged pedestrian enters on an ego-progress trigger and reaches the conflict center 1.75 s later.",
        57.0,
    )
    add(
        "long_08_bus_occluded_pedestrian_001", "occluded_pedestrian", "hidden_pedestrian", 12.0,
        [actor_definition("parked_bus", "vehicle.bus_01"), actor_definition("hidden_pedestrian", "pedestrian.person_01")],
        [static_route(rf, "parked_bus", 70.0, -4.2),
         crossing(rf, "hidden_pedestrian", 77.0, 12.0, 15.2, -5.0, 5.0, 90.0)],
        "A grounded pedestrian emerges on an ego-progress trigger and reaches the conflict center 1.60 s later.",
        55.0,
    )
    add(
        "long_09_stopped_queue_multi_actor_001", "stopped_queue", "queue_tail", 10.5,
        [actor_definition("queue_tail", "vehicle.passenger_01"), actor_definition("queue_front", "vehicle.bus_01")],
        [route_motion(rf, "queue_tail", 9.0, brake_profile(5.8, 10.5, 2.4)),
         static_route(rf, "queue_front", 83.0)],
        "A Tesla brakes at the tail of a stopped queue behind a bus.",
        57.0,
    )
    add(
        "long_10_lead_brake_pedestrian_001", "multi_actor_compound", "lead_nissan", 10.5,
        [actor_definition("lead_nissan", "vehicle.passenger_02"), actor_definition("side_pedestrian", "pedestrian.person_01")],
        [route_motion(rf, "lead_nissan", 10.0, brake_profile(5.8, 10.5, 2.5)),
         crossing(rf, "side_pedestrian", 75.0, 12.0, 15.0, -4.5, 4.5, 90.0)],
        "A braking lead and crossing pedestrian create a compound near-field decision.",
        58.0,
    )
    return cases


def main():
    route = parse_route_rows(load_csv(ROUTE_CSV))
    rf = RouteFrame(route)
    base_env = json.loads(BASE_ENV.read_text(encoding="utf-8"))
    manifest_cases = []
    for case in build_cases(rf):
        sid = case["scenario_id"]
        resolved = {
            "schema_version": "2.0-resolved",
            "source_schema_version": "2.0",
            "scenario_id": sid,
            "source_description": case["description"],
            "duration_s": DURATION_S,
            "fps": FPS,
            "seed": 20260901,
            "coordinate_frame": "ego_initial",
            "camera": {"name": "front", "width_px": 1280, "height_px": 720, "fov_deg": 90.0, "x_m": 1.5, "y_m": 0.0, "z_m": 1.6, "pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0},
            "actors": case["actors"],
            "ego_frames": [{"frame_idx": i, "t_s": i * DT, "x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0, "speed_mps": 0.0, "vx_mps": 0.0, "vy_mps": 0.0} for i in range(int(DURATION_S * FPS) + 1)],
            "actor_frames": [frame for actor_frames in case["frames"] for frame in actor_frames],
        }
        resolved_path = OUTPUT / "resolved" / f"{sid}.resolved_v2.json"
        save_json(resolved_path, resolved)
        environment = copy.deepcopy(base_env)
        environment["scenario_id"] = sid
        environment_path = OUTPUT / "environment" / f"{sid}.environment_v1.json"
        save_json(environment_path, environment)
        manifest_cases.append({
            "scenario_id": sid,
            "category": case["category"],
            "metric_actor_id": case["metric_actor_id"],
            "event_start_s": case["event_start_s"],
            "event_source_start_s": case["event_source_start_s"],
            "trigger_route_progress_m": case["trigger_route_progress_m"],
            "asset_keys": sorted({actor["asset_key"] for actor in case["actors"]}),
            "actor_ids": [actor["actor_id"] for actor in case["actors"]],
            "resolved": str(resolved_path.relative_to(SG_ROOT)),
            "resolved_sha256": sha256(resolved_path),
            "environment": str(environment_path.relative_to(SG_ROOT)),
            "status": "requires_execution_and_visual_acceptance",
        })
    manifest = {
        "schema": "long_safety_critical_suite_manifest_v2",
        "suite_id": "long_safety_critical_suite_v2_triggered",
        "fps": FPS,
        "duration_s": DURATION_S,
        "scenario_count": len(manifest_cases),
        "models": ["tcp", "neat"],
        "conditions": ["carla", "he"],
        "route_csv": str(ROUTE_CSV.relative_to(ROOT)),
        "cases": manifest_cases,
    }
    save_json(OUTPUT / "suite_manifest.json", manifest)
    print(f"Built {len(manifest_cases)} long safety-critical scenarios in {OUTPUT}")


if __name__ == "__main__":
    main()
