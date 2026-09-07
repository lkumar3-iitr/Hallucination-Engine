from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_generator.schema.scenario_schema_v2 import ScenarioSpecV2
from scenario_generator.trajectory.trajectory_resolver_v2 import V2TrajectoryResolver


FPS = 20
DURATION_S = 10.0
DT = 1.0 / FPS
NISSAN = "vehicle.passenger_02"
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def minimum_jerk(value: float) -> float:
    u = clamp01(value)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def transition(t_s: float, start_s: float, end_s: float) -> float:
    if end_s <= start_s:
        raise ValueError("Transition end must be after its start.")
    return minimum_jerk((float(t_s) - start_s) / (end_s - start_s))


def sampled_path(position, stationary_yaw_deg: float = 0.0) -> list[dict]:
    samples = []
    count = int(round(DURATION_S * FPS)) + 1
    epsilon = 1e-3
    for frame_idx in range(count):
        t_s = frame_idx * DT
        x_m, y_m = position(t_s)
        before = max(0.0, t_s - epsilon)
        after = min(DURATION_S, t_s + epsilon)
        bx, by = position(before)
        ax, ay = position(after)
        dx = ax - bx
        dy = ay - by
        yaw_deg = (
            float(stationary_yaw_deg)
            if math.hypot(dx, dy) <= 1e-9
            else math.degrees(math.atan2(dy, dx))
        )
        samples.append(
            {
                "t_s": round(t_s, 8),
                "x_m": round(float(x_m), 8),
                "y_m": round(float(y_m), 8),
                "yaw_deg": round(float(yaw_deg), 8),
            }
        )
    return samples


def straight(x0: float, speed_mps: float, y_m: float = 0.0):
    return sampled_path(lambda t: (x0 + speed_mps * t, y_m))


def stationary(x_m: float, y_m: float, yaw_deg: float = 0.0):
    return sampled_path(lambda _t: (x_m, y_m), stationary_yaw_deg=yaw_deg)


def lane_change(
    x0: float,
    speed_mps: float,
    y0: float,
    y1: float,
    start_s: float,
    end_s: float,
):
    return sampled_path(
        lambda t: (
            x0 + speed_mps * t,
            y0 + (y1 - y0) * transition(t, start_s, end_s),
        )
    )


def two_stage_lane_change(
    x0: float,
    speed_mps: float,
    y0: float,
    y_mid: float,
    y1: float,
    first: tuple[float, float],
    second: tuple[float, float],
):
    def position(t_s):
        first_y = y0 + (y_mid - y0) * transition(t_s, *first)
        second_y = (y1 - y_mid) * transition(t_s, *second)
        return x0 + speed_mps * t_s, first_y + second_y

    return sampled_path(position)


def speed_profile_path(
    x0: float,
    y_m: float,
    initial_speed_mps: float,
    final_speed_mps: float,
    transition_start_s: float,
    transition_end_s: float,
):
    positions = [(0.0, float(x0))]
    x_m = float(x0)
    previous_speed = float(initial_speed_mps)
    count = int(round(DURATION_S * FPS)) + 1
    for frame_idx in range(1, count):
        t_s = frame_idx * DT
        blend = transition(t_s, transition_start_s, transition_end_s)
        speed = initial_speed_mps + (final_speed_mps - initial_speed_mps) * blend
        x_m += 0.5 * (previous_speed + speed) * DT
        positions.append((t_s, x_m))
        previous_speed = speed

    def position(t_s):
        index = max(0, min(count - 1, int(round(t_s * FPS))))
        return positions[index][1], y_m

    return sampled_path(position)


def crossing(x_m: float, y0: float, y1: float, start_s: float, end_s: float):
    return sampled_path(
        lambda t: (x_m, y0 + (y1 - y0) * transition(t, start_s, end_s)),
        stationary_yaw_deg=90.0,
    )


def actor(actor_id: str, frames: list[dict], role: str = "adversary") -> dict:
    first = frames[0]
    return {
        "actor_id": actor_id,
        "actor_type": "vehicle",
        "role": role,
        "asset_key": NISSAN,
        "spawn": {
            "x_m": first["x_m"],
            "y_m": first["y_m"],
            "yaw_deg": first["yaw_deg"],
        },
        "lifecycle": {"spawn_time_s": 0.0, "despawn_time_s": None},
        "motion": {
            "mode": "keyframes",
            "interpolation": "linear",
            "keyframes": frames,
        },
    }


def scenario(case_id: str, description: str, actors: list[dict]) -> dict:
    return {
        "schema_version": "2.0",
        "scenario_id": case_id,
        "description": description,
        "duration_s": DURATION_S,
        "fps": FPS,
        "seed": 0,
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
    }


def cases() -> list[tuple[dict, dict]]:
    return [
        (
            scenario(
                "smooth_01_stopped_vehicle_001",
                "A stationary vehicle blocks the ego lane ahead.",
                [actor("stopped_hazard", stationary(65.0, 0.0))],
            ),
            {"category": "static_obstacle", "metric_actor_id": "stopped_hazard", "event_start_s": 0.0},
        ),
        (
            scenario(
                "smooth_02_lead_hard_brake_001",
                "A lead vehicle smoothly transitions from cruising to a full stop.",
                [actor("braking_lead", speed_profile_path(32.0, 0.0, 7.0, 0.0, 3.0, 5.0))],
            ),
            {"category": "sudden_braking", "metric_actor_id": "braking_lead", "event_start_s": 3.0},
        ),
        (
            scenario(
                "smooth_03_right_cutin_001",
                "An adjacent vehicle performs a smooth right-lane-to-ego-lane cut-in.",
                [actor("cutin_actor", lane_change(30.0, 6.0, -3.5, 0.0, 2.0, 6.0))],
            ),
            {"category": "cut_in", "metric_actor_id": "cutin_actor", "event_start_s": 2.0},
        ),
        (
            scenario(
                "smooth_04_cutout_reveal_001",
                "A lead vehicle smoothly cuts out and reveals a stopped hazard.",
                [
                    actor("cutout_lead", lane_change(24.0, 5.5, 0.0, -3.5, 2.0, 5.5), role="traffic"),
                    actor("revealed_hazard", stationary(62.0, 0.0)),
                ],
            ),
            {"category": "partial_observability_cutout", "metric_actor_id": "revealed_hazard", "event_start_s": 2.0},
        ),
        (
            scenario(
                "smooth_05_oncoming_pass_001",
                "An oncoming vehicle approaches and passes in the opposite lane.",
                [actor("oncoming_actor", straight(90.0, -12.0, 3.5))],
            ),
            {"category": "oncoming", "metric_actor_id": "oncoming_actor", "event_start_s": 3.5},
        ),
        (
            scenario(
                "smooth_06_crossing_vehicle_001",
                "A side-road vehicle smoothly crosses the ego path.",
                [actor("crossing_actor", crossing(30.0, -3.5, 5.5, 2.0, 6.0))],
            ),
            {"category": "crossing_vehicle", "metric_actor_id": "crossing_actor", "event_start_s": 4.0},
        ),
        (
            scenario(
                "smooth_07_ramp_merge_001",
                "A vehicle follows a long smooth merge from a separated right lane.",
                [actor("merging_actor", lane_change(25.0, 6.5, -3.5, 0.0, 2.0, 7.0))],
            ),
            {"category": "merge", "metric_actor_id": "merging_actor", "event_start_s": 2.0},
        ),
        (
            scenario(
                "smooth_08_adversary_overtake_001",
                "A faster adjacent vehicle overtakes ego and merges ahead smoothly.",
                [actor("overtaking_actor", lane_change(-12.0, 8.5, -3.5, 0.0, 5.0, 8.0))],
            ),
            {"category": "adversary_overtake", "metric_actor_id": "overtaking_actor", "event_start_s": 3.0},
        ),
        (
            scenario(
                "smooth_09_hidden_reappearance_001",
                "A persistent target disappears behind a smooth two-lane crossing occluder and reappears.",
                [
                    actor("persistent_target", straight(58.0, 4.0, 0.0)),
                    actor(
                        "moving_occluder",
                        two_stage_lane_change(30.0, 6.0, -3.5, 0.0, 3.5, (1.5, 4.5), (5.5, 8.5)),
                        role="traffic",
                    ),
                ],
            ),
            {"category": "partial_observability_reappearance", "metric_actor_id": "persistent_target", "event_start_s": 1.5},
        ),
        (
            scenario(
                "smooth_10_lead_plus_cutin_001",
                "A braking lead and a smooth adjacent cut-in create a multi-actor conflict.",
                [
                    actor("braking_lead", speed_profile_path(42.0, 0.0, 5.0, 2.0, 5.0, 7.0)),
                    actor("secondary_cutin", lane_change(28.0, 4.5, -3.5, 0.0, 2.0, 5.5), role="traffic"),
                ],
            ),
            {"category": "multi_actor", "metric_actor_id": "braking_lead", "event_start_s": 2.0},
        ),
    ]


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def motion_qa(resolved) -> dict:
    actor_reports = {}
    for actor_info in resolved.actors:
        frames = [frame for frame in resolved.actor_frames if frame.actor_id == actor_info.actor_id]
        yaw_rates = []
        accelerations = []
        previous_speed = None
        for previous, current in zip(frames, frames[1:]):
            dt = current.t_s - previous.t_s
            yaw_delta = ((current.yaw_deg - previous.yaw_deg + 180.0) % 360.0) - 180.0
            yaw_rates.append(abs(yaw_delta / dt))
            speed = math.hypot(current.x_m - previous.x_m, current.y_m - previous.y_m) / dt
            if previous_speed is not None:
                accelerations.append(abs(speed - previous_speed) / dt)
            previous_speed = speed
        actor_reports[actor_info.actor_id] = {
            "max_abs_yaw_rate_deg_s": max(yaw_rates, default=0.0),
            "max_abs_acceleration_mps2": max(accelerations, default=0.0),
        }
    return actor_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ten smooth safety benchmark scenarios.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "smooth_safety_suite_v1",
    )
    args = parser.parse_args()
    resolver = V2TrajectoryResolver()
    manifest = {
        "schema": "smooth_safety_suite_manifest_v1",
        "suite_id": "smooth_safety_suite_v1",
        "fps": FPS,
        "duration_s": DURATION_S,
        "scenario_count": 10,
        "models": ["tcp", "neat", "cilpp", "aimmt"],
        "conditions": ["carla", "he"],
        "cases": [],
    }
    for raw, metadata in cases():
        spec = ScenarioSpecV2.model_validate(raw)
        resolved = resolver.resolve(spec)
        spec_path = args.output_root / "specs" / f"{spec.scenario_id}.scenario_v2.json"
        resolved_path = args.output_root / "resolved" / f"{spec.scenario_id}.resolved_v2.json"
        save_json(spec_path, spec.model_dump(mode="json"))
        save_json(resolved_path, resolved.model_dump(mode="json"))
        manifest["cases"].append(
            {
                "scenario_id": spec.scenario_id,
                **metadata,
                "asset_keys": sorted({item.asset_key for item in spec.actors}),
                "actor_ids": [item.actor_id for item in spec.actors],
                "critical_frames": [0, 40, 80, 100, 120, 160, 200],
                "spec": str(spec_path.relative_to(PROJECT_ROOT)),
                "resolved": str(resolved_path.relative_to(PROJECT_ROOT)),
                "resolved_sha256": sha256(resolved_path),
                "motion_qa": motion_qa(resolved),
                "status": "requires_visual_acceptance",
            }
        )
    save_json(args.output_root / "suite_manifest.json", manifest)
    print(f"Built {len(manifest['cases'])} scenarios in {args.output_root}")


if __name__ == "__main__":
    main()
