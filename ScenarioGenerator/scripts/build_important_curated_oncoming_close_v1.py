from __future__ import annotations

import hashlib
import json
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
ASSET_ROOT = Path(
    r"D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production"
)
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_resolved_all_pairs_clearance_v1.py"
FPS = 20
DURATION_S = 20.0
DT = 1.0 / FPS
LANE_CENTER_M = 4.25

TESLA = "vehicle.tesla.model3"
PATROL = "vehicle.passenger_02"
BUS = "vehicle.bus_01"
PEDESTRIAN = "pedestrian.person_01"

DIMENSIONS = {
    TESLA: {"length_m": 4.792, "width_m": 2.163, "height_m": 1.488},
    PATROL: {"length_m": 5.566, "width_m": 2.150, "height_m": 2.045},
    BUS: {"length_m": 10.273, "width_m": 3.944, "height_m": 4.253},
    PEDESTRIAN: {"length_m": 0.375, "width_m": 0.375, "height_m": 1.860},
}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_to_sg(path: Path) -> str:
    return os.path.relpath(path.resolve(), PROJECT_ROOT.resolve())


def linear_keyframes(x0_m: float, speed_mps: float, y_m: float, yaw_deg: float):
    return [
        {
            "t_s": round(frame_idx * DT, 8),
            "x_m": round(x0_m + speed_mps * frame_idx * DT, 8),
            "y_m": y_m,
            "yaw_deg": yaw_deg,
        }
        for frame_idx in range(int(DURATION_S * FPS) + 1)
    ]


def actor(
    actor_id: str,
    asset_key: str,
    x0_m: float,
    speed_mps: float,
    y_m: float,
    yaw_deg: float = 180.0,
):
    frames = linear_keyframes(x0_m, speed_mps, y_m, yaw_deg)
    return {
        "actor_id": actor_id,
        "actor_type": "pedestrian" if asset_key == PEDESTRIAN else "vehicle",
        "role": "adversary" if asset_key != PEDESTRIAN else "traffic",
        "asset_key": asset_key,
        "dimensions_m": DIMENSIONS[asset_key],
        "spawn": {
            "x_m": frames[0]["x_m"],
            "y_m": frames[0]["y_m"],
            "yaw_deg": frames[0]["yaw_deg"],
        },
        "lifecycle": {"spawn_time_s": 0.0, "despawn_time_s": None},
        "motion": {
            "mode": "keyframes",
            "interpolation": "linear",
            "keyframes": frames,
        },
    }


def main() -> None:
    scenario_id = "curated_oncoming_close_all_assets_400"
    semantic = {
        "schema_version": "2.0",
        "scenario_id": scenario_id,
        "description": (
            "Three oncoming vehicle assets pass the ego sequentially in the "
            "opposing lane while a pedestrian approaches on the far sidewalk."
        ),
        "duration_s": DURATION_S,
        "fps": FPS,
        "seed": 2026090701,
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
        "actors": [
            actor("oncoming_tesla", TESLA, 45.0, -5.0, LANE_CENTER_M),
            actor("oncoming_patrol", PATROL, 100.0, -5.0, LANE_CENTER_M),
            actor("oncoming_bus", BUS, 155.0, -5.0, LANE_CENTER_M),
            actor("oncoming_pedestrian", PEDESTRIAN, 30.0, -1.5, 7.5),
        ],
        "camera": {
            "image_width": 1280,
            "image_height": 720,
            "fov_deg": 90.0,
            "x_m": 1.5,
            "y_m": 0.0,
            "z_m": 1.6,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
            "roll_deg": 0.0,
        },
    }

    spec = ScenarioSpecV2.model_validate(semantic)
    resolved = V2TrajectoryResolver().resolve(spec)
    semantic_path = OUTPUT / "semantic" / f"{scenario_id}.semantic_v2.json"
    resolved_path = OUTPUT / "resolved" / f"{scenario_id}.resolved_v2.json"
    environment_path = OUTPUT / "environment" / f"{scenario_id}.environment_v1.json"
    validation_path = OUTPUT / "validation" / f"{scenario_id}.all_pairs_clearance_v1.json"

    save_json(semantic_path, spec.model_dump(mode="json"))
    save_json(resolved_path, resolved.model_dump(mode="json"))
    save_json(
        environment_path,
        {
            "schema": "scenario_environment_v1",
            "scenario_id": scenario_id,
            "weather_preset": "ClearNoon",
        },
    )

    subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(resolved_path),
            "--asset-root",
            str(ASSET_ROOT),
            "--include-ego",
            "--minimum-clearance-m",
            "1.00",
            "--minimum-bus-clearance-m",
            "1.00",
            "--output",
            str(validation_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    manifest = {
        "schema": "important_curated_oncoming_close_suite_manifest_v1",
        "suite_id": "important_curated_oncoming_close_all_assets_v1",
        "fps": FPS,
        "duration_s": DURATION_S,
        "scenario_count": 1,
        "expected_outcome_counts": {"collision_free": 1, "collision_required": 0},
        "models": ["tcp", "neat", "cilpp", "aimmt"],
        "conditions": ["carla", "he"],
        "route_csv": "driving_models/common/outputs/town10_spawn10_to45_route_v1/route.csv",
        "cases": [
            {
                "scenario_id": scenario_id,
                "category": "oncoming_close_all_assets",
                "metric_actor_id": "oncoming_bus",
                "event_start_s": 0.0,
                "event_source_start_s": 0.0,
                "trigger_route_progress_m": None,
                "pre_trigger_source_frame": None,
                "expected_outcome": "collision_free",
                "duration_s": DURATION_S,
                "fps": FPS,
                "critical_frames": [70, 90, 110, 180, 200, 220, 290, 310, 330],
                "metric_applicability": {
                    "collision": True,
                    "physical_clearance": True,
                    "braking_response": True,
                    "trajectory_tracking": True,
                    "route_ttc": False,
                    "conflict_point_timing": False,
                    "occlusion": False,
                    "comfort": True,
                    "renderer_continuity": True,
                },
                "asset_keys": [TESLA, PATROL, BUS, PEDESTRIAN],
                "actor_ids": [
                    "oncoming_tesla",
                    "oncoming_patrol",
                    "oncoming_bus",
                    "oncoming_pedestrian",
                ],
                "semantic": str(semantic_path.relative_to(OUTPUT)),
                "resolved": rel_to_sg(resolved_path),
                "resolved_sha256": sha256(resolved_path),
                "environment": rel_to_sg(environment_path),
                "validation": rel_to_sg(validation_path),
                "status": "requires_visual_acceptance",
            }
        ],
    }
    save_json(OUTPUT / "oncoming_close_all_assets_suite_manifest_v1.json", manifest)
    print(f"Built {scenario_id}: 400 intervals, 401 authored frames, 4 actors")


if __name__ == "__main__":
    main()
