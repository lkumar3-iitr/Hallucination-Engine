from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_generator.schema.scenario_schema_v2 import ScenarioSpecV2
from scenario_generator.trajectory.trajectory_resolver_v2 import (
    V2TrajectoryResolver,
)


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


def ego(speed_mps: float = 4.0) -> dict:
    return {
        "actor_id": "ego",
        "spawn": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
        "motion": {
            "mode": "maneuver",
            "maneuver": "straight",
            "speed_mps": speed_mps,
            "start_time_s": 0.0,
        },
    }


def actor(
    actor_id: str,
    actor_type: str,
    asset_key: str,
    keyframes: list[dict],
    *,
    role: str = "adversary",
    spawn_time_s: float = 0.0,
    despawn_time_s: float | None = None,
) -> dict:
    first = keyframes[0]
    return {
        "actor_id": actor_id,
        "actor_type": actor_type,
        "role": role,
        "asset_key": asset_key,
        "spawn": {
            "x_m": first["x_m"],
            "y_m": first["y_m"],
            "yaw_deg": first["yaw_deg"],
        },
        "lifecycle": {
            "spawn_time_s": spawn_time_s,
            "despawn_time_s": despawn_time_s,
        },
        "motion": {
            "mode": "keyframes",
            "interpolation": "linear",
            "keyframes": keyframes,
        },
    }


def frame(t_s: float, x_m: float, y_m: float, yaw_deg: float) -> dict:
    return {
        "t_s": t_s,
        "x_m": x_m,
        "y_m": y_m,
        "yaw_deg": yaw_deg,
    }


def scenario(
    scenario_id: str,
    description: str,
    duration_s: float,
    actors: list[dict],
) -> dict:
    return {
        "schema_version": "2.0",
        "scenario_id": scenario_id,
        "description": description,
        "duration_s": duration_s,
        "fps": 20,
        "seed": 0,
        "ego": ego(),
        "actors": actors,
        "camera": CAMERA,
    }


def build_candidates() -> list[dict]:
    return [
        scenario(
            "candidate_oncoming_patrol_001",
            "A Nissan Patrol approaches and passes in the adjacent opposing lane.",
            8.0,
            [
                actor(
                    "oncoming_patrol",
                    "vehicle",
                    "vehicle.passenger_02",
                    [
                        frame(0.0, 52.0, 3.5, 180.0),
                        frame(4.0, 22.0, 3.5, 180.0),
                        frame(8.0, -8.0, 3.5, 180.0),
                    ],
                )
            ],
        ),
        scenario(
            "candidate_pedestrian_crossing_safe_001",
            "A pedestrian crosses well before the ego reaches the crossing point.",
            8.0,
            [
                actor(
                    "crossing_pedestrian",
                    "pedestrian",
                    "pedestrian.person_01",
                    [
                        frame(1.0, 31.0, 6.0, -90.0),
                        frame(3.5, 31.0, 1.0, -90.0),
                        frame(4.5, 31.0, -1.0, -90.0),
                        frame(7.0, 31.0, -6.0, -90.0),
                    ],
                    spawn_time_s=1.0,
                    despawn_time_s=7.0,
                )
            ],
        ),
        scenario(
            "candidate_oncoming_bus_001",
            "A bus approaches in a wider opposing lane with conservative side clearance.",
            10.0,
            [
                actor(
                    "oncoming_bus",
                    "vehicle",
                    "vehicle.bus_01",
                    [
                        frame(0.0, 70.0, 5.0, 180.0),
                        frame(5.0, 35.0, 5.0, 180.0),
                        frame(10.0, 0.0, 5.0, 180.0),
                    ],
                )
            ],
        ),
        scenario(
            "candidate_lead_brake_safe_001",
            "A Tesla lead vehicle decelerates to a stop while preserving pre-contact clearance.",
            8.0,
            [
                actor(
                    "lead_tesla",
                    "vehicle",
                    "vehicle.passenger_01",
                    [
                        frame(0.0, 25.0, 0.0, 0.0),
                        frame(3.0, 34.0, 0.0, 0.0),
                        frame(5.0, 38.0, 0.0, 0.0),
                        frame(8.0, 38.0, 0.0, 0.0),
                    ],
                )
            ],
        ),
        scenario(
            "candidate_multi_lead_oncoming_001",
            "A slowing Tesla lead vehicle is observed while a Nissan passes in the opposing lane.",
            8.0,
            [
                actor(
                    "lead_tesla",
                    "vehicle",
                    "vehicle.passenger_01",
                    [
                        frame(0.0, 25.0, 0.0, 0.0),
                        frame(3.0, 34.0, 0.0, 0.0),
                        frame(5.0, 38.0, 0.0, 0.0),
                        frame(8.0, 38.0, 0.0, 0.0),
                    ],
                    role="traffic",
                ),
                actor(
                    "oncoming_patrol",
                    "vehicle",
                    "vehicle.passenger_02",
                    [
                        frame(0.0, 60.0, 3.5, 180.0),
                        frame(4.0, 28.0, 3.5, 180.0),
                        frame(8.0, -4.0, 3.5, 180.0),
                    ],
                ),
            ],
        ),
    ]


def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CARLA-free candidate ScenarioSpecV2 and resolved files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "candidate_suite_v1",
    )
    args = parser.parse_args()

    resolver = V2TrajectoryResolver()
    index = []
    for raw in build_candidates():
        spec = ScenarioSpecV2.model_validate(raw)
        resolved = resolver.resolve(spec)
        spec_path = args.output_root / "specs" / f"{spec.scenario_id}.scenario_v2.json"
        resolved_path = (
            args.output_root / "resolved" / f"{spec.scenario_id}.resolved_v2.json"
        )
        save_json(spec_path, spec.model_dump(mode="json"))
        save_json(resolved_path, resolved.model_dump(mode="json"))
        index.append(
            {
                "scenario_id": spec.scenario_id,
                "status": "candidate_not_yet_visually_accepted",
                "actor_ids": [item.actor_id for item in spec.actors],
                "spec": str(spec_path.relative_to(PROJECT_ROOT)),
                "resolved": str(resolved_path.relative_to(PROJECT_ROOT)),
            }
        )

    save_json(args.output_root / "candidate_manifest.json", index)
    print(f"Built {len(index)} offline candidates in {args.output_root}")


if __name__ == "__main__":
    main()
