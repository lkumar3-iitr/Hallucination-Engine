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


NISSAN_ASSET_KEY = "vehicle.passenger_02"
FPS = 20
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


def frame(t_s: float, x_m: float, y_m: float, yaw_deg: float = 0.0) -> dict:
    return {
        "t_s": t_s,
        "x_m": x_m,
        "y_m": y_m,
        "yaw_deg": yaw_deg,
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


def patrol(actor_id: str, keyframes: list[dict], *, role: str) -> dict:
    first = keyframes[0]
    return {
        "actor_id": actor_id,
        "actor_type": "vehicle",
        "role": role,
        "asset_key": NISSAN_ASSET_KEY,
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
        "fps": FPS,
        "seed": 0,
        "ego": ego(),
        "actors": actors,
        "camera": CAMERA,
    }


def build_cases() -> list[tuple[dict, dict]]:
    return [
        (
            scenario(
                "po0_cutin_reveal_stopped_patrol_002",
                "A moving lead Patrol initially occludes a stopped Patrol, then "
                "cuts into the right lane and reveals the persistent hazard.",
                6.0,
                [
                    patrol(
                        "occluding_lead",
                        [
                            frame(0.0, 18.0, 0.0),
                            frame(2.0, 26.0, 0.0),
                            frame(2.5, 28.0, -0.5, -15.0),
                            frame(3.0, 30.0, -1.75, -28.0),
                            frame(3.5, 32.0, -3.0, -20.0),
                            frame(4.0, 34.0, -3.5, 0.0),
                            frame(6.0, 42.0, -3.5),
                        ],
                        role="traffic",
                    ),
                    patrol(
                        "hidden_stopped_hazard",
                        [frame(0.0, 36.0, 0.0), frame(6.0, 36.0, 0.0)],
                        role="adversary",
                    ),
                ],
            ),
            {
                "observation_target": "hidden_stopped_hazard",
                "primary_occluder": "occluding_lead",
                "expected_phase_order": ["occluded", "progressive_reveal", "visible"],
                "nominal_transition_s": [2.0, 3.5],
            },
        ),
        (
            scenario(
                "po1_hidden_patrol_reappearance_001",
                "A Patrol remains in the world with one actor ID while a nearer "
                "Patrol crosses its line of sight, causing disappearance and reappearance.",
                7.0,
                [
                    patrol(
                        "persistent_target",
                        [frame(0.0, 42.0, 0.0), frame(7.0, 49.0, 0.0)],
                        role="adversary",
                    ),
                    patrol(
                        "moving_occluder",
                        [
                            frame(0.0, 25.0, -3.5),
                            frame(2.0, 31.0, -3.5),
                            frame(2.5, 32.5, -1.75, 35.0),
                            frame(3.0, 34.0, 0.0, 49.0),
                            frame(3.5, 35.5, 0.0, 20.0),
                            frame(4.0, 37.0, 0.0),
                            frame(4.5, 38.5, 1.75, 35.0),
                            frame(5.0, 40.0, 3.5, 49.0),
                            frame(5.5, 41.5, 3.5, 20.0),
                            frame(7.0, 46.0, 3.5),
                        ],
                        role="traffic",
                    ),
                ],
            ),
            {
                "observation_target": "persistent_target",
                "primary_occluder": "moving_occluder",
                "expected_phase_order": ["visible", "occluded", "visible"],
                "nominal_transition_s": [3.0, 5.0],
                "identity_must_persist": True,
            },
        ),
        (
            scenario(
                "po2_parked_occlusion_crossing_patrol_001",
                "A crossing Patrol approaches from the right behind a stationary "
                "roadside Patrol and progressively emerges into the ego path.",
                7.0,
                [
                    patrol(
                        "parked_occluder",
                        [frame(0.0, 30.0, -3.5), frame(7.0, 30.0, -3.5)],
                        role="traffic",
                    ),
                    patrol(
                        "crossing_hazard",
                        [
                            frame(0.0, 36.0, -8.0, 90.0),
                            frame(3.0, 36.0, -5.0, 90.0),
                            frame(5.0, 36.0, -2.5, 90.0),
                            frame(7.0, 36.0, 0.0, 90.0),
                        ],
                        role="adversary",
                    ),
                ],
            ),
            {
                "observation_target": "crossing_hazard",
                "primary_occluder": "parked_occluder",
                "expected_phase_order": ["occluded", "progressive_reveal", "visible"],
                "nominal_transition_s": [3.0, 6.0],
            },
        ),
    ]


def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Nissan-only partial-observability candidate suite."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "partial_observability_v1",
    )
    args = parser.parse_args()

    resolver = V2TrajectoryResolver()
    manifest = []
    for raw, observability in build_cases():
        spec = ScenarioSpecV2.model_validate(raw)
        resolved = resolver.resolve(spec)
        spec_path = args.output_root / "specs" / f"{spec.scenario_id}.scenario_v2.json"
        resolved_path = (
            args.output_root / "resolved" / f"{spec.scenario_id}.resolved_v2.json"
        )
        save_json(spec_path, spec.model_dump(mode="json"))
        save_json(resolved_path, resolved.model_dump(mode="json"))
        manifest.append(
            {
                "scenario_id": spec.scenario_id,
                "status": "candidate_requires_matched_visual_acceptance",
                "asset_keys": [NISSAN_ASSET_KEY],
                "actor_ids": [item.actor_id for item in spec.actors],
                "observability": observability,
                "spec": str(spec_path.relative_to(PROJECT_ROOT)),
                "resolved": str(resolved_path.relative_to(PROJECT_ROOT)),
            }
        )

    save_json(args.output_root / "candidate_manifest.json", manifest)
    print(f"Built {len(manifest)} partial-observability candidates in {args.output_root}")


if __name__ == "__main__":
    main()
