"""
build_multi_actor_smoke_v1.py

Minimal model-independent multi-actor ScenarioSchema v2 smoke test.

Purpose
-------
Verify that the generic runtime/backend stack supports:

    - multiple actors
    - different assets
    - overlapping lifetimes
    - spawn/despawn lifecycle
    - exact resolved trajectories

No NEAT/TCP/HE/model knowledge belongs here.

Timeline at 20 Hz
-----------------
Tesla:
    active 0.0 -> 6.0 s

Nissan Patrol:
    active 2.0 -> 8.0 s

Therefore approximately:

    frame 0   : Tesla only
    frame 40  : Tesla + Patrol
    frame 121 : Patrol only
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scenario_generator.schema.scenario_schema_v2 import (
    ScenarioSpecV2,
)

from scenario_generator.trajectory.trajectory_resolver_v2 import (
    V2TrajectoryResolver,
)


def save_json(
    path: Path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            data,
            fp,
            indent=2,
        )

        fp.write(
            "\n"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario-id",
        default="multi_actor_smoke_001",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--duration-s",
        type=float,
        default=8.0,
    )

    args = parser.parse_args()

    scenario_dict = {

        "schema_version":
            "2.0",

        "scenario_id":
            args.scenario_id,

        "description":
            (
                "Generic two-vehicle lifecycle smoke test. "
                "Tesla is active first, Nissan Patrol joins after "
                "2 seconds, and Tesla despawns after 6 seconds."
            ),

        "duration_s":
            float(
                args.duration_s
            ),

        "fps":
            int(
                args.fps
            ),

        "seed":
            0,

        # ----------------------------------------------------
        # Ego is only a coordinate-frame placeholder.
        # No physical ego is required by the CARLA actor realizer
        # smoke test.
        # ----------------------------------------------------

        "ego": {

            "actor_id":
                "ego",

            "spawn": {
                "x_m": 0.0,
                "y_m": 0.0,
                "yaw_deg": 0.0,
            },

            "motion": {
                "mode": "maneuver",
                "maneuver": "static",
                "speed_mps": 0.0,
                "start_time_s": 0.0,
            },
        },

        "actors": [

            # =================================================
            # Actor 1
            # =================================================

            {
                "actor_id":
                    "traffic_tesla",

                "actor_type":
                    "vehicle",

                "role":
                    "traffic",

                # Canonical semantic registry key.
                "asset_key":
                    "vehicle.passenger_01",

                "spawn": {
                    "x_m": 12.0,
                    "y_m": 0.0,
                    "yaw_deg": 0.0,
                },

                "lifecycle": {
                    "spawn_time_s": 0.0,
                    "despawn_time_s": 6.0,
                },

                "motion": {

                    "mode":
                        "keyframes",

                    "interpolation":
                        "linear",

                    "keyframes": [

                        {
                            "t_s": 0.0,
                            "x_m": 12.0,
                            "y_m": 0.0,
                            "yaw_deg": 0.0,
                        },

                        {
                            "t_s": 6.0,
                            "x_m": 30.0,
                            "y_m": 0.0,
                            "yaw_deg": 0.0,
                        },
                    ],
                },
            },

            # =================================================
            # Actor 2
            # =================================================

            {
                "actor_id":
                    "traffic_patrol",

                "actor_type":
                    "vehicle",

                "role":
                    "traffic",

                "asset_key":
                    "vehicle.passenger_02",

                "spawn": {
                    "x_m": 25.0,
                    "y_m": 0.0,
                    "yaw_deg": 0.0,
                },

                "lifecycle": {
                    "spawn_time_s": 2.0,
                    "despawn_time_s": None,
                },

                "motion": {

                    "mode":
                        "keyframes",

                    "interpolation":
                        "linear",

                    "keyframes": [

                        {
                            "t_s": 2.0,
                            "x_m": 25.0,
                            "y_m": 0.0,
                            "yaw_deg": 0.0,
                        },

                        {
                            "t_s": 8.0,
                            "x_m": 40.0,
                            "y_m": 0.0,
                            "yaw_deg": 0.0,
                        },
                    ],
                },
            },
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

    scenario = (
        ScenarioSpecV2
        .model_validate(
            scenario_dict
        )
    )

    resolver = (
        V2TrajectoryResolver()
    )

    resolved = resolver.resolve(
        scenario
    )

    spec_path = (
        PROJECT_ROOT
        /
        "outputs"
        /
        "v2_specs"
        /
        (
            args.scenario_id
            +
            ".scenario_v2.json"
        )
    )

    resolved_path = (
        PROJECT_ROOT
        /
        "outputs"
        /
        "v2_resolved"
        /
        (
            args.scenario_id
            +
            ".resolved_v2.json"
        )
    )

    save_json(
        spec_path,
        scenario.model_dump(
            mode="json"
        ),
    )

    save_json(
        resolved_path,
        resolved.model_dump(
            mode="json"
        ),
    )

    print()
    print(
        "=" * 78
    )

    print(
        "MULTI-ACTOR SMOKE SCENARIO BUILT"
    )

    print(
        "=" * 78
    )

    print(
        "scenario:",
        args.scenario_id,
    )

    print(
        "fps:",
        args.fps,
    )

    print(
        "duration:",
        f"{args.duration_s:.2f}s",
    )

    print(
        "actors:",
        len(
            resolved.actors
        ),
    )

    for actor in resolved.actors:

        frames = [
            frame
            for frame
            in resolved.actor_frames
            if frame.actor_id
            ==
            actor.actor_id
        ]

        print(
            f"{actor.actor_id}: "
            f"asset={actor.asset_key} "
            f"frames={len(frames)} "
            f"first="
            f"{frames[0].frame_idx if frames else None} "
            f"last="
            f"{frames[-1].frame_idx if frames else None}"
        )

    print(
        "spec:",
        spec_path,
    )

    print(
        "resolved:",
        resolved_path,
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()