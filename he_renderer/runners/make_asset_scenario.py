"""Derive a matched static-overtake scenario for a registered HE asset."""

import argparse
import json
from pathlib import Path


ASSETS = {
    "patrol": {
        "scenario_id": "static_left_patrol_ego_overtakes_200",
        "actor_id": "left_patrol",
        "actor_type": "vehicle",
        "asset_key": "vehicle.passenger_02",
        "dimensions_m": {
            "length_m": 5.565828800201416,
            "width_m": 2.1499669551849365,
            "height_m": 2.045147180557251,
        },
        "label": "Nissan Patrol",
    },
    "pedestrian": {
        "scenario_id": "static_left_pedestrian_ego_overtakes_200",
        "actor_id": "left_pedestrian",
        "actor_type": "pedestrian",
        "asset_key": "pedestrian.person_01",
        "dimensions_m": {
            "length_m": 0.3753577768802643,
            "width_m": 0.3753577768802643,
            "height_m": 1.8600000143051147,
        },
        "label": "pedestrian 0001",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--asset", choices=sorted(ASSETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    spec = ASSETS[args.asset]
    scenario = json.loads(args.source.read_text())
    scenario["scenario_id"] = spec["scenario_id"]
    scenario["source_description"] = (
        f"Matched {spec['label']} variant of the accepted static-adversary "
        "overtake scenario; ego trajectory, camera, timing, and actor pose "
        "are unchanged."
    )
    actor = scenario["actors"][0]
    old_id = actor["actor_id"]
    actor.update(
        actor_id=spec["actor_id"],
        actor_type=spec["actor_type"],
        asset_key=spec["asset_key"],
        dimensions_m=spec["dimensions_m"],
    )
    for frame in scenario["actor_frames"]:
        if frame["actor_id"] != old_id:
            raise ValueError(f"Unexpected actor id: {frame['actor_id']}")
        frame["actor_id"] = spec["actor_id"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
