"""Derive the static-bus overtake from the accepted Tesla scenario geometry."""
import argparse
import json
from pathlib import Path


BUS_DIMENSIONS = {
    "length_m": 10.272685050964355,
    "width_m": 3.9441518783569336,
    "height_m": 4.252848148345947,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    scenario = json.loads(args.source.read_text())
    scenario["scenario_id"] = "static_left_bus_ego_overtakes_200"
    scenario["source_description"] = (
        "Bus variant of the accepted static-adversary renderer scenario. A "
        "stationary Fuso Rosa remains parallel in the left adjacent lane while "
        "the canonical ego travels forward at 6 m/s and overtakes it completely."
    )
    actor = scenario["actors"][0]
    old_id = actor["actor_id"]
    actor.update(
        actor_id="left_bus",
        asset_key="vehicle.bus_01",
        dimensions_m=BUS_DIMENSIONS,
    )
    for frame in scenario["actor_frames"]:
        if frame["actor_id"] != old_id:
            raise ValueError(f"Unexpected actor id: {frame['actor_id']}")
        frame["actor_id"] = "left_bus"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
