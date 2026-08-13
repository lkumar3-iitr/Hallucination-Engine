from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from scenario_generator.schema.scenario_schema_v2 import (
    ScenarioSpecV2,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "scenario",
        type=str,
    )

    args = parser.parse_args()

    path = Path(
        args.scenario
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    scenario = (
        ScenarioSpecV2.model_validate(
            data
        )
    )

    print()
    print("=" * 64)
    print("ScenarioSchema v2 validation passed")
    print("=" * 64)

    print(
        "scenario_id:",
        scenario.scenario_id,
    )

    print(
        "duration_s:",
        scenario.duration_s,
    )

    print(
        "fps:",
        scenario.fps,
    )

    print(
        "actors:",
        len(scenario.actors),
    )

    print()

    for actor in scenario.actors:
        print(
            f"{actor.actor_id:20s} "
            f"type={actor.actor_type.value:10s} "
            f"role={actor.role.value:10s} "
            f"motion={actor.motion.mode}"
        )

    print("=" * 64)


if __name__ == "__main__":
    main()