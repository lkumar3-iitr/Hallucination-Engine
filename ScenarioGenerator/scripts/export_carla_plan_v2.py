from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from scenario_generator.backends.carla_backend.carla_adapter_v2 import (
    resolved_v2_to_carla_plan,
)

from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedScenarioV2,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "resolved",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    input_path = Path(
        args.resolved
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    resolved = (
        ResolvedScenarioV2.model_validate(
            data
        )
    )

    plan = resolved_v2_to_carla_plan(
        resolved
    )

    if args.output:
        output_path = Path(
            args.output
        )
    else:
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "v2_carla_plans"
            / (
                resolved.scenario_id
                + ".carla_plan_v2.json"
            )
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            plan,
            f,
            indent=2,
        )

        f.write("\n")

    print()
    print("=" * 68)
    print("CARLA v2 plan exported")
    print("=" * 68)

    print(
        "scenario:",
        resolved.scenario_id,
    )

    print(
        "ego frames:",
        len(
            plan["ego"]["frames"]
        ),
    )

    print(
        "actors:",
        len(
            plan["actors"]
        ),
    )

    for actor in plan["actors"]:
        lifecycle = actor[
            "lifecycle"
        ]

        print(
            f"{actor['actor_id']:20s} "
            f"blueprint={actor['blueprint']:18s} "
            f"frames={len(actor['frames']):3d} "
            f"first={lifecycle['first_frame_idx']} "
            f"last={lifecycle['last_frame_idx']}"
        )

    print()
    print(
        "output:",
        output_path,
    )

    print("=" * 68)


if __name__ == "__main__":
    main()