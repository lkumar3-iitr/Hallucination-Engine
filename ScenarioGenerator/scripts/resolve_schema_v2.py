from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)
from scenario_generator.planner.semantic_scenario_v2 import (
    SemanticScenarioV2,
)

from scenario_generator.planner.conditional_scenario_compiler_v2 import (
    compile_semantic_scenario_with_events,
)

from scenario_generator.schema.scenario_schema_v2 import (
    ScenarioSpecV2,
)

from scenario_generator.trajectory.trajectory_resolver_v2 import (
    V2TrajectoryResolver,
)


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


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
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )

        f.write("\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "scenario",
        type=str,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help=(
            "Treat input JSON as SemanticScenarioV2 "
            "and compile semantic/event behavior "
            "offline before trajectory resolution."
        ),
    )
    args = parser.parse_args()

    input_path = Path(
        args.scenario
    )

    input_data = load_json(
        input_path
    )

    if args.semantic:
        semantic = (
            SemanticScenarioV2.model_validate(
                input_data
            )
        )

        scenario = (
            compile_semantic_scenario_with_events(
                semantic
            )
        )

    else:
        scenario = (
            ScenarioSpecV2.model_validate(
                input_data
            )
        )

    resolver = (
        V2TrajectoryResolver()
    )

    resolved = resolver.resolve(
        scenario
    )

    if args.output:
        output_path = Path(
            args.output
        )
    else:
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "v2_resolved"
            / (
                scenario.scenario_id
                + ".resolved_v2.json"
            )
        )

    save_json(
        output_path,
        resolved.model_dump(
            mode="json"
        ),
    )

    counts = Counter(
        frame.actor_id
        for frame
        in resolved.actor_frames
    )

    print()
    print("=" * 68)
    print("ScenarioSchema v2 trajectory resolution complete")
    print("=" * 68)

    print(
        "scenario_id:",
        resolved.scenario_id,
    )

    print(
        "fps:",
        resolved.fps,
    )

    print(
        "duration_s:",
        resolved.duration_s,
    )

    print(
        "ego_frames:",
        len(
            resolved.ego_frames
        ),
    )

    print(
        "actors:",
        len(
            resolved.actors
        ),
    )

    print(
        "actor_frames:",
        len(
            resolved.actor_frames
        ),
    )

    print()

    for actor in resolved.actors:
        print(
            f"{actor.actor_id:20s} "
            f"frames={counts[actor.actor_id]}"
        )

    print()
    print(
        "output:",
        output_path,
    )

    print("=" * 68)


if __name__ == "__main__":
    main()