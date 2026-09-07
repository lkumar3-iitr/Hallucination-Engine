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
from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedScenarioV2,
)


def validate_resolved_structure(
    scenario: ResolvedScenarioV2,
):
    expected_frames = int(
        round(
            scenario.duration_s
            * scenario.fps
        )
    ) + 1

    if len(scenario.ego_frames) != expected_frames:
        raise ValueError(
            "Resolved ego frame count mismatch: "
            f"expected {expected_frames}, "
            f"got {len(scenario.ego_frames)}"
        )

    expected_indices = list(
        range(expected_frames)
    )
    ego_indices = [
        frame.frame_idx
        for frame in scenario.ego_frames
    ]
    if ego_indices != expected_indices:
        raise ValueError(
            "Resolved ego frame indices are not contiguous."
        )

    for frame in scenario.ego_frames:
        expected_t_s = (
            float(frame.frame_idx)
            / float(scenario.fps)
        )
        if abs(frame.t_s - expected_t_s) > 1e-9:
            raise ValueError(
                "Resolved ego frame time mismatch: "
                f"frame {frame.frame_idx} has t_s={frame.t_s}, "
                f"expected {expected_t_s}"
            )

    actor_ids = {
        actor.actor_id
        for actor in scenario.actors
    }
    frame_keys = set()

    for frame in scenario.actor_frames:
        if frame.actor_id not in actor_ids:
            raise ValueError(
                "Resolved actor frame references unknown actor: "
                f"{frame.actor_id!r}"
            )

        if frame.frame_idx >= expected_frames:
            raise ValueError(
                "Resolved actor frame index is outside the scenario: "
                f"{frame.actor_id!r} frame {frame.frame_idx}, "
                f"last valid frame is {expected_frames - 1}"
            )

        key = (
            frame.actor_id,
            frame.frame_idx,
        )
        if key in frame_keys:
            raise ValueError(
                "Duplicate resolved actor frame: "
                f"{key}"
            )
        frame_keys.add(key)

        expected_t_s = (
            float(frame.frame_idx)
            / float(scenario.fps)
        )
        if abs(frame.t_s - expected_t_s) > 1e-9:
            raise ValueError(
                "Resolved actor frame time mismatch: "
                f"{key} has t_s={frame.t_s}, "
                f"expected {expected_t_s}"
            )

    return expected_frames


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

    schema_version = data.get(
        "schema_version",
        "2.0",
    )

    if schema_version == "2.0-resolved":
        scenario = (
            ResolvedScenarioV2.model_validate(
                data
            )
        )
        expected_frames = (
            validate_resolved_structure(
                scenario
            )
        )
        artifact_kind = "resolved"
    elif schema_version == "2.0":
        scenario = (
            ScenarioSpecV2.model_validate(
                data
            )
        )
        expected_frames = None
        artifact_kind = "structured"
    else:
        raise ValueError(
            "Unsupported ScenarioSchema version: "
            f"{schema_version!r}"
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
        "artifact_kind:",
        artifact_kind,
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

    if artifact_kind == "resolved":
        print(
            "expected_frames:",
            expected_frames,
        )
        print(
            "ego_frames:",
            len(scenario.ego_frames),
        )
        print(
            "actor_frames:",
            len(scenario.actor_frames),
        )

        for actor in scenario.actors:
            print(
                f"{actor.actor_id:20s} "
                f"type={actor.actor_type.value:10s} "
                f"role={actor.role.value:10s}"
            )
    else:
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
