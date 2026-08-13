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


from scenario_generator.backends.he_backend.current_he_adapter_v2 import (
    load_he_template,
    resolved_v2_to_current_he_json,
    save_he_json,
)

from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedScenarioV2,
)


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resolved",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--pair-name",
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--background-video",
        default=None,
    )

    parser.add_argument(
        "--ego-pose",
        default=None,
    )

    parser.add_argument(
        "--he-output-dir",
        default=None,
    )

    parser.add_argument(
        "--he-video-name",
        default="he_output.mp4",
    )

    args = parser.parse_args()

    # ========================================================
    # Load resolved physical scenario
    # ========================================================

    resolved_path = Path(
        args.resolved
    )

    resolved = (
        ResolvedScenarioV2.model_validate(
            load_json(
                resolved_path
            )
        )
    )

    # ========================================================
    # Load known-good HE template
    # ========================================================

    template = (
        load_he_template(
            args.template
        )
    )

    # ========================================================
    # Pair naming
    # ========================================================

    pair_name = (
        args.pair_name
        if args.pair_name
        else resolved.scenario_id
    )

    # ========================================================
    # Paths stored INSIDE HE scenario JSON
    #
    # These are intentionally relative to HE_v_0.1 because the
    # compositor is normally executed from that directory.
    # ========================================================

    background_video = (
        args.background_video
        if args.background_video
        else (
            f"recordings/he_pairs/"
            f"{pair_name}/"
            f"background.mp4"
        )
    )

    ego_pose = (
        args.ego_pose
        if args.ego_pose
        else (
            f"recordings/he_pairs/"
            f"{pair_name}/"
            f"background_ego_pose.jsonl"
        )
    )

    he_output_dir = (
        args.he_output_dir
        if args.he_output_dir
        else (
            f"he_outputs/"
            f"{pair_name}_he"
        )
    )

    # ========================================================
    # Convert
    # ========================================================

    he = (
        resolved_v2_to_current_he_json(

            resolved=
                resolved,

            template_json=
                template,

            output_dir=
                he_output_dir,

            output_video_name=
                args.he_video_name,

            use_keyframes=
                True,

            background_video_path=
                background_video,

            ego_pose_path=
                ego_pose,
        )
    )

    # ========================================================
    # Output JSON path
    # ========================================================

    if args.output:

        output_path = Path(
            args.output
        )

    else:

        output_path = (

            PROJECT_ROOT
            / "outputs"
            / "v2_he_scenarios"
            / (
                resolved.scenario_id
                + ".he_scenario_v2.json"
            )
        )

    save_he_json(
        output_path,
        he,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "HE scenario v2 exported"
    )

    print(
        "=" * 72
    )

    print(
        "scenario:",
        resolved.scenario_id,
    )

    print(
        "timeline:",
        he["timeline"][
            "start_frame"
        ],
        "->",
        he["timeline"][
            "end_frame"
        ],
    )

    print(
        "fps:",
        he["timeline"]["fps"],
    )

    print(
        "actors:",
        len(
            he["adversaries"]
        ),
    )

    print()

    for adv in he["adversaries"]:

        print(
            f"{adv['id']:20s} "
            f"active="
            f"{adv['active_start_frame']:3d}"
            f".."
            f"{adv['active_end_frame']:3d} "
            f"keyframes="
            f"{len(adv['motion']['keyframes']):3d}"
        )

    print()

    print(
        "background:",
        he["input"]["video_path"],
    )

    print(
        "ego pose:",
        he["input"]["ego_pose_path"],
    )

    print(
        "HE output:",
        he["output"]["output_dir"],
    )

    print(
        "JSON:",
        output_path,
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()