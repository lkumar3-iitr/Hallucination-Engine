from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scenario_generator.backends.he_backend.current_he_adapter import (
    load_he_template,
    resolved_to_current_he_json,
    save_he_json,
)
from scenario_generator.io.json_io import load_model
from scenario_generator.schema import ResolvedScenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ScenarioGenerator resolved JSON to current HE scenario JSON."
    )

    parser.add_argument(
        "--resolved-json",
        required=True,
        help="Path to ScenarioGenerator *.resolved.json file.",
    )

    parser.add_argument(
        "--template-json",
        required=True,
        help="Path to an existing working HE scenario JSON template.",
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="Output HE scenario JSON path.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional HE output.output_dir override.",
    )

    parser.add_argument(
        "--output-video-name",
        default=None,
        help="Optional HE output.output_video_name override.",
    )

    parser.add_argument(
        "--use-keyframes",
        action="store_true",
        help=(
            "Export actor motion as keyframed_trajectory. "
            "Use this only after the HE compositor supports keyframes."
        ),
    )

    args = parser.parse_args()

    resolved = load_model(args.resolved_json, ResolvedScenario)
    template = load_he_template(args.template_json)

    he_json = resolved_to_current_he_json(
        resolved=resolved,
        template_json=template,
        output_dir=args.output_dir,
        output_video_name=args.output_video_name,
        use_keyframes=args.use_keyframes,
    )

    save_he_json(args.output_json, he_json)

    print(f"Saved current HE scenario JSON: {args.output_json}")


if __name__ == "__main__":
    main()