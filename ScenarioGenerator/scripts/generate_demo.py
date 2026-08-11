from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse
from pathlib import Path

from scenario_generator.backends.he_backend.he_compositor_adapter import (
    resolved_to_he_compositor_json,
)
from scenario_generator.io.json_io import load_model, save_model
from scenario_generator.planner.manual_planner import (
    make_cut_in_vehicle_demo,
    make_oncoming_vehicle_demo,
    make_oncoming_vehicle_with_360_rig_demo,
    make_static_vehicle_demo,
    make_cut_in_from_left_demo,
    make_cut_in_from_right_demo,
    make_fast_oncoming_demo,
    make_slow_static_near_demo,
)
from scenario_generator.schema import ScenarioSpec
from scenario_generator.trajectory.rule_based.rule_based_generator import (
    RuleBasedTrajectoryGenerator,
)
from scenario_generator.validator.scenario_validator import ScenarioValidator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        choices=[
            "oncoming",
            "static",
            "cut_in",
            "oncoming_360",
            "cut_in_right",
            "cut_in_left",
            "fast_oncoming",
            "static_near",
        ],
        default="oncoming",
    )
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="examples/resolved")
    args = parser.parse_args()

    if args.input_json:
        scenario = load_model(args.input_json, ScenarioSpec)
    elif args.demo == "oncoming":
        scenario = make_oncoming_vehicle_demo()
    elif args.demo == "static":
        scenario = make_static_vehicle_demo()
    elif args.demo == "cut_in":
        scenario = make_cut_in_vehicle_demo()
    elif args.demo == "oncoming_360":
        scenario = make_oncoming_vehicle_with_360_rig_demo()
    elif args.demo == "cut_in_right":
        scenario = make_cut_in_from_right_demo()
    elif args.demo == "cut_in_left":
        scenario = make_cut_in_from_left_demo()
    elif args.demo == "fast_oncoming":
        scenario = make_fast_oncoming_demo()
    elif args.demo == "static_near":
        scenario = make_slow_static_near_demo()
    else:
        raise ValueError(args.demo)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validator = ScenarioValidator()
    structured_result = validator.validate_structured(scenario)
    if not structured_result.ok:
        for issue in structured_result.issues:
            print(f"[{issue.severity.upper()}] {issue.message}")
        raise SystemExit("Structured scenario validation failed.")

    generator = RuleBasedTrajectoryGenerator()
    resolved = generator.resolve(scenario)

    resolved_result = validator.validate_resolved(resolved)
    for issue in resolved_result.issues:
        print(f"[{issue.severity.upper()}] {issue.message}")

    resolved_path = out_dir / f"{scenario.scenario_id}.resolved.json"
    he_path = out_dir / f"{scenario.scenario_id}.he_compositor.json"

    save_model(resolved_path, resolved)

    he_json = resolved_to_he_compositor_json(resolved)
    import json

    with he_path.open("w", encoding="utf-8") as f:
        json.dump(he_json, f, indent=2)
        f.write("\n")

    print(f"Saved resolved scenario: {resolved_path}")
    print(f"Saved HE compositor JSON: {he_path}")


if __name__ == "__main__":
    main()
