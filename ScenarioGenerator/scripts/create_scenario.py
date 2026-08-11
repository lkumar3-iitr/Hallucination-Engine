from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scenario_generator.backends.he_backend.current_he_adapter import (
    load_he_template,
    resolved_to_current_he_json,
    save_he_json,
)
from scenario_generator.io.json_io import save_model
from scenario_generator.planner.scenario_factory import make_scenario_from_params
from scenario_generator.schema import ScenarioSpec
from scenario_generator.trajectory.rule_based.rule_based_generator import (
    RuleBasedTrajectoryGenerator,
)
from scenario_generator.validator.scenario_validator import ScenarioValidator
from scenario_generator.visualization.bev_debug_plot import plot_resolved_bev


def _default_output_name(
    scenario_type: str,
    side: str | None,
    start_distance_m: float,
    speed_mps: float,
) -> str:
    side_part = f"_{side}" if side else ""

    def tag(v: float) -> str:
        s = f"{float(v):.2f}".rstrip("0").rstrip(".")
        return s.replace("-", "m").replace(".", "p")

    return f"{scenario_type}{side_part}_d{tag(start_distance_m)}_v{tag(speed_mps)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a parameterized scenario: structured JSON, resolved JSON, "
            "BEV plot, and optionally current HE JSON."
        )
    )
    parser.add_argument(
        "--input-video",
        default=None,
        help=(
            "Optional input video path to write into current HE JSON. "
            "Example: recordings/ego_slow_forward_spawn10_400.mp4"
        ),
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help=(
            "Frame index in the input video where the HE scenario should start. "
            "Example: --start-frame 300 starts rendering at video frame 300."
        ),
    )

    parser.add_argument(
        "--ego-pose-jsonl",
        default=None,
        help=(
            "Optional ego pose JSONL path recorded with the CARLA ego video. "
            "Used later by HE compositor for ego-pose-aware rendering."
        ),
    )

    parser.add_argument(
        "--actor-state-frame",
        default="ego_initial",
        choices=["camera_relative", "ego_initial"],
        help=(
            "Frame in which generated actor states are defined. "
            "camera_relative = old behavior. "
            "ego_initial = actor states are defined relative to ego pose at frame 0."
        ),
    )

    parser.add_argument(
        "--runtime-transform",
        default="none",
        choices=["none", "ego_pose_jsonl"],
        help=(
            "Runtime transform used by HE compositor. "
            "Use ego_pose_jsonl when --ego-pose-jsonl is provided."
        ),
    )

    parser.add_argument(
        "--type",
        required=True,
        choices=["static", "oncoming", "following", "cut_in", "crossing"],
        help="Scenario type.",
    )

    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default=None,
        help="Side for cut_in or crossing.",
    )

    parser.add_argument(
        "--start-distance",
        type=float,
        default=30.0,
        help="Initial forward distance of adversary in meters.",
    )

    parser.add_argument(
        "--lane-y",
        type=float,
        default=0.0,
        help=(
            "Lane lateral y for static/oncoming/following. "
            "For cut_in, side decides start y unless using custom factory later."
        ),
    )

    parser.add_argument(
        "--target-lane-y",
        type=float,
        default=0.0,
        help="Target lateral y for cut_in.",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="Adversary speed in m/s.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Scenario duration in seconds.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Scenario FPS. Use 30 for HE videos.",
    )

    parser.add_argument(
        "--ego-speed",
        type=float,
        default=5.0,
        help="Ego forward speed in m/s.",
    )

    parser.add_argument(
        "--lane-width",
        type=float,
        default=3.5,
        help="Lane width in meters.",
    )

    parser.add_argument(
        "--cut-start",
        type=float,
        default=1.5,
        help="Cut-in start time in seconds.",
    )

    parser.add_argument(
        "--cut-duration",
        type=float,
        default=3.0,
        help="Cut-in lateral transition duration in seconds.",
    )

    parser.add_argument(
        "--scenario-id",
        default=None,
        help="Optional explicit scenario_id.",
    )

    parser.add_argument(
        "--out-root",
        default="examples/generated",
        help="Root folder for generated scenario outputs.",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Optional output folder name.",
    )

    parser.add_argument(
        "--template-json",
        default=None,
        help="Working HE scenario JSON template. Required if --make-he-json is set.",
    )

    parser.add_argument(
        "--make-he-json",
        action="store_true",
        help="Also export current HE JSON.",
    )

    parser.add_argument(
        "--use-keyframes",
        action="store_true",
        help=(
            "Export HE motion as keyframed_trajectory. "
            "Recommended for cut_in and crossing."
        ),
    )

    args = parser.parse_args()

    # Convenience: if an ego pose file is provided, use the general transform mode
    # unless the user explicitly passed another runtime mode.
    if args.ego_pose_jsonl is not None and args.runtime_transform == "none":
        args.runtime_transform = "ego_pose_jsonl"

    scenario = make_scenario_from_params(
        scenario_type=args.type,
        side=args.side,
        start_distance_m=args.start_distance,
        lane_y_m=args.lane_y,
        target_lane_y_m=args.target_lane_y,
        speed_mps=args.speed,
        duration_s=args.duration,
        fps=args.fps,
        ego_speed_mps=args.ego_speed,
        lane_width_m=args.lane_width,
        cut_start_s=args.cut_start,
        cut_duration_s=args.cut_duration,
        scenario_id=args.scenario_id,
    )

    if args.name is not None:
        output_name = args.name
    else:
        output_name = _default_output_name(
            scenario_type=args.type,
            side=args.side,
            start_distance_m=args.start_distance,
            speed_mps=args.speed,
        )

    out_dir = Path(args.out_root) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    structured_path = out_dir / "structured.json"
    resolved_path = out_dir / "resolved.json"
    bev_path = out_dir / "bev.png"
    he_path = out_dir / "current_he.json"

    validator = ScenarioValidator()
    structured_result = validator.validate_structured(scenario)

    if not structured_result.ok:
        for issue in structured_result.issues:
            print(f"[{issue.severity.upper()}] {issue.message}")
        raise SystemExit("Structured scenario validation failed.")

    for issue in structured_result.issues:
        print(f"[{issue.severity.upper()}] {issue.message}")

    generator = RuleBasedTrajectoryGenerator()
    resolved = generator.resolve(scenario)

    resolved_result = validator.validate_resolved(resolved)
    for issue in resolved_result.issues:
        print(f"[{issue.severity.upper()}] {issue.message}")

    save_model(structured_path, scenario)
    save_model(resolved_path, resolved)

    plot_resolved_bev(
        scenario=resolved,
        output_path=bev_path,
        every_n_frames=max(1, int(args.fps // 2)),
        show_frame_labels=False,
    )

    print(f"Saved structured scenario: {structured_path}")
    print(f"Saved resolved scenario  : {resolved_path}")
    print(f"Saved BEV plot           : {bev_path}")

    if args.make_he_json:
        if args.template_json is None:
            raise SystemExit("--template-json is required when --make-he-json is set.")

        template = load_he_template(args.template_json)

        he_output_dir = f"he_outputs/{output_name}"
        he_output_video_name = f"{output_name}.mp4"

        he_json = resolved_to_current_he_json(
            resolved=resolved,
            template_json=template,
            output_dir=he_output_dir,
            output_video_name=he_output_video_name,
            use_keyframes=args.use_keyframes,
        )

        # Override HE timeline so the scenario can start at any frame
        # inside a longer input video.
        scenario_frame_count = int(round(args.duration * args.fps))
        he_json["timeline"] = {
            "start_frame": int(args.start_frame),
            "end_frame": int(args.start_frame + scenario_frame_count),
            "fps": float(args.fps),
        }
        # Optional override for the HE compositor input video.
        # This lets us reuse the same scenario with different CARLA ego videos.
        if args.input_video is not None:
            he_json.setdefault("input", {})
            he_json["input"]["video_path"] = args.input_video

        # Optional ego pose path for future/general ego-motion-aware rendering.
        # This is required when ego is turning and actor states should remain
        # consistent in the ego-initial/world frame.
        if args.ego_pose_jsonl is not None:
            he_json.setdefault("input", {})
            he_json["input"]["ego_pose_path"] = args.ego_pose_jsonl

        # Coordinate mode metadata.
        # Old behavior:
        #   actor_state_frame = camera_relative
        #   runtime_transform = none
        #
        # General ego-pose-aware behavior:
        #   actor_state_frame = ego_initial
        #   runtime_transform = ego_pose_jsonl
        he_json["coordinate_mode"] = {
            "actor_state_frame": args.actor_state_frame,
            "runtime_transform": args.runtime_transform,
        }

        save_he_json(he_path, he_json)

        print(f"Saved current HE JSON    : {he_path}")
        print()
        print("Run from HE root:")
        print(
            "python run_he_temporal_compositor_v1.py ^\n"
            f"  --scenario {he_path.resolve()} ^\n"
            "  --overwrite"
        )


if __name__ == "__main__":
    main()