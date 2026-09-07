"""Build, validate, and run the frozen four-model paper campaign."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "ScenarioGenerator" / "scripts" / "build_paper_safety_suite_v1.py"
VALIDATOR = ROOT / "ScenarioGenerator" / "scripts" / "validate_paper_suite_v1.py"
ORCHESTRATOR = Path(__file__).with_name("run_four_model_benchmark_v1.py")
SUITE = (
    ROOT / "ScenarioGenerator" / "outputs" / "paper_safety_suite_v1"
    / "suite_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "driving_models" / "outputs" / "paper_safety_suite_v1"


def run(command: list[str]) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run 20 frozen scenarios (15 collision-free and 5 collision controls) "
            "through TCP, NEAT, CIL++, and AIM-MT in CARLA and HE."
        )
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--weather-preset", default=None)
    parser.add_argument("--he-silhouette-scale", type=float, default=1.0)
    parser.add_argument(
        "--he-warp-scale-mode",
        choices=[
            "independent",
            "uniform_height_preserve_aspect",
        ],
        default="independent",
    )
    parser.add_argument(
        "--he-viewpoint-lateral-sign",
        type=float,
        choices=[
            -1.0,
            1.0,
        ],
        default=1.0,
    )
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run([sys.executable, str(BUILDER)])
    run([
        sys.executable, str(VALIDATOR),
        "--suite-manifest", str(SUITE),
        "--output", str(args.output_root / "preflight_validation.json"),
    ])

    command = [
        sys.executable, str(ORCHESTRATOR),
        "--suite-manifest", str(SUITE),
        "--asset-root", str(args.asset_root),
        "--output-root", str(args.output_root),
        "--models", "tcp", "neat", "cilpp", "aimmt",
        "--conditions", "carla", "he",
        "--device", args.device,
        "--host", args.host,
        "--port", str(args.port),
        "--he-silhouette-scale", str(args.he_silhouette_scale),
        "--he-warp-scale-mode", str(args.he_warp_scale_mode),
        "--he-viewpoint-lateral-sign", str(args.he_viewpoint_lateral_sign),
        "--resume",
    ]
    if args.weather_preset:
        command.extend(["--weather-preset", args.weather_preset])
    if args.max_frames >= 0:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.dry_run:
        command.append("--dry-run")
    run(command)

    if not args.dry_run:
        run([
            sys.executable, str(VALIDATOR),
            "--suite-manifest", str(SUITE),
            "--run-root", str(args.output_root),
            "--output", str(args.output_root / "postrun_validation.json"),
        ])


if __name__ == "__main__":
    main()
