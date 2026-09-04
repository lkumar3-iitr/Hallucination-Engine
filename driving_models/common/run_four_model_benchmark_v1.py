"""Sequential, resumable CARLA-versus-HE campaign orchestrator."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
RUNNER = THIS_FILE.parent / "generic_he_closed_loop_runner_v1.py"
AGGREGATOR = THIS_FILE.parent / "aggregate_four_model_benchmark_v1.py"
DEFAULT_SUITE = (
    REPO_ROOT
    / "ScenarioGenerator"
    / "outputs"
    / "smooth_safety_suite_v1"
    / "suite_manifest.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "driving_models" / "outputs" / "smooth_safety_suite_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_streaming(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ten scenarios through four models in physical CARLA and HE."
    )
    parser.add_argument("--suite-manifest", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=["tcp", "neat", "cilpp", "aimmt"])
    parser.add_argument("--conditions", nargs="+", choices=["carla", "he"], default=["carla", "he"])
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--spawn-index", type=int, default=10)
    parser.add_argument("--destination-index", type=int, default=-1)
    parser.add_argument("--weather-preset", default=None)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--he-renderer-version", choices=["v1", "v2"], default="v1")
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suite_path = args.suite_manifest.resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    selected = set(args.scenario_id)
    cases = [
        case for case in suite["cases"]
        if not selected or case["scenario_id"] in selected
    ]
    unknown = selected - {case["scenario_id"] for case in suite["cases"]}
    if unknown:
        raise ValueError(f"Unknown scenario IDs: {sorted(unknown)}")

    output_root = args.output_root.resolve()
    campaign_path = output_root / "campaign_manifest.json"
    campaign = {
        "schema": "four_model_benchmark_campaign_v1",
        "campaign_id": f"{suite['suite_id']}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "started_utc": utc_now(),
        "completed_utc": None,
        "suite_manifest": str(suite_path),
        "suite_manifest_sha256": sha256(suite_path),
        "asset_root": str(args.asset_root.resolve()),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "host": args.host,
        "port": args.port,
        "town": args.town,
        "spawn_index": args.spawn_index,
        "destination_index": args.destination_index,
        "weather_preset": args.weather_preset,
        "models": args.models,
        "conditions": args.conditions,
        "he_silhouette_scale": args.he_silhouette_scale,
        "he_warp_scale_mode": args.he_warp_scale_mode,
        "he_viewpoint_lateral_sign": args.he_viewpoint_lateral_sign,
        "he_renderer_version": args.he_renderer_version,
        "runs": [],
    }
    save_json(campaign_path, campaign)

    failures = 0
    for case in cases:
        resolved = (REPO_ROOT / "ScenarioGenerator" / case["resolved"]).resolve()
        if sha256(resolved) != case["resolved_sha256"]:
            raise RuntimeError(f"Resolved scenario hash mismatch: {resolved}")
        for model in args.models:
            for condition in args.conditions:
                run_dir = output_root / case["scenario_id"] / model
                csv_path = run_dir / f"{case['scenario_id']}_{model}_{condition}.csv"
                videos = list(run_dir.glob(f"{case['scenario_id']}_{model}_{condition}_*.mp4"))
                if args.resume and csv_path.exists() and videos:
                    status = "skipped_complete"
                    return_code = 0
                    command = []
                else:
                    command = [
                        sys.executable,
                        str(RUNNER),
                        "--model", model,
                        "--device", args.device,
                        "--resolved", str(resolved),
                        "--asset-root", str(args.asset_root.resolve()),
                        "--condition", condition,
                        "--host", args.host,
                        "--port", str(args.port),
                        "--town", args.town,
                        "--spawn-index", str(args.spawn_index),
                        "--destination-index", str(args.destination_index),
                        "--metric-actor-id", case["metric_actor_id"],
                        "--event-start-s", str(case["event_start_s"]),
                        "--output-root", str(output_root),
                        "--save-video",
                        "--he-renderer-version", args.he_renderer_version,
                        "--he-silhouette-scale", str(args.he_silhouette_scale),
                        "--he-warp-scale-mode", str(args.he_warp_scale_mode),
                        "--he-viewpoint-lateral-sign",
                        str(args.he_viewpoint_lateral_sign),
                    ]
                    if args.weather_preset:
                        command.extend(["--weather-preset", args.weather_preset])
                    if case.get("trigger_route_progress_m") is not None:
                        command.extend([
                            "--trigger-route-progress-m",
                            str(case["trigger_route_progress_m"]),
                            "--event-source-start-s",
                            str(case.get("event_source_start_s", case["event_start_s"])),
                        ])
                    if case.get("pre_trigger_source_frame") is not None:
                        command.extend([
                            "--pre-trigger-source-frame",
                            str(case["pre_trigger_source_frame"]),
                        ])
                    environment_rel = case.get("environment")
                    if environment_rel:
                        command.extend([
                            "--environment-json",
                            str((REPO_ROOT / "ScenarioGenerator" / environment_rel).resolve()),
                        ])
                    route_metrics_rel = suite.get("route_csv")
                    if route_metrics_rel:
                        command.extend([
                            "--route-metrics-csv",
                            str((REPO_ROOT / route_metrics_rel).resolve()),
                        ])
                    if args.max_frames >= 0:
                        command.extend(["--max-frames", str(args.max_frames)])
                    print("\n" + "=" * 100)
                    print(f"{case['scenario_id']} | {model} | {condition}")
                    print(subprocess.list2cmdline(command))
                    print("=" * 100)
                    if args.dry_run:
                        status = "dry_run"
                        return_code = 0
                    else:
                        log_path = run_dir / f"{condition}_runner.log"
                        return_code = run_streaming(command, log_path)
                        status = "complete" if return_code == 0 else "failed"
                campaign["runs"].append(
                    {
                        "scenario_id": case["scenario_id"],
                        "resolved_sha256": case["resolved_sha256"],
                        "model": model,
                        "condition": condition,
                        "status": status,
                        "return_code": return_code,
                        "command": command,
                        "updated_utc": utc_now(),
                    }
                )
                save_json(campaign_path, campaign)
                if return_code != 0:
                    failures += 1
                    if not args.continue_on_error:
                        raise SystemExit(return_code)

    if not args.dry_run:
        subprocess.run(
            [
                sys.executable,
                str(AGGREGATOR),
                "--suite-manifest", str(suite_path),
                "--run-root", str(output_root),
            ],
            cwd=REPO_ROOT,
            check=False,
        )
    campaign["completed_utc"] = utc_now()
    campaign["failure_count"] = failures
    save_json(campaign_path, campaign)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
