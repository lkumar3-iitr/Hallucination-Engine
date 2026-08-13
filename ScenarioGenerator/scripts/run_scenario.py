from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
HE_ROOT = REPO_ROOT / "HE_v_0.1"


# ============================================================
# Helpers
# ============================================================

def run_command(
    command: list[str],
    cwd: Path,
    title: str,
) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        " ".join(
            f'"{x}"' if " " in str(x) else str(x)
            for x in command
        )
    )

    print()

    subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
    )


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Preflight: CARLA plan == HE plan
# ============================================================

def check_backend_trajectory_equivalence(
    carla_plan_path: Path,
    he_scenario_path: Path,
    tolerance: float = 1e-9,
) -> None:
    carla_plan = load_json(
        carla_plan_path
    )

    he_scenario = load_json(
        he_scenario_path
    )

    carla_actors = carla_plan.get(
        "actors",
        [],
    )

    he_actors = he_scenario.get(
        "adversaries",
        [],
    )

    if len(carla_actors) != len(he_actors):
        raise RuntimeError(
            "CARLA/HE actor-count mismatch: "
            f"{len(carla_actors)} vs "
            f"{len(he_actors)}"
        )

    for actor_index, (
        carla_actor,
        he_actor,
    ) in enumerate(
        zip(
            carla_actors,
            he_actors,
        )
    ):
        carla_frames = {
            int(row["frame_idx"]): row
            for row in carla_actor["frames"]
        }

        he_frames = {
            int(row["frame_idx"]): row
            for row in (
                he_actor["motion"]["keyframes"]
            )
        }

        if set(carla_frames) != set(he_frames):
            raise RuntimeError(
                f"Actor {actor_index}: "
                "CARLA/HE frame sets differ."
            )

        for frame_idx in carla_frames:
            c = carla_frames[frame_idx]
            h = he_frames[frame_idx]

            checks = [
                (
                    "x",
                    float(c["local_x_m"]),
                    float(h["x_m"]),
                ),
                (
                    "y",
                    float(c["local_y_m"]),
                    float(h["y_m"]),
                ),
                (
                    "z",
                    float(c["local_z_m"]),
                    float(h["z_m"]),
                ),
                (
                    "yaw",
                    float(c["local_yaw_deg"]),
                    float(h["yaw_deg"]),
                ),
            ]

            for name, a, b in checks:
                if abs(a - b) > tolerance:
                    raise RuntimeError(
                        f"Backend trajectory mismatch: "
                        f"actor={actor_index}, "
                        f"frame={frame_idx}, "
                        f"field={name}, "
                        f"CARLA={a}, HE={b}"
                    )

    print(
        "[PASS] CARLA and HE consume "
        "identical actor trajectories."
    )


# ============================================================
# Real/background synchronization
# ============================================================

def read_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def check_real_background_sync(
    pair_dir: Path,
    tolerance: float = 1e-6,
) -> None:
    real_path = (
        pair_dir
        / "real_ego_pose.jsonl"
    )

    background_path = (
        pair_dir
        / "background_ego_pose.jsonl"
    )

    real = read_jsonl(
        real_path
    )

    background = read_jsonl(
        background_path
    )

    if len(real) != len(background):
        raise RuntimeError(
            "Real/background frame-count mismatch: "
            f"{len(real)} vs {len(background)}"
        )

    keys = [
        "x",
        "y",
        "z",
        "pitch",
        "yaw",
        "roll",
    ]

    print()
    print(
        "Real/background synchronization"
    )
    print(
        "-" * 40
    )

    worst = 0.0

    for key in keys:
        ego_diff = max(
            abs(
                float(r["ego_transform"][key])
                -
                float(b["ego_transform"][key])
            )
            for r, b in zip(
                real,
                background,
            )
        )

        camera_diff = max(
            abs(
                float(r["camera_transform"][key])
                -
                float(b["camera_transform"][key])
            )
            for r, b in zip(
                real,
                background,
            )
        )

        worst = max(
            worst,
            ego_diff,
            camera_diff,
        )

        print(
            f"{key:6s} "
            f"ego={ego_diff:.9f} "
            f"camera={camera_diff:.9f}"
        )

    if worst > tolerance:
        raise RuntimeError(
            "Real/background synchronization failed. "
            f"Worst difference = {worst}"
        )

    print(
        "[PASS] Real/background trajectories "
        "are synchronized."
    )


# ============================================================
# Main
# ============================================================

def main(args) -> None:

    experiment_dir = (
        PROJECT_ROOT
        / args.output_root
        / args.experiment
    ).resolve()

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment not found: "
            f"{experiment_dir}"
        )

    carla_plan_path = (
        experiment_dir
        / "carla_plan.json"
    )

    he_scenario_path = (
        experiment_dir
        / "he_scenario.json"
    )

    manifest_path = (
        experiment_dir
        / "manifest.json"
    )

    for path in [
        carla_plan_path,
        he_scenario_path,
        manifest_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact missing: {path}"
            )

    # --------------------------------------------------------
    # Preflight
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("Scenario execution")
    print("=" * 72)

    print(
        f"Experiment: {args.experiment}"
    )

    print(
        f"Directory:  {experiment_dir}"
    )

    print()

    check_backend_trajectory_equivalence(
        carla_plan_path,
        he_scenario_path,
    )

    pair_dir = (
        HE_ROOT
        / "recordings"
        / "he_pairs"
        / args.experiment
    )

    # --------------------------------------------------------
    # CARLA real adversary
    # --------------------------------------------------------

    if not args.skip_real:
        run_command(
            [
                sys.executable,
                "record_carla_from_plan.py",

                "--plan",
                str(carla_plan_path),

                "--mode",
                "real_adversary",

                "--pair-name",
                args.experiment,

                "--town",
                args.town,

                "--spawn-index",
                str(args.spawn_index),

                "--host",
                args.host,

                "--port",
                str(args.port),
            ],
            cwd=HE_ROOT,
            title=(
                "1/4 - CARLA real adversary"
            ),
        )

    # --------------------------------------------------------
    # CARLA background
    # --------------------------------------------------------

    if not args.skip_background:
        run_command(
            [
                sys.executable,
                "record_carla_from_plan.py",

                "--plan",
                str(carla_plan_path),

                "--mode",
                "background",

                "--pair-name",
                args.experiment,

                "--town",
                args.town,

                "--spawn-index",
                str(args.spawn_index),

                "--host",
                args.host,

                "--port",
                str(args.port),
            ],
            cwd=HE_ROOT,
            title=(
                "2/4 - CARLA background"
            ),
        )

    # --------------------------------------------------------
    # Synchronization check
    # --------------------------------------------------------

    if (
        not args.skip_sync_check
        and (
            pair_dir
            / "real_ego_pose.jsonl"
        ).exists()
        and (
            pair_dir
            / "background_ego_pose.jsonl"
        ).exists()
    ):
        check_real_background_sync(
            pair_dir
        )

    # --------------------------------------------------------
    # HE
    # --------------------------------------------------------

    if not args.skip_he:
        run_command(
            [
                sys.executable,
                "run_he_temporal_compositor_v1.py",

                "--scenario",
                str(he_scenario_path),

                "--overwrite",
            ],
            cwd=HE_ROOT,
            title="3/4 - HE compositor",
        )

    # --------------------------------------------------------
    # Determine HE metadata path from HE scenario
    # --------------------------------------------------------

    he_json = load_json(
        he_scenario_path
    )

    he_output_dir = Path(
        he_json["output"]["output_dir"]
    )

    if not he_output_dir.is_absolute():
        he_output_dir = (
            HE_ROOT
            / he_output_dir
        )

    he_metadata_path = (
        he_output_dir
        / "metadata.json"
    )

    real_bbox_path = (
        pair_dir
        / "real_bbox.jsonl"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if not args.skip_compare:

        if not real_bbox_path.exists():
            raise FileNotFoundError(
                f"Real bbox missing: "
                f"{real_bbox_path}"
            )

        if not he_metadata_path.exists():
            raise FileNotFoundError(
                f"HE metadata missing: "
                f"{he_metadata_path}"
            )

        run_command(
            [
                sys.executable,
                "compare_real_vs_he_bbox.py",

                "--real-bbox",
                str(real_bbox_path),

                "--he-metadata",
                str(he_metadata_path),
            ],
            cwd=HE_ROOT,
            title=(
                "4/4 - CARLA vs HE comparison"
            ),
        )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("Scenario execution complete")
    print("=" * 72)

    print(
        f"Experiment:   {args.experiment}"
    )

    print(
        f"Pair data:    {pair_dir}"
    )

    print(
        f"HE output:    {he_output_dir}"
    )

    print(
        f"HE metadata:  {he_metadata_path}"
    )

    print("=" * 72)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Execute one previously built scenario "
            "through CARLA and HE."
        )
    )

    parser.add_argument(
        "--experiment",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        default="outputs/scenarios",
    )

    parser.add_argument(
        "--town",
        default="Town10HD_Opt",
    )

    parser.add_argument(
        "--spawn-index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--skip-real",
        action="store_true",
    )

    parser.add_argument(
        "--skip-background",
        action="store_true",
    )

    parser.add_argument(
        "--skip-sync-check",
        action="store_true",
    )

    parser.add_argument(
        "--skip-he",
        action="store_true",
    )

    parser.add_argument(
        "--skip-compare",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(
        parse_args()
    )