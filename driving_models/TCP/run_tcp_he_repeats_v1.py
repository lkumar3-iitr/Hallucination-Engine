"""
run_tcp_he_repeats_v1.py

Repeated fresh-process TCP CARLA <-> HE experiments.

Each CARLA or HE condition is executed in a completely fresh
Python process.

Execution order alternates:

    pair 01: CARLA -> HE
    pair 02: HE -> CARLA
    pair 03: CARLA -> HE
    ...

This reduces systematic condition-order bias.

After every pair:
    analyze_tcp_cutin_pair_v1.py

After all pairs:
    analyze_tcp_he_repeats_v1.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]


RUNNER = (
    HE_ROOT
    / "driving_models"
    / "TCP"
    / "tcp_he_pair_experiment_v1.py"
)


PAIR_ANALYZER = (
    HE_ROOT
    / "driving_models"
    / "TCP"
    / "analyze_tcp_cutin_pair_v1.py"
)


REPEAT_ANALYZER = (
    HE_ROOT
    / "driving_models"
    / "TCP"
    / "analyze_tcp_he_repeats_v1.py"
)


DEFAULT_RESOLVED = (
    HE_ROOT
    / "ScenarioGenerator"
    / "outputs"
    / "v2_resolved"
    / "tcp_cutin_001.resolved_v2.json"
)


DEFAULT_OUTPUT_ROOT = (
    HE_ROOT
    / "driving_models"
    / "TCP"
    / "outputs"
    / "tcp_4320_repeats_v1"
)


DEFAULT_VIEW_MATRIX = Path(
    r"D:\HE_Data"
    r"\sprite_bank_grabcut"
    r"\tesla_grabcut_view_matrix_full"
    r"\view_matrix.csv"
)


def run_command(
    cmd,
):

    print()
    print("=" * 88)
    print(
        " ".join(
            str(x)
            for x in cmd
        )
    )
    print("=" * 88)
    print()

    result = subprocess.run(
        cmd,
        cwd=str(
            HE_ROOT
        ),
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Command failed with "
            f"return code {result.returncode}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--start-pair",
        type=int,
        default=1,
        help=(
            "First repeat pair to run. "
            "Useful for resuming an interrupted experiment."
        ),
    )
    parser.add_argument(
        "--resolved",
        default=str(
            DEFAULT_RESOLVED
        ),
    )

    parser.add_argument(
        "--actor-id",
        default="adv_cutin",
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
        "--event-start-s",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--carla-pythonapi",
        default=(
            r"E:\Carla"
            r"\Carla_0.9.15"
            r"\PythonAPI"
            r"\carla"
        ),
    )

    parser.add_argument(
        "--view-matrix-csv",
        default=str(
            DEFAULT_VIEW_MATRIX
        ),
    )

    parser.add_argument(
        "--distance-selection-mode",
        choices=[
            "linear",
            "log",
            "inverse_depth",
        ],
        default="linear",
    )
    parser.add_argument(
        "--he-bottom-y-offset-px",
        type=float,
        default=0.0,
        help=(
            "HE native-camera vertical placement correction. "
            "Negative moves the HE actor upward."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    args = parser.parse_args()
    if (
        args.start_pair < 1
        or
        args.start_pair > args.repeats
    ):
        raise ValueError(
            "--start-pair must be between "
            "1 and --repeats"
        )
    if args.repeats <= 0:

        raise ValueError(
            "--repeats must be > 0"
        )

    resolved_path = Path(
        args.resolved
    ).resolve()

    view_matrix_path = Path(
        args.view_matrix_csv
    ).resolve()

    output_root = Path(
        args.output_root
    ).resolve()

    if not resolved_path.exists():

        raise FileNotFoundError(
            resolved_path
        )

    if not view_matrix_path.exists():

        raise FileNotFoundError(
            view_matrix_path
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as fp:
        resolved_data = json.load(
            fp
        )

    scenario_id = str(
        resolved_data[
            "scenario_id"
        ]
    )

    manifest = {

        "schema":
            "tcp_he_repeat_manifest_v1",

        "scenario_id":
            scenario_id,

        "repeats":
            args.repeats,

        "resolved":
            str(
                resolved_path
            ),

        "actor_id":
            args.actor_id,

        "town":
            args.town,

        "spawn_index":
            args.spawn_index,

        "event_start_s":
            args.event_start_s,

        "view_matrix_csv":
            str(
                view_matrix_path
            ),

        "distance_selection_mode":
            args.distance_selection_mode,

        "python":
            sys.executable,

        "pairs":
            [],
    }

    print()
    print("=" * 88)
    print(
        "TCP CARLA <-> HE REPEATED EXPERIMENT V1"
    )
    print("=" * 88)

    print(
        "scenario :",
        scenario_id,
    )

    print(
        "repeats  :",
        args.repeats,
    )

    print(
        "python   :",
        sys.executable,
    )

    print(
        "sprites  :",
        view_matrix_path,
    )

    print(
        "output   :",
        output_root,
    )

    print("=" * 88)

    # ========================================================
    # Repeated pairs
    # ========================================================

    for pair_idx in range(
        args.start_pair,
        args.repeats + 1,
    ):

        pair_name = (
            f"pair_{pair_idx:02d}"
        )

        pair_root = (
            output_root
            / pair_name
        )

        # Alternate condition order.
        if pair_idx % 2 == 1:

            conditions = [
                "carla",
                "he",
            ]

        else:

            conditions = [
                "he",
                "carla",
            ]

        pair_manifest = {

            "pair_idx":
                pair_idx,

            "pair_name":
                pair_name,

            "condition_order":
                conditions,

            "pair_root":
                str(
                    pair_root
                ),
        }

        manifest[
            "pairs"
        ].append(
            pair_manifest
        )

        pair_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            pair_root
            / "pair_manifest.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                pair_manifest,
                fp,
                indent=2,
            )

        print()
        print("#" * 88)

        print(
            f"PAIR {pair_idx}/{args.repeats}"
        )

        print(
            "order:",
            " -> ".join(
                conditions
            ),
        )

        print("#" * 88)

        # ====================================================
        # CARLA / HE runs
        # ====================================================

        for condition in conditions:

            cmd = [

                sys.executable,

                str(
                    RUNNER
                ),

                "--condition",
                condition,

                "--resolved",
                str(
                    resolved_path
                ),

                "--actor-id",
                args.actor_id,

                "--town",
                args.town,

                "--spawn-index",
                str(
                    args.spawn_index
                ),

                "--event-start-s",
                str(
                    args.event_start_s
                ),

                "--carla-pythonapi",
                args.carla_pythonapi,

                "--output-root",
                str(
                    pair_root
                ),
            ]

            # Only HE needs sprite-bank arguments.
            if condition == "he":

                cmd.extend([
                    "--view-matrix-csv",
                    str(
                        view_matrix_path
                    ),

                    "--distance-selection-mode",
                    args.distance_selection_mode,

                    "--he-bottom-y-offset-px",
                    str(
                        args.he_bottom_y_offset_px
                    ),
                ])

            run_command(
                cmd
            )

        # ====================================================
        # Analyze this CARLA <-> HE pair
        # ====================================================

        scenario_root = (
            pair_root
            / scenario_id
        )

        carla_csv = (
            scenario_root
            / (
                scenario_id
                + "_carla.csv"
            )
        )

        he_csv = (
            scenario_root
            / (
                scenario_id
                + "_he.csv"
            )
        )

        analysis_dir = (
            scenario_root
            / "analysis"
        )

        cmd = [

            sys.executable,

            str(
                PAIR_ANALYZER
            ),

            "--carla",
            str(
                carla_csv
            ),

            "--he",
            str(
                he_csv
            ),

            "--output-dir",
            str(
                analysis_dir
            ),
        ]

        run_command(
            cmd
        )

    # ========================================================
    # Save repeat manifest
    # ========================================================

    manifest_path = (
        output_root
        / "repeat_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            manifest,
            fp,
            indent=2,
        )

    # ========================================================
    # Aggregate all pairs
    # ========================================================

    cmd = [

        sys.executable,

        str(
            REPEAT_ANALYZER
        ),

        "--root",
        str(
            output_root
        ),

        "--scenario",
        scenario_id,

        "--repeats",
        str(
            args.repeats
        ),
    ]

    run_command(
        cmd
    )

    print()
    print("=" * 88)
    print(
        "ALL TCP CARLA <-> HE PAIRS COMPLETE"
    )
    print("=" * 88)

    print(
        "root:",
        output_root,
    )

    print(
        "manifest:",
        manifest_path,
    )

    print("=" * 88)


if __name__ == "__main__":

    main()