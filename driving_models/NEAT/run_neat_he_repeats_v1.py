"""
run_neat_he_repeats_v1.py

Run repeated fresh-process NEAT CARLA <-> HE lead-brake pairs.

Each condition is launched as a completely separate Python process.
Odd pairs run CARLA first; even pairs run HE first to avoid a fixed
condition-order bias.

After each pair, the existing single-pair analyzer is executed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]

RUNNER = (
    HE_ROOT
    / "driving_models"
    / "NEAT"
    / "neat_he_pair_experiment_v1.py"
)

ANALYZER = (
    HE_ROOT
    / "driving_models"
    / "NEAT"
    / "analyze_neat_lead_brake_pair_v1.py"
)

DEFAULT_RESOLVED = (
    HE_ROOT
    / "ScenarioGenerator"
    / "outputs"
    / "v2_resolved"
    / "tcp_lead_brake_001.resolved_v2.json"
)

DEFAULT_OUTPUT_ROOT = (
    HE_ROOT
    / "driving_models"
    / "NEAT"
    / "outputs"
    / "neat_he_repeat_v1"
)


def run_command(cmd):

    print()
    print("=" * 78)
    print(" ".join(str(x) for x in cmd))
    print("=" * 78)
    print()

    result = subprocess.run(
        cmd,
        cwd=str(HE_ROOT),
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Command failed with "
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
        "--resolved",
        default=str(
            DEFAULT_RESOLVED
        ),
    )

    parser.add_argument(
        "--actor-id",
        default="adv_lead",
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
            r"E:\Carla\Carla_0.9.15"
            r"\PythonAPI\carla"
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--debug-every",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    repeat_root = Path(
        args.output_root
    ).resolve()

    repeat_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenario_id = (
        Path(
            args.resolved
        )
        .name
        .replace(
            ".resolved_v2.json",
            "",
        )
    )

    print()
    print("=" * 78)
    print("NEAT CARLA <-> HE REPEATED EXPERIMENT")
    print("=" * 78)
    print("scenario :", scenario_id)
    print("repeats  :", args.repeats)
    print("python   :", sys.executable)
    print("output   :", repeat_root)
    print("=" * 78)

    for repeat_idx in range(
        1,
        args.repeats + 1,
    ):

        pair_name = (
            f"pair_{repeat_idx:02d}"
        )

        pair_root = (
            repeat_root
            / pair_name
        )

        # Alternate execution order.
        if repeat_idx % 2 == 1:

            conditions = [
                "carla",
                "he",
            ]

        else:

            conditions = [
                "he",
                "carla",
            ]

        print()
        print("#" * 78)
        print(
            f"PAIR {repeat_idx}/{args.repeats}"
        )
        print(
            "order:",
            " -> ".join(
                conditions
            ),
        )
        print("#" * 78)

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
                    Path(
                        args.resolved
                    ).resolve()
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

                "--debug-every",
                str(
                    args.debug_every
                ),

                "--carla-pythonapi",
                args.carla_pythonapi,

                "--output-root",
                str(
                    pair_root
                ),
            ]

            run_command(
                cmd
            )

        scenario_root = (
            pair_root
            / scenario_id
        )

        carla_csv = (
            scenario_root
            / "carla"
            / (
                scenario_id
                + "_carla.csv"
            )
        )

        he_csv = (
            scenario_root
            / "he"
            / (
                scenario_id
                + "_he.csv"
            )
        )

        analysis_path = (
            scenario_root
            / "analysis"
            / (
                scenario_id
                + "_pair_summary.json"
            )
        )

        cmd = [
            sys.executable,
            str(
                ANALYZER
            ),

            "--carla",
            str(
                carla_csv
            ),

            "--he",
            str(
                he_csv
            ),

            "--event-start-s",
            str(
                args.event_start_s
            ),

            "--output",
            str(
                analysis_path
            ),
        ]

        run_command(
            cmd
        )

    print()
    print("=" * 78)
    print("ALL NEAT PAIRS COMPLETE")
    print("=" * 78)
    print(
        "output:",
        repeat_root
    )
    print("=" * 78)


if __name__ == "__main__":

    main()