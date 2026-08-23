#!/usr/bin/env python3
"""
run_all_native_asset_banks_v1.py

Sequential production launcher for the Hallucination Engine CARLA-native
sprite banks.

Purpose
-------
Run every production asset bank from one command while preserving the
per-asset resume/validation logic implemented by:

    generate_carla_asset_view_matrix_native_mask_v3.py

Production banks
----------------
1. Tesla Model 3
   - class: vehicle
   - blueprint: vehicle.tesla.model3
   - full dataset resolved by generator:
       360 angles x [5,10,20] m x [0,5,10,20] deg = 4320 views

2. Nissan Patrol 2021
   - class: vehicle
   - blueprint: vehicle.nissan.patrol_2021
   - 4320 views

3. Mitsubishi Fuso Rosa
   - class: bus
   - blueprint: vehicle.mitsubishi.fusorosa
   - full dataset resolved by generator:
       360 angles x [10,15,20,25] m x [0,5,10,20] deg = 5760 views

4. Pedestrian 0001
   - class: pedestrian
   - blueprint: walker.pedestrian.0001
   - 4320 views

Total: 18,720 views.

Behavior
--------
- Runs one asset at a time in the same CARLA server process.
- Invokes the production generator as a subprocess.
- Passes --full-dataset and --resume to every asset.
- Writes one persistent log per asset.
- Stops on a real generator failure instead of silently continuing.
- Re-running this launcher resumes existing asset banks.
- Already completed views are skipped by the production generator.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ASSETS = [
    {
        "asset_id": "tesla_model3_native_full_v3",
        "asset_class": "vehicle",
        "blueprint": "vehicle.tesla.model3",
        "color": "0,0,255",
        "expected_views": 4320,
    },
    {
        "asset_id": "nissan_patrol_2021_native_full_v3",
        "asset_class": "vehicle",
        "blueprint": "vehicle.nissan.patrol_2021",
        "color": "0,0,255",
        "expected_views": 4320,
    },
    {
        "asset_id": "fuso_rosa_bus_native_full_v3",
        "asset_class": "bus",
        "blueprint": "vehicle.mitsubishi.fusorosa",
        "color": None,
        "expected_views": 5760,
    },
    {
        "asset_id": "pedestrian_0001_native_full_v3",
        "asset_class": "pedestrian",
        "blueprint": "walker.pedestrian.0001",
        "color": None,
        "expected_views": 4320,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generator",
        default="generate_carla_asset_view_matrix_native_mask_v3.py",
        help="Production asset-bank generator.",
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch the generator.",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="CARLA RPC port. Server production default is 30000.",
    )

    parser.add_argument(
        "--output-root",
        default="assets/sprite_bank_native_production",
    )

    parser.add_argument(
        "--log-root",
        default="logs/native_asset_generation",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--between-assets-seconds",
        type=float,
        default=5.0,
        help="Short pause after one generator exits before launching the next.",
    )

    parser.add_argument(
        "--start-at",
        choices=[a["asset_id"] for a in ASSETS],
        default=None,
        help=(
            "Optional recovery convenience. Skip launcher entries before this "
            "asset. Normal generator --resume still applies."
        ),
    )

    return parser.parse_args()


def print_banner(args):
    print()
    print("=" * 88)
    print("HE CARLA-NATIVE PRODUCTION ASSET LAUNCHER")
    print("=" * 88)
    print("generator   :", args.generator)
    print("python      :", args.python)
    print("CARLA       : {}:{}".format(args.host, args.port))
    print("output root :", args.output_root)
    print("log root    :", args.log_root)
    print()
    print("Production plan:")

    total = 0

    for index, asset in enumerate(ASSETS, start=1):
        total += int(asset["expected_views"])
        print(
            "  {}. {:36s} {:5d} views  {}".format(
                index,
                asset["asset_id"],
                asset["expected_views"],
                asset["blueprint"],
            )
        )

    print()
    print("Total expected views:", total)
    print("=" * 88)
    print()


def build_command(args, asset):
    command = [
        str(args.python),
        str(args.generator),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--timeout",
        str(args.timeout),
        "--asset-id",
        str(asset["asset_id"]),
        "--asset-class",
        str(asset["asset_class"]),
        "--actor-blueprint",
        str(asset["blueprint"]),
        "--semantic-tag",
        "auto",
        "--output-root",
        str(args.output_root),
        "--full-dataset",
        "--resume",
    ]

    if asset["color"] is not None:
        command.extend(
            [
                "--color",
                str(asset["color"]),
            ]
        )

    return command


def run_asset(args, asset, index, total_assets, log_root):
    command = build_command(args, asset)

    log_path = (
        log_root
        /
        "{}.log".format(
            asset["asset_id"]
        )
    )

    print()
    print("=" * 88)
    print(
        "[Launcher] ASSET {}/{}: {}".format(
            index,
            total_assets,
            asset["asset_id"],
        )
    )
    print("[Launcher] blueprint     :", asset["blueprint"])
    print("[Launcher] expected views:", asset["expected_views"])
    print("[Launcher] log           :", log_path)
    print("[Launcher] command       :")
    print(" ".join(command))
    print("=" * 88)
    print()

    start_time = time.time()

    with log_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as log_file:

        log_file.write("\n")
        log_file.write("=" * 88 + "\n")
        log_file.write(
            "START {}\n".format(
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )
        log_file.write(
            "COMMAND {}\n".format(
                " ".join(command)
            )
        )
        log_file.write("=" * 88 + "\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert process.stdout is not None

        try:
            for line in process.stdout:
                print(
                    line,
                    end="",
                    flush=True,
                )
                log_file.write(
                    line
                )

        except KeyboardInterrupt:
            print()
            print(
                "[Launcher] Keyboard interrupt received; "
                "terminating current generator..."
            )

            process.terminate()

            try:
                process.wait(
                    timeout=15.0
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            raise

        return_code = process.wait()

        elapsed = time.time() - start_time

        log_file.write("\n")
        log_file.write(
            "END {} return_code={} elapsed_seconds={:.1f}\n".format(
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                return_code,
                elapsed,
            )
        )
        log_file.flush()

    if return_code != 0:
        raise RuntimeError(
            "Asset generator failed for {} with return code {}. "
            "Fix/restart CARLA if necessary, then run this same launcher "
            "again; --resume will keep completed views.".format(
                asset["asset_id"],
                return_code,
            )
        )

    print()
    print(
        "[Launcher] COMPLETE: {} in {:.1f} min".format(
            asset["asset_id"],
            elapsed / 60.0,
        )
    )


def main():
    args = parse_args()

    generator_path = Path(
        args.generator
    )

    if not generator_path.is_file():
        raise FileNotFoundError(
            "Generator not found: {}".format(
                generator_path
            )
        )

    output_root = Path(
        args.output_root
    )

    log_root = Path(
        args.log_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_banner(
        args
    )

    assets_to_run = list(
        ASSETS
    )

    if args.start_at is not None:
        start_index = next(
            index
            for index, asset in enumerate(
                assets_to_run
            )
            if asset["asset_id"]
            ==
            args.start_at
        )

        assets_to_run = assets_to_run[
            start_index:
        ]

        print(
            "[Launcher] --start-at active:",
            args.start_at,
        )

    total_assets = len(
        assets_to_run
    )

    launcher_start = time.time()

    for index, asset in enumerate(
        assets_to_run,
        start=1,
    ):
        run_asset(
            args=args,
            asset=asset,
            index=index,
            total_assets=total_assets,
            log_root=log_root,
        )

        if (
            index
            <
            total_assets
            and
            args.between_assets_seconds > 0.0
        ):
            print(
                "[Launcher] Waiting {:.1f}s before next asset...".format(
                    args.between_assets_seconds
                )
            )

            time.sleep(
                args.between_assets_seconds
            )

    elapsed = time.time() - launcher_start

    print()
    print("=" * 88)
    print("ALL PRODUCTION ASSET BANKS COMPLETED")
    print("=" * 88)
    print(
        "Elapsed: {:.2f} hours".format(
            elapsed / 3600.0
        )
    )
    print(
        "Output :",
        output_root,
    )
    print(
        "Logs   :",
        log_root,
    )
    print("=" * 88)
    print()


if __name__ == "__main__":
    main()
