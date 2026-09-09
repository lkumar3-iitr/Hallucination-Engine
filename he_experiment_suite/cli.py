"""Single command-line entry point for final HE experiments."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def forwarded_args(values: list[str]) -> list[str]:
    """Remove argparse's conventional forwarding separator."""
    return values[1:] if values and values[0] == "--" else values


def run(command: list[str], dry_run: bool = False) -> int:
    print(subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate the release installation")
    validate.add_argument("--asset-root", type=Path, required=True)
    validate.add_argument("--profile", choices=("release", "native-only"), default="release")

    test = sub.add_parser("test", help="run renderer and suite regression tests")
    test.add_argument("--dry-run", action="store_true")

    pair = sub.add_parser("pair", help="record one resolved CARLA or HE condition")
    pair.add_argument("--condition", choices=("carla", "he"), required=True)
    pair.add_argument("--dry-run", action="store_true")
    pair.add_argument("args", nargs=argparse.REMAINDER)

    campaign = sub.add_parser("campaign", help="run the frozen four-model campaign")
    campaign.add_argument("--dry-run", action="store_true")
    campaign.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    if args.command == "validate":
        return run(
            [sys.executable, "-m", "he_experiment_suite.validate", "--asset-root", str(args.asset_root), "--profile", args.profile]
        )
    if args.command == "test":
        commands = [
            [sys.executable, "-m", "unittest", "discover", "-s", "he_renderer/tests", "-p", "test_*.py"],
            [sys.executable, "-m", "unittest", "discover", "-s", "he_experiment_suite/tests", "-p", "test_*.py"],
        ]
        for command in commands:
            code = run(command, args.dry_run)
            if code:
                return code
        return 0
    if args.command == "pair":
        launcher = (
            "he_renderer/runners/run_pair.py"
            if args.condition == "he"
            else "driving_models/common/record_scenario_carla_he_pair_v3.py"
        )
        return run(
            [
                sys.executable,
                launcher,
                "--condition",
                args.condition,
                *forwarded_args(args.args),
            ],
            args.dry_run,
        )
    if args.command == "campaign":
        forwarded = forwarded_args(args.args)
        if args.dry_run and "--dry-run" not in forwarded:
            forwarded.append("--dry-run")
        return run(
            [sys.executable, "driving_models/common/run_paper_four_model_campaign_v1.py", *forwarded],
            args.dry_run,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
