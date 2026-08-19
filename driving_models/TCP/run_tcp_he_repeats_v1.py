"""
run_tcp_he_repeats_v1.py

Repeated paired TCP experiment.

Runs:
    CARLA-01 -> HE-01
    CARLA-02 -> HE-02
    ...

Each execution is a fresh Python process, therefore:
    - fresh TCP model
    - fresh TCP PID/controller state
    - fresh CARLA actors/sensors

The underlying pair runner is NOT modified.

Outputs from every run are copied into a permanent repeat folder
before the next execution overwrites the normal working outputs.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
TCP_DIR = THIS_FILE.parent
HE_ROOT = THIS_FILE.parents[2]

PAIR_SCRIPT = (
    TCP_DIR
    / "tcp_he_pair_experiment_v1.py"
)

SCENARIO_ID = None
WORKING_OUTPUT = None
REPEAT_ROOT = None


def copy_if_exists(src, dst):
    if not src.exists():
        return False

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        src,
        dst,
    )

    return True


def archive_existing_outputs():
    """
    Preserve whatever CARLA/HE result currently exists before
    the repeat experiment begins.
    """

    snapshot = (
        REPEAT_ROOT
        / "_pre_repeat_snapshot"
    )

    snapshot.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0

    for path in WORKING_OUTPUT.glob("*"):
        if not path.is_file():
            continue

        dst = (
            snapshot
            / path.name
        )

        shutil.copy2(
            path,
            dst,
        )

        copied += 1

    print(
        f"[archive] preserved {copied} existing files"
    )

    print(
        "[archive]",
        snapshot,
    )


def expected_files(condition):
    prefix = (
        f"{SCENARIO_ID}_{condition}"
    )

    return [
        f"{prefix}.csv",
        f"{prefix}_tcp_input.mp4",
        f"{prefix}_debug.mp4",
    ]


def archive_run(
    condition,
    run_idx,
    log_path,
):
    run_dir = (
        REPEAT_ROOT
        / f"run_{run_idx:02d}"
        / condition
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = []

    for filename in expected_files(
        condition
    ):
        src = (
            WORKING_OUTPUT
            / filename
        )

        dst = (
            run_dir
            / filename
        )

        if copy_if_exists(
            src,
            dst,
        ):
            copied.append(
                filename
            )

    if log_path.exists():
        shutil.copy2(
            log_path,
            run_dir
            / "console.log",
        )

    return (
        run_dir,
        copied,
    )


def run_condition(
    condition,
    run_idx,
    args,
):
    print()
    print("=" * 78)
    print(
        f"RUN {run_idx:02d}/{args.runs:02d} "
        f"CONDITION={condition.upper()}"
    )
    print("=" * 78)

    log_dir = (
        REPEAT_ROOT
        / "_logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / (
            f"run_{run_idx:02d}_"
            f"{condition}.log"
        )
    )

    command = [
        sys.executable,
        str(PAIR_SCRIPT),

        "--condition",
        condition,

        "--resolved",
        args.resolved,

        "--actor-id",
        args.actor_id,

        "--event-start-s",
        str(args.event_start_s),

        "--town",
        args.town,

        "--spawn-index",
        str(args.spawn_index),

        "--carla-pythonapi",
        args.carla_pythonapi,
    ]

    print()
    print(
        "[command]",
        subprocess.list2cmdline(
            command
        ),
    )
    print()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_fp:

        process = subprocess.Popen(
            command,
            cwd=str(HE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert (
            process.stdout
            is not None
        )

        for line in process.stdout:
            print(
                line,
                end="",
            )

            log_fp.write(
                line
            )

        return_code = (
            process.wait()
        )

    if return_code != 0:
        raise RuntimeError(
            f"{condition} run {run_idx} failed "
            f"with exit code {return_code}. "
            f"See {log_path}"
        )

    run_dir, copied = archive_run(
        condition=
            condition,

        run_idx=
            run_idx,

        log_path=
            log_path,
    )

    print()
    print(
        "[saved]",
        run_dir,
    )

    for name in copied:
        print(
            "        ",
            name,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--scenario-id",
        required=True,
    )

    parser.add_argument(
        "--resolved",
        required=True,
    )

    parser.add_argument(
        "--actor-id",
        required=True,
    )

    parser.add_argument(
        "--event-start-s",
        type=float,
        default=4.0,
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
        "--carla-pythonapi",
        default=(
            r"E:\Carla\Carla_0.9.15"
            r"\PythonAPI\carla"
        ),
    )

    parser.add_argument(
        "--no-snapshot",
        action="store_true",
    )

    args = parser.parse_args()
    global SCENARIO_ID
    global WORKING_OUTPUT
    global REPEAT_ROOT

    SCENARIO_ID = (
        args.scenario_id
    )

    WORKING_OUTPUT = (
        TCP_DIR
        / "outputs"
        / "tcp_he_pair_v1"
        / SCENARIO_ID
    )

    REPEAT_ROOT = (
        TCP_DIR
        / "outputs"
        / "tcp_he_repeat_v1"
        / SCENARIO_ID
    )

    if args.runs < 1:
        raise ValueError(
            "--runs must be >= 1"
        )

    if not PAIR_SCRIPT.exists():
        raise FileNotFoundError(
            PAIR_SCRIPT
        )

    REPEAT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("TCP CARLA / HE REPEATABILITY EXPERIMENT")
    print("=" * 78)

    print(
        "runs per condition:",
        args.runs,
    )

    print(
        "scenario:",
        SCENARIO_ID,
    )

    print(
        "output:",
        REPEAT_ROOT,
    )

    if not args.no_snapshot:
        archive_existing_outputs()

    for run_idx in range(
        1,
        args.runs + 1,
    ):
        # Interleave conditions deliberately.
        #
        # This is preferable to CARLA x5 followed by HE x5,
        # because machine/environment drift cannot systematically
        # affect only one condition.

        run_condition(
            condition="carla",
            run_idx=run_idx,
            args=args,
        )

        run_condition(
            condition="he",
            run_idx=run_idx,
            args=args,
        )

    print()
    print("=" * 78)
    print("REPEAT EXPERIMENT COMPLETE")
    print("=" * 78)

    print(
        "results:",
        REPEAT_ROOT,
    )


if __name__ == "__main__":
    main()