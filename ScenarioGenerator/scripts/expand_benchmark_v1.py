from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from scenario_generator.benchmark import (
    BenchmarkSpecV1,
    expand_benchmark,
)


# ============================================================
# JSON helpers
# ============================================================

def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

        f.write(
            "\n"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Expand BenchmarkSpecV1 into deterministic "
            "BenchmarkCaseV1 instances."
        )
    )

    parser.add_argument(
        "--benchmark",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    benchmark_path = Path(
        args.benchmark
    )

    benchmark = (
        BenchmarkSpecV1
        .model_validate(
            load_json(
                benchmark_path
            )
        )
    )

    cases = (
        expand_benchmark(
            benchmark
        )
    )

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

    else:

        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / "benchmarks"
            / benchmark.benchmark_id
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Preserve benchmark source
    # --------------------------------------------------------

    save_json(
        output_dir
        / "benchmark_spec.json",

        benchmark.model_dump(
            mode="json"
        ),
    )

    # --------------------------------------------------------
    # Full expansion manifest
    # --------------------------------------------------------

    expansion = {

        "benchmark_id":
            benchmark.benchmark_id,

        "scenario_type":
            benchmark.scenario_type,

        "num_cases":
            len(
                cases
            ),

        "cases": [
            case.model_dump(
                mode="json"
            )
            for case
            in cases
        ],
    }

    save_json(
        output_dir
        / "benchmark_cases.json",

        expansion,
    )

    # --------------------------------------------------------
    # Also save one JSON per case.
    # --------------------------------------------------------

    cases_dir = (
        output_dir
        / "cases"
    )

    for case in cases:

        save_json(
            cases_dir
            / (
                case.case_id
                + ".json"
            ),

            case.model_dump(
                mode="json"
            ),
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 76
    )

    print(
        "BenchmarkSpecV1 expansion"
    )

    print(
        "=" * 76
    )

    print(
        "benchmark:",
        benchmark.benchmark_id,
    )

    print(
        "scenario type:",
        benchmark.scenario_type,
    )

    print(
        "cases:",
        len(
            cases
        ),
    )

    print(
        "output:",
        output_dir,
    )

    print()

    for case in cases:

        swept_text = ", ".join(
            f"{key}={value}"
            for (
                key,
                value,
            )
            in case.swept_values.items()
        )

        print(
            f"{case.case_index:4d}  "
            f"{case.case_id:42s} "
            f"{swept_text}"
        )

    print(
        "=" * 76
    )


if __name__ == "__main__":
    main()