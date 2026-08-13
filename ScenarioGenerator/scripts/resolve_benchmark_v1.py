"""
resolve_benchmark_v1.py

Milestone 4B driver.

Consumes the BenchmarkCaseV1 manifest produced by M4A.

For each case:

    BenchmarkCaseV1
        ↓
    ScenarioSpecV2
        ↓
    ResolvedScenarioV2
        ↓
    HE capability analysis

No CARLA or HE execution occurs.
"""

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
    BenchmarkCaseV1,
    resolve_benchmark_case,
)


# ============================================================
# IO
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
            allow_nan=False,
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
            "Resolve expanded benchmark cases into "
            "ScenarioSpecV2 / ResolvedScenarioV2 and "
            "perform HE capability preflight."
        )
    )

    parser.add_argument(
        "--benchmark-cases",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.benchmark_cases
    )

    manifest = load_json(
        manifest_path
    )

    benchmark_id = str(
        manifest[
            "benchmark_id"
        ]
    )

    raw_cases = manifest.get(
        "cases",
        [],
    )

    cases = [
        BenchmarkCaseV1
        .model_validate(
            case
        )

        for case
        in raw_cases
    ]

    if args.output_dir:

        output_root = Path(
            args.output_dir
        )

    else:

        output_root = (
            manifest_path.parent
            / "m4b"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Run every case
    # ========================================================

    results = []

    num_supported = 0
    num_ood = 0
    num_errors = 0

    print()
    print("=" * 92)
    print("M4B BENCHMARK RESOLUTION + HE CAPABILITY PREFLIGHT")
    print("=" * 92)

    print(
        "benchmark:",
        benchmark_id,
    )

    print(
        "cases:",
        len(cases),
    )

    print(
        "output:",
        output_root,
    )

    print()

    for case in cases:

        case_dir = (
            output_root
            / case.case_id
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            (
                scenario,
                resolved,
                capability,
            ) = resolve_benchmark_case(
                case
            )

            # ------------------------------------------------
            # Save individual stage outputs.
            # ------------------------------------------------

            save_json(
                case_dir
                / "benchmark_case.json",

                case.model_dump(
                    mode="json"
                ),
            )

            save_json(
                case_dir
                / "scenario_spec_v2.json",

                scenario.model_dump(
                    mode="json"
                ),
            )

            save_json(
                case_dir
                / "resolved_scenario_v2.json",

                resolved.model_dump(
                    mode="json"
                ),
            )

            save_json(
                case_dir
                / "he_capability.json",

                capability,
            )

            status = capability[
                "status"
            ]

            if status == "HE_SUPPORTED":

                num_supported += 1

            else:

                num_ood += 1

            actor_summary_parts = []

            for (
                actor_id,
                actor_report,
            ) in capability[
                "actors"
            ].items():

                actor_summary_parts.append(
                    (
                        f"{actor_id}: "
                        f"{actor_report['supported_placement_frames']}"
                        f"/"
                        f"{actor_report['placement_required_frames']}"
                        " placement-supported, "
                        f"{actor_report['culled_frames']} culled, "
                        f"{actor_report['out_of_domain_frames']} OOD"
                    )
                )

            actor_text = "; ".join(
                actor_summary_parts
            )

            print(
                f"{case.case_index:4d}  "
                f"{case.case_id:42s} "
                f"{status:13s} "
                f"{actor_text}"
            )

            results.append(
                {
                    "case_index":
                        case.case_index,

                    "case_id":
                        case.case_id,

                    "scenario_type":
                        case.scenario_type,

                    "swept_values":
                        case.swept_values,

                    "status":
                        status,

                    "error":
                        None,

                    "total_actor_frames":
                        capability[
                            "total_actor_frames"
                        ],

                    "placement_required_frames":
                        capability[
                            "placement_required_frames"
                        ],

                    "supported_placement_frames":
                        capability[
                            "supported_placement_frames"
                        ],

                    "culled_frames":
                        capability[
                            "culled_frames"
                        ],

                    "out_of_domain_frames":
                        capability[
                            "out_of_domain_frames"
                        ],

                    "actors":
                        capability[
                            "actors"
                        ],
                }
            )

        except Exception as exc:

            num_errors += 1

            error_text = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"{case.case_index:4d}  "
                f"{case.case_id:42s} "
                f"{'ERROR':13s} "
                f"{error_text}"
            )

            results.append(
                {
                    "case_index":
                        case.case_index,

                    "case_id":
                        case.case_id,

                    "scenario_type":
                        case.scenario_type,

                    "swept_values":
                        case.swept_values,

                    "status":
                        "ERROR",

                    "error":
                        error_text,
                }
            )

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "benchmark_id":
            benchmark_id,

        "num_cases":
            len(cases),

        "num_he_supported":
            num_supported,

        "num_he_ood":
            num_ood,

        "num_errors":
            num_errors,

        "cases":
            results,
    }

    save_json(
        output_root
        / "m4b_summary.json",

        summary,
    )

    print()
    print("-" * 92)

    print(
        "Cases:        ",
        len(cases),
    )

    print(
        "HE supported: ",
        num_supported,
    )

    print(
        "HE OOD:       ",
        num_ood,
    )

    print(
        "Errors:       ",
        num_errors,
    )

    print()

    print(
        "Summary:",
        output_root
        / "m4b_summary.json",
    )

    print("=" * 92)


if __name__ == "__main__":
    main()