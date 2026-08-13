"""
aggregate_benchmark_results_v1.py

Milestone M4E.

Aggregate a completed controlled benchmark into:

    benchmark_results.csv
    benchmark_results.json
    benchmark_summary.json

Inputs are existing artifacts only:

    benchmark_cases.json
    m4b/<case>/he_capability.json
    m4d/<case>/execution_report.json
    m4d/<case>/logs/04_compare_depth_*.txt

No CARLA execution.
No HE rendering.
No model tuning.

Important:
    M4D PASS means execution success.
    M4E reports geometric measurements separately.
    No geometric PASS/FAIL threshold is imposed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


# ============================================================
# IO
# ============================================================

def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


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
        f.write("\n")


# ============================================================
# Small helpers
# ============================================================

def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def safe_int(value):
    if value is None:
        return None

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def first_existing(
    dictionary: dict,
    *keys,
    default=None,
):
    for key in keys:
        if key in dictionary:
            return dictionary[key]

    return default


# ============================================================
# Benchmark cases
# ============================================================

def load_benchmark_cases(
    benchmark_root: Path,
):
    path = (
        benchmark_root
        / "benchmark_cases.json"
    )

    data = load_json(path)

    # Support either:
    #
    #   [case, case, ...]
    #
    # or:
    #
    #   {"cases": [...]}
    #
    if isinstance(
        data,
        list,
    ):
        cases = data

    elif isinstance(
        data,
        dict,
    ):
        cases = data.get(
            "cases",
            []
        )

    else:
        raise ValueError(
            "Unsupported benchmark_cases.json structure."
        )

    return {
        str(case["case_id"]):
            case
        for case in cases
    }


# ============================================================
# Comparator log parser
# ============================================================

METRIC_PATTERNS = {

    "rows_evaluated":
        r"Rows evaluated:\s+(\d+)",

    "both_present":
        r"Both present:\s+(\d+)",

    "real_only_present":
        r"Real-only present:\s+(\d+)",

    "he_only_present":
        r"HE-only present:\s+(\d+)",

    "presence_agreement":
        r"Presence agreement:\s+([-+0-9.eE]+)",

    "visibility_comparable":
        r"Visibility comparable:\s+(\d+)",

    "visibility_agreement":
        r"Visibility agreement:\s+([-+0-9.eE]+)",

    "real_visible_he_not":
        r"Real visible / HE not:\s+(\d+)",

    "he_visible_real_not":
        r"HE visible / Real not:\s+(\d+)",

    "both_visible_bbox_frames":
        r"Both visible for bbox:\s+(\d+)",

    "min_real_depth_m":
        r"Min real depth filter:\s+([-+0-9.eE]+)\s*m",

    "mae_cx_px":
        r"MAE cx px:\s+([-+0-9.eE]+)",

    "mae_bottom_y_px":
        r"MAE bottom_y px:\s+([-+0-9.eE]+)",

    "mae_width_px":
        r"MAE width px:\s+([-+0-9.eE]+)",

    "mae_height_px":
        r"MAE height px:\s+([-+0-9.eE]+)",

    "signed_mean_cx_px":
        r"Signed mean cx px:\s+([-+0-9.eE]+)",

    "signed_mean_bottom_y_px":
        r"Signed mean bottom_y px:\s+([-+0-9.eE]+)",

    "signed_mean_width_px":
        r"Signed mean width px:\s+([-+0-9.eE]+)",

    "signed_mean_height_px":
        r"Signed mean height px:\s+([-+0-9.eE]+)",
}


INTEGER_METRICS = {
    "rows_evaluated",
    "both_present",
    "real_only_present",
    "he_only_present",
    "visibility_comparable",
    "real_visible_he_not",
    "he_visible_real_not",
    "both_visible_bbox_frames",
}


def parse_comparison_log(
    path: Path,
):
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    result = {}

    for (
        name,
        pattern,
    ) in METRIC_PATTERNS.items():

        matches = re.findall(
            pattern,
            text,
        )

        if not matches:
            result[name] = None
            continue

        # For a one-actor benchmark this is the actor block.
        # Using the final match also works if a script prints
        # an aggregate block afterward.
        value = matches[-1]

        if name in INTEGER_METRICS:
            result[name] = int(
                value
            )
        else:
            result[name] = float(
                value
            )

    return result


# ============================================================
# Find comparator log
# ============================================================

def find_comparison_log(
    case_m4d_dir: Path,
):
    logs_dir = (
        case_m4d_dir
        / "logs"
    )

    candidates = sorted(
        logs_dir.glob(
            "04_compare_depth_*.txt"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "No geometric comparison log found in "
            f"{logs_dir}"
        )

    if len(candidates) > 1:
        print(
            "[WARN]",
            case_m4d_dir.name,
            "has multiple comparison logs; using:",
            candidates[-1].name,
        )

    return candidates[-1]


# ============================================================
# M4B capability extraction
# ============================================================

def get_actor_capability(
    capability: dict,
):
    """
    Current benchmark has one actor.

    This is intentionally tolerant of either:
        "actors": [...]
    or:
        "actors": {"adv_001": {...}}
    """

    actors = capability.get(
        "actors",
        {}
    )

    if isinstance(
        actors,
        dict,
    ):
        if not actors:
            return {}

        first_key = sorted(
            actors.keys()
        )[0]

        return actors[
            first_key
        ]

    if isinstance(
        actors,
        list,
    ):
        if not actors:
            return {}

        return actors[0]

    return {}


def extract_capability_metrics(
    capability: dict,
):
    actor = get_actor_capability(
        capability
    )

    return {

        "he_capability_status":
            first_existing(
                capability,
                "status",
                "he_status",
                default=None,
            ),

        "active_frames":
            safe_int(
                first_existing(
                    actor,
                    "active_frames",
                    default=None,
                )
            ),

        "placement_required_frames":
            safe_int(
                first_existing(
                    actor,
                    "placement_required_frames",
                    default=None,
                )
            ),

        "supported_placement_frames":
            safe_int(
                first_existing(
                    actor,
                    "supported_placement_frames",
                    "placement_supported_frames",
                    default=None,
                )
            ),

        "culled_frames":
            safe_int(
                first_existing(
                    actor,
                    "culled_frames",
                    default=None,
                )
            ),

        "out_of_domain_frames":
            safe_int(
                first_existing(
                    actor,
                    "out_of_domain_frames",
                    "ood_frames",
                    default=None,
                )
            ),
    }


# ============================================================
# Case parameters
# ============================================================

def extract_case_parameters(
    case: dict,
):
    p = case.get(
        "parameters",
        {}
    )

    swept = case.get(
        "swept_values",
        {}
    )

    return {

        "scenario_type":
            case.get(
                "scenario_type"
            ),

        "side":
            p.get(
                "side"
            ),

        "start_distance_m":
            safe_float(
                p.get(
                    "start_distance_m"
                )
            ),

        "lane_y_m":
            safe_float(
                p.get(
                    "lane_y_m"
                )
            ),

        "target_lane_y_m":
            safe_float(
                p.get(
                    "target_lane_y_m"
                )
            ),

        "lane_width_m":
            safe_float(
                p.get(
                    "lane_width_m"
                )
            ),

        "actor_speed_mps":
            safe_float(
                p.get(
                    "actor_speed_mps"
                )
            ),

        "ego_speed_mps":
            safe_float(
                p.get(
                    "ego_speed_mps"
                )
            ),

        "duration_s":
            safe_float(
                p.get(
                    "duration_s"
                )
            ),

        "fps":
            safe_int(
                p.get(
                    "fps"
                )
            ),

        "cut_start_s":
            safe_float(
                p.get(
                    "cut_start_s"
                )
            ),

        "cut_duration_s":
            safe_float(
                p.get(
                    "cut_duration_s"
                )
            ),

        # Useful traceability of which parameters were actually
        # varied by the benchmark Cartesian expansion.
        "swept_values":
            swept,
    }


# ============================================================
# Aggregate statistics
# ============================================================

def valid_numeric_values(
    rows,
    field,
):
    values = []

    for row in rows:
        value = row.get(
            field
        )

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            if math.isfinite(
                float(value)
            ):
                values.append(
                    float(value)
                )

    return values


def summarize_numeric_field(
    rows,
    field,
):
    values = valid_numeric_values(
        rows,
        field,
    )

    if not values:
        return None

    return {
        "mean":
            mean(
                values
            ),

        "min":
            min(
                values
            ),

        "max":
            max(
                values
            ),
    }


def worst_case(
    rows,
    field,
):
    candidates = [
        row
        for row
        in rows
        if isinstance(
            row.get(field),
            (
                int,
                float,
            ),
        )
    ]

    if not candidates:
        return None

    row = max(
        candidates,
        key=lambda item:
            float(
                item[field]
            ),
    )

    return {
        "case_id":
            row[
                "case_id"
            ],

        "value":
            float(
                row[
                    field
                ]
            ),
    }


# ============================================================
# CSV
# ============================================================

CSV_FIELDS = [

    "case_id",
    "scenario_type",

    "side",
    "start_distance_m",
    "lane_y_m",
    "target_lane_y_m",
    "lane_width_m",

    "actor_speed_mps",
    "ego_speed_mps",

    "duration_s",
    "fps",

    "cut_start_s",
    "cut_duration_s",

    "execution_status",
    "he_capability_status",

    "active_frames",
    "placement_required_frames",
    "supported_placement_frames",
    "culled_frames",
    "out_of_domain_frames",

    "rows_evaluated",

    "both_present",
    "real_only_present",
    "he_only_present",
    "presence_agreement",

    "visibility_comparable",
    "visibility_agreement",
    "real_visible_he_not",
    "he_visible_real_not",

    "both_visible_bbox_frames",
    "min_real_depth_m",

    "mae_cx_px",
    "mae_bottom_y_px",
    "mae_width_px",
    "mae_height_px",

    "signed_mean_cx_px",
    "signed_mean_bottom_y_px",
    "signed_mean_width_px",
    "signed_mean_height_px",

    "comparison_log",
]


def save_csv(
    path: Path,
    rows: list[dict],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for row in rows:

            output = {
                key:
                    row.get(
                        key
                    )
                for key in CSV_FIELDS
            }

            writer.writerow(
                output
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate M4 benchmark execution results."
        )
    )

    parser.add_argument(
        "--benchmark-root",
        required=True,
        help=(
            "Example: "
            "outputs\\benchmarks\\geometry_cutin_smoke_v1"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    benchmark_root = Path(
        args.benchmark_root
    )

    if not benchmark_root.exists():
        raise FileNotFoundError(
            f"Benchmark root not found: "
            f"{benchmark_root}"
        )

    benchmark_id = (
        benchmark_root.name
    )

    cases = load_benchmark_cases(
        benchmark_root
    )

    m4b_root = (
        benchmark_root
        / "m4b"
    )

    m4d_root = (
        benchmark_root
        / "m4d"
    )

    if args.output_dir:
        output_dir = Path(
            args.output_dir
        )
    else:
        output_dir = (
            benchmark_root
            / "m4e"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "M4E BENCHMARK AGGREGATION"
    )
    print(
        "=" * 100
    )

    print(
        "benchmark:",
        benchmark_id,
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

    rows = []
    errors = []

    # ========================================================
    # Each benchmark case
    # ========================================================

    for case_id in sorted(
        cases.keys()
    ):

        case = cases[
            case_id
        ]

        try:

            capability_path = (
                m4b_root
                / case_id
                / "he_capability.json"
            )

            execution_path = (
                m4d_root
                / case_id
                / "execution_report.json"
            )

            if not capability_path.exists():
                raise FileNotFoundError(
                    capability_path
                )

            if not execution_path.exists():
                raise FileNotFoundError(
                    execution_path
                )

            capability = load_json(
                capability_path
            )

            execution = load_json(
                execution_path
            )

            comparison_log = (
                find_comparison_log(
                    m4d_root
                    / case_id
                )
            )

            comparison = (
                parse_comparison_log(
                    comparison_log
                )
            )

            row = {
                "case_id":
                    case_id,
            }

            row.update(
                extract_case_parameters(
                    case
                )
            )

            row.update(
                extract_capability_metrics(
                    capability
                )
            )

            row.update(
                comparison
            )

            row[
                "execution_status"
            ] = execution.get(
                "status",
                "UNKNOWN",
            )

            row[
                "comparison_log"
            ] = str(
                comparison_log
            )

            rows.append(
                row
            )

            print(
                f"[OK] {case_id}"
                f"  cx={row['mae_cx_px']:.3f}"
                f"  by={row['mae_bottom_y_px']:.3f}"
                f"  w={row['mae_width_px']:.3f}"
                f"  h={row['mae_height_px']:.3f}"
                f"  vis={row['visibility_agreement']:.3f}"
            )

        except Exception as exc:

            error_text = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            errors.append(
                {
                    "case_id":
                        case_id,

                    "error":
                        error_text,
                }
            )

            print(
                f"[ERROR] "
                f"{case_id}: "
                f"{error_text}"
            )

    # ========================================================
    # Save raw per-case results
    # ========================================================

    results_json_path = (
        output_dir
        / "benchmark_results.json"
    )

    results_csv_path = (
        output_dir
        / "benchmark_results.csv"
    )

    save_json(
        results_json_path,
        {
            "benchmark_id":
                benchmark_id,

            "cases":
                rows,

            "errors":
                errors,
        },
    )

    save_csv(
        results_csv_path,
        rows,
    )

    # ========================================================
    # Aggregate descriptive statistics
    # ========================================================

    metric_fields = [

        "presence_agreement",
        "visibility_agreement",

        "mae_cx_px",
        "mae_bottom_y_px",
        "mae_width_px",
        "mae_height_px",

        "signed_mean_cx_px",
        "signed_mean_bottom_y_px",
        "signed_mean_width_px",
        "signed_mean_height_px",
    ]

    descriptive = {
        field:
            summarize_numeric_field(
                rows,
                field,
            )
        for field in metric_fields
    }

    mae_fields = [
        "mae_cx_px",
        "mae_bottom_y_px",
        "mae_width_px",
        "mae_height_px",
    ]

    worst_cases = {
        field:
            worst_case(
                rows,
                field,
            )
        for field in mae_fields
    }

    execution_passed = sum(
        1
        for row
        in rows
        if (
            row.get(
                "execution_status"
            )
            ==
            "PASS"
        )
    )

    total_ood_frames = sum(
        row.get(
            "out_of_domain_frames"
        )
        or 0
        for row
        in rows
    )

    total_culled_frames = sum(
        row.get(
            "culled_frames"
        )
        or 0
        for row
        in rows
    )

    total_bbox_frames = sum(
        row.get(
            "both_visible_bbox_frames"
        )
        or 0
        for row
        in rows
    )

    summary = {

        "benchmark_id":
            benchmark_id,

        "cases_expected":
            len(
                cases
            ),

        "cases_aggregated":
            len(
                rows
            ),

        "aggregation_errors":
            len(
                errors
            ),

        "execution_passed":
            execution_passed,

        "all_execution_passed":
            (
                execution_passed
                ==
                len(
                    cases
                )
                and
                len(
                    errors
                )
                == 0
            ),

        "total_culled_frames":
            total_culled_frames,

        "total_out_of_domain_frames":
            total_ood_frames,

        "total_bbox_evaluation_frames":
            total_bbox_frames,

        "descriptive_statistics":
            descriptive,

        "worst_cases":
            worst_cases,

        # Important distinction:
        #
        # No geometric acceptance margin has yet been defined.
        # Therefore this aggregation does NOT label the geometric
        # results PASS or FAIL.
        "geometry_acceptance":
            {
                "status":
                    "NOT_DEFINED",

                "note":
                    (
                        "M4E reports measurements only. "
                        "No geometric equivalence threshold "
                        "has been imposed."
                    ),
            },

        "errors":
            errors,
    }

    summary_path = (
        output_dir
        / "benchmark_summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print(
        "-" * 100
    )

    print(
        "Cases expected:    ",
        len(
            cases
        ),
    )

    print(
        "Cases aggregated:  ",
        len(
            rows
        ),
    )

    print(
        "Execution PASS:    ",
        execution_passed,
    )

    print(
        "Aggregation errors:",
        len(
            errors
        ),
    )

    print(
        "Total OOD frames:  ",
        total_ood_frames,
    )

    print(
        "BBox eval frames:  ",
        total_bbox_frames,
    )

    print()

    print(
        "Per-case mean metrics:"
    )

    for field in [
        "presence_agreement",
        "visibility_agreement",
        "mae_cx_px",
        "mae_bottom_y_px",
        "mae_width_px",
        "mae_height_px",
    ]:

        stats = descriptive.get(
            field
        )

        if stats is None:
            continue

        print(
            f"  {field:26s}"
            f" mean={stats['mean']:.4f}"
            f"  min={stats['min']:.4f}"
            f"  max={stats['max']:.4f}"
        )

    print()
    print(
        "Worst geometric cases:"
    )

    for field in mae_fields:

        worst = worst_cases[
            field
        ]

        if worst is None:
            continue

        print(
            f"  {field:26s}"
            f" {worst['case_id']}"
            f" = {worst['value']:.4f}"
        )

    print()
    print(
        "CSV:    ",
        results_csv_path,
    )

    print(
        "JSON:   ",
        results_json_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "=" * 100
    )

    if errors:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()