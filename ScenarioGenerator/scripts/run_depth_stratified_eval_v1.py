"""
run_depth_stratified_eval_v1.py

M4F: Depth-stratified geometric evaluation.

Reuses already-generated:
    CARLA ground truth
    HE metadata

No CARLA recording.
No HE rendering.
No model/lookup modification.

For every completed benchmark case, run the existing
compare_real_vs_he_bbox_v2.py at several minimum depth thresholds.

Outputs:
    depth_results.csv
    depth_results.json
    depth_summary.json
    logs/<case>/depth_XX.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys

from pathlib import Path
from statistics import mean


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

REPO_ROOT = (
    PROJECT_ROOT.parent
)

HE_ROOT = (
    REPO_ROOT
    / "HE_v_0.1"
)


# ============================================================
# IO
# ============================================================

def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)

def format_metric(
    value,
):
    if value is None:
        return "N/A"

    return f"{float(value):.3f}"

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
# Benchmark cases
# ============================================================

def load_cases(
    benchmark_root: Path,
):

    path = (
        benchmark_root
        / "benchmark_cases.json"
    )

    data = load_json(
        path
    )

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
            [],
        )

    else:

        raise ValueError(
            "Unsupported benchmark_cases.json structure."
        )

    return sorted(
        cases,
        key=lambda case:
            str(
                case[
                    "case_id"
                ]
            ),
    )


# ============================================================
# Comparator output parsing
# ============================================================

PATTERNS = {

    "rows_evaluated":
        r"Rows evaluated:\s+(\d+)",

    "both_present":
        r"Both present:\s+(\d+)",

    "presence_agreement":
        r"Presence agreement:\s+([-+0-9.eE]+)",

    "visibility_comparable":
        r"Visibility comparable:\s+(\d+)",

    "visibility_agreement":
        r"Visibility agreement:\s+([-+0-9.eE]+)",

    "both_visible_bbox_frames":
        r"Both visible for bbox:\s+(\d+)",

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


INTEGER_FIELDS = {
    "rows_evaluated",
    "both_present",
    "visibility_comparable",
    "both_visible_bbox_frames",
}


def parse_comparator_output(
    text: str,
):

    result = {}

    for (
        name,
        pattern,
    ) in PATTERNS.items():

        matches = re.findall(
            pattern,
            text,
        )

        if not matches:

            result[
                name
            ] = None

            continue

        value = matches[-1]

        if name in INTEGER_FIELDS:

            result[
                name
            ] = int(
                value
            )

        else:

            result[
                name
            ] = float(
                value
            )

    return result


# ============================================================
# Run comparator
# ============================================================

def run_comparator(
    *,
    python_exe: str,
    comparator_path: Path,
    real_gt_path: Path,
    he_metadata_path: Path,
    min_depth: float,
    log_path: Path,
):

    command = [

        python_exe,

        str(
            comparator_path
        ),

        "--real-ground-truth",
        str(
            real_gt_path
        ),

        "--he-metadata",
        str(
            he_metadata_path
        ),

        "--min-real-depth",
        str(
            min_depth
        ),
    ]

    process = subprocess.run(

        command,

        cwd=str(
            HE_ROOT
        ),

        stdout=subprocess.PIPE,

        stderr=subprocess.STDOUT,

        text=True,
    )

    output = (
        process.stdout
        or ""
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path.write_text(
        output,
        encoding="utf-8",
    )

    if process.returncode != 0:

        raise RuntimeError(
            "Comparator failed with "
            f"exit code {process.returncode}. "
            f"See {log_path}"
        )

    return parse_comparator_output(
        output
    )


# ============================================================
# Case parameters
# ============================================================

def extract_parameters(
    case: dict,
):

    p = case.get(
        "parameters",
        {}
    )

    return {

        "scenario_type":
            case.get(
                "scenario_type"
            ),

        "start_distance_m":
            p.get(
                "start_distance_m"
            ),

        "actor_speed_mps":
            p.get(
                "actor_speed_mps"
            ),

        "ego_speed_mps":
            p.get(
                "ego_speed_mps"
            ),

        "cut_duration_s":
            p.get(
                "cut_duration_s"
            ),
    }


# ============================================================
# CSV
# ============================================================

CSV_FIELDS = [

    "case_id",

    "scenario_type",

    "start_distance_m",
    "actor_speed_mps",
    "ego_speed_mps",
    "cut_duration_s",

    "min_depth_m",

    "rows_evaluated",
    "both_present",

    "presence_agreement",

    "visibility_comparable",
    "visibility_agreement",

    "both_visible_bbox_frames",

    "mae_cx_px",
    "mae_bottom_y_px",
    "mae_width_px",
    "mae_height_px",

    "signed_mean_cx_px",
    "signed_mean_bottom_y_px",
    "signed_mean_width_px",
    "signed_mean_height_px",

    "log_path",
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

            fieldnames=
                CSV_FIELDS,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    field:
                        row.get(
                            field
                        )

                    for field
                    in CSV_FIELDS
                }
            )


# ============================================================
# Aggregate helpers
# ============================================================

def valid_values(
    rows,
    field,
):

    values = [

        float(
            row[
                field
            ]
        )

        for row
        in rows

        if (
            row.get(
                field
            )
            is not None
        )
    ]

    return values


def summarize_field(
    rows,
    field,
):

    values = valid_values(
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
def weighted_mean_metric(
    rows,
    field,
    weight_field="both_visible_bbox_frames",
):
    weighted_sum = 0.0
    total_weight = 0

    for row in rows:

        value = row.get(
            field
        )

        weight = int(
            row.get(
                weight_field
            )
            or 0
        )

        if (
            value is None
            or weight <= 0
        ):
            continue

        weighted_sum += (
            float(value)
            * weight
        )

        total_weight += weight

    if total_weight == 0:
        return None

    return (
        weighted_sum
        /
        total_weight
    )

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run M4F depth-stratified geometric evaluation."
        )
    )

    parser.add_argument(
        "--benchmark-root",
        required=True,
    )

    parser.add_argument(
        "--depths",
        nargs="+",
        type=float,
        default=[
            15.0,
            20.0,
            25.0,
            30.0,
        ],
    )

    parser.add_argument(
        "--he-python",
        default=sys.executable,
    )

    args = parser.parse_args()

    benchmark_root = Path(
        args.benchmark_root
    )

    comparator_path = (
        HE_ROOT
        / "compare_real_vs_he_bbox_v2.py"
    )

    if not comparator_path.exists():

        raise FileNotFoundError(
            comparator_path
        )

    cases = load_cases(
        benchmark_root
    )

    output_root = (
        benchmark_root
        / "m4f"
    )

    logs_root = (
        output_root
        / "logs"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "M4F DEPTH-STRATIFIED GEOMETRIC EVALUATION"
    )

    print(
        "=" * 100
    )

    print(
        "benchmark:",
        benchmark_root.name,
    )

    print(
        "cases:",
        len(
            cases
        ),
    )

    print(
        "depth thresholds:",
        args.depths,
    )

    print()

    results = []
    errors = []

    # ========================================================
    # Case × depth evaluation
    # ========================================================

    for case in cases:

        case_id = str(
            case[
                "case_id"
            ]
        )

        parameters = (
            extract_parameters(
                case
            )
        )

        real_gt_path = (

            HE_ROOT
            / "recordings"
            / "he_pairs"
            / case_id
            / "real_ground_truth_v2.jsonl"
        )

        he_metadata_path = (

            HE_ROOT
            / "he_outputs"
            / f"{case_id}_he"
            / "metadata.json"
        )

        if not real_gt_path.exists():

            errors.append(
                {
                    "case_id":
                        case_id,

                    "error":
                        (
                            "Missing real ground truth: "
                            f"{real_gt_path}"
                        ),
                }
            )

            continue

        if not he_metadata_path.exists():

            errors.append(
                {
                    "case_id":
                        case_id,

                    "error":
                        (
                            "Missing HE metadata: "
                            f"{he_metadata_path}"
                        ),
                }
            )

            continue

        for min_depth in args.depths:

            depth_name = (
                f"{min_depth:g}"
            )

            log_path = (

                logs_root
                / case_id
                / f"depth_{depth_name}.txt"
            )

            try:

                metrics = run_comparator(

                    python_exe=
                        args.he_python,

                    comparator_path=
                        comparator_path,

                    real_gt_path=
                        real_gt_path,

                    he_metadata_path=
                        he_metadata_path,

                    min_depth=
                        min_depth,

                    log_path=
                        log_path,
                )

                row = {

                    "case_id":
                        case_id,

                    **parameters,

                    "min_depth_m":
                        float(
                            min_depth
                        ),

                    **metrics,

                    "log_path":
                        str(
                            log_path
                        ),
                }

                results.append(
                    row
                )

                print(
                    f"[OK] "
                    f"{case_id}  "
                    f"depth>={min_depth:g}m  "
                    f"frames="
                    f"{metrics['both_visible_bbox_frames']}  "
                    f"cx={format_metric(metrics['mae_cx_px'])}  "
                    f"by={format_metric(metrics['mae_bottom_y_px'])}  "
                    f"w={format_metric(metrics['mae_width_px'])}  "
                    f"h={format_metric(metrics['mae_height_px'])}"
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

                        "min_depth_m":
                            min_depth,

                        "error":
                            error_text,
                    }
                )

                print(
                    "[ERROR]",
                    case_id,
                    f"depth={min_depth:g}:",
                    error_text,
                )

    # ========================================================
    # Save raw results
    # ========================================================

    results_json_path = (
        output_root
        / "depth_results.json"
    )

    results_csv_path = (
        output_root
        / "depth_results.csv"
    )

    save_json(
        results_json_path,

        {
            "benchmark_id":
                benchmark_root.name,

            "depths_m":
                args.depths,

            "results":
                results,

            "errors":
                errors,
        },
    )

    save_csv(
        results_csv_path,
        results,
    )

    # ========================================================
    # Aggregate by depth
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

    depth_summary = {}

    print()
    print(
        "-" * 100
    )

    print(
        "AGGREGATE BY MINIMUM DEPTH"
    )

    print(
        "-" * 100
    )

    for min_depth in args.depths:

        depth_rows = [

            row

            for row
            in results

            if abs(
                float(
                    row[
                        "min_depth_m"
                    ]
                )
                -
                float(
                    min_depth
                )
            )
            < 1e-9
        ]
        valid_bbox_rows = [
            row
            for row in depth_rows
            if (
                int(
                    row.get(
                        "both_visible_bbox_frames"
                    )
                    or 0
                )
                > 0
                and
                row.get(
                    "mae_cx_px"
                )
                is not None
            )
        ]

        zero_bbox_rows = [
            row
            for row in depth_rows
            if int(
                row.get(
                    "both_visible_bbox_frames"
                )
                or 0
            )
            == 0
        ]

        stats = {

            field:
                summarize_field(
                    depth_rows,
                    field,
                )

            for field
            in metric_fields
        }

        total_frames = sum(

            int(
                row[
                    "both_visible_bbox_frames"
                ]
                or 0
            )

            for row
            in depth_rows
        )

        depth_summary[
            str(
                min_depth
            )
        ] = {

            "min_depth_m":
                float(
                    min_depth
                ),

            "cases_total":
                len(
                    depth_rows
                ),

            "cases_with_bbox_samples":
                len(
                    valid_bbox_rows
                ),

            "cases_without_bbox_samples":
                len(
                    zero_bbox_rows
                ),

            "case_ids_without_bbox_samples": [
                row[
                    "case_id"
                ]
                for row
                in zero_bbox_rows
            ],

            "bbox_frames":
                total_frames,

            "macro_metrics":
                stats,

            "frame_weighted_mae": {

                "mae_cx_px":
                    weighted_mean_metric(
                        depth_rows,
                        "mae_cx_px",
                    ),

                "mae_bottom_y_px":
                    weighted_mean_metric(
                        depth_rows,
                        "mae_bottom_y_px",
                    ),

                "mae_width_px":
                    weighted_mean_metric(
                        depth_rows,
                        "mae_width_px",
                    ),

                "mae_height_px":
                    weighted_mean_metric(
                        depth_rows,
                        "mae_height_px",
                    ),
            },
        }

        print()
        print(
            f"Depth >= {min_depth:g} m"
        )

        print(
            f"  cases with bbox samples: "
            f"{len(valid_bbox_rows)}/{len(depth_rows)}"
        )

        if zero_bbox_rows:
            print(
                "  no bbox samples: "
                + ", ".join(
                    row["case_id"]
                    for row
                    in zero_bbox_rows
                )
            )

        print(
            f"  bbox frames: "
            f"{total_frames}"
        )

        for field in [

            "mae_cx_px",
            "mae_bottom_y_px",
            "mae_width_px",
            "mae_height_px",
        ]:

            field_stats = stats[
                field
            ]

            if field_stats is None:
                continue

            print(
                f"  {field:24s}"
                f" mean="
                f"{field_stats['mean']:.4f}"
                f"  min="
                f"{field_stats['min']:.4f}"
                f"  max="
                f"{field_stats['max']:.4f}"
            )

    # ========================================================
    # Save summary
    # ========================================================

    summary_path = (
        output_root
        / "depth_summary.json"
    )

    save_json(
        summary_path,

        {
            "benchmark_id":
                benchmark_root.name,

            "cases":
                len(
                    cases
                ),

            "depth_thresholds_m":
                args.depths,

            "evaluations_expected":
                (
                    len(
                        cases
                    )
                    *
                    len(
                        args.depths
                    )
                ),

            "evaluations_completed":
                len(
                    results
                ),

            "errors":
                errors,

            "by_depth":
                depth_summary,
        },
    )

    print()
    print(
        "=" * 100
    )

    print(
        "Evaluations expected:",
        (
            len(
                cases
            )
            *
            len(
                args.depths
            )
        ),
    )

    print(
        "Evaluations completed:",
        len(
            results
        ),
    )

    print(
        "Errors:",
        len(
            errors
        ),
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

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()