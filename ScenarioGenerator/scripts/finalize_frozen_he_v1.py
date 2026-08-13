"""
finalize_frozen_he_v1.py

Finalizes the frozen HE Placement/Rendering v1 validation package.

This script DOES NOT:
    - run CARLA
    - render new videos
    - modify the placement lookup
    - regenerate scenarios

It only collects already-completed benchmark results and creates a
frozen validation package.

Outputs:
    outputs/frozen_he_v1/
        validation_results.csv
        depth_results.csv
        validation_summary.md
        frozen_manifest.json
        coverage.json

Run:
    cd /d D:\\HallucinationEngine\\ScenarioGenerator

    python scripts\\finalize_frozen_he_v1.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()

SG_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]

HE_ROOT = REPO_ROOT / "HE_v_0.1"


DEFAULT_BENCHMARK_ROOT = (
    SG_ROOT
    / "outputs"
    / "benchmarks"
    / "geometry_cutin_smoke_v1"
)

DEFAULT_OUTPUT_ROOT = (
    SG_ROOT
    / "outputs"
    / "frozen_he_v1"
)


# ---------------------------------------------------------------------
# Frozen HE configuration
# ---------------------------------------------------------------------

LOOKUP_PATH = (
    HE_ROOT
    / "assets"
    / "placement_lookup"
    / "heplacement_v2_lookup_camera_relative_x4_0_100_z05_yaw1.npz"
)


FROZEN_CAMERA = {
    "width": 1280,
    "height": 720,
    "fov_deg": 90.0,

    "camera_x_m": 1.5,
    "camera_y_m": 0.0,
    "camera_z_m": 1.6,

    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
    "roll_deg": 0.0,

    "fx": 640.0,
    "fy": 640.0,
    "cx": 640.0,
    "cy": 360.0,
}


FROZEN_PLACEMENT_DOMAIN = {
    "camera_relative_x_m": [-4.0, 4.0],
    "camera_relative_z_m": [0.0, 100.0],
    "yaw_deg": [0.0, 359.0],

    "visibility_policy": {
        "min_render_depth_m": 0.0,
        "behind_camera": "CULLED",
        "outside_lookup_domain": "PLACEMENT_OOD",
    },
}


DEPTH_THRESHOLDS_M = [
    15.0,
    20.0,
    25.0,
    30.0,
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def read_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_json(
    path: Path,
    data: Any,
) -> None:

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


def read_csv_rows(
    path: Path,
) -> List[Dict[str, str]]:

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def to_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    if text.lower() in {
        "none",
        "null",
        "nan",
        "n/a",
        "na",
    }:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def to_int(
    value: Any,
) -> Optional[int]:

    v = to_float(value)

    if v is None:
        return None

    return int(v)


def get_field(
    row: Dict[str, Any],
    *names: str,
) -> Any:

    # exact first
    for name in names:
        if name in row:
            return row[name]

    # case-insensitive fallback
    lower_map = {
        str(k).lower(): v
        for k, v in row.items()
    }

    for name in names:

        key = name.lower()

        if key in lower_map:
            return lower_map[key]

    return None


def sha256_file(
    path: Path,
) -> Optional[str]:

    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def relative_or_absolute(
    path: Path,
) -> str:

    try:
        return str(
            path.relative_to(
                REPO_ROOT
            )
        )

    except ValueError:
        return str(path)


def weighted_mean(
    rows: List[Dict[str, Any]],
    metric_name: str,
) -> Optional[float]:

    numerator = 0.0
    denominator = 0

    for row in rows:

        metric = to_float(
            get_field(
                row,
                metric_name,
            )
        )

        frames = to_int(
            get_field(
                row,
                "both_visible_bbox_frames",
                "bbox_eval_frames",
                "bbox_frames",
            )
        )

        if (
            metric is None
            or frames is None
            or frames <= 0
        ):
            continue

        numerator += (
            metric
            * frames
        )

        denominator += frames

    if denominator == 0:
        return None

    return (
        numerator
        / denominator
    )


def fmt(
    value: Optional[float],
    digits: int = 3,
) -> str:

    if value is None:
        return "N/A"

    return (
        f"{value:.{digits}f}"
    )


# ---------------------------------------------------------------------
# Coverage / result discovery
# ---------------------------------------------------------------------

def find_he_metadata(
    case_id: str,
) -> Optional[Path]:

    direct_candidates = [

        HE_ROOT
        / "he_outputs"
        / f"{case_id}_he"
        / "metadata.json",

        HE_ROOT
        / "he_outputs"
        / case_id
        / "metadata.json",
    ]

    for path in direct_candidates:

        if path.exists():
            return path

    he_outputs = (
        HE_ROOT
        / "he_outputs"
    )

    if not he_outputs.exists():
        return None

    case_lower = (
        case_id.lower()
    )

    for path in he_outputs.rglob(
        "metadata.json"
    ):

        if (
            case_lower
            in str(
                path.parent
            ).lower()
        ):
            return path

    return None


def build_coverage() -> Dict[str, Any]:

    cases = {

        "controlled_cutin_benchmark": {
            "type": "quantitative_benchmark",
            "case_count": 8,
        },

        "oncoming": {
            "type": "standardized_v2_validation",
            "case_id":
                "geometry_oncoming_pass_smoke_v1_case_0001",
        },

        "multi_actor": {
            "type": "multi_actor_validation",
            "case_id":
                "v2_multi_actor_001",
        },
    }

    # Controlled cut-in benchmark
    cutin_ids = [
        f"geometry_cutin_smoke_v1_case_{i:04d}"
        for i in range(
            1,
            9,
        )
    ]

    cutin_entries = []

    for case_id in cutin_ids:

        gt = (
            HE_ROOT
            / "recordings"
            / "he_pairs"
            / case_id
            / "real_ground_truth_v2.jsonl"
        )

        metadata = find_he_metadata(
            case_id
        )

        cutin_entries.append(
            {
                "case_id":
                    case_id,

                "real_ground_truth_exists":
                    gt.exists(),

                "real_ground_truth":
                    relative_or_absolute(
                        gt
                    ),

                "he_metadata_exists":
                    (
                        metadata
                        is not None
                        and
                        metadata.exists()
                    ),

                "he_metadata":
                    (
                        relative_or_absolute(
                            metadata
                        )
                        if metadata
                        else None
                    ),
            }
        )

    cases[
        "controlled_cutin_benchmark"
    ][
        "cases"
    ] = cutin_entries

    # Oncoming and multi actor
    for key in [
        "oncoming",
        "multi_actor",
    ]:

        case_id = (
            cases[key][
                "case_id"
            ]
        )

        gt = (
            HE_ROOT
            / "recordings"
            / "he_pairs"
            / case_id
            / "real_ground_truth_v2.jsonl"
        )

        metadata = find_he_metadata(
            case_id
        )

        cases[key].update(
            {
                "real_ground_truth_exists":
                    gt.exists(),

                "real_ground_truth":
                    relative_or_absolute(
                        gt
                    ),

                "he_metadata_exists":
                    (
                        metadata
                        is not None
                        and
                        metadata.exists()
                    ),

                "he_metadata":
                    (
                        relative_or_absolute(
                            metadata
                        )
                        if metadata
                        else None
                    ),
            }
        )

    # Older engineering validation scenarios.
    # We do NOT mix their metrics into the canonical M4 benchmark because
    # they were produced during earlier pipeline revisions.
    cases[
        "legacy_engineering_validation"
    ] = {

        "included_in_quantitative_summary":
            False,

        "reason":
            (
                "Earlier development scenarios used "
                "different pipeline revisions. They remain "
                "engineering validation evidence but are not "
                "mixed with the frozen standardized M4 protocol."
            ),

        "scenario_families": [
            "static",
            "following/receding",
            "crossing left-to-right",
            "crossing right-to-left",
            "ego turning",
            "earlier cut-in",
            "earlier oncoming",
        ],
    }

    return cases


# ---------------------------------------------------------------------
# Benchmark analysis
# ---------------------------------------------------------------------

def analyze_depth_results(
    depth_rows: List[Dict[str, str]],
) -> Dict[str, Any]:

    groups: Dict[
        float,
        List[Dict[str, str]],
    ] = defaultdict(list)

    for row in depth_rows:

        depth = to_float(
            get_field(
                row,
                "min_depth_m",
                "min_real_depth_m",
                "depth_m",
                "depth",
            )
        )

        if depth is None:
            continue

        groups[
            depth
        ].append(
            row
        )

    summary: Dict[str, Any] = {}

    metric_names = [
        "mae_cx_px",
        "mae_bottom_y_px",
        "mae_width_px",
        "mae_height_px",
    ]

    for depth in sorted(
        groups.keys()
    ):

        rows = groups[
            depth
        ]

        valid_rows = []

        total_frames = 0

        for row in rows:

            frames = to_int(
                get_field(
                    row,
                    "both_visible_bbox_frames",
                    "bbox_eval_frames",
                    "bbox_frames",
                )
            )

            if (
                frames is not None
                and frames > 0
            ):
                valid_rows.append(
                    row
                )

                total_frames += frames

        macro = {}
        weighted = {}

        for metric in metric_names:

            values = []

            for row in valid_rows:

                value = to_float(
                    get_field(
                        row,
                        metric,
                    )
                )

                if value is not None:
                    values.append(
                        value
                    )

            macro[
                metric
            ] = {

                "mean":
                    (
                        mean(values)
                        if values
                        else None
                    ),

                "min":
                    (
                        min(values)
                        if values
                        else None
                    ),

                "max":
                    (
                        max(values)
                        if values
                        else None
                    ),
            }

            weighted[
                metric
            ] = weighted_mean(
                valid_rows,
                metric,
            )

        no_sample_cases = []

        for row in rows:

            frames = to_int(
                get_field(
                    row,
                    "both_visible_bbox_frames",
                    "bbox_eval_frames",
                    "bbox_frames",
                )
            )

            if (
                frames is None
                or frames <= 0
            ):
                no_sample_cases.append(
                    get_field(
                        row,
                        "case_id",
                    )
                )

        summary[
            str(
                int(depth)
                if depth.is_integer()
                else depth
            )
        ] = {

            "min_depth_m":
                depth,

            "cases_total":
                len(rows),

            "cases_with_bbox_samples":
                len(valid_rows),

            "cases_without_bbox_samples":
                len(rows)
                - len(valid_rows),

            "case_ids_without_bbox_samples":
                no_sample_cases,

            "bbox_frames":
                total_frames,

            "macro":
                macro,

            "frame_weighted_mae":
                weighted,
        }

    return summary


# ---------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------

def write_markdown_summary(
    path: Path,
    benchmark_rows: List[Dict[str, str]],
    depth_summary: Dict[str, Any],
    coverage: Dict[str, Any],
) -> None:

    lines: List[str] = []

    lines.append(
        "# Frozen HE Placement / Rendering v1"
    )

    lines.append("")

    lines.append(
        "**Status:** FROZEN / VALIDATED"
    )

    lines.append("")

    lines.append(
        "This package freezes the HE v1 placement and "
        "rendering configuration after the standardized "
        "CARLA–HE geometric validation."
    )

    lines.append("")

    lines.append(
        "No placement lookup tuning is performed by this "
        "finalization step."
    )

    lines.append("")

    # -------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------

    lines.append(
        "## Frozen camera"
    )

    lines.append("")

    lines.append(
        f"- Resolution: "
        f"{FROZEN_CAMERA['width']} x "
        f"{FROZEN_CAMERA['height']}"
    )

    lines.append(
        f"- Horizontal FOV: "
        f"{FROZEN_CAMERA['fov_deg']} deg"
    )

    lines.append(
        "- Mount: "
        f"x={FROZEN_CAMERA['camera_x_m']} m, "
        f"y={FROZEN_CAMERA['camera_y_m']} m, "
        f"z={FROZEN_CAMERA['camera_z_m']} m"
    )

    lines.append(
        "- Orientation: "
        "pitch=0 deg, yaw=0 deg, roll=0 deg"
    )

    lines.append(
        "- Intrinsics: "
        f"fx={FROZEN_CAMERA['fx']}, "
        f"fy={FROZEN_CAMERA['fy']}, "
        f"cx={FROZEN_CAMERA['cx']}, "
        f"cy={FROZEN_CAMERA['cy']}"
    )

    lines.append("")

    # -------------------------------------------------------------
    # Canonical benchmark
    # -------------------------------------------------------------

    lines.append(
        "## Canonical controlled benchmark"
    )

    lines.append("")

    lines.append(
        f"- Controlled cut-in cases: "
        f"{len(benchmark_rows)}"
    )

    lines.append(
        "- Factors: start distance, actor speed, "
        "cut-in duration"
    )

    lines.append(
        "- HE placement OOD checking enabled"
    )

    lines.append(
        "- Exact CARLA–HE backend state invariant "
        "validated before execution"
    )

    lines.append("")

    # -------------------------------------------------------------
    # Depth table
    # -------------------------------------------------------------

    lines.append(
        "## Depth-stratified geometric result"
    )

    lines.append("")

    lines.append(
        "| Minimum depth | Cases | BBox frames | "
        "cx MAE | bottom-y MAE | width MAE | height MAE |"
    )

    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|"
    )

    for depth_key in sorted(
        depth_summary.keys(),
        key=lambda x: float(x),
    ):

        item = depth_summary[
            depth_key
        ]

        macro = item[
            "macro"
        ]

        lines.append(
            "| "
            f">= {item['min_depth_m']:g} m | "
            f"{item['cases_with_bbox_samples']}/"
            f"{item['cases_total']} | "
            f"{item['bbox_frames']} | "
            f"{fmt(macro['mae_cx_px']['mean'])} | "
            f"{fmt(macro['mae_bottom_y_px']['mean'])} | "
            f"{fmt(macro['mae_width_px']['mean'])} | "
            f"{fmt(macro['mae_height_px']['mean'])} |"
        )

    lines.append("")

    lines.append(
        "The scale and vertical-placement errors decrease "
        "as near-field frames are excluded. Horizontal-center "
        "error is not expected to be strictly monotonic because "
        "it also depends on lateral position and viewpoint."
    )

    lines.append("")

    # -------------------------------------------------------------
    # Frame-weighted table
    # -------------------------------------------------------------

    lines.append(
        "## Frame-weighted MAE"
    )

    lines.append("")

    lines.append(
        "| Minimum depth | cx | bottom-y | width | height |"
    )

    lines.append(
        "|---:|---:|---:|---:|---:|"
    )

    for depth_key in sorted(
        depth_summary.keys(),
        key=lambda x: float(x),
    ):

        item = depth_summary[
            depth_key
        ]

        weighted = item[
            "frame_weighted_mae"
        ]

        lines.append(
            "| "
            f">= {item['min_depth_m']:g} m | "
            f"{fmt(weighted['mae_cx_px'])} | "
            f"{fmt(weighted['mae_bottom_y_px'])} | "
            f"{fmt(weighted['mae_width_px'])} | "
            f"{fmt(weighted['mae_height_px'])} |"
        )

    lines.append("")

    # -------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------

    lines.append(
        "## Validation coverage"
    )

    lines.append("")

    lines.append(
        "- Controlled cut-in benchmark: standardized "
        "quantitative validation."
    )

    lines.append(
        "- Oncoming scenario: standardized V2/M4 "
        "execution validation."
    )

    lines.append(
        "- Multi-actor scenario: V2 multi-actor "
        "execution validation."
    )

    lines.append(
        "- Static, following/receding, crossing, "
        "ego-turning and earlier cut-in/oncoming cases "
        "remain engineering validation evidence from "
        "earlier pipeline stages."
    )

    lines.append("")

    # -------------------------------------------------------------
    # Operating domain
    # -------------------------------------------------------------

    lines.append(
        "## Frozen placement operating domain"
    )

    lines.append("")

    lines.append(
        "- Camera-relative lateral x: "
        "[-4.0, +4.0] m"
    )

    lines.append(
        "- Camera-relative forward z: "
        "[0.0, 100.0] m"
    )

    lines.append(
        "- Actor relative yaw: "
        "[0, 359] deg"
    )

    lines.append(
        "- z <= 0: actor is behind camera and culled."
    )

    lines.append(
        "- Placement outside the lookup domain must be "
        "reported as PLACEMENT_OOD rather than silently "
        "treated as valid."
    )

    lines.append("")

    # -------------------------------------------------------------
    # Known limitations
    # -------------------------------------------------------------

    lines.append(
        "## Known limitations"
    )

    lines.append("")

    lines.append(
        "- Near-field scale and vertical-placement "
        "error is larger than far-field error."
    )

    lines.append(
        "- Partial near-camera clipping is not modeled "
        "as a separate visibility state."
    )

    lines.append(
        "- General scene occlusion is not fully modeled."
    )

    lines.append(
        "- Frozen lookup validity is limited to its "
        "defined camera-relative placement domain."
    )

    lines.append(
        "- These limitations are documented rather than "
        "corrected by fitting the lookup to validation cases."
    )

    lines.append("")

    lines.append(
        "## Freeze decision"
    )

    lines.append("")

    lines.append(
        "The placement lookup, camera convention, viewpoint "
        "calculation, coordinate transforms and V2 rendering "
        "pipeline are frozen after this validation."
    )

    lines.append("")

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    benchmark_root = (
        args.benchmark_root.resolve()
    )

    output_root = (
        args.output_root.resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    m4e_csv = (
        benchmark_root
        / "m4e"
        / "benchmark_results.csv"
    )

    m4e_json = (
        benchmark_root
        / "m4e"
        / "benchmark_results.json"
    )

    m4e_summary = (
        benchmark_root
        / "m4e"
        / "benchmark_summary.json"
    )

    m4f_csv = (
        benchmark_root
        / "m4f"
        / "depth_results.csv"
    )

    m4f_json = (
        benchmark_root
        / "m4f"
        / "depth_results.json"
    )

    m4f_summary = (
        benchmark_root
        / "m4f"
        / "depth_summary.json"
    )

    required = [
        m4e_csv,
        m4f_csv,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        print(
            "\nMissing required canonical result files:"
        )

        for path in missing:
            print(
                f"  {path}"
            )

        raise SystemExit(
            1
        )

    benchmark_rows = read_csv_rows(
        m4e_csv
    )

    depth_rows = read_csv_rows(
        m4f_csv
    )

    depth_summary = analyze_depth_results(
        depth_rows
    )

    coverage = build_coverage()

    # -------------------------------------------------------------
    # Copy canonical result tables
    # -------------------------------------------------------------

    shutil.copy2(
        m4e_csv,
        output_root
        / "validation_results.csv",
    )

    shutil.copy2(
        m4f_csv,
        output_root
        / "depth_results.csv",
    )

    # Keep original JSON outputs as traceability artifacts.
    for source in [
        m4e_json,
        m4e_summary,
        m4f_json,
        m4f_summary,
    ]:

        if source.exists():

            shutil.copy2(
                source,
                output_root
                / source.name,
            )

    # -------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------

    write_json(
        output_root
        / "coverage.json",
        coverage,
    )

    # -------------------------------------------------------------
    # Frozen files and hashes
    # -------------------------------------------------------------

    frozen_files = [

        LOOKUP_PATH,

        HE_ROOT
        / "run_he_temporal_compositor_v2.py",

        HE_ROOT
        / "compare_real_vs_he_bbox_v2.py",

        SG_ROOT
        / "scenario_generator"
        / "backends"
        / "he_backend"
        / "current_he_adapter_v2.py",

        SG_ROOT
        / "scenario_generator"
        / "backends"
        / "carla_backend"
        / "carla_adapter_v2.py",

        SG_ROOT
        / "scenario_generator"
        / "trajectory"
        / "trajectory_resolver_v2.py",
    ]

    file_manifest = []

    print()
    print(
        "Computing frozen-file hashes..."
    )

    for path in frozen_files:

        exists = path.exists()

        digest = (
            sha256_file(path)
            if exists
            else None
        )

        file_manifest.append(
            {
                "path":
                    relative_or_absolute(
                        path
                    ),

                "exists":
                    exists,

                "sha256":
                    digest,
            }
        )

        state = (
            "OK"
            if exists
            else "MISSING"
        )

        print(
            f"  [{state}] "
            f"{relative_or_absolute(path)}"
        )

    # -------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------

    manifest = {

        "name":
            "HE Placement and Rendering v1",

        "status":
            "FROZEN_VALIDATED",

        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "repository_root":
            str(
                REPO_ROOT
            ),

        "canonical_benchmark":
            relative_or_absolute(
                benchmark_root
            ),

        "canonical_quantitative_protocol": {
            "benchmark":
                "geometry_cutin_smoke_v1",

            "number_of_cases":
                len(
                    benchmark_rows
                ),

            "depth_thresholds_m":
                DEPTH_THRESHOLDS_M,

            "geometry_acceptance_threshold":
                None,

            "geometry_acceptance_policy":
                (
                    "No post-hoc acceptance threshold was "
                    "defined. Results are reported as "
                    "characterization metrics."
                ),
        },

        "camera":
            FROZEN_CAMERA,

        "placement_domain":
            FROZEN_PLACEMENT_DOMAIN,

        "placement_lookup":
            relative_or_absolute(
                LOOKUP_PATH
            ),

        "frozen_files":
            file_manifest,

        "coverage_file":
            "coverage.json",

        "canonical_outputs": {
            "validation_results":
                "validation_results.csv",

            "depth_results":
                "depth_results.csv",

            "summary":
                "validation_summary.md",
        },

        "known_limitations": [
            (
                "Near-field scale and vertical-placement "
                "errors are larger than far-field errors."
            ),

            (
                "Partial near-camera clipping is not "
                "modeled as a separate visibility state."
            ),

            (
                "General scene occlusion is not fully "
                "modeled."
            ),

            (
                "Placement validity is restricted to "
                "the frozen lookup operating domain."
            ),
        ],

        "freeze_policy": {
            "lookup_retraining_after_validation":
                False,

            "post_hoc_case_specific_tuning":
                False,

            "camera_configuration_changes":
                False,

            "coordinate_convention_changes":
                False,
        },
    }

    write_json(
        output_root
        / "frozen_manifest.json",
        manifest,
    )

    write_markdown_summary(
        output_root
        / "validation_summary.md",
        benchmark_rows,
        depth_summary,
        coverage,
    )

    # -------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------

    print()
    print(
        "=" * 88
    )

    print(
        "FROZEN HE V1 FINALIZATION"
    )

    print(
        "=" * 88
    )

    print(
        f"Canonical cut-in cases: "
        f"{len(benchmark_rows)}"
    )

    print()

    print(
        "Depth-stratified summary:"
    )

    for depth_key in sorted(
        depth_summary.keys(),
        key=lambda x: float(x),
    ):

        item = depth_summary[
            depth_key
        ]

        macro = item[
            "macro"
        ]

        print()
        print(
            f"Depth >= "
            f"{item['min_depth_m']:g} m"
        )

        print(
            f"  cases: "
            f"{item['cases_with_bbox_samples']}/"
            f"{item['cases_total']}"
        )

        print(
            f"  bbox frames: "
            f"{item['bbox_frames']}"
        )

        print(
            "  MAE "
            f"cx={fmt(macro['mae_cx_px']['mean'])}  "
            f"bottom={fmt(macro['mae_bottom_y_px']['mean'])}  "
            f"width={fmt(macro['mae_width_px']['mean'])}  "
            f"height={fmt(macro['mae_height_px']['mean'])}"
        )

    print()
    print(
        f"Output:"
    )

    print(
        f"  {output_root}"
    )

    print()

    print(
        "Files:"
    )

    for name in [
        "validation_results.csv",
        "depth_results.csv",
        "validation_summary.md",
        "frozen_manifest.json",
        "coverage.json",
    ]:
        print(
            f"  {name}"
        )

    print()
    print(
        "STATUS: HE PLACEMENT / RENDERING V1 "
        "FROZEN AND VALIDATED"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()