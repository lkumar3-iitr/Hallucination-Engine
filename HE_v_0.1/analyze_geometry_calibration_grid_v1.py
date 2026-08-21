#!/usr/bin/env python3

"""
analyze_geometry_calibration_grid_v1.py

Analyze the controlled CARLA distance x viewpoint calibration grid.

This script compares:

    HEPlacement V2.1 geometry
            vs
    CARLA visible instance-mask geometry

It DOES NOT use the HE rendered mask and DOES NOT modify the renderer.

Pipeline
--------
135 CARLA measurements
    -> run current frozen HEPlacement
    -> compute residuals
    -> measure repeatability
    -> collapse 3 repeats into one median per pose
    -> produce 45-pose correction surfaces

Correction targets
------------------
width_scale =
    CARLA_visible_width / HEPlacement_width

height_scale =
    CARLA_visible_height / HEPlacement_height

bottom_offset_px =
    CARLA_visible_bottom_y - HEPlacement_bottom_y

cx_offset_px is recorded diagnostically only.
We do not plan to fit cx unless multi-view evidence requires it.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from heplacement_npz_lookup_adapter import (
    HEPlacementNPZLookupAdapter,
)


# ============================================================
# Utilities
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    if not rows:

        raise RuntimeError(
            f"No rows found: {path}"
        )

    return rows


def write_csv(path, rows):

    if not rows:
        return

    path = Path(path)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def median(values):

    return float(
        np.median(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def mean(values):

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def std(values):

    return float(
        np.std(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def value_range(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return float(
        values.max()
        -
        values.min()
    )


def circular_error_deg(a, b):

    return abs(
        (
            float(a)
            -
            float(b)
            +
            180.0
        )
        %
        360.0
        -
        180.0
    )


# ============================================================
# Production HEPlacement prediction
# ============================================================

def predict_production_box(
    adapter,
    state,
    image_width,
    image_height,
    fx,
    camera_cx,
):
    """
    Mirror the current HEPlacement V2.1 compositor behavior.

    1. NPZ lookup predicts:
         bottom_y
         width
         height
         visibility

    2. cx is replaced analytically:

         cx = camera_cx + fx * x / z
    """

    box = adapter.predict_box(
        state=state,
        image_width=image_width,
        image_height=image_height,
    )

    box = dict(box)

    if not box.get(
        "visible",
        False,
    ):
        return box

    rel_x = float(
        state["x_m"]
    )

    rel_z = float(
        state["z_m"]
    )

    if rel_z <= 1e-9:

        return {
            "visible":
                False,

            "reason":
                "invalid_rel_z",
        }

    analytic_cx = (
        float(camera_cx)
        +
        float(fx)
        *
        (
            rel_x
            /
            rel_z
        )
    )

    box_width = float(
        box["box_width"]
    )

    box["cx"] = float(
        analytic_cx
    )

    box["x1"] = float(
        analytic_cx
        -
        box_width / 2.0
    )

    box["x2"] = float(
        analytic_cx
        +
        box_width / 2.0
    )

    box["source"] = (
        "heplacement_v2_npz_lookup_analytic_cx"
    )

    return box


# ============================================================
# Per-measurement analysis
# ============================================================

def analyze_measurement(
    measurement,
    adapter,
    image_width,
    image_height,
    fx,
    camera_cx,
):

    if not measurement.get(
        "mask_found",
        False,
    ):

        raise RuntimeError(
            "Calibration measurement has "
            "mask_found=False: "
            f"{measurement.get('pose_id')}"
        )

    query = measurement[
        "actual_he_query"
    ]

    mask = measurement[
        "mask_geometry"
    ]

    state = {
        "x_m":
            float(
                query["x_m"]
            ),

        "y_m":
            float(
                query.get(
                    "y_m",
                    0.0,
                )
            ),

        "z_m":
            float(
                query["z_m"]
            ),

        "yaw_deg":
            float(
                query[
                    "relative_yaw_deg"
                ]
            ),
    }

    box = predict_production_box(
        adapter=adapter,
        state=state,
        image_width=image_width,
        image_height=image_height,
        fx=fx,
        camera_cx=camera_cx,
    )

    if not box.get(
        "visible",
        False,
    ):

        raise RuntimeError(
            "HEPlacement predicts invisible "
            f"for pose {measurement['pose_id']} "
            f"repeat {measurement['repeat_id']}: "
            f"{box}"
        )

    he_width = float(
        box["box_width"]
    )

    he_height = float(
        box["box_height"]
    )

    he_bottom = float(
        box["bottom_y"]
    )

    he_cx = float(
        box["cx"]
    )

    carla_width = float(
        mask["width"]
    )

    carla_height = float(
        mask["height"]
    )

    carla_bottom = float(
        mask["bottom_y"]
    )

    carla_cx = float(
        mask["cx"]
    )

    width_scale = (
        carla_width
        /
        max(
            he_width,
            1e-9,
        )
    )

    height_scale = (
        carla_height
        /
        max(
            he_height,
            1e-9,
        )
    )

    bottom_offset = (
        carla_bottom
        -
        he_bottom
    )

    cx_offset = (
        carla_cx
        -
        he_cx
    )

    requested = measurement[
        "requested"
    ]

    return {
        "measurement_index":
            int(
                measurement[
                    "measurement_index"
                ]
            ),

        "pose_id":
            str(
                measurement[
                    "pose_id"
                ]
            ),

        "repeat_id":
            int(
                measurement[
                    "repeat_id"
                ]
            ),

        # --------------------------------------------------------
        # Requested experimental coordinate
        # --------------------------------------------------------

        "requested_distance_m":
            float(
                requested[
                    "distance_m"
                ]
            ),

        "requested_viewpoint_deg":
            float(
                requested[
                    "viewpoint_deg"
                ]
            ),

        # --------------------------------------------------------
        # Actual continuous camera-space coordinate
        # --------------------------------------------------------

        "actual_distance_m":
            float(
                query[
                    "distance_m"
                ]
            ),

        "actual_viewpoint_deg":
            float(
                query[
                    "viewpoint_deg"
                ]
            ),

        "actual_elevation_deg":
            float(
                query[
                    "elevation_deg"
                ]
            ),

        "state_x_m":
            float(
                state["x_m"]
            ),

        "state_y_m":
            float(
                state["y_m"]
            ),

        "state_z_m":
            float(
                state["z_m"]
            ),

        "state_yaw_deg":
            float(
                state["yaw_deg"]
            ),

        # --------------------------------------------------------
        # HEPlacement
        # --------------------------------------------------------

        "he_cx":
            he_cx,

        "he_bottom_y":
            he_bottom,

        "he_width":
            he_width,

        "he_height":
            he_height,

        "he_clamped":
            int(
                bool(
                    box.get(
                        "clamped",
                        False,
                    )
                )
            ),

        # --------------------------------------------------------
        # CARLA visible mask
        # --------------------------------------------------------

        "carla_cx":
            carla_cx,

        "carla_bottom_y":
            carla_bottom,

        "carla_width":
            carla_width,

        "carla_height":
            carla_height,

        "carla_area":
            int(
                mask[
                    "area"
                ]
            ),

        # --------------------------------------------------------
        # Correction target
        # --------------------------------------------------------

        "width_scale_target":
            float(
                width_scale
            ),

        "height_scale_target":
            float(
                height_scale
            ),

        "bottom_offset_target_px":
            float(
                bottom_offset
            ),

        "cx_offset_target_px":
            float(
                cx_offset
            ),
    }


# ============================================================
# Collapse repeated measurements
# ============================================================

def collapse_pose_repeats(rows):

    groups = {}

    for row in rows:

        groups.setdefault(
            row["pose_id"],
            [],
        ).append(row)

    pose_rows = []

    for pose_id in sorted(
        groups.keys(),
        key=lambda k: (
            groups[k][0][
                "requested_distance_m"
            ],
            groups[k][0][
                "requested_viewpoint_deg"
            ],
        ),
    ):

        group = groups[
            pose_id
        ]

        def vals(key):
            return [
                float(r[key])
                for r in group
            ]

        width_scale = vals(
            "width_scale_target"
        )

        height_scale = vals(
            "height_scale_target"
        )

        bottom_offset = vals(
            "bottom_offset_target_px"
        )

        cx_offset = vals(
            "cx_offset_target_px"
        )

        carla_width = vals(
            "carla_width"
        )

        carla_height = vals(
            "carla_height"
        )

        carla_bottom = vals(
            "carla_bottom_y"
        )

        carla_cx = vals(
            "carla_cx"
        )

        pose_rows.append({
            "pose_id":
                pose_id,

            "n":
                len(group),

            "requested_distance_m":
                median(
                    vals(
                        "requested_distance_m"
                    )
                ),

            "requested_viewpoint_deg":
                median(
                    vals(
                        "requested_viewpoint_deg"
                    )
                ),

            "actual_distance_m":
                median(
                    vals(
                        "actual_distance_m"
                    )
                ),

            "actual_viewpoint_deg":
                median(
                    vals(
                        "actual_viewpoint_deg"
                    )
                ),

            "actual_elevation_deg":
                median(
                    vals(
                        "actual_elevation_deg"
                    )
                ),

            # ----------------------------------------------------
            # Median correction target
            # ----------------------------------------------------

            "width_scale_median":
                median(
                    width_scale
                ),

            "height_scale_median":
                median(
                    height_scale
                ),

            "bottom_offset_median_px":
                median(
                    bottom_offset
                ),

            "cx_offset_median_px":
                median(
                    cx_offset
                ),

            # ----------------------------------------------------
            # Repeatability of correction targets
            # ----------------------------------------------------

            "width_scale_std":
                std(
                    width_scale
                ),

            "height_scale_std":
                std(
                    height_scale
                ),

            "bottom_offset_std_px":
                std(
                    bottom_offset
                ),

            "cx_offset_std_px":
                std(
                    cx_offset
                ),

            # ----------------------------------------------------
            # Direct CARLA repeatability
            # ----------------------------------------------------

            "carla_width_median_px":
                median(
                    carla_width
                ),

            "carla_width_range_px":
                value_range(
                    carla_width
                ),

            "carla_height_median_px":
                median(
                    carla_height
                ),

            "carla_height_range_px":
                value_range(
                    carla_height
                ),

            "carla_bottom_median_px":
                median(
                    carla_bottom
                ),

            "carla_bottom_range_px":
                value_range(
                    carla_bottom
                ),

            "carla_cx_median_px":
                median(
                    carla_cx
                ),

            "carla_cx_range_px":
                value_range(
                    carla_cx
                ),
        })

    return pose_rows


# ============================================================
# Matrix exports
# ============================================================

def write_surface_matrix(
    path,
    pose_rows,
    value_key,
):

    distances = sorted({
        float(
            r[
                "requested_distance_m"
            ]
        )
        for r in pose_rows
    })

    viewpoints = sorted({
        float(
            r[
                "requested_viewpoint_deg"
            ]
        )
        for r in pose_rows
    })

    lookup = {
        (
            float(
                r[
                    "requested_distance_m"
                ]
            ),
            float(
                r[
                    "requested_viewpoint_deg"
                ]
            ),
        ):
            float(
                r[
                    value_key
                ]
            )
        for r in pose_rows
    }

    path = Path(path)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["distance_m"]
            +
            [
                f"view_{v:g}_deg"
                for v in viewpoints
            ]
        )

        for distance in distances:

            row = [
                distance
            ]

            for viewpoint in viewpoints:

                row.append(
                    lookup[
                        (
                            distance,
                            viewpoint,
                        )
                    ]
                )

            writer.writerow(row)


# ============================================================
# Console surface
# ============================================================

def print_surface(
    title,
    pose_rows,
    key,
    fmt,
):

    distances = sorted({
        float(
            r[
                "requested_distance_m"
            ]
        )
        for r in pose_rows
    })

    viewpoints = sorted({
        float(
            r[
                "requested_viewpoint_deg"
            ]
        )
        for r in pose_rows
    })

    lookup = {
        (
            float(
                r[
                    "requested_distance_m"
                ]
            ),
            float(
                r[
                    "requested_viewpoint_deg"
                ]
            ),
        ):
            float(
                r[key]
            )
        for r in pose_rows
    }

    print()
    print(title)

    header = (
        "dist "
        +
        " ".join(
            f"{v:>8.0f}"
            for v in viewpoints
        )
    )

    print(header)

    for distance in distances:

        values = []

        for viewpoint in viewpoints:

            value = lookup[
                (
                    distance,
                    viewpoint,
                )
            ]

            values.append(
                format(
                    value,
                    fmt,
                )
            )

        print(
            f"{distance:4.0f} "
            +
            " ".join(
                f"{v:>8}"
                for v in values
            )
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--measurements",
        required=True,
        help=(
            "geometry calibration "
            "measurements.jsonl"
        ),
    )

    parser.add_argument(
        "--scenario-json",
        required=True,
        help=(
            "Frozen production HE scenario. "
            "Used to obtain the exact placement NPZ "
            "and camera intrinsics."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--placement-lookup",
        default=None,
        help=(
            "Optional explicit override of the "
            "scenario placement_lookup_npz."
        ),
    )

    args = parser.parse_args()

    measurements_path = Path(
        args.measurements
    )

    scenario_path = Path(
        args.scenario_json
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    measurements = load_jsonl(
        measurements_path
    )

    scenario = load_json(
        scenario_path
    )

    # ------------------------------------------------------------
    # Do not accidentally calibrate on top of another correction.
    # ------------------------------------------------------------

    runtime_correction = (
        scenario.get(
            "placement_runtime_correction",
            {},
        )
    )

    if runtime_correction.get(
        "enabled",
        False,
    ):

        raise RuntimeError(
            "The supplied scenario has "
            "placement_runtime_correction enabled. "
            "Calibration V1 must analyze the uncorrected "
            "HEPlacement baseline."
        )

    # ------------------------------------------------------------
    # Placement model
    # ------------------------------------------------------------

    if args.placement_lookup:

        lookup_path = Path(
            args.placement_lookup
        )

    else:

        lookup_value = scenario.get(
            "placement_lookup_npz"
        )

        if not lookup_value:

            raise RuntimeError(
                "Scenario has no placement_lookup_npz. "
                "Pass --placement-lookup explicitly."
            )

        lookup_path = Path(
            lookup_value
        )

    adapter = HEPlacementNPZLookupAdapter(
        lookup_path
    )

    # ------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------

    camera = scenario.get(
        "camera",
        {},
    )

    image_width = int(
        camera.get(
            "image_width",
            1280,
        )
    )

    image_height = int(
        camera.get(
            "image_height",
            720,
        )
    )

    intrinsics = camera.get(
        "intrinsics",
        {},
    )

    fx = float(
        intrinsics.get(
            "fx",
            image_width / 2.0,
        )
    )

    camera_cx = float(
        intrinsics.get(
            "cx",
            image_width / 2.0,
        )
    )

    # ------------------------------------------------------------
    # Per-measurement residuals
    # ------------------------------------------------------------

    rows = []

    for measurement in measurements:

        rows.append(
            analyze_measurement(
                measurement=
                    measurement,
                adapter=
                    adapter,
                image_width=
                    image_width,
                image_height=
                    image_height,
                fx=
                    fx,
                camera_cx=
                    camera_cx,
            )
        )

    clamped_count = sum(
        int(
            r[
                "he_clamped"
            ]
        )
        for r in rows
    )

    # ------------------------------------------------------------
    # Requested vs actual sanity
    # ------------------------------------------------------------

    max_distance_request_error = max(
        abs(
            r[
                "actual_distance_m"
            ]
            -
            r[
                "requested_distance_m"
            ]
        )
        for r in rows
    )

    max_view_request_error = max(
        circular_error_deg(
            r[
                "actual_viewpoint_deg"
            ],
            r[
                "requested_viewpoint_deg"
            ],
        )
        for r in rows
    )

    # ------------------------------------------------------------
    # Collapse 3 repeats -> 45 pose medians
    # ------------------------------------------------------------

    pose_rows = collapse_pose_repeats(
        rows
    )

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    per_measurement_path = (
        output_dir
        /
        "per_measurement_residuals.csv"
    )

    pose_path = (
        output_dir
        /
        "pose_medians.csv"
    )

    write_csv(
        per_measurement_path,
        rows,
    )

    write_csv(
        pose_path,
        pose_rows,
    )

    write_surface_matrix(
        output_dir
        /
        "width_scale_surface.csv",
        pose_rows,
        "width_scale_median",
    )

    write_surface_matrix(
        output_dir
        /
        "height_scale_surface.csv",
        pose_rows,
        "height_scale_median",
    )

    write_surface_matrix(
        output_dir
        /
        "bottom_offset_surface.csv",
        pose_rows,
        "bottom_offset_median_px",
    )

    write_surface_matrix(
        output_dir
        /
        "cx_offset_surface.csv",
        pose_rows,
        "cx_offset_median_px",
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    width_values = [
        r[
            "width_scale_median"
        ]
        for r in pose_rows
    ]

    height_values = [
        r[
            "height_scale_median"
        ]
        for r in pose_rows
    ]

    bottom_values = [
        r[
            "bottom_offset_median_px"
        ]
        for r in pose_rows
    ]

    cx_values = [
        r[
            "cx_offset_median_px"
        ]
        for r in pose_rows
    ]

    summary = {
        "measurements":
            len(rows),

        "unique_poses":
            len(pose_rows),

        "placement_lookup":
            str(
                lookup_path
            ),

        "scenario_json":
            str(
                scenario_path
            ),

        "heplacement_clamped_measurements":
            int(
                clamped_count
            ),

        "max_requested_actual_distance_error_m":
            float(
                max_distance_request_error
            ),

        "max_requested_actual_viewpoint_error_deg":
            float(
                max_view_request_error
            ),

        "repeatability": {
            "max_carla_width_range_px":
                max(
                    r[
                        "carla_width_range_px"
                    ]
                    for r in pose_rows
                ),

            "max_carla_height_range_px":
                max(
                    r[
                        "carla_height_range_px"
                    ]
                    for r in pose_rows
                ),

            "max_carla_bottom_range_px":
                max(
                    r[
                        "carla_bottom_range_px"
                    ]
                    for r in pose_rows
                ),

            "max_carla_cx_range_px":
                max(
                    r[
                        "carla_cx_range_px"
                    ]
                    for r in pose_rows
                ),
        },

        "pose_median_correction_ranges": {
            "width_scale_min":
                min(
                    width_values
                ),

            "width_scale_max":
                max(
                    width_values
                ),

            "height_scale_min":
                min(
                    height_values
                ),

            "height_scale_max":
                max(
                    height_values
                ),

            "bottom_offset_min_px":
                min(
                    bottom_values
                ),

            "bottom_offset_max_px":
                max(
                    bottom_values
                ),

            "cx_offset_min_px":
                min(
                    cx_values
                ),

            "cx_offset_max_px":
                max(
                    cx_values
                ),
        },
    }

    summary_path = (
        output_dir
        /
        "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ============================================================
    # Console report
    # ============================================================

    print()
    print("=" * 96)
    print(
        "HE GEOMETRY CALIBRATION GRID ANALYSIS V1"
    )
    print("=" * 96)

    print(
        "Measurements            :",
        len(rows),
    )

    print(
        "Unique controlled poses :",
        len(pose_rows),
    )

    print(
        "HEPlacement clamped     :",
        clamped_count,
    )

    print(
        "Max distance request err:",
        f"{max_distance_request_error:.9f} m",
    )

    print(
        "Max viewpoint request err:",
        f"{max_view_request_error:.9f} deg",
    )

    print()
    print(
        "CARLA repeatability across repeats"
    )

    print(
        "  max width range :",
        f"{summary['repeatability']['max_carla_width_range_px']:.4f}px",
    )

    print(
        "  max height range:",
        f"{summary['repeatability']['max_carla_height_range_px']:.4f}px",
    )

    print(
        "  max bottom range:",
        f"{summary['repeatability']['max_carla_bottom_range_px']:.4f}px",
    )

    print(
        "  max cx range    :",
        f"{summary['repeatability']['max_carla_cx_range_px']:.4f}px",
    )

    print()
    print(
        "45-pose correction range"
    )

    print(
        "  width scale :",
        f"{min(width_values):.6f}",
        "to",
        f"{max(width_values):.6f}",
    )

    print(
        "  height scale:",
        f"{min(height_values):.6f}",
        "to",
        f"{max(height_values):.6f}",
    )

    print(
        "  bottom offset:",
        f"{min(bottom_values):.4f}",
        "to",
        f"{max(bottom_values):.4f}",
        "px",
    )

    print(
        "  cx offset:",
        f"{min(cx_values):.4f}",
        "to",
        f"{max(cx_values):.4f}",
        "px",
    )

    print_surface(
        title=(
            "WIDTH SCALE TARGET "
            "(CARLA / HEPlacement)"
        ),
        pose_rows=pose_rows,
        key="width_scale_median",
        fmt=".4f",
    )

    print_surface(
        title=(
            "HEIGHT SCALE TARGET "
            "(CARLA / HEPlacement)"
        ),
        pose_rows=pose_rows,
        key="height_scale_median",
        fmt=".4f",
    )

    print_surface(
        title=(
            "BOTTOM OFFSET TARGET [px] "
            "(CARLA - HEPlacement)"
        ),
        pose_rows=pose_rows,
        key="bottom_offset_median_px",
        fmt=".2f",
    )

    print_surface(
        title=(
            "CX OFFSET [px] "
            "(diagnostic only)"
        ),
        pose_rows=pose_rows,
        key="cx_offset_median_px",
        fmt=".2f",
    )

    print()
    print(
        "Saved:",
        per_measurement_path,
    )

    print(
        "Saved:",
        pose_path,
    )

    print(
        "Saved:",
        summary_path,
    )

    print("=" * 96)


if __name__ == "__main__":
    main()