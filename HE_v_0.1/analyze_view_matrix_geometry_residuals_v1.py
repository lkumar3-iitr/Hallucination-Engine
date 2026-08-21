#!/usr/bin/env python3

"""
analyze_view_matrix_geometry_residuals_v1.py

Geometry-calibration diagnostic for Hallucination Engine.

Purpose
-------
Compare the geometry requested by HEPlacement against the actual visible
CARLA instance-mask geometry.

IMPORTANT:
    This script does NOT evaluate the HE rendered alpha mask.

That is intentional.

We want to isolate:

    HEPlacement geometry
            vs
    CARLA visible actor geometry

without mixing in:
    - sprite silhouette
    - alpha threshold
    - rasterization
    - subpixel compositor

The resulting correction targets are:

    width_scale_target
        = CARLA_mask_width / HEPlacement_width

    height_scale_target
        = CARLA_mask_height / HEPlacement_height

    bottom_offset_target_px
        = CARLA_mask_bottom_y - HEPlacement_bottom_y

    cx_offset_target_px
        = CARLA_mask_cx - HEPlacement_cx

Continuous calibration coordinates come from the view-matrix query:

    query_distance_m
    relative_angle_deg
    query_elevation_deg

NOT from the discretely selected sprite.

This keeps geometry calibration independent from sprite-bank quantization.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Utility
# ============================================================

def normalize_angle_180(angle_deg):
    return (
        float(angle_deg) + 180.0
    ) % 360.0 - 180.0


def nearest_bin_center(value, step):
    """
    Quantize only for analysis/reporting.

    Calibration per-frame data remains continuous.
    """

    value = float(value)
    step = float(step)

    if step <= 0.0:
        return value

    return (
        round(
            value / step
        )
        * step
    )


def find_mask_path(mask_dir, frame_idx):
    """
    Support the filename forms used by our CARLA recording tools.
    """

    mask_dir = Path(mask_dir)

    candidates = [
        mask_dir
        / f"frame_{frame_idx:06d}_mask.png",

        mask_dir
        / f"frame_{frame_idx:06d}.png",

        mask_dir
        / f"{frame_idx:06d}_mask.png",

        mask_dir
        / f"{frame_idx:06d}.png",
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


def load_binary_mask_geometry(
    path,
    threshold=127,
):
    """
    Return bbox/area geometry for a CARLA instance mask.
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read mask: {path}"
        )

    ys, xs = np.where(
        image > int(threshold)
    )

    if len(xs) == 0:
        return None

    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    )

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    )

    width = int(
        x2 - x1 + 1
    )

    height = int(
        y2 - y1 + 1
    )

    area = int(
        len(xs)
    )

    cx = (
        float(x1 + x2)
        / 2.0
    )

    bottom_y = float(
        y2
    )

    return {
        "x1":
            x1,

        "y1":
            y1,

        "x2":
            x2,

        "y2":
            y2,

        "width":
            width,

        "height":
            height,

        "area":
            area,

        "cx":
            cx,

        "bottom_y":
            bottom_y,
    }


def choose_actor(
    frame,
    actor_id=None,
):
    """
    Select the requested rendered adversary.

    For our current single-actor experiments actor_id may be omitted.
    """

    adversaries = frame.get(
        "adversaries",
        [],
    )

    rendered = [
        actor
        for actor in adversaries
        if actor.get(
            "rendered",
            False,
        )
    ]

    if actor_id is not None:

        for actor in rendered:

            if str(
                actor.get(
                    "id",
                    "",
                )
            ) == str(actor_id):

                return actor

        return None

    if not rendered:
        return None

    if len(rendered) > 1:

        raise RuntimeError(
            "Multiple rendered adversaries found. "
            "Use --actor-id to select one."
        )

    return rendered[0]


def safe_mean(values):

    if not values:
        return None

    return float(
        np.mean(values)
    )


def safe_median(values):

    if not values:
        return None

    return float(
        np.median(values)
    )


def safe_std(values):

    if not values:
        return None

    return float(
        np.std(values)
    )


def safe_percentile(
    values,
    q,
):

    if not values:
        return None

    return float(
        np.percentile(
            values,
            q,
        )
    )


# ============================================================
# Per-frame extraction
# ============================================================

def extract_frame_record(
    frame,
    mask_dir,
    mask_threshold,
    actor_id=None,
):
    """
    Build one calibration record.

    Returns None if no valid rendered actor or CARLA mask exists.
    """

    frame_idx = int(
        frame["frame_idx"]
    )

    actor = choose_actor(
        frame=frame,
        actor_id=actor_id,
    )

    if actor is None:
        return None

    box = actor.get(
        "box",
        {},
    )

    sprite = actor.get(
        "sprite",
        {},
    )

    state = actor.get(
        "state",
        {},
    )

    if not box.get(
        "visible",
        False,
    ):
        return None

    required_box_fields = [
        "cx",
        "bottom_y",
        "box_width",
        "box_height",
    ]

    for field in required_box_fields:

        if field not in box:

            raise RuntimeError(
                f"Frame {frame_idx}: "
                f"missing box field '{field}'"
            )

    mask_path = find_mask_path(
        mask_dir=mask_dir,
        frame_idx=frame_idx,
    )

    if mask_path is None:

        return {
            "_missing_mask":
                True,

            "frame_idx":
                frame_idx,
        }

    carla = load_binary_mask_geometry(
        path=mask_path,
        threshold=mask_threshold,
    )

    if carla is None:

        return {
            "_empty_mask":
                True,

            "frame_idx":
                frame_idx,

            "mask_path":
                str(mask_path),
        }

    # ------------------------------------------------------------
    # HEPlacement target geometry
    # ------------------------------------------------------------

    he_cx = float(
        box["cx"]
    )

    he_bottom = float(
        box["bottom_y"]
    )

    he_width = float(
        box["box_width"]
    )

    he_height = float(
        box["box_height"]
    )

    # ------------------------------------------------------------
    # CARLA visible silhouette geometry
    # ------------------------------------------------------------

    carla_cx = float(
        carla["cx"]
    )

    carla_bottom = float(
        carla["bottom_y"]
    )

    carla_width = float(
        carla["width"]
    )

    carla_height = float(
        carla["height"]
    )

    # ------------------------------------------------------------
    # Correction targets.
    #
    # These are expressed in the direction required to convert:
    #
    #       HEPlacement -> CARLA visible geometry
    # ------------------------------------------------------------

    width_scale_target = (
        carla_width
        /
        max(
            1e-9,
            he_width,
        )
    )

    height_scale_target = (
        carla_height
        /
        max(
            1e-9,
            he_height,
        )
    )

    bottom_offset_target = (
        carla_bottom
        -
        he_bottom
    )

    cx_offset_target = (
        carla_cx
        -
        he_cx
    )

    # Useful inverse ratios for diagnostic comparison with our
    # previous placement-vs-mask analysis.
    placement_over_carla_width = (
        he_width
        /
        max(
            1e-9,
            carla_width,
        )
    )

    placement_over_carla_height = (
        he_height
        /
        max(
            1e-9,
            carla_height,
        )
    )

    # ------------------------------------------------------------
    # Continuous view coordinates.
    #
    # Use query coordinates, not selected sprite coordinates.
    # ------------------------------------------------------------

    query_viewpoint_deg = float(
        sprite.get(
            "relative_angle_deg",
            state.get(
                "yaw_deg",
                0.0,
            ),
        )
    )

    query_viewpoint_deg = (
        normalize_angle_180(
            query_viewpoint_deg
        )
    )

    query_distance_m = float(
        sprite.get(
            "query_distance_m",
            math.sqrt(
                float(
                    state.get(
                        "x_m",
                        0.0,
                    )
                ) ** 2
                +
                float(
                    state.get(
                        "z_m",
                        0.0,
                    )
                ) ** 2
            ),
        )
    )

    query_elevation_deg = float(
        sprite.get(
            "query_elevation_deg",
            0.0,
        )
    )

    selected_angle = float(
        sprite.get(
            "selected_angle",
            0.0,
        )
    )

    selected_distance = float(
        sprite.get(
            "selected_distance_m",
            0.0,
        )
    )

    selected_elevation = float(
        sprite.get(
            "selected_elevation_deg",
            0.0,
        )
    )

    return {
        "_missing_mask":
            False,

        "_empty_mask":
            False,

        "frame_idx":
            frame_idx,

        "actor_id":
            str(
                actor.get(
                    "id",
                    "",
                )
            ),

        # --------------------------------------------------------
        # Camera-relative state
        # --------------------------------------------------------

        "state_x_m":
            float(
                state.get(
                    "x_m",
                    0.0,
                )
            ),

        "state_y_m":
            float(
                state.get(
                    "y_m",
                    0.0,
                )
            ),

        "state_z_m":
            float(
                state.get(
                    "z_m",
                    0.0,
                )
            ),

        "state_yaw_deg":
            float(
                state.get(
                    "yaw_deg",
                    0.0,
                )
            ),

        # --------------------------------------------------------
        # Continuous view-matrix query coordinates
        # --------------------------------------------------------

        "query_viewpoint_deg":
            query_viewpoint_deg,

        "query_distance_m":
            query_distance_m,

        "query_elevation_deg":
            query_elevation_deg,

        # --------------------------------------------------------
        # Discrete selected sprite -- diagnostic only
        # --------------------------------------------------------

        "selected_angle_deg":
            selected_angle,

        "selected_distance_m":
            selected_distance,

        "selected_elevation_deg":
            selected_elevation,

        # --------------------------------------------------------
        # HEPlacement geometry
        # --------------------------------------------------------

        "he_cx":
            he_cx,

        "he_bottom_y":
            he_bottom,

        "he_width":
            he_width,

        "he_height":
            he_height,

        # --------------------------------------------------------
        # CARLA instance-mask geometry
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
                carla["area"]
            ),

        # --------------------------------------------------------
        # Direct HE minus CARLA residuals
        # --------------------------------------------------------

        "he_minus_carla_cx_px":
            he_cx
            -
            carla_cx,

        "he_minus_carla_bottom_px":
            he_bottom
            -
            carla_bottom,

        "he_over_carla_width":
            placement_over_carla_width,

        "he_over_carla_height":
            placement_over_carla_height,

        # --------------------------------------------------------
        # Correction targets:
        # HEPlacement -> CARLA visible geometry
        # --------------------------------------------------------

        "cx_offset_target_px":
            cx_offset_target,

        "bottom_offset_target_px":
            bottom_offset_target,

        "width_scale_target":
            width_scale_target,

        "height_scale_target":
            height_scale_target,

        "mask_path":
            str(
                mask_path
            ),
    }


# ============================================================
# Grouped analysis
# ============================================================

def build_bin_rows(
    rows,
    distance_bin_m,
    angle_bin_deg,
    elevation_bin_deg,
):
    """
    Aggregate correction targets by query-space bins.

    This is for inspection only.

    Later, our controlled calibration collection will put several
    measurements around each exact distance/viewpoint pose.
    """

    grouped = {}

    for row in rows:

        distance_bin = nearest_bin_center(
            row[
                "query_distance_m"
            ],
            distance_bin_m,
        )

        angle_bin = nearest_bin_center(
            row[
                "query_viewpoint_deg"
            ],
            angle_bin_deg,
        )

        elevation_bin = nearest_bin_center(
            row[
                "query_elevation_deg"
            ],
            elevation_bin_deg,
        )

        key = (
            distance_bin,
            angle_bin,
            elevation_bin,
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            row
        )

    output = []

    for key in sorted(
        grouped.keys()
    ):

        distance_bin, angle_bin, elevation_bin = key

        group = grouped[
            key
        ]

        width_scales = [
            r[
                "width_scale_target"
            ]
            for r in group
        ]

        height_scales = [
            r[
                "height_scale_target"
            ]
            for r in group
        ]

        bottom_offsets = [
            r[
                "bottom_offset_target_px"
            ]
            for r in group
        ]

        cx_offsets = [
            r[
                "cx_offset_target_px"
            ]
            for r in group
        ]

        query_distances = [
            r[
                "query_distance_m"
            ]
            for r in group
        ]

        query_angles = [
            r[
                "query_viewpoint_deg"
            ]
            for r in group
        ]

        query_elevations = [
            r[
                "query_elevation_deg"
            ]
            for r in group
        ]

        output.append({
            "distance_bin_m":
                float(
                    distance_bin
                ),

            "viewpoint_bin_deg":
                float(
                    angle_bin
                ),

            "elevation_bin_deg":
                float(
                    elevation_bin
                ),

            "n":
                len(
                    group
                ),

            "query_distance_mean_m":
                safe_mean(
                    query_distances
                ),

            "query_viewpoint_mean_deg":
                safe_mean(
                    query_angles
                ),

            "query_elevation_mean_deg":
                safe_mean(
                    query_elevations
                ),

            "width_scale_mean":
                safe_mean(
                    width_scales
                ),

            "width_scale_median":
                safe_median(
                    width_scales
                ),

            "width_scale_std":
                safe_std(
                    width_scales
                ),

            "height_scale_mean":
                safe_mean(
                    height_scales
                ),

            "height_scale_median":
                safe_median(
                    height_scales
                ),

            "height_scale_std":
                safe_std(
                    height_scales
                ),

            "bottom_offset_mean_px":
                safe_mean(
                    bottom_offsets
                ),

            "bottom_offset_median_px":
                safe_median(
                    bottom_offsets
                ),

            "bottom_offset_std_px":
                safe_std(
                    bottom_offsets
                ),

            "cx_offset_mean_px":
                safe_mean(
                    cx_offsets
                ),

            "cx_offset_median_px":
                safe_median(
                    cx_offsets
                ),

            "cx_offset_std_px":
                safe_std(
                    cx_offsets
                ),
        })

    return output


def build_summary(
    rows,
    missing_masks,
    empty_masks,
):
    """
    Global diagnostic summary.
    """

    width_scale = [
        r[
            "width_scale_target"
        ]
        for r in rows
    ]

    height_scale = [
        r[
            "height_scale_target"
        ]
        for r in rows
    ]

    bottom_offset = [
        r[
            "bottom_offset_target_px"
        ]
        for r in rows
    ]

    cx_offset = [
        r[
            "cx_offset_target_px"
        ]
        for r in rows
    ]

    return {
        "comparable_frames":
            len(
                rows
            ),

        "missing_masks":
            int(
                missing_masks
            ),

        "empty_masks":
            int(
                empty_masks
            ),

        "width_scale_target": {
            "mean":
                safe_mean(
                    width_scale
                ),

            "median":
                safe_median(
                    width_scale
                ),

            "std":
                safe_std(
                    width_scale
                ),

            "p05":
                safe_percentile(
                    width_scale,
                    5,
                ),

            "p95":
                safe_percentile(
                    width_scale,
                    95,
                ),
        },

        "height_scale_target": {
            "mean":
                safe_mean(
                    height_scale
                ),

            "median":
                safe_median(
                    height_scale
                ),

            "std":
                safe_std(
                    height_scale
                ),

            "p05":
                safe_percentile(
                    height_scale,
                    5,
                ),

            "p95":
                safe_percentile(
                    height_scale,
                    95,
                ),
        },

        "bottom_offset_target_px": {
            "mean":
                safe_mean(
                    bottom_offset
                ),

            "median":
                safe_median(
                    bottom_offset
                ),

            "std":
                safe_std(
                    bottom_offset
                ),

            "p05":
                safe_percentile(
                    bottom_offset,
                    5,
                ),

            "p95":
                safe_percentile(
                    bottom_offset,
                    95,
                ),
        },

        "cx_offset_target_px": {
            "mean":
                safe_mean(
                    cx_offset
                ),

            "median":
                safe_median(
                    cx_offset
                ),

            "std":
                safe_std(
                    cx_offset
                ),

            "p05":
                safe_percentile(
                    cx_offset,
                    5,
                ),

            "p95":
                safe_percentile(
                    cx_offset,
                    95,
                ),
        },
    }


# ============================================================
# IO
# ============================================================

def write_csv(
    path,
    rows,
):

    if not rows:
        return

    path = Path(
        path
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(
                    rows[0].keys()
                ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        required=True,
        help="HE compositor metadata.json",
    )

    parser.add_argument(
        "--carla-mask-dir",
        required=True,
        help="CARLA real_instance_masks directory",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--actor-id",
        default=None,
        help=(
            "Optional adversary ID. "
            "Required only for multi-actor scenarios."
        ),
    )

    parser.add_argument(
        "--carla-threshold",
        type=int,
        default=127,
    )

    parser.add_argument(
        "--distance-bin-m",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--angle-bin-deg",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--elevation-bin-deg",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    metadata_path = Path(
        args.metadata
    )

    mask_dir = Path(
        args.carla_mask_dir
    )

    output_dir = Path(
        args.output_dir
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata not found: "
            f"{metadata_path}"
        )

    if not mask_dir.exists():

        raise FileNotFoundError(
            f"CARLA mask directory not found: "
            f"{mask_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(
            f
        )

    rows = []

    missing_masks = 0
    empty_masks = 0

    for frame in metadata.get(
        "frames",
        [],
    ):

        record = extract_frame_record(
            frame=frame,
            mask_dir=mask_dir,
            mask_threshold=
                args.carla_threshold,
            actor_id=
                args.actor_id,
        )

        if record is None:
            continue

        if record.get(
            "_missing_mask",
            False,
        ):

            missing_masks += 1
            continue

        if record.get(
            "_empty_mask",
            False,
        ):

            empty_masks += 1
            continue

        # Internal flags are not useful in CSV.
        record.pop(
            "_missing_mask",
            None,
        )

        record.pop(
            "_empty_mask",
            None,
        )

        rows.append(
            record
        )

    if not rows:

        raise RuntimeError(
            "No comparable HEPlacement/CARLA "
            "mask frames were found."
        )

    bin_rows = build_bin_rows(
        rows=rows,
        distance_bin_m=
            args.distance_bin_m,
        angle_bin_deg=
            args.angle_bin_deg,
        elevation_bin_deg=
            args.elevation_bin_deg,
    )

    summary = build_summary(
        rows=rows,
        missing_masks=
            missing_masks,
        empty_masks=
            empty_masks,
    )

    summary.update({
        "metadata":
            str(
                metadata_path
            ),

        "carla_mask_dir":
            str(
                mask_dir
            ),

        "carla_threshold":
            int(
                args.carla_threshold
            ),

        "distance_bin_m":
            float(
                args.distance_bin_m
            ),

        "angle_bin_deg":
            float(
                args.angle_bin_deg
            ),

        "elevation_bin_deg":
            float(
                args.elevation_bin_deg
            ),

        "important_note": (
            "Correction coordinates use continuous "
            "query distance/viewpoint/elevation, not "
            "discrete selected sprite coordinates."
        ),
    })

    per_frame_path = (
        output_dir
        /
        "per_frame_geometry_residuals.csv"
    )

    bins_path = (
        output_dir
        /
        "geometry_bins.csv"
    )

    summary_path = (
        output_dir
        /
        "summary.json"
    )

    write_csv(
        path=per_frame_path,
        rows=rows,
    )

    write_csv(
        path=bins_path,
        rows=bin_rows,
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
    # Console summary
    # ============================================================

    width_scale = summary[
        "width_scale_target"
    ]

    height_scale = summary[
        "height_scale_target"
    ]

    bottom_offset = summary[
        "bottom_offset_target_px"
    ]

    cx_offset = summary[
        "cx_offset_target_px"
    ]

    print()
    print("=" * 92)
    print(
        "HE VIEW-MATRIX GEOMETRY RESIDUALS V1"
    )
    print("=" * 92)

    print(
        "Comparable frames:",
        len(
            rows
        ),
    )

    print(
        "Missing masks    :",
        missing_masks,
    )

    print(
        "Empty masks      :",
        empty_masks,
    )

    print()
    print(
        "Correction target: "
        "HEPlacement -> CARLA visible mask"
    )

    print()
    print(
        "Width scale"
    )

    print(
        "  mean   :",
        f"{width_scale['mean']:.6f}",
    )

    print(
        "  median :",
        f"{width_scale['median']:.6f}",
    )

    print(
        "  P05/P95:",
        f"{width_scale['p05']:.6f}",
        "/",
        f"{width_scale['p95']:.6f}",
    )

    print()
    print(
        "Height scale"
    )

    print(
        "  mean   :",
        f"{height_scale['mean']:.6f}",
    )

    print(
        "  median :",
        f"{height_scale['median']:.6f}",
    )

    print(
        "  P05/P95:",
        f"{height_scale['p05']:.6f}",
        "/",
        f"{height_scale['p95']:.6f}",
    )

    print()
    print(
        "Bottom offset"
    )

    print(
        "  mean   :",
        f"{bottom_offset['mean']:.4f}px",
    )

    print(
        "  median :",
        f"{bottom_offset['median']:.4f}px",
    )

    print(
        "  P05/P95:",
        f"{bottom_offset['p05']:.4f}",
        "/",
        f"{bottom_offset['p95']:.4f}",
        "px",
    )

    print()
    print(
        "CX offset"
    )

    print(
        "  mean   :",
        f"{cx_offset['mean']:.4f}px",
    )

    print(
        "  median :",
        f"{cx_offset['median']:.4f}px",
    )

    print(
        "  P05/P95:",
        f"{cx_offset['p05']:.4f}",
        "/",
        f"{cx_offset['p95']:.4f}",
        "px",
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  corrected_width  = "
        "HE_width  * width_scale_target"
    )

    print(
        "  corrected_height = "
        "HE_height * height_scale_target"
    )

    print(
        "  corrected_bottom = "
        "HE_bottom + bottom_offset_target_px"
    )

    print(
        "  corrected_cx     = "
        "HE_cx + cx_offset_target_px"
    )

    print()
    print(
        "Saved:",
        per_frame_path,
    )

    print(
        "Saved:",
        bins_path,
    )

    print(
        "Saved:",
        summary_path,
    )

    print("=" * 92)


if __name__ == "__main__":
    main()