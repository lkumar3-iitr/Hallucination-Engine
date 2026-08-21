"""
evaluate_carla_he_mask_equivalence_v1.py

Compare CARLA true instance-segmentation masks against HE rendered
sprite-alpha masks.

This evaluates silhouette/rendering equivalence rather than only
placement-box agreement.

Metrics
-------
Overlap:
    Mask IoU
    Dice coefficient

Area:
    HE / CARLA area ratio
    relative area error

Position:
    silhouette centroid error

Boundary:
    symmetric mean boundary distance
    symmetric p95 boundary distance
    boundary precision / recall / F1 at tolerance

Distance-conditioned:
    metrics grouped by CARLA actor depth

Outputs
-------
summary.json
per_frame.csv
distance_bins.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Helpers
# ============================================================

def finite(x):
    try:
        return math.isfinite(
            float(x)
        )
    except Exception:
        return False


def safe_mean(values):
    values = [
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return math.nan

    return float(
        np.mean(values)
    )


def safe_median(values):
    values = [
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return math.nan

    return float(
        np.median(values)
    )


def safe_percentile(
    values,
    percentile,
):
    values = [
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return math.nan

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


def safe_ratio(
    numerator,
    denominator,
):
    denominator = float(
        denominator
    )

    if abs(
        denominator
    ) < 1e-9:
        return math.nan

    return (
        float(numerator)
        / denominator
    )


def fmt(x):
    if not finite(x):
        return "nan"

    return (
        f"{float(x):.4f}"
    )


# ============================================================
# Real CARLA metadata
# ============================================================

def load_real_bbox_jsonl(path):
    rows = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rec = json.loads(
                line
            )

            frame_idx = int(
                rec[
                    "recorded_frame_idx"
                ]
            )

            bbox = rec.get(
                "bbox",
                {},
            )

            rows[
                frame_idx
            ] = {
                "frame_idx":
                    frame_idx,

                "visible":
                    bool(
                        bbox.get(
                            "visible",
                            False,
                        )
                    ),

                "depth_m":
                    float(
                        bbox.get(
                            "depth_m",
                            math.nan,
                        )
                    ),
            }

    return rows


# ============================================================
# Mask loading
# ============================================================

def load_binary_mask(
    path,
    threshold,
):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read mask: {path}"
        )

    return (
        image > int(threshold)
    )


# ============================================================
# Mask geometry
# ============================================================

def mask_centroid(mask):
    ys, xs = np.where(
        mask
    )

    if len(xs) == 0:
        return (
            math.nan,
            math.nan,
        )

    return (
        float(
            np.mean(xs)
        ),
        float(
            np.mean(ys)
        ),
    )


def mask_bbox(mask):
    ys, xs = np.where(
        mask
    )

    if len(xs) == 0:
        return None

    return {
        "x1":
            int(xs.min()),

        "y1":
            int(ys.min()),

        "x2":
            int(xs.max()),

        "y2":
            int(ys.max()),

        "width":
            int(
                xs.max()
                - xs.min()
                + 1
            ),

        "height":
            int(
                ys.max()
                - ys.min()
                + 1
            ),
    }


# ============================================================
# Overlap metrics
# ============================================================

def mask_overlap_metrics(
    real_mask,
    he_mask,
):
    real_pixels = int(
        np.count_nonzero(
            real_mask
        )
    )

    he_pixels = int(
        np.count_nonzero(
            he_mask
        )
    )

    intersection = int(
        np.count_nonzero(
            real_mask
            &
            he_mask
        )
    )

    union = int(
        np.count_nonzero(
            real_mask
            |
            he_mask
        )
    )

    iou = safe_ratio(
        intersection,
        union,
    )

    dice = safe_ratio(
        2.0 * intersection,
        real_pixels
        + he_pixels,
    )

    area_ratio = safe_ratio(
        he_pixels,
        real_pixels,
    )

    relative_area_error = (
        abs(
            he_pixels
            - real_pixels
        )
        / float(real_pixels)
        if real_pixels > 0
        else math.nan
    )

    return {
        "real_pixels":
            real_pixels,

        "he_pixels":
            he_pixels,

        "intersection_pixels":
            intersection,

        "union_pixels":
            union,

        "mask_iou":
            iou,

        "dice":
            dice,

        "area_ratio":
            area_ratio,

        "relative_area_error":
            relative_area_error,
    }


# ============================================================
# Boundary metrics
# ============================================================

def extract_boundary(mask):
    """
    One-pixel-ish morphological boundary of a binary mask.
    """

    mask_u8 = (
        mask.astype(
            np.uint8
        )
        * 255
    )

    kernel = np.ones(
        (
            3,
            3,
        ),
        dtype=np.uint8,
    )

    eroded = cv2.erode(
        mask_u8,
        kernel,
        iterations=1,
    )

    boundary = (
        mask_u8
        > eroded
    )

    return boundary


def distance_to_boundary(
    source_boundary,
    target_boundary,
):
    """
    For every source boundary pixel, return its distance to the
    nearest target boundary pixel.
    """

    if (
        np.count_nonzero(
            source_boundary
        )
        == 0
    ):
        return np.array(
            [],
            dtype=np.float32,
        )

    if (
        np.count_nonzero(
            target_boundary
        )
        == 0
    ):
        return np.array(
            [],
            dtype=np.float32,
        )

    # cv2.distanceTransform measures distance of non-zero pixels
    # to the nearest zero pixel.
    #
    # Therefore:
    #     target boundary pixels = 0
    #     everything else        = 255
    distance_input = np.where(
        target_boundary,
        0,
        255,
    ).astype(
        np.uint8
    )

    distance_map = cv2.distanceTransform(
        distance_input,
        cv2.DIST_L2,
        5,
    )

    return distance_map[
        source_boundary
    ]


def boundary_metrics(
    real_mask,
    he_mask,
    tolerance_px=2.0,
):
    real_boundary = extract_boundary(
        real_mask
    )

    he_boundary = extract_boundary(
        he_mask
    )

    real_to_he = distance_to_boundary(
        real_boundary,
        he_boundary,
    )

    he_to_real = distance_to_boundary(
        he_boundary,
        real_boundary,
    )

    if (
        real_to_he.size == 0
        or he_to_real.size == 0
    ):
        return {
            "mean_boundary_distance_px":
                math.nan,

            "p95_boundary_distance_px":
                math.nan,

            "max_boundary_distance_px":
                math.nan,

            "boundary_precision":
                math.nan,

            "boundary_recall":
                math.nan,

            "boundary_f1":
                math.nan,

            "real_boundary_pixels":
                int(
                    np.count_nonzero(
                        real_boundary
                    )
                ),

            "he_boundary_pixels":
                int(
                    np.count_nonzero(
                        he_boundary
                    )
                ),
        }

    all_distances = np.concatenate(
        [
            real_to_he,
            he_to_real,
        ]
    )

    # HE boundary pixel is correct if it lies sufficiently
    # close to the CARLA boundary.
    boundary_precision = float(
        np.mean(
            he_to_real
            <= float(
                tolerance_px
            )
        )
    )

    # CARLA boundary pixel is recovered if an HE boundary
    # lies sufficiently close to it.
    boundary_recall = float(
        np.mean(
            real_to_he
            <= float(
                tolerance_px
            )
        )
    )

    if (
        boundary_precision
        + boundary_recall
        > 1e-9
    ):
        boundary_f1 = (
            2.0
            * boundary_precision
            * boundary_recall
            / (
                boundary_precision
                + boundary_recall
            )
        )
    else:
        boundary_f1 = 0.0

    return {
        "mean_boundary_distance_px":
            float(
                np.mean(
                    all_distances
                )
            ),

        "p95_boundary_distance_px":
            float(
                np.percentile(
                    all_distances,
                    95,
                )
            ),

        "max_boundary_distance_px":
            float(
                np.max(
                    all_distances
                )
            ),

        "boundary_precision":
            boundary_precision,

        "boundary_recall":
            boundary_recall,

        "boundary_f1":
            boundary_f1,

        "real_boundary_pixels":
            int(
                np.count_nonzero(
                    real_boundary
                )
            ),

        "he_boundary_pixels":
            int(
                np.count_nonzero(
                    he_boundary
                )
            ),
    }


# ============================================================
# Per-frame comparison
# ============================================================

def evaluate_frame(
    frame_idx,
    real_mask_path,
    he_mask_path,
    real_depth_m,
    real_threshold,
    he_threshold,
    boundary_tolerance_px,
):
    real_mask = load_binary_mask(
        real_mask_path,
        real_threshold,
    )

    he_mask = load_binary_mask(
        he_mask_path,
        he_threshold,
    )

    if (
        real_mask.shape
        != he_mask.shape
    ):
        raise RuntimeError(
            "Mask shape mismatch at frame "
            f"{frame_idx}: "
            f"CARLA={real_mask.shape}, "
            f"HE={he_mask.shape}"
        )

    overlap = mask_overlap_metrics(
        real_mask,
        he_mask,
    )

    boundary = boundary_metrics(
        real_mask,
        he_mask,
        tolerance_px=
            boundary_tolerance_px,
    )

    real_cx, real_cy = (
        mask_centroid(
            real_mask
        )
    )

    he_cx, he_cy = (
        mask_centroid(
            he_mask
        )
    )

    if (
        finite(real_cx)
        and finite(real_cy)
        and finite(he_cx)
        and finite(he_cy)
    ):
        centroid_error_px = math.hypot(
            he_cx
            - real_cx,
            he_cy
            - real_cy,
        )
    else:
        centroid_error_px = math.nan

    real_box = mask_bbox(
        real_mask
    )

    he_box = mask_bbox(
        he_mask
    )

    row = {
        "frame_idx":
            int(frame_idx),

        "real_depth_m":
            float(
                real_depth_m
            ),

        "real_mask_path":
            str(
                real_mask_path
            ),

        "he_mask_path":
            str(
                he_mask_path
            ),

        "real_centroid_x":
            real_cx,

        "real_centroid_y":
            real_cy,

        "he_centroid_x":
            he_cx,

        "he_centroid_y":
            he_cy,

        "centroid_error_px":
            centroid_error_px,

        "real_bbox_width_px":
            (
                real_box["width"]
                if real_box
                else math.nan
            ),

        "real_bbox_height_px":
            (
                real_box["height"]
                if real_box
                else math.nan
            ),

        "he_bbox_width_px":
            (
                he_box["width"]
                if he_box
                else math.nan
            ),

        "he_bbox_height_px":
            (
                he_box["height"]
                if he_box
                else math.nan
            ),

        **overlap,
        **boundary,
    }

    return row


# ============================================================
# Summary
# ============================================================

def summarize_rows(rows):

    def values(key):
        return [
            r.get(
                key,
                math.nan,
            )
            for r in rows
            if finite(
                r.get(
                    key,
                    math.nan,
                )
            )
        ]

    return {
        "num_frames":
            len(rows),

        "mean_mask_iou":
            safe_mean(
                values(
                    "mask_iou"
                )
            ),

        "median_mask_iou":
            safe_median(
                values(
                    "mask_iou"
                )
            ),

        "p10_mask_iou":
            safe_percentile(
                values(
                    "mask_iou"
                ),
                10,
            ),

        "p05_mask_iou":
            safe_percentile(
                values(
                    "mask_iou"
                ),
                5,
            ),

        "fraction_iou_ge_0_50":
            safe_mean(
                [
                    1.0
                    if x >= 0.50
                    else 0.0
                    for x in values(
                        "mask_iou"
                    )
                ]
            ),

        "fraction_iou_ge_0_75":
            safe_mean(
                [
                    1.0
                    if x >= 0.75
                    else 0.0
                    for x in values(
                        "mask_iou"
                    )
                ]
            ),

        "mean_dice":
            safe_mean(
                values(
                    "dice"
                )
            ),

        "median_dice":
            safe_median(
                values(
                    "dice"
                )
            ),

        "mean_area_ratio":
            safe_mean(
                values(
                    "area_ratio"
                )
            ),

        "mean_relative_area_error":
            safe_mean(
                values(
                    "relative_area_error"
                )
            ),

        "mean_centroid_error_px":
            safe_mean(
                values(
                    "centroid_error_px"
                )
            ),

        "median_centroid_error_px":
            safe_median(
                values(
                    "centroid_error_px"
                )
            ),

        "p95_centroid_error_px":
            safe_percentile(
                values(
                    "centroid_error_px"
                ),
                95,
            ),

        "mean_boundary_distance_px":
            safe_mean(
                values(
                    "mean_boundary_distance_px"
                )
            ),

        "mean_p95_boundary_distance_px":
            safe_mean(
                values(
                    "p95_boundary_distance_px"
                )
            ),

        "p95_of_boundary_p95_px":
            safe_percentile(
                values(
                    "p95_boundary_distance_px"
                ),
                95,
            ),

        "mean_boundary_precision":
            safe_mean(
                values(
                    "boundary_precision"
                )
            ),

        "mean_boundary_recall":
            safe_mean(
                values(
                    "boundary_recall"
                )
            ),

        "mean_boundary_f1":
            safe_mean(
                values(
                    "boundary_f1"
                )
            ),
    }


# ============================================================
# Distance bins
# ============================================================

DISTANCE_BINS = [
    (
        "0_5",
        0.0,
        5.0,
    ),
    (
        "5_10",
        5.0,
        10.0,
    ),
    (
        "10_15",
        10.0,
        15.0,
    ),
    (
        "15_20",
        15.0,
        20.0,
    ),
    (
        "20_30",
        20.0,
        30.0,
    ),
    (
        "30_50",
        30.0,
        50.0,
    ),
    (
        "50_plus",
        50.0,
        math.inf,
    ),
]


def build_distance_bins(rows):
    output = []

    for (
        label,
        low,
        high,
    ) in DISTANCE_BINS:

        selected = [
            r
            for r in rows
            if finite(
                r[
                    "real_depth_m"
                ]
            )
            and float(
                r[
                    "real_depth_m"
                ]
            ) >= low
            and float(
                r[
                    "real_depth_m"
                ]
            ) < high
        ]

        summary = summarize_rows(
            selected
        )

        output.append(
            {
                "distance_bin":
                    label,

                "min_depth_m":
                    low,

                "max_depth_m":
                    (
                        high
                        if finite(high)
                        else "inf"
                    ),

                **summary,
            }
        )

    return output


# ============================================================
# CSV output
# ============================================================

def write_csv(
    path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
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
        "--carla-mask-dir",
        required=True,
    )

    parser.add_argument(
        "--he-mask-dir",
        required=True,
    )

    parser.add_argument(
        "--real-bbox",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--carla-threshold",
        type=int,
        default=127,
    )

    parser.add_argument(
        "--he-threshold",
        type=int,
        default=10,
        help=(
            "HE alpha threshold. 10 matches the visible-alpha "
            "threshold used by the view-matrix compositor."
        ),
    )

    parser.add_argument(
        "--boundary-tolerance-px",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--min-real-depth",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    carla_mask_dir = Path(
        args.carla_mask_dir
    )

    he_mask_dir = Path(
        args.he_mask_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    real_bbox_rows = (
        load_real_bbox_jsonl(
            args.real_bbox
        )
    )

    frame_indices = sorted(
        real_bbox_rows.keys()
    )

    rows = []

    missing_carla = []
    missing_he = []

    skipped_not_visible = []

    for frame_idx in frame_indices:

        real_info = (
            real_bbox_rows[
                frame_idx
            ]
        )

        if not real_info[
            "visible"
        ]:
            skipped_not_visible.append(
                frame_idx
            )
            continue

        depth_m = real_info[
            "depth_m"
        ]

        if (
            finite(depth_m)
            and float(depth_m)
            < float(
                args.min_real_depth
            )
        ):
            continue

        carla_mask_path = (
            carla_mask_dir
            / (
                f"frame_"
                f"{frame_idx:06d}"
                f"_mask.png"
            )
        )

        he_mask_path = (
            he_mask_dir
            / (
                f"frame_"
                f"{frame_idx:06d}"
                f"_mask.png"
            )
        )

        if not carla_mask_path.exists():

            missing_carla.append(
                frame_idx
            )

            continue

        if not he_mask_path.exists():

            missing_he.append(
                frame_idx
            )

            continue

        row = evaluate_frame(
            frame_idx=
                frame_idx,

            real_mask_path=
                carla_mask_path,

            he_mask_path=
                he_mask_path,

            real_depth_m=
                depth_m,

            real_threshold=
                args.carla_threshold,

            he_threshold=
                args.he_threshold,

            boundary_tolerance_px=
                args.boundary_tolerance_px,
        )

        rows.append(
            row
        )

    if not rows:
        raise RuntimeError(
            "No comparable CARLA/HE masks found."
        )

    summary_metrics = (
        summarize_rows(
            rows
        )
    )

    distance_bins = (
        build_distance_bins(
            rows
        )
    )

    worst_iou_rows = sorted(
        rows,
        key=lambda r:
            (
                r[
                    "mask_iou"
                ]
                if finite(
                    r[
                        "mask_iou"
                    ]
                )
                else 999.0
            ),
    )[:10]

    summary = {
        "carla_mask_dir":
            str(
                carla_mask_dir
            ),

        "he_mask_dir":
            str(
                he_mask_dir
            ),

        "real_bbox":
            str(
                args.real_bbox
            ),

        "carla_threshold":
            int(
                args.carla_threshold
            ),

        "he_threshold":
            int(
                args.he_threshold
            ),

        "boundary_tolerance_px":
            float(
                args.boundary_tolerance_px
            ),

        "min_real_depth_m":
            float(
                args.min_real_depth
            ),

        "missing_carla_frames":
            missing_carla,

        "missing_he_frames":
            missing_he,

        "skipped_not_visible_frames":
            skipped_not_visible,

        "metrics":
            summary_metrics,

        "distance_bins":
            distance_bins,

        "worst_iou_frames":
            [
                {
                    "frame_idx":
                        int(
                            r[
                                "frame_idx"
                            ]
                        ),

                    "depth_m":
                        float(
                            r[
                                "real_depth_m"
                            ]
                        ),

                    "mask_iou":
                        float(
                            r[
                                "mask_iou"
                            ]
                        ),

                    "dice":
                        float(
                            r[
                                "dice"
                            ]
                        ),

                    "centroid_error_px":
                        float(
                            r[
                                "centroid_error_px"
                            ]
                        ),
                }
                for r in worst_iou_rows
            ],
    }

    with open(
        output_dir
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    write_csv(
        output_dir
        / "per_frame.csv",
        rows,
    )

    write_csv(
        output_dir
        / "distance_bins.csv",
        distance_bins,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "CARLA <-> HE MASK EQUIVALENCE V1"
    )

    print(
        "=" * 78
    )

    print(
        "Comparable frames:",
        summary_metrics[
            "num_frames"
        ],
    )

    print(
        "Missing CARLA masks:",
        len(
            missing_carla
        ),
    )

    print(
        "Missing HE masks:",
        len(
            missing_he
        ),
    )

    print()

    print(
        "Silhouette overlap"
    )

    print(
        "  Mean IoU       :",
        fmt(
            summary_metrics[
                "mean_mask_iou"
            ]
        ),
    )

    print(
        "  Median IoU     :",
        fmt(
            summary_metrics[
                "median_mask_iou"
            ]
        ),
    )

    print(
        "  P10 IoU        :",
        fmt(
            summary_metrics[
                "p10_mask_iou"
            ]
        ),
    )

    print(
        "  P05 IoU        :",
        fmt(
            summary_metrics[
                "p05_mask_iou"
            ]
        ),
    )

    print(
        "  IoU >= 0.50    :",
        fmt(
            summary_metrics[
                "fraction_iou_ge_0_50"
            ]
        ),
    )

    print(
        "  IoU >= 0.75    :",
        fmt(
            summary_metrics[
                "fraction_iou_ge_0_75"
            ]
        ),
    )

    print(
        "  Mean Dice      :",
        fmt(
            summary_metrics[
                "mean_dice"
            ]
        ),
    )

    print()

    print(
        "Silhouette geometry"
    )

    print(
        "  HE/CARLA area  :",
        fmt(
            summary_metrics[
                "mean_area_ratio"
            ]
        ),
    )

    print(
        "  Rel area error :",
        fmt(
            summary_metrics[
                "mean_relative_area_error"
            ]
        ),
    )

    print(
        "  Centroid error :",
        fmt(
            summary_metrics[
                "mean_centroid_error_px"
            ]
        ),
        "px",
    )

    print(
        "  P95 centroid   :",
        fmt(
            summary_metrics[
                "p95_centroid_error_px"
            ]
        ),
        "px",
    )

    print()

    print(
        "Boundary"
    )

    print(
        "  Mean distance  :",
        fmt(
            summary_metrics[
                "mean_boundary_distance_px"
            ]
        ),
        "px",
    )

    print(
        "  Mean P95 dist  :",
        fmt(
            summary_metrics[
                "mean_p95_boundary_distance_px"
            ]
        ),
        "px",
    )

    print(
        "  Boundary Prec. :",
        fmt(
            summary_metrics[
                "mean_boundary_precision"
            ]
        ),
    )

    print(
        "  Boundary Recall:",
        fmt(
            summary_metrics[
                "mean_boundary_recall"
            ]
        ),
    )

    print(
        "  Boundary F1    :",
        fmt(
            summary_metrics[
                "mean_boundary_f1"
            ]
        ),
        f"@ {args.boundary_tolerance_px:.1f}px",
    )

    print()

    print(
        "Distance bins"
    )

    for row in distance_bins:

        if row[
            "num_frames"
        ] <= 0:
            continue

        print(
            "  {:>7s} m  "
            "n={:4d}  "
            "IoU={}  "
            "Dice={}  "
            "Area={}  "
            "Cent={}px  "
            "B-F1={}".format(
                row[
                    "distance_bin"
                ],

                row[
                    "num_frames"
                ],

                fmt(
                    row[
                        "mean_mask_iou"
                    ]
                ),

                fmt(
                    row[
                        "mean_dice"
                    ]
                ),

                fmt(
                    row[
                        "mean_area_ratio"
                    ]
                ),

                fmt(
                    row[
                        "mean_centroid_error_px"
                    ]
                ),

                fmt(
                    row[
                        "mean_boundary_f1"
                    ]
                ),
            )
        )

    print()

    print(
        "Worst IoU frames"
    )

    for row in worst_iou_rows:

        print(
            "  frame={:3d} "
            "depth={:6.2f}m "
            "IoU={} "
            "Dice={} "
            "centroid={}px".format(
                row[
                    "frame_idx"
                ],

                row[
                    "real_depth_m"
                ],

                fmt(
                    row[
                        "mask_iou"
                    ]
                ),

                fmt(
                    row[
                        "dice"
                    ]
                ),

                fmt(
                    row[
                        "centroid_error_px"
                    ]
                ),
            )
        )

    print()

    print(
        "Saved:",
        output_dir
        / "summary.json",
    )

    print(
        "Saved:",
        output_dir
        / "per_frame.csv",
    )

    print(
        "Saved:",
        output_dir
        / "distance_bins.csv",
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()