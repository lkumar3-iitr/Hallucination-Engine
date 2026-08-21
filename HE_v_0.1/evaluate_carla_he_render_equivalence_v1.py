"""
evaluate_carla_he_render_equivalence_v1.py

CARLA <-> Hallucination Engine rendering-equivalence evaluator.

Inputs
------
1. CARLA projected actor bbox JSONL produced by record_carla_he_pair.py
2. HE compositor metadata.json

Metrics
-------
Visibility:
    accuracy
    precision
    recall
    F1
    TP / TN / FP / FN

Geometry:
    center-x error
    bottom-y error
    width error
    height error
    2D center error
    relative width / height error

Overlap:
    bbox IoU
    IoU >= 0.50
    IoU >= 0.75

Temporal:
    frame-to-frame change of the CARLA-vs-HE residual

Distance:
    metrics grouped into depth bins

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

import numpy as np


# ============================================================
# Loading
# ============================================================

def load_real_bbox_jsonl(path):
    rows = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rec = json.loads(line)

            frame_idx = int(
                rec["recorded_frame_idx"]
            )

            bbox = rec.get(
                "bbox",
                {},
            )

            rows[frame_idx] = {
                "frame_idx":
                    frame_idx,

                "visible":
                    bool(
                        bbox.get(
                            "visible",
                            False,
                        )
                    ),

                "cx":
                    float(
                        bbox.get(
                            "cx",
                            math.nan,
                        )
                    ),

                "bottom_y":
                    float(
                        bbox.get(
                            "bottom_y",
                            math.nan,
                        )
                    ),

                "width":
                    float(
                        bbox.get(
                            "box_width",
                            math.nan,
                        )
                    ),

                "height":
                    float(
                        bbox.get(
                            "box_height",
                            math.nan,
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


def load_he_metadata(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        meta = json.load(f)

    rows = {}

    for frame in meta.get(
        "frames",
        [],
    ):
        frame_idx = int(
            frame["frame_idx"]
        )

        adversaries = frame.get(
            "adversaries",
            [],
        )

        if not adversaries:
            rows[frame_idx] = {
                "frame_idx":
                    frame_idx,

                "visible":
                    False,

                "rendered":
                    False,

                "cx":
                    math.nan,

                "bottom_y":
                    math.nan,

                "width":
                    math.nan,

                "height":
                    math.nan,

                "z_m":
                    math.nan,
            }

            continue

        adv = adversaries[0]

        box = adv.get(
            "box",
            {},
        )

        rows[frame_idx] = {
            "frame_idx":
                frame_idx,

            "visible":
                bool(
                    box.get(
                        "visible",
                        False,
                    )
                ),

            "rendered":
                bool(
                    adv.get(
                        "rendered",
                        False,
                    )
                ),

            "cx":
                float(
                    box.get(
                        "cx",
                        math.nan,
                    )
                ),

            "bottom_y":
                float(
                    box.get(
                        "bottom_y",
                        math.nan,
                    )
                ),

            "width":
                float(
                    box.get(
                        "box_width",
                        math.nan,
                    )
                ),

            "height":
                float(
                    box.get(
                        "box_height",
                        math.nan,
                    )
                ),

            "z_m":
                float(
                    box.get(
                        "z_m",
                        math.nan,
                    )
                ),
        }

    return rows, meta


# ============================================================
# Math helpers
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
    if abs(
        float(denominator)
    ) < 1e-9:
        return math.nan

    return (
        float(numerator)
        / float(denominator)
    )


# ============================================================
# Bounding boxes
# ============================================================

def bbox_from_anchor(
    cx,
    bottom_y,
    width,
    height,
):
    """
    Convert our HE/CARLA representation

        cx
        bottom_y
        width
        height

    into x1,y1,x2,y2.
    """

    x1 = (
        float(cx)
        - float(width) / 2.0
    )

    x2 = (
        float(cx)
        + float(width) / 2.0
    )

    y2 = float(
        bottom_y
    )

    y1 = (
        float(bottom_y)
        - float(height)
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def bbox_iou(
    box_a,
    box_b,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        iw
        * ih
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 1e-9:
        return math.nan

    return (
        intersection
        / union
    )


# ============================================================
# Per-frame comparison
# ============================================================

def build_rows(
    real_rows,
    he_rows,
    image_width,
    image_height,
):
    common_frames = sorted(
        set(real_rows.keys())
        & set(he_rows.keys())
    )

    rows = []

    previous_geometry_row = None

    for frame_idx in common_frames:

        real = real_rows[
            frame_idx
        ]

        he = he_rows[
            frame_idx
        ]

        real_visible = bool(
            real["visible"]
        )

        he_visible = (
            bool(
                he["visible"]
            )
            and bool(
                he.get(
                    "rendered",
                    False,
                )
            )
        )

        row = {
            "frame_idx":
                frame_idx,

            "real_visible":
                int(real_visible),

            "he_visible":
                int(he_visible),

            "visibility_match":
                int(
                    real_visible
                    == he_visible
                ),

            "real_depth_m":
                real["depth_m"],

            "he_z_m":
                he["z_m"],

            "real_cx":
                real["cx"],

            "he_cx":
                he["cx"],

            "real_bottom_y":
                real["bottom_y"],

            "he_bottom_y":
                he["bottom_y"],

            "real_width":
                real["width"],

            "he_width":
                he["width"],

            "real_height":
                real["height"],

            "he_height":
                he["height"],
        }

        if (
            real_visible
            and he_visible
            and all(
                finite(v)
                for v in [
                    real["cx"],
                    real["bottom_y"],
                    real["width"],
                    real["height"],
                    he["cx"],
                    he["bottom_y"],
                    he["width"],
                    he["height"],
                ]
            )
        ):
            err_cx = (
                he["cx"]
                - real["cx"]
            )

            err_bottom_y = (
                he["bottom_y"]
                - real["bottom_y"]
            )

            err_width = (
                he["width"]
                - real["width"]
            )

            err_height = (
                he["height"]
                - real["height"]
            )

            real_cy = (
                real["bottom_y"]
                - real["height"] / 2.0
            )

            he_cy = (
                he["bottom_y"]
                - he["height"] / 2.0
            )

            center_error_px = math.hypot(
                he["cx"] - real["cx"],
                he_cy - real_cy,
            )

            real_box = bbox_from_anchor(
                real["cx"],
                real["bottom_y"],
                real["width"],
                real["height"],
            )

            he_box = bbox_from_anchor(
                he["cx"],
                he["bottom_y"],
                he["width"],
                he["height"],
            )

            iou = bbox_iou(
                real_box,
                he_box,
            )

            row.update(
                {
                    "err_cx_px":
                        err_cx,

                    "abs_err_cx_px":
                        abs(err_cx),

                    "err_bottom_y_px":
                        err_bottom_y,

                    "abs_err_bottom_y_px":
                        abs(err_bottom_y),

                    "err_width_px":
                        err_width,

                    "abs_err_width_px":
                        abs(err_width),

                    "err_height_px":
                        err_height,

                    "abs_err_height_px":
                        abs(err_height),

                    "center_error_px":
                        center_error_px,

                    "cx_error_image_norm":
                        abs(err_cx)
                        / float(image_width),

                    "bottom_y_error_image_norm":
                        abs(err_bottom_y)
                        / float(image_height),

                    "width_relative_error":
                        safe_ratio(
                            abs(err_width),
                            real["width"],
                        ),

                    "height_relative_error":
                        safe_ratio(
                            abs(err_height),
                            real["height"],
                        ),

                    "bbox_iou":
                        iou,
                }
            )

            # ----------------------------------------------------
            # Temporal residual change
            #
            # This does not measure object motion itself.
            # It measures how quickly the CARLA-vs-HE error changes.
            # ----------------------------------------------------

            if (
                previous_geometry_row
                is not None
                and previous_geometry_row[
                    "frame_idx"
                ]
                == frame_idx - 1
            ):
                row[
                    "delta_err_cx_px"
                ] = (
                    err_cx
                    - previous_geometry_row[
                        "err_cx_px"
                    ]
                )

                row[
                    "delta_err_bottom_y_px"
                ] = (
                    err_bottom_y
                    - previous_geometry_row[
                        "err_bottom_y_px"
                    ]
                )

                row[
                    "delta_err_width_px"
                ] = (
                    err_width
                    - previous_geometry_row[
                        "err_width_px"
                    ]
                )

                row[
                    "delta_err_height_px"
                ] = (
                    err_height
                    - previous_geometry_row[
                        "err_height_px"
                    ]
                )

            else:
                row[
                    "delta_err_cx_px"
                ] = math.nan

                row[
                    "delta_err_bottom_y_px"
                ] = math.nan

                row[
                    "delta_err_width_px"
                ] = math.nan

                row[
                    "delta_err_height_px"
                ] = math.nan

            previous_geometry_row = row

        else:
            for key in [
                "err_cx_px",
                "abs_err_cx_px",
                "err_bottom_y_px",
                "abs_err_bottom_y_px",
                "err_width_px",
                "abs_err_width_px",
                "err_height_px",
                "abs_err_height_px",
                "center_error_px",
                "cx_error_image_norm",
                "bottom_y_error_image_norm",
                "width_relative_error",
                "height_relative_error",
                "bbox_iou",
                "delta_err_cx_px",
                "delta_err_bottom_y_px",
                "delta_err_width_px",
                "delta_err_height_px",
            ]:
                row[key] = math.nan

            previous_geometry_row = None

        rows.append(
            row
        )

    return rows


# ============================================================
# Summary
# ============================================================

def summarize_geometry(
    rows,
):
    if not rows:
        return {
            "num_frames":
                0
        }

    def values(key):
        return [
            r[key]
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

        "mae_cx_px":
            safe_mean(
                values(
                    "abs_err_cx_px"
                )
            ),

        "median_abs_cx_px":
            safe_median(
                values(
                    "abs_err_cx_px"
                )
            ),

        "p95_abs_cx_px":
            safe_percentile(
                values(
                    "abs_err_cx_px"
                ),
                95,
            ),

        "mae_bottom_y_px":
            safe_mean(
                values(
                    "abs_err_bottom_y_px"
                )
            ),

        "median_abs_bottom_y_px":
            safe_median(
                values(
                    "abs_err_bottom_y_px"
                )
            ),

        "p95_abs_bottom_y_px":
            safe_percentile(
                values(
                    "abs_err_bottom_y_px"
                ),
                95,
            ),

        "mae_width_px":
            safe_mean(
                values(
                    "abs_err_width_px"
                )
            ),

        "mae_height_px":
            safe_mean(
                values(
                    "abs_err_height_px"
                )
            ),

        "mean_signed_cx_px":
            safe_mean(
                values(
                    "err_cx_px"
                )
            ),

        "mean_signed_bottom_y_px":
            safe_mean(
                values(
                    "err_bottom_y_px"
                )
            ),

        "mean_signed_width_px":
            safe_mean(
                values(
                    "err_width_px"
                )
            ),

        "mean_signed_height_px":
            safe_mean(
                values(
                    "err_height_px"
                )
            ),

        "mean_center_error_px":
            safe_mean(
                values(
                    "center_error_px"
                )
            ),

        "mean_width_relative_error":
            safe_mean(
                values(
                    "width_relative_error"
                )
            ),

        "mean_height_relative_error":
            safe_mean(
                values(
                    "height_relative_error"
                )
            ),

        "mean_bbox_iou":
            safe_mean(
                values(
                    "bbox_iou"
                )
            ),

        "median_bbox_iou":
            safe_median(
                values(
                    "bbox_iou"
                )
            ),

        "p10_bbox_iou":
            safe_percentile(
                values(
                    "bbox_iou"
                ),
                10,
            ),

        "fraction_iou_ge_0_50":
            safe_mean(
                [
                    1.0
                    if x >= 0.50
                    else 0.0
                    for x in values(
                        "bbox_iou"
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
                        "bbox_iou"
                    )
                ]
            ),

        "temporal_mae_delta_cx_px":
            safe_mean(
                [
                    abs(x)
                    for x in values(
                        "delta_err_cx_px"
                    )
                ]
            ),

        "temporal_mae_delta_bottom_y_px":
            safe_mean(
                [
                    abs(x)
                    for x in values(
                        "delta_err_bottom_y_px"
                    )
                ]
            ),

        "temporal_mae_delta_width_px":
            safe_mean(
                [
                    abs(x)
                    for x in values(
                        "delta_err_width_px"
                    )
                ]
            ),

        "temporal_mae_delta_height_px":
            safe_mean(
                [
                    abs(x)
                    for x in values(
                        "delta_err_height_px"
                    )
                ]
            ),
    }


def build_visibility_summary(
    rows,
):
    tp = sum(
        1
        for r in rows
        if r["real_visible"]
        and r["he_visible"]
    )

    tn = sum(
        1
        for r in rows
        if not r["real_visible"]
        and not r["he_visible"]
    )

    fp = sum(
        1
        for r in rows
        if not r["real_visible"]
        and r["he_visible"]
    )

    fn = sum(
        1
        for r in rows
        if r["real_visible"]
        and not r["he_visible"]
    )

    total = (
        tp
        + tn
        + fp
        + fn
    )

    accuracy = safe_ratio(
        tp + tn,
        total,
    )

    precision = safe_ratio(
        tp,
        tp + fp,
    )

    recall = safe_ratio(
        tp,
        tp + fn,
    )

    if (
        finite(precision)
        and finite(recall)
        and precision + recall > 1e-9
    ):
        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    else:
        f1 = math.nan

    return {
        "tp":
            tp,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,
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


def build_distance_bins(
    geometry_rows,
):
    output = []

    for (
        label,
        low,
        high,
    ) in DISTANCE_BINS:

        selected = [
            r
            for r in geometry_rows
            if finite(
                r["real_depth_m"]
            )
            and float(
                r["real_depth_m"]
            ) >= low
            and float(
                r["real_depth_m"]
            ) < high
        ]

        summary = summarize_geometry(
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
# Output
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


def fmt(x):
    if not finite(x):
        return "nan"

    return (
        f"{float(x):.4f}"
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--real-bbox",
        required=True,
    )

    parser.add_argument(
        "--he-metadata",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--min-real-depth",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=720,
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    real_rows = load_real_bbox_jsonl(
        args.real_bbox
    )

    he_rows, he_meta = load_he_metadata(
        args.he_metadata
    )

    rows = build_rows(
        real_rows=real_rows,
        he_rows=he_rows,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    visibility_summary = (
        build_visibility_summary(
            rows
        )
    )

    geometry_rows = [
        r
        for r in rows
        if r["real_visible"]
        and r["he_visible"]
        and finite(
            r["real_depth_m"]
        )
        and float(
            r["real_depth_m"]
        )
        >= float(
            args.min_real_depth
        )
        and finite(
            r["bbox_iou"]
        )
    ]

    geometry_summary = (
        summarize_geometry(
            geometry_rows
        )
    )

    distance_bins = (
        build_distance_bins(
            geometry_rows
        )
    )

    summary = {
        "scenario":
            he_meta.get(
                "scenario_id",
                "unknown",
            ),

        "real_bbox":
            str(
                args.real_bbox
            ),

        "he_metadata":
            str(
                args.he_metadata
            ),

        "image_width":
            args.image_width,

        "image_height":
            args.image_height,

        "min_real_depth_m":
            args.min_real_depth,

        "num_common_frames":
            len(rows),

        "visibility":
            visibility_summary,

        "geometry":
            geometry_summary,

        "distance_bins":
            distance_bins,
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
        "=" * 76
    )
    print(
        "CARLA <-> HE RENDER EQUIVALENCE V1"
    )
    print(
        "=" * 76
    )

    print(
        "Scenario:",
        summary["scenario"],
    )

    print(
        "Common frames:",
        len(rows),
    )

    print(
        "Geometry frames:",
        geometry_summary[
            "num_frames"
        ],
    )

    print()

    print(
        "Visibility"
    )

    print(
        "  TP / TN / FP / FN:",
        visibility_summary["tp"],
        "/",
        visibility_summary["tn"],
        "/",
        visibility_summary["fp"],
        "/",
        visibility_summary["fn"],
    )

    print(
        "  Accuracy :",
        fmt(
            visibility_summary[
                "accuracy"
            ]
        ),
    )

    print(
        "  Precision:",
        fmt(
            visibility_summary[
                "precision"
            ]
        ),
    )

    print(
        "  Recall   :",
        fmt(
            visibility_summary[
                "recall"
            ]
        ),
    )

    print(
        "  F1       :",
        fmt(
            visibility_summary[
                "f1"
            ]
        ),
    )

    print()

    print(
        "Geometry"
    )

    print(
        "  MAE cx       :",
        fmt(
            geometry_summary[
                "mae_cx_px"
            ]
        ),
        "px",
    )

    print(
        "  MAE bottom_y :",
        fmt(
            geometry_summary[
                "mae_bottom_y_px"
            ]
        ),
        "px",
    )

    print(
        "  MAE width    :",
        fmt(
            geometry_summary[
                "mae_width_px"
            ]
        ),
        "px",
    )

    print(
        "  MAE height   :",
        fmt(
            geometry_summary[
                "mae_height_px"
            ]
        ),
        "px",
    )

    print(
        "  Center error :",
        fmt(
            geometry_summary[
                "mean_center_error_px"
            ]
        ),
        "px",
    )

    print()

    print(
        "Overlap"
    )

    print(
        "  Mean IoU     :",
        fmt(
            geometry_summary[
                "mean_bbox_iou"
            ]
        ),
    )

    print(
        "  Median IoU   :",
        fmt(
            geometry_summary[
                "median_bbox_iou"
            ]
        ),
    )

    print(
        "  P10 IoU      :",
        fmt(
            geometry_summary[
                "p10_bbox_iou"
            ]
        ),
    )

    print(
        "  IoU >= 0.50  :",
        fmt(
            geometry_summary[
                "fraction_iou_ge_0_50"
            ]
        ),
    )

    print(
        "  IoU >= 0.75  :",
        fmt(
            geometry_summary[
                "fraction_iou_ge_0_75"
            ]
        ),
    )

    print()

    print(
        "Relative geometry"
    )

    print(
        "  Width error  :",
        fmt(
            geometry_summary[
                "mean_width_relative_error"
            ]
        ),
    )

    print(
        "  Height error :",
        fmt(
            geometry_summary[
                "mean_height_relative_error"
            ]
        ),
    )

    print()

    print(
        "Temporal residual change"
    )

    print(
        "  cx       :",
        fmt(
            geometry_summary[
                "temporal_mae_delta_cx_px"
            ]
        ),
        "px/frame",
    )

    print(
        "  bottom_y :",
        fmt(
            geometry_summary[
                "temporal_mae_delta_bottom_y_px"
            ]
        ),
        "px/frame",
    )

    print(
        "  width    :",
        fmt(
            geometry_summary[
                "temporal_mae_delta_width_px"
            ]
        ),
        "px/frame",
    )

    print(
        "  height   :",
        fmt(
            geometry_summary[
                "temporal_mae_delta_height_px"
            ]
        ),
        "px/frame",
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
            "  {:>7s} m  n={:4d}  "
            "IoU={}  cx={}  by={}  w={}  h={}".format(
                row[
                    "distance_bin"
                ],
                row[
                    "num_frames"
                ],
                fmt(
                    row[
                        "mean_bbox_iou"
                    ]
                ),
                fmt(
                    row[
                        "mae_cx_px"
                    ]
                ),
                fmt(
                    row[
                        "mae_bottom_y_px"
                    ]
                ),
                fmt(
                    row[
                        "mae_width_px"
                    ]
                ),
                fmt(
                    row[
                        "mae_height_px"
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
        "=" * 76
    )


if __name__ == "__main__":
    main()