"""
analyze_tcp_actor_mask_equivalence_v1.py

Compare CARLA-native instance masks against HE-native alpha masks
from the TCP adversary-equivalence dump.

Important:
- CARLA masks are binary.
- HE masks are recovered alpha maps (0..255).
- HE is evaluated at multiple alpha thresholds so faint
  interpolation/antialiasing pixels do not distort geometry.
- Startup frames can be excluded explicitly.

This is a diagnostic comparison of matched probe indices.
For strict paper-level visual equivalence, we will later use an
exact same-state replay so ego pose is identical.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]


DEFAULT_ROOT = (
    HE_ROOT
    / "driving_models"
    / "TCP"
    / "outputs"
    / "tcp_native_equiv_smoke"
    / "tcp_cutin_001"
    / "native_equivalence_v1"
)


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path):

    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as fp:

        for line in fp:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(
                    line
                )
            )

    return rows


def load_gray(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image: {path}"
        )

    return image


def binary_geometry(mask):

    ys, xs = np.where(
        mask
    )

    if len(xs) == 0:

        return {
            "visible": False,
            "x1": None,
            "y1": None,
            "x2": None,
            "y2": None,
            "width": 0,
            "height": 0,
            "cx": None,
            "cy": None,
            "bottom_y": None,
            "area": 0,
        }

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

    return {
        "visible": True,

        "x1":
            x1,

        "y1":
            y1,

        "x2":
            x2,

        "y2":
            y2,

        "width":
            int(
                x2 - x1 + 1
            ),

        "height":
            int(
                y2 - y1 + 1
            ),

        "cx":
            float(
                xs.mean()
            ),

        "cy":
            float(
                ys.mean()
            ),

        "bottom_y":
            float(
                y2
            ),

        "area":
            int(
                mask.sum()
            ),
    }


def dice_iou(
    a,
    b,
):

    a = a.astype(
        bool
    )

    b = b.astype(
        bool
    )

    intersection = int(
        np.logical_and(
            a,
            b,
        ).sum()
    )

    union = int(
        np.logical_or(
            a,
            b,
        ).sum()
    )

    area_a = int(
        a.sum()
    )

    area_b = int(
        b.sum()
    )

    if union > 0:

        iou = (
            intersection
            /
            union
        )

    else:

        iou = float(
            "nan"
        )

    denom = (
        area_a
        +
        area_b
    )

    if denom > 0:

        dice = (
            2.0
            * intersection
            /
            denom
        )

    else:

        dice = float(
            "nan"
        )

    return (
        float(iou),
        float(dice),
    )


def boundary_mask(
    mask,
):

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
        >
        eroded
    )

    return boundary


def boundary_f1(
    reference,
    prediction,
    tolerance_px=2,
):

    ref_boundary = (
        boundary_mask(
            reference
        )
    )

    pred_boundary = (
        boundary_mask(
            prediction
        )
    )

    ref_count = int(
        ref_boundary.sum()
    )

    pred_count = int(
        pred_boundary.sum()
    )

    if (
        ref_count == 0
        or
        pred_count == 0
    ):

        return float(
            "nan"
        )

    radius = int(
        tolerance_px
    )

    kernel = np.ones(
        (
            2 * radius + 1,
            2 * radius + 1,
        ),
        dtype=np.uint8,
    )

    ref_dilated = (
        cv2.dilate(
            ref_boundary.astype(
                np.uint8
            ),
            kernel,
            iterations=1,
        )
        > 0
    )

    pred_dilated = (
        cv2.dilate(
            pred_boundary.astype(
                np.uint8
            ),
            kernel,
            iterations=1,
        )
        > 0
    )

    precision = (
        np.logical_and(
            pred_boundary,
            ref_dilated,
        ).sum()
        /
        pred_count
    )

    recall = (
        np.logical_and(
            ref_boundary,
            pred_dilated,
        ).sum()
        /
        ref_count
    )

    if (
        precision + recall
        <= 0.0
    ):

        return 0.0

    return float(
        2.0
        * precision
        * recall
        /
        (
            precision
            +
            recall
        )
    )


def finite_summary(
    values,
):

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    if len(array) == 0:

        return {
            "n": 0,
            "mean": None,
            "sample_std": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    return {
        "n":
            int(
                len(array)
            ),

        "mean":
            float(
                np.mean(
                    array
                )
            ),

        "sample_std":
            (
                float(
                    np.std(
                        array,
                        ddof=1,
                    )
                )
                if len(array) >= 2
                else 0.0
            ),

        "median":
            float(
                np.median(
                    array
                )
            ),

        "p95":
            float(
                np.percentile(
                    array,
                    95,
                )
            ),

        "min":
            float(
                np.min(
                    array
                )
            ),

        "max":
            float(
                np.max(
                    array
                )
            ),
    }


def fmt(
    summary,
    digits=4,
):

    if summary[
        "mean"
    ] is None:

        return "none"

    return (
        f"{summary['mean']:.{digits}f}"
        " +/- "
        f"{summary['sample_std']:.{digits}f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=str(
            DEFAULT_ROOT
        ),
    )

    parser.add_argument(
        "--skip-first",
        type=int,
        default=3,
        help=(
            "Ignore initial sensor/render warm-up frames."
        ),
    )

    parser.add_argument(
        "--he-thresholds",
        default="1,16,32,64,128",
        help=(
            "Comma-separated HE alpha thresholds."
        ),
    )

    parser.add_argument(
        "--boundary-tolerance-px",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).resolve()

    carla_root = (
        root
        / "carla"
    )

    he_root = (
        root
        / "he"
    )

    carla_jsonl = (
        carla_root
        / "native_equivalence_rows.jsonl"
    )

    he_jsonl = (
        he_root
        / "native_equivalence_rows.jsonl"
    )

    carla_rows = load_jsonl(
        carla_jsonl
    )

    he_rows = load_jsonl(
        he_jsonl
    )

    carla_by_idx = {
        int(
            row[
                "probe_idx"
            ]
        ):
            row

        for row
        in carla_rows
    }

    he_by_idx = {
        int(
            row[
                "probe_idx"
            ]
        ):
            row

        for row
        in he_rows
    }

    indices = sorted(
        set(
            carla_by_idx
        )
        &
        set(
            he_by_idx
        )
    )

    indices = [
        idx
        for idx
        in indices
        if idx
        >= args.skip_first
    ]

    thresholds = [
        int(
            value.strip()
        )
        for value
        in args.he_thresholds.split(
            ","
        )
        if value.strip()
    ]

    if args.output_dir is None:

        output_dir = (
            root
            / "analysis"
        )

    else:

        output_dir = Path(
            args.output_dir
        ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}

    # ========================================================
    # Threshold sweep
    # ========================================================

    for threshold in thresholds:

        per_frame = []

        for idx in indices:

            c_row = (
                carla_by_idx[
                    idx
                ]
            )

            h_row = (
                he_by_idx[
                    idx
                ]
            )

            c_mask_path = (
                carla_root
                / "masks_carla"
                / f"{idx:06d}.png"
            )

            h_mask_path = (
                he_root
                / "masks_he"
                / f"{idx:06d}.png"
            )

            c_gray = load_gray(
                c_mask_path
            )

            h_alpha = load_gray(
                h_mask_path
            )

            c_mask = (
                c_gray
                > 0
            )

            h_mask = (
                h_alpha
                >= threshold
            )

            c_geom = (
                binary_geometry(
                    c_mask
                )
            )

            h_geom = (
                binary_geometry(
                    h_mask
                )
            )

            if (
                not c_geom[
                    "visible"
                ]
                or
                not h_geom[
                    "visible"
                ]
            ):

                continue

            iou, dice = (
                dice_iou(
                    c_mask,
                    h_mask,
                )
            )

            width_ratio = (
                h_geom[
                    "width"
                ]
                /
                c_geom[
                    "width"
                ]
            )

            height_ratio = (
                h_geom[
                    "height"
                ]
                /
                c_geom[
                    "height"
                ]
            )

            area_ratio = (
                h_geom[
                    "area"
                ]
                /
                c_geom[
                    "area"
                ]
            )

            centroid_error = math.hypot(
                h_geom[
                    "cx"
                ]
                -
                c_geom[
                    "cx"
                ],

                h_geom[
                    "cy"
                ]
                -
                c_geom[
                    "cy"
                ],
            )

            cx_error = (
                h_geom[
                    "cx"
                ]
                -
                c_geom[
                    "cx"
                ]
            )

            bottom_error = (
                h_geom[
                    "bottom_y"
                ]
                -
                c_geom[
                    "bottom_y"
                ]
            )

            bf1 = boundary_f1(
                reference=
                    c_mask,

                prediction=
                    h_mask,

                tolerance_px=
                    args.boundary_tolerance_px,
            )

            row = {
                "probe_idx":
                    idx,

                "threshold":
                    threshold,

                "carla_t_s":
                    float(
                        c_row[
                            "t_s"
                        ]
                    ),

                "he_t_s":
                    float(
                        h_row[
                            "t_s"
                        ]
                    ),

                "carla_gap_m":
                    float(
                        c_row[
                            "bumper_gap_m"
                        ]
                    ),

                "he_gap_m":
                    float(
                        h_row[
                            "bumper_gap_m"
                        ]
                    ),

                "carla_width_px":
                    c_geom[
                        "width"
                    ],

                "he_width_px":
                    h_geom[
                        "width"
                    ],

                "width_ratio_he_carla":
                    width_ratio,

                "carla_height_px":
                    c_geom[
                        "height"
                    ],

                "he_height_px":
                    h_geom[
                        "height"
                    ],

                "height_ratio_he_carla":
                    height_ratio,

                "carla_area_px":
                    c_geom[
                        "area"
                    ],

                "he_area_px":
                    h_geom[
                        "area"
                    ],

                "area_ratio_he_carla":
                    area_ratio,

                "cx_error_he_minus_carla_px":
                    cx_error,

                "bottom_error_he_minus_carla_px":
                    bottom_error,

                "centroid_error_px":
                    centroid_error,

                "mask_iou":
                    iou,

                "dice":
                    dice,

                "boundary_f1":
                    bf1,
            }

            per_frame.append(
                row
            )

        metric_names = [
            "width_ratio_he_carla",
            "height_ratio_he_carla",
            "area_ratio_he_carla",
            "cx_error_he_minus_carla_px",
            "bottom_error_he_minus_carla_px",
            "centroid_error_px",
            "mask_iou",
            "dice",
            "boundary_f1",
        ]

        summary = {}

        for metric_name in metric_names:

            summary[
                metric_name
            ] = finite_summary(
                [
                    row[
                        metric_name
                    ]
                    for row
                    in per_frame
                ]
            )

        all_results[
            str(
                threshold
            )
        ] = {
            "threshold":
                threshold,

            "frame_count":
                len(
                    per_frame
                ),

            "summary":
                summary,

            "per_frame":
                per_frame,
        }

        # ----------------------------------------------------
        # Per-threshold CSV
        # ----------------------------------------------------

        csv_path = (
            output_dir
            /
            (
                "per_frame_threshold_"
                f"{threshold:03d}.csv"
            )
        )

        if per_frame:

            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as fp:

                writer = (
                    csv.DictWriter(
                        fp,
                        fieldnames=list(
                            per_frame[
                                0
                            ].keys()
                        ),
                    )
                )

                writer.writeheader()

                writer.writerows(
                    per_frame
                )

    # ========================================================
    # JSON
    # ========================================================

    output_json = (
        output_dir
        / "tcp_actor_mask_equivalence_summary.json"
    )

    with output_json.open(
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            {
                "schema":
                    "tcp_actor_mask_equivalence_v1",

                "root":
                    str(
                        root
                    ),

                "skip_first":
                    args.skip_first,

                "thresholds":
                    thresholds,

                "boundary_tolerance_px":
                    args.boundary_tolerance_px,

                "results":
                    all_results,
            },
            fp,
            indent=2,
        )

    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 88)
    print(
        "TCP CARLA <-> HE ACTOR MASK EQUIVALENCE V1"
    )
    print("=" * 88)

    print(
        "Matched frames:",
        len(
            indices
        ),
    )

    print(
        "Skipped startup frames:",
        args.skip_first,
    )

    print()

    for threshold in thresholds:

        result = all_results[
            str(
                threshold
            )
        ]

        s = result[
            "summary"
        ]

        print(
            f"HE alpha >= {threshold}"
        )

        print(
            "  width ratio HE/C :",
            fmt(
                s[
                    "width_ratio_he_carla"
                ]
            ),
        )

        print(
            "  height ratio HE/C:",
            fmt(
                s[
                    "height_ratio_he_carla"
                ]
            ),
        )

        print(
            "  area ratio HE/C  :",
            fmt(
                s[
                    "area_ratio_he_carla"
                ]
            ),
        )

        print(
            "  cx error HE-C    :",
            fmt(
                s[
                    "cx_error_he_minus_carla_px"
                ]
            ),
            "px",
        )

        print(
            "  bottom error HE-C:",
            fmt(
                s[
                    "bottom_error_he_minus_carla_px"
                ]
            ),
            "px",
        )

        print(
            "  centroid error   :",
            fmt(
                s[
                    "centroid_error_px"
                ]
            ),
            "px",
        )

        print(
            "  IoU              :",
            fmt(
                s[
                    "mask_iou"
                ]
            ),
        )

        print(
            "  Dice             :",
            fmt(
                s[
                    "dice"
                ]
            ),
        )

        print(
            "  Boundary F1      :",
            fmt(
                s[
                    "boundary_f1"
                ]
            ),
        )

        print()

    print(
        "Saved:",
        output_json,
    )

    print("=" * 88)


if __name__ == "__main__":

    main()