"""
qa_sprite_view_matrix_v1.py

QA for HE multi-view sprite banks.

Expected filenames:

    d_005p0_e_010p0_angle_000_rgba.png
    d_010p0_e_010p0_angle_180_rgba.png
    d_020p0_e_010p0_angle_359_rgba.png

The important difference from qa_sprite_bank_360_v1.py is that
ANGLE ALONE is not the unique sprite key.

Unique view:
    (distance_m, elevation_deg, angle_deg)

Continuity is evaluated independently within each:
    (distance_m, elevation_deg)
360-degree ring.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


VIEW_RE = re.compile(
    r"d_(\d+)p(\d+)_"
    r"e_(\d+)p(\d+)_"
    r"angle_(\d{3})_rgba\.png$",
    re.IGNORECASE,
)


# ============================================================
# Parsing
# ============================================================

def decode_float(integer_text, fraction_text):
    return float(
        "{}.{}".format(
            integer_text,
            fraction_text,
        )
    )


def parse_view_filename(path):
    m = VIEW_RE.search(
        Path(path).name
    )

    if not m:
        return None

    distance_m = decode_float(
        m.group(1),
        m.group(2),
    )

    elevation_deg = decode_float(
        m.group(3),
        m.group(4),
    )

    angle_deg = int(
        m.group(5)
    ) % 360

    return {
        "distance_m": distance_m,
        "elevation_deg": elevation_deg,
        "angle_deg": angle_deg,
    }


# ============================================================
# Image analysis
# ============================================================

def analyze_rgba(
    path,
    alpha_threshold=10,
):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise RuntimeError(
            "Could not read: {}".format(path)
        )

    if (
        image.ndim != 3
        or image.shape[2] != 4
    ):
        raise RuntimeError(
            "Expected RGBA/BGRA PNG: {}".format(path)
        )

    alpha = image[:, :, 3]

    binary = (
        alpha > int(alpha_threshold)
    ).astype(np.uint8)

    ys, xs = np.where(
        binary > 0
    )

    h, w = binary.shape

    result = {
        "image_width_px": int(w),
        "image_height_px": int(h),
        "valid": False,
        "flags": [],
    }

    if len(xs) == 0:
        result["flags"].append(
            "EMPTY_ALPHA"
        )
        return result

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    visible_w = (
        x2 - x1 + 1
    )

    visible_h = (
        y2 - y1 + 1
    )

    alpha_pixels = int(
        np.sum(binary)
    )

    bbox_area = max(
        1,
        visible_w * visible_h
    )

    fill_ratio = (
        alpha_pixels
        /
        float(bbox_area)
    )

    centroid_x = float(
        np.mean(xs)
    )

    centroid_y = float(
        np.mean(ys)
    )

    # Bottom contact pixels.
    bottom_xs = xs[
        ys >= (y2 - 2)
    ]

    if len(bottom_xs) > 0:
        bottom_anchor_x = float(
            np.mean(bottom_xs)
        )
    else:
        bottom_anchor_x = float(
            (x1 + x2) / 2.0
        )

    bottom_anchor_x_norm = (
        (
            bottom_anchor_x
            - x1
        )
        /
        float(
            max(1, visible_w - 1)
        )
    )

    # Connected components.
    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    component_areas = []

    for label in range(
        1,
        component_count,
    ):
        component_areas.append(
            int(
                stats[
                    label,
                    cv2.CC_STAT_AREA
                ]
            )
        )

    component_areas.sort(
        reverse=True
    )

    largest_component_area = (
        component_areas[0]
        if component_areas
        else 0
    )

    disconnected_pixels = max(
        0,
        alpha_pixels
        -
        largest_component_area
    )

    disconnected_fraction = (
        disconnected_pixels
        /
        float(
            max(1, alpha_pixels)
        )
    )

    significant_components = sum(
        1
        for area in component_areas
        if area
        >= max(
            4,
            int(
                0.001
                *
                alpha_pixels
            ),
        )
    )

    # Alpha touching PNG boundary can indicate clipping.
    touches_left = bool(
        np.any(binary[:, 0])
    )

    touches_right = bool(
        np.any(binary[:, -1])
    )

    touches_top = bool(
        np.any(binary[0, :])
    )

    touches_bottom = bool(
        np.any(binary[-1, :])
    )

    # Normalize silhouette into fixed canvas for angular continuity.
    visible = binary[
        y1:y2 + 1,
        x1:x2 + 1,
    ]

    target = np.zeros(
        (256, 256),
        dtype=np.uint8,
    )

    scale = min(
        240.0 / visible_w,
        240.0 / visible_h,
    )

    rw = max(
        1,
        int(round(
            visible_w * scale
        )),
    )

    rh = max(
        1,
        int(round(
            visible_h * scale
        )),
    )

    resized = cv2.resize(
        visible,
        (rw, rh),
        interpolation=cv2.INTER_NEAREST,
    )

    ox = (
        256 - rw
    ) // 2

    # Bottom align so road-contact geometry matters.
    oy = (
        248 - rh
    )

    oy = max(
        0,
        min(
            256 - rh,
            oy,
        )
    )

    target[
        oy:oy + rh,
        ox:ox + rw,
    ] = resized

    result.update({
        "valid": True,

        "alpha_pixels":
            alpha_pixels,

        "visible_x1":
            x1,

        "visible_y1":
            y1,

        "visible_x2":
            x2,

        "visible_y2":
            y2,

        "visible_width_px":
            int(visible_w),

        "visible_height_px":
            int(visible_h),

        "aspect_ratio":
            float(
                visible_w
                /
                float(
                    max(1, visible_h)
                )
            ),

        "fill_ratio":
            float(fill_ratio),

        "centroid_x_norm":
            float(
                (
                    centroid_x
                    - x1
                )
                /
                max(
                    1,
                    visible_w - 1
                )
            ),

        "centroid_y_norm":
            float(
                (
                    centroid_y
                    - y1
                )
                /
                max(
                    1,
                    visible_h - 1
                )
            ),

        "bottom_anchor_x_norm":
            float(
                bottom_anchor_x_norm
            ),

        "component_count":
            int(
                len(component_areas)
            ),

        "significant_components":
            int(
                significant_components
            ),

        "largest_component_area":
            int(
                largest_component_area
            ),

        "disconnected_fraction":
            float(
                disconnected_fraction
            ),

        "touches_left":
            touches_left,

        "touches_right":
            touches_right,

        "touches_top":
            touches_top,

        "touches_bottom":
            touches_bottom,

        "silhouette":
            target,
    })

    return result


# ============================================================
# Comparison
# ============================================================

def circular_angle_difference(a, b):
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


def relative_difference(a, b):
    a = float(a)
    b = float(b)

    denom = max(
        1e-6,
        abs(a),
        abs(b),
    )

    return abs(
        a - b
    ) / denom


def silhouette_iou(a, b):
    aa = a > 0
    bb = b > 0

    intersection = int(
        np.sum(
            aa & bb
        )
    )

    union = int(
        np.sum(
            aa | bb
        )
    )

    if union == 0:
        return 1.0

    return (
        intersection
        /
        float(union)
    )


def neighbor_metrics(
    previous,
    current,
):
    return {
        "width_delta":
            relative_difference(
                previous[
                    "visible_width_px"
                ],
                current[
                    "visible_width_px"
                ],
            ),

        "height_delta":
            relative_difference(
                previous[
                    "visible_height_px"
                ],
                current[
                    "visible_height_px"
                ],
            ),

        "area_delta":
            relative_difference(
                previous[
                    "alpha_pixels"
                ],
                current[
                    "alpha_pixels"
                ],
            ),

        "anchor_delta":
            abs(
                previous[
                    "bottom_anchor_x_norm"
                ]
                -
                current[
                    "bottom_anchor_x_norm"
                ]
            ),

        "shape_iou":
            silhouette_iou(
                previous[
                    "silhouette"
                ],
                current[
                    "silhouette"
                ],
            ),
    }


# ============================================================
# Robust thresholds
# ============================================================

def robust_limit(
    values,
    multiplier=6.0,
    minimum=None,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return (
            float(minimum)
            if minimum is not None
            else math.inf
        )

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(
                values
                -
                median
            )
        )
    )

    robust_sigma = (
        1.4826
        *
        mad
    )

    limit = (
        median
        +
        multiplier
        *
        robust_sigma
    )

    if minimum is not None:
        limit = max(
            float(minimum),
            limit,
        )

    return float(limit)


# ============================================================
# QA
# ============================================================

def run_qa(
    bank_dir,
    output_dir,
    alpha_threshold,
):
    bank_dir = Path(
        bank_dir
    )

    if (
        bank_dir.name.lower()
        ==
        "rgba"
    ):
        rgba_dir = bank_dir
        bank_root = (
            bank_dir.parent
        )
    else:
        bank_root = bank_dir
        rgba_dir = (
            bank_dir
            /
            "rgba"
        )

    if not rgba_dir.exists():
        raise FileNotFoundError(
            "RGBA directory not found: {}".format(
                rgba_dir
            )
        )

    output_dir = Path(
        output_dir
        if output_dir
        else (
            bank_root
            /
            "qa_view_matrix"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = sorted(
        rgba_dir.glob(
            "*_rgba.png"
        )
    )

    print()
    print("=" * 80)
    print(
        "HE SPRITE VIEW MATRIX QA V1"
    )
    print("=" * 80)

    print(
        "RGBA:",
        rgba_dir
    )

    print(
        "QA output:",
        output_dir
    )

    print()
    print(
        "PNG files found:",
        len(paths)
    )

    records = []

    invalid_names = []

    for idx, path in enumerate(
        paths,
        start=1,
    ):
        view = parse_view_filename(
            path
        )

        if view is None:
            invalid_names.append(
                str(path)
            )
            continue

        analysis = analyze_rgba(
            path=path,
            alpha_threshold=alpha_threshold,
        )

        record = {
            **view,

            "path":
                str(path),

            **analysis,
        }

        records.append(
            record
        )

        if (
            idx % 100 == 0
            or idx == len(paths)
        ):
            print(
                "Analyzed {}/{}".format(
                    idx,
                    len(paths),
                )
            )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    keys = [
        (
            r["distance_m"],
            r["elevation_deg"],
            r["angle_deg"],
        )
        for r in records
    ]

    unique_keys = set(
        keys
    )

    duplicate_view_count = (
        len(keys)
        -
        len(unique_keys)
    )

    distances = sorted(
        set(
            r["distance_m"]
            for r in records
        )
    )

    elevations = sorted(
        set(
            r["elevation_deg"]
            for r in records
        )
    )

    rings = {}

    for record in records:
        ring_key = (
            record["distance_m"],
            record["elevation_deg"],
        )

        rings.setdefault(
            ring_key,
            {}
        )[
            record["angle_deg"]
        ] = record

    missing_views = []

    for distance in distances:
        for elevation in elevations:
            ring = rings.get(
                (
                    distance,
                    elevation
                ),
                {},
            )

            for angle in range(
                360
            ):
                if (
                    angle
                    not in ring
                ):
                    missing_views.append(
                        (
                            distance,
                            elevation,
                            angle,
                        )
                    )

    # --------------------------------------------------------
    # Per-sprite immediate flags
    # --------------------------------------------------------

    for record in records:

        flags = record[
            "flags"
        ]

        if not record[
            "valid"
        ]:
            continue

        if (
            record[
                "disconnected_fraction"
            ]
            >
            0.01
        ):
            flags.append(
                "DISCONNECTED_ALPHA"
            )

        if (
            record[
                "significant_components"
            ]
            >
            3
        ):
            flags.append(
                "MANY_COMPONENTS"
            )

        if (
            record["touches_left"]
            or
            record["touches_right"]
            or
            record["touches_top"]
            or
            record["touches_bottom"]
        ):
            flags.append(
                "ALPHA_TOUCHES_IMAGE_BORDER"
            )

    # --------------------------------------------------------
    # Angular continuity PER RING
    # --------------------------------------------------------

    neighbor_rows = []

    for (
        distance,
        elevation
    ), ring in sorted(
        rings.items()
    ):

        if len(ring) != 360:
            continue

        for angle in range(
            360
        ):
            next_angle = (
                angle + 1
            ) % 360

            current = ring[
                angle
            ]

            nxt = ring[
                next_angle
            ]

            if (
                not current[
                    "valid"
                ]
                or
                not nxt[
                    "valid"
                ]
            ):
                continue

            metrics = neighbor_metrics(
                current,
                nxt,
            )

            neighbor_rows.append({
                "distance_m":
                    distance,

                "elevation_deg":
                    elevation,

                "angle_a":
                    angle,

                "angle_b":
                    next_angle,

                **metrics,
            })

    width_limit = robust_limit(
        [
            r["width_delta"]
            for r in neighbor_rows
        ],
        minimum=0.05,
    )

    height_limit = robust_limit(
        [
            r["height_delta"]
            for r in neighbor_rows
        ],
        minimum=0.05,
    )

    area_limit = robust_limit(
        [
            r["area_delta"]
            for r in neighbor_rows
        ],
        minimum=0.08,
    )

    anchor_limit = robust_limit(
        [
            r["anchor_delta"]
            for r in neighbor_rows
        ],
        minimum=0.08,
    )

    # Shape IoU is inverted:
    # low value = suspicious.
    iou_values = np.asarray(
        [
            r["shape_iou"]
            for r in neighbor_rows
        ],
        dtype=np.float64,
    )

    if len(iou_values):
        iou_median = float(
            np.median(
                iou_values
            )
        )

        iou_mad = float(
            np.median(
                np.abs(
                    iou_values
                    -
                    iou_median
                )
            )
        )

        iou_limit = min(
            0.80,
            iou_median
            -
            6.0
            *
            1.4826
            *
            iou_mad,
        )

        iou_limit = max(
            0.20,
            iou_limit,
        )

    else:
        iou_limit = 0.0

    flagged_transitions = []

    for row in neighbor_rows:

        flags = []

        if (
            row["width_delta"]
            >
            width_limit
        ):
            flags.append(
                "WIDTH_JUMP"
            )

        if (
            row["height_delta"]
            >
            height_limit
        ):
            flags.append(
                "HEIGHT_JUMP"
            )

        if (
            row["area_delta"]
            >
            area_limit
        ):
            flags.append(
                "ALPHA_AREA_JUMP"
            )

        if (
            row["anchor_delta"]
            >
            anchor_limit
        ):
            flags.append(
                "ANCHOR_JUMP"
            )

        if (
            row["shape_iou"]
            <
            iou_limit
        ):
            flags.append(
                "SHAPE_JUMP"
            )

        if flags:

            row["flags"] = flags

            flagged_transitions.append(
                row
            )

    # --------------------------------------------------------
    # Save per-sprite CSV
    # --------------------------------------------------------

    csv_path = (
        output_dir
        /
        "sprite_metrics.csv"
    )

    fieldnames = [
        "distance_m",
        "elevation_deg",
        "angle_deg",
        "valid",
        "alpha_pixels",
        "visible_width_px",
        "visible_height_px",
        "aspect_ratio",
        "fill_ratio",
        "centroid_x_norm",
        "centroid_y_norm",
        "bottom_anchor_x_norm",
        "component_count",
        "significant_components",
        "disconnected_fraction",
        "touches_left",
        "touches_right",
        "touches_top",
        "touches_bottom",
        "flags",
        "path",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:

            writer.writerow({
                key: (
                    "|".join(
                        record["flags"]
                    )
                    if key == "flags"
                    else record.get(
                        key,
                        ""
                    )
                )
                for key in fieldnames
            })

    transition_csv = (
        output_dir
        /
        "flagged_transitions.csv"
    )

    transition_fields = [
        "distance_m",
        "elevation_deg",
        "angle_a",
        "angle_b",
        "width_delta",
        "height_delta",
        "area_delta",
        "anchor_delta",
        "shape_iou",
        "flags",
    ]

    with open(
        transition_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=transition_fields,
        )

        writer.writeheader()

        for row in flagged_transitions:

            writer.writerow({
                key: (
                    "|".join(
                        row["flags"]
                    )
                    if key == "flags"
                    else row.get(
                        key,
                        ""
                    )
                )
                for key in transition_fields
            })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    flagged_sprites = [
        r
        for r in records
        if r["flags"]
    ]

    summary = {
        "png_count":
            len(paths),

        "recognized_view_count":
            len(records),

        "unique_view_count":
            len(unique_keys),

        "duplicate_view_count":
            duplicate_view_count,

        "distances_m":
            distances,

        "elevations_deg":
            elevations,

        "ring_count":
            len(rings),

        "missing_view_count":
            len(missing_views),

        "invalid_filename_count":
            len(invalid_names),

        "flagged_sprite_count":
            len(flagged_sprites),

        "flagged_transition_count":
            len(flagged_transitions),

        "thresholds": {
            "width_delta":
                width_limit,

            "height_delta":
                height_limit,

            "area_delta":
                area_limit,

            "anchor_delta":
                anchor_limit,

            "shape_iou_min":
                iou_limit,
        },
    }

    summary_path = (
        output_dir
        /
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        "STRUCTURE"
    )
    print("=" * 80)

    print(
        "Recognized views:",
        len(records)
    )

    print(
        "Unique views:",
        len(unique_keys)
    )

    print(
        "Duplicate views:",
        duplicate_view_count
    )

    print(
        "Distances:",
        distances
    )

    print(
        "Elevations:",
        elevations
    )

    print(
        "Rings:",
        len(rings)
    )

    print(
        "Missing views:",
        len(missing_views)
    )

    print()
    print("=" * 80)
    print(
        "QUALITY"
    )
    print("=" * 80)

    print(
        "Flagged sprites:",
        len(flagged_sprites),
        "/",
        len(records),
    )

    print(
        "Flagged angular transitions:",
        len(flagged_transitions),
        "/",
        len(neighbor_rows),
    )

    print()
    print(
        "Thresholds:"
    )

    print(
        "  width delta :",
        width_limit
    )

    print(
        "  height delta:",
        height_limit
    )

    print(
        "  area delta  :",
        area_limit
    )

    print(
        "  anchor delta:",
        anchor_limit
    )

    print(
        "  shape IoU   :",
        iou_limit
    )

    print()
    print(
        "Saved:",
        csv_path
    )

    print(
        "Saved:",
        transition_csv
    )

    print(
        "Saved:",
        summary_path
    )

    print()
    print(
        "[QA] DONE"
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bank-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    run_qa(
        bank_dir=args.bank_dir,
        output_dir=args.output_dir,
        alpha_threshold=args.alpha_threshold,
    )


if __name__ == "__main__":
    main()