"""
qa_sprite_bank_360_v1.py

QA for one complete 360-degree HE sprite surface.

Expected input:
    one actor
    one distance
    one elevation
    angles 000 ... 359

Example:
    assets/sprite_bank_grabcut_test/
        tesla_360_d10_e10_qa/
            rgba/
                d_010p0_e_010p0_angle_000_rgba.png
                ...
                d_010p0_e_010p0_angle_359_rgba.png

Checks:
    - all 360 angles exist
    - empty/broken alpha
    - visible bbox width/height
    - alpha area
    - aspect ratio
    - alpha centroid
    - lower silhouette anchor
    - crop-edge contact
    - disconnected foreground components
    - neighbor-to-neighbor continuity
    - normalized silhouette IoU between neighboring angles

Outputs:
    qa_metrics.csv
    qa_summary.json
    flagged_angles.txt
    contact_sheets/contact_sheet_000_029.png
    ...
    flagged_review_*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ANGLE_RE = re.compile(
    r"angle_(\d{3})",
    re.IGNORECASE,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bank-dir",
        required=True,
        help=(
            "Bank directory containing rgba/, "
            "or the rgba directory itself."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "QA output directory. "
            "Default: <bank-dir>/qa_360"
        ),
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--sheet-size",
        type=int,
        default=30,
        help="Number of angles per contact sheet.",
    )

    parser.add_argument(
        "--sheet-cols",
        type=int,
        default=5,
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def relative_difference(
    a,
    b,
):
    """
    Symmetric relative difference.
    """

    a = float(
        a
    )

    b = float(
        b
    )

    denom = max(
        1e-9,
        (
            abs(
                a
            )
            +
            abs(
                b
            )
        )
        /
        2.0,
    )

    return (
        abs(
            a
            -
            b
        )
        /
        denom
    )


def robust_upper_threshold(
    values,
    minimum_threshold,
    mad_multiplier=6.0,
):
    """
    Robust outlier threshold:

        max(
            engineering minimum,
            median + k * MAD
        )

    This avoids flagging normal perspective changes while still
    finding sudden one-angle failures.
    """

    values = np.asarray(
        [
            float(v)
            for v in values
            if np.isfinite(
                v
            )
        ],
        dtype=np.float64,
    )

    if len(
        values
    ) == 0:

        return float(
            minimum_threshold
        )

    median = float(
        np.median(
            values
        )
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

    return max(
        float(
            minimum_threshold
        ),
        median
        +
        float(
            mad_multiplier
        )
        *
        robust_sigma,
    )


def extract_angle(
    path,
):

    match = ANGLE_RE.search(
        path.name
    )

    if match is None:
        return None

    return int(
        match.group(
            1
        )
    )


def resolve_rgba_dir(
    bank_dir,
):

    bank_dir = Path(
        bank_dir
    )

    if (
        bank_dir.name.lower()
        ==
        "rgba"
    ):

        return bank_dir

    candidate = (
        bank_dir
        /
        "rgba"
    )

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find rgba directory under: "
        f"{bank_dir}"
    )


# ============================================================
# Alpha / shape analysis
# ============================================================

def normalized_silhouette(
    alpha,
    alpha_threshold,
    canvas_size=256,
    content_size=224,
):
    """
    Normalize a tight silhouette onto a fixed canvas.

    Aspect ratio is preserved.

    This makes adjacent-view silhouette IoU meaningful even though
    source crops have different pixel dimensions.
    """

    binary = (
        alpha
        >
        int(
            alpha_threshold
        )
    ).astype(
        np.uint8
    )

    ys, xs = np.where(
        binary > 0
    )

    canvas = np.zeros(
        (
            canvas_size,
            canvas_size,
        ),
        dtype=np.uint8,
    )

    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):

        return canvas

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

    crop = binary[
        y1:
        y2 + 1,
        x1:
        x2 + 1,
    ]

    crop_h, crop_w = (
        crop.shape[
            :2
        ]
    )

    scale = min(
        float(
            content_size
        )
        /
        max(
            1,
            crop_w
        ),

        float(
            content_size
        )
        /
        max(
            1,
            crop_h
        ),
    )

    new_w = max(
        1,
        int(
            round(
                crop_w
                *
                scale
            )
        ),
    )

    new_h = max(
        1,
        int(
            round(
                crop_h
                *
                scale
            )
        ),
    )

    resized = cv2.resize(
        crop,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    x0 = (
        canvas_size
        -
        new_w
    ) // 2

    y0 = (
        canvas_size
        -
        new_h
    ) // 2

    canvas[
        y0:
        y0 + new_h,
        x0:
        x0 + new_w,
    ] = resized

    return canvas


def silhouette_iou(
    a,
    b,
):

    a = (
        a > 0
    )

    b = (
        b > 0
    )

    intersection = int(
        np.sum(
            a
            &
            b
        )
    )

    union = int(
        np.sum(
            a
            |
            b
        )
    )

    if union == 0:
        return 1.0

    return (
        float(
            intersection
        )
        /
        float(
            union
        )
    )


def analyze_sprite(
    path,
    angle,
    alpha_threshold,
):

    image = Image.open(
        path
    ).convert(
        "RGBA"
    )

    rgba = np.asarray(
        image
    )

    alpha = rgba[
        :,
        :,
        3
    ]

    image_h, image_w = (
        alpha.shape[
            :2
        ]
    )

    binary = (
        alpha
        >
        int(
            alpha_threshold
        )
    ).astype(
        np.uint8
    )

    ys, xs = np.where(
        binary > 0
    )

    result = {
        "angle_deg":
            int(
                angle
            ),

        "path":
            str(
                path
            ),

        "image_width_px":
            int(
                image_w
            ),

        "image_height_px":
            int(
                image_h
            ),

        "valid":
            False,

        "flags":
            [],
    }

    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):

        result[
            "flags"
        ].append(
            "EMPTY_ALPHA"
        )

        result[
            "silhouette"
        ] = np.zeros(
            (
                256,
                256,
            ),
            dtype=np.uint8,
        )

        return result

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

    visible_width = (
        x2
        -
        x1
        +
        1
    )

    visible_height = (
        y2
        -
        y1
        +
        1
    )

    alpha_pixels = int(
        np.sum(
            binary
        )
    )

    bbox_area = max(
        1,
        visible_width
        *
        visible_height,
    )

    fill_ratio = (
        float(
            alpha_pixels
        )
        /
        float(
            bbox_area
        )
    )

    aspect_ratio = (
        float(
            visible_width
        )
        /
        max(
            1.0,
            float(
                visible_height
            ),
        )
    )

    centroid_x = float(
        np.mean(
            xs
        )
    )

    centroid_y = float(
        np.mean(
            ys
        )
    )

    centroid_x_norm = (
        centroid_x
        -
        x1
    ) / max(
        1.0,
        float(
            visible_width
            -
            1
        ),
    )

    centroid_y_norm = (
        centroid_y
        -
        y1
    ) / max(
        1.0,
        float(
            visible_height
            -
            1
        ),
    )

    # --------------------------------------------------------
    # Production sprite anchor
    #
    # Match the actual generator:
    #
    #   anchor_x = midpoint of visible alpha bounds
    #   anchor_y = lowest visible alpha pixel
    #
    # Do NOT use the mean of the bottom-most pixels. For cars,
    # that quantity jumps between wheels during normal 1-degree
    # rotation and creates false QA alarms.
    # --------------------------------------------------------

    anchor_x = (
        float(
            x1
        )
        +
        float(
            x2
        )
    ) / 2.0

    anchor_y = float(
        y2
    )

    anchor_x_norm = (
        anchor_x
        -
        float(
            x1
        )
    ) / max(
        1.0,
        float(
            visible_width
            -
            1
        ),
    )

    anchor_y_norm = (
        anchor_y
        -
        float(
            y1
        )
    ) / max(
        1.0,
        float(
            visible_height
            -
            1
        ),
    )
    # --------------------------------------------------------
    # Crop margins
    # --------------------------------------------------------

    margin_left = int(
        x1
    )

    margin_right = int(
        image_w
        -
        1
        -
        x2
    )

    margin_top = int(
        y1
    )

    margin_bottom = int(
        image_h
        -
        1
        -
        y2
    )

    minimum_margin = min(
        margin_left,
        margin_right,
        margin_top,
        margin_bottom,
    )

    if minimum_margin <= 1:

        result[
            "flags"
        ].append(
            "ALPHA_TOUCHES_CROP_EDGE"
        )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    (
        component_count,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    component_areas = []

    for component in range(
        1,
        component_count,
    ):

        area = int(
            stats[
                component,
                cv2.CC_STAT_AREA,
            ]
        )

        component_areas.append(
            area
        )

    component_areas.sort(
        reverse=True
    )

    meaningful_components = 0

    second_component_fraction = 0.0

    if component_areas:

        largest = max(
            1,
            component_areas[
                0
            ],
        )

        meaningful_min = max(
            3,
            int(
                round(
                    largest
                    *
                    0.001
                )
            ),
        )

        meaningful_components = sum(
            area
            >=
            meaningful_min

            for area
            in component_areas
        )

        if len(
            component_areas
        ) >= 2:

            second_component_fraction = (
                float(
                    component_areas[
                        1
                    ]
                )
                /
                float(
                    largest
                )
            )

    if (
        second_component_fraction
        >
        0.03
    ):

        result[
            "flags"
        ].append(
            "LARGE_DISCONNECTED_COMPONENT"
        )

    result.update({
        "valid":
            True,

        "visible_x1":
            x1,

        "visible_y1":
            y1,

        "visible_x2":
            x2,

        "visible_y2":
            y2,

        "visible_width_px":
            int(
                visible_width
            ),

        "visible_height_px":
            int(
                visible_height
            ),

        "alpha_pixels":
            alpha_pixels,

        "fill_ratio":
            float(
                fill_ratio
            ),

        "aspect_ratio":
            float(
                aspect_ratio
            ),

        "centroid_x_norm":
            float(
                centroid_x_norm
            ),

        "centroid_y_norm":
            float(
                centroid_y_norm
            ),

        "anchor_x_norm":
            float(
                anchor_x_norm
            ),

        "anchor_y_norm":
            float(
                anchor_y_norm
            ),
        "margin_left_px":
            margin_left,

        "margin_right_px":
            margin_right,

        "margin_top_px":
            margin_top,

        "margin_bottom_px":
            margin_bottom,

        "minimum_margin_px":
            minimum_margin,

        "meaningful_component_count":
            int(
                meaningful_components
            ),

        "second_component_fraction":
            float(
                second_component_fraction
            ),

        "silhouette":
            normalized_silhouette(
                alpha,
                alpha_threshold,
            ),
    })

    return result


# ============================================================
# Contact sheets
# ============================================================

def build_contact_sheet(
    records,
    angles,
    output_path,
    sheet_cols,
):
    """
    White-background contact sheet.

    Flagged angles are labelled with FLAG.
    """

    if not angles:
        return

    cell_w = 340
    cell_h = 220
    label_h = 36

    rows = int(
        math.ceil(
            len(
                angles
            )
            /
            float(
                sheet_cols
            )
        )
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_cols
            *
            cell_w,

            rows
            *
            (
                cell_h
                +
                label_h
            ),
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    record_by_angle = {
        int(
            record[
                "angle_deg"
            ]
        ):
            record
        for record in records
    }

    for index, angle in enumerate(
        angles
    ):

        row = (
            index
            //
            sheet_cols
        )

        col = (
            index
            %
            sheet_cols
        )

        x0 = (
            col
            *
            cell_w
        )

        y0 = (
            row
            *
            (
                cell_h
                +
                label_h
            )
        )

        record = record_by_angle.get(
            int(
                angle
            )
        )

        if record is None:

            draw.text(
                (
                    x0 + 8,
                    y0 + 8,
                ),
                f"{angle:03d}  MISSING",
                fill="red",
            )

            continue

        flags = record.get(
            "flags",
            []
        )

        if flags:

            label = (
                f"{angle:03d}  FLAG"
            )

            label_color = (
                "red"
            )

        else:

            label = (
                f"{angle:03d}"
            )

            label_color = (
                "black"
            )

        draw.text(
            (
                x0 + 8,
                y0 + 8,
            ),
            label,
            fill=label_color,
        )

        try:

            rgba = Image.open(
                record[
                    "path"
                ]
            ).convert(
                "RGBA"
            )

            white = Image.new(
                "RGBA",
                rgba.size,
                (
                    255,
                    255,
                    255,
                    255,
                ),
            )

            white.alpha_composite(
                rgba
            )

            thumb = ImageOps.contain(
                white.convert(
                    "RGB"
                ),
                (
                    cell_w - 16,
                    cell_h - 16,
                ),
            )

            px = (
                x0
                +
                (
                    cell_w
                    -
                    thumb.width
                )
                //
                2
            )

            py = (
                y0
                +
                label_h
                +
                (
                    cell_h
                    -
                    thumb.height
                )
                //
                2
            )

            sheet.paste(
                thumb,
                (
                    px,
                    py,
                ),
            )

        except Exception as exc:

            draw.text(
                (
                    x0 + 8,
                    y0 + 60,
                ),
                f"ERROR: {exc}",
                fill="red",
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path
    )


# ============================================================
# Main QA
# ============================================================

def main():

    args = parse_args()

    bank_dir = Path(
        args.bank_dir
    )

    rgba_dir = resolve_rgba_dir(
        bank_dir
    )

    if args.output_dir is None:

        if (
            bank_dir.name.lower()
            ==
            "rgba"
        ):

            output_dir = (
                bank_dir.parent
                /
                "qa_360"
            )

        else:

            output_dir = (
                bank_dir
                /
                "qa_360"
            )

    else:

        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    contact_dir = (
        output_dir
        /
        "contact_sheets"
    )

    contact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 88)
    print(
        "HE SPRITE BANK 360 QA V1"
    )
    print("=" * 88)

    print(
        "RGBA:",
        rgba_dir.resolve(),
    )

    print(
        "QA output:",
        output_dir.resolve(),
    )

    # ========================================================
    # Discover files
    # ========================================================

    paths = sorted(
        rgba_dir.glob(
            "*_rgba.png"
        )
    )

    angle_to_paths = {}

    unrecognized = []

    for path in paths:

        angle = extract_angle(
            path
        )

        if angle is None:

            unrecognized.append(
                path
            )

            continue

        angle_to_paths.setdefault(
            int(
                angle
            ),
            [],
        ).append(
            path
        )

    duplicate_angles = sorted(
        angle
        for (
            angle,
            angle_paths,
        )
        in angle_to_paths.items()
        if len(
            angle_paths
        )
        >
        1
    )

    expected_angles = list(
        range(
            360
        )
    )

    missing_angles = [
        angle
        for angle
        in expected_angles
        if angle
        not in
        angle_to_paths
    ]

    print()
    print(
        "PNG files found:",
        len(
            paths
        ),
    )

    print(
        "Unique recognized angles:",
        len(
            angle_to_paths
        ),
        "/ 360",
    )

    print(
        "Missing angles:",
        len(
            missing_angles
        ),
    )

    print(
        "Duplicate angles:",
        len(
            duplicate_angles
        ),
    )

    # ========================================================
    # Analyze each angle
    # ========================================================

    records = []

    record_by_angle = {}

    for angle in expected_angles:

        if angle not in angle_to_paths:
            continue

        path = angle_to_paths[
            angle
        ][0]

        record = analyze_sprite(
            path=path,

            angle=angle,

            alpha_threshold=(
                args.alpha_threshold
            ),
        )

        records.append(
            record
        )

        record_by_angle[
            angle
        ] = record

    # ========================================================
    # Circular neighbor differences
    # ========================================================

    neighbor_rows = []

    for angle in expected_angles:

        current = record_by_angle.get(
            angle
        )

        previous_angle = (
            angle
            -
            1
        ) % 360

        previous = record_by_angle.get(
            previous_angle
        )

        if (
            current is None
            or
            previous is None
            or
            not current.get(
                "valid",
                False,
            )
            or
            not previous.get(
                "valid",
                False,
            )
        ):

            continue

        row = {
            "angle":
                angle,

            "previous_angle":
                previous_angle,

            "area_delta":
                relative_difference(
                    current[
                        "alpha_pixels"
                    ],
                    previous[
                        "alpha_pixels"
                    ],
                ),

            "width_delta":
                relative_difference(
                    current[
                        "visible_width_px"
                    ],
                    previous[
                        "visible_width_px"
                    ],
                ),

            "height_delta":
                relative_difference(
                    current[
                        "visible_height_px"
                    ],
                    previous[
                        "visible_height_px"
                    ],
                ),

            "aspect_delta":
                relative_difference(
                    current[
                        "aspect_ratio"
                    ],
                    previous[
                        "aspect_ratio"
                    ],
                ),

            "centroid_x_delta":
                abs(
                    current[
                        "centroid_x_norm"
                    ]
                    -
                    previous[
                        "centroid_x_norm"
                    ]
                ),

            "centroid_y_delta":
                abs(
                    current[
                        "centroid_y_norm"
                    ]
                    -
                    previous[
                        "centroid_y_norm"
                    ]
                ),

            "bottom_anchor_delta":
                abs(
                    current[
                        "bottom_anchor_x_norm"
                    ]
                    -
                    previous[
                        "bottom_anchor_x_norm"
                    ]
                ),

            "shape_iou":
                silhouette_iou(
                    current[
                        "silhouette"
                    ],
                    previous[
                        "silhouette"
                    ],
                ),
        }

        row[
            "shape_loss"
        ] = (
            1.0
            -
            row[
                "shape_iou"
            ]
        )

        neighbor_rows.append(
            row
        )

    # ========================================================
    # Learn robust thresholds from the 360-degree surface
    # ========================================================

    threshold_specs = {
        "area_delta": (
            0.06
        ),

        "width_delta": (
            0.04
        ),

        "height_delta": (
            0.04
        ),

        "aspect_delta": (
            0.06
        ),

        "centroid_x_delta": (
            0.025
        ),

        "centroid_y_delta": (
            0.025
        ),

        "shape_loss": (
            0.10
        ),
    }

    learned_thresholds = {}

    for (
        key,
        minimum_threshold,
    ) in threshold_specs.items():

        learned_thresholds[
            key
        ] = (
            robust_upper_threshold(
                [
                    row[
                        key
                    ]
                    for row
                    in neighbor_rows
                ],

                minimum_threshold=(
                    minimum_threshold
                ),
            )
        )

    print()
    print("-" * 88)
    print(
        "CONTINUITY THRESHOLDS"
    )
    print("-" * 88)

    for key in threshold_specs:

        print(
            f"{key:24s}: "
            f"{learned_thresholds[key]:.5f}"
        )

    # ========================================================
    # Apply continuity flags
    # ========================================================

    neighbor_by_angle = {
        int(
            row[
                "angle"
            ]
        ):
            row
        for row
        in neighbor_rows
    }

    continuity_flag_names = {
        "area_delta":
            "AREA_JUMP",

        "width_delta":
            "WIDTH_JUMP",

        "height_delta":
            "HEIGHT_JUMP",

        "aspect_delta":
            "ASPECT_JUMP",

        "centroid_x_delta":
            "CENTROID_X_JUMP",

        "centroid_y_delta":
            "CENTROID_Y_JUMP",

        "shape_loss":
            "SHAPE_JUMP",
    }

    for angle, record in record_by_angle.items():

        row = neighbor_by_angle.get(
            angle
        )

        if row is None:
            continue

        for (
            key,
            flag_name,
        ) in continuity_flag_names.items():

            if (
                row[
                    key
                ]
                >
                learned_thresholds[
                    key
                ]
            ):

                record[
                    "flags"
                ].append(
                    flag_name
                )

        record[
            "previous_angle"
        ] = int(
            row[
                "previous_angle"
            ]
        )

        record[
            "neighbor_area_delta"
        ] = float(
            row[
                "area_delta"
            ]
        )

        record[
            "neighbor_width_delta"
        ] = float(
            row[
                "width_delta"
            ]
        )

        record[
            "neighbor_height_delta"
        ] = float(
            row[
                "height_delta"
            ]
        )

        record[
            "neighbor_aspect_delta"
        ] = float(
            row[
                "aspect_delta"
            ]
        )

        record[
            "neighbor_centroid_x_delta"
        ] = float(
            row[
                "centroid_x_delta"
            ]
        )

        record[
            "neighbor_centroid_y_delta"
        ] = float(
            row[
                "centroid_y_delta"
            ]
        )

        record[
            "neighbor_bottom_anchor_delta"
        ] = float(
            row[
                "bottom_anchor_delta"
            ]
        )

        record[
            "neighbor_shape_iou"
        ] = float(
            row[
                "shape_iou"
            ]
        )

    # ========================================================
    # Contact sheets: 30 consecutive angles each
    # ========================================================

    sheet_size = int(
        args.sheet_size
    )

    for start_angle in range(
        0,
        360,
        sheet_size,
    ):

        end_angle = min(
            359,
            start_angle
            +
            sheet_size
            -
            1,
        )

        angles = list(
            range(
                start_angle,
                end_angle
                +
                1,
            )
        )

        output_path = (
            contact_dir
            /
            (
                f"contact_sheet_"
                f"{start_angle:03d}_"
                f"{end_angle:03d}.png"
            )
        )

        build_contact_sheet(
            records=records,

            angles=angles,

            output_path=(
                output_path
            ),

            sheet_cols=int(
                args.sheet_cols
            ),
        )

    # ========================================================
    # Flagged review sheets
    # ========================================================

    flagged_angles = sorted(
        angle
        for (
            angle,
            record,
        )
        in record_by_angle.items()
        if record[
            "flags"
        ]
    )

    review_angles = set()

    for angle in flagged_angles:

        for offset in (
            -2,
            -1,
            0,
            1,
            2,
        ):

            review_angles.add(
                (
                    angle
                    +
                    offset
                )
                %
                360
            )

    review_angles = sorted(
        review_angles
    )

    review_chunk_size = 30

    for chunk_index in range(
        0,
        len(
            review_angles
        ),
        review_chunk_size,
    ):

        chunk = review_angles[
            chunk_index:
            chunk_index
            +
            review_chunk_size
        ]

        sheet_number = (
            chunk_index
            //
            review_chunk_size
            +
            1
        )

        build_contact_sheet(
            records=records,

            angles=chunk,

            output_path=(
                output_dir
                /
                (
                    f"flagged_review_"
                    f"{sheet_number:02d}.png"
                )
            ),

            sheet_cols=int(
                args.sheet_cols
            ),
        )

    # ========================================================
    # CSV
    # ========================================================

    csv_path = (
        output_dir
        /
        "qa_metrics.csv"
    )

    csv_fields = [
        "angle_deg",
        "path",

        "image_width_px",
        "image_height_px",

        "visible_width_px",
        "visible_height_px",

        "alpha_pixels",
        "fill_ratio",
        "aspect_ratio",

        "centroid_x_norm",
        "centroid_y_norm",
        "bottom_anchor_x_norm",

        "margin_left_px",
        "margin_right_px",
        "margin_top_px",
        "margin_bottom_px",
        "minimum_margin_px",

        "meaningful_component_count",
        "second_component_fraction",

        "previous_angle",

        "neighbor_area_delta",
        "neighbor_width_delta",
        "neighbor_height_delta",
        "neighbor_aspect_delta",

        "neighbor_centroid_x_delta",
        "neighbor_centroid_y_delta",

        "neighbor_bottom_anchor_delta",
        "neighbor_shape_iou",

        "flags",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                csv_fields
            ),
        )

        writer.writeheader()

        for record in records:

            row = {}

            for field in csv_fields:

                if field == "flags":

                    row[
                        field
                    ] = ";".join(
                        record.get(
                            "flags",
                            [],
                        )
                    )

                else:

                    row[
                        field
                    ] = record.get(
                        field,
                        "",
                    )

            writer.writerow(
                row
            )

    # ========================================================
    # Flagged-angle text report
    # ========================================================

    flagged_path = (
        output_dir
        /
        "flagged_angles.txt"
    )

    with open(
        flagged_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "HE SPRITE BANK 360 QA\n"
        )

        f.write(
            "=" * 80
            +
            "\n\n"
        )

        f.write(
            f"Missing angles: "
            f"{missing_angles}\n"
        )

        f.write(
            f"Duplicate angles: "
            f"{duplicate_angles}\n\n"
        )

        for angle in flagged_angles:

            record = (
                record_by_angle[
                    angle
                ]
            )

            f.write(
                f"{angle:03d}: "
                +
                ", ".join(
                    record[
                        "flags"
                    ]
                )
                +
                "\n"
            )

    # ========================================================
    # JSON summary
    # ========================================================

    summary = {
        "bank_dir":
            str(
                bank_dir.resolve()
            ),

        "rgba_dir":
            str(
                rgba_dir.resolve()
            ),

        "png_count":
            int(
                len(
                    paths
                )
            ),

        "recognized_angle_count":
            int(
                len(
                    angle_to_paths
                )
            ),

        "missing_angles":
            missing_angles,

        "duplicate_angles":
            duplicate_angles,

        "unrecognized_files":
            [
                str(
                    path
                )
                for path
                in unrecognized
            ],

        "flagged_angle_count":
            int(
                len(
                    flagged_angles
                )
            ),

        "flagged_angles":
            flagged_angles,

        "continuity_thresholds":
            {
                key:
                    float(
                        value
                    )
                for (
                    key,
                    value,
                )
                in learned_thresholds.items()
            },
    }

    summary_path = (
        output_dir
        /
        "qa_summary.json"
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

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 88)
    print(
        "QA SUMMARY"
    )
    print("=" * 88)

    print(
        "Expected angles        : 360"
    )

    print(
        "Recognized angles      :",
        len(
            angle_to_paths
        ),
    )

    print(
        "Missing angles         :",
        len(
            missing_angles
        ),
    )

    print(
        "Duplicate angles       :",
        len(
            duplicate_angles
        ),
    )

    print(
        "Automatically flagged  :",
        len(
            flagged_angles
        ),
    )

    if flagged_angles:

        print(
            "Flagged:",
            " ".join(
                f"{a:03d}"
                for a
                in flagged_angles
            ),
        )

    print()
    print(
        "Metrics:",
        csv_path.resolve(),
    )

    print(
        "Flags:",
        flagged_path.resolve(),
    )

    print(
        "Summary:",
        summary_path.resolve(),
    )

    print(
        "Contact sheets:",
        contact_dir.resolve(),
    )

    print("=" * 88)


if __name__ == "__main__":

    main()