"""
analyze_production_anchor_continuity_v1.py

Analyze temporal/geometric continuity of the new production
CARLA HE sprite banks.

The production dataset contains:

    360 azimuth angles
    x distances
    x elevations

For every fixed (distance, elevation) ring, this script compares
adjacent angles:

    0 -> 1
    1 -> 2
    ...
    358 -> 359
    359 -> 0

The main goal is to determine whether the current HE sprite
anchor can jump even though the physical CARLA geometry remains
continuous.

Fields analyzed
---------------
Current HE anchor:
    anchor_x
    anchor_y

Normalized HE anchor:
    anchor_x / sprite_width
    anchor_y / sprite_height

Physical ground/support projection:
    projected_ground_anchor_x_px
    projected_ground_anchor_y_px

Physical projected bbox:
    projected_bbox_center_x_px
    projected_bbox_bottom_y_px

Visible-mask centroid:
    alpha_centroid_x_full_px
    alpha_centroid_y_full_px

Visible dimensions:
    visible_width_px
    visible_height_px

Physical reference dimensions:
    he_reference_projected_width_px
    he_reference_projected_height_px

Usage
-----
python analyze_production_anchor_continuity_v1.py ^
  --csv D:\\...\\view_matrix.csv ^
  --top 40

Outputs
-------
Console:
    dataset summary
    worst discontinuities

CSV:
    <bank>/anchor_continuity_analysis_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Production view_matrix.csv",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="Number of largest adjacent-angle jumps to print.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV. Default: "
            "<asset-bank>/anchor_continuity_analysis_v1.csv"
        ),
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def as_float(
    row,
    key,
):

    value = row.get(
        key,
        "",
    )

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    try:

        result = float(
            text
        )

    except Exception:

        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def delta(
    a,
    b,
):

    if (
        a is None
        or
        b is None
    ):
        return None

    return float(
        b - a
    )


def abs_or_none(
    value,
):

    if value is None:
        return None

    return abs(
        float(value)
    )


def fmt(
    value,
    digits=4,
):

    if value is None:
        return "NA"

    return (
        f"{float(value):.{digits}f}"
    )


def normalized(
    value,
    size,
):

    if (
        value is None
        or
        size is None
        or
        float(size) <= 1.0
    ):
        return None

    # Pixel coordinates span [0, size-1].
    return (
        float(value)
        /
        float(
            size - 1.0
        )
    )


# ============================================================
# Load
# ============================================================

def load_rows(
    csv_path,
):

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(
                f
            )
        )

    if not rows:

        raise RuntimeError(
            f"No rows loaded from {csv_path}"
        )

    return rows


# ============================================================
# Build normalized record
# ============================================================

def prepare_row(
    row,
):

    sprite_w = as_float(
        row,
        "sprite_width_px",
    )

    sprite_h = as_float(
        row,
        "sprite_height_px",
    )

    anchor_x = as_float(
        row,
        "anchor_x",
    )

    anchor_y = as_float(
        row,
        "anchor_y",
    )

    return {
        "raw":
            row,

        "angle_deg":
            int(
                round(
                    as_float(
                        row,
                        "angle_deg",
                    )
                )
            ) % 360,

        "distance_m":
            as_float(
                row,
                "distance_m",
            ),

        "elevation_deg":
            as_float(
                row,
                "elevation_deg",
            ),

        "sprite_width_px":
            sprite_w,

        "sprite_height_px":
            sprite_h,

        "anchor_x":
            anchor_x,

        "anchor_y":
            anchor_y,

        "anchor_x_norm":
            normalized(
                anchor_x,
                sprite_w,
            ),

        "anchor_y_norm":
            normalized(
                anchor_y,
                sprite_h,
            ),

        "ground_x":
            as_float(
                row,
                "projected_ground_anchor_x_px",
            ),

        "ground_y":
            as_float(
                row,
                "projected_ground_anchor_y_px",
            ),

        "bbox_center_x":
            as_float(
                row,
                "projected_bbox_center_x_px",
            ),

        "bbox_bottom_y":
            as_float(
                row,
                "projected_bbox_bottom_y_px",
            ),

        "alpha_centroid_x":
            as_float(
                row,
                "alpha_centroid_x_full_px",
            ),

        "alpha_centroid_y":
            as_float(
                row,
                "alpha_centroid_y_full_px",
            ),

        "visible_width":
            as_float(
                row,
                "visible_width_px",
            ),

        "visible_height":
            as_float(
                row,
                "visible_height_px",
            ),

        "reference_width":
            as_float(
                row,
                "he_reference_projected_width_px",
            ),

        "reference_height":
            as_float(
                row,
                "he_reference_projected_height_px",
            ),

        "bbox_center_distance_m":
            as_float(
                row,
                "bbox_center_distance_m",
            ),

        "nearest_bbox_depth_m":
            as_float(
                row,
                "nearest_bbox_depth_m",
            ),

        "farthest_bbox_depth_m":
            as_float(
                row,
                "farthest_bbox_depth_m",
            ),

        "qa_pass":
            str(
                row.get(
                    "qa_pass",
                    "",
                )
            ),

        "qa_flags":
            str(
                row.get(
                    "qa_flags",
                    "",
                )
            ),

        "qa_warnings":
            str(
                row.get(
                    "qa_warnings",
                    "",
                )
            ),
    }


# ============================================================
# Adjacent-angle comparison
# ============================================================

def compare_pair(
    current,
    nxt,
):

    result = {
        "distance_m":
            current[
                "distance_m"
            ],

        "elevation_deg":
            current[
                "elevation_deg"
            ],

        "angle_from_deg":
            current[
                "angle_deg"
            ],

        "angle_to_deg":
            nxt[
                "angle_deg"
            ],
    }

    fields = [
        "anchor_x",
        "anchor_y",

        "anchor_x_norm",
        "anchor_y_norm",

        "ground_x",
        "ground_y",

        "bbox_center_x",
        "bbox_bottom_y",

        "alpha_centroid_x",
        "alpha_centroid_y",

        "visible_width",
        "visible_height",

        "reference_width",
        "reference_height",

        "bbox_center_distance_m",
        "nearest_bbox_depth_m",
        "farthest_bbox_depth_m",
    ]

    for key in fields:

        value_a = current[
            key
        ]

        value_b = nxt[
            key
        ]

        d = delta(
            value_a,
            value_b,
        )

        result[
            key + "_from"
        ] = value_a

        result[
            key + "_to"
        ] = value_b

        result[
            "delta_" + key
        ] = d

        result[
            "abs_delta_" + key
        ] = abs_or_none(
            d
        )

    # Useful interpretation:
    #
    # If the normalized source anchor moves by 0.1,
    # then a 60 px rendered sprite can move by roughly
    # 6 px purely from source-anchor change.
    #
    # This is not the exact runtime affine transform,
    # but it is a convenient continuity metric.

    anchor_norm_jump = (
        result.get(
            "abs_delta_anchor_x_norm"
        )
    )

    for rendered_width in (
        40,
        60,
        80,
        100,
        150,
        200,
    ):

        key = (
            "estimated_anchor_shift_"
            f"at_{rendered_width}px"
        )

        if anchor_norm_jump is None:

            result[
                key
            ] = None

        else:

            result[
                key
            ] = (
                float(
                    anchor_norm_jump
                )
                *
                float(
                    rendered_width
                )
            )

    result[
        "from_qa_pass"
    ] = current[
        "qa_pass"
    ]

    result[
        "to_qa_pass"
    ] = nxt[
        "qa_pass"
    ]

    result[
        "from_qa_flags"
    ] = current[
        "qa_flags"
    ]

    result[
        "to_qa_flags"
    ] = nxt[
        "qa_flags"
    ]

    result[
        "from_qa_warnings"
    ] = current[
        "qa_warnings"
    ]

    result[
        "to_qa_warnings"
    ] = nxt[
        "qa_warnings"
    ]

    return result


# ============================================================
# Analysis
# ============================================================

def build_transitions(
    rows,
):

    prepared = [
        prepare_row(
            row
        )
        for row in rows
    ]

    rings = {}

    for row in prepared:

        key = (
            float(
                row[
                    "distance_m"
                ]
            ),
            float(
                row[
                    "elevation_deg"
                ]
            ),
        )

        rings.setdefault(
            key,
            {},
        )

        angle = int(
            row[
                "angle_deg"
            ]
        )

        if angle in rings[
            key
        ]:

            raise RuntimeError(
                "Duplicate view: "
                f"distance={key[0]} "
                f"elevation={key[1]} "
                f"angle={angle}"
            )

        rings[
            key
        ][
            angle
        ] = row

    transitions = []

    for (
        distance_m,
        elevation_deg,
    ), angle_map in sorted(
        rings.items()
    ):

        missing = [
            angle
            for angle in range(
                360
            )
            if angle
            not in angle_map
        ]

        if missing:

            raise RuntimeError(
                "Incomplete 360-degree ring: "
                f"d={distance_m} "
                f"e={elevation_deg}; "
                f"missing={missing[:20]}"
            )

        for angle in range(
            360
        ):

            next_angle = (
                angle + 1
            ) % 360

            transitions.append(
                compare_pair(
                    angle_map[
                        angle
                    ],
                    angle_map[
                        next_angle
                    ],
                )
            )

    return (
        prepared,
        rings,
        transitions,
    )


# ============================================================
# Reporting
# ============================================================

def print_top(
    transitions,
    field,
    top_n,
    title,
):

    metric = (
        "abs_delta_"
        + field
    )

    valid = [
        row
        for row in transitions
        if row.get(
            metric
        )
        is not None
    ]

    valid.sort(
        key=lambda row:
            row[
                metric
            ],
        reverse=True,
    )

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    for row in valid[
        :top_n
    ]:

        print(
            "d={:5.1f} "
            "e={:5.1f} "
            "a={:03d}->{:03d} "
            "{}: {} -> {} "
            "delta={}".format(
                float(
                    row[
                        "distance_m"
                    ]
                ),
                float(
                    row[
                        "elevation_deg"
                    ]
                ),
                int(
                    row[
                        "angle_from_deg"
                    ]
                ),
                int(
                    row[
                        "angle_to_deg"
                    ]
                ),
                field,
                fmt(
                    row[
                        field
                        + "_from"
                    ],
                    6,
                ),
                fmt(
                    row[
                        field
                        + "_to"
                    ],
                    6,
                ),
                fmt(
                    row[
                        "delta_"
                        + field
                    ],
                    6,
                ),
            )
        )


def print_anchor_summary(
    transitions,
    top_n,
):

    metric = (
        "abs_delta_anchor_x_norm"
    )

    valid = [
        row
        for row in transitions
        if row.get(
            metric
        )
        is not None
    ]

    valid.sort(
        key=lambda row:
            row[
                metric
            ],
        reverse=True,
    )

    print()
    print("=" * 118)
    print(
        "WORST NORMALIZED HE ANCHOR-X DISCONTINUITIES"
    )
    print("=" * 118)

    for row in valid[
        :top_n
    ]:

        print(
            "d={:5.1f} "
            "e={:5.1f} "
            "a={:03d}->{:03d} | "
            "anchorX {:8.3f}->{:8.3f} | "
            "norm {:8.5f}->{:8.5f} "
            "dNorm={:8.5f} | "
            "~shift@60px={:6.2f}px "
            "@100px={:6.2f}px "
            "@200px={:6.2f}px | "
            "ground dX={} | "
            "bboxCenter dX={}".format(
                float(
                    row[
                        "distance_m"
                    ]
                ),
                float(
                    row[
                        "elevation_deg"
                    ]
                ),
                int(
                    row[
                        "angle_from_deg"
                    ]
                ),
                int(
                    row[
                        "angle_to_deg"
                    ]
                ),
                float(
                    row[
                        "anchor_x_from"
                    ]
                ),
                float(
                    row[
                        "anchor_x_to"
                    ]
                ),
                float(
                    row[
                        "anchor_x_norm_from"
                    ]
                ),
                float(
                    row[
                        "anchor_x_norm_to"
                    ]
                ),
                float(
                    row[
                        "delta_anchor_x_norm"
                    ]
                ),
                float(
                    row[
                        "estimated_anchor_shift_at_60px"
                    ]
                ),
                float(
                    row[
                        "estimated_anchor_shift_at_100px"
                    ]
                ),
                float(
                    row[
                        "estimated_anchor_shift_at_200px"
                    ]
                ),
                fmt(
                    row.get(
                        "delta_ground_x"
                    ),
                    3,
                ),
                fmt(
                    row.get(
                        "delta_bbox_center_x"
                    ),
                    3,
                ),
            )
        )


def write_output(
    path,
    transitions,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not transitions:
        return

    fieldnames = list(
        transitions[
            0
        ].keys()
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
            transitions
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    csv_path = Path(
        args.csv
    ).resolve()

    rows = load_rows(
        csv_path
    )

    (
        prepared,
        rings,
        transitions,
    ) = build_transitions(
        rows
    )

    distances = sorted(
        {
            row[
                "distance_m"
            ]
            for row in prepared
        }
    )

    elevations = sorted(
        {
            row[
                "elevation_deg"
            ]
            for row in prepared
        }
    )

    angles = sorted(
        {
            row[
                "angle_deg"
            ]
            for row in prepared
        }
    )

    qa_failed = [
        row
        for row in prepared
        if str(
            row[
                "qa_pass"
            ]
        ).strip().lower()
        not in {
            "true",
            "1",
        }
    ]

    distance_errors = []

    for row in prepared:

        requested = (
            row[
                "distance_m"
            ]
        )

        measured = (
            row[
                "bbox_center_distance_m"
            ]
        )

        if (
            requested is not None
            and
            measured is not None
        ):

            distance_errors.append(
                abs(
                    requested
                    -
                    measured
                )
            )

    print()
    print("=" * 100)
    print(
        "PRODUCTION HE ASSET CONTINUITY ANALYSIS"
    )
    print("=" * 100)

    print(
        "CSV:",
        csv_path,
    )

    print(
        "Rows:",
        len(
            rows
        ),
    )

    print(
        "Angles:",
        len(
            angles
        ),
        angles[
            :5
        ],
        "...",
        angles[
            -5:
        ],
    )

    print(
        "Distances:",
        distances,
    )

    print(
        "Elevations:",
        elevations,
    )

    print(
        "Rings:",
        len(
            rings
        ),
    )

    print(
        "Adjacent transitions:",
        len(
            transitions
        ),
    )

    print(
        "QA failed views:",
        len(
            qa_failed
        ),
    )

    if distance_errors:

        print(
            "Max |requested - measured "
            "bbox-center distance|:",
            max(
                distance_errors
            ),
            "m",
        )

    print_anchor_summary(
        transitions,
        args.top,
    )

    print_top(
        transitions,
        "ground_x",
        min(
            args.top,
            20,
        ),
        (
            "WORST PHYSICAL GROUND-ANCHOR "
            "X CHANGES"
        ),
    )

    print_top(
        transitions,
        "bbox_center_x",
        min(
            args.top,
            20,
        ),
        (
            "WORST PROJECTED PHYSICAL "
            "BBOX-CENTER X CHANGES"
        ),
    )

    print_top(
        transitions,
        "alpha_centroid_x",
        min(
            args.top,
            20,
        ),
        (
            "WORST VISIBLE-ALPHA "
            "CENTROID X CHANGES"
        ),
    )

    print_top(
        transitions,
        "visible_width",
        min(
            args.top,
            20,
        ),
        (
            "WORST VISIBLE-WIDTH CHANGES"
        ),
    )

    print_top(
        transitions,
        "reference_width",
        min(
            args.top,
            20,
        ),
        (
            "WORST PHYSICAL REFERENCE-WIDTH "
            "CHANGES"
        ),
    )

    if args.output:

        output_path = Path(
            args.output
        ).resolve()

    else:

        output_path = (
            csv_path.parent
            /
            "anchor_continuity_analysis_v1.csv"
        )

    write_output(
        output_path,
        transitions,
    )

    print()
    print("=" * 100)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 100)

    print(
        "Output:",
        output_path,
    )


if __name__ == "__main__":
    main()