"""
compare_sprite_native_geometry_continuous_v1.py

Offline diagnostic for generalized HE visual geometry.

Compares:

    1. CARLA instance-mask geometry              [ground truth]
    2. Current render-dimension proxy            [4.2 x 1.8 x 1.5]
    3. Sprite-native nearest geometry
    4. Sprite-native continuous geometry

Continuous sprite-native geometry interpolates in:

    viewpoint angle     : circular linear interpolation
    elevation           : linear interpolation
    depth               : inverse-depth interpolation

No rendering, CARLA execution, or TCP execution is performed.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


# ============================================================
# Basic utilities
# ============================================================

def focal_length_px(
    image_width_px,
    horizontal_fov_deg,
):
    return (
        float(image_width_px)
        /
        (
            2.0
            *
            math.tan(
                math.radians(
                    float(horizontal_fov_deg)
                )
                /
                2.0
            )
        )
    )


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


def lerp(
    a,
    b,
    t,
):
    return (
        float(a)
        +
        float(t)
        *
        (
            float(b)
            -
            float(a)
        )
    )


# ============================================================
# Load sprite-native measurements
# ============================================================

def load_geometry_table(path):

    table = {}

    angles = set()
    elevations = set()
    distances = set()

    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as fp:

        reader = csv.DictReader(
            fp
        )

        for row in reader:

            angle = int(
                float(
                    row[
                        "angle_deg"
                    ]
                )
            )

            elevation = float(
                row[
                    "elevation_deg"
                ]
            )

            distance = float(
                row[
                    "distance_m"
                ]
            )

            source_fx = float(
                row[
                    "fx_px"
                ]
            )

            source_fy = float(
                row[
                    "fy_px"
                ]
            )

            alpha_w = float(
                row[
                    "alpha_width_px"
                ]
            )

            alpha_h = float(
                row[
                    "alpha_height_px"
                ]
            )

            # Normalize apparent image geometry by focal length.
            #
            # This makes the geometry portable to another camera.
            q_width = (
                alpha_w
                /
                source_fx
            )

            q_height = (
                alpha_h
                /
                source_fy
            )

            key = (
                angle,
                elevation,
                distance,
            )

            table[
                key
            ] = {
                "q_width":
                    q_width,

                "q_height":
                    q_height,
            }

            angles.add(
                angle
            )

            elevations.add(
                elevation
            )

            distances.add(
                distance
            )

    return {
        "table":
            table,

        "angles":
            sorted(
                angles
            ),

        "elevations":
            sorted(
                elevations
            ),

        "distances":
            sorted(
                distances
            ),
    }


# ============================================================
# Distance interpolation
# ============================================================

def bracket_linear(
    value,
    available,
):
    """
    Find lower / upper anchors.

    Values outside the available range are clamped for this
    diagnostic. Our current TCP probe is inside the range.
    """

    value = float(
        value
    )

    available = sorted(
        float(x)
        for x in available
    )

    if value <= available[0]:
        return (
            available[0],
            available[0],
            0.0,
        )

    if value >= available[-1]:
        return (
            available[-1],
            available[-1],
            0.0,
        )

    for idx in range(
        len(available) - 1
    ):

        low = available[
            idx
        ]

        high = available[
            idx + 1
        ]

        if (
            low
            <=
            value
            <=
            high
        ):

            if abs(
                high - low
            ) < 1e-12:

                t = 0.0

            else:

                t = (
                    value
                    -
                    low
                ) / (
                    high
                    -
                    low
                )

            return (
                low,
                high,
                t,
            )

    raise RuntimeError(
        "Could not bracket value."
    )


def interpolate_distance_inverse_depth(
    geometry,
    angle_deg,
    elevation_deg,
    depth_m,
):
    """
    Interpolate normalized geometry q in inverse-depth space.

    q = pixels / focal_length
    """

    table = geometry[
        "table"
    ]

    distances = geometry[
        "distances"
    ]

    z = float(
        depth_m
    )

    z0, z1, _ = bracket_linear(
        z,
        distances,
    )

    key0 = (
        int(
            angle_deg
        ),
        float(
            elevation_deg
        ),
        float(
            z0
        ),
    )

    key1 = (
        int(
            angle_deg
        ),
        float(
            elevation_deg
        ),
        float(
            z1
        ),
    )

    g0 = table.get(
        key0
    )

    g1 = table.get(
        key1
    )

    if (
        g0 is None
        or
        g1 is None
    ):
        return None

    if abs(
        z1 - z0
    ) < 1e-12:

        return {
            "q_width":
                float(
                    g0[
                        "q_width"
                    ]
                ),

            "q_height":
                float(
                    g0[
                        "q_height"
                    ]
                ),

            "distance_low_m":
                z0,

            "distance_high_m":
                z1,

            "distance_weight":
                0.0,
        }

    # --------------------------------------------------------
    # Interpolate using inverse depth:
    #
    #     u = 1 / Z
    # --------------------------------------------------------

    u = (
        1.0
        /
        z
    )

    u0 = (
        1.0
        /
        z0
    )

    u1 = (
        1.0
        /
        z1
    )

    t = (
        u - u0
    ) / (
        u1 - u0
    )

    q_width = lerp(
        g0[
            "q_width"
        ],
        g1[
            "q_width"
        ],
        t,
    )

    q_height = lerp(
        g0[
            "q_height"
        ],
        g1[
            "q_height"
        ],
        t,
    )

    return {
        "q_width":
            q_width,

        "q_height":
            q_height,

        "distance_low_m":
            z0,

        "distance_high_m":
            z1,

        "distance_weight":
            float(
                t
            ),
    }


# ============================================================
# Continuous viewpoint interpolation
# ============================================================

def circular_angle_bracket(
    angle_deg,
):
    """
    The production bank contains every integer angle 0..359.

    Example:

        348.43
            ->
        348, 349, 0.43

    Wraps correctly:

        359.7
            ->
        359, 0, 0.7
    """

    angle = (
        float(
            angle_deg
        )
        %
        360.0
    )

    low = int(
        math.floor(
            angle
        )
    )

    high = (
        low + 1
    ) % 360

    t = (
        angle
        -
        math.floor(
            angle
        )
    )

    return (
        low,
        high,
        float(
            t
        ),
    )


def continuous_geometry(
    geometry,
    viewpoint_angle_deg,
    elevation_deg,
    depth_m,
):
    """
    Trilinear-style interpolation:

        angle:
            circular linear interpolation

        elevation:
            linear interpolation

        distance:
            inverse-depth interpolation
    """

    angle0, angle1, angle_t = (
        circular_angle_bracket(
            viewpoint_angle_deg
        )
    )

    elev0, elev1, elev_t = (
        bracket_linear(
            elevation_deg,
            geometry[
                "elevations"
            ],
        )
    )

    # --------------------------------------------------------
    # Evaluate geometry at the four angle/elevation corners.
    # Each corner performs its own inverse-depth interpolation.
    # --------------------------------------------------------

    g00 = (
        interpolate_distance_inverse_depth(
            geometry=
                geometry,

            angle_deg=
                angle0,

            elevation_deg=
                elev0,

            depth_m=
                depth_m,
        )
    )

    g10 = (
        interpolate_distance_inverse_depth(
            geometry=
                geometry,

            angle_deg=
                angle1,

            elevation_deg=
                elev0,

            depth_m=
                depth_m,
        )
    )

    g01 = (
        interpolate_distance_inverse_depth(
            geometry=
                geometry,

            angle_deg=
                angle0,

            elevation_deg=
                elev1,

            depth_m=
                depth_m,
        )
    )

    g11 = (
        interpolate_distance_inverse_depth(
            geometry=
                geometry,

            angle_deg=
                angle1,

            elevation_deg=
                elev1,

            depth_m=
                depth_m,
        )
    )

    if any(
        value is None
        for value in [
            g00,
            g10,
            g01,
            g11,
        ]
    ):
        return None

    # --------------------------------------------------------
    # First interpolate horizontally in viewpoint angle.
    # --------------------------------------------------------

    q_width_e0 = lerp(
        g00[
            "q_width"
        ],
        g10[
            "q_width"
        ],
        angle_t,
    )

    q_height_e0 = lerp(
        g00[
            "q_height"
        ],
        g10[
            "q_height"
        ],
        angle_t,
    )

    q_width_e1 = lerp(
        g01[
            "q_width"
        ],
        g11[
            "q_width"
        ],
        angle_t,
    )

    q_height_e1 = lerp(
        g01[
            "q_height"
        ],
        g11[
            "q_height"
        ],
        angle_t,
    )

    # --------------------------------------------------------
    # Then interpolate vertically in elevation.
    # --------------------------------------------------------

    q_width = lerp(
        q_width_e0,
        q_width_e1,
        elev_t,
    )

    q_height = lerp(
        q_height_e0,
        q_height_e1,
        elev_t,
    )

    return {
        "q_width":
            q_width,

        "q_height":
            q_height,

        "angle_low_deg":
            angle0,

        "angle_high_deg":
            angle1,

        "angle_weight":
            angle_t,

        "elevation_low_deg":
            elev0,

        "elevation_high_deg":
            elev1,

        "elevation_weight":
            elev_t,

        "distance_low_m":
            g00[
                "distance_low_m"
            ],

        "distance_high_m":
            g00[
                "distance_high_m"
            ],

        "distance_weight":
            g00[
                "distance_weight"
            ],
    }


# ============================================================
# Statistics
# ============================================================

def summarize(
    name,
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            actual
        )
        &
        np.isfinite(
            predicted
        )
        &
        (
            actual > 0
        )
    )

    actual = actual[
        valid
    ]

    predicted = predicted[
        valid
    ]

    if len(
        actual
    ) == 0:

        print(
            name,
            ": no valid samples",
        )

        return

    error = (
        predicted
        -
        actual
    )

    abs_error = np.abs(
        error
    )

    relative_error = (
        abs_error
        /
        actual
    )

    ratio = (
        predicted
        /
        actual
    )

    print(
        name
    )

    print(
        "  samples       :",
        len(
            actual
        ),
    )

    print(
        "  MAE [px]      :",
        f"{np.mean(abs_error):.4f}",
    )

    print(
        "  P95 error [px]:",
        f"{np.percentile(abs_error, 95):.4f}",
    )

    print(
        "  bias [px]     :",
        f"{np.mean(error):+.4f}",
    )

    print(
        "  mean ratio    :",
        f"{np.mean(ratio):.4f}",
    )

    print(
        "  mean rel err  :",
        f"{100.0 * np.mean(relative_error):.2f}%"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--carla-jsonl",
        required=True,
    )

    parser.add_argument(
        "--he-jsonl",
        required=True,
    )

    parser.add_argument(
        "--geometry-csv",
        required=True,
    )

    parser.add_argument(
        "--skip-first",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--target-width",
        type=int,
        default=900,
    )

    parser.add_argument(
        "--target-height",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--target-fov",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "driving_models/TCP/outputs/"
            "sprite_native_continuous_v1/"
            "comparison_rows.csv"
        ),
    )

    args = parser.parse_args()

    geometry = (
        load_geometry_table(
            args.geometry_csv
        )
    )

    carla_rows = load_jsonl(
        args.carla_jsonl
    )

    he_rows = load_jsonl(
        args.he_jsonl
    )

    carla_by_probe = {
        int(
            row[
                "probe_idx"
            ]
        ):
        row
        for row in carla_rows
    }

    he_by_probe = {
        int(
            row[
                "probe_idx"
            ]
        ):
        row
        for row in he_rows
    }

    probe_ids = sorted(
        set(
            carla_by_probe
        )
        &
        set(
            he_by_probe
        )
    )

    target_fx = focal_length_px(
        args.target_width,
        args.target_fov,
    )

    target_fy = (
        target_fx
    )

    output_rows = []

    for probe_idx in probe_ids:

        if (
            probe_idx
            <
            args.skip_first
        ):
            continue

        carla_row = (
            carla_by_probe[
                probe_idx
            ]
        )

        he_row = (
            he_by_probe[
                probe_idx
            ]
        )

        carla_geometry = (
            carla_row.get(
                "carla_mask_geometry"
            )
        )

        he_info = (
            he_row.get(
                "he_mask_info"
            )
        )

        if (
            not carla_geometry
            or
            not carla_geometry.get(
                "visible",
                False,
            )
            or
            not he_info
            or
            not he_info.get(
                "found",
                False,
            )
        ):
            continue

        viewpoint = float(
            he_info[
                "viewpoint_angle_deg"
            ]
        )

        query_elevation = float(
            he_info[
                "query_elevation_deg"
            ]
        )

        depth_m = float(
            he_info[
                "depth_m"
            ]
        )

        # ====================================================
        # Existing nearest sprite-native geometry
        # ====================================================

        selected_angle = int(
            he_info[
                "selected_angle"
            ]
        )

        selected_elevation = float(
            he_info[
                "selected_elevation_deg"
            ]
        )

        nearest = (
            interpolate_distance_inverse_depth(
                geometry=
                    geometry,

                angle_deg=
                    selected_angle,

                elevation_deg=
                    selected_elevation,

                depth_m=
                    depth_m,
            )
        )

        # ====================================================
        # Continuous geometry
        # ====================================================

        continuous = (
            continuous_geometry(
                geometry=
                    geometry,

                viewpoint_angle_deg=
                    viewpoint,

                elevation_deg=
                    query_elevation,

                depth_m=
                    depth_m,
            )
        )

        if (
            nearest is None
            or
            continuous is None
        ):
            continue

        nearest_w = (
            target_fx
            *
            nearest[
                "q_width"
            ]
        )

        nearest_h = (
            target_fy
            *
            nearest[
                "q_height"
            ]
        )

        continuous_w = (
            target_fx
            *
            continuous[
                "q_width"
            ]
        )

        continuous_h = (
            target_fy
            *
            continuous[
                "q_height"
            ]
        )

        output_rows.append(
            {
                "probe_idx":
                    probe_idx,

                "depth_m":
                    depth_m,

                "query_viewpoint_deg":
                    viewpoint,

                "selected_angle_deg":
                    selected_angle,

                "query_elevation_deg":
                    query_elevation,

                "selected_elevation_deg":
                    selected_elevation,

                "carla_width_px":
                    float(
                        carla_geometry[
                            "width"
                        ]
                    ),

                "carla_height_px":
                    float(
                        carla_geometry[
                            "height"
                        ]
                    ),

                "proxy_width_px":
                    float(
                        he_info[
                            "projected_box_width_px"
                        ]
                    ),

                "proxy_height_px":
                    float(
                        he_info[
                            "projected_box_height_px"
                        ]
                    ),

                "nearest_width_px":
                    float(
                        nearest_w
                    ),

                "nearest_height_px":
                    float(
                        nearest_h
                    ),

                "continuous_width_px":
                    float(
                        continuous_w
                    ),

                "continuous_height_px":
                    float(
                        continuous_h
                    ),

                "angle_low_deg":
                    continuous[
                        "angle_low_deg"
                    ],

                "angle_high_deg":
                    continuous[
                        "angle_high_deg"
                    ],

                "angle_weight":
                    continuous[
                        "angle_weight"
                    ],

                "elevation_low_deg":
                    continuous[
                        "elevation_low_deg"
                    ],

                "elevation_high_deg":
                    continuous[
                        "elevation_high_deg"
                    ],

                "elevation_weight":
                    continuous[
                        "elevation_weight"
                    ],

                "distance_low_m":
                    continuous[
                        "distance_low_m"
                    ],

                "distance_high_m":
                    continuous[
                        "distance_high_m"
                    ],

                "distance_weight":
                    continuous[
                        "distance_weight"
                    ],
            }
        )

    if not output_rows:
        raise RuntimeError(
            "No valid matched rows."
        )

    output_path = Path(
        args.output_csv
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    carla_w = [
        row[
            "carla_width_px"
        ]
        for row in output_rows
    ]

    carla_h = [
        row[
            "carla_height_px"
        ]
        for row in output_rows
    ]

    print()
    print("=" * 88)
    print(
        "CARLA VS CONTINUOUS SPRITE-NATIVE GEOMETRY V1"
    )
    print("=" * 88)

    print(
        "matched usable frames:",
        len(
            output_rows
        ),
    )

    print(
        "target camera:",
        f"{args.target_width}x{args.target_height}",
        f"FOV={args.target_fov:.1f}",
        f"fx=fy={target_fx:.4f}",
    )

    print()

    summarize(
        "CURRENT PROXY - WIDTH",
        carla_w,
        [
            row[
                "proxy_width_px"
            ]
            for row in output_rows
        ],
    )

    print()

    summarize(
        "CURRENT PROXY - HEIGHT",
        carla_h,
        [
            row[
                "proxy_height_px"
            ]
            for row in output_rows
        ],
    )

    print()

    summarize(
        "SPRITE NEAREST - WIDTH",
        carla_w,
        [
            row[
                "nearest_width_px"
            ]
            for row in output_rows
        ],
    )

    print()

    summarize(
        "SPRITE NEAREST - HEIGHT",
        carla_h,
        [
            row[
                "nearest_height_px"
            ]
            for row in output_rows
        ],
    )

    print()

    summarize(
        "SPRITE CONTINUOUS - WIDTH",
        carla_w,
        [
            row[
                "continuous_width_px"
            ]
            for row in output_rows
        ],
    )

    print()

    summarize(
        "SPRITE CONTINUOUS - HEIGHT",
        carla_h,
        [
            row[
                "continuous_height_px"
            ]
            for row in output_rows
        ],
    )

    print()

    print(
        "CSV:",
        output_path,
    )

    print("=" * 88)


if __name__ == "__main__":
    main()