"""
analyze_sprite_native_geometry_v1.py

Diagnostic for generalized HE sprite-native geometry.

Purpose
-------
Test whether the 4320 sprite bank itself contains enough geometric
information to predict apparent actor size without asset-specific
render dimensions such as:

    4.2 x 1.8 x 1.5 m

For every sprite we measure the same visible-alpha bounding box used by
the production compositor and compute perspective-normalized apparent
width/height:

    K_w = alpha_width_px  * source_depth_m / fx
    K_h = alpha_height_px * source_depth_m / fy

For a fixed viewpoint angle and elevation, K_w and K_h should remain
approximately constant across the 5 m, 10 m and 20 m captures if the
sprite bank is geometrically self-consistent.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


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


def visible_alpha_bbox(
    sprite_rgba,
    alpha_threshold=10,
):
    """
    Same visibility rule as production compositor:

        alpha > alpha_threshold
    """

    alpha = sprite_rgba[:, :, 3]

    ys, xs = np.where(
        alpha > int(alpha_threshold)
    )

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(
            "Sprite has no visible alpha pixels."
        )

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": int(
            x2 - x1 + 1
        ),
        "height": int(
            y2 - y1 + 1
        ),
    }


def resolve_rgba_path(
    csv_path,
    recorded_path,
):
    """
    view_matrix.csv contains the original Ubuntu path.

    On Windows the sprite bank is located beside view_matrix.csv:

        <bank_root>/view_matrix.csv
        <bank_root>/rgba/<filename>
    """

    filename = Path(
        recorded_path
    ).name

    local_path = (
        csv_path.parent
        / "rgba"
        / filename
    )

    if local_path.exists():
        return local_path

    original = Path(
        recorded_path
    )

    if original.exists():
        return original

    raise FileNotFoundError(
        f"Could not resolve sprite: {recorded_path}"
    )


def coefficient_of_variation(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(
            values,
            ddof=0,
        )
    )

    if abs(mean) < 1e-12:
        return float("nan")

    return (
        std
        /
        abs(mean)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--view-matrix-csv",
        required=True,
    )

    parser.add_argument(
        "--source-width",
        type=int,
        default=1920,
    )

    parser.add_argument(
        "--source-height",
        type=int,
        default=1080,
    )

    parser.add_argument(
        "--source-fov",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "driving_models/TCP/outputs/"
            "sprite_native_geometry_v1"
        ),
    )

    args = parser.parse_args()

    csv_path = Path(
        args.view_matrix_csv
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fx = focal_length_px(
        image_width_px=
            args.source_width,

        horizontal_fov_deg=
            args.source_fov,
    )

    # CARLA pinhole camera uses square pixels.
    fy = fx

    rows_out = []

    groups = defaultdict(
        list
    )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as fp:

        reader = csv.DictReader(
            fp
        )

        for record in reader:

            angle_deg = int(
                float(
                    record[
                        "angle_deg"
                    ]
                )
            )

            distance_m = float(
                record[
                    "distance_m"
                ]
            )

            elevation_deg = float(
                record[
                    "elevation_deg"
                ]
            )

            rgba_path = resolve_rgba_path(
                csv_path=
                    csv_path,

                recorded_path=
                    record[
                        "rgba_path"
                    ],
            )

            sprite_rgba = cv2.imread(
                str(
                    rgba_path
                ),
                cv2.IMREAD_UNCHANGED,
            )

            if sprite_rgba is None:
                raise RuntimeError(
                    f"Could not load {rgba_path}"
                )

            if (
                sprite_rgba.ndim != 3
                or
                sprite_rgba.shape[2] != 4
            ):
                raise RuntimeError(
                    f"Expected RGBA image: {rgba_path}"
                )

            bbox = visible_alpha_bbox(
                sprite_rgba=
                    sprite_rgba,

                alpha_threshold=
                    args.alpha_threshold,
            )

            alpha_w = float(
                bbox[
                    "width"
                ]
            )

            alpha_h = float(
                bbox[
                    "height"
                ]
            )

            # Source camera explicitly looks at target centre.
            # Therefore the target centre lies on the optical axis
            # at the Euclidean capture distance.
            source_depth_m = (
                distance_m
            )

            k_w = (
                alpha_w
                *
                source_depth_m
                /
                fx
            )

            k_h = (
                alpha_h
                *
                source_depth_m
                /
                fy
            )

            row = {
                "angle_deg":
                    angle_deg,

                "distance_m":
                    distance_m,

                "elevation_deg":
                    elevation_deg,

                "alpha_width_px":
                    alpha_w,

                "alpha_height_px":
                    alpha_h,

                "source_depth_m":
                    source_depth_m,

                "fx_px":
                    fx,

                "fy_px":
                    fy,

                "k_width_m":
                    k_w,

                "k_height_m":
                    k_h,

                "rgba_path":
                    str(
                        rgba_path
                    ),
            }

            rows_out.append(
                row
            )

            groups[
                (
                    angle_deg,
                    elevation_deg,
                )
            ].append(
                row
            )

    # ============================================================
    # Per-viewpoint consistency
    # ============================================================

    group_rows = []

    for (
        angle_deg,
        elevation_deg,
    ), group in sorted(
        groups.items()
    ):

        group = sorted(
            group,
            key=lambda item:
                item[
                    "distance_m"
                ],
        )

        kw_values = [
            item[
                "k_width_m"
            ]
            for item in group
        ]

        kh_values = [
            item[
                "k_height_m"
            ]
            for item in group
        ]

        kw_mean = float(
            np.mean(
                kw_values
            )
        )

        kh_mean = float(
            np.mean(
                kh_values
            )
        )

        kw_std = float(
            np.std(
                kw_values
            )
        )

        kh_std = float(
            np.std(
                kh_values
            )
        )

        group_rows.append(
            {
                "angle_deg":
                    angle_deg,

                "elevation_deg":
                    elevation_deg,

                "num_distances":
                    len(
                        group
                    ),

                "k_width_mean_m":
                    kw_mean,

                "k_width_std_m":
                    kw_std,

                "k_width_cv":
                    coefficient_of_variation(
                        kw_values
                    ),

                "k_height_mean_m":
                    kh_mean,

                "k_height_std_m":
                    kh_std,

                "k_height_cv":
                    coefficient_of_variation(
                        kh_values
                    ),
            }
        )

    # ============================================================
    # Save raw rows
    # ============================================================

    raw_path = (
        output_dir
        / "sprite_native_geometry_rows.csv"
    )

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                rows_out[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows_out
        )

    group_path = (
        output_dir
        / "sprite_native_geometry_groups.csv"
    )

    with group_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                group_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            group_rows
        )

    # ============================================================
    # Global summary
    # ============================================================

    kw_cvs = np.asarray(
        [
            row[
                "k_width_cv"
            ]
            for row in group_rows
        ],
        dtype=np.float64,
    )

    kh_cvs = np.asarray(
        [
            row[
                "k_height_cv"
            ]
            for row in group_rows
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 80)
    print(
        "SPRITE-NATIVE GEOMETRY CONSISTENCY V1"
    )
    print("=" * 80)

    print(
        "sprites:",
        len(
            rows_out
        ),
    )

    print(
        "viewpoint/elevation groups:",
        len(
            group_rows
        ),
    )

    print()

    print(
        "source camera:"
    )

    print(
        f"  {args.source_width}x"
        f"{args.source_height}, "
        f"FOV={args.source_fov:.1f}"
    )

    print(
        f"  fx=fy={fx:.4f} px"
    )

    print()

    print(
        "Width K consistency across distance"
    )

    print(
        "  mean CV :",
        f"{np.mean(kw_cvs):.6f}",
    )

    print(
        "  median  :",
        f"{np.median(kw_cvs):.6f}",
    )

    print(
        "  P95     :",
        f"{np.percentile(kw_cvs, 95):.6f}",
    )

    print(
        "  max     :",
        f"{np.max(kw_cvs):.6f}",
    )

    print()

    print(
        "Height K consistency across distance"
    )

    print(
        "  mean CV :",
        f"{np.mean(kh_cvs):.6f}",
    )

    print(
        "  median  :",
        f"{np.median(kh_cvs):.6f}",
    )

    print(
        "  P95     :",
        f"{np.percentile(kh_cvs, 95):.6f}",
    )

    print(
        "  max     :",
        f"{np.max(kh_cvs):.6f}",
    )

    print()

    # Show a few intuitive examples.
    for angle in [
        0,
        45,
        90,
        135,
        180,
    ]:

        candidates = [
            row
            for row in group_rows
            if (
                row[
                    "angle_deg"
                ]
                ==
                angle
                and
                abs(
                    row[
                        "elevation_deg"
                    ]
                )
                <
                1e-6
            )
        ]

        if not candidates:
            continue

        row = candidates[
            0
        ]

        print(
            f"angle={angle:3d}, "
            f"elev=0: "
            f"Kw={row['k_width_mean_m']:.4f} "
            f"(CV={row['k_width_cv']:.4f}), "
            f"Kh={row['k_height_mean_m']:.4f} "
            f"(CV={row['k_height_cv']:.4f})"
        )

    print()

    print(
        "raw:",
        raw_path,
    )

    print(
        "groups:",
        group_path,
    )

    print("=" * 80)


if __name__ == "__main__":
    main()