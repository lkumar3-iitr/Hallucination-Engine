"""
compare_sprite_native_geometry_to_carla_v1.py

Offline comparison of:

    1. CARLA visible instance-mask geometry       [ground truth]
    2. Current render-dimensions proxy projection
    3. Sprite-native empirical interpolation
    4. Sprite-native effective-depth fit

No CARLA execution and no TCP execution are required.
"""

import argparse
import csv
import json
import math
from pathlib import Path

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


def geometry_key(
    angle_deg,
    elevation_deg,
):
    return (
        int(
            round(
                float(angle_deg)
            )
        ),
        round(
            float(elevation_deg),
            6,
        ),
    )


def load_raw_geometry(path):
    groups = {}

    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as fp:

        reader = csv.DictReader(
            fp
        )

        for row in reader:

            key = geometry_key(
                row[
                    "angle_deg"
                ],
                row[
                    "elevation_deg"
                ],
            )

            item = {
                "distance_m":
                    float(
                        row[
                            "distance_m"
                        ]
                    ),

                "alpha_width_px":
                    float(
                        row[
                            "alpha_width_px"
                        ]
                    ),

                "alpha_height_px":
                    float(
                        row[
                            "alpha_height_px"
                        ]
                    ),

                "fx_px":
                    float(
                        row[
                            "fx_px"
                        ]
                    ),

                "fy_px":
                    float(
                        row[
                            "fy_px"
                        ]
                    ),
            }

            groups.setdefault(
                key,
                [],
            ).append(
                item
            )

    for key in groups:
        groups[key] = sorted(
            groups[key],
            key=lambda item:
                item[
                    "distance_m"
                ],
        )

    return groups


def load_fits(path):
    fits = {}

    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as fp:

        reader = csv.DictReader(
            fp
        )

        for row in reader:

            key = geometry_key(
                row[
                    "angle_deg"
                ],
                row[
                    "elevation_deg"
                ],
            )

            fits[key] = {
                "width_extent_m":
                    float(
                        row[
                            "width_extent_m"
                        ]
                    ),

                "width_depth_offset_m":
                    float(
                        row[
                            "width_depth_offset_m"
                        ]
                    ),

                "height_extent_m":
                    float(
                        row[
                            "height_extent_m"
                        ]
                    ),

                "height_depth_offset_m":
                    float(
                        row[
                            "height_depth_offset_m"
                        ]
                    ),
            }

    return fits


def empirical_interpolate(
    samples,
    target_depth_m,
    target_fx,
    target_fy,
):
    """
    Interpolate normalized visible geometry q = pixels / focal_length
    linearly in inverse depth.

    Only interpolation inside the measured distance range is done.
    """

    z = float(
        target_depth_m
    )

    if len(samples) < 2:
        return None

    if (
        z
        <
        samples[0][
            "distance_m"
        ]
        or
        z
        >
        samples[-1][
            "distance_m"
        ]
    ):
        return None

    low = None
    high = None

    for idx in range(
        len(samples) - 1
    ):
        a = samples[
            idx
        ]

        b = samples[
            idx + 1
        ]

        if (
            a[
                "distance_m"
            ]
            <=
            z
            <=
            b[
                "distance_m"
            ]
        ):
            low = a
            high = b
            break

    if low is None or high is None:
        return None

    z0 = float(
        low[
            "distance_m"
        ]
    )

    z1 = float(
        high[
            "distance_m"
        ]
    )

    u = 1.0 / z
    u0 = 1.0 / z0
    u1 = 1.0 / z1

    if abs(
        u1 - u0
    ) < 1e-12:
        t = 0.0
    else:
        t = (
            u - u0
        ) / (
            u1 - u0
        )

    source_fx_0 = float(
        low[
            "fx_px"
        ]
    )

    source_fx_1 = float(
        high[
            "fx_px"
        ]
    )

    source_fy_0 = float(
        low[
            "fy_px"
        ]
    )

    source_fy_1 = float(
        high[
            "fy_px"
        ]
    )

    qw0 = (
        float(
            low[
                "alpha_width_px"
            ]
        )
        /
        source_fx_0
    )

    qw1 = (
        float(
            high[
                "alpha_width_px"
            ]
        )
        /
        source_fx_1
    )

    qh0 = (
        float(
            low[
                "alpha_height_px"
            ]
        )
        /
        source_fy_0
    )

    qh1 = (
        float(
            high[
                "alpha_height_px"
            ]
        )
        /
        source_fy_1
    )

    qw = (
        qw0
        +
        t
        *
        (
            qw1
            -
            qw0
        )
    )

    qh = (
        qh0
        +
        t
        *
        (
            qh1
            -
            qh0
        )
    )

    return {
        "width_px":
            float(
                target_fx
            )
            *
            qw,

        "height_px":
            float(
                target_fy
            )
            *
            qh,

        "distance_low_m":
            z0,

        "distance_high_m":
            z1,

        "weight":
            float(
                t
            ),
    }


def fitted_predict(
    fit,
    depth_m,
    target_fx,
    target_fy,
):
    z = float(
        depth_m
    )

    dw = float(
        fit[
            "width_depth_offset_m"
        ]
    )

    dh = float(
        fit[
            "height_depth_offset_m"
        ]
    )

    denom_w = (
        z - dw
    )

    denom_h = (
        z - dh
    )

    if (
        denom_w <= 1e-6
        or
        denom_h <= 1e-6
    ):
        return None

    return {
        "width_px":
            (
                float(
                    target_fx
                )
                *
                float(
                    fit[
                        "width_extent_m"
                    ]
                )
                /
                denom_w
            ),

        "height_px":
            (
                float(
                    target_fy
                )
                *
                float(
                    fit[
                        "height_extent_m"
                    ]
                )
                /
                denom_h
            ),
    }


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
            actual > 0.0
        )
    )

    actual = actual[
        valid
    ]

    predicted = predicted[
        valid
    ]

    if len(actual) == 0:
        print(
            f"{name}: no valid samples"
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
        len(actual),
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
        "--raw-geometry-csv",
        required=True,
    )

    parser.add_argument(
        "--fit-csv",
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
            "sprite_native_carla_compare_v1/"
            "comparison_rows.csv"
        ),
    )

    args = parser.parse_args()

    carla_rows = load_jsonl(
        args.carla_jsonl
    )

    he_rows = load_jsonl(
        args.he_jsonl
    )

    raw_geometry = load_raw_geometry(
        args.raw_geometry_csv
    )

    fits = load_fits(
        args.fit_csv
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

    common_probe_ids = sorted(
        set(
            carla_by_probe
        )
        &
        set(
            he_by_probe
        )
    )

    target_fx = focal_length_px(
        image_width_px=
            args.target_width,

        horizontal_fov_deg=
            args.target_fov,
    )

    # CARLA pinhole camera uses square pixels.
    target_fy = target_fx

    output_rows = []

    for probe_idx in common_probe_ids:

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

        angle_deg = int(
            he_info[
                "selected_angle"
            ]
        )

        elevation_deg = float(
            he_info[
                "selected_elevation_deg"
            ]
        )

        target_depth_m = float(
            he_info[
                "depth_m"
            ]
        )

        key = geometry_key(
            angle_deg,
            elevation_deg,
        )

        samples = (
            raw_geometry.get(
                key
            )
        )

        fit = fits.get(
            key
        )

        if (
            samples is None
            or
            fit is None
        ):
            continue

        empirical = (
            empirical_interpolate(
                samples=
                    samples,

                target_depth_m=
                    target_depth_m,

                target_fx=
                    target_fx,

                target_fy=
                    target_fy,
            )
        )

        fitted = (
            fitted_predict(
                fit=
                    fit,

                depth_m=
                    target_depth_m,

                target_fx=
                    target_fx,

                target_fy=
                    target_fy,
            )
        )

        row = {
            "probe_idx":
                probe_idx,

            "depth_m":
                target_depth_m,

            "selected_angle":
                angle_deg,

            "selected_elevation_deg":
                elevation_deg,

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

            "empirical_width_px":
                (
                    float(
                        empirical[
                            "width_px"
                        ]
                    )
                    if empirical
                    else
                    float(
                        "nan"
                    )
                ),

            "empirical_height_px":
                (
                    float(
                        empirical[
                            "height_px"
                        ]
                    )
                    if empirical
                    else
                    float(
                        "nan"
                    )
                ),

            "fitted_width_px":
                (
                    float(
                        fitted[
                            "width_px"
                        ]
                    )
                    if fitted
                    else
                    float(
                        "nan"
                    )
                ),

            "fitted_height_px":
                (
                    float(
                        fitted[
                            "height_px"
                        ]
                    )
                    if fitted
                    else
                    float(
                        "nan"
                    )
                ),
        }

        output_rows.append(
            row
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
    print("=" * 84)
    print(
        "CARLA VS SPRITE-NATIVE GEOMETRY V1"
    )
    print("=" * 84)

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
        "SPRITE EMPIRICAL - WIDTH",
        carla_w,
        [
            row[
                "empirical_width_px"
            ]
            for row in output_rows
        ],
    )

    print()

    summarize(
        "SPRITE EMPIRICAL - HEIGHT",
        carla_h,
        [
            row[
                "empirical_height_px"
            ]
            for row in output_rows
        ],
    )

    print()

    summarize(
        "SPRITE FITTED - WIDTH",
        carla_w,
        [
            row[
                "fitted_width_px"
            ]
            for row in output_rows
        ],
    )

    print()

    summarize(
        "SPRITE FITTED - HEIGHT",
        carla_h,
        [
            row[
                "fitted_height_px"
            ]
            for row in output_rows
        ],
    )

    print()

    print(
        "CSV:",
        output_path,
    )

    print("=" * 84)


if __name__ == "__main__":
    main()