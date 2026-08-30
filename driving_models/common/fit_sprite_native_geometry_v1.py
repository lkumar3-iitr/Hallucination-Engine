"""
fit_sprite_native_geometry_v1.py

Fit a generic sprite-native apparent-geometry model.

For every fixed:

    viewpoint angle
    elevation

we have sprite silhouette measurements at:

    5 m
    10 m
    20 m

Model
-----

    width_px(Z)  = fx * A_w / (Z - delta_w)
    height_px(Z) = fy * A_h / (Z - delta_h)

where:

    A_w / A_h:
        apparent geometric extent inferred automatically from sprites

    delta_w / delta_h:
        effective depth offset of the visible silhouette relative
        to the actor centre

No hand-selected vehicle render dimensions are used.

The fitting equation is linear:

    fx / width_px = (1 / A_w) * Z - delta_w / A_w

and similarly for height.

This script reports:

    - fitted parameters
    - in-sample relative error
    - leave-one-distance-out prediction error
    - parameter distributions
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def fit_effective_depth_model(
    z_values,
    pixel_values,
    focal_px,
):
    """
    Fit:

        p(Z) = f * A / (Z - delta)

    Rearranged:

        f / p = m * Z + b

    with:

        A     = 1 / m
        delta = -b / m
    """

    z = np.asarray(
        z_values,
        dtype=np.float64,
    )

    p = np.asarray(
        pixel_values,
        dtype=np.float64,
    )

    if len(z) < 2:
        raise ValueError(
            "Need at least two points."
        )

    if np.any(p <= 0):
        raise ValueError(
            "Pixel values must be positive."
        )

    y = (
        float(focal_px)
        /
        p
    )

    X = np.column_stack(
        [
            z,
            np.ones_like(
                z
            ),
        ]
    )

    solution, _, _, _ = (
        np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )
    )

    slope = float(
        solution[0]
    )

    intercept = float(
        solution[1]
    )

    if slope <= 0.0:
        return {
            "valid": False,
            "reason": (
                "non_positive_slope"
            ),
        }

    apparent_extent_m = (
        1.0
        /
        slope
    )

    depth_offset_m = (
        -intercept
        /
        slope
    )

    return {
        "valid":
            True,

        "slope":
            slope,

        "intercept":
            intercept,

        "apparent_extent_m":
            float(
                apparent_extent_m
            ),

        "depth_offset_m":
            float(
                depth_offset_m
            ),
    }


def predict_pixels(
    depth_m,
    focal_px,
    apparent_extent_m,
    depth_offset_m,
):
    denominator = (
        float(depth_m)
        -
        float(depth_offset_m)
    )

    if denominator <= 1e-6:
        return float(
            "nan"
        )

    return (
        float(focal_px)
        *
        float(apparent_extent_m)
        /
        denominator
    )


def relative_error(
    predicted,
    actual,
):
    if (
        not math.isfinite(
            predicted
        )
        or
        abs(
            actual
        )
        < 1e-12
    ):
        return float(
            "nan"
        )

    return (
        abs(
            float(predicted)
            -
            float(actual)
        )
        /
        abs(
            float(actual)
        )
    )


def summarize(
    name,
    values,
):
    arr = np.asarray(
        [
            value
            for value in values
            if math.isfinite(
                value
            )
        ],
        dtype=np.float64,
    )

    print(
        name
    )

    if len(arr) == 0:
        print(
            "  no valid values"
        )
        return

    print(
        "  count  :",
        len(
            arr
        ),
    )

    print(
        "  mean   :",
        f"{np.mean(arr):.6f}",
    )

    print(
        "  median :",
        f"{np.median(arr):.6f}",
    )

    print(
        "  P95    :",
        f"{np.percentile(arr, 95):.6f}",
    )

    print(
        "  max    :",
        f"{np.max(arr):.6f}",
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-csv",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "driving_models/TCP/outputs/"
            "sprite_native_geometry_fit_v1"
        ),
    )

    parser.add_argument(
        "--geometry-version",
        default="sprite_geometry_v1",
        help=(
            "Version string embedded in the runtime sprite_geometry_v1.csv."
        ),
    )

    parser.add_argument(
        "--geometry-alpha-threshold",
        type=int,
        default=64,
        help=(
            "Alpha threshold used when the input silhouette rows were measured. "
            "This value is embedded in sprite_geometry_v1.csv and is reused by "
            "the runtime compositor."
        ),
    )

    args = parser.parse_args()

    if not (
        1
        <= int(args.geometry_alpha_threshold)
        <= 255
    ):
        raise ValueError(
            "--geometry-alpha-threshold must be in [1, 255]."
        )

    input_csv = Path(
        args.input_csv
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Load measurements
    # ============================================================

    groups = defaultdict(
        list
    )

    # Runtime geometry table.  Unlike sprite_native_geometry_fits.csv,
    # this preserves the measured per-distance silhouettes because
    # SpriteNativeGeometryV1 performs interpolation/extrapolation itself.
    runtime_rows = []

    with input_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as fp:

        reader = csv.DictReader(
            fp
        )

        for row in reader:

            parsed = {
                "angle_deg":
                    int(
                        float(
                            row[
                                "angle_deg"
                            ]
                        )
                    ),

                "elevation_deg":
                    float(
                        row[
                            "elevation_deg"
                        ]
                    ),

                "distance_m":
                    float(
                        row[
                            "distance_m"
                        ]
                    ),

                "source_depth_m":
                    float(
                        row.get(
                            "source_depth_m",
                            row[
                                "distance_m"
                            ],
                        )
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

            runtime_rows.append(
                {
                    "geometry_version":
                        str(
                            args.geometry_version
                        ),

                    "geometry_alpha_threshold":
                        int(
                            args.geometry_alpha_threshold
                        ),

                    "angle_deg":
                        int(
                            parsed[
                                "angle_deg"
                            ]
                        ),

                    "elevation_deg":
                        float(
                            parsed[
                                "elevation_deg"
                            ]
                        ),

                    "source_depth_m":
                        float(
                            parsed[
                                "source_depth_m"
                            ]
                        ),

                    "alpha_width_px":
                        float(
                            parsed[
                                "alpha_width_px"
                            ]
                        ),

                    "alpha_height_px":
                        float(
                            parsed[
                                "alpha_height_px"
                            ]
                        ),

                    "fx_px":
                        float(
                            parsed[
                                "fx_px"
                            ]
                        ),

                    "fy_px":
                        float(
                            parsed[
                                "fy_px"
                            ]
                        ),
                }
            )

            groups[
                (
                    parsed[
                        "angle_deg"
                    ],
                    parsed[
                        "elevation_deg"
                    ],
                )
            ].append(
                parsed
            )

    # ============================================================
    # Fit every angle/elevation group
    # ============================================================

    fit_rows = []

    prediction_rows = []

    for (
        angle_deg,
        elevation_deg,
    ), samples in sorted(
        groups.items()
    ):

        samples = sorted(
            samples,
            key=lambda item:
                item[
                    "distance_m"
                ],
        )

        z_values = [
            row[
                "distance_m"
            ]
            for row in samples
        ]

        width_values = [
            row[
                "alpha_width_px"
            ]
            for row in samples
        ]

        height_values = [
            row[
                "alpha_height_px"
            ]
            for row in samples
        ]

        fx = float(
            samples[0][
                "fx_px"
            ]
        )

        fy = float(
            samples[0][
                "fy_px"
            ]
        )

        width_fit = (
            fit_effective_depth_model(
                z_values=
                    z_values,

                pixel_values=
                    width_values,

                focal_px=
                    fx,
            )
        )

        height_fit = (
            fit_effective_depth_model(
                z_values=
                    z_values,

                pixel_values=
                    height_values,

                focal_px=
                    fy,
            )
        )

        if (
            not width_fit[
                "valid"
            ]
            or
            not height_fit[
                "valid"
            ]
        ):
            continue

        width_errors = []
        height_errors = []

        # ========================================================
        # In-sample predictions
        # ========================================================

        for sample in samples:

            depth_m = float(
                sample[
                    "distance_m"
                ]
            )

            actual_w = float(
                sample[
                    "alpha_width_px"
                ]
            )

            actual_h = float(
                sample[
                    "alpha_height_px"
                ]
            )

            predicted_w = (
                predict_pixels(
                    depth_m=
                        depth_m,

                    focal_px=
                        fx,

                    apparent_extent_m=
                        width_fit[
                            "apparent_extent_m"
                        ],

                    depth_offset_m=
                        width_fit[
                            "depth_offset_m"
                        ],
                )
            )

            predicted_h = (
                predict_pixels(
                    depth_m=
                        depth_m,

                    focal_px=
                        fy,

                    apparent_extent_m=
                        height_fit[
                            "apparent_extent_m"
                        ],

                    depth_offset_m=
                        height_fit[
                            "depth_offset_m"
                        ],
                )
            )

            width_err = (
                relative_error(
                    predicted=
                        predicted_w,

                    actual=
                        actual_w,
                )
            )

            height_err = (
                relative_error(
                    predicted=
                        predicted_h,

                    actual=
                        actual_h,
                )
            )

            width_errors.append(
                width_err
            )

            height_errors.append(
                height_err
            )

            prediction_rows.append(
                {
                    "mode":
                        "in_sample",

                    "angle_deg":
                        angle_deg,

                    "elevation_deg":
                        elevation_deg,

                    "distance_m":
                        depth_m,

                    "actual_width_px":
                        actual_w,

                    "predicted_width_px":
                        predicted_w,

                    "width_relative_error":
                        width_err,

                    "actual_height_px":
                        actual_h,

                    "predicted_height_px":
                        predicted_h,

                    "height_relative_error":
                        height_err,
                }
            )

        # ========================================================
        # Leave-one-distance-out validation
        #
        # With 3 distances:
        #
        #   fit 10,20 -> predict 5
        #   fit 5,20  -> predict 10
        #   fit 5,10  -> predict 20
        # ========================================================

        loo_width_errors = []
        loo_height_errors = []

        for holdout_idx in range(
            len(
                samples
            )
        ):

            train_samples = [
                sample
                for idx, sample
                in enumerate(
                    samples
                )
                if idx != holdout_idx
            ]

            test_sample = (
                samples[
                    holdout_idx
                ]
            )

            train_z = [
                sample[
                    "distance_m"
                ]
                for sample
                in train_samples
            ]

            train_w = [
                sample[
                    "alpha_width_px"
                ]
                for sample
                in train_samples
            ]

            train_h = [
                sample[
                    "alpha_height_px"
                ]
                for sample
                in train_samples
            ]

            loo_width_fit = (
                fit_effective_depth_model(
                    z_values=
                        train_z,

                    pixel_values=
                        train_w,

                    focal_px=
                        fx,
                )
            )

            loo_height_fit = (
                fit_effective_depth_model(
                    z_values=
                        train_z,

                    pixel_values=
                        train_h,

                    focal_px=
                        fy,
                )
            )

            test_depth = float(
                test_sample[
                    "distance_m"
                ]
            )

            actual_w = float(
                test_sample[
                    "alpha_width_px"
                ]
            )

            actual_h = float(
                test_sample[
                    "alpha_height_px"
                ]
            )

            if loo_width_fit[
                "valid"
            ]:

                pred_w = (
                    predict_pixels(
                        depth_m=
                            test_depth,

                        focal_px=
                            fx,

                        apparent_extent_m=
                            loo_width_fit[
                                "apparent_extent_m"
                            ],

                        depth_offset_m=
                            loo_width_fit[
                                "depth_offset_m"
                            ],
                    )
                )

                err_w = (
                    relative_error(
                        predicted=
                            pred_w,

                        actual=
                            actual_w,
                    )
                )

            else:

                pred_w = float(
                    "nan"
                )

                err_w = float(
                    "nan"
                )

            if loo_height_fit[
                "valid"
            ]:

                pred_h = (
                    predict_pixels(
                        depth_m=
                            test_depth,

                        focal_px=
                            fy,

                        apparent_extent_m=
                            loo_height_fit[
                                "apparent_extent_m"
                            ],

                        depth_offset_m=
                            loo_height_fit[
                                "depth_offset_m"
                            ],
                    )
                )

                err_h = (
                    relative_error(
                        predicted=
                            pred_h,

                        actual=
                            actual_h,
                    )
                )

            else:

                pred_h = float(
                    "nan"
                )

                err_h = float(
                    "nan"
                )

            loo_width_errors.append(
                err_w
            )

            loo_height_errors.append(
                err_h
            )

            prediction_rows.append(
                {
                    "mode":
                        "leave_one_out",

                    "angle_deg":
                        angle_deg,

                    "elevation_deg":
                        elevation_deg,

                    "distance_m":
                        test_depth,

                    "actual_width_px":
                        actual_w,

                    "predicted_width_px":
                        pred_w,

                    "width_relative_error":
                        err_w,

                    "actual_height_px":
                        actual_h,

                    "predicted_height_px":
                        pred_h,

                    "height_relative_error":
                        err_h,
                }
            )

        fit_rows.append(
            {
                "angle_deg":
                    angle_deg,

                "elevation_deg":
                    elevation_deg,

                "num_distances":
                    len(
                        samples
                    ),

                "width_extent_m":
                    width_fit[
                        "apparent_extent_m"
                    ],

                "width_depth_offset_m":
                    width_fit[
                        "depth_offset_m"
                    ],

                "width_fit_mean_relative_error":
                    float(
                        np.nanmean(
                            width_errors
                        )
                    ),

                "width_loo_mean_relative_error":
                    float(
                        np.nanmean(
                            loo_width_errors
                        )
                    ),

                "height_extent_m":
                    height_fit[
                        "apparent_extent_m"
                    ],

                "height_depth_offset_m":
                    height_fit[
                        "depth_offset_m"
                    ],

                "height_fit_mean_relative_error":
                    float(
                        np.nanmean(
                            height_errors
                        )
                    ),

                "height_loo_mean_relative_error":
                    float(
                        np.nanmean(
                            loo_height_errors
                        )
                    ),
            }
        )

    # ============================================================
    # Save self-contained runtime geometry table
    # ============================================================

    runtime_rows.sort(
        key=lambda item: (
            int(
                item[
                    "angle_deg"
                ]
            ),
            float(
                item[
                    "elevation_deg"
                ]
            ),
            float(
                item[
                    "source_depth_m"
                ]
            ),
        )
    )

    runtime_path = (
        output_dir
        / "sprite_geometry_v1.csv"
    )

    with runtime_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "geometry_version",
                "geometry_alpha_threshold",
                "angle_deg",
                "elevation_deg",
                "source_depth_m",
                "alpha_width_px",
                "alpha_height_px",
                "fx_px",
                "fy_px",
            ],
        )

        writer.writeheader()

        writer.writerows(
            runtime_rows
        )

    # ============================================================
    # Save diagnostic fitted geometry table
    # ============================================================

    fit_path = (
        output_dir
        / "sprite_native_geometry_fits.csv"
    )

    with fit_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                fit_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            fit_rows
        )

    predictions_path = (
        output_dir
        / "sprite_native_geometry_predictions.csv"
    )

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                prediction_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            prediction_rows
        )

    # ============================================================
    # Global statistics
    # ============================================================

    width_fit_errors = [
        row[
            "width_relative_error"
        ]
        for row in prediction_rows
        if row[
            "mode"
        ] == "in_sample"
    ]

    height_fit_errors = [
        row[
            "height_relative_error"
        ]
        for row in prediction_rows
        if row[
            "mode"
        ] == "in_sample"
    ]

    width_loo_errors = [
        row[
            "width_relative_error"
        ]
        for row in prediction_rows
        if row[
            "mode"
        ] == "leave_one_out"
    ]

    height_loo_errors = [
        row[
            "height_relative_error"
        ]
        for row in prediction_rows
        if row[
            "mode"
        ] == "leave_one_out"
    ]

    width_extents = [
        row[
            "width_extent_m"
        ]
        for row in fit_rows
    ]

    width_offsets = [
        row[
            "width_depth_offset_m"
        ]
        for row in fit_rows
    ]

    height_extents = [
        row[
            "height_extent_m"
        ]
        for row in fit_rows
    ]

    height_offsets = [
        row[
            "height_depth_offset_m"
        ]
        for row in fit_rows
    ]

    print()
    print("=" * 84)
    print(
        "SPRITE-NATIVE EFFECTIVE-DEPTH GEOMETRY FIT V1"
    )
    print("=" * 84)

    print(
        "groups:",
        len(
            fit_rows
        ),
    )

    print()

    summarize(
        "Width in-sample relative error",
        width_fit_errors,
    )

    print()

    summarize(
        "Height in-sample relative error",
        height_fit_errors,
    )

    print()

    summarize(
        "Width leave-one-distance-out relative error",
        width_loo_errors,
    )

    print()

    summarize(
        "Height leave-one-distance-out relative error",
        height_loo_errors,
    )

    print()

    summarize(
        "Fitted width apparent extent [m]",
        width_extents,
    )

    print()

    summarize(
        "Fitted width depth offset [m]",
        width_offsets,
    )

    print()

    summarize(
        "Fitted height apparent extent [m]",
        height_extents,
    )

    print()

    summarize(
        "Fitted height depth offset [m]",
        height_offsets,
    )

    # ============================================================
    # Representative views
    # ============================================================

    print()
    print(
        "Representative elevation=0 fits"
    )

    for angle in [
        0,
        45,
        90,
        135,
        180,
    ]:

        matches = [
            row
            for row in fit_rows
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
                1e-9
            )
        ]

        if not matches:
            continue

        row = matches[
            0
        ]

        print(
            f"  angle={angle:3d}: "
            f"Aw={row['width_extent_m']:.4f}, "
            f"dw={row['width_depth_offset_m']:.4f}, "
            f"Wloo={row['width_loo_mean_relative_error']:.4f}; "
            f"Ah={row['height_extent_m']:.4f}, "
            f"dh={row['height_depth_offset_m']:.4f}, "
            f"Hloo={row['height_loo_mean_relative_error']:.4f}"
        )

    print()

    print(
        "runtime    :",
        runtime_path,
    )

    print(
        "fits       :",
        fit_path,
    )

    print(
        "predictions:",
        predictions_path,
    )

    print("=" * 84)


if __name__ == "__main__":
    main()