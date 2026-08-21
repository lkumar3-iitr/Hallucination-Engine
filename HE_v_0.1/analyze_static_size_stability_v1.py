#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_mask_geometry(path, threshold=10):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return None

    ys, xs = np.where(
        image > int(threshold)
    )

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())

    y1 = int(ys.min())
    y2 = int(ys.max())

    return {
        "x1":
            x1,

        "x2":
            x2,

        "y1":
            y1,

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
            0.5
            *
            (
                x1 + x2
            ),

        "bottom_y":
            float(
                y2
            ),
    }


def infer_direction(values):

    if len(values) < 2:
        return 0

    delta = (
        float(values[-1])
        -
        float(values[0])
    )

    if abs(delta) < 1e-9:
        return 0

    return (
        1
        if delta > 0
        else -1
    )


def opposite_direction_violation(
    delta,
    expected_direction,
    tolerance,
):

    if expected_direction > 0:
        return (
            delta
            <
            -float(tolerance)
        )

    if expected_direction < 0:
        return (
            delta
            >
            float(tolerance)
        )

    return (
        abs(delta)
        >
        float(tolerance)
    )


def local_residual(values, i):

    return (
        float(values[i])
        -
        0.5
        *
        (
            float(values[i - 1])
            +
            float(values[i + 1])
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        required=True,
    )

    parser.add_argument(
        "--mask-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--violation-tolerance-px",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    with open(
        args.metadata,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    mask_dir = Path(
        args.mask_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = []

    for list_idx, frame in enumerate(
        metadata["frames"]
    ):

        frame_idx = int(
            frame.get(
                "frame_idx",
                list_idx,
            )
        )

        adversaries = frame.get(
            "adversaries",
            [],
        )

        if not adversaries:
            continue

        actor = adversaries[0]

        if not actor.get(
            "rendered",
            False,
        ):
            continue

        box = actor.get(
            "box",
            {},
        )

        if not box.get(
            "visible",
            False,
        ):
            continue

        mask_path = (
            mask_dir
            /
            f"frame_{frame_idx:06d}_mask.png"
        )

        mask_geom = load_mask_geometry(
            mask_path,
            threshold=
                args.mask_threshold,
        )

        if mask_geom is None:
            continue

        sprite = actor.get(
            "sprite",
            {},
        )

        resize_info = actor.get(
            "view_matrix_resize",
            {},
        )

        frames.append({
            "frame_idx":
                frame_idx,

            "target_width":
                float(
                    box["box_width"]
                ),

            "target_height":
                float(
                    box["box_height"]
                ),

            "target_cx":
                float(
                    box["cx"]
                ),

            "target_bottom_y":
                float(
                    box["bottom_y"]
                ),

            "rendered_width":
                float(
                    mask_geom["width"]
                ),

            "rendered_height":
                float(
                    mask_geom["height"]
                ),

            "rendered_cx":
                float(
                    mask_geom["cx"]
                ),

            "rendered_bottom_y":
                float(
                    mask_geom[
                        "bottom_y"
                    ]
                ),

            "selected_angle":
                float(
                    sprite.get(
                        "selected_angle",
                        0.0,
                    )
                ),

            "selected_distance":
                float(
                    sprite.get(
                        "selected_distance_m",
                        0.0,
                    )
                ),

            "selected_elevation":
                float(
                    sprite.get(
                        "selected_elevation_deg",
                        0.0,
                    )
                ),

            "render_mode":
                str(
                    resize_info.get(
                        "mode",
                        "",
                    )
                ),

            "scale_error_x":
                float(
                    resize_info.get(
                        "scale_error_x",
                        0.0,
                    )
                ),

            "scale_error_y":
                float(
                    resize_info.get(
                        "scale_error_y",
                        0.0,
                    )
                ),

            "placement_error_x":
                float(
                    resize_info.get(
                        "placement_error_x",
                        0.0,
                    )
                ),

            "placement_error_y":
                float(
                    resize_info.get(
                        "placement_error_y",
                        0.0,
                    )
                ),
        })

    if len(frames) < 3:
        raise RuntimeError(
            "Need at least 3 rendered frames."
        )

    target_widths = [
        f["target_width"]
        for f in frames
    ]

    target_heights = [
        f["target_height"]
        for f in frames
    ]

    rendered_widths = [
        f["rendered_width"]
        for f in frames
    ]

    rendered_heights = [
        f["rendered_height"]
        for f in frames
    ]

    width_direction = infer_direction(
        target_widths
    )

    height_direction = infer_direction(
        target_heights
    )

    rows = []

    for i, frame in enumerate(
        frames
    ):

        row = dict(
            frame
        )

        if i == 0:

            row.update({
                "d_target_width":
                    0.0,

                "d_target_height":
                    0.0,

                "d_rendered_width":
                    0.0,

                "d_rendered_height":
                    0.0,

                "target_width_violation":
                    0,

                "target_height_violation":
                    0,

                "rendered_width_violation":
                    0,

                "rendered_height_violation":
                    0,

                "width_local_residual":
                    0.0,

                "height_local_residual":
                    0.0,
            })

        else:

            d_tw = (
                frame[
                    "target_width"
                ]
                -
                frames[i - 1][
                    "target_width"
                ]
            )

            d_th = (
                frame[
                    "target_height"
                ]
                -
                frames[i - 1][
                    "target_height"
                ]
            )

            d_rw = (
                frame[
                    "rendered_width"
                ]
                -
                frames[i - 1][
                    "rendered_width"
                ]
            )

            d_rh = (
                frame[
                    "rendered_height"
                ]
                -
                frames[i - 1][
                    "rendered_height"
                ]
            )

            row.update({
                "d_target_width":
                    float(d_tw),

                "d_target_height":
                    float(d_th),

                "d_rendered_width":
                    float(d_rw),

                "d_rendered_height":
                    float(d_rh),

                "target_width_violation":
                    int(
                        opposite_direction_violation(
                            d_tw,
                            width_direction,
                            args.violation_tolerance_px,
                        )
                    ),

                "target_height_violation":
                    int(
                        opposite_direction_violation(
                            d_th,
                            height_direction,
                            args.violation_tolerance_px,
                        )
                    ),

                "rendered_width_violation":
                    int(
                        opposite_direction_violation(
                            d_rw,
                            width_direction,
                            args.violation_tolerance_px,
                        )
                    ),

                "rendered_height_violation":
                    int(
                        opposite_direction_violation(
                            d_rh,
                            height_direction,
                            args.violation_tolerance_px,
                        )
                    ),
            })

            if (
                0 < i
                <
                len(frames) - 1
            ):

                row[
                    "width_local_residual"
                ] = float(
                    local_residual(
                        rendered_widths,
                        i,
                    )
                )

                row[
                    "height_local_residual"
                ] = float(
                    local_residual(
                        rendered_heights,
                        i,
                    )
                )

            else:

                row[
                    "width_local_residual"
                ] = 0.0

                row[
                    "height_local_residual"
                ] = 0.0

        row[
            "rendered_target_width_ratio"
        ] = (
            frame["rendered_width"]
            /
            max(
                1e-6,
                frame["target_width"],
            )
        )

        row[
            "rendered_target_height_ratio"
        ] = (
            frame["rendered_height"]
            /
            max(
                1e-6,
                frame["target_height"],
            )
        )

        rows.append(
            row
        )

    csv_path = (
        output_dir
        /
        "static_size_stability.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                list(
                    rows[0].keys()
                ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    target_w_violations = sum(
        r[
            "target_width_violation"
        ]
        for r in rows
    )

    target_h_violations = sum(
        r[
            "target_height_violation"
        ]
        for r in rows
    )

    rendered_w_violations = sum(
        r[
            "rendered_width_violation"
        ]
        for r in rows
    )

    rendered_h_violations = sum(
        r[
            "rendered_height_violation"
        ]
        for r in rows
    )

    width_residuals = [
        abs(
            r[
                "width_local_residual"
            ]
        )
        for r in rows[1:-1]
    ]

    height_residuals = [
        abs(
            r[
                "height_local_residual"
            ]
        )
        for r in rows[1:-1]
    ]

    width_ratios = [
        r[
            "rendered_target_width_ratio"
        ]
        for r in rows
    ]

    height_ratios = [
        r[
            "rendered_target_height_ratio"
        ]
        for r in rows
    ]

    print()
    print("=" * 88)
    print(
        "HE STATIC SIZE STABILITY V1"
    )
    print("=" * 88)

    print(
        "Frames:",
        len(rows),
    )

    print()
    print(
        "Expected direction"
    )

    print(
        "  width :",
        (
            "increasing"
            if width_direction > 0
            else
            "decreasing"
            if width_direction < 0
            else
            "constant"
        ),
    )

    print(
        "  height:",
        (
            "increasing"
            if height_direction > 0
            else
            "decreasing"
            if height_direction < 0
            else
            "constant"
        ),
    )

    print()
    print(
        "Direction violations"
    )

    print(
        f"  HEPlacement width : "
        f"{target_w_violations}"
    )

    print(
        f"  HEPlacement height: "
        f"{target_h_violations}"
    )

    print(
        f"  Rendered width    : "
        f"{rendered_w_violations}"
    )

    print(
        f"  Rendered height   : "
        f"{rendered_h_violations}"
    )

    print()
    print(
        "Rendered size local residual"
    )

    print(
        "  width median:",
        f"{np.median(width_residuals):.4f}px",
    )

    print(
        "  width P95   :",
        f"{np.percentile(width_residuals, 95):.4f}px",
    )

    print(
        "  width max   :",
        f"{np.max(width_residuals):.4f}px",
    )

    print(
        "  height median:",
        f"{np.median(height_residuals):.4f}px",
    )

    print(
        "  height P95   :",
        f"{np.percentile(height_residuals, 95):.4f}px",
    )

    print(
        "  height max   :",
        f"{np.max(height_residuals):.4f}px",
    )

    print()
    print(
        "Rendered / target size ratio"
    )

    print(
        "  width mean/std :",
        f"{np.mean(width_ratios):.6f}",
        "/",
        f"{np.std(width_ratios):.6f}",
    )

    print(
        "  height mean/std:",
        f"{np.mean(height_ratios):.6f}",
        "/",
        f"{np.std(height_ratios):.6f}",
    )

    max_scale_error_x = max(
        abs(
            r[
                "scale_error_x"
            ]
        )
        for r in rows
    )

    max_scale_error_y = max(
        abs(
            r[
                "scale_error_y"
            ]
        )
        for r in rows
    )

    max_place_error_x = max(
        abs(
            r[
                "placement_error_x"
            ]
        )
        for r in rows
    )

    max_place_error_y = max(
        abs(
            r[
                "placement_error_y"
            ]
        )
        for r in rows
    )

    print()
    print(
        "Continuous-transform sanity"
    )

    print(
        "  max scale error x:",
        f"{max_scale_error_x:.12f}",
    )

    print(
        "  max scale error y:",
        f"{max_scale_error_y:.12f}",
    )

    print(
        "  max placement error x:",
        f"{max_place_error_x:.12f}px",
    )

    print(
        "  max placement error y:",
        f"{max_place_error_y:.12f}px",
    )

    print()
    print(
        "Largest rendered width residual frames"
    )

    worst = sorted(
        rows[1:-1],
        key=lambda r:
            abs(
                r[
                    "width_local_residual"
                ]
            ),
        reverse=True,
    )[:15]

    print(
        "frame   targetW  renderW  dW   residual   dist  elev"
    )

    for r in worst:

        print(
            f"{r['frame_idx']:5d}  "
            f"{r['target_width']:7.2f}  "
            f"{r['rendered_width']:7.2f}  "
            f"{r['d_rendered_width']:4.0f}  "
            f"{r['width_local_residual']:8.3f}  "
            f"{r['selected_distance']:4.0f}  "
            f"{r['selected_elevation']:4.0f}"
        )

    print()
    print(
        "Saved:",
        csv_path,
    )

    print("=" * 88)


if __name__ == "__main__":
    main()