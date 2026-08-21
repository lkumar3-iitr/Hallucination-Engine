import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean(values):
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


def median(values):
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


def fmt(x):
    if not finite(x):
        return "nan"

    return f"{float(x):.4f}"


def load_mask_rows(path):
    rows = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            frame_idx = int(
                row["frame_idx"]
            )

            rows[frame_idx] = row

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
            continue

        adv = adversaries[0]

        sprite = adv.get(
            "sprite",
            {},
        )

        state = adv.get(
            "state",
            {},
        )

        rows[frame_idx] = {
            "query_angle_deg":
                float(
                    sprite.get(
                        "relative_angle_deg",
                        math.nan,
                    )
                ),

            "selected_angle_deg":
                float(
                    sprite.get(
                        "selected_angle",
                        math.nan,
                    )
                ),

            "angle_error_deg":
                float(
                    sprite.get(
                        "angle_error_deg",
                        math.nan,
                    )
                ),

            "query_distance_m":
                float(
                    sprite.get(
                        "query_distance_m",
                        math.nan,
                    )
                ),

            "selected_distance_m":
                float(
                    sprite.get(
                        "selected_distance_m",
                        math.nan,
                    )
                ),

            "query_elevation_deg":
                float(
                    sprite.get(
                        "query_elevation_deg",
                        math.nan,
                    )
                ),

            "selected_elevation_deg":
                float(
                    sprite.get(
                        "selected_elevation_deg",
                        math.nan,
                    )
                ),

            "state_yaw_deg":
                float(
                    state.get(
                        "yaw_deg",
                        math.nan,
                    )
                ),
        }

    return rows


def angle_bin(angle):
    """
    5-degree bins:
        0-5
        5-10
        10-15
        ...
    """

    angle = abs(
        float(angle)
    )

    low = (
        int(angle // 5.0)
        * 5
    )

    high = low + 5

    return (
        f"{low:02d}_{high:02d}",
        low,
        high,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask-csv",
        required=True,
    )

    parser.add_argument(
        "--he-metadata",
        required=True,
    )

    parser.add_argument(
        "--output-csv",
        required=True,
    )

    args = parser.parse_args()

    mask_rows = load_mask_rows(
        args.mask_csv
    )

    he_rows = load_he_metadata(
        args.he_metadata
    )

    combined = []

    for frame_idx in sorted(
        set(mask_rows.keys())
        & set(he_rows.keys())
    ):

        m = mask_rows[
            frame_idx
        ]

        h = he_rows[
            frame_idx
        ]

        row = {
            "frame_idx":
                frame_idx,

            **h,

            "mask_iou":
                float(
                    m["mask_iou"]
                ),

            "dice":
                float(
                    m["dice"]
                ),

            "area_ratio":
                float(
                    m["area_ratio"]
                ),

            "centroid_error_px":
                float(
                    m[
                        "centroid_error_px"
                    ]
                ),

            "boundary_f1":
                float(
                    m[
                        "boundary_f1"
                    ]
                ),

            "boundary_distance_px":
                float(
                    m[
                        "mean_boundary_distance_px"
                    ]
                ),
        }

        combined.append(
            row
        )

    bins = {}

    for row in combined:

        label, low, high = angle_bin(
            row[
                "state_yaw_deg"
            ]
        )

        if label not in bins:
            bins[label] = {
                "label":
                    label,

                "low":
                    low,

                "high":
                    high,

                "rows":
                    [],
            }

        bins[label][
            "rows"
        ].append(
            row
        )

    summaries = []

    for label in sorted(
        bins.keys()
    ):

        group = bins[
            label
        ]

        rows = group[
            "rows"
        ]

        summary = {
            "angle_bin":
                label,

            "angle_min_deg":
                group["low"],

            "angle_max_deg":
                group["high"],

            "num_frames":
                len(rows),

            "mean_mask_iou":
                mean(
                    [
                        r["mask_iou"]
                        for r in rows
                    ]
                ),

            "median_mask_iou":
                median(
                    [
                        r["mask_iou"]
                        for r in rows
                    ]
                ),

            "mean_dice":
                mean(
                    [
                        r["dice"]
                        for r in rows
                    ]
                ),

            "mean_area_ratio":
                mean(
                    [
                        r["area_ratio"]
                        for r in rows
                    ]
                ),

            "mean_centroid_error_px":
                mean(
                    [
                        r[
                            "centroid_error_px"
                        ]
                        for r in rows
                    ]
                ),

            "mean_boundary_f1":
                mean(
                    [
                        r["boundary_f1"]
                        for r in rows
                    ]
                ),

            "mean_boundary_distance_px":
                mean(
                    [
                        r[
                            "boundary_distance_px"
                        ]
                        for r in rows
                    ]
                ),
        }

        summaries.append(
            summary
        )

    output_path = Path(
        args.output_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if summaries:

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    summaries[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                summaries
            )

    print()
    print(
        "=" * 78
    )

    print(
        "MASK EQUIVALENCE BY VIEWPOINT V1"
    )

    print(
        "=" * 78
    )

    print(
        "Frames:",
        len(combined),
    )

    print()

    print(
        "yaw range | n | IoU | Dice | Area | Centroid | Boundary F1 | Boundary dist"
    )

    print(
        "-" * 100
    )

    for row in summaries:

        print(
            "{:>2d}-{:>2d} deg | "
            "{:3d} | "
            "{} | "
            "{} | "
            "{} | "
            "{} px | "
            "{} | "
            "{} px".format(
                row[
                    "angle_min_deg"
                ],

                row[
                    "angle_max_deg"
                ],

                row[
                    "num_frames"
                ],

                fmt(
                    row[
                        "mean_mask_iou"
                    ]
                ),

                fmt(
                    row[
                        "mean_dice"
                    ]
                ),

                fmt(
                    row[
                        "mean_area_ratio"
                    ]
                ),

                fmt(
                    row[
                        "mean_centroid_error_px"
                    ]
                ),

                fmt(
                    row[
                        "mean_boundary_f1"
                    ]
                ),

                fmt(
                    row[
                        "mean_boundary_distance_px"
                    ]
                ),
            )
        )

    print()

    best = max(
        summaries,
        key=lambda r:
            r[
                "mean_mask_iou"
            ],
    )

    worst = min(
        summaries,
        key=lambda r:
            r[
                "mean_mask_iou"
            ],
    )

    print(
        "Best IoU bin :",
        best[
            "angle_bin"
        ],
        fmt(
            best[
                "mean_mask_iou"
            ]
        ),
    )

    print(
        "Worst IoU bin:",
        worst[
            "angle_bin"
        ],
        fmt(
            worst[
                "mean_mask_iou"
            ]
        ),
    )

    print()
    print(
        "Saved:",
        output_path,
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()