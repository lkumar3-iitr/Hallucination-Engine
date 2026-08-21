import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


DISTANCE_BINS = [
    (5.0, 10.0, "5_10"),
    (10.0, 15.0, "10_15"),
    (15.0, 20.0, "15_20"),
    (20.0, 30.0, "20_30"),
    (30.0, 50.0, "30_50"),
    (50.0, float("inf"), "50_plus"),
]


def mask_geometry(path, threshold):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return None

    mask = image > threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    area = int(mask.sum())

    bbox_area = width * height

    return {
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
        "width": width,
        "height": height,
        "cx": 0.5 * (x1 + x2),
        "bottom_y": float(y2),
        "area": area,
        "bbox_area": bbox_area,
        "fill_ratio": (
            area / bbox_area
            if bbox_area > 0
            else 0.0
        ),
    }


def distance_bin(depth):
    for lo, hi, name in DISTANCE_BINS:
        if lo <= depth < hi:
            return name

    return "other"


def mean(rows, key):
    values = [
        float(r[key])
        for r in rows
        if r.get(key) not in (
            None,
            "",
        )
    ]

    if not values:
        return float("nan")

    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        required=True,
    )

    parser.add_argument(
        "--carla-mask-dir",
        required=True,
    )

    parser.add_argument(
        "--he-mask-dir",
        default=None,
    )

    parser.add_argument(
        "--mask-csv",
        default=None,
        help=(
            "Optional mask-equivalence per_frame.csv. "
            "Used for real_depth_m."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata)

    carla_mask_dir = Path(
        args.carla_mask_dir
    )

    he_mask_dir = (
        Path(args.he_mask_dir)
        if args.he_mask_dir
        else None
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    evaluator_rows = {}

    if args.mask_csv:

        with open(
            args.mask_csv,
            "r",
            encoding="utf-8",
            newline="",
        ) as f:

            for row in csv.DictReader(f):
                evaluator_rows[
                    int(row["frame_idx"])
                ] = row

    rows = []

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

        placement = actor.get(
            "box"
        )

        if not placement:
            continue

        if not placement.get(
            "visible",
            False,
        ):
            continue

        carla_path = (
            carla_mask_dir
            / f"frame_{frame_idx:06d}_mask.png"
        )

        carla = mask_geometry(
            carla_path,
            127,
        )

        if carla is None:
            continue

        if frame_idx in evaluator_rows:

            depth = float(
                evaluator_rows[
                    frame_idx
                ]["real_depth_m"]
            )

        else:

            depth = float(
                placement.get(
                    "z_m",
                    actor.get(
                        "state",
                        {},
                    ).get(
                        "z_m",
                        float("nan"),
                    ),
                )
            )

        placement_width = float(
            placement["box_width"]
        )

        placement_height = float(
            placement["box_height"]
        )

        placement_cx = float(
            placement["cx"]
        )

        placement_bottom = float(
            placement["bottom_y"]
        )

        row = {
            "frame_idx": frame_idx,
            "depth_m": depth,
            "distance_bin": (
                distance_bin(depth)
            ),

            "placement_width_px":
                placement_width,

            "placement_height_px":
                placement_height,

            "carla_mask_width_px":
                carla["width"],

            "carla_mask_height_px":
                carla["height"],

            "placement_to_carla_width":
                placement_width
                / carla["width"],

            "placement_to_carla_height":
                placement_height
                / carla["height"],

            "placement_cx_error_px":
                placement_cx
                - carla["cx"],

            "placement_bottom_error_px":
                placement_bottom
                - carla["bottom_y"],

            "carla_mask_area":
                carla["area"],

            "carla_mask_fill_ratio":
                carla["fill_ratio"],
        }

        if he_mask_dir is not None:

            he_path = (
                he_mask_dir
                / f"frame_{frame_idx:06d}_mask.png"
            )

            he = mask_geometry(
                he_path,
                10,
            )

            if he is not None:

                row[
                    "he_mask_width_px"
                ] = he["width"]

                row[
                    "he_mask_height_px"
                ] = he["height"]

                row[
                    "he_mask_area"
                ] = he["area"]

                row[
                    "he_mask_fill_ratio"
                ] = he["fill_ratio"]

                row[
                    "he_to_carla_width"
                ] = (
                    he["width"]
                    / carla["width"]
                )

                row[
                    "he_to_carla_height"
                ] = (
                    he["height"]
                    / carla["height"]
                )

                row[
                    "he_to_carla_area"
                ] = (
                    he["area"]
                    / carla["area"]
                )

                row[
                    "he_to_carla_fill"
                ] = (
                    he["fill_ratio"]
                    / carla["fill_ratio"]
                )

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No comparable frames found."
        )

    csv_path = (
        output_dir
        / "per_frame.csv"
    )

    keys = list(rows[0].keys())

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 96)
    print(
        "PLACEMENT VS CARLA INSTANCE MASK V1"
    )
    print("=" * 96)

    print(
        "bin       n   "
        "P/C width  "
        "P/C height  "
        "P-C bottom  "
        "P-C cx"
        "      "
        "HE/C area   "
        "HE/C fill"
    )

    print("-" * 96)

    for _, _, bin_name in DISTANCE_BINS:

        subset = [
            r
            for r in rows
            if r["distance_bin"]
            == bin_name
        ]

        if not subset:
            continue

        print(
            f"{bin_name:9s} "
            f"{len(subset):3d}  "
            f"{mean(subset, 'placement_to_carla_width'):9.4f}  "
            f"{mean(subset, 'placement_to_carla_height'):10.4f}  "
            f"{mean(subset, 'placement_bottom_error_px'):10.3f}  "
            f"{mean(subset, 'placement_cx_error_px'):7.3f}  "
            f"{mean(subset, 'he_to_carla_area'):10.4f}  "
            f"{mean(subset, 'he_to_carla_fill'):9.4f}"
        )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  P/C width,height > 1:"
        " placement target itself"
        " is larger than CARLA's"
        " visible mask bbox."
    )

    print(
        "  HE/C fill > 1:"
        " HE silhouette is fuller"
        " inside its bounding box."
    )

    print(
        "  Increasing bottom error:"
        " possible close-range"
        " vertical placement bias."
    )

    print()
    print(
        "Saved:",
        csv_path,
    )

    print("=" * 96)


if __name__ == "__main__":
    main()