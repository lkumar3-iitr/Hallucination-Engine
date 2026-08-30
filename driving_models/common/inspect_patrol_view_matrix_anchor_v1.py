
"""
inspect_patrol_view_matrix_anchor_v1.py

Inspect the exact production Nissan Patrol source sprites selected by the
close-range sweep.

For each query pose, report:
- selected angle/distance/elevation
- source image size
- visible alpha bbox
- stored ground anchor
- anchor relative to visible bbox
- vertical/horizontal gap from visible silhouette to stored anchor

No CARLA connection required.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def normalize_angle_180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def circular_error(a, b):
    return abs(normalize_angle_180(float(a) - float(b)))


def get_alpha_bbox(path, threshold=10):
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if bgra is None:
        raise RuntimeError(f"Could not read: {path}")

    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise RuntimeError(f"Expected RGBA/BGRA image: {path}")

    alpha = bgra[:, :, 3]
    ys, xs = np.where(alpha > int(threshold))

    if len(xs) == 0:
        return None

    return {
        "x1": int(xs.min()),
        "x2": int(xs.max()),
        "y1": int(ys.min()),
        "y2": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
        "image_width": int(bgra.shape[1]),
        "image_height": int(bgra.shape[0]),
    }


def f(row, name, default=None):
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument(
        "--folder",
        default="nissan_patrol_2021_native_full_v3",
    )
    parser.add_argument("--alpha-threshold", type=int, default=10)
    args = parser.parse_args()

    asset_dir = Path(args.asset_root) / args.folder
    csv_path = asset_dir / "view_matrix.csv"
    rgba_dir = asset_dir / "rgba"

    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    print("=" * 100)
    print("PATROL VIEW-MATRIX SOURCE ANCHOR INSPECTION")
    print("=" * 100)
    print("[CSV]", csv_path)
    print("[fields]")
    print(fieldnames)
    print()

    # Exact selected views from the latest sweep.
    queries = [
        # z, query_angle, selected_angle, query_elev, selected_elev
        (20.0, 171.469, 171, 2.8, 5.0),
        (15.0, 168.690, 169, 3.7, 5.0),
        (12.0, 165.964, 166, 4.5, 5.0),
        (10.0, 163.301, 163, 5.4, 5.0),
        (8.0, 159.444, 159, 6.6, 5.0),
        (7.0, 156.801, 157, 7.4, 5.0),
        (6.0, 153.435, 153, 8.3, 10.0),
        (5.0, 149.036, 149, 9.6, 10.0),
        (4.0, 143.130, 143, 11.1, 10.0),
        (3.5, 139.399, 139, 12.0, 10.0),
        (3.0, 135.000, 135, 13.1, 10.0),
    ]

    available_distances = sorted(
        {
            float(r["distance_m"])
            for r in rows
        }
    )

    index = {}
    for r in rows:
        key = (
            int(round(float(r["angle_deg"]))) % 360,
            float(r["distance_m"]),
            float(r["elevation_deg"]),
        )
        index[key] = r

    print(
        "z     selA selD selE  imgWxH    alphaWxH  "
        "anchor(x,y)  alphaBottom  dy(anchor-bottom)  "
        "anchorY/alphaH"
    )
    print("-" * 100)

    for (
        z,
        _query_angle,
        sel_angle,
        _query_elev,
        sel_elev,
    ) in queries:

        # The current runtime uses nearest linear physical distance.
        # Use lateral = -3 m and target vertical coordinate consistent
        # with the current test to recover the same query distance.
        horizontal = math.sqrt(z * z + 3.0 * 3.0)

        # TCP camera z=2.0, target_height=0.75:
        target_y = -2.0 + 0.75
        query_distance = math.sqrt(
            horizontal * horizontal
            + target_y * target_y
        )

        sel_distance = min(
            available_distances,
            key=lambda d: abs(query_distance - d),
        )

        key = (
            int(sel_angle),
            float(sel_distance),
            float(sel_elev),
        )

        row = index.get(key)

        if row is None:
            print(f"z={z}: missing key {key}")
            continue

        rgba_text = str(row.get("rgba_path", "")).replace("\\", "/")
        filename = rgba_text.split("/")[-1]
        rgba_path = rgba_dir / filename

        bbox = get_alpha_bbox(
            rgba_path,
            threshold=args.alpha_threshold,
        )

        if bbox is None:
            print(f"z={z}: no alpha: {rgba_path}")
            continue

        anchor_x = f(row, "anchor_x", float("nan"))
        anchor_y = f(row, "anchor_y", float("nan"))

        dy = anchor_y - float(bbox["y2"])
        dx_center = (
            anchor_x
            - 0.5 * (bbox["x1"] + bbox["x2"])
        )

        print(
            f"{z:4.1f}  "
            f"{sel_angle:4d} "
            f"{sel_distance:4.0f} "
            f"{sel_elev:4.0f}  "
            f"{bbox['image_width']:4d}x{bbox['image_height']:<4d} "
            f"{bbox['width']:4d}x{bbox['height']:<4d} "
            f"({anchor_x:7.2f},{anchor_y:7.2f}) "
            f"{bbox['y2']:11d} "
            f"{dy:18.2f} "
            f"{anchor_y / max(1.0, float(bbox['height'])):14.3f}"
        )

        print(
            "      ",
            f"alpha_bbox=({bbox['x1']},{bbox['y1']})-"
            f"({bbox['x2']},{bbox['y2']})",
            f"anchor_minus_alpha_center_x={dx_center:+.2f}",
            f"file={filename}",
        )

    print()
    print("Interpretation:")
    print(
        "  dy > 0 : stored ground anchor is BELOW the visible alpha bottom."
    )
    print(
        "  dy < 0 : stored ground anchor is ABOVE/inside the visible alpha bottom."
    )
    print(
        "A large |dy| that changes with selected view can create close-range "
        "vertical clipping when the full sprite is scaled around that anchor."
    )


if __name__ == "__main__":
    main()
