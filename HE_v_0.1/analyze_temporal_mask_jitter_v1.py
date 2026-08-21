import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_mask(path, threshold=10):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return None

    return image > threshold


def mask_geometry(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    return {
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,

        "width": x2 - x1 + 1,
        "height": y2 - y1 + 1,

        "cx": 0.5 * (x1 + x2),
        "bottom_y": float(y2),
    }


def normalized_shape(mask, geom, size=256):

    crop = mask[
        geom["y1"]:geom["y2"] + 1,
        geom["x1"]:geom["x2"] + 1,
    ].astype(np.uint8)

    norm = cv2.resize(
        crop,
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )

    return norm > 0


def mask_iou(a, b):

    inter = np.logical_and(
        a,
        b,
    ).sum()

    union = np.logical_or(
        a,
        b,
    ).sum()

    if union == 0:
        return 1.0

    return float(
        inter / union
    )


def circular_changed(a, b):

    delta = (
        (float(a) - float(b) + 180.0)
        % 360.0
    ) - 180.0

    return abs(delta) > 1e-6


def mean(rows, key):

    if not rows:
        return float("nan")

    return float(
        np.mean([
            float(r[key])
            for r in rows
        ])
    )


def percentile(rows, key, q):

    if not rows:
        return float("nan")

    return float(
        np.percentile(
            [
                float(r[key])
                for r in rows
            ],
            q,
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

        sprite = actor.get(
            "sprite",
            {},
        )

        mask_path = (
            mask_dir
            / f"frame_{frame_idx:06d}_mask.png"
        )

        mask = read_mask(
            mask_path,
            10,
        )

        if mask is None:
            continue

        geom = mask_geometry(
            mask
        )

        if geom is None:
            continue

        shape = normalized_shape(
            mask,
            geom,
        )

        frames.append({
            "frame_idx":
                frame_idx,

            "mask":
                mask,

            "shape":
                shape,

            "cx":
                geom["cx"],

            "bottom_y":
                geom["bottom_y"],

            "width":
                geom["width"],

            "height":
                geom["height"],

            "angle":
                float(
                    sprite.get(
                        "selected_angle",
                        0.0,
                    )
                ),

            "distance":
                float(
                    sprite.get(
                        "selected_distance_m",
                        0.0,
                    )
                ),

            "elevation":
                float(
                    sprite.get(
                        "selected_elevation_deg",
                        0.0,
                    )
                ),
        })

    if len(frames) < 3:
        raise RuntimeError(
            "Need at least 3 mask frames."
        )

    rows = []

    for i in range(
        1,
        len(frames) - 1,
    ):

        prev = frames[i - 1]
        cur = frames[i]
        nxt = frames[i + 1]

        def residual(key):
            return (
                cur[key]
                -
                0.5
                * (
                    prev[key]
                    +
                    nxt[key]
                )
            )

        r_cx = residual("cx")
        r_bottom = residual("bottom_y")
        r_width = residual("width")
        r_height = residual("height")

        geometry_jitter = float(
            np.sqrt(
                r_cx ** 2
                +
                r_bottom ** 2
                +
                0.25 * r_width ** 2
                +
                0.25 * r_height ** 2
            )
        )

        angle_switch = circular_changed(
            cur["angle"],
            prev["angle"],
        )

        distance_switch = (
            abs(
                cur["distance"]
                -
                prev["distance"]
            )
            > 1e-6
        )

        elevation_switch = (
            abs(
                cur["elevation"]
                -
                prev["elevation"]
            )
            > 1e-6
        )

        any_switch = (
            angle_switch
            or distance_switch
            or elevation_switch
        )

        temporal_shape_iou = mask_iou(
            prev["shape"],
            cur["shape"],
        )

        rows.append({
            "frame_idx":
                cur["frame_idx"],

            "cx":
                cur["cx"],

            "bottom_y":
                cur["bottom_y"],

            "width":
                cur["width"],

            "height":
                cur["height"],

            "selected_angle":
                cur["angle"],

            "selected_distance":
                cur["distance"],

            "selected_elevation":
                cur["elevation"],

            "angle_switch":
                int(angle_switch),

            "distance_switch":
                int(distance_switch),

            "elevation_switch":
                int(elevation_switch),

            "any_switch":
                int(any_switch),

            "geometry_jitter_px":
                geometry_jitter,

            "temporal_shape_iou":
                temporal_shape_iou,
        })

    csv_path = (
        output_dir
        / "temporal_mask_jitter.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    switch_rows = [
        r
        for r in rows
        if r["any_switch"]
    ]

    nonswitch_rows = [
        r
        for r in rows
        if not r["any_switch"]
    ]

    print()
    print("=" * 92)
    print(
        "HE TEMPORAL MASK JITTER V1"
    )
    print("=" * 92)

    print(
        f"Analyzed frames      : {len(rows)}"
    )

    print(
        f"Sprite-switch frames : {len(switch_rows)}"
    )

    print()
    print("Final-mask geometry jitter")

    print(
        f"  median : "
        f"{percentile(rows, 'geometry_jitter_px', 50):.4f} px"
    )

    print(
        f"  P95    : "
        f"{percentile(rows, 'geometry_jitter_px', 95):.4f} px"
    )

    print(
        f"  max    : "
        f"{max(r['geometry_jitter_px'] for r in rows):.4f} px"
    )

    print()
    print("BBox-normalized consecutive shape IoU")

    print(
        f"  all        : "
        f"{mean(rows, 'temporal_shape_iou'):.4f}"
    )

    print(
        f"  switch     : "
        f"{mean(switch_rows, 'temporal_shape_iou'):.4f}"
    )

    print(
        f"  non-switch : "
        f"{mean(nonswitch_rows, 'temporal_shape_iou'):.4f}"
    )

    print()
    print("Geometry jitter")

    print(
        f"  switch     : "
        f"{mean(switch_rows, 'geometry_jitter_px'):.4f} px"
    )

    print(
        f"  non-switch : "
        f"{mean(nonswitch_rows, 'geometry_jitter_px'):.4f} px"
    )

    print()
    print("Lowest temporal shape-IoU frames")

    worst = sorted(
        rows,
        key=lambda r:
            r["temporal_shape_iou"],
    )[:20]

    print(
        "frame   IoU    jitter   angle  dist  elev  switch"
    )

    for r in worst:

        print(
            f"{r['frame_idx']:5d}  "
            f"{r['temporal_shape_iou']:.4f}  "
            f"{r['geometry_jitter_px']:7.3f}  "
            f"{r['selected_angle']:6.1f}  "
            f"{r['selected_distance']:4.0f}  "
            f"{r['selected_elevation']:4.0f}  "
            f"{r['any_switch']}"
        )

    print()
    print(
        "Saved:",
        csv_path,
    )

    print("=" * 92)


if __name__ == "__main__":
    main()