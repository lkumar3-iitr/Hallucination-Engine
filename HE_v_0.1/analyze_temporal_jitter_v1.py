import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def circular_delta_deg(a, b):
    return (
        (float(a) - float(b) + 180.0)
        % 360.0
    ) - 180.0


def percentile(values, q):
    if not values:
        return float("nan")

    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
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
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    metadata_path = Path(
        args.metadata
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

        box = actor.get(
            "box",
            {},
        )

        sprite = actor.get(
            "sprite",
            {},
        )

        resize = actor.get(
            "view_matrix_resize",
            {},
        )

        paste = actor.get(
            "paste",
            {},
        )

        if not box.get(
            "visible",
            False,
        ):
            continue

        rows.append({
            "frame_idx":
                frame_idx,

            "cx":
                float(box["cx"]),

            "bottom_y":
                float(box["bottom_y"]),

            "width":
                float(box["box_width"]),

            "height":
                float(box["box_height"]),

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

            "paste_x":
                float(
                    paste.get(
                        "x1",
                        0.0,
                    )
                ),

            "paste_y":
                float(
                    paste.get(
                        "y1",
                        0.0,
                    )
                ),

            "sprite_width":
                float(
                    paste.get(
                        "sprite_width",
                        resize.get(
                            "resized_width",
                            0.0,
                        ),
                    )
                ),

            "sprite_height":
                float(
                    paste.get(
                        "sprite_height",
                        resize.get(
                            "resized_height",
                            0.0,
                        ),
                    )
                ),
        })

    if len(rows) < 3:
        raise RuntimeError(
            "Need at least three rendered frames."
        )

    output_rows = []

    for i in range(
        1,
        len(rows) - 1,
    ):

        prev = rows[i - 1]
        cur = rows[i]
        nxt = rows[i + 1]

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

        angle_switch = (
            abs(
                circular_delta_deg(
                    cur["selected_angle"],
                    prev["selected_angle"],
                )
            )
            > 1e-6
        )

        distance_switch = (
            abs(
                cur["selected_distance"]
                -
                prev["selected_distance"]
            )
            > 1e-6
        )

        elevation_switch = (
            abs(
                cur["selected_elevation"]
                -
                prev["selected_elevation"]
            )
            > 1e-6
        )

        any_switch = (
            angle_switch
            or distance_switch
            or elevation_switch
        )

        r_cx = residual("cx")
        r_bottom = residual("bottom_y")
        r_width = residual("width")
        r_height = residual("height")

        r_paste_x = residual(
            "paste_x"
        )

        r_paste_y = residual(
            "paste_y"
        )

        r_sprite_w = residual(
            "sprite_width"
        )

        r_sprite_h = residual(
            "sprite_height"
        )

        # Geometry jitter score.
        #
        # cx/bottom are full weight.
        # width/height are half weight.
        geometry_score = math.sqrt(
            r_cx * r_cx
            +
            r_bottom * r_bottom
            +
            0.25
            * r_width * r_width
            +
            0.25
            * r_height * r_height
        )

        paste_score = math.sqrt(
            r_paste_x * r_paste_x
            +
            r_paste_y * r_paste_y
            +
            0.25
            * r_sprite_w * r_sprite_w
            +
            0.25
            * r_sprite_h * r_sprite_h
        )

        output_rows.append({
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
                cur["selected_angle"],

            "selected_distance":
                cur["selected_distance"],

            "selected_elevation":
                cur["selected_elevation"],

            "angle_switch":
                int(angle_switch),

            "distance_switch":
                int(distance_switch),

            "elevation_switch":
                int(elevation_switch),

            "any_switch":
                int(any_switch),

            "residual_cx_px":
                r_cx,

            "residual_bottom_px":
                r_bottom,

            "residual_width_px":
                r_width,

            "residual_height_px":
                r_height,

            "residual_paste_x_px":
                r_paste_x,

            "residual_paste_y_px":
                r_paste_y,

            "residual_sprite_width_px":
                r_sprite_w,

            "residual_sprite_height_px":
                r_sprite_h,

            "geometry_jitter_score":
                geometry_score,

            "paste_jitter_score":
                paste_score,
        })

    csv_path = (
        output_dir
        / "temporal_jitter.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    geometry_scores = [
        r["geometry_jitter_score"]
        for r in output_rows
    ]

    paste_scores = [
        r["paste_jitter_score"]
        for r in output_rows
    ]

    switch_rows = [
        r
        for r in output_rows
        if r["any_switch"]
    ]

    nonswitch_rows = [
        r
        for r in output_rows
        if not r["any_switch"]
    ]

    def mean_score(
        subset,
        key,
    ):
        if not subset:
            return float("nan")

        return float(
            np.mean([
                r[key]
                for r in subset
            ])
        )

    print()
    print("=" * 88)
    print(
        "HE TEMPORAL JITTER ANALYSIS V1"
    )
    print("=" * 88)

    print(
        f"Analyzed frames       : "
        f"{len(output_rows)}"
    )

    print(
        f"Sprite-switch frames  : "
        f"{len(switch_rows)}"
    )

    print()
    print("Geometry residual")

    print(
        f"  median : "
        f"{percentile(geometry_scores, 50):.4f} px"
    )

    print(
        f"  P95    : "
        f"{percentile(geometry_scores, 95):.4f} px"
    )

    print(
        f"  max    : "
        f"{max(geometry_scores):.4f} px"
    )

    print()
    print("Final paste residual")

    print(
        f"  median : "
        f"{percentile(paste_scores, 50):.4f} px"
    )

    print(
        f"  P95    : "
        f"{percentile(paste_scores, 95):.4f} px"
    )

    print(
        f"  max    : "
        f"{max(paste_scores):.4f} px"
    )

    print()
    print("Switch vs non-switch")

    print(
        "  Geometry:"
    )

    print(
        f"    switch     "
        f"{mean_score(switch_rows, 'geometry_jitter_score'):.4f} px"
    )

    print(
        f"    non-switch "
        f"{mean_score(nonswitch_rows, 'geometry_jitter_score'):.4f} px"
    )

    print(
        "  Paste:"
    )

    print(
        f"    switch     "
        f"{mean_score(switch_rows, 'paste_jitter_score'):.4f} px"
    )

    print(
        f"    non-switch "
        f"{mean_score(nonswitch_rows, 'paste_jitter_score'):.4f} px"
    )

    print()
    print("Largest jitter frames")
    print(
        "frame  score  paste  "
        "angle  dist  elev  switch"
    )

    worst = sorted(
        output_rows,
        key=lambda r:
            r["paste_jitter_score"],
        reverse=True,
    )[:20]

    for r in worst:

        print(
            f"{r['frame_idx']:5d}  "
            f"{r['geometry_jitter_score']:6.3f}  "
            f"{r['paste_jitter_score']:6.3f}  "
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

    print("=" * 88)


if __name__ == "__main__":
    main()