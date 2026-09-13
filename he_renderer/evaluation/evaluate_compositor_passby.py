"""Evaluate the production hybrid compositor on a frozen CARLA pass-by."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from he_renderer.compositor import HESpriteRendererCompositor
from he_renderer.selector import transform


def read_frame(capture, index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame {index}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--close-bank", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir(exist_ok=True)
    setup = json.loads((args.reference / "setup.json").read_text())
    records = [json.loads(line) for line in (args.reference / "frames.jsonl").read_text().splitlines()]
    camera = setup["camera"]
    width, height, fov = camera["width"], camera["height"], camera["fov_deg"]
    metadata = json.loads((args.bank / "asset_metadata.json").read_text())
    bbox = metadata["physical_bbox"]
    dimensions = SimpleNamespace(
        length_m=float(bbox["length_m"]),
        width_m=float(bbox["width_m"]),
        height_m=float(bbox["height_m"]),
    )
    compositor = HESpriteRendererCompositor()
    physical = cv2.VideoCapture(str(args.reference / "carla_reference.mp4"))
    background = cv2.VideoCapture(str(args.reference / "background_only.mp4"))
    writer = cv2.VideoWriter(
        str(args.output / "hybrid_comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        setup["fps"], (width * 2, height + 45),
    )
    rows = []
    try:
        for index, record in enumerate(records):
            carla_frame = read_frame(physical, index)
            clean_frame = read_frame(background, index)
            actor_tf = transform(record["actor_transform"])
            camera_tf = transform(record["camera_transform"])
            actor = SimpleNamespace(
                actor_id="target",
                canonical_asset_key=setup["asset_id"],
                carla_blueprint=setup["carla_blueprint"],
                world_x_m=actor_tf.location.x,
                world_y_m=actor_tf.location.y,
                world_z_m=actor_tf.location.z,
                world_yaw_deg=actor_tf.rotation.yaw,
                physical_dimensions=dimensions,
                he_view_matrix_csv=str(args.bank / "view_matrix.csv"),
                he_close_view_matrix_csvs={
                    "left": [str(bank / "view_matrix.csv") for bank in args.close_bank],
                    "right": [str(bank / "view_matrix.csv") for bank in args.close_bank],
                },
            )
            rendered = compositor.render(
                cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB), camera_tf, [actor],
                width, height, fov,
            )
            he_frame = cv2.cvtColor(rendered.rgb, cv2.COLOR_RGB2BGR)
            result = rendered.actor_results[0]
            mode = result.he_metadata.get("sprite_mode", "")
            difference = np.max(np.abs(
                he_frame.astype(np.int16) - clean_frame.astype(np.int16)
            ), axis=2)
            predicted = difference > 2
            gt = cv2.imread(str(args.reference / record["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
            union = np.count_nonzero(predicted | gt)
            iou = float(np.count_nonzero(predicted & gt) / union) if union else 1.0
            rows.append({
                "frame": index,
                "iou": iou,
                "gt_pixels": int(np.count_nonzero(gt)),
                "predicted_pixels": int(np.count_nonzero(predicted)),
                "sprite_mode": mode,
                "selected_angle": result.he_metadata.get("selected_angle", ""),
                "selected_distance_m": result.he_metadata.get("selected_distance_m", ""),
            })
            panel = np.zeros((height + 45, width * 2, 3), np.uint8)
            panel[45:, :width] = carla_frame
            panel[45:, width:] = he_frame
            cv2.putText(panel, f"CARLA | HE hybrid  frame={index}  IoU={iou:.3f}  mode={mode}",
                        (8, 29), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            writer.write(panel)
            if index % 5 == 0:
                cv2.imwrite(str(args.output / "frames" / f"frame_{index:06d}.jpg"), panel)
    finally:
        physical.release()
        background.release()
        writer.release()

    with (args.output / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=rows[0])
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    visible = [row for row in rows if row["gt_pixels"]]
    summary = {
        "frames": len(rows),
        "visible_frames": len(visible),
        "mean_visible_iou": float(np.mean([row["iou"] for row in visible])),
        "minimum_visible_iou": min(row["iou"] for row in visible),
        "mode_counts": {
            mode: sum(row["sprite_mode"] == mode for row in rows)
            for mode in sorted({row["sprite_mode"] for row in rows})
        },
        "frame_90": rows[90],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
