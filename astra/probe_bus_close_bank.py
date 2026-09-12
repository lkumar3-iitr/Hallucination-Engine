"""Compare native-bank ASTRA with pose-nearest close-bank bus appearance."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from passby_renderer import PassbyRenderer
from selector import composite_to_box, transform


def angle_delta(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def mask_iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def align_visible_fragment(background, premultiplied_rgb, alpha, box):
    ys, xs = np.where(alpha > 10 / 255)
    if not len(xs):
        return background.copy(), np.zeros(background.shape[:2], np.float32)
    rgb_crop = premultiplied_rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alpha_crop = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    x1, y1, x2, y2 = map(int, box)
    width, height = x2 - x1, y2 - y1
    rgb = cv2.resize(rgb_crop, (width, height), interpolation=cv2.INTER_LINEAR)
    a = cv2.resize(alpha_crop, (width, height), interpolation=cv2.INTER_LINEAR)
    output = background.copy()
    output[y1:y2, x1:x2] = np.rint(np.clip(
        rgb + background[y1:y2, x1:x2] * (1 - a[:, :, None]), 0, 255
    )).astype(np.uint8)
    alpha_canvas = np.zeros(background.shape[:2], np.float32)
    alpha_canvas[y1:y2, x1:x2] = a
    return output, alpha_canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native-bank", type=Path, required=True)
    parser.add_argument("--close-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=30)
    parser.add_argument("--end", type=int, default=61)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    setup = json.loads((args.reference / "setup.json").read_text())
    frames = [json.loads(line) for line in
              (args.reference / "frames.jsonl").read_text().splitlines()]
    with (args.close_bank / "view_matrix.csv").open(newline="", encoding="utf-8") as file:
        close_rows = list(csv.DictReader(file))
    renderer = PassbyRenderer(args.native_bank, args.output.parent / "cache")
    cap = cv2.VideoCapture(str(args.reference / "carla_reference.mp4"))
    writer = cv2.VideoWriter(
        str(args.output / "close_bank_probe.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        setup["fps"], (setup["camera"]["width"] * 3, setup["camera"]["height"]),
    )
    results = []
    try:
        for index in range(args.end + 1):
            ok, carla_rgb = cap.read()
            if not ok:
                raise RuntimeError(f"Missing reference frame {index}")
            if index < args.start:
                continue
            row = frames[index]
            actor = transform(row["actor_transform"])
            camera = transform(row["camera_transform"])
            actor_inverse = np.asarray(actor.get_inverse_matrix())
            camera_local = (actor_inverse @ np.array([
                camera.location.x, camera.location.y, camera.location.z, 1.0
            ]))[:3]
            relative_yaw = (actor.rotation.yaw - camera.rotation.yaw + 180) % 360 - 180
            candidates = [r for r in close_rows if angle_delta(
                float(r["close_relative_yaw_deg"]), relative_yaw) <= 0.1]
            if not candidates:
                raise RuntimeError(f"No close-bank yaw at frame {index}")
            selected = min(candidates, key=lambda r:
                (float(r["camera_bus_local_x_m"]) - camera_local[0]) ** 2 +
                (float(r["camera_bus_local_y_m"]) - camera_local[1]) ** 2)

            background = np.zeros_like(carla_rgb)
            native_rgb, native_alpha, _ = renderer.render(
                background, actor, camera, setup["camera"]["fov_deg"]
            )
            ys, xs = np.where(native_alpha > 10 / 255)
            if not len(xs):
                continue
            native_box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            close_sprite = cv2.imread(
                str(args.close_bank / selected["rgba_relpath"]), cv2.IMREAD_UNCHANGED
            )
            runtime_fx = setup["camera"]["width"] / (
                2 * np.tan(np.deg2rad(setup["camera"]["fov_deg"]) / 2)
            )
            scale = runtime_fx / float(selected["camera_fx_px"])
            source_x1 = float(selected["crop_x1_px"])
            source_y1 = float(selected["crop_y1_px"])
            target_x1 = ((source_x1 - float(selected["camera_cx_px"])) * scale
                         + setup["camera"]["width"] / 2)
            target_y1 = ((source_y1 - float(selected["camera_cy_px"])) * scale
                         + setup["camera"]["height"] / 2)
            calibrated_box = [
                target_x1,
                target_y1,
                target_x1 + close_sprite.shape[1] * scale,
                target_y1 + close_sprite.shape[0] * scale,
            ]
            close_rgb, close_alpha = composite_to_box(
                background, close_sprite, calibrated_box
            )
            close_rgb, close_alpha = align_visible_fragment(
                background, close_rgb, close_alpha, native_box
            )
            gt = cv2.imread(str(args.reference / row["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
            native_mask, close_mask = native_alpha > 10 / 255, close_alpha > 10 / 255
            record = {
                "frame": index,
                "camera_local_x": float(camera_local[0]),
                "camera_local_y": float(camera_local[1]),
                "selected_x": float(selected["camera_bus_local_x_m"]),
                "selected_y": float(selected["camera_bus_local_y_m"]),
                "selected_path": selected["rgba_relpath"],
                "native_iou": mask_iou(gt, native_mask),
                "close_iou": mask_iou(gt, close_mask),
            }
            results.append(record)
            panel = np.concatenate((carla_rgb, native_rgb, close_rgb), axis=1)
            cv2.putText(panel, f"CARLA | native ASTRA | close bank  frame {index}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            cv2.putText(panel, f"native IoU {record['native_iou']:.3f}  close IoU {record['close_iou']:.3f}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            writer.write(panel)
            if index in (35, 38, 40, 43, 45, 50, 55, 60):
                cv2.imwrite(str(args.output / f"frame_{index:06d}.png"), panel)
    finally:
        cap.release()
        writer.release()

    with (args.output / "rows.csv").open("w", newline="") as file:
        output = csv.DictWriter(file, fieldnames=results[0].keys())
        output.writeheader()
        output.writerows(results)
    summary = {
        "frames": len(results),
        "native_mean_iou": float(np.mean([r["native_iou"] for r in results])),
        "close_mean_iou": float(np.mean([r["close_iou"] for r in results])),
        "close_better_frames": sum(r["close_iou"] > r["native_iou"] for r in results),
        "selection_uses_gt": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
