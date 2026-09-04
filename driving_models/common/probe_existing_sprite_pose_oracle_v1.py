#!/usr/bin/env python3
"""Search existing HE sprite banks against frozen CARLA reference frames.

This is a diagnostic, not a production renderer. Candidate sprites are aligned
to the CARLA visible-mask center, bottom, and height with one uniform scale so
the remaining mask IoU primarily measures silhouette/viewpoint compatibility.
An optional pure camera-rotation homography tests whether a centered 4320 view
contains the correct surface appearance but uses the wrong optical axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import carla
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-dir", type=Path, required=True)
    parser.add_argument("--he-dir", type=Path, required=True)
    parser.add_argument("--far-bank", type=Path, required=True)
    parser.add_argument("--close-left-bank", type=Path, required=True)
    parser.add_argument("--close-right-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--angle-radius-deg", type=int, default=25)
    parser.add_argument("--top", type=int, default=5)
    return parser.parse_args()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_csv(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def transform_from_dict(value):
    return carla.Transform(
        carla.Location(
            x=float(value["x"]), y=float(value["y"]), z=float(value["z"])
        ),
        carla.Rotation(
            pitch=float(value["pitch"]),
            yaw=float(value["yaw"]),
            roll=float(value["roll"]),
        ),
    )


def rotation_matrix(transform):
    return np.asarray(transform.get_matrix(), dtype=np.float64)[:3, :3]


def intrinsic(width, height, fov_deg):
    focal = float(width) / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.asarray([
        [focal, 0.0, float(width) / 2.0],
        [0.0, focal, float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def source_path(bank_dir, row):
    relative = row.get("rgba_relpath", "").strip()
    if relative:
        return bank_dir / relative
    path = Path(row["rgba_path"])
    if path.is_file():
        return path
    return bank_dir / "rgba" / path.name


def alpha_bbox(alpha):
    ys, xs = np.where(alpha > 10)
    if not len(xs):
        return None
    return {
        "x1": float(xs.min()), "x2": float(xs.max()),
        "y1": float(ys.min()), "y2": float(ys.max()),
        "width": float(xs.max() - xs.min() + 1),
        "height": float(ys.max() - ys.min() + 1),
        "cx": (float(xs.min()) + float(xs.max())) / 2.0,
        "bottom": float(ys.max()),
    }


def mask_iou(first, second):
    union = np.count_nonzero(first | second)
    if not union:
        return 1.0
    return float(np.count_nonzero(first & second) / union)


def rotation_homography(
    row, source_actor_tf, source_camera_tf, target_actor_tf,
    target_camera_tf, target_width, target_height, target_fov,
):
    source_width = float(row["image_width_px"])
    source_height = float(row["image_height_px"])
    source_fov = float(row["fov_deg"])
    source_k = intrinsic(source_width, source_height, source_fov)
    target_k = intrinsic(target_width, target_height, target_fov)

    source_camera_to_actor = (
        rotation_matrix(source_actor_tf).T
        @ rotation_matrix(source_camera_tf)
    )
    actor_to_target_camera = (
        rotation_matrix(target_camera_tf).T
        @ rotation_matrix(target_actor_tf)
    )
    ray_rotation = actor_to_target_camera @ source_camera_to_actor

    # CARLA camera rays are [forward, right, up], while image homogeneous
    # coordinates use [right, -up, forward]. Convert around the rotations.
    camera_to_image_axes = np.asarray([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)
    image_to_camera_axes = np.linalg.inv(camera_to_image_axes)
    full_h = (
        target_k
        @ camera_to_image_axes
        @ ray_rotation
        @ image_to_camera_axes
        @ np.linalg.inv(source_k)
    )
    crop_translation = np.asarray([
        [1.0, 0.0, float(row["crop_x1_px"])],
        [0.0, 1.0, float(row["crop_y1_px"])],
        [0.0, 0.0, 1.0],
    ])
    return full_h @ crop_translation


def align_uniform(sprite, initial_h, target_bbox, width, height):
    ys, xs = np.where(sprite[:, :, 3] > 10)
    if not len(xs):
        return None
    points = np.column_stack((xs, ys, np.ones(len(xs), dtype=np.float64)))
    transformed = (initial_h @ points.T).T
    valid = np.abs(transformed[:, 2]) > 1e-8
    transformed = transformed[valid, :2] / transformed[valid, 2:3]
    if not len(transformed):
        return None
    current_x1, current_y1 = np.min(transformed, axis=0)
    current_x2, current_y2 = np.max(transformed, axis=0)
    current_height = current_y2 - current_y1 + 1.0
    if current_height <= 1e-6:
        return None
    scale = float(target_bbox["height_px"]) / current_height
    current_cx = (current_x1 + current_x2) / 2.0
    target_cx = float(target_bbox["center_x_px"])
    target_bottom = float(target_bbox["bottom_y_px"])
    affine = np.asarray([
        [scale, 0.0, target_cx - scale * current_cx],
        [0.0, scale, target_bottom - scale * current_y2],
        [0.0, 0.0, 1.0],
    ])
    final_h = affine @ initial_h
    return cv2.warpPerspective(
        sprite, final_h, (int(width), int(height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def candidate_rows(frame, far_rows, close_rows_by_side, radius):
    mode = frame.get("he_sprite_mode")
    if mode == "view_matrix":
        center = int(round(float(frame["he_selected_angle_deg"]))) % 360
        distance = float(frame["he_selected_distance_m"])
        elevation = float(frame["he_selected_elevation_deg"])
        allowed = {(center + delta) % 360 for delta in range(-radius, radius + 1)}
        rows = [
            row for row in far_rows
            if int(float(row["angle_deg"])) % 360 in allowed
            and abs(float(row["distance_m"]) - distance) < 1e-6
            and abs(float(row["elevation_deg"]) - elevation) < 1e-6
        ]
        return "far", center, rows

    close = frame.get("he_cartesian_close") or {}
    sources = close.get("source_weights") or []
    if not sources:
        return None, None, []
    dominant = max(sources, key=lambda row: float(row["weight"]))
    side = str(close["side"])
    center = int(round(float(dominant["relative_yaw_deg"]))) % 360
    forward = float(dominant["forward_m"])
    right = float(dominant["right_m"])
    allowed = {(center + delta) % 360 for delta in range(-radius, radius + 1)}
    rows = [
        row for row in close_rows_by_side[side]
        if int(round(float(row["close_relative_yaw_deg"]))) % 360 in allowed
        and abs(float(row["close_forward_m"]) - forward) < 1e-6
        and abs(float(row["close_right_m"]) - right) < 1e-6
    ]
    return side, center, rows


def source_transforms(kind, row):
    if kind == "far":
        actor_tf = carla.Transform()
        camera_tf = carla.Transform(rotation=carla.Rotation(
            pitch=-float(row["elevation_deg"]),
            yaw=float(row["angle_deg"]) - 180.0,
        ))
        value = int(float(row["angle_deg"])) % 360
        return actor_tf, camera_tf, value

    actor_tf = carla.Transform(rotation=carla.Rotation(
        yaw=float(row["close_relative_yaw_deg"])
    ))
    camera_tf = carla.Transform()
    value = int(round(float(row["close_relative_yaw_deg"]))) % 360
    return actor_tf, camera_tf, value


def neutral_tile(rgba, label, width=400, height=330):
    canvas = np.full((height, width, 3), 214, dtype=np.uint8)
    if rgba is not None:
        rgb = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        canvas[:rgba.shape[0], :rgba.shape[1]] = np.clip(
            rgb * alpha + canvas[:rgba.shape[0], :rgba.shape[1]] * (1.0 - alpha),
            0, 255,
        ).astype(np.uint8)
    cv2.rectangle(canvas, (0, 300), (width - 1, height - 1), (245, 245, 245), -1)
    cv2.putText(
        canvas, label, (8, 321), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (20, 20, 20), 1, cv2.LINE_AA,
    )
    return canvas


def video_frames(path, wanted):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    result = {}
    for index in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            result[index] = frame
    capture.release()
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_jsonl(args.he_dir / "frames.jsonl")
    far_rows = load_csv(args.far_bank / "view_matrix.csv")
    close_rows = {
        "left": load_csv(args.close_left_bank / "view_matrix.csv"),
        "right": load_csv(args.close_right_bank / "view_matrix.csv"),
    }
    bank_dirs = {
        "far": args.far_bank,
        "left": args.close_left_bank,
        "right": args.close_right_bank,
    }
    carla_video = video_frames(
        args.carla_dir / "carla_reference.mp4", set(args.frames)
    )
    he_video = video_frames(args.he_dir / "he_replay.mp4", set(args.frames))
    setup = json.loads((args.he_dir / "setup.json").read_text(encoding="utf-8"))
    camera = setup["camera"]
    width, height = int(camera["width"]), int(camera["height"])
    output_rows = []

    for frame_index in args.frames:
        frame = frames[frame_index]
        target_mask = cv2.imread(
            str(args.carla_dir / frame["mask_path"]), cv2.IMREAD_GRAYSCALE
        ) > 0
        target_bbox = frame.get("mask_bbox")
        # HE metadata stores the HE mask box, so measure the CARLA box directly.
        target_bbox = alpha_bbox(target_mask.astype(np.uint8) * 255)
        if target_bbox is None:
            continue
        target_bbox = {
            "height_px": target_bbox["height"],
            "center_x_px": target_bbox["cx"],
            "bottom_y_px": target_bbox["bottom"],
        }
        kind, center_value, rows = candidate_rows(
            frame, far_rows, close_rows, args.angle_radius_deg
        )
        target_actor = transform_from_dict(frame["actor_transform"])
        target_camera = transform_from_dict(frame["camera_transform"])
        ranked = []

        for row in rows:
            path = source_path(bank_dirs[kind], row)
            sprite = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if sprite is None or sprite.shape[2] != 4:
                continue
            source_actor, source_camera, value = source_transforms(kind, row)
            identity = np.eye(3, dtype=np.float64)
            plain = align_uniform(sprite, identity, target_bbox, width, height)
            rotation = rotation_homography(
                row, source_actor, source_camera, target_actor, target_camera,
                width, height, float(camera["fov_deg"]),
            )
            corrected = align_uniform(
                sprite, rotation, target_bbox, width, height
            )
            if plain is None or corrected is None:
                continue
            plain_iou = mask_iou(plain[:, :, 3] > 10, target_mask)
            corrected_iou = mask_iou(corrected[:, :, 3] > 10, target_mask)
            ranked.append({
                "value": value,
                "row": row,
                "plain": plain,
                "corrected": corrected,
                "plain_iou": plain_iou,
                "corrected_iou": corrected_iou,
                "offset_deg": ((value - center_value + 180) % 360) - 180,
            })

        if not ranked:
            continue
        best_plain = max(ranked, key=lambda item: item["plain_iou"])
        best_corrected = max(ranked, key=lambda item: item["corrected_iou"])
        top = sorted(
            ranked, key=lambda item: item["corrected_iou"], reverse=True
        )[:args.top]
        output_rows.append({
            "frame": frame_index,
            "mode": frame.get("he_sprite_mode"),
            "runtime_geometric_angle_deg": frame.get(
                "carla_geometric_view_angle_deg"
            ),
            "current_selected_value_deg": center_value,
            "best_plain_value_deg": best_plain["value"],
            "best_plain_offset_deg": best_plain["offset_deg"],
            "best_plain_mask_iou_normalized": best_plain["plain_iou"],
            "best_rotation_value_deg": best_corrected["value"],
            "best_rotation_offset_deg": best_corrected["offset_deg"],
            "best_rotation_mask_iou_normalized": best_corrected["corrected_iou"],
        })

        reference = carla_video[frame_index].copy()
        current = he_video[frame_index].copy()
        cv2.putText(reference, "CARLA physical", (8, 22), 0, 0.55, (255, 255, 255), 1)
        cv2.putText(current, "Current HE", (8, 22), 0, 0.55, (255, 255, 255), 1)
        tiles = [
            neutral_tile(
                cv2.cvtColor(reference, cv2.COLOR_BGR2BGRA), "CARLA reference"
            ),
            neutral_tile(
                cv2.cvtColor(current, cv2.COLOR_BGR2BGRA), "Current Candidate 12"
            ),
            neutral_tile(
                best_plain["plain"],
                f"plain best={best_plain['value']} IoU={best_plain['plain_iou']:.3f}",
            ),
        ]
        tiles.extend(
            neutral_tile(
                item["corrected"],
                f"rot value={item['value']} d={item['offset_deg']:+d} IoU={item['corrected_iou']:.3f}",
            )
            for item in top
        )
        columns = 4
        rows_count = math.ceil(len(tiles) / columns)
        sheet = np.full((rows_count * 330, columns * 400, 3), 230, dtype=np.uint8)
        for index, tile in enumerate(tiles):
            y = (index // columns) * 330
            x = (index % columns) * 400
            sheet[y:y + 330, x:x + 400] = tile
        cv2.imwrite(str(args.output_dir / f"frame_{frame_index:06d}.png"), sheet)

    csv_path = args.output_dir / "oracle.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"frames={len(output_rows)}")
    print(f"oracle={csv_path}")


if __name__ == "__main__":
    main()
