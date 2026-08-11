#!/usr/bin/env python3
"""
temporal_sprite_insert_fixed_angle.py

First temporal HE test:
  - fixed RGBA sprite angle
  - smooth scale change over frames
  - alpha compositing
  - output video + masks + metadata

This does NOT use:
  - refiner
  - HEBoxPred
  - angle switching
  - 3D projection

Example:
  python temporal_sprite_insert_fixed_angle.py ^
    --input-video "road video.mp4" ^
    --sprite-path assets\\sprite_bank_rgba\\vehicle_blue_sedan\\rgba\\angle_180_rgba.png ^
    --output-dir he_outputs\\oncoming_fixed_angle_test ^
    --start-frame 0 ^
    --end-frame 150 ^
    --start-height 70 ^
    --end-height 260 ^
    --center-x-ratio 0.50 ^
    --bottom-y-ratio 0.92
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-video", required=True)
    parser.add_argument("--sprite-path", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)

    parser.add_argument("--start-height", type=int, default=70)
    parser.add_argument("--end-height", type=int, default=260)

    parser.add_argument("--center-x-ratio", type=float, default=0.50)
    parser.add_argument("--bottom-y-ratio", type=float, default=0.92)

    parser.add_argument("--x-drift-ratio", type=float, default=0.0,
                        help="Optional horizontal drift from start to end as ratio of frame width.")

    parser.add_argument("--output-fps", type=float, default=0.0,
                        help="0 means use input video FPS.")

    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-masks", action="store_true")

    return parser.parse_args()


def load_rgba(path):
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if rgba is None:
        raise RuntimeError(f"Could not read sprite: {path}")

    if rgba.shape[2] != 4:
        raise RuntimeError("Sprite must be RGBA/BGRA PNG with alpha channel.")

    # cv2 loads as BGRA. Convert to RGBA.
    rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)

    return rgba


def resize_sprite_keep_aspect(sprite_rgba, target_h):
    h, w = sprite_rgba.shape[:2]

    if h <= 0 or w <= 0:
        raise RuntimeError("Invalid sprite size.")

    scale = target_h / float(h)
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(target_h)))

    resized = cv2.resize(
        sprite_rgba,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )

    return resized


def alpha_composite(frame_rgb, sprite_rgba, x1, y1):
    """
    frame_rgb: HxWx3 RGB uint8
    sprite_rgba: hxwx4 RGBA uint8
    x1,y1: top-left paste position

    Returns:
      composited RGB frame
      full-frame mask uint8
    """
    frame_h, frame_w = frame_rgb.shape[:2]
    spr_h, spr_w = sprite_rgba.shape[:2]

    x2 = x1 + spr_w
    y2 = y1 + spr_h

    # Clip to frame.
    paste_x1 = max(0, x1)
    paste_y1 = max(0, y1)
    paste_x2 = min(frame_w, x2)
    paste_y2 = min(frame_h, y2)

    if paste_x2 <= paste_x1 or paste_y2 <= paste_y1:
        mask_full = np.zeros((frame_h, frame_w), dtype=np.uint8)
        return frame_rgb, mask_full

    sprite_x1 = paste_x1 - x1
    sprite_y1 = paste_y1 - y1
    sprite_x2 = sprite_x1 + (paste_x2 - paste_x1)
    sprite_y2 = sprite_y1 + (paste_y2 - paste_y1)

    sprite_crop = sprite_rgba[sprite_y1:sprite_y2, sprite_x1:sprite_x2]

    sprite_rgb = sprite_crop[:, :, :3].astype(np.float32)
    alpha = sprite_crop[:, :, 3:4].astype(np.float32) / 255.0

    roi = frame_rgb[paste_y1:paste_y2, paste_x1:paste_x2].astype(np.float32)

    blended = sprite_rgb * alpha + roi * (1.0 - alpha)
    frame_rgb[paste_y1:paste_y2, paste_x1:paste_x2] = np.clip(blended, 0, 255).astype(np.uint8)

    mask_full = np.zeros((frame_h, frame_w), dtype=np.uint8)
    mask_full[paste_y1:paste_y2, paste_x1:paste_x2] = sprite_crop[:, :, 3]

    return frame_rgb, mask_full


def smoothstep(t):
    """
    Smooth interpolation from 0 to 1.
    """
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def main():
    args = parse_args()

    input_video = Path(args.input_video)
    sprite_path = Path(args.sprite_path)
    output_dir = Path(args.output_dir)

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"

    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    sprite_rgba = load_rgba(sprite_path)

    cap = cv2.VideoCapture(str(input_video))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = args.output_fps if args.output_fps > 0 else input_fps

    if args.end_frame < 0:
        end_frame = total_frames - 1
    else:
        end_frame = min(args.end_frame, total_frames - 1)

    start_frame = max(0, args.start_frame)

    if end_frame <= start_frame:
        raise RuntimeError("end-frame must be greater than start-frame.")

    output_video_path = output_dir / "temporal_he_fixed_angle.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (frame_w, frame_h)
    )

    metadata = {
        "input_video": str(input_video),
        "sprite_path": str(sprite_path),
        "output_video": str(output_video_path),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "input_fps": input_fps,
        "output_fps": fps,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_height": args.start_height,
        "end_height": args.end_height,
        "center_x_ratio": args.center_x_ratio,
        "bottom_y_ratio": args.bottom_y_ratio,
        "x_drift_ratio": args.x_drift_ratio,
        "frames": []
    }

    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if start_frame <= frame_idx <= end_frame:
            denom = max(1, end_frame - start_frame)
            t = (frame_idx - start_frame) / float(denom)
            s = smoothstep(t)

            target_h = int(round(args.start_height * (1.0 - s) + args.end_height * s))

            sprite_resized = resize_sprite_keep_aspect(sprite_rgba, target_h)

            spr_h, spr_w = sprite_resized.shape[:2]

            center_x = int(round(frame_w * (args.center_x_ratio + args.x_drift_ratio * s)))
            bottom_y = int(round(frame_h * args.bottom_y_ratio))

            x1 = int(round(center_x - spr_w / 2.0))
            y1 = int(round(bottom_y - spr_h))

            frame_rgb, mask = alpha_composite(frame_rgb, sprite_resized, x1, y1)

            metadata["frames"].append({
                "frame_idx": frame_idx,
                "active": True,
                "t": float(t),
                "smooth_t": float(s),
                "target_height": int(target_h),
                "sprite_width": int(spr_w),
                "sprite_height": int(spr_h),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x1 + spr_w),
                "y2": int(y1 + spr_h),
                "center_x": int(center_x),
                "bottom_y": int(bottom_y)
            })

            if args.save_masks:
                cv2.imwrite(
                    str(masks_dir / f"frame_{frame_idx:06d}_mask.png"),
                    mask
                )

        else:
            metadata["frames"].append({
                "frame_idx": frame_idx,
                "active": False
            })

        out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)

        if args.save_frames:
            cv2.imwrite(
                str(frames_dir / f"frame_{frame_idx:06d}.png"),
                out_bgr
            )

        frame_idx += 1

    cap.release()
    writer.release()

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[TemporalHE] Done.")
    print("[TemporalHE] Output:", output_video_path)
    print("[TemporalHE] Metadata:", output_dir / "metadata.json")


if __name__ == "__main__":
    main()