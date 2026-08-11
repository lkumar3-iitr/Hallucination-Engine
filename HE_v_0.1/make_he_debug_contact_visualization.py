#!/usr/bin/env python3
"""
make_he_debug_contact_visualization.py

Visual review tool for HE temporal compositor outputs.

Reads:
  - metadata.json from run_he_temporal_compositor_v1.py
  - generated HE video or debug video

Creates:
  - review video with overlays:
      projected box
      pasted sprite box
      bottom contact point
      center trajectory
      bottom/contact trajectory
      selected sprite angle
      relative angle
      z distance

Example:
  python make_he_debug_contact_visualization.py ^
    --metadata he_outputs\\cut_in_left_to_center_001\\metadata.json ^
    --video he_outputs\\cut_in_left_to_center_001\\cut_in_left_to_center_001.mp4 ^
    --output he_outputs\\cut_in_left_to_center_001\\review_contact_cut_in.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--video", required=True, help="Path to HE output video")
    parser.add_argument("--output", required=True, help="Output review/debug video path")

    parser.add_argument("--trail-length", type=int, default=90)
    parser.add_argument("--draw-frame-index", action="store_true")
    parser.add_argument("--draw-all-adversaries", action="store_true")

    return parser.parse_args()


# ============================================================
# Loading
# ============================================================

def load_metadata(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_frame_lookup(metadata):
    lookup = {}

    for frame_meta in metadata.get("frames", []):
        idx = int(frame_meta["frame_idx"])
        lookup[idx] = frame_meta

    return lookup


# ============================================================
# Drawing helpers
# ============================================================

def safe_int(x):
    return int(round(float(x)))


def draw_label(img, text, org, font_scale=0.5, thickness=1):
    x, y = org

    cv2.putText(
        img,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_projected_box(img, box):
    if not box or not box.get("visible", False):
        return

    x1 = safe_int(box["x1"])
    y1 = safe_int(box["y1"])
    x2 = safe_int(box["x2"])
    y2 = safe_int(box["y2"])

    # Blue projected physics/camera box.
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


def draw_paste_box(img, paste):
    if not paste:
        return

    x1 = safe_int(paste["x1"])
    y1 = safe_int(paste["y1"])
    x2 = safe_int(paste["x2"])
    y2 = safe_int(paste["y2"])

    # Green actual sprite paste box.
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)


def draw_contact_and_center(img, box):
    if not box or not box.get("visible", False):
        return None, None

    cx = safe_int(box["cx"])
    bottom_y = safe_int(box["bottom_y"])

    x1 = safe_int(box["x1"])
    y1 = safe_int(box["y1"])
    x2 = safe_int(box["x2"])
    y2 = safe_int(box["y2"])

    center_y = safe_int((y1 + y2) / 2.0)

    # Bottom contact point.
    cv2.circle(img, (cx, bottom_y), 5, (0, 0, 255), -1)

    # Box center point.
    cv2.circle(img, (cx, center_y), 4, (255, 255, 0), -1)

    # Vertical line from center to contact.
    cv2.line(img, (cx, center_y), (cx, bottom_y), (255, 255, 0), 1)

    return (cx, bottom_y), (cx, center_y)


def draw_trail(img, points, color, thickness=2):
    if len(points) < 2:
        return

    pts = points[-1 * len(points):]

    for i in range(1, len(pts)):
        p1 = pts[i - 1]
        p2 = pts[i]

        if p1 is None or p2 is None:
            continue

        cv2.line(img, p1, p2, color, thickness)


def draw_info_panel(img, frame_idx, adv_meta, draw_frame_index=True):
    h, w = img.shape[:2]

    panel_x = 10
    panel_y = 24
    line_h = 22

    if draw_frame_index:
        draw_label(img, f"frame: {frame_idx}", (panel_x, panel_y), 0.55, 1)
        panel_y += line_h

    if adv_meta is None:
        draw_label(img, "no active adversary", (panel_x, panel_y), 0.55, 1)
        return

    state = adv_meta.get("state", {})
    box = adv_meta.get("box", {})
    sprite = adv_meta.get("sprite", {})
    paste = adv_meta.get("paste", {})

    adv_id = adv_meta.get("id", "adv")
    rendered = adv_meta.get("rendered", False)

    z_m = state.get("z_m", None)
    x_m = state.get("x_m", None)
    yaw = state.get("yaw_deg", None)

    selected_angle = sprite.get("selected_angle", None)
    rel_angle = sprite.get("relative_angle_deg", adv_meta.get("relative_angle_deg", None))

    lines = [
        f"id: {adv_id} rendered={rendered}",
        f"x={x_m:.2f} m  z={z_m:.2f} m  yaw={yaw:.1f} deg" if z_m is not None else "",
        f"rel_angle={rel_angle:.1f}  sprite={selected_angle:03d}" if selected_angle is not None else "",
    ]

    if box and box.get("visible", False):
        lines.append(
            f"box h={box.get('box_height', 0):.1f} w={box.get('box_width', 0):.1f} "
            f"bottom_y={box.get('bottom_y', 0):.1f}"
        )

    if paste:
        lines.append(
            f"paste: x1={paste.get('x1')} y1={paste.get('y1')} "
            f"w={paste.get('sprite_width')} h={paste.get('sprite_height')}"
        )

    for line in lines:
        if line:
            draw_label(img, line, (panel_x, panel_y), 0.52, 1)
            panel_y += line_h

    # Legend.
    legend_y = h - 78
    draw_label(img, "blue box: projected vehicle box", (10, legend_y), 0.48, 1)
    draw_label(img, "green box: pasted sprite box", (10, legend_y + 20), 0.48, 1)
    draw_label(img, "red dot/trail: bottom contact", (10, legend_y + 40), 0.48, 1)
    draw_label(img, "cyan dot/trail: box center", (10, legend_y + 60), 0.48, 1)


def get_first_rendered_adversary(frame_meta):
    adversaries = frame_meta.get("adversaries", [])

    for adv in adversaries:
        if adv.get("rendered", False):
            return adv

    if len(adversaries) > 0:
        return adversaries[0]

    return None


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    metadata = load_metadata(args.metadata)
    frame_lookup = build_frame_lookup(metadata)

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = float(metadata.get("output_fps", 30.0))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    contact_trail = []
    center_trail = []

    frame_idx = 0

    print("[Review] Reading video:", video_path)
    print("[Review] Reading metadata:", args.metadata)
    print("[Review] Writing output:", output_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_meta = frame_lookup.get(frame_idx, None)

        if frame_meta is None:
            writer.write(frame)
            frame_idx += 1
            continue

        adversaries = frame_meta.get("adversaries", [])

        if args.draw_all_adversaries:
            selected_adversaries = adversaries
        else:
            first = get_first_rendered_adversary(frame_meta)
            selected_adversaries = [first] if first is not None else []

        current_contact = None
        current_center = None

        for adv_meta in selected_adversaries:
            if adv_meta is None:
                continue

            box = adv_meta.get("box", {})
            paste = adv_meta.get("paste", {})

            draw_projected_box(frame, box)
            draw_paste_box(frame, paste)

            contact, center = draw_contact_and_center(frame, box)

            if contact is not None:
                current_contact = contact

            if center is not None:
                current_center = center

        contact_trail.append(current_contact)
        center_trail.append(current_center)

        if len(contact_trail) > args.trail_length:
            contact_trail = contact_trail[-args.trail_length:]

        if len(center_trail) > args.trail_length:
            center_trail = center_trail[-args.trail_length:]

        draw_trail(frame, contact_trail, (0, 0, 255), thickness=2)
        draw_trail(frame, center_trail, (255, 255, 0), thickness=1)

        panel_adv = selected_adversaries[0] if selected_adversaries else None

        draw_info_panel(
            frame,
            frame_idx=frame_idx,
            adv_meta=panel_adv,
            draw_frame_index=args.draw_frame_index
        )

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"[Review] frame {frame_idx}")

        frame_idx += 1

    cap.release()
    writer.release()

    print("\n[Review] Done.")
    print("[Review] Output:", output_path)
    print("[Review] Frames:", frame_idx)


if __name__ == "__main__":
    main()