#!/usr/bin/env python3
"""
render_forward_passby_oracle_v1.py

Offline preview renderer for the "4320 bank is enough" hypothesis.

Goal
----
Generate a qualitative pass-by video for the forward camera using:
  - the original 4320 Tesla bank,
  - clipping at image boundaries,
  - full-bank brute-force sprite selection per frame,
  - a simple full-object 2D placement box derived from 3D actor geometry.

This is a PREVIEW / thought-experiment renderer, not yet the final HE runtime.
It is intentionally offline and consumes an existing CARLA reference recording.

Expected input directory
------------------------
A CARLA reference directory produced by record_scenario_carla_he_pair_v3.py:

    setup.json
    frames.jsonl
    masks/frame_XXXXXX.png
    carla_reference.mp4

Outputs
-------
    oracle_passby_preview.mp4         composite on inpainted background
    oracle_passby_side_by_side.mp4    CARLA vs oracle
    oracle_passby_rows.csv            per-frame selected source and scores
    frames_preview/frame_XXXXXX.png   per-frame preview frames
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Render forward-camera pass-by preview using full 4320 search and clipping."
    )
    p.add_argument("--carla-dir", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--csv-name", default="view_matrix.csv")
    p.add_argument("--alpha-threshold", type=int, default=10)

    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=-1)
    p.add_argument("--frame-step", type=int, default=1)

    p.add_argument("--actor-length-m", type=float, default=4.792)
    p.add_argument("--actor-width-m", type=float, default=2.163)
    p.add_argument("--actor-height-m", type=float, default=1.488)

    p.add_argument("--inpaint-radius", type=float, default=4.0)
    p.add_argument("--mask-dilate-px", type=int, default=9)

    p.add_argument("--save-frame-step", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--temporal-angle-window-deg", type=int, default=999,
                   help=("Optional continuity window around previous winning angle. "
                         "999 means full search every frame."))
    return p


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def first_float(row: dict, keys: Sequence[str], default=None):
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        txt = str(value).strip()
        if not txt:
            continue
        try:
            return float(txt)
        except Exception:
            pass
    return default


def source_path(bank_dir: Path, row: dict) -> Path:
    rel = str(row.get("rgba_relpath", "")).strip()
    if rel:
        return (bank_dir / rel).resolve()
    raw = str(row.get("rgba_path", "")).strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return p.resolve()
        q = bank_dir / raw
        if q.is_file():
            return q.resolve()
        q = bank_dir / "rgba" / p.name
        if q.is_file():
            return q.resolve()
    raise FileNotFoundError("Could not resolve source sprite path from CSV row.")


def circular_delta_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def alpha_bbox(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def rotation_matrix_deg(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    # Local basis: x forward, y right, z up.
    p = math.radians(float(pitch_deg))
    y = math.radians(float(yaw_deg))
    r = math.radians(float(roll_deg))

    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]], dtype=np.float64)
    return rz @ ry @ rx


def intrinsic(width: int, height: int, fov_deg: float) -> Tuple[float, float, float, float]:
    focal = float(width) / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fx = fy = focal
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return fx, fy, cx, cy


def dict_pose_to_parts(d: dict):
    loc = np.array([float(d["x"]), float(d["y"]), float(d["z"])], dtype=np.float64)
    rot = rotation_matrix_deg(
        float(d.get("pitch", 0.0)),
        float(d.get("yaw", 0.0)),
        float(d.get("roll", 0.0)),
    )
    return loc, rot


def project_actor_bbox(frame: dict, width: int, height: int, fov_deg: float,
                       length_m: float, width_m: float, height_m: float):
    actor_tf = frame.get("actor_transform")
    camera_tf = frame.get("camera_transform")
    if actor_tf is None or camera_tf is None:
        return None

    actor_loc, actor_rot = dict_pose_to_parts(actor_tf)
    camera_loc, camera_rot = dict_pose_to_parts(camera_tf)
    fx, fy, cx, cy = intrinsic(width, height, fov_deg)

    hx = float(length_m) / 2.0
    hy = float(width_m) / 2.0
    hz = float(height_m) / 2.0

    local_corners = np.array([
        [+hx, +hy, +hz], [+hx, +hy, -hz], [+hx, -hy, +hz], [+hx, -hy, -hz],
        [-hx, +hy, +hz], [-hx, +hy, -hz], [-hx, -hy, +hz], [-hx, -hy, -hz],
    ], dtype=np.float64)

    world = (actor_rot @ local_corners.T).T + actor_loc[None, :]
    pc = (camera_rot.T @ (world - camera_loc[None, :]).T).T

    forward = pc[:, 0]
    valid = forward > 1e-4
    if not np.any(valid):
        return None

    right = pc[valid, 1]
    up = pc[valid, 2]
    forward = pc[valid, 0]

    u = fx * (right / forward) + cx
    v = cy - fy * (up / forward)

    if len(u) == 0:
        return None

    x1 = int(math.floor(np.min(u)))
    x2 = int(math.ceil(np.max(u)))
    y1 = int(math.floor(np.min(v)))
    y2 = int(math.ceil(np.max(v)))

    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "width": int(x2 - x1 + 1),
        "height": int(y2 - y1 + 1),
    }


# ---------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------

@dataclass
class Candidate:
    angle_deg: float
    distance_m: float
    elevation_deg: float
    rgba_crop: np.ndarray
    mask_crop: np.ndarray
    path: Path


def load_bank(bank_dir: Path, csv_name: str, alpha_threshold: int) -> List[Candidate]:
    rows = read_csv(bank_dir / csv_name)
    out = []
    for row in rows:
        angle = first_float(row, ("angle_deg",))
        distance = first_float(row, ("distance_m",))
        elevation = first_float(row, ("elevation_deg",))
        path = source_path(bank_dir, row)
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
            continue
        alpha = rgba[:, :, 3] > int(alpha_threshold)
        if not np.any(alpha):
            continue
        ys, xs = np.where(alpha)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        rgba_crop = rgba[y1:y2+1, x1:x2+1].copy()
        mask_crop = (rgba_crop[:, :, 3] > int(alpha_threshold)).astype(np.uint8)
        out.append(Candidate(
            angle_deg=float(angle) % 360.0,
            distance_m=float(distance),
            elevation_deg=float(elevation),
            rgba_crop=rgba_crop,
            mask_crop=mask_crop,
            path=path,
        ))
    return out


# ---------------------------------------------------------------------
# Rendering / scoring
# ---------------------------------------------------------------------

def render_mask_full(candidate: Candidate, box: dict, width: int, height: int):
    w = max(1, int(box["width"]))
    h = max(1, int(box["height"]))
    resized = cv2.resize(candidate.mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((height, width), dtype=np.uint8)

    x1 = int(box["x1"])
    y1 = int(box["y1"])
    x2 = x1 + w
    y2 = y1 + h

    dx1 = max(0, -x1)
    dy1 = max(0, -y1)
    dx2 = max(0, x2 - width)
    dy2 = max(0, y2 - height)

    sx1 = dx1
    sy1 = dy1
    sx2 = w - dx2
    sy2 = h - dy2

    if sx1 >= sx2 or sy1 >= sy2:
        return canvas

    tx1 = max(0, x1)
    ty1 = max(0, y1)
    tx2 = tx1 + (sx2 - sx1)
    ty2 = ty1 + (sy2 - sy1)

    canvas[ty1:ty2, tx1:tx2] = (resized[sy1:sy2, sx1:sx2] > 0).astype(np.uint8)
    return canvas


def render_rgba_full(candidate: Candidate, box: dict, width: int, height: int):
    w = max(1, int(box["width"]))
    h = max(1, int(box["height"]))
    resized = cv2.resize(candidate.rgba_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    x1 = int(box["x1"])
    y1 = int(box["y1"])
    x2 = x1 + w
    y2 = y1 + h

    dx1 = max(0, -x1)
    dy1 = max(0, -y1)
    dx2 = max(0, x2 - width)
    dy2 = max(0, y2 - height)

    sx1 = dx1
    sy1 = dy1
    sx2 = w - dx2
    sy2 = h - dy2

    if sx1 >= sx2 or sy1 >= sy2:
        return canvas

    tx1 = max(0, x1)
    ty1 = max(0, y1)
    tx2 = tx1 + (sx2 - sx1)
    ty2 = ty1 + (sy2 - sy1)

    canvas[ty1:ty2, tx1:tx2] = resized[sy1:sy2, sx1:sx2]
    return canvas


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return float(inter / union) if union > 0 else 1.0


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.count_nonzero(a & b))
    denom = int(np.count_nonzero(a) + np.count_nonzero(b))
    return float(2 * inter / denom) if denom > 0 else 1.0


def overlay_rgba(background_bgr: np.ndarray, rgba_full: np.ndarray):
    out = background_bgr.copy()
    rgb = rgba_full[:, :, :3].astype(np.float32)
    alpha = (rgba_full[:, :, 3:4].astype(np.float32) / 255.0)
    out = np.clip(rgb * alpha + out.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
    return out


def inpaint_background(rgb_bgr: np.ndarray, mask_u8: np.ndarray, dilate_px: int, radius: float):
    k = max(1, int(dilate_px))
    kernel = np.ones((k, k), dtype=np.uint8)
    inpaint_mask = cv2.dilate((mask_u8 > 0).astype(np.uint8) * 255, kernel, iterations=1)
    return cv2.inpaint(rgb_bgr, inpaint_mask, float(radius), cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def draw_text_block(img: np.ndarray, lines: Sequence[str], x: int, y: int):
    yy = y
    for line in lines:
        cv2.putText(img, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        yy += 20


def stack_preview(carla_bgr: np.ndarray, preview_bgr: np.ndarray, mask_u8: np.ndarray, lines: Sequence[str]):
    mask_rgb = np.repeat((mask_u8 > 0)[:, :, None].astype(np.uint8) * 255, 3, axis=2)
    h, w = carla_bgr.shape[:2]
    header_h = 74
    canvas = np.zeros((header_h + 2 * h, 2 * w, 3), dtype=np.uint8)
    canvas[header_h:header_h+h, :w] = carla_bgr
    canvas[header_h:header_h+h, w:2*w] = preview_bgr
    canvas[header_h+h:header_h+2*h, :w] = mask_rgb
    diff = cv2.absdiff(carla_bgr, preview_bgr)
    canvas[header_h+h:header_h+2*h, w:2*w] = diff
    cv2.putText(canvas, "CARLA RGB (physical Tesla)", (10, header_h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Oracle preview (inpainted background + selected 4320 sprite)", (w + 10, header_h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CARLA target mask", (10, header_h + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Absolute RGB difference", (w + 10, header_h + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
    draw_text_block(canvas, lines, 10, 22)
    return canvas


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    carla_dir = args.carla_dir.resolve()
    bank_dir = args.bank.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "frames_preview"
    preview_dir.mkdir(exist_ok=True)

    setup = load_json(carla_dir / "setup.json")
    frames = load_jsonl(carla_dir / "frames.jsonl")

    camera_cfg = setup.get("camera") or {}
    width = int(camera_cfg.get("width", camera_cfg.get("image_width", 1280)))
    height = int(camera_cfg.get("height", camera_cfg.get("image_height", 720)))
    fov_deg = float(camera_cfg.get("fov_deg", 90.0))
    fps = float(setup.get("fps", 20.0))

    frame_end = len(frames) - 1 if int(args.frame_end) < 0 else min(int(args.frame_end), len(frames) - 1)
    wanted = list(range(max(0, int(args.frame_start)), frame_end + 1, max(1, int(args.frame_step))))

    print("=" * 92)
    print("FORWARD-CAMERA 4320 ORACLE PASS-BY PREVIEW")
    print("=" * 92)
    print("[CARLA dir] ", carla_dir)
    print("[bank]      ", bank_dir)
    print("[output]    ", out_dir)
    print("[camera]    ", f"{width}x{height}, FOV={fov_deg:.3f}")
    print("[frames]    ", f"{wanted[0]}..{wanted[-1]} step {max(1, int(args.frame_step))}")
    print("=" * 92)

    bank = load_bank(bank_dir, args.csv_name, int(args.alpha_threshold))
    print(f"[bank loaded] {len(bank)} candidates")

    cap = cv2.VideoCapture(str(carla_dir / "carla_reference.mp4"))
    if not cap.isOpened():
        raise RuntimeError("Could not open carla_reference.mp4")

    preview_video = cv2.VideoWriter(
        str(out_dir / "oracle_passby_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    side_video = cv2.VideoWriter(
        str(out_dir / "oracle_passby_side_by_side.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (2 * width, 74 + 2 * height),
    )
    if not preview_video.isOpened() or not side_video.isOpened():
        raise RuntimeError("Could not open one of the output videos.")

    rows = []
    previous_best_angle = None

    start_all = time.time()
    for scenario_frame in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(scenario_frame))
        ok, carla_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read CARLA RGB frame {scenario_frame}")

        frame = frames[scenario_frame]
        mask_path = carla_dir / str(frame["mask_path"])
        target_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if target_gray is None:
            raise RuntimeError(f"Could not read mask {mask_path}")
        target_mask = (target_gray > 0).astype(np.uint8)

        visible = bool(np.any(target_mask))
        projected_box = None
        best = None
        top = []

        if visible:
            projected_box = project_actor_bbox(
                frame=frame,
                width=width,
                height=height,
                fov_deg=fov_deg,
                length_m=float(args.actor_length_m),
                width_m=float(args.actor_width_m),
                height_m=float(args.actor_height_m),
            )

        if visible and projected_box is not None and projected_box["width"] > 1 and projected_box["height"] > 1:
            candidate_iter = bank
            if previous_best_angle is not None and int(args.temporal_angle_window_deg) < 360:
                radius = int(args.temporal_angle_window_deg)
                candidate_iter = [
                    c for c in bank
                    if abs(circular_delta_deg(c.angle_deg, previous_best_angle)) <= radius
                ]

            scored = []
            for cand in candidate_iter:
                cand_mask = render_mask_full(cand, projected_box, width, height)
                score_iou = iou(cand_mask, target_mask)
                score_dice = dice(cand_mask, target_mask)
                scored.append((score_iou, score_dice, cand))

            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            top = scored[:max(1, int(args.top_k))]
            score_iou, score_dice, best_cand = top[0]
            best = {
                "candidate": best_cand,
                "iou": float(score_iou),
                "dice": float(score_dice),
            }
            previous_best_angle = float(best_cand.angle_deg)

        # Build preview frame.
        bg = inpaint_background(
            carla_bgr,
            target_mask,
            dilate_px=int(args.mask_dilate_px),
            radius=float(args.inpaint_radius),
        )

        if best is not None:
            rgba_full = render_rgba_full(best["candidate"], projected_box, width, height)
            preview_bgr = overlay_rgba(bg, rgba_full)
            source_angle = best["candidate"].angle_deg
            source_dist = best["candidate"].distance_m
            source_elev = best["candidate"].elevation_deg
            score_iou = best["iou"]
            score_dice = best["dice"]
        else:
            preview_bgr = bg
            source_angle = source_dist = source_elev = None
            score_iou = score_dice = None

        geom_angle = frame.get("carla_geometric_view_angle_deg")
        rel = frame.get("camera_relative") or {}
        lines = [
            f"frame={scenario_frame} visible={int(visible)}  d={rel.get('camera_distance_m')}",
            f"geom_angle={geom_angle}  selected_angle={source_angle}  source_d={source_dist}  source_elev={source_elev}",
            f"mask_IoU={score_iou}  Dice={score_dice}  projected_box={projected_box}",
        ]
        stacked = stack_preview(carla_bgr, preview_bgr, target_mask, lines)

        preview_video.write(preview_bgr)
        side_video.write(stacked)

        if (scenario_frame - wanted[0]) % max(1, int(args.save_frame_step)) == 0:
            cv2.imwrite(str(preview_dir / f"frame_{scenario_frame:06d}.png"), stacked)

        top_summary = ""
        if top:
            top_summary = " | ".join(
                f"{cand.angle_deg:.0f}/{cand.distance_m:.0f}/{cand.elevation_deg:.0f}:IoU={score_iou:.4f}"
                for score_iou, _score_dice, cand in top[:3]
            )

        rows.append({
            "scenario_frame": int(scenario_frame),
            "visible": int(visible),
            "camera_distance_m": rel.get("camera_distance_m"),
            "carla_geometric_view_angle_deg": geom_angle,
            "selected_angle_deg": source_angle,
            "selected_source_distance_m": source_dist,
            "selected_source_elevation_deg": source_elev,
            "selected_iou": score_iou,
            "selected_dice": score_dice,
            "projected_box_x1": None if projected_box is None else projected_box["x1"],
            "projected_box_y1": None if projected_box is None else projected_box["y1"],
            "projected_box_x2": None if projected_box is None else projected_box["x2"],
            "projected_box_y2": None if projected_box is None else projected_box["y2"],
            "projected_box_w": None if projected_box is None else projected_box["width"],
            "projected_box_h": None if projected_box is None else projected_box["height"],
            "top3": top_summary,
        })

        elapsed = time.time() - start_all
        print(
            f"[frame {scenario_frame:04d}] visible={int(visible)} "
            f"geom={geom_angle} selected={source_angle} "
            f"IoU={score_iou} elapsed={elapsed:.1f}s"
        )

    cap.release()
    preview_video.release()
    side_video.release()

    with (out_dir / "oracle_passby_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 92)
    print("PASS-BY PREVIEW COMPLETE")
    print("=" * 92)
    print("[preview] ", out_dir / "oracle_passby_preview.mp4")
    print("[side-by-side] ", out_dir / "oracle_passby_side_by_side.mp4")
    print("[rows]    ", out_dir / "oracle_passby_rows.csv")
    print("[frames]  ", preview_dir)
    print("=" * 92)


if __name__ == "__main__":
    main()
