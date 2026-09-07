#!/usr/bin/env python3
"""
oracle_4320_silhouette.py

Offline oracle experiment for Hallucination Engine.

Purpose
-------
Test the strongest form of the "4320 bank is sufficient" hypothesis:

    Given the exact CARLA visible target box and target instance mask,
    can ANY sprite in the original 4320 bank reproduce the CARLA
    silhouette closely?

This script deliberately does NOT use the HE runtime sprite-angle formula,
distance selector, interpolation, close-range Cartesian bank, or learned
placement model. Every bank sprite is allowed to compete.

Expected CARLA reference directory
----------------------------------
A directory produced by:
    driving_models/common/record_scenario_carla_he_pair_v3.py
with:
    setup.json
    frames.jsonl
    masks/frame_XXXXXX.png
    carla_reference.mp4       (optional, only for diagnostic video)

Expected original sprite bank
-----------------------------
A directory containing:
    view_matrix.csv
and RGBA PNGs referenced by rgba_relpath or rgba_path.

The bank is expected to contain the original 4320 combinations:
    360 angles x 4 distances x 3 elevations
but the code searches whatever rows are actually present and reports the grid.

Core normalization
------------------
For every bank sprite:
  1. Read alpha.
  2. Crop to its non-transparent alpha bounds.
  3. Resize the cropped silhouette to a canonical square.

For every CARLA target:
  1. Read its instance mask.
  2. Crop to the exact visible CARLA mask box.
  3. Resize to the same canonical square.

Thus source capture distance, source crop size, target FOV magnification,
and target pixel box size do NOT directly influence the coarse silhouette
search. Elevation and azimuth remain as true appearance variables.

After the full-bank coarse search, the best candidates are refined at the
native CARLA box resolution by resizing their alpha crops directly into the
exact CARLA GT box.

Primary ranking metric
----------------------
Symmetric boundary Chamfer distance, normalized by the mask diagonal.
Lower is better.

Secondary metrics:
    IoU
    Dice
    Boundary F1 at 1/2/3 pixels (native refinement)

Outputs
-------
    oracle_results.csv
    oracle_summary.json
    top_candidates/frame_XXXXXX.csv
    best_masks/frame_XXXXXX.png
    diagnostics/frame_XXXXXX.png
    oracle_diagnostic.mp4     (if carla_reference.mp4 exists)

No CARLA Python package is required. This is completely offline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full-bank 4320 silhouette oracle against CARLA GT masks."
    )
    p.add_argument("--carla-dir", type=Path, required=True,
                   help="CARLA reference run directory (setup.json, frames.jsonl, masks/).")
    p.add_argument("--bank", type=Path, required=True,
                   help="Original 4320 sprite-bank directory containing view_matrix.csv.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--he-dir", type=Path, default=None,
                   help=("Optional HE replay directory from the same pair. "
                         "Used ONLY to log current HE requested/selected angles; "
                         "it never affects oracle search."))

    p.add_argument("--csv-name", default="view_matrix.csv")
    p.add_argument("--alpha-threshold", type=int, default=10,
                   help="Alpha threshold used to define source sprite silhouette.")
    p.add_argument("--canonical-size", type=int, default=160,
                   help="Square resolution used for the all-4320 coarse search.")
    p.add_argument("--native-refine-k", type=int, default=32,
                   help="Number of coarse winners refined at native GT box resolution.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of candidates saved per frame.")

    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=-1,
                   help="Inclusive scenario frame. -1 means final frame.")
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--frames", type=int, nargs="*", default=None,
                   help="Optional explicit scenario-frame list. Overrides start/end/step.")

    p.add_argument("--min-visible-pixels", type=int, default=20)
    p.add_argument("--min-box-width", type=int, default=4,
                   help="Skip degenerate visible boxes narrower than this.")
    p.add_argument("--min-box-height", type=int, default=4,
                   help="Skip degenerate visible boxes shorter than this.")
    p.add_argument("--skip-border-touching", action="store_true",
                   help=("Skip targets whose visible CARLA mask touches any image border. "
                         "Use this for the strict 'full silhouette visible' oracle."))
    p.add_argument("--border-margin-px", type=int, default=0,
                   help="Additional margin used when deciding whether a target touches the border.")
    p.add_argument("--strict-4320", action="store_true",
                   help="Fail unless the bank contains exactly 4320 usable rows.")

    p.add_argument("--no-video", action="store_true",
                   help="Do not generate oracle_diagnostic.mp4.")
    p.add_argument("--save-every-diagnostic", action="store_true",
                   help="Save a diagnostic PNG for every evaluated frame.")
    p.add_argument("--diagnostic-step", type=int, default=10,
                   help="Otherwise save diagnostic PNG every N evaluated frames.")
    p.add_argument("--video-fps", type=float, default=0.0,
                   help="0 = use CARLA setup FPS.")

    p.add_argument("--batch-size", type=int, default=256,
                   help="Candidate batch size for vectorized coarse scoring.")
    return p


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    bank_index: int
    angle_deg: float
    distance_m: float
    elevation_deg: float
    path: Path
    row: Dict[str, str]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

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
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
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
        candidate = bank_dir / raw
        if candidate.is_file():
            return candidate.resolve()
        candidate = bank_dir / "rgba" / p.name
        if candidate.is_file():
            return candidate.resolve()

    # Last-resort aliases, useful if the CSV schema changed slightly.
    for key in ("path", "sprite_path", "file", "filename"):
        raw = str(row.get(key, "")).strip()
        if not raw:
            continue
        p = bank_dir / raw
        if p.is_file():
            return p.resolve()
        p2 = bank_dir / "rgba" / Path(raw).name
        if p2.is_file():
            return p2.resolve()

    raise FileNotFoundError(
        "Could not resolve RGBA sprite path from CSV row. "
        f"Available keys={sorted(row.keys())}"
    )


def alpha_crop_from_rgba(path: Path, alpha_threshold: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read sprite: {path}")
    if image.ndim != 3 or image.shape[2] < 4:
        raise RuntimeError(f"Sprite is not RGBA: {path} shape={image.shape}")

    alpha = image[:, :, 3]
    mask = alpha > int(alpha_threshold)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError(f"Sprite alpha is empty: {path}")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def resize_binary(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    # INTER_NEAREST preserves a hard silhouette and avoids alpha-threshold bias.
    out = cv2.resize(
        mask.astype(np.uint8),
        (max(1, int(width)), max(1, int(height))),
        interpolation=cv2.INTER_NEAREST,
    )
    return (out > 0).astype(np.uint8)


def inner_boundary(mask_u8: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return ((mask_u8 > 0) & (eroded == 0)).astype(np.uint8)


def distance_to_boundary(boundary_u8: np.ndarray) -> np.ndarray:
    # OpenCV distanceTransform measures distance from nonzero pixels to zeros.
    # Make boundary pixels zero and all other pixels nonzero.
    src = np.where(boundary_u8 > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, 3)


def safe_f1(precision: float, recall: float) -> float:
    den = precision + recall
    return 0.0 if den <= 1e-12 else 2.0 * precision * recall / den


def boundary_f1(a_boundary: np.ndarray, b_boundary: np.ndarray, tolerance_px: int) -> float:
    a = (a_boundary > 0).astype(np.uint8)
    b = (b_boundary > 0).astype(np.uint8)
    if np.count_nonzero(a) == 0 or np.count_nonzero(b) == 0:
        return 0.0

    k = 2 * int(tolerance_px) + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    b_dil = cv2.dilate(b, kernel, iterations=1)
    a_dil = cv2.dilate(a, kernel, iterations=1)

    precision = float(np.count_nonzero((a > 0) & (b_dil > 0))) / float(np.count_nonzero(a))
    recall = float(np.count_nonzero((b > 0) & (a_dil > 0))) / float(np.count_nonzero(b))
    return safe_f1(precision, recall)


def pair_metrics(candidate: np.ndarray, target: np.ndarray) -> dict:
    c = candidate > 0
    t = target > 0

    inter = int(np.count_nonzero(c & t))
    union = int(np.count_nonzero(c | t))
    c_area = int(np.count_nonzero(c))
    t_area = int(np.count_nonzero(t))

    iou = float(inter / union) if union else 1.0
    dice = float(2 * inter / (c_area + t_area)) if (c_area + t_area) else 1.0

    c_edge = inner_boundary(c.astype(np.uint8))
    t_edge = inner_boundary(t.astype(np.uint8))
    c_edge_n = int(np.count_nonzero(c_edge))
    t_edge_n = int(np.count_nonzero(t_edge))

    # A zero-edge silhouette is a degenerate comparison, not a perfect match.
    if c_edge_n == 0 or t_edge_n == 0:
        return {
            "iou": iou,
            "dice": dice,
            "boundary_chamfer_px": float("inf"),
            "boundary_chamfer_norm": float("inf"),
            "boundary_f1_1px": 0.0,
            "boundary_f1_2px": 0.0,
            "boundary_f1_3px": 0.0,
        }

    c_dt = distance_to_boundary(c_edge)
    t_dt = distance_to_boundary(t_edge)

    c_to_t = float(np.sum(t_dt[c_edge > 0], dtype=np.float64) / c_edge_n)
    t_to_c = float(np.sum(c_dt[t_edge > 0], dtype=np.float64) / t_edge_n)
    chamfer = 0.5 * (c_to_t + t_to_c)

    h, w = target.shape[:2]
    diagonal = math.hypot(float(w), float(h))
    chamfer_norm = float(chamfer / max(1e-9, diagonal))

    return {
        "iou": iou,
        "dice": dice,
        "boundary_chamfer_px": chamfer,
        "boundary_chamfer_norm": chamfer_norm,
        "boundary_f1_1px": boundary_f1(c_edge, t_edge, 1),
        "boundary_f1_2px": boundary_f1(c_edge, t_edge, 2),
        "boundary_f1_3px": boundary_f1(c_edge, t_edge, 3),
    }


def circular_delta_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def pose_derived_values(frame: dict) -> dict:
    camera = frame.get("camera_transform") or {}
    actor = frame.get("actor_transform") or {}

    result = {
        "camera_actor_center_distance_m": None,
        "camera_actor_horizontal_distance_m": None,
        "camera_position_elevation_deg": None,
    }
    try:
        dx = float(camera["x"]) - float(actor["x"])
        dy = float(camera["y"]) - float(actor["y"])
        dz = float(camera["z"]) - float(actor["z"])
        horiz = math.hypot(dx, dy)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        elev = math.degrees(math.atan2(dz, max(1e-12, horiz)))
        result.update({
            "camera_actor_center_distance_m": dist,
            "camera_actor_horizontal_distance_m": horiz,
            "camera_position_elevation_deg": elev,
        })
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Bank loading and normalization
# ---------------------------------------------------------------------------

def load_candidates(bank_dir: Path, csv_name: str) -> List[Candidate]:
    csv_path = bank_dir / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"Bank CSV not found: {csv_path}")

    rows = read_csv(csv_path)
    candidates = []
    failures = []

    for idx, row in enumerate(rows):
        try:
            angle = first_float(row, ("angle_deg", "view_angle_deg", "yaw_deg"))
            distance = first_float(row, ("distance_m", "camera_distance_m", "d_m"))
            elevation = first_float(row, ("elevation_deg", "camera_elevation_deg", "elev_deg"))
            if angle is None or distance is None or elevation is None:
                raise ValueError(
                    f"missing angle/distance/elevation fields in row keys={sorted(row.keys())}"
                )
            path = source_path(bank_dir, row)
            candidates.append(Candidate(
                bank_index=idx,
                angle_deg=float(angle) % 360.0,
                distance_m=float(distance),
                elevation_deg=float(elevation),
                path=path,
                row=row,
            ))
        except Exception as exc:
            failures.append((idx, str(exc)))

    if failures:
        print(f"[warning] {len(failures)} CSV rows could not be used.")
        for idx, msg in failures[:5]:
            print(f"  row {idx}: {msg}")
        if len(failures) > 5:
            print("  ...")

    if not candidates:
        raise RuntimeError("No usable bank candidates were found.")
    return candidates


def summarize_grid(candidates: Sequence[Candidate]) -> dict:
    angles = sorted({round(float(c.angle_deg), 6) for c in candidates})
    distances = sorted({round(float(c.distance_m), 6) for c in candidates})
    elevations = sorted({round(float(c.elevation_deg), 6) for c in candidates})
    return {
        "count": len(candidates),
        "angle_count": len(angles),
        "distance_count": len(distances),
        "elevation_count": len(elevations),
        "angles": angles,
        "distances_m": distances,
        "elevations_deg": elevations,
        "cartesian_product_count": len(angles) * len(distances) * len(elevations),
    }


def preload_normalized_bank(
    candidates: Sequence[Candidate],
    canonical_size: int,
    alpha_threshold: int,
):
    n = len(candidates)
    s = int(canonical_size)

    masks = np.zeros((n, s, s), dtype=np.uint8)
    edges = np.zeros((n, s, s), dtype=np.uint8)
    # Keep float32. float16 can overflow in large reductions during
    # boundary-distance scoring and is not worth the diagnostic ambiguity.
    edge_dt = np.zeros((n, s, s), dtype=np.float32)

    original_crops: List[np.ndarray] = [None] * n

    start = time.time()
    for i, candidate in enumerate(candidates):
        crop = alpha_crop_from_rgba(candidate.path, alpha_threshold)
        original_crops[i] = crop

        norm = resize_binary(crop, s, s)
        edge = inner_boundary(norm)
        dt = distance_to_boundary(edge)

        masks[i] = norm
        edges[i] = edge
        edge_dt[i] = dt.astype(np.float32)

        if i % 250 == 0 or i == n - 1:
            elapsed = time.time() - start
            print(f"\r[preload] {i+1:4d}/{n} sprites  elapsed={elapsed:6.1f}s", end="", flush=True)
    print()

    area = masks.reshape(n, -1).sum(axis=1).astype(np.int32)
    edge_count = edges.reshape(n, -1).sum(axis=1).astype(np.int32)
    return masks, edges, edge_dt, area, edge_count, original_crops


# ---------------------------------------------------------------------------
# Full-bank coarse search
# ---------------------------------------------------------------------------

def coarse_search(
    target_norm: np.ndarray,
    candidate_masks: np.ndarray,
    candidate_edges: np.ndarray,
    candidate_edge_dt: np.ndarray,
    candidate_area: np.ndarray,
    candidate_edge_count: np.ndarray,
    batch_size: int,
):
    target = (target_norm > 0).astype(np.uint8)
    target_bool = target.astype(bool)
    target_area = int(np.count_nonzero(target_bool))
    target_edge = inner_boundary(target)
    target_edge_bool = target_edge.astype(bool)
    target_edge_count = max(1, int(np.count_nonzero(target_edge_bool)))
    target_dt = distance_to_boundary(target_edge).astype(np.float32)

    n = candidate_masks.shape[0]
    scores = np.empty(n, dtype=np.float32)
    ious = np.empty(n, dtype=np.float32)
    dices = np.empty(n, dtype=np.float32)

    for begin in range(0, n, int(batch_size)):
        end = min(n, begin + int(batch_size))

        cm = candidate_masks[begin:end].astype(bool)
        ce = candidate_edges[begin:end].astype(bool)
        cdt = candidate_edge_dt[begin:end].astype(np.float32)

        inter = np.count_nonzero(cm & target_bool[None, :, :], axis=(1, 2)).astype(np.float32)
        union = (
            candidate_area[begin:end].astype(np.float32)
            + float(target_area)
            - inter
        )
        iou = np.divide(inter, np.maximum(union, 1.0))

        dice_den = candidate_area[begin:end].astype(np.float32) + float(target_area)
        dice = np.divide(2.0 * inter, np.maximum(dice_den, 1.0))

        # Candidate boundary -> target boundary.
        c_to_t_sum = np.sum(
            ce.astype(np.float32) * target_dt[None, :, :],
            axis=(1, 2),
        )
        c_to_t = c_to_t_sum / np.maximum(candidate_edge_count[begin:end], 1)

        # Target boundary -> candidate boundary, using precomputed candidate DT.
        t_to_c_sum = np.sum(
            cdt * target_edge_bool[None, :, :].astype(np.float32),
            axis=(1, 2),
        )
        t_to_c = t_to_c_sum / float(target_edge_count)

        chamfer = 0.5 * (c_to_t + t_to_c)
        diagonal = math.hypot(float(target.shape[1]), float(target.shape[0]))
        chamfer_norm = chamfer / max(1e-9, diagonal)

        scores[begin:end] = chamfer_norm.astype(np.float32)
        ious[begin:end] = iou.astype(np.float32)
        dices[begin:end] = dice.astype(np.float32)

    # Primary: lower Chamfer. Secondary: higher IoU.
    order = np.lexsort((-ious, scores))
    return order, scores, ious, dices


# ---------------------------------------------------------------------------
# Native refinement
# ---------------------------------------------------------------------------

def native_refine(
    coarse_indices: Sequence[int],
    target_crop: np.ndarray,
    candidates: Sequence[Candidate],
    original_crops: Sequence[np.ndarray],
):
    h, w = target_crop.shape[:2]
    rows = []

    for idx in coarse_indices:
        candidate_native = resize_binary(original_crops[int(idx)], w, h)
        metrics = pair_metrics(candidate_native, target_crop)
        c = candidates[int(idx)]
        rows.append({
            "candidate_index": int(idx),
            "bank_index": int(c.bank_index),
            "angle_deg": float(c.angle_deg),
            "distance_m": float(c.distance_m),
            "elevation_deg": float(c.elevation_deg),
            "path": str(c.path),
            "native_mask": candidate_native,
            **metrics,
        })

    rows.sort(
        key=lambda r: (
            float(r["boundary_chamfer_norm"]),
            -float(r["iou"]),
        )
    )
    return rows


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def read_video_frame(cap: Optional[cv2.VideoCapture], frame_index: int, width: int, height: int):
    if cap is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return frame


def fit_panel(image: np.ndarray, panel_w: int, panel_h: int, label: str) -> np.ndarray:
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(panel_w / max(1, w), (panel_h - 34) / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (panel_w - nw) // 2
    y = (panel_h - 34 - nh) // 2 + 28
    canvas[y:y+nh, x:x+nw] = resized
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_diagnostic(
    rgb_bgr: np.ndarray,
    target_full: np.ndarray,
    oracle_full: np.ndarray,
    title_lines: Sequence[str],
) -> np.ndarray:
    target_rgb = np.repeat((target_full > 0)[:, :, None].astype(np.uint8) * 255, 3, axis=2)
    oracle_rgb = np.repeat((oracle_full > 0)[:, :, None].astype(np.uint8) * 255, 3, axis=2)

    diff = np.zeros_like(target_rgb)
    # White = overlap, red = CARLA-only, green = oracle-only.
    t = target_full > 0
    o = oracle_full > 0
    diff[t & o] = (255, 255, 255)
    diff[t & ~o] = (0, 0, 255)
    diff[~t & o] = (0, 255, 0)

    panel_w = 640
    panel_h = 390
    top = np.hstack([
        fit_panel(rgb_bgr, panel_w, panel_h, "CARLA RGB"),
        fit_panel(target_rgb, panel_w, panel_h, "CARLA target silhouette"),
    ])
    bottom = np.hstack([
        fit_panel(oracle_rgb, panel_w, panel_h, "Best 4320 oracle silhouette"),
        fit_panel(diff, panel_w, panel_h, "Difference: white overlap / red GT / green oracle"),
    ])
    canvas = np.vstack([top, bottom])

    header_h = 28 + 22 * len(title_lines)
    out = np.zeros((canvas.shape[0] + header_h, canvas.shape[1], 3), dtype=np.uint8)
    out[header_h:] = canvas
    cv2.putText(out, "HE 4320 FULL-BANK SILHOUETTE ORACLE", (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    for i, line in enumerate(title_lines):
        cv2.putText(out, str(line), (10, 46 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    carla_dir = args.carla_dir.resolve()
    bank_dir = args.bank.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    top_dir = out_dir / "top_candidates"
    mask_out_dir = out_dir / "best_masks"
    diag_dir = out_dir / "diagnostics"
    top_dir.mkdir(exist_ok=True)
    mask_out_dir.mkdir(exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    setup_path = carla_dir / "setup.json"
    frames_path = carla_dir / "frames.jsonl"
    if not setup_path.is_file():
        raise FileNotFoundError(f"Missing {setup_path}")
    if not frames_path.is_file():
        raise FileNotFoundError(f"Missing {frames_path}")

    setup = load_json(setup_path)
    frames = load_jsonl(frames_path)
    if not frames:
        raise RuntimeError("frames.jsonl is empty.")

    he_frame_by_scenario = {}
    if args.he_dir is not None:
        he_frames_path = args.he_dir.resolve() / "frames.jsonl"
        if not he_frames_path.is_file():
            raise FileNotFoundError(f"Optional HE frames file not found: {he_frames_path}")
        he_frames = load_jsonl(he_frames_path)
        he_frame_by_scenario = {
            int(row.get("scenario_frame", i)): row
            for i, row in enumerate(he_frames)
        }

    camera_cfg = setup.get("camera") or {}
    width = int(camera_cfg.get("width", camera_cfg.get("image_width", 1280)))
    height = int(camera_cfg.get("height", camera_cfg.get("image_height", 720)))
    fov_deg = float(camera_cfg.get("fov_deg", camera_cfg.get("fov", 90.0)))
    fps = float(setup.get("fps", 20.0))

    candidates = load_candidates(bank_dir, args.csv_name)
    grid = summarize_grid(candidates)

    print("=" * 92)
    print("HE 4320 FULL-BANK SILHOUETTE ORACLE")
    print("=" * 92)
    print("[CARLA dir]      ", carla_dir)
    print("[bank]           ", bank_dir)
    print("[output]         ", out_dir)
    print("[camera]         ", f"{width}x{height}, FOV={fov_deg:.3f} deg")
    print("[bank rows]      ", grid["count"])
    print("[angles]         ", grid["angle_count"])
    print("[distances m]    ", grid["distances_m"])
    print("[elevations deg] ", grid["elevations_deg"])
    print("[grid product]   ", grid["cartesian_product_count"])
    print("=" * 92)

    if args.strict_4320 and len(candidates) != 4320:
        raise RuntimeError(
            f"--strict-4320 requested but usable candidate count is {len(candidates)}"
        )

    # Full-bank normalized cache.
    (
        candidate_masks,
        candidate_edges,
        candidate_edge_dt,
        candidate_area,
        candidate_edge_count,
        original_crops,
    ) = preload_normalized_bank(
        candidates,
        canonical_size=int(args.canonical_size),
        alpha_threshold=int(args.alpha_threshold),
    )

    # Frame selection.
    if args.frames:
        wanted = [int(v) for v in args.frames]
    else:
        last = len(frames) - 1 if int(args.frame_end) < 0 else min(int(args.frame_end), len(frames) - 1)
        wanted = list(range(max(0, int(args.frame_start)), last + 1, max(1, int(args.frame_step))))

    frame_by_scenario = {
        int(row.get("scenario_frame", i)): row
        for i, row in enumerate(frames)
    }

    # Optional CARLA video.
    video_path = carla_dir / "carla_reference.mp4"
    cap = None
    if video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap = None

    video_writer = None
    result_rows = []
    evaluated_count = 0
    start_all = time.time()

    for scenario_frame in wanted:
        frame = frame_by_scenario.get(int(scenario_frame))
        if frame is None:
            print(f"[skip] scenario frame {scenario_frame}: not found in frames.jsonl")
            continue

        mask_rel = frame.get("mask_path")
        if not mask_rel:
            print(f"[skip] frame {scenario_frame}: no mask_path")
            continue
        target_path = carla_dir / str(mask_rel)
        target_gray = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if target_gray is None:
            print(f"[skip] frame {scenario_frame}: could not read {target_path}")
            continue
        target_full = (target_gray > 0).astype(np.uint8)

        visible_pixels = int(np.count_nonzero(target_full))
        if visible_pixels < int(args.min_visible_pixels):
            continue

        bbox = mask_bbox(target_full)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        gt_w = int(x2 - x1 + 1)
        gt_h = int(y2 - y1 + 1)

        margin = max(0, int(args.border_margin_px))
        touches_border = bool(
            x1 <= margin
            or y1 <= margin
            or x2 >= (target_full.shape[1] - 1 - margin)
            or y2 >= (target_full.shape[0] - 1 - margin)
        )

        if gt_w < int(args.min_box_width) or gt_h < int(args.min_box_height):
            print(
                f"[skip] frame {scenario_frame}: degenerate visible box "
                f"{gt_w}x{gt_h}"
            )
            continue

        if args.skip_border_touching and touches_border:
            print(
                f"[skip] frame {scenario_frame}: target touches image border "
                f"bbox=({x1},{y1})-({x2},{y2})"
            )
            continue

        target_crop = target_full[y1:y2+1, x1:x2+1]

        target_norm = resize_binary(
            target_crop,
            int(args.canonical_size),
            int(args.canonical_size),
        )

        order, coarse_scores, coarse_ious, coarse_dices = coarse_search(
            target_norm=target_norm,
            candidate_masks=candidate_masks,
            candidate_edges=candidate_edges,
            candidate_edge_dt=candidate_edge_dt,
            candidate_area=candidate_area,
            candidate_edge_count=candidate_edge_count,
            batch_size=int(args.batch_size),
        )

        refine_k = min(int(args.native_refine_k), len(order))
        refined = native_refine(
            coarse_indices=order[:refine_k],
            target_crop=target_crop,
            candidates=candidates,
            original_crops=original_crops,
        )
        best = refined[0]

        # Build full-frame oracle mask using exact CARLA GT box.
        oracle_full = np.zeros_like(target_full)
        oracle_full[y1:y2+1, x1:x2+1] = best["native_mask"]
        cv2.imwrite(
            str(mask_out_dir / f"frame_{scenario_frame:06d}.png"),
            oracle_full * 255,
        )

        # Save top candidate table.
        top_rows = refined[:min(int(args.top_k), len(refined))]
        top_csv_path = top_dir / f"frame_{scenario_frame:06d}.csv"
        with top_csv_path.open("w", encoding="utf-8", newline="") as f:
            fields = [
                "rank", "candidate_index", "bank_index",
                "angle_deg", "distance_m", "elevation_deg",
                "boundary_chamfer_px", "boundary_chamfer_norm",
                "boundary_f1_1px", "boundary_f1_2px", "boundary_f1_3px",
                "iou", "dice", "path",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rank, row in enumerate(top_rows, 1):
                writer.writerow({
                    "rank": rank,
                    **{k: row[k] for k in fields if k != "rank"},
                })

        relative = frame.get("camera_relative") or {}
        derived = pose_derived_values(frame)
        geom_angle = frame.get("carla_geometric_view_angle_deg")

        # CARLA-condition frames intentionally have no HE selection.
        # If an HE replay directory was supplied, read current HE selection
        # from the matching scenario frame for comparison only.
        he_frame = he_frame_by_scenario.get(int(scenario_frame), {})
        current_angle = he_frame.get(
            "he_selected_angle_deg",
            frame.get("he_selected_angle_deg"),
        )
        requested_angle = he_frame.get(
            "he_requested_viewpoint_angle_deg",
            frame.get("he_requested_viewpoint_angle_deg"),
        )

        result = {
            "scenario_frame": int(scenario_frame),
            "t_s": float(frame.get("t_s", scenario_frame / max(1e-9, fps))),
            "visible_pixels": visible_pixels,

            "gt_x1": x1, "gt_y1": y1, "gt_x2": x2, "gt_y2": y2,
            "gt_width_px": gt_w, "gt_height_px": gt_h,
            "target_touches_image_border": touches_border,

            "camera_fov_deg": fov_deg,
            "camera_pitch_deg": first_float(frame.get("camera_transform") or {}, ("pitch",), None),
            "camera_yaw_deg": first_float(frame.get("camera_transform") or {}, ("yaw",), None),
            "camera_roll_deg": first_float(frame.get("camera_transform") or {}, ("roll",), None),

            "camera_forward_m": relative.get("camera_forward_m"),
            "camera_right_m": relative.get("camera_right_m"),
            "camera_up_m": relative.get("camera_up_m"),
            "camera_distance_m": relative.get("camera_distance_m"),
            **derived,

            "carla_geometric_view_angle_deg": geom_angle,
            "current_he_requested_angle_deg": requested_angle,
            "current_he_selected_angle_deg": current_angle,

            "oracle_angle_deg": float(best["angle_deg"]),
            "oracle_source_distance_m": float(best["distance_m"]),
            "oracle_source_elevation_deg": float(best["elevation_deg"]),
            "oracle_source_path": str(best["path"]),

            "oracle_boundary_chamfer_px": float(best["boundary_chamfer_px"]),
            "oracle_boundary_chamfer_norm": float(best["boundary_chamfer_norm"]),
            "oracle_boundary_f1_1px": float(best["boundary_f1_1px"]),
            "oracle_boundary_f1_2px": float(best["boundary_f1_2px"]),
            "oracle_boundary_f1_3px": float(best["boundary_f1_3px"]),
            "oracle_iou": float(best["iou"]),
            "oracle_dice": float(best["dice"]),

            "oracle_minus_geometric_angle_deg": (
                circular_delta_deg(float(best["angle_deg"]), float(geom_angle))
                if geom_angle is not None else None
            ),
            "oracle_minus_current_selected_angle_deg": (
                circular_delta_deg(float(best["angle_deg"]), float(current_angle))
                if current_angle is not None else None
            ),
        }
        result_rows.append(result)

        rgb = read_video_frame(cap, scenario_frame, width, height)
        title_lines = [
            (
                f"frame={scenario_frame}  d={derived['camera_actor_center_distance_m']!s} m  "
                f"target_elev={derived['camera_position_elevation_deg']!s} deg  FOV={fov_deg:.1f}"
            ),
            (
                f"geom_angle={geom_angle}  current_HE={current_angle}  "
                f"oracle={best['angle_deg']:.1f} deg"
            ),
            (
                f"oracle source: d={best['distance_m']:.1f} m  elev={best['elevation_deg']:.1f} deg  "
                f"Chamfer={best['boundary_chamfer_px']:.3f}px  "
                f"IoU={best['iou']:.4f}  Dice={best['dice']:.4f}"
            ),
        ]
        diag = make_diagnostic(rgb, target_full, oracle_full, title_lines)

        if (
            args.save_every_diagnostic
            or evaluated_count % max(1, int(args.diagnostic_step)) == 0
        ):
            cv2.imwrite(
                str(diag_dir / f"frame_{scenario_frame:06d}.png"),
                diag,
            )

        if not args.no_video:
            if video_writer is None:
                out_fps = float(args.video_fps) if float(args.video_fps) > 0 else fps
                video_writer = cv2.VideoWriter(
                    str(out_dir / "oracle_diagnostic.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    out_fps,
                    (diag.shape[1], diag.shape[0]),
                )
                if not video_writer.isOpened():
                    raise RuntimeError("Could not open diagnostic video writer.")
            video_writer.write(diag)

        evaluated_count += 1
        elapsed = time.time() - start_all
        print(
            f"[frame {scenario_frame:04d}] "
            f"GT={gt_w}x{gt_h} "
            f"oracle=(a={best['angle_deg']:.1f}, d={best['distance_m']:.1f}, "
            f"e={best['elevation_deg']:.1f}) "
            f"chamfer={best['boundary_chamfer_px']:.3f}px "
            f"IoU={best['iou']:.4f} "
            f"elapsed={elapsed:.1f}s"
        )

    if cap is not None:
        cap.release()
    if video_writer is not None:
        video_writer.release()

    if not result_rows:
        raise RuntimeError("No visible frames were evaluated.")

    # Results CSV.
    results_path = out_dir / "oracle_results.csv"
    fieldnames = list(result_rows[0].keys())
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    # Summary.
    chamfers = np.asarray([r["oracle_boundary_chamfer_px"] for r in result_rows], dtype=float)
    chamfers_norm = np.asarray([r["oracle_boundary_chamfer_norm"] for r in result_rows], dtype=float)
    ious = np.asarray([r["oracle_iou"] for r in result_rows], dtype=float)
    dices = np.asarray([r["oracle_dice"] for r in result_rows], dtype=float)
    elevs = [float(r["oracle_source_elevation_deg"]) for r in result_rows]
    dists = [float(r["oracle_source_distance_m"]) for r in result_rows]

    def counts(values):
        out = {}
        for v in values:
            key = f"{v:g}"
            out[key] = out.get(key, 0) + 1
        return out

    summary = {
        "experiment": "full_4320_exact_gt_box_silhouette_oracle_v1",
        "carla_dir": str(carla_dir),
        "he_dir_optional": str(args.he_dir.resolve()) if args.he_dir is not None else None,
        "bank_dir": str(bank_dir),
        "camera": camera_cfg,
        "bank_grid": grid,
        "normalization": {
            "source_alpha_crop": True,
            "source_distance_pixel_scale_removed": True,
            "target_exact_visible_gt_box_used": True,
            "coarse_canonical_square_px": int(args.canonical_size),
            "elevation_is_searched_not_removed": True,
            "all_bank_rows_allowed_to_compete": True,
            "skip_border_touching": bool(args.skip_border_touching),
            "min_box_width": int(args.min_box_width),
            "min_box_height": int(args.min_box_height),
        },
        "evaluated_frames": len(result_rows),
        "metrics": {
            "boundary_chamfer_px_mean": float(np.mean(chamfers)),
            "boundary_chamfer_px_median": float(np.median(chamfers)),
            "boundary_chamfer_px_p95": float(np.percentile(chamfers, 95)),
            "boundary_chamfer_norm_mean": float(np.mean(chamfers_norm)),
            "boundary_chamfer_norm_median": float(np.median(chamfers_norm)),
            "iou_mean": float(np.mean(ious)),
            "iou_median": float(np.median(ious)),
            "dice_mean": float(np.mean(dices)),
            "dice_median": float(np.median(dices)),
        },
        "oracle_source_distance_counts": counts(dists),
        "oracle_source_elevation_counts": counts(elevs),
    }
    summary_path = out_dir / "oracle_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 92)
    print("ORACLE COMPLETE")
    print("=" * 92)
    print("[evaluated frames] ", len(result_rows))
    print("[Chamfer mean px]  ", f"{np.mean(chamfers):.4f}")
    print("[Chamfer median px]", f"{np.median(chamfers):.4f}")
    print("[IoU mean]         ", f"{np.mean(ious):.4f}")
    print("[Dice mean]        ", f"{np.mean(dices):.4f}")
    print("[results]          ", results_path)
    print("[summary]          ", summary_path)
    if not args.no_video:
        print("[video]            ", out_dir / "oracle_diagnostic.mp4")
    print("=" * 92)


if __name__ == "__main__":
    main()
