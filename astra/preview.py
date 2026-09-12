"""RGB inspection of evaluated choices with GT standing in for external boxFinder."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate import DEFAULT_BANK, ROOT
from selector import Selector, composite_to_box


def read_frame(cap, frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selector = Selector(args.bank, ROOT / "artifacts/cache")
    rows = list(csv.DictReader((args.evaluation / "rows.csv").open()))
    rows = [r for r in rows if r["reference"] == args.reference.name]
    frames = {r["scenario_frame"]: r for r in map(json.loads, (args.reference / "frames.jsonl").read_text().splitlines())}
    setup = json.loads((args.reference / "setup.json").read_text())
    camera = setup["camera"]
    cap = cv2.VideoCapture(str(args.reference / "carla_reference.mp4"))
    bg_cap = None
    full_writer = None
    if args.background:
        bg_setup = json.loads((args.background / "setup.json").read_text())
        bg_frames = {r["scenario_frame"]: r for r in map(json.loads, (args.background / "frames.jsonl").read_text().splitlines())}
        for key in ("width", "height", "fov_deg"):
            if camera[key] != bg_setup["camera"][key]:
                raise ValueError("Background camera intrinsics differ")
        for row in rows:
            frame = int(row["frame"])
            for key, value in frames[frame]["camera_transform"].items():
                if abs(value-bg_frames[frame]["camera_transform"][key]) > 1e-4:
                    raise ValueError("Background camera pose mismatch")
        bg_cap = cv2.VideoCapture(str(args.background / "background_only.mp4"))
        full_writer = cv2.VideoWriter(str(args.output / "clean_background_gt_box.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                     (camera["width"]*2, camera["height"]+40))
        if not full_writer.isOpened():
            raise RuntimeError("Full preview writer failed")
    writer = cv2.VideoWriter(str(args.output / "rgb_selection_comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 440))
    if not writer.isOpened():
        raise RuntimeError("RGB preview writer failed")
    try:
        for row in rows:
            frame = int(row["frame"])
            original = read_frame(cap, frame)
            mask = cv2.imread(str(args.reference / frames[frame]["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
            ys, xs = np.where(mask)
            box = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
            x1, y1, x2, y2 = box
            sprite = cv2.imread(str(selector.path(selector.rows[int(row["selected_index"])])), cv2.IMREAD_UNCHANGED)
            blank = np.full_like(original, 210)
            selected, alpha = composite_to_box(blank, sprite, box)
            gt_cutout = blank.copy()
            gt_cutout[mask] = original[mask]
            patches = [gt_cutout[y1:y2, x1:x2], selected[y1:y2, x1:x2]]
            panel = np.full((440, 1280, 3), 210, np.uint8)
            for col, patch in enumerate(patches):
                scale = min(610/patch.shape[1], 350/patch.shape[0])
                patch = cv2.resize(patch, (round(patch.shape[1]*scale), round(patch.shape[0]*scale)))
                x = col*640 + (640-patch.shape[1])//2
                y = 50+(350-patch.shape[0])//2
                panel[y:y+patch.shape[0], x:x+patch.shape[1]] = patch
            for col, label in enumerate(("CARLA target", "ASTRA selected sprite / external GT box")):
                cv2.putText(panel, label, (col*640+15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 1)
            cv2.putText(panel, f"frame={frame}  key={row['selected_key']}  silhouette IoU={float(row['native_selected_iou']):.3f}",
                        (15, 425), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1)
            writer.write(panel)
            if frame % 10 == 0:
                cv2.imwrite(str(args.output / f"rgb_{frame:06d}.png"), panel)
            if bg_cap is not None:
                background = read_frame(bg_cap, frame)
                composite, _ = composite_to_box(background, sprite, box)
                full = np.zeros((camera["height"]+40, camera["width"]*2, 3), np.uint8)
                full[40:, :camera["width"]] = original
                full[40:, camera["width"]:] = composite
                cv2.putText(full, f"CARLA | ASTRA on recorded clean background (no inpainting); GT box; frame {frame}",
                            (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
                full_writer.write(full)
    finally:
        cap.release()
        writer.release()
        if bg_cap is not None:
            bg_cap.release()
            full_writer.release()
    (args.output / "preview_manifest.json").write_text(json.dumps({
        "frames": [int(r["frame"]) for r in rows], "fps": 10,
        "box_source": "GT mask full box on fully visible frames only",
        "selection_source": "bank-only visual hull, poses, intrinsics",
        "background": str(args.background) if args.background else "solid RGB inspection surface",
        "inpainting": False, "note": "Eligible frames only, gaps are not a continuous trajectory."}, indent=2))
    print(f"Wrote RGB previews to {args.output}")


if __name__ == "__main__":
    main()
