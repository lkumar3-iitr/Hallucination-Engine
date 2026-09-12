"""Full-image evaluation including clipped, empty, and side-pass frames. No GT placement."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate import DEFAULT_BANK, ROOT
from passby_renderer import PassbyRenderer, geometry
from selector import transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--box-mode", choices=["existing", "anchor_fixed", "hull"], default="hull")
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir(exist_ok=True)
    engine = PassbyRenderer(args.bank, ROOT / "artifacts/cache")
    setup = json.loads((args.reference / "setup.json").read_text())
    camera = setup["camera"]
    frames = [json.loads(line) for line in (args.reference / "frames.jsonl").read_text().splitlines() if line]
    cap = cv2.VideoCapture(str(args.reference / "carla_reference.mp4"))
    bg = None
    if args.background:
        bg = cv2.VideoCapture(str(args.background / "background_only.mp4"))
        bg_setup = json.loads((args.background / "setup.json").read_text())
        bg_frames = [json.loads(line) for line in (args.background / "frames.jsonl").read_text().splitlines() if line]
        for k in ("width", "height", "fov_deg"):
            assert bg_setup["camera"][k] == camera[k]
        for index, row in enumerate(frames[:args.end+1]):
            for key, value in row["camera_transform"].items():
                assert abs(value-bg_frames[index]["camera_transform"][key]) < 1e-4
    width, height = camera["width"], camera["height"]
    writer = cv2.VideoWriter(str(args.output / "passby_comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             setup["fps"], (width*2, height+70))
    if not cap.isOpened() or not writer.isOpened() or (bg is not None and not bg.isOpened()):
        raise RuntimeError("Video open failed")
    cache = geometry.SpriteCache()
    dimensions = {"length_m": engine.selector.extents[0]*2, "width_m": engine.selector.extents[1]*2,
                  "height_m": engine.selector.extents[2]*2}
    rows = []
    try:
        for index, row in enumerate(frames[:args.end+1]):
            ok, original = cap.read()
            if not ok:
                raise RuntimeError(f"Frame {index} missing")
            if bg is not None:
                ok, background = bg.read()
                if not ok:
                    raise RuntimeError("Background frame missing")
            else:
                background = np.full_like(original, 210)
            actor, cam = transform(row["actor_transform"]), transform(row["camera_transform"])
            reason = ""
            try:
                result, alpha, meta = engine.render(background, actor, cam, camera["fov_deg"], args.box_mode)
            except ValueError as exc:
                result, alpha, meta = background.copy(), np.zeros((height, width)), {}
                reason = str(exc)
            # Only now read runtime GT. Never skip a clipped/empty/failed frame.
            gt = cv2.imread(str(args.reference / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                raise FileNotFoundError(row["mask_path"])
            gt = gt > 0
            predicted = alpha > (10/255)
            union = np.count_nonzero(gt | predicted)
            score = float(np.count_nonzero(gt & predicted)/union) if union else 1.0
            baseline_pixels, baseline_reason = None, None
            if args.baseline:
                _, base_meta = geometry.render_he_actor_view_matrix(
                    np.zeros_like(original), actor, cam, dimensions, engine.bank_config, engine.view_matrix,
                    cache, width, height, camera["fov_deg"], geometry_mode="physical_bbox_silhouette",
                    projection_mode="oriented_2p5d_support", view_selection_mode="projected_bbox_match")
                baseline_pixels = base_meta.get("rendered", False)
                baseline_reason = base_meta.get("reason") or (base_meta.get("box") or {}).get("reason")
            record = {"frame": index, "gt_pixels": int(gt.sum()), "predicted_pixels": int(predicted.sum()),
                      "iou": score, "reason": reason, "key": json.dumps(meta.get("key")),
                      "distance": meta.get("center_distance_m"),
                      "baseline_rendered": baseline_pixels, "baseline_reason": baseline_reason,
                      "box": json.dumps(meta.get("virtual_box"))}
            rows.append(record)
            panel = np.zeros((height+70, width*2, 3), np.uint8)
            panel[70:, :width] = original
            panel[70:, width:] = result
            cv2.putText(panel, f"CARLA | ASTRA side-pass / no GT selection or placement / {args.box_mode} size", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
            cv2.putText(panel, f"frame {index} IoU {score:.3f} gt {gt.sum()} HE {predicted.sum()} {reason}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            writer.write(panel)
            if index % 5 == 0:
                cv2.imwrite(str(args.output / "frames" / f"frame_{index:06d}.jpg"), panel)
            if index % 10 == 0:
                print(record, flush=True)
    finally:
        cap.release()
        writer.release()
        if bg is not None:
            bg.release()
    with (args.output / "rows.csv").open("w", newline="") as file:
        output = csv.DictWriter(file, fieldnames=rows[0].keys())
        output.writeheader()
        output.writerows(rows)
    def stats(items):
        return {"count": len(items), "mean_iou": float(np.mean([r["iou"] for r in items])) if items else None,
                "min_iou": min((r["iou"] for r in items), default=None)}
    summary = {"all": stats(rows), "gt_visible": stats([r for r in rows if r["gt_pixels"]]),
               "late_visible_frame40plus": stats([r for r in rows if r["gt_pixels"] and r["frame"] >= 40]),
               "dropouts": [r["frame"] for r in rows if r["gt_pixels"] > 20 and r["predicted_pixels"] == 0],
               "ghosts": [r["frame"] for r in rows if r["gt_pixels"] == 0 and r["predicted_pixels"] > 20],
               "errors": [r for r in rows if r["reason"]], "box_mode": args.box_mode,
               "runtime_GT": False, "background": str(args.background),
               "bank": str(args.bank), "asset_id": setup.get("asset_id"),
               "carla_blueprint": setup.get("carla_blueprint")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
