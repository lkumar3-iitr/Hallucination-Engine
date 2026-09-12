"""Evaluate selection with the same external GT box for every sprite.

Fully visible frames only: a visible clipped box is not an unoccluded object box.
Runtime selection happens before reading the GT mask. No evaluation labels train
the visual hull, which uses only existing bank captures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from selector import Selector, canonical, crop_mask, iou_scores, transform


DEFAULT_BANK = Path(r"D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production\tesla_model3_native_full_v3")
ROOT = Path(__file__).resolve().parent


def summarize(rows):
    if not rows:
        return {"count": 0}
    return {"count": len(rows), **{
        key: {"mean": float(np.mean([r[key] for r in rows])),
              "min": float(min(r[key] for r in rows))}
        for key in ("selected_iou", "nearest_iou", "oracle_iou", "native_selected_iou",
                    "hull_gt_iou", "oracle_gap")},
        "within_0.02_of_oracle": float(np.mean([r["oracle_gap"] <= 0.02 for r in rows])),
        "within_0.05_of_oracle": float(np.mean([r["oracle_gap"] <= 0.05 for r in rows]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--references", type=Path, nargs="+", default=sorted(
        (ROOT.parent / "web/carla_reference").glob("*/setup.json")))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluation_v1")
    parser.add_argument("--spacing", type=float, default=0.035)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    if args.frame_step < 1 or args.size < 8 or args.spacing <= 0:
        parser.error("frame-step and spacing must be positive; size must be at least 8")
    args.output.mkdir(parents=True, exist_ok=True)
    selector = Selector(args.bank, ROOT / "artifacts/cache", args.size)
    selector.build_hull(spacing=args.spacing)
    results, skipped = [], []
    for ref in args.references:
        ref = ref.parent if ref.is_file() else ref
        setup = json.loads((ref / "setup.json").read_text())
        camera = setup["camera"]
        frames = [json.loads(line) for line in (ref / "frames.jsonl").read_text().splitlines() if line]
        out = args.output / ref.name
        out.mkdir(exist_ok=True)
        video = cv2.VideoWriter(str(out / "selection_comparison.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                setup["fps"]/args.frame_step, (1280, 320))
        if not video.isOpened():
            raise RuntimeError("Video writer failed")
        try:
            for row in frames[::args.frame_step]:
                frame = row["scenario_frame"]
                if not row.get("actor_transform"):
                    continue
                actor, cam = transform(row["actor_transform"]), transform(row["camera_transform"])
                start = time.perf_counter()
                try:
                    selected = selector.select(actor, cam, camera["width"], camera["height"], camera["fov_deg"])
                except ValueError as exc:
                    skipped.append({"reference": ref.name, "frame": frame, "reason": str(exc)})
                    continue
                elapsed = time.perf_counter()-start
                # This is the first access to target image evidence.
                gt = cv2.imread(str(ref / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    raise FileNotFoundError(ref / row["mask_path"])
                gt = gt > 0
                if gt.sum() < 20 or any(np.any(edge) for edge in (gt[0], gt[-1], gt[:, 0], gt[:, -1])):
                    skipped.append({"reference": ref.name, "frame": frame, "reason": "empty_small_or_border_touching"})
                    continue
                target = canonical(gt, args.size)
                scores = iou_scores(selector.masks, target)
                winner = int(np.argmax(scores))
                index, nearest = selected["index"], selected["baseline_index"]
                gt_crop = crop_mask(gt)
                native = cv2.resize(crop_mask(selector.alpha(index)).astype(np.uint8),
                                    (gt_crop.shape[1], gt_crop.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                native_iou = float(np.count_nonzero(native & gt_crop)/np.count_nonzero(native | gt_crop))
                record = {"reference": ref.name, "frame": frame,
                          "distance_m": selected["query"][1],
                          "query_angle": selected["query"][0], "query_elevation": selected["query"][2],
                          "selected_index": index, "selected_key": json.dumps(selected["key"]),
                          "oracle_index": winner, "oracle_key": json.dumps(selector.keys[winner].tolist()),
                          "nearest_index": nearest,
                          "selected_iou": float(scores[index]), "nearest_iou": float(scores[nearest]),
                          "oracle_iou": float(scores[winner]), "native_selected_iou": native_iou,
                          "oracle_gap": float(scores[winner]-scores[index]),
                          "hull_gt_iou": float(iou_scores(selected["predicted_mask"][None], target)[0]),
                          "selection_seconds": elapsed}
                results.append(record)
                masks = [target, selected["predicted_mask"], selector.masks[nearest],
                         selector.masks[index], selector.masks[winner]]
                labels = ["CARLA GT", "Bank hull", f"Nearest {scores[nearest]:.3f}",
                          f"ASTRA {scores[index]:.3f}", f"Oracle {scores[winner]:.3f}"]
                panel = np.full((320, 1280, 3), 28, np.uint8)
                for col, (mask, label) in enumerate(zip(masks, labels)):
                    rendered = cv2.resize(mask.astype(np.uint8)*255, (240, 240), interpolation=cv2.INTER_NEAREST)
                    panel[45:285, col*256+8:col*256+248] = rendered[:, :, None]
                    cv2.putText(panel, label, (col*256+8, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                cv2.putText(panel, f"frame {frame}  d={record['distance_m']:.2f}  key={selected['key']}",
                            (8, 310), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                video.write(panel)
                if frame % 5 == 0 or record["oracle_gap"] > .08:
                    cv2.imwrite(str(out / f"frame_{frame:06d}.png"), panel)
                if frame % 10 == 0:
                    print(f"{ref.name} frame={frame} selected={scores[index]:.3f} oracle={scores[winner]:.3f} gap={record['oracle_gap']:.3f}", flush=True)
        finally:
            video.release()
    if not results:
        raise RuntimeError("No eligible evaluation frames")
    with (args.output / "rows.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {"protocol": "Fully visible, GT external box, exhaustive canonical IoU oracle; no runtime GT in selector",
               "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("selector.py", "evaluate.py")},
               "bank": str(args.bank), "bank_fingerprint": selector.fingerprint,
               "hull": selector.hull_config, "canonical_size": args.size,
               "all": summarize(results), "close_under_10m": summarize([r for r in results if r["distance_m"] < 10]),
               "references": {name: summarize([r for r in results if r["reference"] == name])
                              for name in sorted({r["reference"] for r in results})},
               "skipped": skipped}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"all": summary["all"], "close": summary["close_under_10m"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
