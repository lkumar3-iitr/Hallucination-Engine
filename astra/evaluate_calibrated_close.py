"""Evaluate calibrated close sampling against frozen, held-out CARLA poses."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .calibrated_close import CalibratedCloseBank, composite, coverage_weight
from .passby_renderer import PassbyRenderer
from .selector import transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--background", type=Path,
                        help="Separate matched clean-background recorder directory")
    parser.add_argument("--close-bank", type=Path, required=True)
    parser.add_argument("--native-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side-half-width", type=float, default=1.6)
    parser.add_argument("--hull", action="store_true")
    parser.add_argument("--refine-hull", action="store_true")
    parser.add_argument("--close-only-hull", action="store_true")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save one review JPEG every N frames; zero saves none")
    args = parser.parse_args()
    if args.close_only_hull and not (args.hull and args.refine_hull):
        parser.error("--close-only-hull requires --hull --refine-hull")
    if args.refine_hull and not args.hull:
        parser.error("--refine-hull requires --hull")
    root = Path(__file__).resolve().parent
    if root not in args.output.resolve().parents:
        parser.error("Output must be inside astra")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"frames").mkdir()
    setup = json.loads((args.reference/"setup.json").read_text())
    records = [json.loads(line) for line in (args.reference/"frames.jsonl").read_text().splitlines()]
    background_dir = args.background or args.reference
    if args.background:
        background_setup = json.loads((background_dir/"setup.json").read_text())
        background_records = [json.loads(line) for line in (background_dir/"frames.jsonl").read_text().splitlines()]
        if background_setup["camera"] != setup["camera"] or background_setup["fps"] != setup["fps"]:
            raise ValueError("Background calibration does not match reference")
        if len(records) != len(background_records):
            raise ValueError("Background/reference frame counts differ")
        for reference_row, background_row in zip(records, background_records):
            if reference_row["scenario_frame"] != background_row["scenario_frame"]:
                raise ValueError("Background/reference frame identities differ")
            for key in ("camera_transform", "actor_transform"):
                a, b = reference_row[key], background_row[key]
                if a.keys() != b.keys() or any(abs(float(a[k])-float(b[k])) > 1e-4 for k in a):
                    raise ValueError(f"Background/reference {key} mismatch")
    geometry = json.loads((args.close_bank/"geometry.json").read_text())
    sampler = CalibratedCloseBank(args.close_bank/"view_matrix.csv", geometry["center"], args.side_half_width)
    native = PassbyRenderer(args.native_bank, root/"artifacts/cache")
    if args.hull:
        from .hull_rays import HullRays
        sampler.hull = (HullRays.from_bounds(native.selector.center, float(np.linalg.norm(native.selector.extents)))
                        if args.close_only_hull else HullRays(native.selector.faces))
        if args.refine_hull:
            sampler.hull.refine(sampler, root/"artifacts/cache")
    w, h, fov = (setup["camera"][key] for key in ("width", "height", "fov_deg"))
    physical = cv2.VideoCapture(str(args.reference/"carla_reference.mp4"))
    clean = cv2.VideoCapture(str(background_dir/"background_only.mp4"))
    video = cv2.VideoWriter(str(args.output/"comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            setup["fps"], (w*3, h+40))
    if not video.isOpened():
        raise IOError("Video writer failed")
    rows = []
    with (args.output/"metadata.jsonl").open("w") as log:
        try:
            for i, record in enumerate(records):
                ok_a, gt_image = physical.read()
                ok_b, background = clean.read()
                if not ok_a or not ok_b:
                    raise IOError(f"Missing video frame {i}")
                actor, camera = transform(record["actor_transform"]), transform(record["camera_transform"])
                result = sampler.render(actor, camera, w, h, fov)
                if result is None:
                    image, alpha, meta = native.render(background, actor, camera, fov)
                    mode = "native"
                else:
                    layer, meta = result
                    # Smooth the finite bank boundary; interior capture weights are continuous.
                    weight = coverage_weight(sampler, meta["query_actor_local"])
                    if weight < 1:
                        native_image, native_alpha, _ = native.render(np.zeros_like(background), actor, camera, fov)
                        layer = weight*layer + (1-weight)*np.dstack((native_image, native_alpha))
                    image, alpha = composite(background, layer), layer[:, :, 3]
                    meta["close_weight"] = float(weight)
                    mode = "calibrated" if weight == 1 else "boundary_blend"
                mask_image = cv2.imread(str(args.reference/record["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask_image is None:
                    raise IOError(record["mask_path"])
                gt, predicted = mask_image > 0, alpha > 10/255
                union = np.count_nonzero(gt | predicted)
                intersection = np.count_nonzero(gt & predicted)
                row = dict(frame=i, iou=intersection/union if union else 1., gt_pixels=int(gt.sum()),
                           predicted_pixels=int(predicted.sum()), mode=mode)
                rows.append(row)
                log.write(json.dumps({**row, "metadata": meta})+"\n")
                overlay = background.copy()
                colors = np.zeros_like(overlay)
                colors[gt] = (0, 255, 0)
                colors[predicted] = (0, 0, 255)
                colors[gt & predicted] = (0, 160, 220)
                active = gt | predicted
                overlay[active] = (overlay[active]*.3 + colors[active]*.7).astype(np.uint8)
                panel = np.zeros((h+40, w*3, 3), np.uint8)
                panel[40:] = np.hstack((gt_image, image, overlay))
                cv2.putText(panel, f"CARLA | ASTRA candidate | masks  frame {i} IoU {row['iou']:.3f} {mode}",
                            (6, 26), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                video.write(panel)
                if args.save_every > 0 and i % args.save_every == 0:
                    cv2.imwrite(str(args.output/"frames"/f"frame_{i:06d}.jpg"), panel)
                if i % 25 == 0:
                    print(row, flush=True)
        finally:
            physical.release()
            clean.release()
            video.release()
    with (args.output/"rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    visible = [r for r in rows if r["gt_pixels"]]
    summary = dict(frames=len(rows), visible_frames=len(visible),
                   mean_visible_iou=float(np.mean([r["iou"] for r in visible])) if visible else None,
                   dropouts=sum(r["gt_pixels"] > 0 and r["predicted_pixels"] == 0 for r in rows),
                   ghosts=sum(r["gt_pixels"] == 0 and r["predicted_pixels"] > 0 for r in rows),
                   alpha_threshold=10/255, side_half_width=args.side_half_width,
                   parallax_proxy="bank_visual_hull" if args.hull else "actor_side_plane",
                   close_hull_refinement=args.refine_hull,
                   close_only_hull=args.close_only_hull,
                   reference=str(args.reference.resolve()), bank=str(args.close_bank.resolve()))
    summary["background"] = str(background_dir.resolve())
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
