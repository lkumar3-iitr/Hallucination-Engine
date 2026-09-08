"""Validate full-sweep and actual recorder integration artifacts."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {"sweeps": {}, "videos": {}, "code_sha256": {}}
    for yaw in (0, -60, -90):
        capture = ROOT / f"artifacts/complete_sweeps_v2/yaw_{yaw}"
        assert json.loads((capture / "COMPLETE.json").read_text())["frames"] == 125
        assert len(list((capture / "masks").glob("*.png"))) == 125
        output = ROOT / f"artifacts/validated_passby_yaw_{yaw}"
        with (output / "rows.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert [int(r["frame"]) for r in rows] == list(range(125))
        summary = json.loads((output / "summary.json").read_text())
        assert not summary["dropouts"] and not summary["ghosts"] and not summary["errors"]
        report["sweeps"][str(yaw)] = {"frames": 125, "visible_mean_iou": summary["gt_visible"]["mean_iou"],
            "baseline_dropouts": [int(r["frame"]) for r in rows if int(r["gt_pixels"]) > 20 and r["baseline_rendered"] == "False"],
            "renderer_dropouts": summary["dropouts"], "renderer_ghosts": summary["ghosts"]}
        path = output / "passby_comparison.mp4"
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 125
        for frame in (0, 50, 62, 70, 80, 124):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            assert ok and image.std() > 5
            if frame == 62:
                cv2.imwrite(str(output / "alongside_verified.jpg"), image)
        cap.release()
        report["videos"][str(path.relative_to(ROOT))] = 125
    integration = ROOT / "artifacts/recorder_integration_v2"
    setup = json.loads((integration / "setup.json").read_text())
    assert setup["he_renderer_backend"] == "he_sprite_renderer_v1"
    assert setup["runtime_ground_truth_used"] is False
    frames = [json.loads(line) for line in (integration / "frames.jsonl").read_text().splitlines()]
    assert len(frames) == len(list((integration / "masks").glob("*.png"))) == 80
    original = ROOT.parent / "web/carla_reference/static_left_tesla_ego_overtakes_200"
    originals = [json.loads(line) for line in (original / "frames.jsonl").read_text().splitlines()]
    scores, missing, ghosts = [], [], []
    max_pose_error = 0
    for frame, row in enumerate(frames):
        gt = cv2.imread(str(original / originals[frame]["mask_path"]), 0)
        predicted = cv2.imread(str(integration / row["mask_path"]), 0)
        assert gt is not None and predicted is not None
        gt, predicted = gt > 0, predicted > 0
        if gt.any():
            scores.append(float(np.count_nonzero(gt & predicted)/np.count_nonzero(gt | predicted)))
        if gt.sum() > 20 and not predicted.any():
            missing.append(frame)
        if not gt.any() and predicted.sum() > 20:
            ghosts.append(frame)
        max_pose_error = max(max_pose_error, max(abs(v-originals[frame]["camera_transform"][k])
                                               for k, v in row["camera_transform"].items()))
    assert not missing and not ghosts and max_pose_error < 1e-4
    video = integration / "he_replay.mp4"
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 80
    for frame in (0, 40, 55, 60, 79):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        assert ok and image.std() > 5
        if frame == 55:
            cv2.imwrite(str(integration / "verified_frame_55.jpg"), image)
    cap.release()
    report["recorder_integration"] = {"frames": 80, "GT_visible_frames": len(scores),
        "mean_visible_mask_iou": float(np.mean(scores)), "dropouts": missing,
        "ghosts": ghosts, "max_camera_pose_component_error": max_pose_error,
        "mask_threshold_note": "recorder uses reconstructed nonzero alpha, unlike sweep alpha >10/255"}
    report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
    (ROOT / "artifacts/passby_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "code_sha256"}, indent=2))


if __name__ == "__main__":
    main()
