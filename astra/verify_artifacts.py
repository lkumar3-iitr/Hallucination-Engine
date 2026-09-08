"""Check recorded evidence counts, score claims, and video decodability."""
import csv
import hashlib
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent


def main():
    checks = {}
    for name, expected in (("evaluation_full_v1", 64), ("independent_evaluation_v1", 74)):
        folder = ROOT / "artifacts" / name
        with (folder / "rows.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        summary = json.loads((folder / "summary.json").read_text())
        assert len(rows) == summary["all"]["count"] == expected
        assert max(float(r["oracle_gap"]) for r in rows) < .02
        assert all(0 <= float(r["selected_iou"]) <= float(r["oracle_iou"]) <= 1 for r in rows)
        checks[name] = {"rows": len(rows), "max_oracle_gap": max(float(r["oracle_gap"]) for r in rows)}
    capture = ROOT / "artifacts/carla_independent_v4"
    assert json.loads((capture / "COMPLETE.json").read_text())["frames"] == 96
    assert len(list((capture / "masks").glob("*.png"))) == 96
    assert len(list((capture / "rgb").glob("*.png"))) == 96
    checks["videos"] = {}
    for name in ("preview_forward", "preview_close_yaw60", "preview_independent"):
        folder = ROOT / "artifacts" / name
        manifest = json.loads((folder / "preview_manifest.json").read_text())
        for path in folder.glob("*.mp4"):
            cap = cv2.VideoCapture(str(path))
            assert cap.isOpened(), str(path)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert count == len(manifest["frames"]), (path, count)
            for frame in (0, count//2, count-1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                assert ok and image.std() > 5, (path, frame)
                if path.stem == "clean_background_gt_box" and frame == count-1:
                    cv2.imwrite(str(folder / "clean_background_last.png"), image)
            cap.release()
            checks["videos"][str(path.relative_to(ROOT))] = count
    checks["current_code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in ROOT.glob("*.py")}
    (ROOT / "artifacts/verification.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
