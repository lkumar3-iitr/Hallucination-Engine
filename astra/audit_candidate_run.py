"""Audit camera consistency and coverage changes from the recorded candidate log."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .calibrated_close import coverage_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--camera-count", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if root not in args.run.resolve().parents:
        parser.error("Run must be inside astra")
    receipt = json.loads((args.run/"CANDIDATE_RUN.json").read_text())
    bank_path = Path(receipt["candidate_bank"])
    config = json.loads((bank_path/"config.json").read_text())
    bank = SimpleNamespace(axes=[[config["forward_min"], config["forward_max"]],
                                 sorted(config["right"]), sorted(config["up"])])
    records = [json.loads(line) for line in (args.run/"renderer_metadata.jsonl").read_text().splitlines()]
    if [r["render_call"] for r in records] != list(range(len(records))):
        raise ValueError("Missing or reordered render calls")
    if len(records) % args.camera_count:
        raise ValueError("Incomplete final camera group")
    modes, transitions = collections.Counter(), []
    disagreements, max_delta, previous = [], 0., None
    for start in range(0, len(records), args.camera_count):
        cameras = []
        for record in records[start:start+args.camera_count]:
            bus = next(a["metadata"] for a in record["actors"] if a["actor_id"] == "parked_bus")
            modes[bus["candidate_mode"]] += 1
            if "query_actor_local" in bus:
                max_delta = max(max_delta, abs(coverage_weight(bank, bus["query_actor_local"])-bus["close_weight"]))
            cameras.append(bus)
        current = cameras[0]["candidate_mode"]
        if current != previous:
            transitions.append({"frame": start//args.camera_count, "mode": current})
            previous = current
        if current != "native_fallback":
            reference = cameras[0].get("sources", [])
            for camera in cameras[1:]:
                other = camera.get("sources", [])
                if ([r["node"] for r in reference] != [r["node"] for r in other]
                        or not np.allclose([r["weight"] for r in reference], [r["weight"] for r in other], atol=1e-6)):
                    disagreements.append(start//args.camera_count)
    source_files = ["calibrated_close.py", "calibrated_bus_renderer.py", "calibrated_compositor.py", "hull_rays.py",
                    "capture_aimed_close.py", "run_calibrated_closed_loop.py", "evaluate_calibrated_close.py"]
    summary = {"frames": len(records)//args.camera_count, "camera_count": args.camera_count,
               "mode_counts_per_camera_call": dict(modes), "mode_transitions": transitions,
               "camera_source_disagreement_frames": sorted(set(disagreements)),
               "maximum_all_axis_guard_weight_change": max_delta,
               "current_code_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_files},
               "bank_csv_sha256": hashlib.sha256((bank_path/"view_matrix.csv").read_bytes()).hexdigest()}
    (args.run/"CONSISTENCY_AUDIT.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if disagreements or max_delta > 1e-9:
        raise ValueError("Camera consistency or recorded-run equivalence check failed")


if __name__ == "__main__":
    main()
