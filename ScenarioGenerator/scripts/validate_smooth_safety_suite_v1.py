from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
COMMON_DIR = REPO_ROOT / "driving_models" / "common"
for path in (PROJECT_ROOT, COMMON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from he_asset_registry_v1 import HEAssetRegistry
from oriented_box_metrics_v1 import rectangle_clearance, rectangle_corners
from scenario_generator.schema.resolved_schema_v2 import ResolvedScenarioV2


def dimensions(actor, registry):
    if actor.dimensions_m is not None:
        return actor.dimensions_m
    asset = registry.resolve(actor.asset_key)
    if asset.physical_bbox is None:
        raise ValueError(f"No physical dimensions for {actor.asset_key}")
    return asset.physical_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.5)
    parser.add_argument("--ego-asset-key", default="vehicle.passenger_01")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    suite = json.loads(args.suite_manifest.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.suite_manifest.parent / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = HEAssetRegistry(args.asset_root)
    ego_asset = registry.resolve(args.ego_asset_key)
    if ego_asset.physical_bbox is None:
        raise ValueError("Ego asset has no physical dimensions.")
    ego_dims = ego_asset.physical_bbox
    reports = []
    failures = []
    for case in suite["cases"]:
        resolved_path = PROJECT_ROOT / case["resolved"]
        resolved = ResolvedScenarioV2.model_validate(
            json.loads(resolved_path.read_text(encoding="utf-8"))
        )
        ego_by_frame = {frame.frame_idx: frame for frame in resolved.ego_frames}
        info = {actor.actor_id: actor for actor in resolved.actors}
        actor_by_frame = {}
        for frame in resolved.actor_frames:
            actor_by_frame.setdefault(frame.frame_idx, {})[frame.actor_id] = frame
        pairs = [("ego", actor_id) for actor_id in sorted(info)]
        actor_ids = sorted(info)
        pairs.extend(
            (actor_ids[first], actor_ids[second])
            for first in range(len(actor_ids))
            for second in range(first + 1, len(actor_ids))
        )
        for first_id, second_id in pairs:
            minimum = None
            overlap_frames = []
            for frame_idx, ego_frame in ego_by_frame.items():
                actors = actor_by_frame.get(frame_idx, {})
                if second_id not in actors or (first_id != "ego" and first_id not in actors):
                    continue
                if first_id == "ego":
                    first_frame = ego_frame
                    first_dims = ego_dims
                else:
                    first_frame = actors[first_id]
                    first_dims = dimensions(info[first_id], registry)
                second_frame = actors[second_id]
                second_dims = dimensions(info[second_id], registry)
                first_box = rectangle_corners(
                    first_frame.x_m, first_frame.y_m, first_frame.yaw_deg,
                    first_dims.length_m, first_dims.width_m,
                )
                second_box = rectangle_corners(
                    second_frame.x_m, second_frame.y_m, second_frame.yaw_deg,
                    second_dims.length_m, second_dims.width_m,
                )
                clearance_m, overlaps = rectangle_clearance(first_box, second_box)
                sample = {
                    "frame_idx": frame_idx,
                    "t_s": float(ego_frame.t_s),
                    "clearance_m": float(clearance_m),
                }
                if minimum is None or sample["clearance_m"] < minimum["clearance_m"]:
                    minimum = sample
                if overlaps:
                    overlap_frames.append(frame_idx)
            passed = (
                minimum is not None
                and not overlap_frames
                and minimum["clearance_m"] + 1e-9 >= args.minimum_clearance_m
            )
            report = {
                "scenario_id": resolved.scenario_id,
                "first_id": first_id,
                "second_id": second_id,
                "status": "PASS" if passed else "FAIL",
                "minimum_clearance_m": minimum["clearance_m"] if minimum else None,
                "minimum_frame_idx": minimum["frame_idx"] if minimum else None,
                "minimum_t_s": minimum["t_s"] if minimum else None,
                "overlap_frame_count": len(overlap_frames),
                "first_overlap_frame": overlap_frames[0] if overlap_frames else None,
            }
            reports.append(report)
            if not passed:
                failures.append(report)
    csv_path = output_dir / "physical_clearance.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    summary = {
        "schema": "smooth_safety_suite_validation_v1",
        "suite_id": suite["suite_id"],
        "status": "PASS" if not failures else "FAIL",
        "minimum_clearance_required_m": args.minimum_clearance_m,
        "pair_checks": len(reports),
        "failed_checks": len(failures),
        "failures": failures,
        "csv": str(csv_path.resolve()),
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
