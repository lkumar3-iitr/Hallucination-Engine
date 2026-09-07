from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
COMMON_DIR = REPO_ROOT / "driving_models" / "common"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(COMMON_DIR))

from he_asset_registry_v1 import HEAssetRegistry
from oriented_box_metrics_v1 import rectangle_clearance, rectangle_corners
from scenario_generator.schema.resolved_schema_v2 import ResolvedScenarioV2


def actor_dimensions(actor_info, registry):
    if actor_info.dimensions_m is not None:
        return actor_info.dimensions_m
    asset = registry.resolve(actor_info.asset_key)
    if asset.physical_bbox is None:
        raise ValueError(f"Asset {actor_info.asset_key!r} has no physical bbox.")
    return asset.physical_bbox


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ego/actor and actor/actor footprint clearance."
    )
    parser.add_argument("resolved_json")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--ego-asset-key", default="vehicle.passenger_01")
    parser.add_argument("--minimum-clearance-m", type=float, default=1.0)
    parser.add_argument("--minimum-bus-clearance-m", type=float, default=1.5)
    parser.add_argument("--include-ego", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scenario = ResolvedScenarioV2.model_validate_json(
        Path(args.resolved_json).read_text(encoding="utf-8")
    )
    registry = HEAssetRegistry(args.asset_root)
    ego_asset = registry.resolve(args.ego_asset_key)
    if ego_asset.physical_bbox is None:
        raise ValueError(f"Ego asset {args.ego_asset_key!r} has no physical bbox.")

    actors_by_id = {actor.actor_id: actor for actor in scenario.actors}
    dimensions = {
        actor.actor_id: actor_dimensions(actor, registry)
        for actor in scenario.actors
    }
    if args.include_ego:
        dimensions["ego"] = ego_asset.physical_bbox
    actor_frames = {}
    for frame in scenario.actor_frames:
        actor_frames.setdefault(frame.frame_idx, {})[frame.actor_id] = frame
    ego_frames = {frame.frame_idx: frame for frame in scenario.ego_frames}

    minimum = None
    failures = []
    overlap_frames = []
    pair_summaries = {}
    frame_indices = sorted(actor_frames)
    if args.include_ego:
        frame_indices = sorted(set(frame_indices) & set(ego_frames))
    for frame_idx in frame_indices:
        present = dict(actor_frames.get(frame_idx, {}))
        if args.include_ego:
            present["ego"] = ego_frames[frame_idx]
        for first_id, second_id in combinations(sorted(present), 2):
            first = present[first_id]
            second = present[second_id]
            first_dims = dimensions[first_id]
            second_dims = dimensions[second_id]
            first_box = rectangle_corners(
                first.x_m, first.y_m, first.yaw_deg,
                first_dims.length_m, first_dims.width_m,
            )
            second_box = rectangle_corners(
                second.x_m, second.y_m, second.yaw_deg,
                second_dims.length_m, second_dims.width_m,
            )
            clearance_m, overlaps = rectangle_clearance(first_box, second_box)
            pair_key = f"{first_id}__{second_id}"
            sample = {
                "frame_idx": int(frame_idx),
                "t_s": float(getattr(first, "t_s", frame_idx / scenario.fps)),
                "pair": pair_key,
                "clearance_m": float(clearance_m),
                "overlaps": bool(overlaps),
            }
            if minimum is None or sample["clearance_m"] < minimum["clearance_m"]:
                minimum = sample
            current = pair_summaries.get(pair_key)
            if current is None or sample["clearance_m"] < current["clearance_m"]:
                pair_summaries[pair_key] = sample
            if overlaps:
                overlap_frames.append(sample)

    for pair, sample in sorted(pair_summaries.items()):
        threshold = float(args.minimum_clearance_m)
        if "bus" in pair:
            threshold = max(threshold, float(args.minimum_bus_clearance_m))
        if sample["clearance_m"] + 1e-9 < threshold:
            failures.append(
                f"{pair}: minimum clearance {sample['clearance_m']:.3f} m "
                f"below {threshold:.3f} m at frame {sample['frame_idx']}"
            )

    report = {
        "schema": "resolved_all_pairs_clearance_validation_v1",
        "status": "PASS" if not failures and not overlap_frames else "FAIL",
        "scenario_id": scenario.scenario_id,
        "resolved_path": str(Path(args.resolved_json).resolve()),
        "minimum_clearance": minimum,
        "pair_minimum_clearances": pair_summaries,
        "minimum_clearance_required_m": float(args.minimum_clearance_m),
        "minimum_bus_clearance_required_m": float(args.minimum_bus_clearance_m),
        "include_ego": bool(args.include_ego),
        "overlap_frame_count": len(overlap_frames),
        "first_overlap": overlap_frames[0] if overlap_frames else None,
        "failures": failures,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
