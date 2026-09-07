"""Validate paper-suite structure and optional completed campaign outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "driving_models" / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from oriented_box_metrics_v1 import rectangle_clearance, rectangle_corners


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def structural_validation(suite_path: Path, suite):
    failures = []
    counts = {"collision_free": 0, "collision_required": 0}
    minimum_actor_clearance = {}
    for case in suite["cases"]:
        expected = case.get("expected_outcome")
        if expected not in counts:
            failures.append(f"{case['scenario_id']}: invalid expected_outcome={expected!r}")
            continue
        counts[expected] += 1
        resolved_path = suite_path.parents[2] / case["resolved"]
        if not resolved_path.exists() or sha256(resolved_path) != case["resolved_sha256"]:
            failures.append(f"{case['scenario_id']}: resolved hash mismatch")
            continue
        data = json.loads(resolved_path.read_text(encoding="utf-8"))
        actor_info = {actor["actor_id"]: actor for actor in data["actors"]}
        states = {}
        for row in data["actor_frames"]:
            states.setdefault(int(row["frame_idx"]), {})[row["actor_id"]] = row
        minimum = float("inf")
        overlap_frames = 0
        for frame in states.values():
            for first_id, second_id in itertools.combinations(sorted(frame), 2):
                first, second = frame[first_id], frame[second_id]
                first_dims = actor_info[first_id]["dimensions_m"]
                second_dims = actor_info[second_id]["dimensions_m"]
                first_box = rectangle_corners(
                    first["x_m"], first["y_m"], first["yaw_deg"],
                    first_dims["length_m"], first_dims["width_m"],
                )
                second_box = rectangle_corners(
                    second["x_m"], second["y_m"], second["yaw_deg"],
                    second_dims["length_m"], second_dims["width_m"],
                )
                clearance, overlaps = rectangle_clearance(first_box, second_box)
                minimum = min(minimum, clearance)
                overlap_frames += int(overlaps)
        minimum_actor_clearance[case["scenario_id"]] = (
            None if minimum == float("inf") else minimum
        )
        if expected == "collision_free" and overlap_frames:
            failures.append(
                f"{case['scenario_id']}: {overlap_frames} authored actor-actor overlaps"
            )
        applicability = case.get("metric_applicability") or {}
        if not applicability.get("collision", False):
            failures.append(f"{case['scenario_id']}: collision metric must be applicable")
        if applicability.get("route_ttc") and applicability.get("conflict_point_timing"):
            failures.append(f"{case['scenario_id']}: route TTC and conflict timing both enabled")
    if counts != suite.get("expected_outcome_counts"):
        failures.append(
            f"outcome counts {counts} != {suite.get('expected_outcome_counts')}"
        )
    return failures, counts, minimum_actor_clearance


def campaign_validation(run_root: Path, suite):
    failures = []
    rows_out = []
    for case in suite["cases"]:
        expected_collision = case["expected_outcome"] == "collision_required"
        event_frame = int(round(float(case["event_source_start_s"]) * float(suite["fps"])))
        for model in suite["models"]:
            for condition in suite["conditions"]:
                path = (
                    run_root / case["scenario_id"] / model
                    / f"{case['scenario_id']}_{model}_{condition}.csv"
                )
                if not path.exists():
                    failures.append(f"missing {path}")
                    continue
                rows = read_csv(path)
                collision = any(row.get("physical_overlap") == "1" for row in rows)
                trigger_rows = [r for r in rows if r.get("trigger_relative_frame") == "0"]
                trigger_ok = bool(trigger_rows) if case.get("trigger_route_progress_m") is not None else True
                source_ok = all(
                    int(r["actor_source_frame"]) - event_frame
                    == int(r["trigger_relative_frame"])
                    for r in rows
                    if r.get("trigger_relative_frame", "") != ""
                    and int(r["trigger_relative_frame"]) >= 0
                )
                outcome_ok = collision == expected_collision
                if not (trigger_ok and source_ok and outcome_ok):
                    failures.append(
                        f"{case['scenario_id']}/{model}/{condition}: "
                        f"trigger={trigger_ok} source={source_ok} collision={collision} "
                        f"expected={expected_collision}"
                    )
                rows_out.append({
                    "scenario_id": case["scenario_id"], "model": model,
                    "condition": condition, "expected_collision": int(expected_collision),
                    "observed_collision": int(collision), "trigger_ok": int(trigger_ok),
                    "source_clock_ok": int(source_ok), "outcome_ok": int(outcome_ok),
                })
    return failures, rows_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    suite_path = args.suite_manifest.resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    failures, counts, clearances = structural_validation(suite_path, suite)
    campaign_rows = []
    if args.run_root is not None:
        campaign_failures, campaign_rows = campaign_validation(args.run_root.resolve(), suite)
        failures.extend(campaign_failures)
    result = {
        "valid": not failures,
        "scenario_count": len(suite["cases"]),
        "outcome_counts": counts,
        "minimum_authored_actor_clearance_m": clearances,
        "campaign_rows": campaign_rows,
        "failures": failures,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
