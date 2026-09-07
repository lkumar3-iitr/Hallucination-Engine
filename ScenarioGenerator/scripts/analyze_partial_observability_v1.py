from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def intersection_fraction(target: dict | None, occluder: dict | None) -> float:
    if target is None or occluder is None:
        return 0.0
    target_area = max(0.0, target["x2"] - target["x1"]) * max(
        0.0, target["y2"] - target["y1"]
    )
    if target_area <= 0.0:
        return 0.0
    width = max(
        0.0,
        min(target["x2"], occluder["x2"])
        - max(target["x1"], occluder["x1"]),
    )
    height = max(
        0.0,
        min(target["y2"], occluder["y2"])
        - max(target["y1"], occluder["y1"]),
    )
    return min(1.0, width * height / target_area)


def actor_map(row: dict, source: str) -> dict[str, dict]:
    actors = row["actors"]
    if source == "carla":
        return {actor["actor_id"]: actor for actor in actors}
    return {actor["actor_id"]: actor for actor in actors}


def actor_box(actor: dict, source: str) -> dict | None:
    if source == "carla":
        return actor["bbox"]
    return actor["he_metadata"]["rendered_alpha_bbox"]


def actor_depth(actor: dict, source: str) -> float:
    if source == "carla":
        return float(actor["bbox"]["depth_m"])
    return float(actor["distance_forward_m"])


def measurements(
    rows: list[dict], source: str, target_id: str, occluder_id: str
) -> list[dict]:
    result = []
    for row in rows:
        actors = actor_map(row, source)
        target = actors[target_id]
        occluder = actors[occluder_id]
        target_depth = actor_depth(target, source)
        occluder_depth = actor_depth(occluder, source)
        target_box = actor_box(target, source)
        occluder_box = actor_box(occluder, source)
        fraction = 0.0
        if target_box is not None and occluder_depth < target_depth:
            fraction = intersection_fraction(target_box, occluder_box)
        visible_fraction = 0.0 if target_box is None else 1.0 - fraction
        frame_idx = (
            row["recorded_frame_idx"]
            if source == "carla"
            else row["scenario_frame"]
        )
        result.append(
            {
                "frame_idx": int(frame_idx),
                "t_s": float(row["t_s"]),
                "occluded_fraction": float(fraction),
                "visible_fraction": float(visible_fraction),
                "target_in_view": target_box is not None,
                "occluder_in_view": occluder_box is not None,
                "target_depth_m": target_depth,
                "occluder_depth_m": occluder_depth,
            }
        )
    return result


def first_frame_at_or_above(rows: list[dict], visible_fraction: float):
    for row in rows:
        if row["visible_fraction"] + 1e-12 >= visible_fraction:
            return {"frame_idx": row["frame_idx"], "t_s": row["t_s"]}
    return None


def summarize(rows: list[dict]) -> dict:
    minimum = min(rows, key=lambda row: row["visible_fraction"])
    maximum = max(rows, key=lambda row: row["occluded_fraction"])
    return {
        "frames": len(rows),
        "target_in_view_frames": sum(row["target_in_view"] for row in rows),
        "initial_occluded_fraction": rows[0]["occluded_fraction"],
        "terminal_visible_fraction": rows[-1]["visible_fraction"],
        "minimum_visible_fraction": {
            "fraction": minimum["visible_fraction"],
            "frame_idx": minimum["frame_idx"],
            "t_s": minimum["t_s"],
        },
        "maximum_occluded_fraction": {
            "fraction": maximum["occluded_fraction"],
            "frame_idx": maximum["frame_idx"],
            "t_s": maximum["t_s"],
        },
        "first_5_percent_visible": first_frame_at_or_above(rows, 0.05),
        "first_50_percent_visible": first_frame_at_or_above(rows, 0.50),
        "first_95_percent_visible": first_frame_at_or_above(rows, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-ground-truth", type=Path, required=True)
    parser.add_argument("--he-frames", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--occluder-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    carla = measurements(
        load_jsonl(args.carla_ground_truth),
        "carla",
        args.target_id,
        args.occluder_id,
    )
    he = measurements(
        load_jsonl(args.he_frames), "he", args.target_id, args.occluder_id
    )
    carla_by_frame = {row["frame_idx"]: row for row in carla}
    he_by_frame = {row["frame_idx"]: row for row in he}
    common_frames = sorted(set(carla_by_frame) & set(he_by_frame))

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "occlusion_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_idx",
                "t_s",
                "carla_occluded_fraction",
                "carla_visible_fraction",
                "he_occluded_fraction",
                "he_visible_fraction",
                "absolute_visible_fraction_error",
            ],
        )
        writer.writeheader()
        for frame_idx in common_frames:
            c = carla_by_frame[frame_idx]
            h = he_by_frame[frame_idx]
            writer.writerow(
                {
                    "frame_idx": frame_idx,
                    "t_s": c["t_s"],
                    "carla_occluded_fraction": c["occluded_fraction"],
                    "carla_visible_fraction": c["visible_fraction"],
                    "he_occluded_fraction": h["occluded_fraction"],
                    "he_visible_fraction": h["visible_fraction"],
                    "absolute_visible_fraction_error": abs(
                        c["visible_fraction"] - h["visible_fraction"]
                    ),
                }
            )

    report = {
        "schema": "partial_observability_analysis_v1",
        "metric": "axis_aligned_projected_box_overlap_proxy",
        "target_id": args.target_id,
        "occluder_id": args.occluder_id,
        "common_frames": len(common_frames),
        "carla": summarize(carla),
        "he": summarize(he),
        "limitations": [
            "Projected box overlap is not visible instance-mask overlap.",
            "Silhouette gaps and self-occlusion are not represented.",
        ],
        "timeseries_csv": str(csv_path.resolve()),
    }
    report_path = args.output_root / "occlusion_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
