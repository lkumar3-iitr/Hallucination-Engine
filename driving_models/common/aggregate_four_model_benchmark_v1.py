"""Create paper-ready condition, paired, frame, and model-CI CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


MODELS = ("tcp", "neat", "cilpp", "aimmt")


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str, default=None):
    value = str(row.get(key, "")).strip()
    if not value:
        return default
    try:
        result = float(value)
    except ValueError:
        return default
    return result if math.isfinite(result) else default


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def mean(values):
    values = finite(values)
    return statistics.fmean(values) if values else math.nan


def percentile(values, q):
    values = sorted(finite(values))
    if not values:
        return math.nan
    position = (len(values) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def first_after(rows, event_s, predicate):
    for row in rows:
        t_s = number(row, "t_s")
        if t_s is not None and t_s + 1e-12 >= event_s and predicate(row):
            return row
    return None


def event_time_s(rows, case):
    for row in rows:
        if str(row.get("trigger_relative_frame", "")).strip() == "0":
            value = number(row, "t_s")
            if value is not None:
                return value
    return float(case["event_start_s"])


def metric_enabled(case, metric, default=True):
    applicability = case.get("metric_applicability") or {}
    return bool(applicability.get(metric, default))


def ttc_number(row, case=None):
    """Return only operationally meaningful finite route TTC values."""
    if case is not None and not metric_enabled(case, "route_ttc", True):
        return None
    lateral_categories = {
        "crossing_vehicle",
        "crossing_bus",
        "pedestrian_crossing",
        "occluded_pedestrian",
    }
    if case is not None and case.get("category") in lateral_categories:
        return None
    value = number(row, "route_ttc_s")
    if value is None or value < 0.0 or value > 60.0:
        return None
    return value


def condition_summary(rows, case, model, condition):
    event_s = event_time_s(rows, case)
    speeds = [number(row, "ego_speed_mps") for row in rows]
    accelerations = [number(row, "ego_acceleration_mps2") for row in rows]
    jerks = [number(row, "ego_jerk_mps3") for row in rows]
    clearances = [number(row, "nearest_actor_clearance_m") for row in rows]
    center_distances = [number(row, "nearest_actor_center_distance_m") for row in rows]
    deviations = [number(row, "route_deviation_m") for row in rows]
    progress = [number(row, "ego_route_progress_m") for row in rows]
    ttc = [ttc_number(row, case) for row in rows]
    brake = [number(row, "brake", 0.0) for row in rows]
    overlap_rows = [row for row in rows if (number(row, "physical_overlap", 0.0) or 0.0) > 0.5]
    expected_collision = case.get("expected_outcome") == "collision_required"
    outcome_matched = bool(overlap_rows) == expected_collision
    first_brake = first_after(
        rows, event_s, lambda row: (number(row, "brake", 0.0) or 0.0) > 0.1
    )
    first_collision = overlap_rows[0] if overlap_rows else None
    valid_progress = finite(progress)
    expected_frames = int(round(float(case.get("duration_s", 10.0)) * float(case.get("fps", 20.0)))) + 1
    frame_indices = [int(number(row, "scenario_frame", -1)) for row in rows]
    event_window_complete = bool(
        rows
        and frame_indices == list(range(frame_indices[0], frame_indices[0] + len(rows)))
    )
    trigger_required = case.get("trigger_route_progress_m") is not None
    trigger_observed = any(
        str(row.get("trigger_relative_frame", "")).strip() == "0"
        for row in rows
    )
    event_observed = trigger_observed if trigger_required else bool(rows)
    return {
        "scenario_id": case["scenario_id"],
        "category": case["category"],
        "model": model,
        "condition": condition,
        "expected_outcome": case.get("expected_outcome", "unspecified"),
        "metric_applicability_json": json.dumps(
            case.get("metric_applicability", {}), sort_keys=True
        ),
        "route_ttc_applicable": int(metric_enabled(case, "route_ttc", True)),
        "conflict_point_timing_applicable": int(
            metric_enabled(case, "conflict_point_timing", False)
        ),
        "occlusion_applicable": int(metric_enabled(case, "occlusion", False)),
        "comfort_applicable": int(metric_enabled(case, "comfort", True)),
        "frames": len(rows),
        "expected_frames": expected_frames,
        "completed": int(len(rows) == expected_frames),
        "event_window_complete": int(event_window_complete),
        "event_observed": int(event_observed),
        "duration_s": number(rows[-1], "t_s", 0.0) if rows else 0.0,
        "collision": int(bool(overlap_rows)),
        "collision_frames": len(overlap_rows),
        "first_collision_t_s": number(first_collision, "t_s", math.nan) if first_collision else math.nan,
        "minimum_actor_clearance_m": min(finite(clearances), default=math.nan),
        "minimum_actor_center_distance_m": min(finite(center_distances), default=math.nan),
        "minimum_finite_ttc_s": min(finite(ttc), default=math.nan),
        "route_progress_m": (max(valid_progress) - min(valid_progress)) if valid_progress else math.nan,
        "route_deviation_mean_m": mean(deviations),
        "route_deviation_p95_m": percentile(deviations, 0.95),
        "route_deviation_max_m": max(finite(deviations), default=math.nan),
        "speed_mean_mps": mean(speeds),
        "speed_min_mps": min(finite(speeds), default=math.nan),
        "speed_max_mps": max(finite(speeds), default=math.nan),
        "acceleration_mean_mps2": mean(accelerations),
        "maximum_deceleration_mps2": min(finite(accelerations), default=math.nan),
        "maximum_abs_jerk_mps3": (
            max((abs(value) for value in finite(jerks)), default=math.nan)
            if metric_enabled(case, "comfort", True) else math.nan
        ),
        "jerk_rms_mps3": (
            math.sqrt(mean([value * value for value in finite(jerks)]))
            if metric_enabled(case, "comfort", True) else math.nan
        ),
        "steer_abs_mean": mean([abs(number(row, "steer", 0.0) or 0.0) for row in rows]),
        "throttle_mean": mean([number(row, "throttle") for row in rows]),
        "brake_mean": mean(brake),
        "maximum_brake": max(finite(brake), default=math.nan),
        "braking_fraction": mean([float(value > 0.1) for value in finite(brake)]),
        "braking_onset_t_s": number(first_brake, "t_s", math.nan) if first_brake else math.nan,
        "reaction_time_s": (
            number(first_brake, "t_s") - event_s if first_brake else math.nan
        ),
        "success": int(len(rows) == expected_frames and outcome_matched),
        "event_window_success": int(
            event_window_complete and event_observed and outcome_matched
        ),
    }


PAIR_FIELDS = (
    "ego_speed_mps",
    "ego_acceleration_mps2",
    "ego_jerk_mps3",
    "steer",
    "throttle",
    "brake",
    "route_deviation_m",
    "ego_route_progress_m",
    "nearest_actor_clearance_m",
    "route_ttc_s",
)


def paired_rows(carla_rows, he_rows, case, model):
    carla = {int(number(row, "scenario_frame")): row for row in carla_rows}
    he = {int(number(row, "scenario_frame")): row for row in he_rows}
    output = []
    for frame_idx in sorted(set(carla) & set(he)):
        c_row = carla[frame_idx]
        h_row = he[frame_idx]
        row = {
            "scenario_id": case["scenario_id"],
            "category": case["category"],
            "model": model,
            "scenario_frame": frame_idx,
            "t_s": number(c_row, "t_s"),
        }
        cx, cy = number(c_row, "ego_x"), number(c_row, "ego_y")
        hx, hy = number(h_row, "ego_x"), number(h_row, "ego_y")
        row["ego_xy_error_m"] = (
            math.hypot(cx - hx, cy - hy)
            if None not in (cx, cy, hx, hy)
            else math.nan
        )
        for field in PAIR_FIELDS:
            if field == "ego_jerk_mps3" and not metric_enabled(case, "comfort", True):
                carla_value = None
                he_value = None
            elif field == "route_ttc_s":
                carla_value = ttc_number(c_row, case)
                he_value = ttc_number(h_row, case)
            else:
                carla_value = number(c_row, field)
                he_value = number(h_row, field)
            row[f"carla_{field}"] = carla_value if carla_value is not None else math.nan
            row[f"he_{field}"] = he_value if he_value is not None else math.nan
            row[f"abs_error_{field}"] = (
                abs(carla_value - he_value)
                if carla_value is not None and he_value is not None
                else math.nan
            )
        output.append(row)
    return output


def paired_summary(rows, carla_summary, he_summary, case, model):
    result = {
        "scenario_id": case["scenario_id"],
        "category": case["category"],
        "model": model,
        "common_frames": len(rows),
        "ego_xy_error_mean_m": mean([row["ego_xy_error_m"] for row in rows]),
        "ego_xy_error_p95_m": percentile([row["ego_xy_error_m"] for row in rows], 0.95),
        "ego_xy_error_max_m": max(finite([row["ego_xy_error_m"] for row in rows]), default=math.nan),
        "collision_agreement": int(carla_summary["collision"] == he_summary["collision"]),
        "collision_carla": carla_summary["collision"],
        "collision_he": he_summary["collision"],
        "minimum_clearance_delta_m": abs(carla_summary["minimum_actor_clearance_m"] - he_summary["minimum_actor_clearance_m"]),
        "braking_onset_delta_s": abs(carla_summary["braking_onset_t_s"] - he_summary["braking_onset_t_s"]),
        "reaction_time_delta_s": abs(carla_summary["reaction_time_s"] - he_summary["reaction_time_s"]),
        "success_agreement": int(carla_summary["success"] == he_summary["success"]),
        "event_window_success_agreement": int(
            carla_summary["event_window_success"]
            == he_summary["event_window_success"]
        ),
        "event_window_success_carla": carla_summary["event_window_success"],
        "event_window_success_he": he_summary["event_window_success"],
    }
    for field in PAIR_FIELDS:
        result[f"{field}_mae"] = mean([row[f"abs_error_{field}"] for row in rows])
    return result


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ci_rows(pair_summaries, models):
    metrics = [
        "ego_xy_error_mean_m",
        "ego_speed_mps_mae",
        "ego_acceleration_mps2_mae",
        "ego_jerk_mps3_mae",
        "steer_mae",
        "brake_mae",
        "route_deviation_m_mae",
        "ego_route_progress_m_mae",
        "nearest_actor_clearance_m_mae",
    ]
    output = []
    for model in models:
        model_rows = [row for row in pair_summaries if row["model"] == model]
        for metric in metrics:
            values = finite([row.get(metric) for row in model_rows])
            average = mean(values)
            sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) >= 2 else math.nan
            half_width = 1.96 * sem if math.isfinite(sem) else math.nan
            output.append(
                {
                    "model": model,
                    "metric": metric,
                    "scenario_count": len(values),
                    "mean": average,
                    "ci95_low": average - half_width if math.isfinite(half_width) else math.nan,
                    "ci95_high": average + half_width if math.isfinite(half_width) else math.nan,
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    suite = json.loads(args.suite_manifest.read_text(encoding="utf-8"))
    configured_models = tuple(suite.get("models") or MODELS)
    models = tuple(
        model for model in configured_models
        if any(
            (args.run_root / case["scenario_id"] / model).exists()
            for case in suite["cases"]
        )
    )
    output_dir = args.output_dir or args.run_root / "paper_tables"
    condition_rows = []
    pair_rows = []
    frame_rows = []
    missing = []
    for case in suite["cases"]:
        case = {**suite, **case}
        for model in models:
            model_dir = args.run_root / case["scenario_id"] / model
            paths = {
                condition: model_dir / f"{case['scenario_id']}_{model}_{condition}.csv"
                for condition in ("carla", "he")
            }
            if not all(path.exists() for path in paths.values()):
                missing.append({"scenario_id": case["scenario_id"], "model": model})
                continue
            raw = {condition: load_csv(path) for condition, path in paths.items()}
            summaries = {
                condition: condition_summary(raw[condition], case, model, condition)
                for condition in ("carla", "he")
            }
            condition_rows.extend(summaries.values())
            paired = paired_rows(raw["carla"], raw["he"], case, model)
            frame_rows.extend(paired)
            pair_rows.append(paired_summary(paired, summaries["carla"], summaries["he"], case, model))
    write_csv(output_dir / "condition_metrics.csv", condition_rows)
    write_csv(output_dir / "paired_metrics.csv", pair_rows)
    write_csv(output_dir / "paired_frame_metrics.csv", frame_rows)
    completed_models = tuple(
        model for model in models
        if any(row["model"] == model for row in pair_rows)
    )
    write_csv(output_dir / "model_paired_ci95.csv", ci_rows(pair_rows, completed_models))
    (output_dir / "aggregation_status.json").write_text(
        json.dumps(
            {
                "suite_id": suite["suite_id"],
                "complete_pairs": len(pair_rows),
                "expected_pairs": len(suite["cases"]) * len(models),
                "missing": missing,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Aggregated {len(pair_rows)} model-scenario pairs into {output_dir}")


if __name__ == "__main__":
    main()
