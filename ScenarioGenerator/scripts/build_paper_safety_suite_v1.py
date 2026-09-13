"""Build the frozen 20-case paper suite: 15 safe cases and 5 collision controls."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import build_long_safety_critical_suite_v1 as base


OUTPUT = base.SG_ROOT / "outputs" / "paper_safety_suite_v1"


def metric_applicability(category: str, collision_required: bool = False):
    lateral = category in {
        "crossing_vehicle", "crossing_bus", "pedestrian_crossing",
        "occluded_pedestrian", "multi_pedestrian_crossing",
        "collision_positive_control",
    }
    return {
        "collision": True,
        "physical_clearance": True,
        "braking_response": True,
        "trajectory_tracking": True,
        "route_ttc": not lateral,
        "conflict_point_timing": lateral,
        "occlusion": category in {
            "partial_observability_cutout", "occluded_pedestrian"
        },
        "comfort": not collision_required,
        "renderer_continuity": category in {
            "cutin_then_brake", "partial_observability_cutout",
            "crossing_vehicle", "crossing_bus", "occluded_pedestrian",
            "collision_positive_control",
        },
    }


def make_case(
    scenario_id, category, metric_actor_id, event_start_s,
    trigger_route_progress_m, actors, frames, description,
    expected_outcome="collision_free", pre_trigger_source_frame=None,
):
    collision_required = expected_outcome == "collision_required"
    return {
        "scenario_id": scenario_id,
        "category": category,
        "metric_actor_id": metric_actor_id,
        "event_start_s": float(event_start_s),
        "event_source_start_s": float(event_start_s),
        "trigger_route_progress_m": trigger_route_progress_m,
        "pre_trigger_source_frame": pre_trigger_source_frame,
        "actors": actors,
        "frames": frames,
        "description": description,
        "expected_outcome": expected_outcome,
        "metric_applicability": metric_applicability(
            category, collision_required=collision_required
        ),
    }


def collision_crossing(
    rf, scenario_id, actor_id, asset_key, conflict_s,
    trigger_s, event_s, start_left, end_left,
):
    # At event_s the actor is exactly at lane center. Frame event-1 stages it
    # off center; the route trigger advances to the center frame in one tick.
    frames = base.crossing(
        rf, actor_id, conflict_s,
        event_s - 0.05, event_s + 0.05,
        start_left, end_left, 90.0 if end_left > start_left else -90.0,
    )
    return make_case(
        scenario_id=scenario_id,
        category="collision_positive_control",
        metric_actor_id=actor_id,
        event_start_s=event_s,
        trigger_route_progress_m=trigger_s,
        actors=[base.actor_definition(actor_id, asset_key, spawn_s=event_s - 0.05)],
        frames=[frames],
        description=(
            "Predeclared unavoidable collision-positive control. The actor "
            "enters the conflict center on the first trigger-relative frame."
        ),
        expected_outcome="collision_required",
        pre_trigger_source_frame=int(round(event_s * base.FPS)) - 1,
    )


def build_cases(rf):
    cases = []
    for case in base.build_cases(rf):
        copied = copy.deepcopy(case)
        if copied["scenario_id"] == "long_09_stopped_queue_multi_actor_001":
            copied["frames"][1] = base.static_route(rf, "queue_front", 92.0)
            copied["description"] = (
                "A Tesla brakes at a collision-free queue tail with a bus "
                "stopped farther ahead."
            )
        copied["expected_outcome"] = "collision_free"
        copied["pre_trigger_source_frame"] = None
        copied["metric_applicability"] = metric_applicability(copied["category"])
        cases.append(copied)

    cases.extend([
        make_case(
            "paper_11_follow_tesla_brake_001", "lead_emergency_brake",
            "lead_tesla_2", 12.0, 62.0,
            [base.actor_definition("lead_tesla_2", "vehicle.passenger_01")],
            [base.route_motion(rf, "lead_tesla_2", 12.0,
                               base.brake_profile(6.2, 12.0, 2.8))],
            "Tesla lead performs a controlled close-range stop.",
        ),
        make_case(
            "paper_12_crossing_tesla_reverse_001", "crossing_vehicle",
            "crossing_tesla", 11.5, 60.0,
            [base.actor_definition("crossing_tesla", "vehicle.passenger_01")],
            [base.crossing(rf, "crossing_tesla", 76.0, 11.5, 15.5,
                           7.0, -7.0, -90.0)],
            "Tesla crosses from the opposite side with a route trigger.",
        ),
        make_case(
            "paper_13_pedestrian_reverse_001", "pedestrian_crossing",
            "reverse_pedestrian", 12.0, 59.0,
            [base.actor_definition("reverse_pedestrian", "pedestrian.person_01")],
            [base.crossing(rf, "reverse_pedestrian", 73.0, 12.0, 15.5,
                           4.5, -4.5, -90.0)],
            "Pedestrian crosses from right to left at a controlled conflict point.",
        ),
        make_case(
            "paper_14_two_pedestrians_001", "multi_pedestrian_crossing",
            "pedestrian_a", 11.0, 57.0,
            [
                base.actor_definition("pedestrian_a", "pedestrian.person_01"),
                base.actor_definition("pedestrian_b", "pedestrian.person_01"),
            ],
            [
                base.crossing(rf, "pedestrian_a", 72.0, 11.0, 14.0,
                              -4.5, 4.5, 90.0),
                base.crossing(rf, "pedestrian_b", 78.0, 13.5, 16.5,
                              4.5, -4.5, -90.0),
            ],
            "Two pedestrians cross sequentially at separated conflict points.",
        ),
        make_case(
            "paper_15_offset_stopped_hazards_001", "stopped_queue",
            "stopped_nissan", 10.0, 60.0,
            [
                base.actor_definition("stopped_nissan", "vehicle.passenger_02"),
                base.actor_definition("offset_bus", "vehicle.bus_01"),
            ],
            [
                base.static_route(rf, "stopped_nissan", 82.0),
                base.static_route(rf, "offset_bus", 95.0, -4.2),
            ],
            "Stopped in-lane Nissan with a spatially separated offset bus.",
        ),
    ])

    cases.extend([
        collision_crossing(rf, "paper_16_collision_nissan_001", "collision_nissan",
                           "vehicle.passenger_02", 65.0, 64.0, 12.0, -5.0, 5.0),
        collision_crossing(rf, "paper_17_collision_tesla_001", "collision_tesla",
                           "vehicle.passenger_01", 70.0, 69.0, 12.0, 5.0, -5.0),
        collision_crossing(rf, "paper_18_collision_bus_001", "collision_bus",
                           "vehicle.bus_01", 78.0, 76.0, 12.0, -7.0, 7.0),
        collision_crossing(rf, "paper_19_collision_pedestrian_001", "collision_pedestrian",
                           "pedestrian.person_01", 68.0, 68.0, 12.0, -2.0, 2.0),
        collision_crossing(rf, "paper_20_collision_pedestrian_reverse_001",
                           "collision_pedestrian_reverse", "pedestrian.person_01",
                           75.0, 75.0, 12.0, 2.0, -2.0),
    ])
    return cases


def main():
    route = base.parse_route_rows(base.load_csv(base.ROUTE_CSV))
    rf = base.RouteFrame(route)
    base_env = json.loads(base.BASE_ENV.read_text(encoding="utf-8"))
    manifest_cases = []
    for case in build_cases(rf):
        sid = case["scenario_id"]
        resolved = {
            "schema_version": "2.0-resolved",
            "source_schema_version": "2.0",
            "scenario_id": sid,
            "source_description": case["description"],
            "duration_s": base.DURATION_S,
            "fps": base.FPS,
            "seed": 20260901,
            "coordinate_frame": "ego_initial",
            "camera": {
                "name": "front", "width_px": 1280, "height_px": 720,
                "fov_deg": 90.0, "x_m": 1.5, "y_m": 0.0, "z_m": 1.6,
                "pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0,
            },
            "actors": case["actors"],
            "ego_frames": [
                {"frame_idx": i, "t_s": i * base.DT, "x_m": 0.0,
                 "y_m": 0.0, "yaw_deg": 0.0, "speed_mps": 0.0,
                 "vx_mps": 0.0, "vy_mps": 0.0}
                for i in range(int(base.DURATION_S * base.FPS) + 1)
            ],
            "actor_frames": [row for rows in case["frames"] for row in rows],
        }
        resolved_path = OUTPUT / "resolved" / f"{sid}.resolved_v2.json"
        base.save_json(resolved_path, resolved)
        environment = copy.deepcopy(base_env)
        environment["scenario_id"] = sid
        environment_path = OUTPUT / "environment" / f"{sid}.environment_v1.json"
        base.save_json(environment_path, environment)
        manifest_cases.append({
            "scenario_id": sid,
            "category": case["category"],
            "metric_actor_id": case["metric_actor_id"],
            "event_start_s": case["event_start_s"],
            "event_source_start_s": case["event_source_start_s"],
            "trigger_route_progress_m": case["trigger_route_progress_m"],
            "pre_trigger_source_frame": case.get("pre_trigger_source_frame"),
            "trigger_gated_actor_ids": case.get("trigger_gated_actor_ids", []),
            "expected_outcome": case["expected_outcome"],
            "duration_s": base.DURATION_S,
            "fps": base.FPS,
            "metric_applicability": case["metric_applicability"],
            "asset_keys": sorted({a["asset_key"] for a in case["actors"]}),
            "actor_ids": [a["actor_id"] for a in case["actors"]],
            "resolved": str(resolved_path.relative_to(base.SG_ROOT)),
            "resolved_sha256": base.sha256(resolved_path),
            "environment": str(environment_path.relative_to(base.SG_ROOT)),
            "status": "requires_smoke_and_visual_acceptance",
        })
    manifest = {
        "schema": "paper_safety_suite_manifest_v1",
        "suite_id": "paper_safety_suite_v1_15_safe_5_collision",
        "fps": base.FPS,
        "duration_s": base.DURATION_S,
        "scenario_count": len(manifest_cases),
        "expected_outcome_counts": {"collision_free": 15, "collision_required": 5},
        "models": ["tcp", "neat", "cilpp", "aimmt"],
        "conditions": ["carla", "he"],
        "route_csv": str(base.ROUTE_CSV.relative_to(base.ROOT)),
        "cases": manifest_cases,
    }
    base.save_json(OUTPUT / "suite_manifest.json", manifest)
    print(f"Built {len(manifest_cases)} paper scenarios in {OUTPUT}")


if __name__ == "__main__":
    main()
