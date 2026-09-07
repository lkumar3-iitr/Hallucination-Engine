import pytest

from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedScenarioV2,
)
from scripts.validate_resolved_clearance_v1 import (
    rectangle_clearance,
    rectangle_corners,
)
from scripts.validate_schema_v2 import validate_resolved_structure


def make_resolved_scenario():
    frame_base = {
        "x_m": 0.0,
        "y_m": 0.0,
        "yaw_deg": 0.0,
        "speed_mps": 0.0,
        "vx_mps": 0.0,
        "vy_mps": 0.0,
    }
    return ResolvedScenarioV2.model_validate({
        "scenario_id": "resolved_validation_test",
        "duration_s": 0.05,
        "fps": 20,
        "camera": {},
        "actors": [{
            "actor_id": "adversary",
            "actor_type": "vehicle",
            "role": "adversary",
            "asset_key": "vehicle.passenger_02",
            "spawn_time_s": 0.0,
            "despawn_time_s": None,
        }],
        "ego_frames": [
            {"frame_idx": 0, "t_s": 0.0, **frame_base},
            {"frame_idx": 1, "t_s": 0.05, **frame_base},
        ],
        "actor_frames": [
            {
                "frame_idx": 0,
                "t_s": 0.0,
                "actor_id": "adversary",
                **frame_base,
            },
            {
                "frame_idx": 1,
                "t_s": 0.05,
                "actor_id": "adversary",
                **frame_base,
            },
        ],
    })


def test_resolved_structure_accepts_contiguous_consistent_frames():
    scenario = make_resolved_scenario()

    assert validate_resolved_structure(scenario) == 2


def test_resolved_structure_rejects_inconsistent_ego_time():
    scenario = make_resolved_scenario()
    scenario.ego_frames[1].t_s = 0.04

    with pytest.raises(ValueError, match="ego frame time mismatch"):
        validate_resolved_structure(scenario)


def test_resolved_structure_rejects_duplicate_actor_frame():
    scenario = make_resolved_scenario()
    scenario.actor_frames.append(scenario.actor_frames[-1].model_copy())

    with pytest.raises(ValueError, match="Duplicate resolved actor frame"):
        validate_resolved_structure(scenario)


def test_rectangle_clearance_uses_physical_footprints():
    ego = rectangle_corners(0.0, 0.0, 0.0, 4.0, 2.0)
    actor_with_one_metre_gap = rectangle_corners(
        5.0,
        0.0,
        0.0,
        4.0,
        2.0,
    )
    actor_overlapping = rectangle_corners(3.0, 0.0, 0.0, 4.0, 2.0)

    clearance_m, overlaps = rectangle_clearance(
        ego,
        actor_with_one_metre_gap,
    )
    assert clearance_m == pytest.approx(1.0)
    assert overlaps is False

    clearance_m, overlaps = rectangle_clearance(ego, actor_overlapping)
    assert clearance_m == pytest.approx(0.0)
    assert overlaps is True
