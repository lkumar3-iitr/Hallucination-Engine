import pytest

from scenario_generator.planner.behavior_compiler_v2 import (
    SemanticBehaviorV2,
    compile_semantic_behavior,
    lane_relation_to_offset_m,
)
from scenario_generator.schema.scenario_schema_v2 import (
    Pose2DV2,
)
from scenario_generator.trajectory.trajectory_resolver_v2 import (
    resolve_sequence_state,
)


def test_lane_relation_offsets():
    assert lane_relation_to_offset_m(
        "ego_lane", 3.5
    ) == pytest.approx(0.0)

    assert lane_relation_to_offset_m(
        "left_adjacent", 3.5
    ) == pytest.approx(3.5)

    assert lane_relation_to_offset_m(
        "right_adjacent", 3.5
    ) == pytest.approx(-3.5)


def test_left_lane_and_return_compile():
    behavior = SemanticBehaviorV2.model_validate({
        "initial_lane": "ego_lane",
        "initial_speed_mps": 4.0,
        "steps": [
            {
                "action": "lane_change",
                "target_lane": "left_adjacent",
                "duration_s": 3.0,
            },
            {
                "action": "lane_change",
                "target_lane": "ego_lane",
                "duration_s": 3.0,
            },
        ],
    })

    motion = compile_semantic_behavior(
        behavior,
        lane_width_m=3.5,
    )

    assert motion.steps[0].lateral_delta_m == pytest.approx(
        3.5
    )

    assert motion.steps[1].lateral_delta_m == pytest.approx(
        -3.5
    )


def test_initial_left_lane_to_ego_lane():
    behavior = SemanticBehaviorV2.model_validate({
        "initial_lane": "left_adjacent",
        "initial_speed_mps": 4.0,
        "steps": [
            {
                "action": "lane_change",
                "target_lane": "ego_lane",
                "duration_s": 3.0,
            },
        ],
    })

    motion = compile_semantic_behavior(
        behavior,
        lane_width_m=3.5,
    )

    assert motion.steps[0].lateral_delta_m == pytest.approx(
        -3.5
    )


def test_invalid_lane_width_rejected():
    behavior = SemanticBehaviorV2.model_validate({
        "initial_speed_mps": 4.0,
        "steps": [
            {
                "action": "cruise",
                "duration_s": 1.0,
            },
        ],
    })

    with pytest.raises(ValueError):
        compile_semantic_behavior(
            behavior,
            lane_width_m=0.0,
        )


def test_compiler_to_resolver_integration():
    behavior = SemanticBehaviorV2.model_validate({
        "initial_lane": "ego_lane",
        "initial_speed_mps": 4.0,
        "steps": [
            {
                "action": "lane_change",
                "target_lane": "left_adjacent",
                "duration_s": 4.0,
            },
            {
                "action": "lane_change",
                "target_lane": "ego_lane",
                "duration_s": 4.0,
            },
        ],
    })

    motion = compile_semantic_behavior(
        behavior,
        lane_width_m=3.5,
    )

    spawn = Pose2DV2(
        x_m=0.0,
        y_m=0.0,
        yaw_deg=0.0,
    )

    left = resolve_sequence_state(
        spawn,
        motion,
        t_s=4.0,
        active_start_s=0.0,
    )

    back = resolve_sequence_state(
        spawn,
        motion,
        t_s=8.0,
        active_start_s=0.0,
    )

    assert left.y_m == pytest.approx(3.5)
    assert left.x_m == pytest.approx(16.0)

    assert back.y_m == pytest.approx(0.0)
    assert back.x_m == pytest.approx(32.0)