import math

import pytest

from scenario_generator.schema.scenario_schema_v2 import (
    Pose2DV2,
    SequenceMotionV2,
)
from scenario_generator.trajectory.trajectory_resolver_v2 import (
    resolve_sequence_state,
)


def make_sequence():
    return SequenceMotionV2.model_validate({
        "mode": "sequence",
        "initial_speed_mps": 3.0,
        "steps": [
            {
                "step": "cruise",
                "duration_s": 5.0,
            },
            {
                "step": "lane_change",
                "lateral_delta_m": 3.5,
                "duration_s": 2.0,
            },
            {
                "step": "brake",
                "target_speed_mps": 0.0,
                "deceleration_mps2": 3.0,
            },
            {
                "step": "hold",
                "duration_s": 5.0,
            },
            {
                "step": "accelerate",
                "target_speed_mps": 8.0,
                "acceleration_mps2": 4.0,
            },
        ],
    })


def resolve(t_s):
    return resolve_sequence_state(
        Pose2DV2(
            x_m=0.0,
            y_m=0.0,
            yaw_deg=0.0,
        ),
        make_sequence(),
        t_s=t_s,
        active_start_s=0.0,
    )


def test_sequence_initial_state():
    s = resolve(0.0)

    assert s.x_m == pytest.approx(0.0)
    assert s.y_m == pytest.approx(0.0)
    assert s.speed_mps == pytest.approx(3.0)


def test_cruise_endpoint():
    s = resolve(5.0)

    assert s.x_m == pytest.approx(15.0)
    assert s.y_m == pytest.approx(0.0)
    assert s.speed_mps == pytest.approx(3.0)


def test_lane_change_midpoint():
    s = resolve(6.0)

    assert s.x_m == pytest.approx(18.0)
    assert s.y_m == pytest.approx(1.75)

    assert s.yaw_deg > 0.0


def test_lane_change_endpoint():
    s = resolve(7.0)

    assert s.x_m == pytest.approx(21.0)
    assert s.y_m == pytest.approx(3.5)
    assert s.yaw_deg == pytest.approx(0.0)
    assert s.speed_mps == pytest.approx(3.0)


def test_braking_reaches_stop():
    s = resolve(8.0)

    assert s.x_m == pytest.approx(22.5)
    assert s.y_m == pytest.approx(3.5)
    assert s.speed_mps == pytest.approx(0.0)


def test_hold_does_not_move():
    s0 = resolve(8.0)
    s1 = resolve(13.0)

    assert s1.x_m == pytest.approx(s0.x_m)
    assert s1.y_m == pytest.approx(s0.y_m)
    assert s1.speed_mps == pytest.approx(0.0)


def test_acceleration():
    s = resolve(14.0)

    assert s.x_m == pytest.approx(24.5)
    assert s.speed_mps == pytest.approx(4.0)

    s = resolve(15.0)

    assert s.x_m == pytest.approx(30.5)
    assert s.speed_mps == pytest.approx(8.0)


def test_resolution_is_stateless():
    # Calling later times first must not affect the result.
    later = resolve(15.0)
    earlier = resolve(6.0)
    later_again = resolve(15.0)

    assert later_again.x_m == pytest.approx(later.x_m)
    assert later_again.y_m == pytest.approx(later.y_m)
    assert earlier.y_m == pytest.approx(1.75)


def test_lane_change_respects_actor_heading():
    motion = SequenceMotionV2.model_validate({
        "initial_speed_mps": 4.0,
        "steps": [
            {
                "step": "lane_change",
                "lateral_delta_m": 3.0,
                "duration_s": 3.0,
            }
        ],
    })

    spawn = Pose2DV2(
        x_m=0.0,
        y_m=0.0,
        yaw_deg=90.0,
    )

    state = resolve_sequence_state(
        spawn,
        motion,
        t_s=3.0,
        active_start_s=0.0,
    )

    # Vehicle faced +y.
    # Its local-left direction therefore points toward -x.
    assert state.x_m == pytest.approx(-3.0)
    assert state.y_m == pytest.approx(12.0)
    assert state.yaw_deg == pytest.approx(90.0)