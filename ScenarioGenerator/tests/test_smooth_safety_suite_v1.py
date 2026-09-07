import math

from scripts.build_smooth_safety_suite_v1 import (
    DURATION_S,
    FPS,
    lane_change,
    minimum_jerk,
)


def test_minimum_jerk_has_exact_endpoints():
    assert minimum_jerk(-1.0) == 0.0
    assert minimum_jerk(0.0) == 0.0
    assert minimum_jerk(1.0) == 1.0
    assert minimum_jerk(2.0) == 1.0


def test_lane_change_is_frame_sampled_and_tangent_aligned():
    frames = lane_change(30.0, 6.0, -3.5, 0.0, 2.0, 6.0)

    assert len(frames) == int(DURATION_S * FPS) + 1
    assert frames[0]["y_m"] == -3.5
    assert frames[-1]["y_m"] == 0.0
    assert frames[0]["yaw_deg"] == 0.0
    assert frames[-1]["yaw_deg"] == 0.0
    assert max(abs(frame["yaw_deg"]) for frame in frames) < 20.0
    assert all(
        math.isclose(frame["t_s"], index / FPS, abs_tol=1e-8)
        for index, frame in enumerate(frames)
    )
