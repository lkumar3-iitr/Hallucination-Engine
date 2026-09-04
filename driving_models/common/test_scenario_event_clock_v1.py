from scenario_event_clock_v1 import RouteTriggeredEventClock


def test_untriggered_clock_uses_execution_frame():
    clock = RouteTriggeredEventClock(None, event_source_frame=200)
    assert clock.source_frame(17) == 17


def test_triggered_clock_stages_then_replays_relative_frames():
    clock = RouteTriggeredEventClock(50.0, event_source_frame=220)
    assert clock.source_frame(100) == 100
    assert clock.source_frame(250) == 220
    assert clock.observe_for_next_frame(251, 49.99) == 220
    assert clock.observe_for_next_frame(252, 50.01) == 220
    assert clock.trigger_frame == 252
    assert clock.relative_frame(252) == 0
    assert clock.source_frame(262) == 230
    assert clock.trigger_observed_progress_m == 50.01


def test_triggered_clock_can_hold_explicit_pretrigger_pose():
    clock = RouteTriggeredEventClock(
        50.0, event_source_frame=220, pre_trigger_source_frame=219
    )
    assert clock.source_frame(250) == 219
    assert clock.observe_for_next_frame(251, 50.01) == 220
    assert clock.source_frame(252) == 221
