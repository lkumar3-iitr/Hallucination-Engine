import pytest

from scenario_generator.planner.semantic_scenario_v2 import (
    SemanticScenarioV2,
    compile_semantic_scenario,
)
from scenario_generator.planner.event_conditions_v2 import (
    ActorStoppedConditionV2,
    DistanceBelowConditionV2,
    RelativeLongitudinalConditionV2,
    TimeConditionV2,
    resolve_condition_time,
)
from scenario_generator.trajectory.trajectory_resolver_v2 import (
    V2TrajectoryResolver,
)


def make_resolved():
    semantic = SemanticScenarioV2.model_validate({
        "scenario_id": "event_test",
        "duration_s": 15.0,
        "fps": 20,
        "ego_speed_mps": 5.0,

        "actors": [
            {
                "actor_id": "lead",
                "longitudinal_offset_m": 15.0,

                "behavior": {
                    "initial_speed_mps": 3.0,

                    "steps": [
                        {
                            "action": "cruise",
                            "duration_s": 5.0,
                        },
                        {
                            "action": "brake_to",
                            "target_speed_mps": 0.0,
                            "deceleration_mps2": 3.0,
                        },
                        {
                            "action": "hold",
                            "duration_s": 5.0,
                        },
                    ],
                },
            }
        ],
    })

    scenario = compile_semantic_scenario(
        semantic
    )

    return V2TrajectoryResolver().resolve(
        scenario
    )


def test_actor_stopped_condition():
    resolved = make_resolved()

    t = resolve_condition_time(
        resolved,
        ActorStoppedConditionV2(
            actor_id="lead",
        ),
    )

    assert t == pytest.approx(6.0)


def test_distance_below_condition():
    resolved = make_resolved()

    t = resolve_condition_time(
        resolved,
        DistanceBelowConditionV2(
            actor_a_id="ego",
            actor_b_id="lead",
            distance_m=10.0,
        ),
    )

    assert t == pytest.approx(2.5)


def test_ego_passes_actor():
    resolved = make_resolved()

    t = resolve_condition_time(
        resolved,
        RelativeLongitudinalConditionV2(
            subject_id="ego",
            reference_id="lead",
            relation="ahead_by",
            margin_m=0.0,
        ),
    )

    # Lead stops at x ~= 31.5 m.
    # Ego travels at 5 m/s.
    # First 20 FPS frame at/after passing is ~6.30 s.
    assert t == pytest.approx(
        6.30,
        abs=0.051,
    )


def test_ahead_by_margin():
    resolved = make_resolved()

    t = resolve_condition_time(
        resolved,
        RelativeLongitudinalConditionV2(
            subject_id="ego",
            reference_id="lead",
            relation="ahead_by",
            margin_m=2.0,
        ),
    )

    assert t == pytest.approx(
        6.70,
        abs=0.051,
    )


def test_not_before_constraint():
    resolved = make_resolved()

    t = resolve_condition_time(
        resolved,
        TimeConditionV2(
            t_s=2.0,
        ),
        not_before_s=4.0,
    )

    assert t == pytest.approx(4.0)