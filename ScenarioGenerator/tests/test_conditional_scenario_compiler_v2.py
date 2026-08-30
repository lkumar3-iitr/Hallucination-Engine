import pytest

from scenario_generator.planner.conditional_scenario_compiler_v2 import (
    compile_semantic_scenario_with_events,
)
from scenario_generator.planner.semantic_scenario_v2 import (
    SemanticScenarioV2,
)


def make_ego_pass_scenario():
    return SemanticScenarioV2.model_validate({
        "scenario_id": "hold_until_test",
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
                            "action": "hold_until",
                            "condition": {
                                "kind": "relative_longitudinal",
                                "subject_id": "ego",
                                "reference_id": "lead",
                                "relation": "ahead_by",
                                "margin_m": 2.0,
                            },
                        },
                        {
                            "action": "accelerate_to",
                            "target_speed_mps": 6.0,
                            "acceleration_mps2": 3.0,
                        },
                    ],
                },
            }
        ],
    })


def test_hold_until_compiles_to_fixed_hold():
    spec = compile_semantic_scenario_with_events(
        make_ego_pass_scenario()
    )

    motion = spec.actors[0].motion

    assert motion.mode == "sequence"

    assert motion.steps[2].step == "hold"

    assert motion.steps[2].duration_s == pytest.approx(
        0.70,
        abs=0.051,
    )


def test_no_semantic_condition_reaches_physical_motion():
    spec = compile_semantic_scenario_with_events(
        make_ego_pass_scenario()
    )

    actor = spec.actors[0]

    motion_dump = actor.motion.model_dump()

    text = str(motion_dump)

    assert "hold_until" not in text
    assert "relative_longitudinal" not in text

    step_types = [
        step.step
        for step in actor.motion.steps
    ]

    assert step_types == [
        "cruise",
        "brake",
        "hold",
        "accelerate",
    ]


def test_minimum_hold_duration_is_respected():
    semantic = SemanticScenarioV2.model_validate({
        "scenario_id": "minimum_hold_test",
        "duration_s": 10.0,
        "fps": 20,
        "ego_speed_mps": 5.0,

        "actors": [
            {
                "actor_id": "car",
                "longitudinal_offset_m": 10.0,

                "behavior": {
                    "initial_speed_mps": 0.0,

                    "steps": [
                        {
                            "action": "hold_until",
                            "min_duration_s": 2.0,

                            "condition": {
                                "kind": "time",
                                "t_s": 0.5,
                            },
                        },
                        {
                            "action": "accelerate_to",
                            "target_speed_mps": 4.0,
                            "acceleration_mps2": 2.0,
                        },
                    ],
                },
            }
        ],
    })

    spec = compile_semantic_scenario_with_events(
        semantic
    )

    hold = spec.actors[0].motion.steps[0]

    assert hold.step == "hold"

    assert hold.duration_s == pytest.approx(
        2.0
    )


def test_actor_to_actor_dependency():
    semantic = SemanticScenarioV2.model_validate({
        "scenario_id": "actor_dependency_test",
        "duration_s": 12.0,
        "fps": 20,
        "ego_speed_mps": 5.0,

        "actors": [
            {
                "actor_id": "trigger",
                "longitudinal_offset_m": 20.0,

                "behavior": {
                    "initial_speed_mps": 2.0,

                    "steps": [
                        {
                            "action": "cruise",
                            "duration_s": 2.0,
                        },
                        {
                            "action": "brake_to",
                            "target_speed_mps": 0.0,
                            "deceleration_mps2": 2.0,
                        },
                        {
                            "action": "hold",
                            "duration_s": 5.0,
                        },
                    ],
                },
            },

            {
                "actor_id": "waiter",
                "longitudinal_offset_m": 5.0,

                "behavior": {
                    "initial_speed_mps": 0.0,

                    "steps": [
                        {
                            "action": "hold_until",

                            "condition": {
                                "kind": "actor_stopped",
                                "actor_id": "trigger",
                            },
                        },
                        {
                            "action": "accelerate_to",
                            "target_speed_mps": 4.0,
                            "acceleration_mps2": 2.0,
                        },
                    ],
                },
            },
        ],
    })

    spec = compile_semantic_scenario_with_events(
        semantic
    )

    actors = {
        actor.actor_id: actor
        for actor in spec.actors
    }

    hold = actors[
        "waiter"
    ].motion.steps[0]

    # Trigger:
    # cruise 2 s at 2 m/s
    # then brake 2 -> 0 at 2 m/s^2 = 1 s
    #
    # therefore stopped at ~3 s.
    assert hold.duration_s == pytest.approx(
        3.0,
        abs=0.051,
    )
def test_cruise_until_compiles_to_fixed_cruise():
    semantic = SemanticScenarioV2.model_validate({
        "scenario_id": "cruise_until_test",
        "duration_s": 10.0,
        "fps": 20,
        "ego_speed_mps": 5.0,

        "actors": [
            {
                "actor_id": "adv",
                "longitudinal_offset_m": 0.0,

                "behavior": {
                    "initial_speed_mps": 9.0,

                    "steps": [
                        {
                            "action": "cruise_until",
                            "speed_mps": 9.0,

                            "condition": {
                                "kind": "relative_longitudinal",
                                "subject_id": "adv",
                                "reference_id": "ego",
                                "relation": "ahead_by",
                                "margin_m": 8.0,
                            },
                        },
                        {
                            "action": "brake_to",
                            "target_speed_mps": 0.0,
                            "deceleration_mps2": 3.0,
                        },
                    ],
                },
            }
        ],
    })

    spec = compile_semantic_scenario_with_events(
        semantic
    )

    motion = spec.actors[0].motion

    assert motion.steps[0].step == "cruise"

    assert motion.steps[0].duration_s == pytest.approx(
        2.0,
        abs=0.051,
    )

    assert motion.steps[0].speed_mps == pytest.approx(
        9.0
    )

    assert motion.steps[1].step == "brake"


def test_cruise_until_condition_does_not_reach_physical_motion():
    semantic = SemanticScenarioV2.model_validate({
        "scenario_id": "cruise_condition_boundary_test",
        "duration_s": 10.0,
        "fps": 20,
        "ego_speed_mps": 5.0,

        "actors": [
            {
                "actor_id": "adv",
                "longitudinal_offset_m": 0.0,

                "behavior": {
                    "initial_speed_mps": 9.0,

                    "steps": [
                        {
                            "action": "cruise_until",

                            "condition": {
                                "kind": "relative_longitudinal",
                                "subject_id": "adv",
                                "reference_id": "ego",
                                "relation": "ahead_by",
                                "margin_m": 8.0,
                            },
                        },
                    ],
                },
            }
        ],
    })

    spec = compile_semantic_scenario_with_events(
        semantic
    )

    motion_text = str(
        spec.actors[0].motion.model_dump()
    )

    assert "cruise_until" not in motion_text
    assert "relative_longitudinal" not in motion_text

    assert (
        spec.actors[0].motion.steps[0].step
        == "cruise"
    )

def test_conditional_compilation_is_deterministic():
    semantic = make_ego_pass_scenario()

    first = compile_semantic_scenario_with_events(
        semantic
    )

    second = compile_semantic_scenario_with_events(
        semantic
    )

    assert (
        first.model_dump()
        ==
        second.model_dump()
    )