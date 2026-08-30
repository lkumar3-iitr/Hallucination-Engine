import pytest

from scenario_generator.planner.semantic_scenario_v2 import (
    SemanticScenarioV2,
    compile_semantic_scenario,
)
from scenario_generator.trajectory.trajectory_resolver_v2 import (
    V2TrajectoryResolver,
)


def make_multi_actor_scenario():
    return SemanticScenarioV2.model_validate({
        "scenario_id": "multi_semantic_test",
        "duration_s": 15.0,
        "fps": 20,
        "ego_speed_mps": 5.0,
        "lane_width_m": 3.5,

        "actors": [
            {
                "actor_id": "lead",
                "longitudinal_offset_m": 15.0,
                "behavior": {
                    "initial_lane": "ego_lane",
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
            },

            {
                "actor_id": "left_car",
                "longitudinal_offset_m": 8.0,
                "behavior": {
                    "initial_lane": "left_adjacent",
                    "initial_speed_mps": 4.0,
                    "steps": [
                        {
                            "action": "cruise",
                            "duration_s": 4.0,
                        },
                        {
                            "action": "lane_change",
                            "target_lane": "ego_lane",
                            "duration_s": 3.0,
                        },
                        {
                            "action": "cruise",
                            "duration_s": 3.0,
                        },
                    ],
                },
            },

            {
                "actor_id": "right_car",
                "longitudinal_offset_m": 25.0,
                "behavior": {
                    "initial_lane": "right_adjacent",
                    "initial_speed_mps": 2.0,
                    "steps": [
                        {
                            "action": "cruise",
                            "duration_s": 10.0,
                        },
                    ],
                },
            },
        ],
    })


def test_multi_actor_compilation():
    semantic = make_multi_actor_scenario()
    scenario = compile_semantic_scenario(semantic)

    assert len(scenario.actors) == 3

    ids = {
        actor.actor_id
        for actor in scenario.actors
    }

    assert ids == {
        "lead",
        "left_car",
        "right_car",
    }


def test_semantic_lane_spawn_positions():
    scenario = compile_semantic_scenario(
        make_multi_actor_scenario()
    )

    actors = {
        actor.actor_id: actor
        for actor in scenario.actors
    }

    assert actors["lead"].spawn.y_m == pytest.approx(0.0)
    assert actors["left_car"].spawn.y_m == pytest.approx(3.5)
    assert actors["right_car"].spawn.y_m == pytest.approx(-3.5)

    assert actors["lead"].spawn.x_m == pytest.approx(15.0)
    assert actors["left_car"].spawn.x_m == pytest.approx(8.0)
    assert actors["right_car"].spawn.x_m == pytest.approx(25.0)


def test_all_semantic_actors_compile_to_sequences():
    scenario = compile_semantic_scenario(
        make_multi_actor_scenario()
    )

    for actor in scenario.actors:
        assert actor.motion.mode == "sequence"


def test_duplicate_actor_ids_rejected():
    with pytest.raises(ValueError):
        SemanticScenarioV2.model_validate({
            "scenario_id": "duplicate_test",
            "duration_s": 10.0,

            "actors": [
                {
                    "actor_id": "car",
                    "behavior": {
                        "initial_speed_mps": 2.0,
                        "steps": [
                            {
                                "action": "cruise",
                                "duration_s": 3.0,
                            }
                        ],
                    },
                },
                {
                    "actor_id": "car",
                    "behavior": {
                        "initial_speed_mps": 2.0,
                        "steps": [
                            {
                                "action": "cruise",
                                "duration_s": 3.0,
                            }
                        ],
                    },
                },
            ],
        })


def test_multi_actor_resolves_to_frames():
    semantic = make_multi_actor_scenario()

    scenario = compile_semantic_scenario(
        semantic
    )

    resolved = V2TrajectoryResolver().resolve(
        scenario
    )

    actor_ids = {
        frame.actor_id
        for frame in resolved.actor_frames
    }

    assert "lead" in actor_ids
    assert "left_car" in actor_ids
    assert "right_car" in actor_ids

    assert len(resolved.ego_frames) > 0
    assert len(resolved.actor_frames) > 0