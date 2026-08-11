from scenario_generator.planner.manual_planner import make_oncoming_vehicle_demo
from scenario_generator.trajectory.rule_based.rule_based_generator import RuleBasedTrajectoryGenerator


def test_oncoming_x_decreases():
    scenario = make_oncoming_vehicle_demo()
    resolved = RuleBasedTrajectoryGenerator().resolve(scenario)

    actor_frames = [f for f in resolved.frames if f.actor_id == "adv_001"]
    assert actor_frames[0].x_m > actor_frames[-1].x_m
