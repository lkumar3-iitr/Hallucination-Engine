import unittest
from types import SimpleNamespace

from generic_he_closed_loop_runner_v1 import filter_trigger_gated_actors


class TriggerGatedActorsTest(unittest.TestCase):
    def setUp(self):
        self.bus = SimpleNamespace(actor_id="parked_bus")
        self.pedestrian = SimpleNamespace(actor_id="hidden_pedestrian")
        self.states = [self.bus, self.pedestrian]

    def test_gated_actor_is_absent_before_trigger(self):
        actual = filter_trigger_gated_actors(
            self.states, None, ["hidden_pedestrian"]
        )
        self.assertEqual([state.actor_id for state in actual], ["parked_bus"])

    def test_all_actors_are_present_after_trigger(self):
        actual = filter_trigger_gated_actors(
            self.states, 200, ["hidden_pedestrian"]
        )
        self.assertEqual(actual, self.states)


if __name__ == "__main__":
    unittest.main()
