import unittest
from he_renderer.evaluation.run_runtime_campaign import validate_terminal_summary


class TerminalCampaignTests(unittest.TestCase):
    def test_valid_contact_including_initial_contact(self):
        for frame in (0, 243, 800):
            validate_terminal_summary(dict(frames=frame, stop_on_collision=True,
                termination_reason='collision', collision_event=dict(
                    physical_overlap=True, criterion='scenario_actor_2d_footprint_overlap',
                    scenario_frame=frame, rendered=False, model_executed=False)))

    def test_complete_noncollision(self):
        validate_terminal_summary(dict(frames=801, stop_on_collision=True,
            termination_reason='horizon', collision_event=None))

    def test_partial_without_collision_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_terminal_summary(dict(frames=244, stop_on_collision=True,
                termination_reason='horizon', collision_event=None))

    def test_collision_label_without_receipt_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_terminal_summary(dict(frames=244, stop_on_collision=True,
                termination_reason='collision', collision_event=None))

    def test_old_protocol_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_terminal_summary(dict(frames=801))
