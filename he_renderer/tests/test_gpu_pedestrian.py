import unittest
from types import SimpleNamespace
from unittest.mock import patch

from he_renderer.calibrated.gpu_pedestrian import PedestrianSelector


class PedestrianPolicyTests(unittest.TestCase):
    def test_distilled_choice_is_not_replaced(self):
        selector = object.__new__(PedestrianSelector)
        selected = {'index': 12, 'selection_mode': 'distilled'}
        selector.teacher = SimpleNamespace(distilled_select=lambda *args: selected)
        with patch('he_renderer.calibrated.gpu_native_projection.GPUNativeProjection.select') as fallback:
            self.assertIs(selector.select(None,None,400,300,100), selected)
            fallback.assert_not_called()

    def test_rejected_distilled_uses_exact_gpu_choice(self):
        selector = object.__new__(PedestrianSelector)
        selector.teacher = SimpleNamespace(distilled_select=lambda *args: None)
        with patch('he_renderer.calibrated.gpu_native_projection.GPUNativeProjection.select', return_value={'index': 8}) as fallback:
            result = selector.select(None,None,400,300,100)
        fallback.assert_called_once()
        self.assertEqual(result, dict(index=8,selection_mode='exact',distilled_neighbor_similarity=None))


if __name__ == '__main__':
    unittest.main()
