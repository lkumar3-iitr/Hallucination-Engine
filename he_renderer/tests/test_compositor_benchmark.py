import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from he_renderer.evaluation.benchmark_compositor import camera_specs, clear_queries


class BenchmarkTests(unittest.TestCase):
    def test_literal_cameras_and_reject_computed_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'adapter.py'
            path.write_text('def camera_specs():\n return [CameraSpec(width=400, yaw_deg=-60)]\n')
            self.assertEqual(camera_specs(path), [{'width': 400, 'yaw_deg': -60}])
            path.write_text('def camera_specs():\n return [CameraSpec(width=calculate())]\n')
            with self.assertRaises(ValueError):
                camera_specs(path)

    def test_query_cache_reset_preserves_engine(self):
        selector = SimpleNamespace(clear=lambda: calls.append(True))
        calls = []
        close_hull = SimpleNamespace(_intersection_cache=object())
        depth_hull = SimpleNamespace(_intersection_cache=object())
        compositor = SimpleNamespace(
            base=SimpleNamespace(engines={}),
            gpu_engines={'a': SimpleNamespace(native=SimpleNamespace(selector=selector))},
            bus_engines={'a': SimpleNamespace(close=SimpleNamespace(hull=close_hull))},
            depth_hulls={'p': depth_hull})
        clear_queries(compositor)
        self.assertEqual(calls, [True])
        self.assertIsNone(close_hull._intersection_cache)
        self.assertIsNone(depth_hull._intersection_cache)


if __name__ == '__main__':
    unittest.main()
