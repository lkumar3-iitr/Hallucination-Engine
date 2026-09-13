import unittest
from unittest.mock import Mock
import numpy as np
import torch
from he_renderer.calibrated.hull_rays import HullRays


class IntersectionCacheTests(unittest.TestCase):
    def test_exact_match_and_invalidations(self):
        hull = HullRays.__new__(HullRays)
        hull.volume = torch.zeros((1, 1, 2, 2, 2))
        hull.spacing = .1
        hull.low, hull.high = np.zeros(3), np.ones(3)
        hull._intersect_uncached = Mock(side_effect=lambda o, d:
            (np.zeros_like(d), np.ones(d.shape[:2], bool)))
        origin, rays = np.zeros(3), np.ones((2, 2, 3))
        first = hull.intersect(origin, rays)
        self.assertIs(first, hull.intersect(origin.copy(), rays.copy()))
        self.assertEqual(hull._intersect_uncached.call_count, 1)
        self.assertFalse(first[0].flags.writeable)
        rays[0, 0, 0] += 1e-9
        hull.intersect(origin, rays)
        origin[0] += 1e-9
        hull.intersect(origin, rays)
        hull.volume.add_(1)
        hull.intersect(origin, rays)
        self.assertEqual(hull._intersect_uncached.call_count, 4)
        with torch.inference_mode():
            hull.volume = torch.zeros((1, 1, 2, 2, 2))
        hull.intersect(origin, rays)
        self.assertEqual(hull._intersect_uncached.call_count, 5)


if __name__ == '__main__':
    unittest.main()
