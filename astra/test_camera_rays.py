import unittest

import numpy as np

from astra.calibrated_close import camera_rays


class CameraRayTests(unittest.TestCase):
    def test_cached_calibration_and_read_only(self):
        rays = camera_rays(400, 300, 100)
        self.assertIs(rays, camera_rays(400, 300, 100))
        self.assertFalse(rays.flags.writeable)
        np.testing.assert_array_equal(rays[150, 200], [1, 0, 0])
        self.assertFalse(np.array_equal(rays, camera_rays(400, 300, 90)))


if __name__ == '__main__':
    unittest.main()
