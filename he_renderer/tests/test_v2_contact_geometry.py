import unittest

import carla
import numpy as np

from he_renderer.calibrated.passby_renderer import aimed_camera, UnsupportedCameraGeometry


class ContactGeometryTests(unittest.TestCase):
    def test_recorded_inside_pose_is_explicitly_unsupported(self):
        actor = carla.Transform(carla.Location(x=47.66026306152344,
            y=-55.21013259887695, z=.05585484579205513), carla.Rotation(yaw=89.9765625))
        camera = carla.Transform(carla.Location(x=46.59035873413086,
            y=-57.568328857421875, z=2.0342886447906494))
        center = np.array([.028382908552885056, .000002249287945232936, 1.0167187452316284])
        extents = np.array([2.782914400100708, 1.0749834775924683, 1.0225735902786255])
        with self.assertRaises(UnsupportedCameraGeometry) as caught:
            aimed_camera(actor, camera, center, extents)
        self.assertEqual(caught.exception.reason, 'camera_inside_or_on_physical_bbox')

    def test_near_surface_is_not_mislabelled_center(self):
        with self.assertRaises(UnsupportedCameraGeometry) as caught:
            aimed_camera(carla.Transform(), carla.Transform(carla.Location(x=1.02)),
                         np.zeros(3), np.ones(3))
        self.assertEqual(caught.exception.reason, 'camera_near_physical_bbox_surface')
        self.assertAlmostEqual(caught.exception.distance_m, .02, places=6)

    def test_supported_external_view_is_unchanged(self):
        camera = carla.Transform(carla.Location(x=2))
        virtual, distance = aimed_camera(carla.Transform(), camera, np.zeros(3), np.ones(3))
        self.assertAlmostEqual(distance, 1.)
        self.assertAlmostEqual(abs(virtual.rotation.yaw), 180.)
        self.assertEqual(virtual.location, camera.location)


if __name__ == '__main__':
    unittest.main()
