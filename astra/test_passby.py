import unittest

import carla
import numpy as np

from passby_renderer import aimed_camera, camera_rotation_map, reproject_sprite
from selector import project, relative_matrix


class SidePassTests(unittest.TestCase):
    def test_rotation_matches_exact_3d_projection(self):
        actor = carla.Transform(carla.Location(x=3, y=-3.5))
        camera = carla.Transform(carla.Location(z=1.6), carla.Rotation(pitch=-5, yaw=-40, roll=8))
        virtual, _ = aimed_camera(actor, camera, np.array([0, 0, .75]))
        points = np.array([[x, y, z] for x in (-2, 2) for y in (-1, 1) for z in (0, 1.5)])
        native, _ = project(points, relative_matrix(actor, camera), 640, 640, 640, 360)
        centered, _ = project(points, relative_matrix(actor, virtual), 640, 640, 640, 360)
        mapping = camera_rotation_map(virtual, camera, 1280, 720, 90)
        mapped = np.c_[native, np.ones(len(native))] @ mapping.T
        np.testing.assert_allclose(mapped[:, :2]/mapped[:, 2:], centered, atol=.001)

    def test_aiming_remains_valid_as_ego_passes_actor_center(self):
        actor = carla.Transform()
        for x in (-4, -.01, 0, .01, 4):
            camera = carla.Transform(carla.Location(x=x, y=3.5, z=1.6))
            virtual, distance = aimed_camera(actor, camera, np.array([0, 0, .75]))
            uv, depth = project(np.array([[0, 0, .75]]), relative_matrix(actor, virtual), 640, 640, 640, 360)
            np.testing.assert_allclose(uv, [[640, 360]], atol=.001)
            self.assertGreater(depth[0], 3.5)

    def test_behind_camera_does_not_reappear(self):
        background = np.zeros((128, 128, 3), np.uint8)
        sprite = np.full((32, 32, 4), 255, np.uint8)
        target = carla.Transform()
        source = carla.Transform(rotation=carla.Rotation(yaw=180))
        result, alpha = reproject_sprite(background, sprite, (32, 32, 96, 96),
                                         camera_rotation_map(source, target, 128, 128, 90))
        self.assertFalse(alpha.any())

    def test_camera_plane_crossing_can_have_visible_fragment(self):
        background = np.zeros((128, 128, 3), np.uint8)
        sprite = np.full((32, 32, 4), 255, np.uint8)
        target = carla.Transform()
        source = carla.Transform(rotation=carla.Rotation(yaw=90))
        _, alpha = reproject_sprite(background, sprite, (-64, 32, 192, 96),
                                    camera_rotation_map(source, target, 128, 128, 90))
        self.assertGreater(np.count_nonzero(alpha), 0)
        self.assertFalse(alpha[:, :64].any())

    def test_translation_is_rejected(self):
        with self.assertRaises(ValueError):
            camera_rotation_map(carla.Transform(), carla.Transform(carla.Location(x=1)), 128, 128, 90)


if __name__ == "__main__":
    unittest.main()
