import unittest

import carla
import numpy as np

from he_renderer.renderer import (
    aimed_camera,
    camera_rotation_map,
    reproject_sprite,
    reproject_sprite_reference,
)
from he_renderer.selector import iou_scores, packed_iou_scores, project, relative_matrix


class SidePassTests(unittest.TestCase):
    def test_compiled_reprojection_matches_reference_visibility(self):
        rng = np.random.default_rng(20260908)
        background = rng.integers(0, 256, (72, 128, 3), dtype=np.uint8)
        sprite = rng.integers(0, 256, (31, 53, 4), dtype=np.uint8)
        sprite[:, :, 3] = np.where(sprite[:, :, 3] > 90, sprite[:, :, 3], 0)
        box = [-8.25, 11.5, 102.75, 68.25]
        matrix = np.array([[1, .04, -3], [-.02, 1, 2], [8e-5, -5e-5, 1.]])

        expected_rgb, expected_alpha = reproject_sprite_reference(
            background, sprite, box, matrix
        )
        actual_rgb, actual_alpha = reproject_sprite(background, sprite, box, matrix)

        np.testing.assert_array_equal(actual_alpha > .04, expected_alpha > .04)
        self.assertLessEqual(float(np.max(np.abs(actual_alpha-expected_alpha))), 1/255)
        self.assertLessEqual(
            int(np.max(np.abs(actual_rgb.astype(np.int16)-expected_rgb.astype(np.int16)))), 2
        )

    def test_packed_iou_matches_boolean_iou(self):
        rng = np.random.default_rng(20260908)
        masks = rng.random((37, 128, 128)) < 0.31
        masks[0] = False
        masks[1] = True

        for density in (0.0, 0.01, 0.5, 1.0):
            target = rng.random((128, 128)) < density
            expected = iou_scores(masks, target)
            packed = np.packbits(masks.reshape((len(masks), -1)), axis=1)
            actual = packed_iou_scores(packed, target)
            np.testing.assert_array_equal(actual, expected)
            areas = np.count_nonzero(masks, axis=(1, 2))
            precomputed = packed_iou_scores(packed, target, areas)
            np.testing.assert_array_equal(precomputed, expected)

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

    def test_bus_sized_bbox_stays_in_front_of_virtual_camera(self):
        actor = carla.Transform()
        center = np.array([-.445, -.121, 2.127])
        extents = np.array([5.136, 1.972, 2.126])
        corners = np.array([
            center + np.array([x, y, z]) * extents
            for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
        ])
        for x in (-4.2, -1.5, 0, 1.5, 4.2):
            camera = carla.Transform(carla.Location(x=x, y=3.5, z=1.55))
            virtual, _ = aimed_camera(actor, camera, center, extents)
            _, depth = project(
                corners, relative_matrix(actor, virtual), 640, 640, 640, 360
            )
            self.assertTrue(np.all(depth > 0), depth)

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
