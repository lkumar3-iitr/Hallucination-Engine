import unittest

import carla
import numpy as np

from he_renderer.selector import canonical, composite_to_box, iou_scores, project, relative_matrix


class GeometryTests(unittest.TestCase):
    def test_carla_axes(self):
        points = np.array([[10, 0, 0], [10, 2, 1], [-1, 0, 0]], dtype=float)
        uv, depth = project(points, np.eye(4), 640, 640, 640, 360)
        np.testing.assert_allclose(uv[:2], [[640, 360], [768, 296]])
        self.assertTrue(np.isnan(uv[2]).all())

    def test_common_world_transform_invariance(self):
        actor = carla.Transform(carla.Location(x=3, y=1), carla.Rotation(yaw=20))
        camera = carla.Transform(carla.Location(x=-6, y=-2, z=1.6))
        world = np.asarray(carla.Transform(carla.Location(x=40, y=20), carla.Rotation(yaw=73)).get_matrix())
        expected = relative_matrix(actor, camera)
        actual = np.linalg.inv(world @ np.asarray(camera.get_matrix())) @ (world @ np.asarray(actor.get_matrix()))
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_canonical_translation_scale_invariance(self):
        small = np.zeros((8, 9), bool)
        small[2:6, 3:7] = True
        large = np.repeat(np.repeat(small, 2, 0), 2, 1)
        np.testing.assert_array_equal(canonical(small, 32), canonical(large, 32))

    def test_iou_identity(self):
        target = np.eye(8, dtype=bool)
        np.testing.assert_allclose(iou_scores(np.stack([target, ~target]), target), [1, 0])

    def test_clipping_does_not_rescale_visible_fragment(self):
        background = np.zeros((12, 12, 3), np.uint8)
        sprite = np.full((4, 4, 4), 255, np.uint8)
        sprite[:, :2, :3] = [0, 0, 255]
        sprite[:, 2:, :3] = [255, 0, 0]
        full, _ = composite_to_box(background, sprite, (2, 2, 10, 10))
        clipped, alpha = composite_to_box(background, sprite, (-4, 2, 4, 10))
        np.testing.assert_array_equal(clipped[2:10, :4], full[2:10, 6:10])
        self.assertEqual(np.count_nonzero(alpha), 32)

    def test_same_color_still_has_alpha(self):
        background = np.full((8, 8, 3), 100, np.uint8)
        sprite = np.full((3, 3, 4), 100, np.uint8)
        sprite[:, :, 3] = 255
        output, alpha = composite_to_box(background, sprite, (2, 2, 6, 6))
        np.testing.assert_array_equal(output, background)
        self.assertEqual(np.count_nonzero(alpha), 16)

    def test_offscreen_and_invalid_box(self):
        background = np.zeros((8, 8, 3), np.uint8)
        sprite = np.full((4, 4, 4), 255, np.uint8)
        output, alpha = composite_to_box(background, sprite, (-20, 0, -10, 10))
        self.assertFalse(alpha.any())
        with self.assertRaises(ValueError):
            composite_to_box(background, sprite, (3, 2, 2, 4))


if __name__ == "__main__":
    unittest.main()
