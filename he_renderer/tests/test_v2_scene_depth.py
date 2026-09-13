import unittest
import numpy as np
import carla

from he_renderer.calibrated.scene_depth import occlude, surface_depth


class DepthTests(unittest.TestCase):
    def test_foreground_background_and_invalid_depth(self):
        rgb = np.full((2, 3, 3), 200, np.uint8)
        background = np.full_like(rgb, 20)
        alpha = np.ones((2, 3), np.float32)
        actor = np.full((2, 3), 10.)
        scene = np.array([[5., 20., 10.], [np.nan, 0., np.inf]])
        out, a, meta = occlude(rgb, background, alpha, actor, scene)
        self.assertEqual(meta['occluded_pixels'], 1)
        self.assertEqual(meta['invalid_scene_pixels'], 3)
        np.testing.assert_array_equal(out[0, 0], background[0, 0])
        self.assertEqual(a[0, 0], 0)
        np.testing.assert_array_equal(out[1], rgb[1])

    def test_unknown_surface_retained_and_shape_rejected(self):
        rgb = np.ones((2, 2, 3), np.uint8)
        alpha = np.ones((2, 2), np.float32)
        out, a, meta = occlude(rgb, rgb*0, alpha, np.full((2, 2), np.nan), alpha)
        np.testing.assert_array_equal(out, rgb)
        self.assertEqual(meta['unknown_surface_pixels'], 4)
        with self.assertRaises(ValueError):
            occlude(rgb, rgb, alpha, alpha, np.ones((1, 2)))

    def test_camera_forward_not_ray_distance(self):
        class PlaneHull:
            def intersect(self, origin, directions):
                return origin+directions*10, np.ones(directions.shape[:2], bool)
        actor = carla.Transform()
        camera = carla.Transform(carla.Location(z=2), carla.Rotation(yaw=60))
        result = surface_depth(PlaneHull(), actor, camera, 8, 6, 100)
        np.testing.assert_allclose(result, 10, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
