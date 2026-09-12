"""Geometry and interpolation regressions for the isolated close candidate."""
import csv
import tempfile
import unittest
from pathlib import Path

import carla
import cv2
import numpy as np

try:
    from .calibrated_close import CalibratedCloseBank, bracket, composite, coverage_weight
except ImportError:
    from calibrated_close import CalibratedCloseBank, bracket, composite, coverage_weight


class CalibratedCloseTests(unittest.TestCase):
    def test_all_bank_boundaries_fade_to_native(self):
        from types import SimpleNamespace
        bank = SimpleNamespace(axes=[[-12, 12], [-4.5, -3.5, 3.5, 4.5], [-.3, .7]])
        self.assertEqual(coverage_weight(bank, [0, -4, 0]), 1)
        for query in ([12, -4, 0], [0, -4.5, 0], [0, -3.5, 0], [0, -4, -.3], [0, -4, .7]):
            self.assertEqual(coverage_weight(bank, query), 0)
        self.assertLess(coverage_weight(bank, [0, -4.5+1e-6, 0]), 1e-8)

    def test_brackets_are_continuous_and_normalized(self):
        for q in np.linspace(-2, 2, 101):
            pairs = bracket([-2, 0, 2], q)
            self.assertAlmostEqual(sum(w for _, w in pairs), 1)
            self.assertAlmostEqual(sum(v*w for v, w in pairs), q)
        self.assertIsNone(bracket([-2, 0, 2], 2.1))

    def test_premultiplied_composite(self):
        bg = np.full((2, 2, 3), 100, np.uint8)
        layer = np.zeros((2, 2, 4), np.float32)
        layer[:, :, :3], layer[:, :, 3] = 50, .5
        np.testing.assert_array_equal(composite(bg, layer), bg)

    def test_exact_capture_rotation_and_world_invariance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source = np.arange(8*8*4, dtype=np.uint8).reshape(8, 8, 4)
            source[:, :, 3] = 255
            cv2.imwrite(str(path/"source.png"), source)
            row = dict(close_forward_m=0, close_right_m=-4, close_target_up_m=-2,
                       rgba_relpath="source.png", camera_fx_px=4, camera_fy_px=4,
                       camera_cx_px=4, camera_cy_px=4, crop_x1_px=0, crop_y1_px=0)
            for prefix, xyz, yaw in (("actor", (0, 0, 0), 0), ("camera", (0, 4, 2), -90)):
                row.update({f"{prefix}_location_{a}_m": v for a, v in zip("xyz", xyz)})
                row.update({f"{prefix}_{a}_deg": yaw if a == "yaw" else 0 for a in ("pitch", "yaw", "roll")})
            with (path/"view_matrix.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            sampler = CalibratedCloseBank(path/"view_matrix.csv", [0, 0, 0], 1)
            actor = carla.Transform()
            camera = carla.Transform(carla.Location(y=4, z=2), carla.Rotation(yaw=-90))
            layer, metadata = sampler.render(actor, camera, 8, 8, 90)
            np.testing.assert_allclose(layer[:, :, :3], source[:, :, :3], atol=1e-5)
            np.testing.assert_array_equal(layer[:, :, 3], np.ones((8, 8)))
            self.assertEqual(len(metadata["sources"]), 1)
            # Rigidly moving the whole setup must not change the selected view.
            actor = carla.Transform(carla.Location(x=10, y=20, z=30), carla.Rotation(yaw=90))
            camera = carla.Transform(carla.Location(x=6, y=20, z=32), carla.Rotation(yaw=0))
            moved, _ = sampler.render(actor, camera, 8, 8, 90)
            # CARLA matrices use float32; invariance is sub-byte, not bitwise.
            np.testing.assert_allclose(moved, layer, atol=1e-3)
            # Opposite-side geometry must not borrow an incompatible bank.
            camera = carla.Transform(carla.Location(y=-4, z=2), carla.Rotation(yaw=90))
            self.assertIsNone(sampler.render(carla.Transform(), camera, 8, 8, 90))

    def test_hull_empty_view_is_valid(self):
        try:
            from .hull_rays import HullRays
        except ImportError:
            from hull_rays import HullRays
        face = np.array([[[-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]],
                         [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]]], np.float32)
        hull = HullRays(face, spacing=.5, device="cpu")
        points, valid = hull.intersect(np.array([10, 0, 0]), np.array([[[1., 0, 0]]]))
        self.assertFalse(valid.any())
        self.assertTrue(np.isfinite(points).all())


if __name__ == "__main__":
    unittest.main()
