import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cartesian = load_module(
    "cartesian_close_generator",
    SCRIPT_DIR / "generate_carla_cartesian_close_native_mask_v1.py",
)
native = load_module(
    "native_mask_generator",
    SCRIPT_DIR / "generate_carla_asset_view_matrix_native_mask_v3.py",
)


class BusPoseClassificationTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            bus_bbox_half_length_m=5.136342525482178,
            bus_bbox_half_width_m=1.9720759391784668,
            bus_camera_clearance_m=0.10,
            right_offsets_m=[3.0],
        )

    def test_parallel_alongside_pose_is_retained(self):
        pose = cartesian.classify_bus_pose(0.25, 3.0, 0.0, self.args)
        self.assertFalse(pose["camera_inside_expanded_bbox"])
        self.assertEqual(pose["nearest_surface"], "left")
        self.assertAlmostEqual(
            pose["nearest_surface_distance_m"],
            3.0 - self.args.bus_bbox_half_width_m,
        )

    def test_perpendicular_pose_inside_bus_is_rejected(self):
        pose = cartesian.classify_bus_pose(0.25, 3.0, 90.0, self.args)
        self.assertTrue(pose["camera_inside_expanded_bbox"])
        self.assertEqual(pose["nearest_surface"], "inside")

    def test_manifest_combines_both_sides_without_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            accepted, rejected = cartesian.write_bus_manifests(
                Path(temp_dir),
                self.args,
                forward_values=[0.25, 7.0],
                relative_yaws=[0.0, 90.0],
            )
            self.assertEqual(len(accepted) + len(rejected), 8)
            self.assertEqual(
                sorted({row["close_right_m"] for row in accepted + rejected}),
                [-3.0, 3.0],
            )
            keys = {
                (
                    row["angle_deg"],
                    row["distance_m"],
                    row["elevation_deg"],
                )
                for row in accepted
            }
            self.assertEqual(len(keys), len(accepted))


class NativeManifestTests(unittest.TestCase):
    def test_signed_irregular_manifest_is_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "views.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["angle_deg", "distance_m", "elevation_deg"],
                )
                writer.writeheader()
                writer.writerows([
                    {"angle_deg": 0, "distance_m": -3.0, "elevation_deg": 0.25},
                    {"angle_deg": 90, "distance_m": 4.0, "elevation_deg": 7.0},
                ])
            args = SimpleNamespace(view_manifest=path)
            views = native.requested_views_from_args(args)
            self.assertEqual(views, [(0, -3.0, 0.25), (90, 4.0, 7.0)])
            self.assertEqual(
                native.requested_views_sha256(views),
                native.requested_views_sha256(list(views)),
            )


if __name__ == "__main__":
    unittest.main()
