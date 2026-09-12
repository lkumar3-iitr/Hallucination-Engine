"""Asset-bank binding and generic entry point regression tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astra.calibrated_asset_renderer import CalibratedAssetRenderer, CalibratedBusRenderer


class AssetBindingTests(unittest.TestCase):
    def test_failed_pedestrian_candidate_is_not_enabled(self):
        manifest = json.loads((Path(__file__).parent/"calibrated_assets_candidate_v1.json").read_text())
        entries = {entry["blueprint"]: entry for entry in manifest["assets"]}
        self.assertFalse(entries["walker.pedestrian.0001"]["enabled"])
        self.assertTrue(entries["vehicle.tesla.model3"].get("enabled", True))

    def test_rejects_wrong_asset_before_loading_hull(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"asset_metadata.json").write_text(json.dumps({"carla_blueprint": "vehicle.tesla.model3"}))
            (path/"config.json").write_text(json.dumps({"blueprint": "walker.pedestrian.0001"}))
            with self.assertRaisesRegex(ValueError, "different actor"):
                CalibratedAssetRenderer(path, path)

    def test_each_asset_uses_same_rendering_core(self):
        for blueprint in ("vehicle.tesla.model3", "vehicle.nissan.patrol_2021",
                          "walker.pedestrian.0001", "vehicle.mitsubishi.fusorosa"):
            with self.subTest(blueprint=blueprint), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path/"asset_metadata.json").write_text(json.dumps({"carla_blueprint": blueprint}))
                (path/"config.json").write_text(json.dumps({"blueprint": blueprint}))
                with patch.object(CalibratedBusRenderer, "__init__", return_value=None) as initialize:
                    CalibratedAssetRenderer(path, path)
                    initialize.assert_called_once_with(path, path)
                self.assertIs(CalibratedAssetRenderer.render, CalibratedBusRenderer.render)


if __name__ == "__main__":
    unittest.main()
