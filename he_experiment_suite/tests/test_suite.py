import tempfile
import unittest
from pathlib import Path

from he_experiment_suite.assets import missing_required, validate_assets
from he_experiment_suite.cli import forwarded_args
from he_experiment_suite.validate import validate_repository


class FinalSuiteTest(unittest.TestCase):
    def test_repository_dependencies_exist(self):
        missing = [(name, path) for name, path, exists in validate_repository() if not exists]
        self.assertEqual([], missing)

    def test_empty_asset_root_fails_release_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            checks = validate_assets(Path(temporary), release=True)
        self.assertTrue(missing_required(checks))

    def test_close_banks_are_optional_in_native_only_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            checks = validate_assets(Path(temporary), release=False)
        close_checks = [check for check in checks if ":close:" in check.name]
        self.assertTrue(close_checks)
        self.assertTrue(all(not check.required for check in close_checks))

    def test_forwarding_separator_is_not_sent_to_child(self):
        self.assertEqual(["--resolved", "case.json"], forwarded_args(["--", "--resolved", "case.json"]))
        self.assertEqual(["--resolved", "case.json"], forwarded_args(["--resolved", "case.json"]))


if __name__ == "__main__":
    unittest.main()
