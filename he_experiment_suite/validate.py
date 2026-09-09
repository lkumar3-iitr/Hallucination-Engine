"""Validate code, scenario, adapter, and external-asset dependencies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assets import (
    REPO_ROOT,
    missing_required,
    validate_assets,
    validate_locked_identities,
)


CONFIG_ROOT = Path(__file__).with_name("configs")


def _load(name: str) -> dict:
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def validate_repository() -> list[tuple[str, Path, bool]]:
    checks: list[tuple[str, Path, bool]] = []
    required = {
        "renderer": "he_renderer/renderer.py",
        "compositor": "he_renderer/compositor.py",
        "scenario_generator": "ScenarioGenerator/scenario_generator/__init__.py",
        "closed_loop_runner": "driving_models/common/generic_he_closed_loop_runner_v1.py",
        "pair_recorder": "driving_models/common/record_scenario_carla_he_pair_v3.py",
    }
    required.update(
        {
            f"adapter:{name}": spec["adapter"]
            for name, spec in _load("models.json")["models"].items()
        }
    )
    required.update(
        {
            f"scenario:{name}": path
            for name, path in _load("scenarios.json")["scenarios"].items()
        }
    )
    for name, relative in required.items():
        path = REPO_ROOT / relative
        checks.append((name, path, path.exists()))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("release", "native-only"), default="release"
    )
    args = parser.parse_args(argv)

    failed = False
    for name, path, exists in validate_repository():
        print(f"[{'OK' if exists else 'MISSING'}] {name}: {path}")
        failed |= not exists

    asset_items = validate_assets(args.asset_root.resolve(), args.profile == "release")
    for item in asset_items:
        state = "OK" if item.exists else ("MISSING" if item.required else "OPTIONAL-MISSING")
        print(f"[{state}] {item.name}: {item.path}")
    failed |= bool(missing_required(asset_items))

    if args.profile == "release":
        lock_items = validate_locked_identities(args.asset_root.resolve())
        for item in lock_items:
            print(f"[{'OK' if item.exists else 'HASH-MISMATCH'}] {item.name}: {item.path}")
        failed |= bool(missing_required(lock_items))

    print("VALIDATION PASSED" if not failed else "VALIDATION FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
