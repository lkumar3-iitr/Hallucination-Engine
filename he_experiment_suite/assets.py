"""Validate the external binary asset release used by HE."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_CONFIG = Path(__file__).with_name("configs") / "assets.json"
ASSET_LOCK = Path(__file__).with_name("assets") / "asset_release_v1.lock.json"


@dataclass(frozen=True)
class ValidationItem:
    name: str
    path: Path
    required: bool
    exists: bool


def load_asset_config() -> dict:
    return json.loads(ASSET_CONFIG.read_text(encoding="utf-8"))


def validate_assets(asset_root: Path, release: bool = True) -> list[ValidationItem]:
    config = load_asset_config()
    layout = config["asset_root_layout"]
    native_root = asset_root / layout["native_bank_root"]
    close_root = asset_root / layout["close_bank_root"]
    items: list[ValidationItem] = []

    for name, spec in config["assets"].items():
        manifest = REPO_ROOT / spec["manifest"]
        native = native_root / spec["native_bank"]
        items.append(ValidationItem(f"{name}:manifest", manifest, True, manifest.is_file()))
        items.append(ValidationItem(f"{name}:native", native, True, native.is_dir()))

        selector = native / "selector_teacher_v1.npz"
        selector_required = release and bool(spec.get("require_distilled_selector"))
        items.append(
            ValidationItem(f"{name}:selector", selector, selector_required, selector.is_file())
        )
        for bank in spec.get("close_banks", []):
            path = close_root / bank
            items.append(ValidationItem(f"{name}:close:{bank}", path, release, path.is_dir()))
    return items


def missing_required(items: list[ValidationItem]) -> list[ValidationItem]:
    return [item for item in items if item.required and not item.exists]


def validate_locked_identities(asset_root: Path) -> list[ValidationItem]:
    """Check small identity-critical files without hashing every sprite PNG."""
    lock = json.loads(ASSET_LOCK.read_text(encoding="utf-8"))
    items: list[ValidationItem] = []
    for bank_relative, bank in lock["banks"].items():
        bank_path = asset_root / bank_relative
        for relative, expected in bank["sha256"].items():
            path = bank_path / relative
            matches = False
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                matches = digest == expected
            items.append(
                ValidationItem(f"lock:{bank_relative}:{relative}", path, True, matches)
            )
    return items
