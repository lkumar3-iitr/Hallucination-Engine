"""
he_asset_registry_v1.py

Generic asset registry for Hallucination Engine experiments.

Purpose
-------
Map a simulator-independent scenario asset key to:

    1. the CARLA realization
    2. the HE production sprite bank

without putting asset-specific paths or CARLA blueprint names inside:

    - ScenarioGenerator
    - driving-model adapters
    - experiment runners
    - HE renderer

Architecture
------------
Scenario actor
    asset_key="vehicle.passenger_01"
              |
              v
        HEAssetRegistry
          /        \\
         /          \\
        v            v
CARLA blueprint    HE view matrix
                      |
                      v
                generic renderer


The large sprite banks remain outside the Git repository.

Example external root:

D:\\HallucinationEngine-asset\\HE_v_0.1\\assets\\
    sprite_bank_native_production\\

Adding a new asset should normally require only:

    1. placing its production bank under the asset root
    2. adding one entry to he_asset_manifest_v1.json

No NEAT/TCP/model-specific code should change.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from dataclasses import dataclass

from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Tuple,
)


# ============================================================
# Defaults
# ============================================================

THIS_FILE = Path(
    __file__
).resolve()

DEFAULT_MANIFEST = (
    THIS_FILE.parent
    /
    "he_asset_manifest_v1.json"
)

ASSET_ROOT_ENV = (
    "HE_NATIVE_ASSET_ROOT"
)


# ============================================================
# Runtime asset definition
# ============================================================
@dataclass(
    frozen=True
)
class AssetPhysicalBBox:
    """
    Exact physical CARLA bounding box metadata stored with the
    generated production asset bank.

    This is physical actor geometry, not HE rendering geometry.
    """

    extent_x_m: float
    extent_y_m: float
    extent_z_m: float

    local_center_x_m: float
    local_center_y_m: float
    local_center_z_m: float
    local_bottom_z_m: float

    length_m: float
    width_m: float
    height_m: float
@dataclass(
    frozen=True
)
class HEAssetDefinition:
    """
    Fully resolved asset definition.

    Note
    ----
    Physical scenario dimensions are intentionally NOT required here.

    Physical dimensions belong to scenario truth.

    This registry describes how that actor is realized by CARLA and HE.
    """

    key: str

    aliases: Tuple[
        str,
        ...
    ]

    asset_dir: Path

    view_matrix_csv: Path

    close_view_matrix_csvs: Dict[
        str,
        Path,
    ]

    asset_metadata_json: Optional[
        Path
    ]

    generation_config_json: Optional[
        Path
    ]

    qa_summary_json: Optional[
        Path
    ]

    asset_id: str

    asset_class: str

    carla_blueprint: str

    view_count: int

    generation_statuses: Tuple[
        str,
        ...
    ]

    angles_deg: Tuple[
        float,
        ...
    ]

    distances_m: Tuple[
        float,
        ...
    ]

    elevations_deg: Tuple[
        float,
        ...
    ]

    target_heights_m: Tuple[
        float,
        ...
    ]

    image_widths_px: Tuple[
        int,
        ...
    ]

    image_heights_px: Tuple[
        int,
        ...
    ]

    capture_fovs_deg: Tuple[
        float,
        ...
    ]

    physical_bbox: Optional[
        AssetPhysicalBBox
    ]

    manifest_data: Dict[
        str,
        Any,
    ]

    def has_complete_generation(
        self,
    ) -> bool:

        if not self.generation_statuses:

            return True

        return all(
            status.lower()
            ==
            "complete"

            for status
            in self.generation_statuses
        )

    def summary(
        self,
    ) -> Dict[str, Any]:

        return {
            "key":
                self.key,

            "aliases":
                list(
                    self.aliases
                ),

            "asset_id":
                self.asset_id,

            "asset_class":
                self.asset_class,

            "carla_blueprint":
                self.carla_blueprint,

            "asset_dir":
                str(
                    self.asset_dir
                ),

            "view_matrix_csv":
                str(
                    self.view_matrix_csv
                ),

            "close_view_matrix_csvs": {
                side: str(path)
                for side, path
                in self.close_view_matrix_csvs.items()
            },

            "view_count":
                self.view_count,

            "generation_statuses":
                list(
                    self.generation_statuses
                ),

            "complete":
                self.has_complete_generation(),

            "angles":
                len(
                    self.angles_deg
                ),

            "distances_m":
                list(
                    self.distances_m
                ),

            "elevations_deg":
                list(
                    self.elevations_deg
                ),

            "target_heights_m":
                list(
                    self.target_heights_m
                ),

            "image_widths_px":
                list(
                    self.image_widths_px
                ),

            "image_heights_px":
                list(
                    self.image_heights_px
                ),

            "capture_fovs_deg":
                list(
                    self.capture_fovs_deg
                ),

            "physical_bbox":
                (
                    {
                        "extent_x_m":
                            self.physical_bbox.extent_x_m,
                        "extent_y_m":
                            self.physical_bbox.extent_y_m,
                        "extent_z_m":
                            self.physical_bbox.extent_z_m,
                        "local_center_x_m":
                            self.physical_bbox.local_center_x_m,
                        "local_center_y_m":
                            self.physical_bbox.local_center_y_m,
                        "local_center_z_m":
                            self.physical_bbox.local_center_z_m,
                        "local_bottom_z_m":
                            self.physical_bbox.local_bottom_z_m,
                        "length_m":
                            self.physical_bbox.length_m,
                        "width_m":
                            self.physical_bbox.width_m,
                        "height_m":
                            self.physical_bbox.height_m,
                    }
                    if self.physical_bbox is not None
                    else None
                ),
        }


# ============================================================
# CSV helpers
# ============================================================

def _clean_string(
    value,
) -> str:

    if value is None:

        return ""

    return str(
        value
    ).strip()


def _float_or_none(
    value,
):

    text = _clean_string(
        value
    )

    if text == "":

        return None

    try:

        return float(
            text
        )

    except ValueError:

        return None


def _int_or_none(
    value,
):

    number = _float_or_none(
        value
    )

    if number is None:

        return None

    return int(
        round(
            number
        )
    )


def _unique_strings(
    rows,
    key,
) -> Tuple[str, ...]:

    values = {
        _clean_string(
            row.get(
                key
            )
        )

        for row in rows
    }

    values.discard(
        ""
    )

    return tuple(
        sorted(
            values
        )
    )


def _unique_floats(
    rows,
    key,
) -> Tuple[float, ...]:

    values = set()

    for row in rows:

        value = _float_or_none(
            row.get(
                key
            )
        )

        if value is not None:

            values.add(
                float(
                    value
                )
            )

    return tuple(
        sorted(
            values
        )
    )


def _unique_ints(
    rows,
    key,
) -> Tuple[int, ...]:

    values = set()

    for row in rows:

        value = _int_or_none(
            row.get(
                key
            )
        )

        if value is not None:

            values.add(
                int(
                    value
                )
            )

    return tuple(
        sorted(
            values
        )
    )


def _single_required_string(
    rows,
    key,
    source,
) -> str:

    values = _unique_strings(
        rows,
        key,
    )

    if not values:

        raise ValueError(
            f"{source}: "
            f"missing required CSV field {key!r}"
        )

    if len(
        values
    ) != 1:

        raise ValueError(
            f"{source}: "
            f"expected one {key!r}, "
            f"found {values}"
        )

    return values[
        0
    ]


# ============================================================
# JSON helper
# ============================================================

def _load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as fp:

        return json.load(
            fp
        )

def _parse_physical_bbox_from_metadata(
    metadata,
):
    """
    Parse exact physical bbox dimensions from asset_metadata.json.

    Returns None if the metadata file or physical_bbox block is missing.
    """

    if not isinstance(
        metadata,
        dict,
    ):

        return None

    physical_bbox = metadata.get(
        "physical_bbox"
    )

    if not isinstance(
        physical_bbox,
        dict,
    ):

        return None

    required = [
        "extent_x_m",
        "extent_y_m",
        "extent_z_m",
        "local_center_x_m",
        "local_center_y_m",
        "local_center_z_m",
        "local_bottom_z_m",
        "length_m",
        "width_m",
        "height_m",
    ]

    for key in required:

        if key not in physical_bbox:

            raise ValueError(
                "asset_metadata.json physical_bbox is missing "
                f"required field {key!r}"
            )

    return AssetPhysicalBBox(
        extent_x_m=float(
            physical_bbox["extent_x_m"]
        ),
        extent_y_m=float(
            physical_bbox["extent_y_m"]
        ),
        extent_z_m=float(
            physical_bbox["extent_z_m"]
        ),
        local_center_x_m=float(
            physical_bbox["local_center_x_m"]
        ),
        local_center_y_m=float(
            physical_bbox["local_center_y_m"]
        ),
        local_center_z_m=float(
            physical_bbox["local_center_z_m"]
        ),
        local_bottom_z_m=float(
            physical_bbox["local_bottom_z_m"]
        ),
        length_m=float(
            physical_bbox["length_m"]
        ),
        width_m=float(
            physical_bbox["width_m"]
        ),
        height_m=float(
            physical_bbox["height_m"]
        ),
    )
# ============================================================
# Registry
# ============================================================

class HEAssetRegistry:
    """
    Registry joining semantic scenario assets with external HE banks.

    Lookup accepts:
        - canonical semantic key
        - manifest alias
        - asset_id
        - production folder name
        - CARLA blueprint

    The last two are useful for compatibility/debugging, but new
    ScenarioGenerator scenarios should prefer canonical semantic keys.
    """

    def __init__(
        self,
        asset_root,
        manifest_path=DEFAULT_MANIFEST,
    ):

        self.asset_root = Path(
            asset_root
        ).expanduser().resolve()

        self.manifest_path = Path(
            manifest_path
        ).expanduser().resolve()

        if not self.asset_root.exists():

            raise FileNotFoundError(
                "HE production asset root does not exist:\n"
                f"{self.asset_root}"
            )

        if not self.asset_root.is_dir():

            raise NotADirectoryError(
                "HE production asset root is not a directory:\n"
                f"{self.asset_root}"
            )

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                "HE asset manifest does not exist:\n"
                f"{self.manifest_path}"
            )

        manifest = _load_json(
            self.manifest_path
        )

        schema_version = str(
            manifest.get(
                "schema_version",
                "",
            )
        )

        if schema_version != "1.0":

            raise ValueError(
                "Unsupported HE asset manifest schema: "
                f"{schema_version!r}"
            )

        entries = manifest.get(
            "assets",
            [],
        )

        if not isinstance(
            entries,
            list,
        ):

            raise ValueError(
                "Manifest field 'assets' "
                "must be a list."
            )

        self._assets: Dict[
            str,
            HEAssetDefinition,
        ] = {}

        self._lookup: Dict[
            str,
            HEAssetDefinition,
        ] = {}

        for entry in entries:

            asset = self._load_asset(
                entry
            )

            if asset.key in self._assets:

                raise ValueError(
                    "Duplicate canonical asset key: "
                    f"{asset.key!r}"
                )

            self._assets[
                asset.key
            ] = asset

            lookup_names = {
                asset.key,
                asset.asset_id,
                asset.asset_dir.name,
                asset.carla_blueprint,
            }

            lookup_names.update(
                asset.aliases
            )

            for lookup_name in lookup_names:

                lookup_name = (
                    str(
                        lookup_name
                    )
                    .strip()
                )

                if lookup_name == "":

                    continue

                existing = self._lookup.get(
                    lookup_name
                )

                if (
                    existing is not None
                    and
                    existing.key
                    !=
                    asset.key
                ):

                    raise ValueError(
                        "Asset lookup alias collision: "
                        f"{lookup_name!r} maps to both "
                        f"{existing.key!r} and "
                        f"{asset.key!r}"
                    )

                self._lookup[
                    lookup_name
                ] = asset

    # ========================================================
    # Asset loading
    # ========================================================

    def _load_asset(
        self,
        entry,
    ) -> HEAssetDefinition:

        if not isinstance(
            entry,
            dict,
        ):

            raise ValueError(
                "Each manifest asset entry "
                "must be an object."
            )

        key = _clean_string(
            entry.get(
                "key"
            )
        )

        if key == "":

            raise ValueError(
                "Manifest asset entry "
                "is missing 'key'."
            )

        folder = _clean_string(
            entry.get(
                "folder"
            )
        )

        if folder == "":

            raise ValueError(
                f"{key}: manifest entry "
                "is missing 'folder'."
            )

        aliases_raw = entry.get(
            "aliases",
            [],
        )

        if aliases_raw is None:

            aliases_raw = []

        if not isinstance(
            aliases_raw,
            list,
        ):

            raise ValueError(
                f"{key}: aliases must be a list."
            )

        aliases = tuple(
            _clean_string(
                alias
            )

            for alias
            in aliases_raw

            if _clean_string(
                alias
            )
        )

        asset_dir = (
            self.asset_root
            /
            folder
        ).resolve()

        if not asset_dir.exists():

            raise FileNotFoundError(
                f"{key}: production asset folder "
                f"does not exist:\n{asset_dir}"
            )

        view_matrix_csv = (
            asset_dir
            /
            "view_matrix.csv"
        )

        if not view_matrix_csv.exists():

            raise FileNotFoundError(
                f"{key}: missing view_matrix.csv:\n"
                f"{view_matrix_csv}"
            )

        close_view_matrix_csvs = {}
        close_banks = entry.get(
            "close_banks",
            {},
        )
        if close_banks is None:
            close_banks = {}
        if not isinstance(close_banks, dict):
            raise ValueError(
                f"{key}: close_banks must be an object."
            )
        for side, close_folder in close_banks.items():
            side_key = _clean_string(side).lower()
            folder_text = _clean_string(close_folder)
            if not side_key or not folder_text:
                continue
            close_dir = (
                self.asset_root
                /
                folder_text
            ).resolve()
            close_csv = close_dir / "view_matrix.csv"
            if not close_csv.exists():
                raise FileNotFoundError(
                    f"{key}: missing close-bank view_matrix.csv "
                    f"for {side_key}:\n{close_csv}"
                )
            close_view_matrix_csvs[side_key] = close_csv

        # ----------------------------------------------------
        # Read production view matrix
        # ----------------------------------------------------

        with view_matrix_csv.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as fp:

            rows = list(
                csv.DictReader(
                    fp
                )
            )

        if not rows:

            raise ValueError(
                f"{key}: empty view_matrix.csv"
            )

        asset_id = (
            _single_required_string(
                rows,
                "asset_id",
                view_matrix_csv,
            )
        )

        asset_class = (
            _single_required_string(
                rows,
                "asset_class",
                view_matrix_csv,
            )
        )

        carla_blueprint = (
            _single_required_string(
                rows,
                "carla_blueprint",
                view_matrix_csv,
            )
        )

        statuses = _unique_strings(
            rows,
            "generation_status",
        )

        angles = _unique_floats(
            rows,
            "angle_deg",
        )

        distances = _unique_floats(
            rows,
            "distance_m",
        )

        elevations = _unique_floats(
            rows,
            "elevation_deg",
        )

        target_heights = _unique_floats(
            rows,
            "target_height_m",
        )

        image_widths = _unique_ints(
            rows,
            "image_width_px",
        )

        image_heights = _unique_ints(
            rows,
            "image_height_px",
        )

        capture_fovs = _unique_floats(
            rows,
            "fov_deg",
        )

        # ----------------------------------------------------
        # Sidecars
        # ----------------------------------------------------

        metadata_path = (
            asset_dir
            /
            "asset_metadata.json"
        )

        generation_config_path = (
            asset_dir
            /
            "generation_config.json"
        )

        qa_summary_path = (
            asset_dir
            /
            "qa_summary.json"
        )

        metadata_json = (
            _load_json(
                metadata_path
            )
            if metadata_path.exists()
            else None
        )

        physical_bbox = (
            _parse_physical_bbox_from_metadata(
                metadata_json
            )
        )

        return HEAssetDefinition(
            key=key,

            aliases=aliases,

            asset_dir=asset_dir,

            view_matrix_csv=
                view_matrix_csv,

            close_view_matrix_csvs=
                close_view_matrix_csvs,

            asset_metadata_json=(
                metadata_path
                if metadata_path.exists()
                else None
            ),

            generation_config_json=(
                generation_config_path
                if generation_config_path.exists()
                else None
            ),

            qa_summary_json=(
                qa_summary_path
                if qa_summary_path.exists()
                else None
            ),

            asset_id=asset_id,

            asset_class=
                asset_class,

            carla_blueprint=
                carla_blueprint,

            view_count=len(
                rows
            ),

            generation_statuses=
                statuses,

            angles_deg=
                angles,

            distances_m=
                distances,

            elevations_deg=
                elevations,

            target_heights_m=
                target_heights,

            image_widths_px=
                image_widths,

            image_heights_px=
                image_heights,

            capture_fovs_deg=
                capture_fovs,

            physical_bbox=
                physical_bbox,

            manifest_data=dict(
                entry
            ),
        )

    # ========================================================
    # Queries
    # ========================================================

    @property
    def canonical_keys(
        self,
    ) -> Tuple[str, ...]:

        return tuple(
            self._assets.keys()
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self._assets
        )

    def __contains__(
        self,
        asset_key,
    ) -> bool:

        return (
            str(
                asset_key
            )
            in
            self._lookup
        )

    def resolve(
        self,
        asset_key,
    ) -> HEAssetDefinition:

        lookup_key = str(
            asset_key
        ).strip()

        try:

            return self._lookup[
                lookup_key
            ]

        except KeyError as exc:

            raise KeyError(
                "Unknown HE asset key "
                f"{lookup_key!r}. "
                "Known canonical keys: "
                f"{', '.join(self.canonical_keys)}"
            ) from exc

    def get(
        self,
        asset_key,
        default=None,
    ):

        return self._lookup.get(
            str(
                asset_key
            ).strip(),
            default,
        )

    def assets(
        self,
    ) -> Tuple[
        HEAssetDefinition,
        ...
    ]:

        return tuple(
            self._assets.values()
        )

    def total_views(
        self,
    ) -> int:

        return sum(
            asset.view_count
            for asset
            in self._assets.values()
        )

    # ========================================================
    # Validation
    # ========================================================

    def validate(
        self,
    ) -> Tuple[str, ...]:
        """
        Return human-readable validation problems.

        Empty tuple means registry validation passed.
        """

        problems = []

        for asset in self.assets():

            if not asset.has_complete_generation():

                problems.append(
                    f"{asset.key}: "
                    "generation status is not completely "
                    f"'complete': "
                    f"{asset.generation_statuses}"
                )

            if not asset.angles_deg:

                problems.append(
                    f"{asset.key}: no angles"
                )

            if not asset.distances_m:

                problems.append(
                    f"{asset.key}: no distances"
                )

            if not asset.elevations_deg:

                problems.append(
                    f"{asset.key}: no elevations"
                )

            rgba_dir = (
                asset.asset_dir
                /
                "rgba"
            )

            if not rgba_dir.exists():

                problems.append(
                    f"{asset.key}: "
                    f"missing RGBA directory "
                    f"{rgba_dir}"
                )

        return tuple(
            problems
        )

    # ========================================================
    # Summary
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, Any]:

        return {
            "asset_root":
                str(
                    self.asset_root
                ),

            "manifest":
                str(
                    self.manifest_path
                ),

            "asset_count":
                len(
                    self
                ),

            "total_views":
                self.total_views(),

            "assets": [
                asset.summary()
                for asset
                in self.assets()
            ],

            "validation_problems":
                list(
                    self.validate()
                ),
        }


# ============================================================
# Asset root resolution
# ============================================================

def resolve_asset_root(
    command_line_root=None,
) -> Path:

    if command_line_root:

        return Path(
            command_line_root
        ).expanduser().resolve()

    env_root = os.environ.get(
        ASSET_ROOT_ENV,
        "",
    ).strip()

    if env_root:

        return Path(
            env_root
        ).expanduser().resolve()

    raise RuntimeError(
        "HE production asset root was not specified.\n\n"
        "Either pass:\n"
        "  --asset-root <path>\n\n"
        "or set environment variable:\n"
        f"  {ASSET_ROOT_ENV}=<path>"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--asset-root",
        default=None,
        help=(
            "Production bank root. "
            "If omitted, HE_NATIVE_ASSET_ROOT is used."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    parser.add_argument(
        "--lookup",
        default=None,
        help=(
            "Optional semantic key / alias / "
            "CARLA blueprint to resolve."
        ),
    )

    args = parser.parse_args()

    root = resolve_asset_root(
        args.asset_root
    )

    registry = HEAssetRegistry(
        asset_root=root,
        manifest_path=
            args.manifest,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "HE ASSET REGISTRY V1"
    )

    print(
        "=" * 78
    )

    print(
        "root:",
        registry.asset_root,
    )

    print(
        "assets:",
        len(
            registry
        ),
    )

    print(
        "total views:",
        registry.total_views(),
    )

    print()

    for asset in registry.assets():

        print(
            f"{asset.key}"
        )

        print(
            f"  folder:    "
            f"{asset.asset_dir.name}"
        )

        print(
            f"  asset_id:  "
            f"{asset.asset_id}"
        )

        print(
            f"  class:     "
            f"{asset.asset_class}"
        )

        print(
            f"  blueprint: "
            f"{asset.carla_blueprint}"
        )

        print(
            f"  views:     "
            f"{asset.view_count}"
        )

        print(
            f"  status:    "
            f"{asset.generation_statuses}"
        )

        print(
            f"  angles:    "
            f"{len(asset.angles_deg)}"
        )

        print(
            f"  distances: "
            f"{list(asset.distances_m)}"
        )

        print(
            f"  elevation: "
            f"{list(asset.elevations_deg)}"
        )

        if asset.physical_bbox is not None:

            print(
                f"  physical: "
                f"L={asset.physical_bbox.length_m:.3f} "
                f"W={asset.physical_bbox.width_m:.3f} "
                f"H={asset.physical_bbox.height_m:.3f}"
            )

        print()

    problems = registry.validate()

    if problems:

        print(
            "VALIDATION PROBLEMS"
        )

        for problem in problems:

            print(
                " -",
                problem,
            )

    else:

        print(
            "validation: PASS"
        )

    if args.lookup is not None:

        asset = registry.resolve(
            args.lookup
        )

        print()
        print(
            "LOOKUP"
        )

        print(
            f"  requested:  "
            f"{args.lookup}"
        )

        print(
            f"  canonical:  "
            f"{asset.key}"
        )

        print(
            f"  blueprint:  "
            f"{asset.carla_blueprint}"
        )

        print(
            f"  view matrix:"
        )

        print(
            f"    "
            f"{asset.view_matrix_csv}"
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()
