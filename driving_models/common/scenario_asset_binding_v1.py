"""
scenario_asset_binding_v1.py

Generic bridge between:

    ResolvedScenarioRuntime
            and
    HEAssetRegistry

This module binds a scenario actor's simulator-independent asset_key
to its concrete CARLA + HE realization.

It deliberately contains NO:
    - CARLA world/spawn code
    - HE rendering code
    - camera configuration
    - NEAT/TCP/model logic
    - route logic
    - collision logic

Architecture
------------
ResolvedScenarioRuntime
        |
        | actor_id
        | asset_key
        | physical state
        v
ScenarioAssetBinder
        |
        +---- physical scenario actor
        |
        +---- CARLA blueprint
        |
        +---- HE production bank

This is the object future generic execution backends consume.
"""

from __future__ import annotations

import argparse

from dataclasses import dataclass

from pathlib import Path

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)


# ============================================================
# Local common modules
# ============================================================

from resolved_scenario_runtime_v1 import (
    ActiveRuntimeActor,
    ResolvedScenarioRuntime,
    RuntimeActorInfo,
    RuntimeActorState,
    RuntimeDimensions,
)

from he_asset_registry_v1 import (
    DEFAULT_MANIFEST,
    HEAssetDefinition,
    HEAssetRegistry,
    resolve_asset_root,
)


# ============================================================
# Bound actor
# ============================================================

@dataclass(
    frozen=True
)
class BoundScenarioActor:
    """
    One active physical scenario actor plus its realization metadata.

    scenario_info:
        Model-independent static scenario information.

    state:
        Exact resolved physical state for this frame.

    asset:
        Concrete CARLA + HE realization resolved from asset_key.
    """

    scenario_info: RuntimeActorInfo

    state: RuntimeActorState

    asset: HEAssetDefinition

    resolved_physical_dimensions: RuntimeDimensions

    physical_dimensions_source: str

    @property
    def actor_id(
        self,
    ) -> str:

        return self.scenario_info.actor_id

    @property
    def asset_key(
        self,
    ) -> str:

        return self.asset.key

    @property
    def carla_blueprint(
        self,
    ) -> str:

        return self.asset.carla_blueprint

    @property
    def view_matrix_csv(
        self,
    ) -> Path:

        return self.asset.view_matrix_csv


# ============================================================
# Compatibility
# ============================================================

def actor_asset_classes_compatible(
    scenario_actor_type: str,
    asset_class: str,
) -> bool:
    """
    Determine whether scenario actor type and registered asset class
    are semantically compatible.

    CARLA / asset generation may distinguish:
        vehicle
        bus
        truck

    while ScenarioSchema intentionally keeps the physical class broad:
        vehicle
        pedestrian
        cyclist
        unknown
    """

    scenario_type = (
        str(
            scenario_actor_type
        )
        .strip()
        .lower()
    )

    asset_type = (
        str(
            asset_class
        )
        .strip()
        .lower()
    )

    vehicle_asset_classes = {
        "vehicle",
        "car",
        "sedan",
        "suv",
        "bus",
        "truck",
        "van",
    }

    pedestrian_asset_classes = {
        "pedestrian",
        "walker",
        "person",
    }

    cyclist_asset_classes = {
        "cyclist",
        "bicycle",
        "bike",
    }

    if scenario_type == "vehicle":

        return (
            asset_type
            in
            vehicle_asset_classes
        )

    if scenario_type == "pedestrian":

        return (
            asset_type
            in
            pedestrian_asset_classes
        )

    if scenario_type == "cyclist":

        return (
            asset_type
            in
            cyclist_asset_classes
        )

    if scenario_type == "unknown":

        return True

    return (
        scenario_type
        ==
        asset_type
    )


# ============================================================
# Binder
# ============================================================

class ScenarioAssetBinder:
    """
    Resolve every scenario actor to an HEAssetDefinition.

    Binding is performed once when the experiment starts.

    Per-frame execution then only retrieves already-resolved objects.
    """

    def __init__(
        self,
        runtime: ResolvedScenarioRuntime,
        registry: HEAssetRegistry,
    ):

        self.runtime = runtime
        self.registry = registry

        self._assets_by_actor_id: Dict[
            str,
            HEAssetDefinition,
        ] = {}

        self._resolved_dimensions_by_actor_id: Dict[
            str,
            RuntimeDimensions,
        ] = {}

        self._dimension_source_by_actor_id: Dict[
            str,
            str,
        ] = {}

        self._validation_errors = []

        self._validation_warnings = []

        self._bind_all()

    # ========================================================
    # Initial binding
    # ========================================================

    def _bind_all(
        self,
    ):

        for actor_id in (
            self.runtime.actor_ids
        ):

            info = (
                self.runtime.actor_info(
                    actor_id
                )
            )

            # ------------------------------------------------
            # Asset key required for execution realization
            # ------------------------------------------------

            if (
                info.asset_key is None
                or
                str(
                    info.asset_key
                ).strip() == ""
            ):

                self._validation_errors.append(
                    f"{actor_id}: "
                    "scenario actor has no asset_key"
                )

                continue

            # ------------------------------------------------
            # Resolve semantic key / alias
            # ------------------------------------------------

            try:

                asset = (
                    self.registry.resolve(
                        info.asset_key
                    )
                )

            except KeyError as exc:

                self._validation_errors.append(
                    f"{actor_id}: "
                    f"cannot resolve asset_key "
                    f"{info.asset_key!r}: "
                    f"{exc}"
                )

                continue

            # ------------------------------------------------
            # Actor-class compatibility
            # ------------------------------------------------

            if not actor_asset_classes_compatible(
                info.actor_type,
                asset.asset_class,
            ):

                self._validation_errors.append(
                    f"{actor_id}: "
                    f"scenario actor_type="
                    f"{info.actor_type!r} "
                    f"is incompatible with "
                    f"asset_class="
                    f"{asset.asset_class!r}"
                )

                continue

            # ------------------------------------------------
            # Physical dimensions
            # ------------------------------------------------
            #
            # Prefer scenario-provided physical dimensions.
            #
            # If they are absent, fall back to the exact physical
            # CARLA bbox stored in asset_metadata.json.
            #
            # This is still physical truth, not HE render geometry.
            # ------------------------------------------------

            resolved_dimensions = None
            dimension_source = None

            if info.dimensions_m is not None:

                resolved_dimensions = info.dimensions_m
                dimension_source = "scenario"

            elif asset.physical_bbox is not None:

                resolved_dimensions = RuntimeDimensions(
                    length_m=asset.physical_bbox.length_m,
                    width_m=asset.physical_bbox.width_m,
                    height_m=asset.physical_bbox.height_m,
                )

                dimension_source = "asset_metadata"

                self._validation_warnings.append(
                    f"{actor_id}: "
                    "scenario dimensions_m missing; using "
                    "asset_metadata physical_bbox dimensions"
                )

            else:

                self._validation_errors.append(
                    f"{actor_id}: "
                    "no physical dimensions available from either "
                    "scenario dimensions_m or asset_metadata physical_bbox"
                )

                continue

            self._assets_by_actor_id[
                actor_id
            ] = asset

            self._resolved_dimensions_by_actor_id[
                actor_id
            ] = resolved_dimensions

            self._dimension_source_by_actor_id[
                actor_id
            ] = dimension_source

    # ========================================================
    # Validation
    # ========================================================

    @property
    def validation_errors(
        self,
    ) -> Tuple[str, ...]:

        return tuple(
            self._validation_errors
        )

    @property
    def validation_warnings(
        self,
    ) -> Tuple[str, ...]:

        return tuple(
            self._validation_warnings
        )

    def is_valid(
        self,
    ) -> bool:

        return (
            len(
                self._validation_errors
            )
            ==
            0
        )

    def require_valid(
        self,
    ):

        if self.is_valid():

            return

        message = [
            "Scenario asset binding failed:"
        ]

        for error in (
            self.validation_errors
        ):

            message.append(
                f"  - {error}"
            )

        raise RuntimeError(
            "\n".join(
                message
            )
        )

    # ========================================================
    # Static actor queries
    # ========================================================

    def asset_for_actor(
        self,
        actor_id: str,
    ) -> HEAssetDefinition:

        if (
            actor_id
            not in
            self._assets_by_actor_id
        ):

            if (
                actor_id
                not in
                self.runtime.actor_ids
            ):

                raise KeyError(
                    f"Unknown scenario actor_id: "
                    f"{actor_id!r}"
                )

            raise RuntimeError(
                f"Actor {actor_id!r} "
                "did not bind to a valid asset."
            )

        return self._assets_by_actor_id[
            actor_id
        ]

    def resolved_dimensions_for_actor(
        self,
        actor_id: str,
    ) -> RuntimeDimensions:

        try:

            return self._resolved_dimensions_by_actor_id[
                actor_id
            ]

        except KeyError as exc:

            raise RuntimeError(
                f"Actor {actor_id!r} "
                "has no resolved physical dimensions."
            ) from exc

    def physical_dimension_source_for_actor(
        self,
        actor_id: str,
    ) -> str:

        try:

            return self._dimension_source_by_actor_id[
                actor_id
            ]

        except KeyError as exc:

            raise RuntimeError(
                f"Actor {actor_id!r} "
                "has no physical dimension source."
            ) from exc

    def view_matrix_for_actor(
        self,
        actor_id: str,
    ) -> Path:

        return (
            self.asset_for_actor(
                actor_id
            )
            .view_matrix_csv
        )

    # ========================================================
    # Per-frame queries
    # ========================================================

    def active_actors(
        self,
        frame_idx: int,
    ) -> Tuple[
        BoundScenarioActor,
        ...
    ]:
        """
        Return every active actor at frame_idx with its concrete
        CARLA + HE asset realization already attached.
        """

        active = []

        for runtime_actor in (
            self.runtime.active_actors(
                frame_idx
            )
        ):

            asset = (
                self.asset_for_actor(
                    runtime_actor
                    .info
                    .actor_id
                )
            )

            active.append(
                BoundScenarioActor(
                    scenario_info=
                        runtime_actor.info,

                    state=
                        runtime_actor.state,

                    asset=
                        asset,

                    resolved_physical_dimensions=
                        self.resolved_dimensions_for_actor(
                            runtime_actor.info.actor_id
                        ),

                    physical_dimensions_source=
                        self.physical_dimension_source_for_actor(
                            runtime_actor.info.actor_id
                        ),
                )
            )

        return tuple(
            active
        )

    # ========================================================
    # Summary
    # ========================================================

    def summary(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        actors = []

        for actor_id in (
            self.runtime.actor_ids
        ):

            info = (
                self.runtime.actor_info(
                    actor_id
                )
            )

            asset = (
                self._assets_by_actor_id
                .get(
                    actor_id
                )
            )

            actors.append({
                "actor_id":
                    actor_id,

                "actor_type":
                    info.actor_type,

                "scenario_asset_key":
                    info.asset_key,

                "canonical_asset_key":
                    (
                        asset.key
                        if asset is not None
                        else None
                    ),

                "carla_blueprint":
                    (
                        asset.carla_blueprint
                        if asset is not None
                        else None
                    ),

                "asset_class":
                    (
                        asset.asset_class
                        if asset is not None
                        else None
                    ),

                "view_count":
                    (
                        asset.view_count
                        if asset is not None
                        else None
                    ),

                "view_matrix_csv":
                    (
                        str(
                            asset.view_matrix_csv
                        )
                        if asset is not None
                        else None
                    ),

                "has_physical_dimensions":
                    (
                        actor_id
                        in
                        self._resolved_dimensions_by_actor_id
                    ),

                "physical_dimensions_source":
                    self._dimension_source_by_actor_id.get(
                        actor_id
                    ),

                "physical_dimensions":
                    (
                        {
                            "length_m":
                                self._resolved_dimensions_by_actor_id[actor_id].length_m,
                            "width_m":
                                self._resolved_dimensions_by_actor_id[actor_id].width_m,
                            "height_m":
                                self._resolved_dimensions_by_actor_id[actor_id].height_m,
                        }
                        if actor_id in self._resolved_dimensions_by_actor_id
                        else None
                    ),
            })

        return {
            "scenario_id":
                self.runtime.scenario_id,

            "scenario_actor_count":
                self.runtime.actor_count(),

            "bound_actor_count":
                len(
                    self._assets_by_actor_id
                ),

            "valid":
                self.is_valid(),

            "errors":
                list(
                    self.validation_errors
                ),

            "warnings":
                list(
                    self.validation_warnings
                ),

            "actors":
                actors,
        }


# ============================================================
# Convenience construction
# ============================================================

def load_bound_scenario(
    resolved_json,
    asset_root,
    manifest_path=DEFAULT_MANIFEST,
):
    """
    Convenience helper intended for future generic runners.
    """

    runtime = (
        ResolvedScenarioRuntime
        .from_json(
            resolved_json
        )
    )

    registry = HEAssetRegistry(
        asset_root=
            asset_root,

        manifest_path=
            manifest_path,
    )

    binder = ScenarioAssetBinder(
        runtime=
            runtime,

        registry=
            registry,
    )

    binder.require_valid()

    return (
        runtime,
        registry,
        binder,
    )


# ============================================================
# CLI smoke test
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "resolved_json",
    )

    parser.add_argument(
        "--asset-root",
        default=None,
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    asset_root = (
        resolve_asset_root(
            args.asset_root
        )
    )

    (
        runtime,
        registry,
        binder,
    ) = load_bound_scenario(
        resolved_json=
            args.resolved_json,

        asset_root=
            asset_root,

        manifest_path=
            args.manifest,
    )

    print()
    print(
        "=" * 78
    )

    print(
        "SCENARIO ASSET BINDING V1"
    )

    print(
        "=" * 78
    )

    print(
        "scenario:",
        runtime.scenario_id,
    )

    print(
        "scenario actors:",
        runtime.actor_count(),
    )

    print(
        "registered assets:",
        len(
            registry
        ),
    )

    print(
        "binding valid:",
        binder.is_valid(),
    )

    print()

    summary = binder.summary()

    for actor in summary[
        "actors"
    ]:

        print(
            actor[
                "actor_id"
            ]
        )

        print(
            "  scenario key:",
            actor[
                "scenario_asset_key"
            ],
        )

        print(
            "  canonical key:",
            actor[
                "canonical_asset_key"
            ],
        )

        print(
            "  class:",
            actor[
                "asset_class"
            ],
        )

        print(
            "  CARLA:",
            actor[
                "carla_blueprint"
            ],
        )

        print(
            "  views:",
            actor[
                "view_count"
            ],
        )

        print(
            "  physical dimensions:",
            (
                "present"
                if actor[
                    "has_physical_dimensions"
                ]
                else
                "missing"
            ),
        )

        print(
            "  physical source:",
            actor[
                "physical_dimensions_source"
            ],
        )

        if actor[
            "physical_dimensions"
        ] is not None:

            dims = actor[
                "physical_dimensions"
            ]

            print(
                "  physical size:",
                f"L={dims['length_m']:.3f} "
                f"W={dims['width_m']:.3f} "
                f"H={dims['height_m']:.3f}"
            )

        print(
            "  HE matrix:",
            actor[
                "view_matrix_csv"
            ],
        )

        print()

    if binder.validation_warnings:

        print(
            "WARNINGS"
        )

        for warning in (
            binder.validation_warnings
        ):

            print(
                " -",
                warning,
            )

        print()

    frame_idx = int(
        args.frame
    )

    print(
        f"ACTIVE ACTORS AT FRAME "
        f"{frame_idx}"
    )

    active = binder.active_actors(
        frame_idx
    )

    if not active:

        print(
            "  none"
        )

    for actor in active:

        print(
            f"  {actor.actor_id}"
        )

        print(
            f"    key="
            f"{actor.asset.key}"
        )

        print(
            f"    blueprint="
            f"{actor.carla_blueprint}"
        )

        print(
            f"    position="
            f"("
            f"{actor.state.x_m:.3f}, "
            f"{actor.state.y_m:.3f}"
            f")"
        )

        print(
            f"    yaw="
            f"{actor.state.yaw_deg:.3f}"
        )

        print(
            f"    speed="
            f"{actor.state.speed_mps:.3f}"
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()