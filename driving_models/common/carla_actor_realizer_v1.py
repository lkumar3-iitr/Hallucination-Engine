"""
carla_actor_realizer_v1.py

Generic CARLA realization backend for resolved scenario actors.

Purpose
-------
Take backend-independent ExecutionActorState objects and realize them
as CARLA actors.

This module handles:
    - arbitrary number of scenario actors
    - lifecycle spawn/despawn
    - semantic asset -> CARLA blueprint already resolved upstream
    - exact scripted transforms
    - deterministic kinematic actor motion
    - actor-type-aware ground-height resolution

It deliberately contains NO:
    - NEAT logic
    - TCP logic
    - model cameras
    - model inference
    - HE compositing
    - route/safety metrics

Expected experiment timing
--------------------------
Before world.tick() for frame N:

    realizer.apply_frame(N)

Then:

    world.tick()

Therefore all RGB/depth sensors produced by that tick observe the
scenario actors at resolved frame N.

This is the same timing convention used by the validated
single-adversary NEAT experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import carla

from he_asset_registry_v1 import (
    DEFAULT_MANIFEST,
)

from scenario_execution_runtime_v1 import (
    ExecutionActorState,
    ExecutionWorldOrigin,
    load_execution_runtime,
)


# ============================================================
# Realized actor information
# ============================================================

@dataclass
class CarlaRealizedActor:

    actor_id: str

    carla_actor: object

    blueprint_id: str

    spawned_frame_idx: int

    last_applied_frame_idx: int


# ============================================================
# Grounding
# ============================================================

def _asset_carla_options(
    execution,
    actor_id: str,
):
    """
    Optional backend configuration may later be placed in the
    manifest as:

        "carla": {
            "grounding_mode": "driving_lane",
            "z_offset_m": 0.05,
            "attributes": {
                "color": "0,0,255"
            }
        }

    No such fields are required.
    """

    asset = (
        execution
        .binder
        .asset_for_actor(
            actor_id
        )
    )

    raw = (
        asset
        .manifest_data
        .get(
            "carla",
            {},
        )
    )

    if not isinstance(
        raw,
        dict,
    ):

        return {}

    return raw


def _default_grounding_mode(
    actor_state: ExecutionActorState,
):

    actor_type = (
        str(
            actor_state.actor_type
        )
        .strip()
        .lower()
    )

    if actor_type == "vehicle":

        return "driving_lane"

    if actor_type == "pedestrian":

        return "nearest_lane"

    if actor_type == "cyclist":

        return "nearest_lane"

    return "origin"


def _default_z_offset_m(
    actor_state: ExecutionActorState,
):

    actor_type = (
        str(
            actor_state.actor_type
        )
        .strip()
        .lower()
    )

    if actor_type == "vehicle":

        return 0.05

    # Walkers should normally use the lane/sidewalk surface
    # directly rather than receiving the vehicle suspension
    # clearance used above.
    return 0.0


def resolve_actor_ground_z(
    world,
    actor_state: ExecutionActorState,
    execution,
):
    """
    Resolve Z only.

    IMPORTANT:
    Scenario x/y are never replaced with waypoint x/y.

    This keeps the exact ScenarioGenerator physical trajectory while
    allowing the 2D scenario schema to be grounded to the CARLA map.
    """

    options = _asset_carla_options(
        execution,
        actor_state.actor_id,
    )

    mode = str(
        options.get(
            "grounding_mode",
            _default_grounding_mode(
                actor_state
            ),
        )
    ).strip().lower()

    z_offset_m = float(
        options.get(
            "z_offset_m",
            _default_z_offset_m(
                actor_state
            ),
        )
    )

    location = carla.Location(
        x=float(
            actor_state.world_x_m
        ),

        y=float(
            actor_state.world_y_m
        ),

        z=float(
            actor_state.world_z_m
        ),
    )

    carla_map = world.get_map()

    waypoint = None

    if mode == "driving_lane":

        waypoint = (
            carla_map
            .get_waypoint(
                location,

                project_to_road=True,

                lane_type=(
                    carla.LaneType.Driving
                ),
            )
        )

    elif mode == "nearest_lane":

        waypoint = (
            carla_map
            .get_waypoint(
                location,

                project_to_road=True,

                lane_type=(
                    carla.LaneType.Any
                ),
            )
        )

    elif mode == "origin":

        waypoint = None

    else:

        raise ValueError(
            f"{actor_state.actor_id}: "
            f"unsupported grounding_mode={mode!r}"
        )

    if waypoint is not None:

        base_z = float(
            waypoint
            .transform
            .location
            .z
        )

    else:

        base_z = float(
            actor_state.world_z_m
        )

    return (
        base_z
        +
        z_offset_m
    )


# ============================================================
# CARLA transform
# ============================================================

def execution_state_to_carla_transform(
    world,
    actor_state: ExecutionActorState,
    execution,
):
    """
    Convert already-world-resolved numeric execution state into
    CARLA Transform.

    SG -> world conversion does NOT happen here.
    That is owned by ScenarioExecutionRuntime.
    """

    z_m = resolve_actor_ground_z(
        world=world,

        actor_state=
            actor_state,

        execution=
            execution,
    )

    return carla.Transform(
        carla.Location(
            x=float(
                actor_state.world_x_m
            ),

            y=float(
                actor_state.world_y_m
            ),

            z=float(
                z_m
            ),
        ),

        carla.Rotation(
            pitch=0.0,

            yaw=float(
                actor_state.world_yaw_deg
            ),

            roll=0.0,
        ),
    )


# ============================================================
# Realizer
# ============================================================

class CarlaActorRealizer:
    """
    Maintain the set of physical CARLA actors corresponding to the
    currently active resolved scenario actors.
    """

    def __init__(
        self,
        world,
        execution,
    ):

        self.world = world

        self.execution = execution

        self._actors: Dict[
            str,
            CarlaRealizedActor,
        ] = {}

    # ========================================================
    # Query
    # ========================================================

    @property
    def actor_ids(
        self,
    ) -> Tuple[str, ...]:

        return tuple(
            self._actors.keys()
        )

    def get_actor(
        self,
        actor_id: str,
    ):

        realized = (
            self._actors.get(
                actor_id
            )
        )

        if realized is None:

            return None

        return realized.carla_actor

    # ========================================================
    # Blueprint
    # ========================================================

    def _prepare_blueprint(
        self,
        actor_state: ExecutionActorState,
    ):

        blueprint = (
            self.world
            .get_blueprint_library()
            .find(
                actor_state.carla_blueprint
            )
        )

        if blueprint.has_attribute(
            "role_name"
        ):

            blueprint.set_attribute(
                "role_name",
                (
                    "scenario_"
                    +
                    actor_state.actor_id
                ),
            )

        # ----------------------------------------------------
        # Optional asset-specific CARLA attributes.
        #
        # Example manifest:
        #
        # "carla": {
        #   "attributes": {
        #       "color": "0,0,255"
        #   }
        # }
        # ----------------------------------------------------

        options = _asset_carla_options(
            self.execution,
            actor_state.actor_id,
        )

        attributes = options.get(
            "attributes",
            {},
        )

        if attributes is None:

            attributes = {}

        if not isinstance(
            attributes,
            dict,
        ):

            raise ValueError(
                f"{actor_state.actor_id}: "
                "manifest carla.attributes "
                "must be an object"
            )

        for name, value in (
            attributes.items()
        ):

            if not blueprint.has_attribute(
                str(
                    name
                )
            ):

                raise ValueError(
                    f"{actor_state.actor_id}: "
                    f"CARLA blueprint "
                    f"{actor_state.carla_blueprint!r} "
                    f"has no attribute "
                    f"{name!r}"
                )

            blueprint.set_attribute(
                str(
                    name
                ),
                str(
                    value
                ),
            )

        return blueprint

    # ========================================================
    # Spawn
    # ========================================================

    def _spawn_actor(
        self,
        actor_state: ExecutionActorState,
    ):

        blueprint = (
            self._prepare_blueprint(
                actor_state
            )
        )

        transform = (
            execution_state_to_carla_transform(
                world=
                    self.world,

                actor_state=
                    actor_state,

                execution=
                    self.execution,
            )
        )

        actor = (
            self.world
            .try_spawn_actor(
                blueprint,
                transform,
            )
        )

        if actor is None:

            raise RuntimeError(
                "Could not spawn scenario actor "
                f"{actor_state.actor_id!r} "
                "using blueprint "
                f"{actor_state.carla_blueprint!r} "
                "at "
                f"({transform.location.x:.3f}, "
                f"{transform.location.y:.3f}, "
                f"{transform.location.z:.3f})"
            )

        # ----------------------------------------------------
        # Exact resolved trajectory owns actor pose.
        #
        # No TrafficManager or free physics should alter the
        # scenario truth.
        # ----------------------------------------------------

        try:

            actor.set_simulate_physics(
                False
            )

        except Exception:

            # Some future CARLA actor types may not expose
            # physics toggling. Their transform is still driven
            # explicitly every scenario frame.
            pass

        self._actors[
            actor_state.actor_id
        ] = CarlaRealizedActor(
            actor_id=
                actor_state.actor_id,

            carla_actor=
                actor,

            blueprint_id=
                actor_state.carla_blueprint,

            spawned_frame_idx=
                actor_state.frame_idx,

            last_applied_frame_idx=
                actor_state.frame_idx,
        )

        return actor

    # ========================================================
    # Update
    # ========================================================

    def _update_actor(
        self,
        actor_state: ExecutionActorState,
    ):

        realized = self._actors[
            actor_state.actor_id
        ]

        transform = (
            execution_state_to_carla_transform(
                world=
                    self.world,

                actor_state=
                    actor_state,

                execution=
                    self.execution,
            )
        )

        realized.carla_actor.set_transform(
            transform
        )

        realized.last_applied_frame_idx = (
            actor_state.frame_idx
        )

    # ========================================================
    # Destroy
    # ========================================================

    def _destroy_actor(
        self,
        actor_id: str,
    ):

        realized = (
            self._actors.pop(
                actor_id,
                None,
            )
        )

        if realized is None:

            return

        try:

            realized.carla_actor.destroy()

        except Exception:

            pass

    # ========================================================
    # Frame realization
    # ========================================================

    def apply_frame(
        self,
        frame_idx: int,
    ):
        """
        Reconcile CARLA actors with resolved scenario frame_idx.

        Call BEFORE world.tick() so the sensors produced by that
        tick observe this exact scenario state.
        """

        frame_idx = int(
            frame_idx
        )

        target_states = {
            actor.actor_id:
                actor

            for actor
            in self.execution.active_actors(
                frame_idx
            )
        }

        target_ids = set(
            target_states.keys()
        )

        current_ids = set(
            self._actors.keys()
        )

        # ----------------------------------------------------
        # Despawn actors that no longer exist.
        # ----------------------------------------------------

        for actor_id in sorted(
            current_ids
            -
            target_ids
        ):

            self._destroy_actor(
                actor_id
            )

        # ----------------------------------------------------
        # Spawn/update every active actor.
        # ----------------------------------------------------

        for actor_id in sorted(
            target_ids
        ):

            state = target_states[
                actor_id
            ]

            if actor_id not in self._actors:

                self._spawn_actor(
                    state
                )

            else:

                realized = self._actors[
                    actor_id
                ]

                # A scenario actor ID must never silently switch
                # blueprint during its lifetime.
                if (
                    realized.blueprint_id
                    !=
                    state.carla_blueprint
                ):

                    raise RuntimeError(
                        f"{actor_id}: CARLA blueprint "
                        "changed during actor lifetime: "
                        f"{realized.blueprint_id!r} -> "
                        f"{state.carla_blueprint!r}"
                    )

                self._update_actor(
                    state
                )

        return tuple(
            target_states[
                actor_id
            ]

            for actor_id
            in sorted(
                target_ids
            )
        )

    # ========================================================
    # Cleanup
    # ========================================================

    def destroy_all(
        self,
    ):

        for actor_id in list(
            self._actors.keys()
        ):

            self._destroy_actor(
                actor_id
            )


# ============================================================
# CLI backend smoke test
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "resolved_json",
    )

    parser.add_argument(
        "--asset-root",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--town",
        default="Town10HD_Opt",
    )

    parser.add_argument(
        "--origin-spawn-index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    client = carla.Client(
        args.host,
        args.port,
    )

    client.set_timeout(
        20.0
    )

    world = client.load_world(
        args.town
    )

    original_settings = (
        world.get_settings()
    )

    realizer = None

    try:

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        # Runtime FPS is loaded below; for this smoke test the
        # current scenarios are 20 Hz.
        settings.fixed_delta_seconds = (
            1.0
            /
            20.0
        )

        world.apply_settings(
            settings
        )

        spawn_points = (
            world
            .get_map()
            .get_spawn_points()
        )

        if not spawn_points:

            raise RuntimeError(
                "CARLA map has no spawn points."
            )

        spawn_idx = (
            int(
                args.origin_spawn_index
            )
            %
            len(
                spawn_points
            )
        )

        origin_tf = (
            spawn_points[
                spawn_idx
            ]
        )

        origin = ExecutionWorldOrigin(
            x_m=float(
                origin_tf.location.x
            ),

            y_m=float(
                origin_tf.location.y
            ),

            z_m=float(
                origin_tf.location.z
            ),

            yaw_deg=float(
                origin_tf.rotation.yaw
            ),
        )

        execution = (
            load_execution_runtime(
                resolved_json=
                    args.resolved_json,

                asset_root=
                    args.asset_root,

                manifest_path=
                    args.manifest,

                origin=
                    origin,
            )
        )

        # Use actual scenario FPS.
        settings = (
            world.get_settings()
        )

        settings.fixed_delta_seconds = (
            1.0
            /
            float(
                execution.runtime.fps
            )
        )

        world.apply_settings(
            settings
        )

        realizer = CarlaActorRealizer(
            world=
                world,

            execution=
                execution,
        )

        print()
        print(
            "=" * 78
        )

        print(
            "CARLA ACTOR REALIZER V1"
        )

        print(
            "=" * 78
        )

        print(
            "scenario:",
            execution.runtime.scenario_id,
        )

        print(
            "town:",
            args.town,
        )

        print(
            "origin spawn:",
            spawn_idx,
        )

        print(
            "origin:",
            (
                origin.x_m,
                origin.y_m,
                origin.z_m,
                origin.yaw_deg,
            ),
        )

        print()

        start_frame = int(
            args.start_frame
        )

        end_frame = min(
            execution.runtime.last_frame_idx
            +
            1,

            start_frame
            +
            int(
                args.frames
            ),
        )

        for frame_idx in range(
            start_frame,
            end_frame,
        ):

            states = (
                realizer.apply_frame(
                    frame_idx
                )
            )

            carla_frame = (
                world.tick()
            )

            print(
                f"[{frame_idx:04d}] "
                f"CARLA={carla_frame} "
                f"active={len(states)}"
            )

            for state in states:

                actor = (
                    realizer.get_actor(
                        state.actor_id
                    )
                )

                tf = (
                    actor.get_transform()
                )

                print(
                    "  "
                    f"{state.actor_id} "
                    f"{state.carla_blueprint} "
                    f"x={tf.location.x:.3f} "
                    f"y={tf.location.y:.3f} "
                    f"z={tf.location.z:.3f} "
                    f"yaw={tf.rotation.yaw:.3f}"
                )

        print()
        print(
            "smoke test: PASS"
        )

        print(
            "=" * 78
        )

    finally:

        if realizer is not None:

            realizer.destroy_all()

        try:

            world.apply_settings(
                original_settings
            )

        except Exception:

            pass


if __name__ == "__main__":

    main()