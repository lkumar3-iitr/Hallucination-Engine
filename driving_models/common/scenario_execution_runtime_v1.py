"""
scenario_execution_runtime_v1.py

Generic execution-coordinate runtime.

Purpose
-------
Join:

    ResolvedScenarioRuntime
        +
    ScenarioAssetBinder
        +
    one execution-world origin

and produce backend-ready NUMERIC actor states.

This module deliberately has NO dependency on:
    - CARLA
    - NEAT
    - TCP
    - PyTorch
    - HE renderer
    - camera configuration

It does NOT spawn actors.

It only performs the single validated conversion:

ScenarioGenerator frame:
    +x = ego-initial forward
    +y = ego-initial left
    +yaw = counter-clockwise / left

Execution world:
    position =
        origin
        + forward * SG_x
        - right * SG_y

    world_yaw =
        origin_yaw - SG_yaw

This mapping is already used by the validated
NEAT/TCP ScenarioGenerator execution.

Architecture
------------
ResolvedScenarioRuntime
          |
          v
ScenarioAssetBinder
          |
          v
ScenarioExecutionRuntime
          |
          +------ CARLA backend
          |
          +------ HE backend
          |
          +------ future backend
"""

from __future__ import annotations

import argparse
import math

from dataclasses import dataclass

from pathlib import Path

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)


from resolved_scenario_runtime_v1 import (
    RuntimeDimensions,
)

from he_asset_registry_v1 import (
    DEFAULT_MANIFEST,
)

from scenario_asset_binding_v1 import (
    BoundScenarioActor,
    load_bound_scenario,
)


# ============================================================
# World origin
# ============================================================

@dataclass(
    frozen=True
)
class ExecutionWorldOrigin:
    """
    Origin of ScenarioGenerator coordinates in the execution world.

    In CARLA experiments this is normally the canonicalized initial
    ego pose.

    This remains numeric so this module does not depend on CARLA.
    """

    x_m: float
    y_m: float
    z_m: float

    yaw_deg: float


# ============================================================
# Execution actor state
# ============================================================

@dataclass(
    frozen=True
)
class ExecutionActorState:
    """
    Fully bound physical actor state in execution-world coordinates.

    physical_dimensions:
        Scenario truth dimensions.

    carla_blueprint:
        CARLA realization metadata.

    he_view_matrix_csv:
        HE realization metadata.

    No renderer/model assumptions are made here.
    """

    frame_idx: int
    t_s: float

    actor_id: str
    actor_type: str
    role: str

    scenario_asset_key: Optional[
        str
    ]

    canonical_asset_key: str

    carla_blueprint: str

    he_view_matrix_csv: Path

    physical_dimensions: Optional[
        RuntimeDimensions
    ]

    physical_dimensions_source: Optional[
        str
    ]

    # --------------------------------------------------------
    # ScenarioGenerator coordinates
    # --------------------------------------------------------

    sg_x_m: float
    sg_y_m: float
    sg_yaw_deg: float

    sg_vx_mps: float
    sg_vy_mps: float

    # --------------------------------------------------------
    # Execution world coordinates
    # --------------------------------------------------------

    world_x_m: float
    world_y_m: float
    world_z_m: float

    world_yaw_deg: float

    world_vx_mps: float
    world_vy_mps: float

    speed_mps: float


# ============================================================
# Coordinate helpers
# ============================================================

def yaw_forward_right(
    yaw_deg: float,
):
    """
    Return world XY unit vectors corresponding to:

        forward
        right

    for the supplied execution-world yaw.

    This is numerically identical to the validated runner helper.
    """

    yaw_rad = math.radians(
        float(
            yaw_deg
        )
    )

    forward_x = math.cos(
        yaw_rad
    )

    forward_y = math.sin(
        yaw_rad
    )

    right_x = -math.sin(
        yaw_rad
    )

    right_y = math.cos(
        yaw_rad
    )

    return (
        (
            forward_x,
            forward_y,
        ),
        (
            right_x,
            right_y,
        ),
    )


def sg_position_to_world(
    origin: ExecutionWorldOrigin,
    sg_x_m: float,
    sg_y_m: float,
):
    """
    Convert ScenarioGenerator XY into execution-world XY.

    SG:
        +x = forward
        +y = left

    therefore:

        world =
            origin
            + forward * sg_x
            - right * sg_y
    """

    (
        forward,
        right,
    ) = yaw_forward_right(
        origin.yaw_deg
    )

    world_x = (
        float(
            origin.x_m
        )
        +
        forward[0]
        *
        float(
            sg_x_m
        )
        -
        right[0]
        *
        float(
            sg_y_m
        )
    )

    world_y = (
        float(
            origin.y_m
        )
        +
        forward[1]
        *
        float(
            sg_x_m
        )
        -
        right[1]
        *
        float(
            sg_y_m
        )
    )

    return (
        float(
            world_x
        ),
        float(
            world_y
        ),
    )


def sg_velocity_to_world(
    origin: ExecutionWorldOrigin,
    sg_vx_mps: float,
    sg_vy_mps: float,
):
    """
    Rotate ScenarioGenerator velocity into execution-world XY.

    Same coordinate convention as position:

        SG +x = forward
        SG +y = left

    so:

        v_world =
            forward * SG_vx
            - right * SG_vy
    """

    (
        forward,
        right,
    ) = yaw_forward_right(
        origin.yaw_deg
    )

    world_vx = (
        forward[0]
        *
        float(
            sg_vx_mps
        )
        -
        right[0]
        *
        float(
            sg_vy_mps
        )
    )

    world_vy = (
        forward[1]
        *
        float(
            sg_vx_mps
        )
        -
        right[1]
        *
        float(
            sg_vy_mps
        )
    )

    return (
        float(
            world_vx
        ),
        float(
            world_vy
        ),
    )


def sg_yaw_to_world(
    origin: ExecutionWorldOrigin,
    sg_yaw_deg: float,
) -> float:
    """
    Validated ScenarioGenerator -> CARLA/world yaw mapping.

        world yaw =
            initial ego yaw
            - ScenarioGenerator yaw
    """

    return float(
        origin.yaw_deg
        -
        float(
            sg_yaw_deg
        )
    )


# ============================================================
# Bound actor conversion
# ============================================================

def bound_actor_to_execution_state(
    actor: BoundScenarioActor,
    origin: ExecutionWorldOrigin,
) -> ExecutionActorState:

    state = actor.state

    (
        world_x,
        world_y,
    ) = sg_position_to_world(
        origin=origin,

        sg_x_m=
            state.x_m,

        sg_y_m=
            state.y_m,
    )

    (
        world_vx,
        world_vy,
    ) = sg_velocity_to_world(
        origin=origin,

        sg_vx_mps=
            state.vx_mps,

        sg_vy_mps=
            state.vy_mps,
    )

    world_yaw = sg_yaw_to_world(
        origin=origin,

        sg_yaw_deg=
            state.yaw_deg,
    )

    return ExecutionActorState(
        frame_idx=
            state.frame_idx,

        t_s=
            state.t_s,

        actor_id=
            actor.actor_id,

        actor_type=
            actor.scenario_info.actor_type,

        role=
            actor.scenario_info.role,

        scenario_asset_key=
            actor.scenario_info.asset_key,

        canonical_asset_key=
            actor.asset.key,

        carla_blueprint=
            actor.asset.carla_blueprint,

        he_view_matrix_csv=
            actor.asset.view_matrix_csv,

        physical_dimensions=
            actor.resolved_physical_dimensions,

        physical_dimensions_source=
            actor.physical_dimensions_source,

        sg_x_m=
            state.x_m,

        sg_y_m=
            state.y_m,

        sg_yaw_deg=
            state.yaw_deg,

        sg_vx_mps=
            state.vx_mps,

        sg_vy_mps=
            state.vy_mps,

        world_x_m=
            world_x,

        world_y_m=
            world_y,

        # ----------------------------------------------------
        # IMPORTANT
        #
        # This is only the reference/world-origin Z.
        #
        # CARLA-specific road snapping belongs in the CARLA
        # realization backend, NOT here.
        # ----------------------------------------------------

        world_z_m=
            float(
                origin.z_m
            ),

        world_yaw_deg=
            world_yaw,

        world_vx_mps=
            world_vx,

        world_vy_mps=
            world_vy,

        speed_mps=
            state.speed_mps,
    )


# ============================================================
# Runtime
# ============================================================

class ScenarioExecutionRuntime:
    """
    Backend-independent per-frame execution state provider.
    """

    def __init__(
        self,
        binder,
        origin: ExecutionWorldOrigin,
    ):

        self.binder = binder

        self.runtime = (
            binder.runtime
        )

        self.registry = (
            binder.registry
        )

        self.origin = origin

        binder.require_valid()

    # ========================================================
    # Per-frame actors
    # ========================================================

    def active_actors(
        self,
        frame_idx: int,
    ) -> Tuple[
        ExecutionActorState,
        ...
    ]:

        return tuple(
            bound_actor_to_execution_state(
                actor=
                    actor,

                origin=
                    self.origin,
            )

            for actor
            in self.binder.active_actors(
                frame_idx
            )
        )

    def actor_state(
        self,
        actor_id: str,
        frame_idx: int,
    ) -> Optional[
        ExecutionActorState
    ]:

        for actor in self.active_actors(
            frame_idx
        ):

            if (
                actor.actor_id
                ==
                actor_id
            ):

                return actor

        return None

    # ========================================================
    # Summary
    # ========================================================

    def summary(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        return {
            "scenario_id":
                self.runtime.scenario_id,

            "actor_count":
                self.runtime.actor_count(),

            "fps":
                self.runtime.fps,

            "duration_s":
                self.runtime.duration_s,

            "origin": {
                "x_m":
                    self.origin.x_m,

                "y_m":
                    self.origin.y_m,

                "z_m":
                    self.origin.z_m,

                "yaw_deg":
                    self.origin.yaw_deg,
            },
        }


# ============================================================
# Convenience constructor
# ============================================================

def load_execution_runtime(
    resolved_json,
    asset_root,
    origin: ExecutionWorldOrigin,
    manifest_path=DEFAULT_MANIFEST,
):

    (
        _runtime,
        _registry,
        binder,
    ) = load_bound_scenario(
        resolved_json=
            resolved_json,

        asset_root=
            asset_root,

        manifest_path=
            manifest_path,
    )

    return ScenarioExecutionRuntime(
        binder=
            binder,

        origin=
            origin,
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
        required=True,
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

    parser.add_argument(
        "--origin-x",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--origin-y",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--origin-z",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--origin-yaw",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    origin = ExecutionWorldOrigin(
        x_m=
            args.origin_x,

        y_m=
            args.origin_y,

        z_m=
            args.origin_z,

        yaw_deg=
            args.origin_yaw,
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

    print()
    print(
        "=" * 78
    )

    print(
        "SCENARIO EXECUTION RUNTIME V1"
    )

    print(
        "=" * 78
    )

    print(
        "scenario:",
        execution.runtime.scenario_id,
    )

    print(
        "actors:",
        execution.runtime.actor_count(),
    )

    print(
        "origin:",
        (
            execution.origin.x_m,
            execution.origin.y_m,
            execution.origin.z_m,
            execution.origin.yaw_deg,
        ),
    )

    print()

    frame_idx = int(
        args.frame
    )

    print(
        f"ACTIVE EXECUTION ACTORS AT FRAME "
        f"{frame_idx}"
    )

    actors = execution.active_actors(
        frame_idx
    )

    if not actors:

        print(
            "  none"
        )

    for actor in actors:

        print(
            f"  {actor.actor_id}"
        )

        print(
            f"    asset="
            f"{actor.canonical_asset_key}"
        )

        print(
            f"    blueprint="
            f"{actor.carla_blueprint}"
        )

        print(
            f"    SG pose="
            f"("
            f"{actor.sg_x_m:.3f}, "
            f"{actor.sg_y_m:.3f}, "
            f"{actor.sg_yaw_deg:.3f}"
            f")"
        )

        print(
            f"    world pose="
            f"("
            f"{actor.world_x_m:.3f}, "
            f"{actor.world_y_m:.3f}, "
            f"{actor.world_z_m:.3f}, "
            f"{actor.world_yaw_deg:.3f}"
            f")"
        )

        print(
            f"    world velocity="
            f"("
            f"{actor.world_vx_mps:.3f}, "
            f"{actor.world_vy_mps:.3f}"
            f")"
        )

        print(
            f"    speed="
            f"{actor.speed_mps:.3f}"
        )

        if actor.physical_dimensions is not None:

            print(
                f"    physical dims="
                f"L={actor.physical_dimensions.length_m:.3f} "
                f"W={actor.physical_dimensions.width_m:.3f} "
                f"H={actor.physical_dimensions.height_m:.3f} "
                f"[source={actor.physical_dimensions_source}]"
            )

        print(
            f"    HE matrix="
            f"{actor.he_view_matrix_csv}"
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()