"""
resolved_scenario_runtime_v1.py

Generic runtime index for ScenarioGenerator ResolvedScenarioV2 artifacts.

Purpose
-------
Convert the flattened, backend-independent resolved scenario JSON into
an efficient runtime interface for execution backends.

This module has NO knowledge of:
    - CARLA
    - Hallucination Engine rendering
    - NEAT
    - TCP
    - cameras
    - model checkpoints
    - image coordinates

Typical use
-----------
runtime = ResolvedScenarioRuntime.from_json(path)

ego = runtime.ego_state(frame_idx)

for actor in runtime.active_actors(frame_idx):
    print(actor.info.actor_id)
    print(actor.state.x_m, actor.state.y_m)

Architecture
------------
ScenarioGenerator
      |
      v
ResolvedScenarioV2 JSON
      |
      v
ResolvedScenarioRuntime
      |
      +---- CARLA backend
      |
      +---- HE backend
      |
      +---- future backends
"""

from __future__ import annotations

import argparse
import json

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
# Runtime data types
# ============================================================

@dataclass(
    frozen=True
)
class RuntimeDimensions:
    """
    Physical actor dimensions.

    These are scenario / collision dimensions.
    They are NOT HE rendering dimensions.
    """

    length_m: float
    width_m: float
    height_m: float


@dataclass(
    frozen=True
)
class RuntimeActorInfo:
    """
    Static information describing one scenario actor.
    """

    actor_id: str
    actor_type: str
    role: str

    asset_key: Optional[str]

    dimensions_m: Optional[
        RuntimeDimensions
    ]

    spawn_time_s: float

    despawn_time_s: Optional[
        float
    ]


@dataclass(
    frozen=True
)
class RuntimeActorState:
    """
    Exact model-independent physical actor state
    for one resolved frame.
    """

    frame_idx: int
    t_s: float

    actor_id: str

    x_m: float
    y_m: float
    yaw_deg: float

    speed_mps: float

    vx_mps: float
    vy_mps: float


@dataclass(
    frozen=True
)
class RuntimeEgoState:
    """
    Resolved ego placeholder state.

    In closed-loop experiments the actual ego state normally comes
    from the simulator/model-controlled vehicle. This resolved state
    remains useful for scripted scenarios and validation.
    """

    frame_idx: int
    t_s: float

    x_m: float
    y_m: float
    yaw_deg: float

    speed_mps: float

    vx_mps: float
    vy_mps: float


@dataclass(
    frozen=True
)
class ActiveRuntimeActor:
    """
    Convenience pair containing both static actor information
    and its exact current physical state.
    """

    info: RuntimeActorInfo
    state: RuntimeActorState


# ============================================================
# Parsing helpers
# ============================================================

def _required(
    data: Dict[str, Any],
    key: str,
):

    if key not in data:

        raise ValueError(
            f"Missing required field: {key}"
        )

    return data[
        key
    ]


def _parse_dimensions(
    value,
):

    if value is None:

        return None

    return RuntimeDimensions(
        length_m=float(
            _required(
                value,
                "length_m",
            )
        ),

        width_m=float(
            _required(
                value,
                "width_m",
            )
        ),

        height_m=float(
            _required(
                value,
                "height_m",
            )
        ),
    )


def _parse_actor_info(
    row,
):

    return RuntimeActorInfo(
        actor_id=str(
            _required(
                row,
                "actor_id",
            )
        ),

        actor_type=str(
            _required(
                row,
                "actor_type",
            )
        ),

        role=str(
            _required(
                row,
                "role",
            )
        ),

        asset_key=(
            None
            if row.get(
                "asset_key"
            ) is None
            else str(
                row.get(
                    "asset_key"
                )
            )
        ),

        dimensions_m=
            _parse_dimensions(
                row.get(
                    "dimensions_m"
                )
            ),

        spawn_time_s=float(
            _required(
                row,
                "spawn_time_s",
            )
        ),

        despawn_time_s=(
            None
            if row.get(
                "despawn_time_s"
            ) is None
            else float(
                row[
                    "despawn_time_s"
                ]
            )
        ),
    )


def _parse_actor_state(
    row,
):

    return RuntimeActorState(
        frame_idx=int(
            _required(
                row,
                "frame_idx",
            )
        ),

        t_s=float(
            _required(
                row,
                "t_s",
            )
        ),

        actor_id=str(
            _required(
                row,
                "actor_id",
            )
        ),

        x_m=float(
            _required(
                row,
                "x_m",
            )
        ),

        y_m=float(
            _required(
                row,
                "y_m",
            )
        ),

        yaw_deg=float(
            _required(
                row,
                "yaw_deg",
            )
        ),

        speed_mps=float(
            _required(
                row,
                "speed_mps",
            )
        ),

        vx_mps=float(
            _required(
                row,
                "vx_mps",
            )
        ),

        vy_mps=float(
            _required(
                row,
                "vy_mps",
            )
        ),
    )


def _parse_ego_state(
    row,
):

    return RuntimeEgoState(
        frame_idx=int(
            _required(
                row,
                "frame_idx",
            )
        ),

        t_s=float(
            _required(
                row,
                "t_s",
            )
        ),

        x_m=float(
            _required(
                row,
                "x_m",
            )
        ),

        y_m=float(
            _required(
                row,
                "y_m",
            )
        ),

        yaw_deg=float(
            _required(
                row,
                "yaw_deg",
            )
        ),

        speed_mps=float(
            _required(
                row,
                "speed_mps",
            )
        ),

        vx_mps=float(
            _required(
                row,
                "vx_mps",
            )
        ),

        vy_mps=float(
            _required(
                row,
                "vy_mps",
            )
        ),
    )


# ============================================================
# Runtime
# ============================================================

class ResolvedScenarioRuntime:
    """
    Indexed model-independent runtime representation.

    Actor states are indexed as:

        actor_id -> frame_idx -> RuntimeActorState

    This supports any number of actors and arbitrary lifecycle
    spawn/despawn behavior without backend-specific logic.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        source_path: Optional[Path] = None,
    ):

        self.source_path = (
            Path(
                source_path
            ).resolve()
            if source_path is not None
            else None
        )

        self.raw = data

        # ----------------------------------------------------
        # Scenario metadata
        # ----------------------------------------------------

        self.schema_version = str(
            _required(
                data,
                "schema_version",
            )
        )

        if (
            self.schema_version
            !=
            "2.0-resolved"
        ):

            raise ValueError(
                "Unsupported resolved scenario schema: "
                f"{self.schema_version!r}. "
                "Expected '2.0-resolved'."
            )

        self.source_schema_version = str(
            data.get(
                "source_schema_version",
                "",
            )
        )

        self.scenario_id = str(
            _required(
                data,
                "scenario_id",
            )
        )

        self.description = data.get(
            "source_description"
        )

        self.duration_s = float(
            _required(
                data,
                "duration_s",
            )
        )

        self.fps = int(
            _required(
                data,
                "fps",
            )
        )

        self.seed = int(
            data.get(
                "seed",
                0,
            )
        )

        self.coordinate_frame = str(
            data.get(
                "coordinate_frame",
                "ego_initial",
            )
        )

        self.camera_metadata = data.get(
            "camera"
        )

        # ----------------------------------------------------
        # Static actor information
        # ----------------------------------------------------

        self._actors_by_id: Dict[
            str,
            RuntimeActorInfo,
        ] = {}

        for row in data.get(
            "actors",
            [],
        ):

            info = _parse_actor_info(
                row
            )

            if (
                info.actor_id
                in
                self._actors_by_id
            ):

                raise ValueError(
                    "Duplicate actor information for "
                    f"{info.actor_id!r}"
                )

            self._actors_by_id[
                info.actor_id
            ] = info

        # ----------------------------------------------------
        # Ego frames
        # ----------------------------------------------------

        self._ego_by_frame: Dict[
            int,
            RuntimeEgoState,
        ] = {}

        for row in data.get(
            "ego_frames",
            [],
        ):

            state = _parse_ego_state(
                row
            )

            if (
                state.frame_idx
                in
                self._ego_by_frame
            ):

                raise ValueError(
                    "Duplicate ego frame_idx: "
                    f"{state.frame_idx}"
                )

            self._ego_by_frame[
                state.frame_idx
            ] = state

        # ----------------------------------------------------
        # Actor frames
        # ----------------------------------------------------

        self._actor_frames: Dict[
            str,
            Dict[
                int,
                RuntimeActorState,
            ],
        ] = {
            actor_id: {}
            for actor_id
            in self._actors_by_id
        }

        for row in data.get(
            "actor_frames",
            [],
        ):

            state = _parse_actor_state(
                row
            )

            if (
                state.actor_id
                not in
                self._actors_by_id
            ):

                raise ValueError(
                    "Actor frame references unknown actor_id "
                    f"{state.actor_id!r}"
                )

            actor_table = self._actor_frames[
                state.actor_id
            ]

            if (
                state.frame_idx
                in
                actor_table
            ):

                raise ValueError(
                    "Duplicate actor state for "
                    f"actor={state.actor_id!r}, "
                    f"frame={state.frame_idx}"
                )

            actor_table[
                state.frame_idx
            ] = state

        # ----------------------------------------------------
        # Frame range
        # ----------------------------------------------------

        all_indices = set(
            self._ego_by_frame.keys()
        )

        for actor_table in (
            self._actor_frames.values()
        ):

            all_indices.update(
                actor_table.keys()
            )

        if all_indices:

            self.first_frame_idx = min(
                all_indices
            )

            self.last_frame_idx = max(
                all_indices
            )

        else:

            self.first_frame_idx = 0
            self.last_frame_idx = -1

        self.frame_count = (
            self.last_frame_idx
            -
            self.first_frame_idx
            +
            1
            if self.last_frame_idx >= 0
            else 0
        )

    # ========================================================
    # Construction
    # ========================================================

    @classmethod
    def from_json(
        cls,
        path,
    ) -> "ResolvedScenarioRuntime":

        path = Path(
            path
        ).resolve()

        with path.open(
            "r",
            encoding="utf-8",
        ) as fp:

            data = json.load(
                fp
            )

        return cls(
            data=data,
            source_path=path,
        )

    # ========================================================
    # Scenario queries
    # ========================================================

    @property
    def actor_ids(
        self,
    ) -> Tuple[str, ...]:

        return tuple(
            self._actors_by_id.keys()
        )

    @property
    def actors_by_id(
        self,
    ):

        return dict(
            self._actors_by_id
        )

    def actor_count(
        self,
    ) -> int:

        return len(
            self._actors_by_id
        )

    def actor_info(
        self,
        actor_id: str,
    ) -> RuntimeActorInfo:

        try:

            return self._actors_by_id[
                actor_id
            ]

        except KeyError as exc:

            raise KeyError(
                f"Unknown actor_id: {actor_id!r}"
            ) from exc

    # ========================================================
    # Frame queries
    # ========================================================

    def ego_state(
        self,
        frame_idx: int,
    ) -> Optional[RuntimeEgoState]:

        return self._ego_by_frame.get(
            int(
                frame_idx
            )
        )

    def actor_state(
        self,
        actor_id: str,
        frame_idx: int,
    ) -> Optional[RuntimeActorState]:

        if (
            actor_id
            not in
            self._actor_frames
        ):

            raise KeyError(
                f"Unknown actor_id: {actor_id!r}"
            )

        return self._actor_frames[
            actor_id
        ].get(
            int(
                frame_idx
            )
        )

    def is_actor_active(
        self,
        actor_id: str,
        frame_idx: int,
    ) -> bool:

        return (
            self.actor_state(
                actor_id,
                frame_idx,
            )
            is not None
        )

    def active_actors(
        self,
        frame_idx: int,
    ) -> Tuple[
        ActiveRuntimeActor,
        ...
    ]:
        """
        Return every actor that physically exists at frame_idx.

        Lifecycle is represented directly by presence/absence of
        resolved actor frames; no extra backend lifecycle logic
        is needed.
        """

        frame_idx = int(
            frame_idx
        )

        active = []

        for actor_id, info in (
            self._actors_by_id.items()
        ):

            state = self._actor_frames[
                actor_id
            ].get(
                frame_idx
            )

            if state is None:

                continue

            active.append(
                ActiveRuntimeActor(
                    info=info,
                    state=state,
                )
            )

        return tuple(
            active
        )

    def active_actor_ids(
        self,
        frame_idx: int,
    ) -> Tuple[str, ...]:

        return tuple(
            actor.info.actor_id
            for actor in self.active_actors(
                frame_idx
            )
        )

    def actor_frame_indices(
        self,
        actor_id: str,
    ) -> Tuple[int, ...]:

        if (
            actor_id
            not in
            self._actor_frames
        ):

            raise KeyError(
                f"Unknown actor_id: {actor_id!r}"
            )

        return tuple(
            sorted(
                self._actor_frames[
                    actor_id
                ].keys()
            )
        )

    def iter_frames(
        self,
    ) -> Iterable[int]:

        if self.last_frame_idx < 0:

            return

        for frame_idx in range(
            self.first_frame_idx,
            self.last_frame_idx + 1,
        ):

            yield frame_idx

    def time_for_frame(
        self,
        frame_idx: int,
    ) -> float:
        """
        Prefer exact resolved time when available.
        Fall back to frame_idx / fps.
        """

        ego = self.ego_state(
            frame_idx
        )

        if ego is not None:

            return float(
                ego.t_s
            )

        for actor in self.active_actors(
            frame_idx
        ):

            return float(
                actor.state.t_s
            )

        return (
            float(
                frame_idx
            )
            /
            float(
                self.fps
            )
        )

    # ========================================================
    # Validation summary
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, Any]:

        actor_summaries = []

        for actor_id in self.actor_ids:

            info = self.actor_info(
                actor_id
            )

            frame_indices = (
                self.actor_frame_indices(
                    actor_id
                )
            )

            actor_summaries.append({
                "actor_id":
                    actor_id,

                "actor_type":
                    info.actor_type,

                "role":
                    info.role,

                "asset_key":
                    info.asset_key,

                "spawn_time_s":
                    info.spawn_time_s,

                "despawn_time_s":
                    info.despawn_time_s,

                "resolved_frames":
                    len(
                        frame_indices
                    ),

                "first_frame":
                    (
                        frame_indices[0]
                        if frame_indices
                        else None
                    ),

                "last_frame":
                    (
                        frame_indices[-1]
                        if frame_indices
                        else None
                    ),
            })

        return {
            "scenario_id":
                self.scenario_id,

            "schema_version":
                self.schema_version,

            "fps":
                self.fps,

            "duration_s":
                self.duration_s,

            "frame_count":
                self.frame_count,

            "first_frame_idx":
                self.first_frame_idx,

            "last_frame_idx":
                self.last_frame_idx,

            "actor_count":
                self.actor_count(),

            "actors":
                actor_summaries,
        }


# ============================================================
# CLI smoke test
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "resolved_json",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    runtime = (
        ResolvedScenarioRuntime
        .from_json(
            args.resolved_json
        )
    )

    summary = runtime.summary()

    print()
    print(
        "=" * 78
    )

    print(
        "RESOLVED SCENARIO RUNTIME V1"
    )

    print(
        "=" * 78
    )

    print(
        "scenario:",
        summary[
            "scenario_id"
        ],
    )

    print(
        "schema:",
        summary[
            "schema_version"
        ],
    )

    print(
        "fps:",
        summary[
            "fps"
        ],
    )

    print(
        "duration:",
        f"{summary['duration_s']:.3f}s",
    )

    print(
        "frames:",
        summary[
            "frame_count"
        ],
    )

    print(
        "actors:",
        summary[
            "actor_count"
        ],
    )

    print()

    for actor in summary[
        "actors"
    ]:

        print(
            f"{actor['actor_id']}: "
            f"type={actor['actor_type']} "
            f"role={actor['role']} "
            f"asset={actor['asset_key']} "
            f"frames={actor['resolved_frames']} "
            f"first={actor['first_frame']} "
            f"last={actor['last_frame']}"
        )

    print()

    frame_idx = int(
        args.frame
    )

    print(
        f"active actors at frame {frame_idx}:"
    )

    active = runtime.active_actors(
        frame_idx
    )

    if not active:

        print(
            "  none"
        )

    for actor in active:

        state = actor.state

        print(
            f"  {actor.info.actor_id}: "
            f"x={state.x_m:.3f} "
            f"y={state.y_m:.3f} "
            f"yaw={state.yaw_deg:.3f} "
            f"speed={state.speed_mps:.3f}"
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()