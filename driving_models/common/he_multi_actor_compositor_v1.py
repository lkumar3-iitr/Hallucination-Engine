from __future__ import annotations

import math

from dataclasses import dataclass

from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

import carla
import numpy as np

from he_camera_renderer import (
    SpriteCache,
    load_view_matrix_sprite_bank,
    render_he_actor_view_matrix,
)


# ============================================================
# Small robust field readers
# ============================================================

_MISSING = object()


def _read_attr(
    obj,
    *names,
    default=_MISSING,
):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)

    if default is not _MISSING:
        return default

    tried = ", ".join(repr(x) for x in names)
    raise AttributeError(
        f"Could not find any of attributes: {tried}"
    )


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ActorCameraState:
    actor_id: str
    asset_key: str
    carla_blueprint: str

    world_x: float
    world_y: float
    world_z: float
    world_yaw_deg: float

    rel_x_m: float
    rel_y_m: float
    rel_z_m: float
    rel_yaw_deg: float

    distance_forward_m: float
    distance_euclidean_m: float

    physical_length_m: float
    physical_width_m: float
    physical_height_m: float

    he_view_matrix_csv: str


@dataclass
class ActorRenderResult:
    actor_id: str
    asset_key: str
    carla_blueprint: str

    render_order: int

    rel_x_m: float
    rel_y_m: float
    rel_z_m: float
    rel_yaw_deg: float

    distance_forward_m: float
    distance_euclidean_m: float

    he_view_matrix_csv: str

    rendered: bool
    he_metadata: Dict[str, Any]


@dataclass
class MultiActorCompositeResult:
    rgb: np.ndarray
    actor_results: List[ActorRenderResult]


# ============================================================
# Geometry helpers
# ============================================================

def normalize_angle_180(
    angle_deg,
):
    return (
        float(angle_deg) + 180.0
    ) % 360.0 - 180.0


def yaw_forward_right(
    yaw_deg,
):
    yaw = math.radians(
        float(yaw_deg)
    )

    forward = np.array(
        [
            math.cos(yaw),
            math.sin(yaw),
        ],
        dtype=np.float64,
    )

    right = np.array(
        [
            -math.sin(yaw),
            math.cos(yaw),
        ],
        dtype=np.float64,
    )

    return forward, right


def actor_camera_relative_state(
    actor_world_x,
    actor_world_y,
    actor_world_z,
    actor_world_yaw_deg,
    camera_tf,
):
    cam_x = float(
        camera_tf.location.x
    )
    cam_y = float(
        camera_tf.location.y
    )
    cam_z = float(
        camera_tf.location.z
    )
    cam_yaw = float(
        camera_tf.rotation.yaw
    )

    forward, right = yaw_forward_right(
        cam_yaw
    )

    dx = float(actor_world_x) - cam_x
    dy = float(actor_world_y) - cam_y
    dz = float(actor_world_z) - cam_z

    rel_z_m = float(
        dx * forward[0]
        + dy * forward[1]
    )

    rel_x_m = float(
        dx * right[0]
        + dy * right[1]
    )

    rel_y_m = float(dz)

    rel_yaw_deg = normalize_angle_180(
        float(actor_world_yaw_deg) - cam_yaw
    )

    distance_euclidean_m = math.sqrt(
        dx * dx + dy * dy + dz * dz
    )

    return {
        "rel_x_m": rel_x_m,
        "rel_y_m": rel_y_m,
        "rel_z_m": rel_z_m,
        "rel_yaw_deg": rel_yaw_deg,
        "distance_forward_m": rel_z_m,
        "distance_euclidean_m": float(
            distance_euclidean_m
        ),
    }


def make_actor_transform(
    world_x,
    world_y,
    world_z,
    world_yaw_deg,
):
    return carla.Transform(
        carla.Location(
            x=float(world_x),
            y=float(world_y),
            z=float(world_z),
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=float(world_yaw_deg),
            roll=0.0,
        ),
    )


def physical_dimensions_dict(
    actor,
):
    dims = _read_attr(
        actor,
        "physical_dimensions",
        "dimensions",
    )

    if dims is None:
        raise RuntimeError(
            "Execution actor is missing physical_dimensions."
        )

    return {
        "length_m": float(
            _read_attr(
                dims,
                "length_m",
                "length",
            )
        ),
        "width_m": float(
            _read_attr(
                dims,
                "width_m",
                "width",
            )
        ),
        "height_m": float(
            _read_attr(
                dims,
                "height_m",
                "height",
            )
        ),
    }


def actor_to_camera_state(
    actor,
    camera_tf,
):
    actor_id = str(
        _read_attr(
            actor,
            "actor_id",
            "id",
        )
    )

    asset_key = str(
        _read_attr(
            actor,
            "canonical_asset_key",
            "scenario_asset_key",
            "asset_key",
            "asset",
            default="unknown_asset",
        )
    )

    carla_blueprint = str(
        _read_attr(
            actor,
            "carla_blueprint",
            "blueprint",
            default="unknown_blueprint",
        )
    )

    world_x = float(
        _read_attr(
            actor,
            "world_x_m",
            "world_x",
        )
    )
    world_y = float(
        _read_attr(
            actor,
            "world_y_m",
            "world_y",
        )
    )
    world_z = float(
        _read_attr(
            actor,
            "world_z_m",
            "world_z",
        )
    )
    world_yaw_deg = float(
        _read_attr(
            actor,
            "world_yaw_deg",
            "world_yaw",
            "yaw_deg",
        )
    )

    rel = actor_camera_relative_state(
        actor_world_x=world_x,
        actor_world_y=world_y,
        actor_world_z=world_z,
        actor_world_yaw_deg=world_yaw_deg,
        camera_tf=camera_tf,
    )

    dims = physical_dimensions_dict(
        actor
    )

    he_view_matrix_csv = str(
        _read_attr(
            actor,
            "he_view_matrix_csv",
            "view_matrix_csv",
        )
    )

    return ActorCameraState(
        actor_id=actor_id,
        asset_key=asset_key,
        carla_blueprint=carla_blueprint,

        world_x=world_x,
        world_y=world_y,
        world_z=world_z,
        world_yaw_deg=world_yaw_deg,

        rel_x_m=rel["rel_x_m"],
        rel_y_m=rel["rel_y_m"],
        rel_z_m=rel["rel_z_m"],
        rel_yaw_deg=rel["rel_yaw_deg"],

        distance_forward_m=rel["distance_forward_m"],
        distance_euclidean_m=rel["distance_euclidean_m"],

        physical_length_m=dims["length_m"],
        physical_width_m=dims["width_m"],
        physical_height_m=dims["height_m"],

        he_view_matrix_csv=he_view_matrix_csv,
    )


# ============================================================
# Multi-actor compositor
# ============================================================

class HEMultiActorCompositorV1:
    """
    Multi-actor HE compositor using actor-specific 4320 view-matrix banks.

    V1 responsibilities:
      - use actor-specific HE banks
      - sort actors far -> near
      - sequentially render into one RGB frame

    Not yet included:
      - scene-depth occlusion
      - per-pixel actor depth compositing
    """

    def __init__(
        self,
        distance_selection_mode="linear",
        bottom_y_offset_px=0.0,
        min_forward_distance_m=0.1,
    ):
        self.distance_selection_mode = str(
            distance_selection_mode
        )

        self.bottom_y_offset_px = float(
            bottom_y_offset_px
        )

        self.min_forward_distance_m = float(
            min_forward_distance_m
        )

        self.sprite_cache = SpriteCache()

        self._view_matrix_cache: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self._sprite_bank_cache: Dict[
            str,
            Dict[str, Any],
        ] = {}

    # --------------------------------------------------------
    # Bank loading
    # --------------------------------------------------------

    def _get_sprite_bank_for_csv(
        self,
        view_matrix_csv,
        camera_height_m,
    ):
        key = str(
            Path(view_matrix_csv).resolve()
        )

        if key not in self._sprite_bank_cache:
            self._sprite_bank_cache[key] = {
                "mode": "view_matrix",
                "view_matrix_csvs": [key],

                # same physical target used elsewhere
                "target_height_m": 0.75,

                # actor base height already comes from world/camera geometry
                "vertical_mode": "state_y",

                # informative here because state_y is used
                "camera_height_m": float(
                    camera_height_m
                ),

                "distance_selection_mode":
                    self.distance_selection_mode,
            }

        return self._sprite_bank_cache[key]

    def _get_view_matrix_for_csv(
        self,
        view_matrix_csv,
        camera_height_m,
    ):
        key = str(
            Path(view_matrix_csv).resolve()
        )

        if key not in self._view_matrix_cache:
            sprite_bank = self._get_sprite_bank_for_csv(
                view_matrix_csv=view_matrix_csv,
                camera_height_m=camera_height_m,
            )

            self._view_matrix_cache[key] = (
                load_view_matrix_sprite_bank(
                    sprite_bank
                )
            )

        return self._view_matrix_cache[key]

    # --------------------------------------------------------
    # Candidate preparation
    # --------------------------------------------------------

    def prepare_candidates(
        self,
        active_actors: Iterable[Any],
        camera_tf,
    ):
        candidates = []

        for actor in active_actors:
            state = actor_to_camera_state(
                actor=actor,
                camera_tf=camera_tf,
            )

            if (
                state.distance_forward_m
                <= self.min_forward_distance_m
            ):
                continue

            candidates.append(
                state
            )

        candidates.sort(
            key=lambda row:
                float(
                    row.distance_forward_m
                ),
            reverse=True,
        )

        return candidates

    # --------------------------------------------------------
    # Main render
    # --------------------------------------------------------

    def render(
        self,
        base_rgb,
        camera_tf,
        active_actors,
        width,
        height,
        fov,
    ):
        if base_rgb is None:
            raise ValueError(
                "base_rgb is None"
            )

        rgb = base_rgb.copy()

        actor_results: List[
            ActorRenderResult
        ] = []

        camera_height_m = float(
            camera_tf.location.z
        )

        candidates = self.prepare_candidates(
            active_actors=active_actors,
            camera_tf=camera_tf,
        )

        for render_order, state in enumerate(
            candidates
        ):
            actor_tf = make_actor_transform(
                world_x=state.world_x,
                world_y=state.world_y,
                world_z=state.world_z,
                world_yaw_deg=state.world_yaw_deg,
            )

            dimensions = {
                "length_m":
                    state.physical_length_m,
                "width_m":
                    state.physical_width_m,
                "height_m":
                    state.physical_height_m,
            }

            sprite_bank = self._get_sprite_bank_for_csv(
                view_matrix_csv=state.he_view_matrix_csv,
                camera_height_m=camera_height_m,
            )

            view_matrix = self._get_view_matrix_for_csv(
                view_matrix_csv=state.he_view_matrix_csv,
                camera_height_m=camera_height_m,
            )

            rgb, he_meta = render_he_actor_view_matrix(
                base_rgb=rgb,
                actor_tf=actor_tf,
                camera_tf=camera_tf,
                dimensions=dimensions,
                sprite_bank=sprite_bank,
                view_matrix=view_matrix,
                sprite_cache=self.sprite_cache,
                width=int(width),
                height=int(height),
                fov=float(fov),
                bottom_y_offset_px=float(
                    self.bottom_y_offset_px
                ),
            )

            actor_results.append(
                ActorRenderResult(
                    actor_id=state.actor_id,
                    asset_key=state.asset_key,
                    carla_blueprint=state.carla_blueprint,

                    render_order=int(
                        render_order
                    ),

                    rel_x_m=state.rel_x_m,
                    rel_y_m=state.rel_y_m,
                    rel_z_m=state.rel_z_m,
                    rel_yaw_deg=state.rel_yaw_deg,

                    distance_forward_m=
                        state.distance_forward_m,
                    distance_euclidean_m=
                        state.distance_euclidean_m,

                    he_view_matrix_csv=
                        state.he_view_matrix_csv,

                    rendered=bool(
                        he_meta.get(
                            "rendered",
                            False,
                        )
                    ),
                    he_metadata=dict(
                        he_meta
                    ),
                )
            )

        return MultiActorCompositeResult(
            rgb=rgb,
            actor_results=actor_results,
        )


# ============================================================
# Simple debug printer
# ============================================================

def print_composite_summary(
    result: MultiActorCompositeResult,
):
    print("=" * 78)
    print(
        "HE MULTI-ACTOR COMPOSITOR V1"
    )
    print("=" * 78)

    print(
        "actors rendered:",
        len(
            result.actor_results
        ),
    )

    for row in result.actor_results:
        print()
        print(
            row.actor_id
        )
        print(
            "  order:",
            row.render_order,
        )
        print(
            "  asset:",
            row.asset_key,
        )
        print(
            "  blueprint:",
            row.carla_blueprint,
        )
        print(
            "  rel pos:",
            f"({row.rel_x_m:.3f}, "
            f"{row.rel_y_m:.3f}, "
            f"{row.rel_z_m:.3f})",
        )
        print(
            "  forward depth:",
            f"{row.distance_forward_m:.3f}",
        )
        print(
            "  euclidean depth:",
            f"{row.distance_euclidean_m:.3f}",
        )
        print(
            "  rendered:",
            row.rendered,
        )

        meta = row.he_metadata

        print(
            "  selected angle:",
            meta.get(
                "selected_angle"
            )
        )
        print(
            "  viewpoint angle:",
            meta.get(
                "viewpoint_angle_deg"
            )
        )
        print(
            "  query distance:",
            meta.get(
                "query_distance_m"
            )
        )
        print(
            "  selected distance:",
            meta.get(
                "selected_distance_m"
            )
        )
        print(
            "  query elevation:",
            meta.get(
                "query_elevation_deg"
            )
        )
        print(
            "  selected elevation:",
            meta.get(
                "selected_elevation_deg"
            )
        )
        print(
            "  sprite:",
            meta.get(
                "sprite_path"
            )
        )

    print("=" * 78)