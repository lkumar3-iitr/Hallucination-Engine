from __future__ import annotations

import math
import csv

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
    alpha_composite_rgb,
    apply_scene_depth_occlusion_to_sprite,
    load_view_matrix_sprite_bank,
    render_he_actor_view_matrix,
    warp_view_matrix_sprite_to_box_subpixel,
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

    he_close_view_matrix_csvs: Dict[str, Any]


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

    he_close_view_matrix_csvs: Dict[str, Any]

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
    close_csvs = _read_attr(
        actor,
        "he_close_view_matrix_csvs",
        default={},
    )
    if close_csvs is None:
        close_csvs = {}
    close_csvs = {
        str(side): (
            [str(path) for path in paths]
            if isinstance(paths, (list, tuple))
            else str(paths)
        )
        for side, paths in dict(close_csvs).items()
    }

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

        he_close_view_matrix_csvs=close_csvs,
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
      - apply CARLA scene-depth occlusion to every actor sprite
    """

    fallback_far_geometry_mode = "proxy"
    centered_close_geometry_mode = "proxy"
    centered_close_projection_mode = "center_depth_billboard"
    center_depth_alpha_bbox_anchor = False
    close_inner_margin_cells = 0.5
    close_outer_margin_cells = 0.5
    close_inner_min_bbox_clearance_m = None
    close_max_yaw_error_deg = None

    def __init__(
        self,
        distance_selection_mode="linear",
        bottom_y_offset_px=0.0,
        min_forward_distance_m=0.1,
        silhouette_scale=1.0,
        warp_scale_mode="independent",
        viewpoint_lateral_sign=1.0,
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
        self.silhouette_scale = float(silhouette_scale)
        self.warp_scale_mode = str(warp_scale_mode)
        self.viewpoint_lateral_sign = float(viewpoint_lateral_sign)

        self.sprite_cache = SpriteCache()

        self._view_matrix_cache: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self._sprite_bank_cache: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self._close_bank_cache: Dict[
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

                "warp_scale_mode":
                    self.warp_scale_mode,

                "viewpoint_lateral_sign":
                    self.viewpoint_lateral_sign,
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

    def _get_close_bank_for_csv(
        self,
        view_matrix_csv,
    ):
        key = str(
            Path(view_matrix_csv).resolve()
        )
        if key not in self._close_bank_cache:
            path = Path(key)
            with path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError(
                    f"Empty close Cartesian bank: {path}"
                )
            def close_key(row):
                return (
                    round(float(row["close_forward_m"]), 6),
                    round(float(row["close_right_m"]), 6),
                    round(float(row["close_relative_yaw_deg"]) % 360.0, 6),
                )
            self._close_bank_cache[key] = {
                "csv": path,
                "root": path.parent,
                "rows": rows,
                "forward_values": sorted({
                    float(row["close_forward_m"])
                    for row in rows
                }),
                "right_values": sorted({
                    float(row["close_right_m"])
                    for row in rows
                }),
                "up_values": sorted({
                    float(row["close_target_up_m"])
                    for row in rows
                }),
                "yaw_values": sorted({
                    float(row["close_relative_yaw_deg"]) % 360.0
                    for row in rows
                }),
                "index": {
                    close_key(row): row
                    for row in rows
                },
            }
        return self._close_bank_cache[key]

    def _get_close_banks_for_side(self, configured, query_up_m):
        paths = configured if isinstance(configured, (list, tuple)) else [configured]
        banks = [self._get_close_bank_for_csv(path) for path in paths]
        return sorted(
            banks,
            key=lambda bank: min(
                abs(float(up_m) - float(query_up_m))
                for up_m in bank["up_values"]
            ),
        )

    @staticmethod
    def _close_row_path(
        bank,
        row,
    ) -> Path:
        rel = row.get(
            "rgba_relpath",
            "",
        )
        if rel:
            return (bank["root"] / rel).resolve()
        return (bank["root"] / "rgba" / Path(row["rgba_path"]).name).resolve()

    @staticmethod
    def _angle_error_deg(
        first,
        second,
    ) -> float:
        return abs(
            (
                float(first)
                -
                float(second)
                +
                180.0
            )
            %
            360.0
            -
            180.0
        )

    def _close_yaw_is_supported(self, query_yaw, selected_yaw) -> bool:
        if self.close_max_yaw_error_deg is None:
            return True
        return self._angle_error_deg(query_yaw, selected_yaw) <= float(
            self.close_max_yaw_error_deg
        )

    @staticmethod
    def _bracket(values, query):
        ordered = sorted(float(value) for value in values)
        if query <= ordered[0]:
            return ordered[0], ordered[0], 0.0
        if query >= ordered[-1]:
            return ordered[-1], ordered[-1], 0.0
        for lower, upper in zip(ordered, ordered[1:]):
            if lower <= query <= upper:
                weight = (float(query) - lower) / (upper - lower)
                return lower, upper, weight
        raise RuntimeError("Could not bracket Cartesian close-bank query")

    def _select_close_bank(
        self,
        state: ActorCameraState,
        actor_tf,
        camera_tf,
        view_matrix,
    ):
        if not state.he_close_view_matrix_csvs:
            return None

        physical_bbox = view_matrix.get("physical_bbox") or {}
        local_bbox_center = np.asarray([
            float(physical_bbox.get("local_center_x_m", 0.0)),
            float(physical_bbox.get("local_center_y_m", 0.0)),
            float(physical_bbox.get("local_center_z_m", 0.0)),
            1.0,
        ], dtype=np.float64)
        actor_to_world = np.asarray(actor_tf.get_matrix(), dtype=np.float64)
        world_to_camera = np.asarray(
            camera_tf.get_inverse_matrix(),
            dtype=np.float64,
        )
        camera_center = world_to_camera @ actor_to_world @ local_bbox_center
        forward_m = float(camera_center[0])
        right_m = float(camera_center[1])
        up_m = float(camera_center[2])

        # Close banks are generated and indexed from the physical bounding-box
        # center, which can lie on the opposite side of the camera centerline
        # from the actor origin during an oblique cut-in.  Use that same point
        # for side selection as well as for the Cartesian lookup.
        side = "right" if right_m > 0.0 else "left"
        configured_banks = state.he_close_view_matrix_csvs.get(side)
        if not configured_banks:
            return None

        bank = self._get_close_banks_for_side(configured_banks, up_m)[0]

        forward_min = min(bank["forward_values"])
        forward_max = max(bank["forward_values"])
        right_min = min(bank["right_values"])
        right_max = max(bank["right_values"])
        up_min = min(bank["up_values"])
        up_max = max(bank["up_values"])

        right_steps = [
            upper - lower
            for lower, upper in zip(
                bank["right_values"],
                bank["right_values"][1:],
            )
        ]
        right_step = (
            min(right_steps)
            if right_steps
            else 0.0
        )
        inner_margin = self.close_inner_margin_cells * right_step
        outer_margin = self.close_outer_margin_cells * right_step
        if side == "right":
            right_query_min = right_min - inner_margin
            right_query_max = right_max + outer_margin
        else:
            right_query_min = right_min - outer_margin
            right_query_max = right_max + inner_margin
        if self.close_inner_min_bbox_clearance_m is not None:
            minimum_abs_right = (
                abs(float(physical_bbox.get("extent_y_m", 0.0)))
                + float(self.close_inner_min_bbox_clearance_m)
            )
            if side == "right":
                right_query_min = max(right_query_min, minimum_abs_right)
            else:
                right_query_max = min(right_query_max, -minimum_abs_right)

        if not (
            forward_min <= forward_m <= forward_max
            and
            right_query_min <= right_m <= right_query_max
        ):
            return None

        bank_up_m = min(
            bank["up_values"],
            key=lambda value: abs(float(value) - up_m),
        )
        query_elevation_deg = math.degrees(math.atan2(up_m, forward_m))
        bank_elevation_deg = math.degrees(
            math.atan2(float(bank_up_m), forward_m)
        )
        if abs(query_elevation_deg - bank_elevation_deg) > 10.0:
            return None

        selected_yaw = min(
            bank["yaw_values"],
            key=lambda value: self._angle_error_deg(value, state.rel_yaw_deg),
        )
        if not self._close_yaw_is_supported(state.rel_yaw_deg, selected_yaw):
            return None
        forward_lower, forward_upper, forward_weight = self._bracket(
            bank["forward_values"], forward_m
        )
        right_lower, right_upper, right_weight = self._bracket(
            bank["right_values"], right_m
        )
        interpolation_rows = []
        interpolation_keys = {
            (
                round(selected_forward, 6),
                round(selected_right, 6),
                round(selected_yaw % 360.0, 6),
            )
            for selected_forward in {forward_lower, forward_upper}
            for selected_right in {right_lower, right_upper}
        }
        for key in interpolation_keys:
            candidate = bank["index"].get(key)
            if candidate is not None:
                interpolation_rows.append(candidate)
        # Some edge poses are intentionally absent because the actor is only
        # partially projectable in the capture viewport. Renormalize the
        # available interpolation weights below instead of discarding a useful
        # close view or extrapolating beyond the sampled half-cell.
        if not interpolation_rows:
            return None
        row = min(
            interpolation_rows,
            key=lambda candidate: (
                (float(candidate["close_forward_m"]) - forward_m) ** 2
                + (float(candidate["close_right_m"]) - right_m) ** 2
            ),
        )

        return {
            "bank": bank,
            "row": row,
            "side": side,
            "eligibility": {
                "forward_min_m": float(forward_min),
                "forward_max_m": float(forward_max),
                "right_min_m": float(right_query_min),
                "right_max_m": float(right_query_max),
            },
            "interpolation": {
                "rows": interpolation_rows,
                "forward_lower": forward_lower,
                "forward_upper": forward_upper,
                "forward_weight": forward_weight,
                "right_lower": right_lower,
                "right_upper": right_upper,
                "right_weight": right_weight,
                "selected_yaw": selected_yaw,
            },
            "query": {
                "forward_m": forward_m,
                "right_m": right_m,
                "up_m": up_m,
                "relative_yaw_deg": float(state.rel_yaw_deg),
            },
        }

    def _close_selection_diagnostic(
        self,
        state: ActorCameraState,
        actor_tf,
        camera_tf,
        view_matrix,
    ):
        if not state.he_close_view_matrix_csvs:
            return {
                "active": False,
                "reason": "no_close_banks_for_actor",
            }

        physical_bbox = view_matrix.get("physical_bbox") or {}
        local_bbox_center = np.asarray([
            float(physical_bbox.get("local_center_x_m", 0.0)),
            float(physical_bbox.get("local_center_y_m", 0.0)),
            float(physical_bbox.get("local_center_z_m", 0.0)),
            1.0,
        ], dtype=np.float64)
        camera_center = (
            np.asarray(camera_tf.get_inverse_matrix(), dtype=np.float64)
            @ np.asarray(actor_tf.get_matrix(), dtype=np.float64)
            @ local_bbox_center
        )
        forward_m = float(camera_center[0])
        right_m = float(camera_center[1])
        up_m = float(camera_center[2])

        side = "right" if right_m > 0.0 else "left"
        configured_banks = state.he_close_view_matrix_csvs.get(side)
        if not configured_banks:
            return {
                "active": False,
                "reason": f"missing_{side}_close_bank",
                "side": side,
                "available_sides": sorted(
                    state.he_close_view_matrix_csvs
                ),
            }

        bank = self._get_close_banks_for_side(configured_banks, up_m)[0]

        forward_min = min(bank["forward_values"])
        forward_max = max(bank["forward_values"])
        right_min = min(bank["right_values"])
        right_max = max(bank["right_values"])
        up_min = min(bank["up_values"])
        up_max = max(bank["up_values"])

        right_steps = [
            upper - lower
            for lower, upper in zip(
                bank["right_values"],
                bank["right_values"][1:],
            )
        ]
        right_step = (
            min(right_steps)
            if right_steps
            else 0.0
        )
        inner_margin = self.close_inner_margin_cells * right_step
        outer_margin = self.close_outer_margin_cells * right_step
        if side == "right":
            right_query_min = right_min - inner_margin
            right_query_max = right_max + outer_margin
        else:
            right_query_min = right_min - outer_margin
            right_query_max = right_max + inner_margin
        if self.close_inner_min_bbox_clearance_m is not None:
            minimum_abs_right = (
                abs(float(physical_bbox.get("extent_y_m", 0.0)))
                + float(self.close_inner_min_bbox_clearance_m)
            )
            if side == "right":
                right_query_min = max(right_query_min, minimum_abs_right)
            else:
                right_query_max = min(right_query_max, -minimum_abs_right)

        reasons = []
        if not forward_min <= forward_m <= forward_max:
            reasons.append("forward_outside_close_bank")
        if not right_query_min <= right_m <= right_query_max:
            reasons.append("right_outside_close_bank")
        if forward_m > 0.0:
            bank_up_m = min(
                bank["up_values"],
                key=lambda value: abs(float(value) - up_m),
            )
            elevation_delta_deg = abs(
                math.degrees(math.atan2(up_m, forward_m))
                -
                math.degrees(math.atan2(float(bank_up_m), forward_m))
            )
            if elevation_delta_deg > 10.0:
                reasons.append("elevation_outside_close_bank")
        else:
            elevation_delta_deg = None
        selected_yaw = min(
            bank["yaw_values"],
            key=lambda value: self._angle_error_deg(value, state.rel_yaw_deg),
        )
        yaw_delta_deg = self._angle_error_deg(
            selected_yaw,
            state.rel_yaw_deg,
        )
        if not self._close_yaw_is_supported(state.rel_yaw_deg, selected_yaw):
            reasons.append("yaw_outside_close_bank")

        return {
            "active": False,
            "reason": ",".join(reasons) or "eligible_but_not_selected",
            "side": side,
            "query": {
                "forward_m": forward_m,
                "right_m": right_m,
                "up_m": up_m,
                "relative_yaw_deg": float(state.rel_yaw_deg),
            },
            "ranges": {
                "forward_m": [
                    forward_min,
                    forward_max,
                ],
                "right_m": [
                    right_query_min,
                    right_query_max,
                ],
                "up_m": [
                    up_min,
                    up_max,
                ],
            },
            "elevation_delta_deg": elevation_delta_deg,
            "yaw_delta_deg": yaw_delta_deg,
        }

    def _render_close_cartesian(
        self,
        base_rgb,
        state: ActorCameraState,
        actor_tf,
        camera_tf,
        view_matrix,
        width,
        height,
        fov,
        scene_depth_m=None,
        fallback_meta=None,
    ):
        selection = self._select_close_bank(
            state,
            actor_tf,
            camera_tf,
            view_matrix,
        )
        if selection is None:
            return None

        bank = selection["bank"]
        row = selection["row"]
        sprite_path = self._close_row_path(
            bank,
            row,
        )
        if not sprite_path.exists():
            return None

        sprite_rgba = self.sprite_cache.load_rgba(sprite_path)
        capture_width = float(row["image_width_px"])
        capture_height = float(row["image_height_px"])
        capture_fx = float(row["camera_fx_px"])
        runtime_fx = float(width) / (
            2.0 * math.tan(math.radians(float(fov)) / 2.0)
        )
        query = selection["query"]
        query_forward = max(float(query["forward_m"]), 1e-6)
        query_center_x = (
            float(width) / 2.0
            + runtime_fx * float(query["right_m"]) / query_forward
        )
        query_center_y = (
            float(height) / 2.0
            - runtime_fx * float(query["up_m"]) / query_forward
        )

        def runtime_visible_box(candidate):
            selected_forward = max(
                float(candidate["close_forward_m"]), 1e-6
            )
            candidate_capture_fx = float(candidate["camera_fx_px"])
            scale = (
                runtime_fx / candidate_capture_fx
                * selected_forward / query_forward
            )
            source_center_x = (
                float(candidate["camera_cx_px"])
                + candidate_capture_fx
                * float(candidate["close_right_m"])
                / selected_forward
            )
            source_center_y = (
                float(candidate["camera_cy_px"])
                - float(candidate["camera_fy_px"])
                * float(candidate["close_target_up_m"])
                / selected_forward
            )
            return {
                "x1": query_center_x + (
                    float(candidate["visible_x1_full_px"]) - source_center_x
                ) * scale,
                "x2": query_center_x + (
                    float(candidate["visible_x2_full_px"]) - source_center_x
                ) * scale,
                "y1": query_center_y + (
                    float(candidate["visible_y1_full_px"]) - source_center_y
                ) * scale,
                "y2": query_center_y + (
                    float(candidate["visible_y2_full_px"]) - source_center_y
                ) * scale,
                "scale": scale,
            }

        interpolation = selection["interpolation"]
        weighted_boxes = []
        for candidate in interpolation["rows"]:
            candidate_forward = float(candidate["close_forward_m"])
            candidate_right = float(candidate["close_right_m"])
            if interpolation["forward_lower"] == interpolation["forward_upper"]:
                forward_factor = 1.0
            elif candidate_forward == interpolation["forward_lower"]:
                forward_factor = 1.0 - interpolation["forward_weight"]
            else:
                forward_factor = interpolation["forward_weight"]
            if interpolation["right_lower"] == interpolation["right_upper"]:
                right_factor = 1.0
            elif candidate_right == interpolation["right_lower"]:
                right_factor = 1.0 - interpolation["right_weight"]
            else:
                right_factor = interpolation["right_weight"]
            weighted_boxes.append((
                forward_factor * right_factor,
                runtime_visible_box(candidate),
            ))
        total_weight = sum(weight for weight, unused in weighted_boxes)
        if total_weight <= 0.0:
            return None
        target = {
            name: sum(weight * box[name] for weight, box in weighted_boxes)
            / total_weight
            for name in ("x1", "x2", "y1", "y2")
        }
        target_x1 = target["x1"]
        target_x2 = target["x2"]
        target_y1 = target["y1"]
        target_y2 = target["y2"]
        projection_scale = runtime_visible_box(row)["scale"]

        forward_max = max(bank["forward_values"])
        right_min = min(bank["right_values"])
        right_max = max(bank["right_values"])
        forward_overflow = max(float(query["forward_m"]) - forward_max, 0.0)
        right_overflow = max(
            right_min - float(query["right_m"]),
            float(query["right_m"]) - right_max,
            0.0,
        )
        close_geometry_weight_linear = max(
            0.0,
            min(
                1.0,
                1.0 - forward_overflow / 1.5,
                1.0 - right_overflow / 1.0,
            ),
        )
        close_geometry_weight = (
            close_geometry_weight_linear
            * close_geometry_weight_linear
            * (3.0 - 2.0 * close_geometry_weight_linear)
        )
        fallback_width = None if fallback_meta is None else fallback_meta.get(
            "target_box_width_px"
        )
        fallback_height = None if fallback_meta is None else fallback_meta.get(
            "target_box_height_px"
        )
        if (
            close_geometry_weight < 1.0
            and fallback_width not in (None, "")
            and fallback_height not in (None, "")
        ):
            close_width = target_x2 - target_x1 + 1.0
            close_height = target_y2 - target_y1 + 1.0
            blended_width = (
                close_geometry_weight * close_width
                + (1.0 - close_geometry_weight) * float(fallback_width)
            )
            blended_height = (
                close_geometry_weight * close_height
                + (1.0 - close_geometry_weight) * float(fallback_height)
            )
            center_x = (target_x1 + target_x2) / 2.0
            bottom_y = target_y2
            target_x1 = center_x - (blended_width - 1.0) / 2.0
            target_x2 = center_x + (blended_width - 1.0) / 2.0
            target_y1 = bottom_y - (blended_height - 1.0)

        alpha_y, alpha_x = np.where(sprite_rgba[:, :, 3] > 10)
        if len(alpha_x) == 0 or len(alpha_y) == 0:
            return None
        source_anchor_x = (float(alpha_x.min()) + float(alpha_x.max())) / 2.0
        source_anchor_y = float(alpha_y.max())
        warped_rgba, resize_info = warp_view_matrix_sprite_to_box_subpixel(
            sprite_rgba=sprite_rgba,
            frame_w=int(width),
            frame_h=int(height),
            target_cx=(target_x1 + target_x2) / 2.0,
            target_bottom_y=target_y2,
            target_box_w=max(1.0, target_x2 - target_x1 + 1.0),
            target_box_h=max(1.0, target_y2 - target_y1 + 1.0),
            anchor_x=source_anchor_x,
            anchor_y=source_anchor_y,
            alpha_threshold=10,
        )
        if resize_info.get("fully_outside_frame", False):
            return None
        paste = resize_info["paste"]
        paste_x = int(paste["x1"])
        paste_y = int(paste["y1"])
        nearest_depth = max(
            0.05,
            query_forward
            + float(row["nearest_bbox_depth_m"])
            - float(row["actor_depth_m"]),
        )
        warped_rgba, occlusion_meta = apply_scene_depth_occlusion_to_sprite(
            sprite_rgba=warped_rgba,
            scene_depth_m=scene_depth_m,
            paste_x1=paste_x,
            paste_y1=paste_y,
            actor_nearest_depth_m=nearest_depth,
        )
        output, full_mask = alpha_composite_rgb(
            frame_rgb=base_rgb.copy(),
            sprite_rgba=warped_rgba,
            x1=paste_x,
            y1=paste_y,
            global_alpha=1.0,
        )
        rendered = bool(np.any(full_mask > 0))
        requested_view_angle = (
            math.degrees(math.atan2(
                float(query["right_m"]),
                float(query["forward_m"]),
            ))
            - float(query["relative_yaw_deg"])
            + 180.0
        ) % 360.0
        selected_view_angle = (
            math.degrees(math.atan2(
                float(row["close_right_m"]),
                float(row["close_forward_m"]),
            ))
            - float(row["close_relative_yaw_deg"])
            + 180.0
        ) % 360.0
        return output, {
            "rendered": rendered,
            "reason": "" if rendered else "close_cartesian_outside_viewport",
            "sprite_mode": "cartesian_close",
            "selected_angle": float(selected_view_angle),
            "viewpoint_angle_deg": float(requested_view_angle),
            "selected_distance_m": float(row["bbox_center_distance_m"]),
            "query_distance_m": float(selection["query"]["forward_m"]),
            "selected_elevation_deg": float(row["elevation_deg"]),
            "query_elevation_deg": "",
            "sprite_path": str(sprite_path),
            "sprite_width": int(warped_rgba.shape[1]),
            "sprite_height": int(warped_rgba.shape[0]),
            "paste_x1": paste_x,
            "paste_y1": paste_y,
            "anchor_mode": "cartesian_close_bbox_center_reprojection",
            "target_box_width_px": float(target_x2 - target_x1 + 1.0),
            "target_box_height_px": float(target_y2 - target_y1 + 1.0),
            "warp_scale_mode": "cartesian_close_focal_height_adapted",
            "resize_info": resize_info,
            "scene_occlusion": occlusion_meta,
            "cartesian_close": {
                "active": True,
                "side": selection["side"],
                "query": selection["query"],
                "selected": {
                    "forward_m": float(row["close_forward_m"]),
                    "right_m": float(row["close_right_m"]),
                    "up_m": float(row["close_target_up_m"]),
                    "relative_yaw_deg": float(row["close_relative_yaw_deg"]),
                },
                "runtime_projection": {
                    "runtime_fx_px": runtime_fx,
                    "capture_fx_px": capture_fx,
                    "projection_scale": projection_scale,
                    "target_visible_box": {
                        "x1": target_x1,
                        "y1": target_y1,
                        "x2": target_x2,
                        "y2": target_y2,
                    },
                    "interpolation": {
                        "forward_lower": interpolation["forward_lower"],
                        "forward_upper": interpolation["forward_upper"],
                        "forward_weight": interpolation["forward_weight"],
                        "right_lower": interpolation["right_lower"],
                        "right_upper": interpolation["right_upper"],
                        "right_weight": interpolation["right_weight"],
                        "selected_yaw": interpolation["selected_yaw"],
                        "close_geometry_weight": close_geometry_weight,
                        "close_geometry_weight_linear": (
                            close_geometry_weight_linear
                        ),
                    },
                },
                "qa_pass": row.get("qa_pass") == "True",
                "qa_flags": row.get("qa_flags", ""),
            },
        }

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
        scene_depth_m=None,
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

            # Compute the 4320 result during the Cartesian overlap as the
            # handoff geometry reference. Only the selected result is exposed;
            # this does not composite two actor sprites into the final frame.
            centered_close_4320 = (
                float(state.rel_z_m) <= 7.0
                and abs(float(state.rel_x_m)) < 3.0
            )
            fallback_rgb, fallback_meta = render_he_actor_view_matrix(
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
                bottom_y_offset_px=float(self.bottom_y_offset_px),
                projection_mode=(
                    self.centered_close_projection_mode
                    if centered_close_4320
                    else "oriented_2p5d_support"
                ),
                geometry_mode=(
                    self.centered_close_geometry_mode
                    if centered_close_4320
                    else self.fallback_far_geometry_mode
                ),
                scene_depth_m=scene_depth_m,
                silhouette_scale=self.silhouette_scale,
                warp_scale_mode=self.warp_scale_mode,
                viewpoint_lateral_sign=self.viewpoint_lateral_sign,
                center_depth_alpha_bbox_anchor=(
                    self.center_depth_alpha_bbox_anchor
                ),
                camera_rotation_reprojection=bool(
                    getattr(self, "camera_rotation_reprojection", False)
                ),
                camera_rotation_reprojection_min_width_fraction=float(
                    getattr(
                        self,
                        "camera_rotation_reprojection_min_width_fraction",
                        0.20,
                    )
                ),
                camera_rotation_reprojection_min_bearing_deg=float(
                    getattr(
                        self,
                        "camera_rotation_reprojection_min_bearing_deg",
                        30.0,
                    )
                ),
            )
            close_result = self._render_close_cartesian(
                base_rgb=rgb,
                state=state,
                actor_tf=actor_tf,
                camera_tf=camera_tf,
                view_matrix=view_matrix,
                width=int(width),
                height=int(height),
                fov=float(fov),
                scene_depth_m=scene_depth_m,
                fallback_meta=fallback_meta,
            )

            if close_result is not None:
                rgb, he_meta = close_result
            else:
                rgb, he_meta = fallback_rgb, fallback_meta
                he_meta["cartesian_close"] = (
                    self._close_selection_diagnostic(
                        state,
                        actor_tf,
                        camera_tf,
                        view_matrix,
                    )
                )
                he_meta["asset_selection_policy"] = {
                    "source": "view_matrix_4320",
                    "centered_close_4320": bool(centered_close_4320),
                    "projection_mode": he_meta.get("projection_mode"),
                }

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

                    he_close_view_matrix_csvs=
                        state.he_close_view_matrix_csvs,

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
