"""
tcp_he_pair_experiment_v1.py

First TCP <-> Hallucination Engine behavioral-equivalence experiment.

Conditions
----------
carla:
    The resolved adversary trajectory is rendered by a real CARLA vehicle.

he:
    No physical adversary vehicle is spawned.
    The exact same resolved physical trajectory is rendered as an HE sprite
    directly into TCP's native 900x256 RGB camera.

Important
---------
TCP is unchanged.

The scenario trajectory is unchanged.

The only intended experimental variable is:

    CARLA-rendered actor
            vs
    HE-rendered actor

The script records:
    - exact RGB input seen by TCP
    - annotated debug MP4
    - TCP controls
    - ego state
    - adversary physical state
    - bumper gap
    - TTC
    - virtual collision
    - HE sprite/placement metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import carla


# ============================================================
# Existing validated TCP integration
# ============================================================

from tcp_carla_0915_probe import (
    HE_ROOT,
    DEFAULT_TCP_ROOT,
    DEFAULT_CHECKPOINT,
    carla_image_to_rgb,
    get_vehicle_speed_mps,
    load_tcp_model,
    make_camera,
    prepare_tcp_rgb_native,
)

from tcp_carla_0915_route_probe import (
    TCPFusionState,
    build_route,
    closest_route_index,
    command_name,
    distance_2d,
    import_carla_agents,
    route_length,
    run_tcp_route,
)

from tcp_carla_0915_closed_loop import (
    import_tcp_route_planner,
    make_gnss,
    make_imu,
    get_named_sensor_frame,
    build_tcp_gps_plan,
    get_tcp_navigation,
    make_bootstrap_result,
)


# ============================================================
# Existing validated HE compositor utilities
# ============================================================

HE_RUNTIME_DIR = (
    HE_ROOT
    / "HE_v_0.1"
)

if str(HE_RUNTIME_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(HE_RUNTIME_DIR),
    )

from run_he_temporal_compositor_v1 import (
    SpriteCache,
    alpha_composite_rgb,
    compute_viewpoint_sprite_angle,
    discover_available_sprite_angles,
    resize_sprite_to_box,
    select_sprite,
)


DEFAULT_RESOLVED = (
    HE_ROOT
    / "ScenarioGenerator"
    / "outputs"
    / "v2_resolved"
    / "tcp_lead_brake_001.resolved_v2.json"
)

DEFAULT_SPRITE_ROOT = (
    HE_RUNTIME_DIR
    / "assets"
    / "sprite_bank_rgba"
    / "vehicle_blue_sedan"
)

DEFAULT_OUTPUT_ROOT = (
    HE_ROOT
    / "driving_models"
    / "TCP"
    / "outputs"
    / "tcp_he_pair_v1"
)


# ============================================================
# Basic utilities
# ============================================================

def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def normalize_angle_180(angle_deg):
    return (
        float(angle_deg) + 180.0
    ) % 360.0 - 180.0


def yaw_forward_right(yaw_deg):
    """
    CARLA horizontal frame.

    yaw=0:
        forward = +X
        right   = +Y
    """

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


# ============================================================
# Resolved Scenario v2
# ============================================================

def load_resolved_scenario(
    path: Path,
    actor_id: str,
):
    data = load_json(
        path
    )

    if (
        data.get("schema_version")
        != "2.0-resolved"
    ):
        raise ValueError(
            "Expected resolved ScenarioSchema v2."
        )

    actors = {
        row["actor_id"]: row
        for row in data.get(
            "actors",
            [],
        )
    }

    if actor_id not in actors:
        raise KeyError(
            f"Actor {actor_id!r} not found "
            f"in resolved scenario."
        )

    actor_info = (
        actors[
            actor_id
        ]
    )

    frames = [
        row
        for row
        in data.get(
            "actor_frames",
            [],
        )
        if row["actor_id"]
        == actor_id
    ]

    frames = sorted(
        frames,
        key=lambda row:
            int(row["frame_idx"]),
    )

    if not frames:
        raise RuntimeError(
            f"No resolved frames for "
            f"{actor_id}."
        )

    frame_ids = [
        int(row["frame_idx"])
        for row in frames
    ]

    expected = list(
        range(
            frame_ids[0],
            frame_ids[-1] + 1,
        )
    )

    if frame_ids != expected:
        raise RuntimeError(
            "Actor trajectory is not "
            "frame-contiguous."
        )

    return (
        data,
        actor_info,
        frames,
    )


def validate_tcp_camera(
    scenario,
):
    camera = scenario[
        "camera"
    ]

    checks = {
        "image_width":
            900,

        "image_height":
            256,

        "fov_deg":
            100.0,

        "x_m":
            -1.5,

        "y_m":
            0.0,

        "z_m":
            2.0,
    }

    for key, expected in (
        checks.items()
    ):
        actual = float(
            camera[key]
        )

        if abs(
            actual
            - float(expected)
        ) > 1e-6:
            raise RuntimeError(
                "Scenario camera is not "
                "TCP native camera:\n"
                f"{key}: "
                f"{actual} != {expected}"
            )


# ============================================================
# ScenarioGenerator physical frame -> CARLA world
# ============================================================

def sg_actor_to_world_transform(
    ego0_tf,
    actor_frame,
    world,
):
    """
    ScenarioGenerator v2:

        +x = forward
        +y = left
        +yaw = left

    CARLA local:

        forward = +local z in old HE notation
        right   = +local x
        positive CARLA yaw = right/clockwise

    Therefore:

        local forward =  SG x
        local right   = -SG y
        local yaw     = -SG yaw
    """

    forward, right = (
        yaw_forward_right(
            ego0_tf.rotation.yaw
        )
    )

    sg_forward = float(
        actor_frame["x_m"]
    )

    sg_left = float(
        actor_frame["y_m"]
    )

    world_xy = (
        np.array(
            [
                float(
                    ego0_tf.location.x
                ),
                float(
                    ego0_tf.location.y
                ),
            ],
            dtype=np.float64,
        )
        + forward
        * sg_forward
        - right
        * sg_left
    )

    loc = carla.Location(
        x=float(
            world_xy[0]
        ),
        y=float(
            world_xy[1]
        ),
        z=float(
            ego0_tf.location.z
        ),
    )

    waypoint = (
        world
        .get_map()
        .get_waypoint(
            loc,
            project_to_road=True,
            lane_type=(
                carla.LaneType.Driving
            ),
        )
    )

    if waypoint is not None:
        loc.z = float(
            waypoint
            .transform
            .location
            .z
            + 0.05
        )

    world_yaw = (
        float(
            ego0_tf.rotation.yaw
        )
        -
        float(
            actor_frame[
                "yaw_deg"
            ]
        )
    )

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=world_yaw,
            roll=0.0,
        ),
    )


# ============================================================
# TCP-camera geometric projection for HE
# ============================================================

def make_camera_intrinsic(
    width,
    height,
    fov_deg,
):
    focal = (
        float(width)
        /
        (
            2.0
            * math.tan(
                math.radians(
                    float(fov_deg)
                )
                / 2.0
            )
        )
    )

    k = np.identity(
        3,
        dtype=np.float64,
    )

    k[0, 0] = focal
    k[1, 1] = focal

    k[0, 2] = (
        float(width)
        / 2.0
    )

    k[1, 2] = (
        float(height)
        / 2.0
    )

    return k


def project_world_point(
    point,
    world_to_camera,
    k,
):
    """
    CARLA camera:

        X = forward
        Y = right
        Z = up

    image camera convention:

        [Y, -Z, X]
    """

    p = np.array(
        [
            float(point.x),
            float(point.y),
            float(point.z),
            1.0,
        ],
        dtype=np.float64,
    )

    p_cam = (
        world_to_camera
        @ p
    )

    depth = float(
        p_cam[0]
    )

    if depth <= 0.05:
        return None

    image_vector = (
        k
        @ np.array(
            [
                p_cam[1],
                -p_cam[2],
                p_cam[0],
            ],
            dtype=np.float64,
        )
    )

    u = float(
        image_vector[0]
        / image_vector[2]
    )

    v = float(
        image_vector[1]
        / image_vector[2]
    )

    return {
        "u":
            u,

        "v":
            v,

        "depth":
            depth,

        "camera_right_m":
            float(
                p_cam[1]
            ),

        "camera_up_m":
            float(
                p_cam[2]
            ),
    }


def project_virtual_actor(
    actor_tf,
    camera_tf,
    dimensions,
    width=900,
    height=256,
    fov=100.0,
):
    actor_height = float(
        dimensions[
            "height_m"
        ]
    )

    actor_width = float(
        dimensions[
            "width_m"
        ]
    )

    k = make_camera_intrinsic(
        width,
        height,
        fov,
    )

    world_to_camera = (
        np.array(
            camera_tf
            .get_inverse_matrix(),
            dtype=np.float64,
        )
    )

    bottom = carla.Location(
        x=float(
            actor_tf.location.x
        ),
        y=float(
            actor_tf.location.y
        ),
        z=float(
            actor_tf.location.z
        ),
    )

    top = carla.Location(
        x=float(
            actor_tf.location.x
        ),
        y=float(
            actor_tf.location.y
        ),
        z=float(
            actor_tf.location.z
            + actor_height
        ),
    )

    p_bottom = project_world_point(
        bottom,
        world_to_camera,
        k,
    )

    p_top = project_world_point(
        top,
        world_to_camera,
        k,
    )

    if (
        p_bottom is None
        or p_top is None
    ):
        return {
            "visible":
                False,

            "reason":
                "behind_camera",
        }

    depth = float(
        p_bottom["depth"]
    )

    if depth <= 1.0:
        return {
            "visible":
                False,

            "reason":
                "too_close",

            "depth_m":
                depth,
        }

    cx = float(
        p_bottom["u"]
    )

    bottom_y = float(
        p_bottom["v"]
    )

    box_height = abs(
        float(
            p_bottom["v"]
        )
        -
        float(
            p_top["v"]
        )
    )

    focal = float(
        k[0, 0]
    )

    box_width = (
        focal
        * actor_width
        / max(
            depth,
            1e-6,
        )
    )

    x1 = (
        cx
        - box_width / 2.0
    )

    x2 = (
        cx
        + box_width / 2.0
    )

    y1 = (
        bottom_y
        - box_height
    )

    y2 = (
        bottom_y
    )

    margin = 30.0

    visible = not (
        x2 < -margin
        or x1
        > float(width) + margin
        or y2 < -margin
        or y1
        > float(height) + margin
    )

    actor_relative_yaw = (
        normalize_angle_180(
            float(
                actor_tf.rotation.yaw
            )
            -
            float(
                camera_tf.rotation.yaw
            )
        )
    )

    return {
        "visible":
            bool(visible),

        "cx":
            cx,

        "bottom_y":
            bottom_y,

        "box_width":
            box_width,

        "box_height":
            box_height,

        "x1":
            x1,

        "x2":
            x2,

        "y1":
            y1,

        "y2":
            y2,

        "depth_m":
            depth,

        "camera_right_m":
            float(
                p_bottom[
                    "camera_right_m"
                ]
            ),

        "actor_relative_yaw_deg":
            actor_relative_yaw,
    }

# ============================================================
# Sprite alpha trimming
# ============================================================

def trim_sprite_to_visible_alpha(
    sprite_rgba,
    alpha_threshold=2,
    padding_px=2,
):
    """
    Remove transparent padding around an RGBA sprite.

    Why:
        Our sprite-bank images contain transparent crop margin.
        If the entire RGBA canvas is resized to the projected
        vehicle height, the visible car becomes too small.

    This trims the transparent border while retaining a tiny
    padding for anti-aliased edge pixels.

    Returns:
        trimmed_rgba
        metadata
    """

    if (
        sprite_rgba.ndim != 3
        or sprite_rgba.shape[2] != 4
    ):
        raise ValueError(
            "Expected HxWx4 RGBA sprite."
        )

    alpha = (
        sprite_rgba[
            :,
            :,
            3
        ]
    )

    ys, xs = np.where(
        alpha
        >
        int(
            alpha_threshold
        )
    )

    # Safety fallback.
    if (
        len(xs) == 0
        or len(ys) == 0
    ):
        return (
            sprite_rgba,
            {
                "trimmed":
                    False,

                "reason":
                    "empty_alpha",
            },
        )

    h, w = (
        sprite_rgba.shape[
            :2
        ]
    )

    x1 = max(
        0,
        int(xs.min())
        - int(padding_px),
    )

    x2 = min(
        w,
        int(xs.max())
        + 1
        + int(padding_px),
    )

    y1 = max(
        0,
        int(ys.min())
        - int(padding_px),
    )

    y2 = min(
        h,
        int(ys.max())
        + 1
        + int(padding_px),
    )

    trimmed = (
        sprite_rgba[
            y1:y2,
            x1:x2
        ]
        .copy()
    )

    return (
        trimmed,
        {
            "trimmed":
                True,

            "raw_width":
                int(w),

            "raw_height":
                int(h),

            "trimmed_width":
                int(
                    trimmed.shape[1]
                ),

            "trimmed_height":
                int(
                    trimmed.shape[0]
                ),

            "trim_x1":
                int(x1),

            "trim_y1":
                int(y1),

            "trim_x2":
                int(x2),

            "trim_y2":
                int(y2),
        },
    )
def alpha_composite_sprite_subpixel(
    frame_rgb,
    sprite_rgba,
    center_x,
    bottom_y,
    target_box_height,
    global_alpha=1.0,
):
    """
    Render an RGBA sprite with continuous scale and continuous placement.

    Unlike:
        resize -> integer width/height -> integer paste position

    this uses one affine transformation directly into the final image.

    Therefore smooth physical motion remains smooth at the rasterization
    stage and we avoid 1-pixel left/right/up/down oscillation.

    The RGB background itself is NOT resized or warped.
    """

    frame_h, frame_w = (
        frame_rgb.shape[:2]
    )

    src_h, src_w = (
        sprite_rgba.shape[:2]
    )

    if src_h < 2 or src_w < 2:
        return (
            frame_rgb,
            {
                "rendered": False,
                "reason": "sprite_too_small",
            },
        )

    # --------------------------------------------------------
    # Continuous scale
    # --------------------------------------------------------

    source_height_span = float(
        src_h - 1
    )

    scale = (
        float(target_box_height)
        /
        source_height_span
    )

    # Physical anchor in source sprite:
    # bottom-center.
    src_anchor_x = (
        float(src_w - 1)
        / 2.0
    )

    src_anchor_y = float(
        src_h - 1
    )

    # Translation that maps sprite bottom-center exactly onto
    # the continuously projected image bottom-center.
    tx = (
        float(center_x)
        -
        scale * src_anchor_x
    )

    ty = (
        float(bottom_y)
        -
        scale * src_anchor_y
    )

    affine = np.array(
        [
            [
                scale,
                0.0,
                tx,
            ],
            [
                0.0,
                scale,
                ty,
            ],
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Premultiplied-alpha representation
    #
    # Important for clean interpolation around transparent edges.
    # --------------------------------------------------------

    sprite = (
        sprite_rgba
        .astype(
            np.float32
        )
    )

    alpha = (
        sprite[
            :,
            :,
            3:4
        ]
        / 255.0
    )

    premult_rgb = (
        sprite[
            :,
            :,
            :3
        ]
        * alpha
    )

    premult_rgba = np.concatenate(
        [
            premult_rgb,
            alpha,
        ],
        axis=2,
    )

    # --------------------------------------------------------
    # Sub-pixel affine render directly into 900x256 canvas.
    # --------------------------------------------------------

    warped = cv2.warpAffine(
        premult_rgba,
        affine,
        (
            frame_w,
            frame_h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )

    warped_rgb = (
        warped[
            :,
            :,
            :3
        ]
    )

    warped_alpha = (
        warped[
            :,
            :,
            3:4
        ]
    )

    warped_alpha = np.clip(
        warped_alpha
        * float(global_alpha),
        0.0,
        1.0,
    )

    # premult_rgb must receive the same global-alpha scaling.
    warped_rgb = (
        warped_rgb
        * float(global_alpha)
    )

    background = (
        frame_rgb
        .astype(
            np.float32
        )
    )

    output = (
        warped_rgb
        +
        background
        * (
            1.0
            -
            warped_alpha
        )
    )

    output = np.clip(
        output,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    target_width = (
        float(src_w - 1)
        * scale
    )

    target_height = (
        float(src_h - 1)
        * scale
    )

    meta = {
        "rendered":
            True,

        "scale":
            float(scale),

        "target_width":
            float(target_width),

        "target_height":
            float(target_height),

        "continuous_x1":
            float(
                center_x
                -
                target_width / 2.0
            ),

        "continuous_y1":
            float(
                bottom_y
                -
                target_height
            ),

        "continuous_x2":
            float(
                center_x
                +
                target_width / 2.0
            ),

        "continuous_y2":
            float(
                bottom_y
            ),
    }

    return (
        output,
        meta,
    )
# ============================================================
# HE renderer
# ============================================================

def render_he_actor(
    base_rgb,
    actor_tf,
    camera_tf,
    dimensions,
    sprite_bank,
    available_angles,
    sprite_cache,
):
    frame = (
        base_rgb.copy()
    )

    box = (
        project_virtual_actor(
            actor_tf=actor_tf,
            camera_tf=camera_tf,
            dimensions=dimensions,
        )
    )

    meta = {
        "rendered":
            False,

        "box":
            box,

        "selected_angle":
            None,

        "angle_error_deg":
            None,
    }

    if not box.get(
        "visible",
        False,
    ):
        return frame, meta

    viewpoint_state = {
        "x_m":
            float(
                box[
                    "camera_right_m"
                ]
            ),

        "z_m":
            float(
                box[
                    "depth_m"
                ]
            ),

        "yaw_deg":
            float(
                box[
                    "actor_relative_yaw_deg"
                ]
            ),
    }

    viewpoint_angle = (
        compute_viewpoint_sprite_angle(
            viewpoint_state
        )
    )

    sprite_info = (
        select_sprite(
            sprite_bank=
                sprite_bank,

            relative_angle_deg=
                viewpoint_angle,

            available_angles=
                available_angles,
        )
    )

    if not sprite_info[
        "exists"
    ]:
        return frame, meta

    sprite_rgba_raw = (
        sprite_cache.load_rgba(
            sprite_info[
                "sprite_path"
            ]
        )
    )

    # Remove transparent crop margin before fitting the
    # sprite to the projected physical vehicle height.
    (
        sprite_rgba,
        trim_meta,
    ) = trim_sprite_to_visible_alpha(
        sprite_rgba_raw,
        alpha_threshold=2,
        padding_px=2,
    )

    (
        frame,
        subpixel_meta,
    ) = alpha_composite_sprite_subpixel(
        frame_rgb=
            frame,

        sprite_rgba=
            sprite_rgba,

        center_x=
            box[
                "cx"
            ],

        bottom_y=
            box[
                "bottom_y"
            ],

        target_box_height=
            box[
                "box_height"
            ],

        global_alpha=
            1.0,
    )

    meta.update({
        "rendered":
            True,

        "viewpoint_angle_deg":
            float(
                viewpoint_angle
            ),

        "sprite_trim":
            trim_meta,

        "selected_angle":
            int(
                sprite_info[
                    "selected_angle"
                ]
            ),

        "angle_error_deg":
            float(
                sprite_info[
                    "angle_error_deg"
                ]
            ),

        "paste_x1":
    float(
        subpixel_meta[
            "continuous_x1"
        ]
    ),

        "paste_y1":
            float(
                subpixel_meta[
                    "continuous_y1"
                ]
            ),

        "sprite_width":
            float(
                subpixel_meta[
                    "target_width"
                ]
            ),

        "sprite_height":
            float(
                subpixel_meta[
                    "target_height"
                ]
            ),

        "subpixel_render":
            True,

        "subpixel_scale":
            float(
                subpixel_meta[
                    "scale"
                ]
            ),
    })

    return frame, meta


# ============================================================
# Shared virtual safety metrics
# ============================================================

def compute_virtual_metrics(
    ego_tf,
    ego0_tf,
    ego_speed_mps,
    actor_frame,
    dimensions,
    ego,
):
    forward, right = (
        yaw_forward_right(
            ego0_tf.rotation.yaw
        )
    )

    delta = np.array(
        [
            float(
                ego_tf.location.x
                -
                ego0_tf.location.x
            ),

            float(
                ego_tf.location.y
                -
                ego0_tf.location.y
            ),
        ],
        dtype=np.float64,
    )

    ego_progress = float(
        np.dot(
            delta,
            forward,
        )
    )

    ego_right = float(
        np.dot(
            delta,
            right,
        )
    )

    # SG +y = left.
    ego_left = (
        -ego_right
    )

    actor_progress = float(
        actor_frame[
            "x_m"
        ]
    )

    actor_left = float(
        actor_frame[
            "y_m"
        ]
    )

    longitudinal_center_distance = (
        actor_progress
        -
        ego_progress
    )

    lateral_distance = abs(
        actor_left
        -
        ego_left
    )

    ego_length = float(
        ego.bounding_box.extent.x
        * 2.0
    )

    ego_width = float(
        ego.bounding_box.extent.y
        * 2.0
    )

    actor_length = float(
        dimensions[
            "length_m"
        ]
    )

    actor_width = float(
        dimensions[
            "width_m"
        ]
    )

    half_length_sum = (
        0.5
        *
        (
            ego_length
            +
            actor_length
        )
    )

    half_width_sum = (
        0.5
        *
        (
            ego_width
            +
            actor_width
        )
    )

    bumper_gap = (
        longitudinal_center_distance
        -
        half_length_sum
    )

    actor_speed = float(
        actor_frame[
            "speed_mps"
        ]
    )

    closing_speed = (
        float(
            ego_speed_mps
        )
        -
        actor_speed
    )

    if (
        bumper_gap > 0.0
        and closing_speed > 1e-3
    ):
        ttc = (
            bumper_gap
            /
            closing_speed
        )
    else:
        ttc = float(
            "inf"
        )

    virtual_collision = (
        abs(
            longitudinal_center_distance
        )
        <= half_length_sum
        and
        lateral_distance
        <= half_width_sum
    )

    return {
        "ego_progress_m":
            ego_progress,

        "ego_left_m":
            ego_left,

        "actor_progress_m":
            actor_progress,

        "actor_left_m":
            actor_left,

        "center_distance_m":
            longitudinal_center_distance,

        "lateral_distance_m":
            lateral_distance,

        "bumper_gap_m":
            bumper_gap,

        "closing_speed_mps":
            closing_speed,

        "ttc_s":
            ttc,

        "virtual_collision":
            bool(
                virtual_collision
            ),
    }


# ============================================================
# Debug MP4
# ============================================================

def make_debug_frame(
    rgb,
    condition,
    frame_idx,
    actor_frame,
    speed_mps,
    tcp,
    metrics,
    nav,
    he_meta,
):
    frame = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    overlay = (
        frame.copy()
    )

    cv2.rectangle(
        overlay,
        (0, 0),
        (900, 118),
        (0, 0, 0),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0.0,
    )

    ttc = (
        metrics[
            "ttc_s"
        ]
    )

    if math.isfinite(
        ttc
    ):
        ttc_text = (
            f"{ttc:.2f}s"
        )
    else:
        ttc_text = "inf"

    lines = [
        (
            f"{condition.upper()}  "
            f"Frame {frame_idx:04d}  "
            f"t={float(actor_frame['t_s']):.2f}s"
        ),

        (
            f"Ego {speed_mps:.2f}m/s  "
            f"Actor {float(actor_frame['speed_mps']):.2f}m/s  "
            f"Gap {metrics['bumper_gap_m']:.2f}m  "
            f"Lat {metrics['lateral_distance_m']:.2f}m  "
            f"TTC {ttc_text}"
        ),

        (
            f"TCP "
            f"S={tcp['steer']:+.3f} "
            f"T={tcp['throttle']:.3f} "
            f"B={tcp['brake']:.3f} "
            f"status={tcp['status_used']}"
        ),

        (
            f"CMD "
            f"{command_name(nav['command'])}  "
            f"Target "
            f"({nav['target_point'][0]:+.2f},"
            f"{nav['target_point'][1]:+.2f})"
        ),
    ]

    if condition == "he":
        lines.append(
            "HE "
            f"rendered="
            f"{int(he_meta.get('rendered', False))} "
            f"angle="
            f"{he_meta.get('selected_angle')}"
        )

    y = 20

    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 22

    return frame


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "carla",
            "he",
        ],
    )

    parser.add_argument(
        "--resolved",
        default=str(
            DEFAULT_RESOLVED
        ),
        help=(
            "Path to resolved ScenarioGenerator v2 JSON."
        ),
    )

    parser.add_argument(
        "--actor-id",
        default="adv_lead",
        help=(
            "Adversary actor_id inside the resolved scenario."
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
        "--spawn-index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--destination-index",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--desired-route-distance-m",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--route-sampling-resolution",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--carla-pythonapi",
        default=None,
    )

    parser.add_argument(
        "--tcp-root",
        default=str(
            DEFAULT_TCP_ROOT
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=str(
            DEFAULT_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--sprite-root",
        default=str(
            DEFAULT_SPRITE_ROOT
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--event-start-s",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--max-route-deviation-m",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--deviation-patience-frames",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    args = (
        parser.parse_args()
    )

    # ========================================================
    # Scenario
    # ========================================================

    resolved_path = Path(
        args.resolved
    ).resolve()

    (
        scenario,
        actor_info,
        actor_frames,
    ) = load_resolved_scenario(
        resolved_path,
        args.actor_id,
    )

    validate_tcp_camera(
        scenario
    )

    fps = float(
        scenario["fps"]
    )

    if abs(
        fps - 20.0
    ) > 1e-6:
        raise RuntimeError(
            "This first TCP pair experiment "
            "expects 20 Hz."
        )

    if args.max_frames > 0:
        actor_frames = (
            actor_frames[
                :args.max_frames
            ]
        )

    dimensions = (
        actor_info.get(
            "dimensions_m"
        )
        or
        {
            "length_m":
                4.2,

            "width_m":
                1.8,

            "height_m":
                1.5,
        }
    )

    blueprint_name = (
        actor_info.get(
            "asset_key"
        )
    )

    if (
        blueprint_name is None
        or not
        blueprint_name.startswith(
            "vehicle."
        )
    ):
        blueprint_name = (
            "vehicle.tesla.model3"
        )

    print(
        "[scenario]",
        scenario[
            "scenario_id"
        ],
    )

    print(
        "[resolved]",
        resolved_path,
    )

    print(
        "[frames]",
        len(
            actor_frames
        ),
    )

    print(
        "[fps]",
        fps,
    )

    print(
        "[condition]",
        args.condition,
    )

    print(
        "[adversary blueprint]",
        blueprint_name,
    )

    # ========================================================
    # CARLA navigation imports
    # ========================================================

    (
        _BasicAgent,
        GlobalRoutePlanner,
    ) = import_carla_agents(
        args.carla_pythonapi
    )

    tcp_root = Path(
        args.tcp_root
    ).resolve()

    RoutePlanner = (
        import_tcp_route_planner(
            tcp_root
        )
    )

    # ========================================================
    # TCP
    # ========================================================

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not
        torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested "
            "but unavailable."
        )

    if device.type == "cuda":
        print(
            "[GPU]",
            torch.cuda.get_device_name(
                0
            ),
        )

    net, config = (
        load_tcp_model(
            tcp_root=
                tcp_root,

            checkpoint_path=
                Path(
                    args.checkpoint
                ).resolve(),

            device=
                device,
        )
    )

    print(
        "[TCP] pred_len:",
        config.pred_len,
    )

    # ========================================================
    # HE sprite bank
    # ========================================================

    sprite_bank = {
        "root":
            str(
                Path(
                    args.sprite_root
                ).resolve()
            ),

        "rgba_dir":
            "rgba",

        "angle_format":
            "angle_{angle:03d}_rgba.png",
    }

    sprite_cache = None
    available_angles = None

    if args.condition == "he":

        available_angles = (
            discover_available_sprite_angles(
                sprite_bank
            )
        )

        sprite_cache = (
            SpriteCache()
        )

        print(
            "[HE sprites]",
            sprite_bank[
                "root"
            ],
        )

        print(
            "[HE angles]",
            len(
                available_angles
            ),
        )

    # ========================================================
    # Outputs
    # ========================================================

    output_dir = (
        Path(
            args.output_root
        )
        /
        scenario[
            "scenario_id"
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        scenario[
            "scenario_id"
        ]
        + "_"
        + args.condition
    )

    csv_path = (
        output_dir
        /
        (
            stem
            + ".csv"
        )
    )

    input_video_path = (
        output_dir
        /
        (
            stem
            + "_tcp_input.mp4"
        )
    )

    debug_video_path = (
        output_dir
        /
        (
            stem
            + "_debug.mp4"
        )
    )

    # ========================================================
    # CARLA
    # ========================================================

    client = carla.Client(
        args.host,
        args.port,
    )

    client.set_timeout(
        20.0
    )

    print(
        "[CARLA] loading:",
        args.town,
    )

    world = client.load_world(
        args.town
    )

    original_settings = (
        world.get_settings()
    )

    ego = None
    adversary = None

    camera = None
    gnss_sensor = None
    imu_sensor = None

    input_writer = None
    debug_writer = None
    csv_file = None

    try:

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = (
            True
        )

        settings.fixed_delta_seconds = (
            1.0 / fps
        )

        world.apply_settings(
            settings
        )

        carla_map = (
            world.get_map()
        )

        spawn_points = (
            carla_map
            .get_spawn_points()
        )

        spawn_idx = (
            args.spawn_index
            % len(
                spawn_points
            )
        )

        # ====================================================
        # Ego
        # ====================================================

        ego_bp = (
            world
            .get_blueprint_library()
            .filter(
                "vehicle.tesla.model3"
            )[0]
        )

        if ego_bp.has_attribute(
            "role_name"
        ):
            ego_bp.set_attribute(
                "role_name",
                "hero",
            )

        ego = world.try_spawn_actor(
            ego_bp,
            spawn_points[
                spawn_idx
            ],
        )

        if ego is None:
            raise RuntimeError(
                "Could not spawn ego."
            )

        # ====================================================
        # Route
        # ====================================================

        grp = GlobalRoutePlanner(
            carla_map,
            float(
                args.route_sampling_resolution
            ),
        )

        (
            route,
            destination_idx,
        ) = build_route(
            grp,
            spawn_points,
            spawn_idx,
            args.destination_index,
            args.desired_route_distance_m,
        )

        destination_location = (
            route[-1][0]
            .transform
            .location
        )

        print(
            "[route] destination:",
            destination_idx,
        )

        print(
            "[route] length:",
            f"{route_length(route):.1f} m",
        )

        tcp_gps_plan = (
            build_tcp_gps_plan(
                carla_map,
                route,
            )
        )

        tcp_route_planner = (
            RoutePlanner(
                4.0,
                50.0,
            )
        )

        tcp_route_planner.set_route(
            tcp_gps_plan,
            gps=True,
        )

        # ====================================================
        # TCP native sensors
        # ====================================================

        camera = make_camera(
            world=world,
            ego=ego,

            width=900,
            height=256,
            fov=100,

            x=-1.5,
            y=0.0,
            z=2.0,

            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        )

        gnss_sensor = make_gnss(
            world,
            ego,
        )

        imu_sensor = make_imu(
            world,
            ego,
        )

        camera_queue = (
            queue.Queue()
        )

        gnss_queue = (
            queue.Queue()
        )

        imu_queue = (
            queue.Queue()
        )

        camera.listen(
            camera_queue.put
        )

        gnss_sensor.listen(
            gnss_queue.put
        )

        imu_sensor.listen(
            imu_queue.put
        )

        # ====================================================
        # Stabilize ego
        # ====================================================

        hold_control = (
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )

        print(
            "[init] stabilizing ego..."
        )

        for _ in range(5):

            ego.apply_control(
                hold_control
            )

            frame = (
                world.tick()
            )

            get_named_sensor_frame(
                camera_queue,
                frame,
                "RGB camera",
            )

            get_named_sensor_frame(
                gnss_queue,
                frame,
                "GNSS",
            )

            get_named_sensor_frame(
                imu_queue,
                frame,
                "IMU",
            )

        # Exact common initial state.

        ego.set_transform(
            spawn_points[
                spawn_idx
            ]
        )

        ego.set_target_velocity(
            carla.Vector3D(
                0.0,
                0.0,
                0.0,
            )
        )

        ego.set_target_angular_velocity(
            carla.Vector3D(
                0.0,
                0.0,
                0.0,
            )
        )

        ego.apply_control(
            hold_control
        )

        ego0_tf = (
            ego.get_transform()
        )

        # ====================================================
        # CARLA adversary condition
        # ====================================================

        actor_tf0 = (
            sg_actor_to_world_transform(
                ego0_tf=
                    ego0_tf,

                actor_frame=
                    actor_frames[0],

                world=
                    world,
            )
        )

        if args.condition == "carla":

            adv_bp = (
                world
                .get_blueprint_library()
                .find(
                    blueprint_name
                )
            )

            if adv_bp.has_attribute(
                "role_name"
            ):
                adv_bp.set_attribute(
                    "role_name",
                    "scenario_adversary",
                )

            # Match the sprite-bank source color.
            if adv_bp.has_attribute(
                "color"
            ):
                adv_bp.set_attribute(
                    "color",
                    "0,0,255",
                )

            adversary = (
                world.try_spawn_actor(
                    adv_bp,
                    actor_tf0,
                )
            )

            if adversary is None:
                raise RuntimeError(
                    "Could not spawn "
                    "CARLA adversary."
                )

            # Scenario controls the pose.
            # Traffic Manager / vehicle AI must not alter it.
            adversary.set_simulate_physics(
                False
            )

            bb = (
                adversary.bounding_box
            )

            print(
                "[CARLA adversary bbox]",
                "L=%.3f W=%.3f H=%.3f"
                %
                (
                    2.0
                    * float(
                        bb.extent.x
                    ),

                    2.0
                    * float(
                        bb.extent.y
                    ),

                    2.0
                    * float(
                        bb.extent.z
                    ),
                ),
            )

        # ====================================================
        # First synchronized experimental frame
        # ====================================================

        current_frame = (
            world.tick()
        )

        current_image = (
            get_named_sensor_frame(
                camera_queue,
                current_frame,
                "RGB camera",
            )
        )

        current_gnss = (
            get_named_sensor_frame(
                gnss_queue,
                current_frame,
                "GNSS",
            )
        )

        current_imu = (
            get_named_sensor_frame(
                imu_queue,
                current_frame,
                "IMU",
            )
        )

        # ====================================================
        # Video
        # ====================================================

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        input_writer = (
            cv2.VideoWriter(
                str(
                    input_video_path
                ),
                fourcc,
                fps,
                (
                    900,
                    256,
                ),
            )
        )

        debug_writer = (
            cv2.VideoWriter(
                str(
                    debug_video_path
                ),
                fourcc,
                fps,
                (
                    900,
                    256,
                ),
            )
        )

        if (
            not input_writer.isOpened()
            or
            not debug_writer.isOpened()
        ):
            raise RuntimeError(
                "Could not create MP4 writer."
            )

        # ====================================================
        # CSV
        # ====================================================

        csv_fields = [
            "condition",
            "probe_idx",
            "carla_frame",
            "t_s",

            "ego_x",
            "ego_y",
            "ego_z",
            "ego_yaw",
            "ego_speed_mps",

            "actor_x_sg_m",
            "actor_y_sg_m",
            "actor_yaw_sg_deg",
            "actor_speed_mps",

            "actor_world_x",
            "actor_world_y",
            "actor_world_z",
            "actor_world_yaw",

            "bumper_gap_m",
            "center_distance_m",
            "lateral_distance_m",
            "closing_speed_mps",
            "ttc_s",
            "virtual_collision",

            "route_index",
            "route_deviation_m",
            "destination_distance_m",

            "command_name",
            "command_value",
            "target_x",
            "target_y",

            "tcp_steer",
            "tcp_throttle",
            "tcp_brake",

            "tcp_status_used",
            "tcp_status_next",

            "desired_speed",
            "pred_speed",

            "he_rendered",
            "he_depth_m",
            "he_cx",
            "he_bottom_y",
            "he_box_width",
            "he_box_height",
            "he_selected_angle",
            "he_angle_error_deg",
            "he_viewpoint_angle_deg",
            "he_sprite_width",
            "he_sprite_height",
            "bootstrap",
        ]

        csv_file = open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        # ====================================================
        # Runtime state
        # ====================================================

        fusion = (
            TCPFusionState()
        )

        route_idx = 0
        deviation_counter = 0

        completed_frames = 0

        min_gap = float(
            "inf"
        )

        any_virtual_collision = (
            False
        )

        first_brake_after_event = (
            None
        )

        # ====================================================
        # Banner
        # ====================================================

        print()
        print("=" * 78)
        print(
            "TCP <-> HE PAIRED EXPERIMENT V1"
        )
        print("=" * 78)

        print(
            "condition :",
            args.condition,
        )

        print(
            "scenario  :",
            scenario[
                "scenario_id"
            ],
        )

        print(
            "TCP input : 900x256 FOV100"
        )

        print(
            "frames    :",
            len(
                actor_frames
            ),
        )

        print("=" * 78)
        print()

        # ====================================================
        # Experiment loop
        # ====================================================

        for i, actor_frame in enumerate(
            actor_frames
        ):

            # ------------------------------------------------
            # Current physical adversary truth
            # ------------------------------------------------

            actor_tf = (
                sg_actor_to_world_transform(
                    ego0_tf=
                        ego0_tf,

                    actor_frame=
                        actor_frame,

                    world=
                        world,
                )
            )

            # ------------------------------------------------
            # Base RGB
            # ------------------------------------------------

            base_rgb = (
                prepare_tcp_rgb_native(
                    carla_image_to_rgb(
                        current_image
                    )
                )
            )

            # ------------------------------------------------
            # Only experimental difference
            # ------------------------------------------------

            if args.condition == "he":

                tcp_rgb, he_meta = (
                    render_he_actor(
                        base_rgb=
                            base_rgb,

                        actor_tf=
                            actor_tf,

                        camera_tf=
                            current_image.transform,

                        dimensions=
                            dimensions,

                        sprite_bank=
                            sprite_bank,

                        available_angles=
                            available_angles,

                        sprite_cache=
                            sprite_cache,
                    )
                )

            else:

                tcp_rgb = (
                    base_rgb
                )

                he_meta = {
                    "rendered":
                        False,

                    "box":
                        {},
                }

            # ------------------------------------------------
            # TCP navigation/state
            # ------------------------------------------------

            speed_mps = (
                get_vehicle_speed_mps(
                    ego
                )
            )

            nav = (
                get_tcp_navigation(
                    tcp_route_planner,
                    current_gnss,
                    current_imu,
                )
            )

            bootstrap = (
                i == 0
            )

            if bootstrap:

                tcp = (
                    make_bootstrap_result()
                )

            else:

                tcp = (
                    run_tcp_route(
                        net,
                        tcp_rgb,
                        speed_mps,
                        nav[
                            "target_point"
                        ],
                        nav[
                            "command_value"
                        ],
                        device,
                        fusion,
                    )
                )

            # ------------------------------------------------
            # TCP controls ego - unchanged
            # ------------------------------------------------

            ego.apply_control(
                carla.VehicleControl(
                    steer=float(
                        tcp[
                            "steer"
                        ]
                    ),

                    throttle=float(
                        tcp[
                            "throttle"
                        ]
                    ),

                    brake=float(
                        tcp[
                            "brake"
                        ]
                    ),

                    hand_brake=False,

                    manual_gear_shift=False,
                )
            )

            # ------------------------------------------------
            # Current ego state
            # ------------------------------------------------

            ego_tf = (
                ego.get_transform()
            )

            ego_loc = (
                ego_tf.location
            )

            route_idx = (
                closest_route_index(
                    route,
                    ego_loc,
                    route_idx,
                )
            )

            route_loc = (
                route[
                    route_idx
                ][0]
                .transform
                .location
            )

            route_deviation = (
                distance_2d(
                    ego_loc,
                    route_loc,
                )
            )

            destination_distance = (
                distance_2d(
                    ego_loc,
                    destination_location,
                )
            )

            # ------------------------------------------------
            # Shared virtual safety metrics
            # ------------------------------------------------

            metrics = (
                compute_virtual_metrics(
                    ego_tf=
                        ego_tf,

                    ego0_tf=
                        ego0_tf,

                    ego_speed_mps=
                        speed_mps,

                    actor_frame=
                        actor_frame,

                    dimensions=
                        dimensions,

                    ego=
                        ego,
                )
            )

            min_gap = min(
                min_gap,
                float(
                    metrics[
                        "bumper_gap_m"
                    ]
                ),
            )

            if metrics[
                "virtual_collision"
            ]:
                any_virtual_collision = (
                    True
                )

            if (
                first_brake_after_event
                is None
                and
                float(
                    actor_frame[
                        "t_s"
                    ]
                )
                >=
                args.event_start_s
                and
                float(
                    tcp[
                        "brake"
                    ]
                )
                >= 0.5
            ):
                first_brake_after_event = {
                    "frame_idx":
                        i,

                    "t_s":
                        float(
                            actor_frame[
                                "t_s"
                            ]
                        ),

                    "gap_m":
                        float(
                            metrics[
                                "bumper_gap_m"
                            ]
                        ),
                }

            # ------------------------------------------------
            # Route safety
            # ------------------------------------------------

            if (
                route_deviation
                >
                args.max_route_deviation_m
            ):
                deviation_counter += 1
            else:
                deviation_counter = 0

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            he_box = (
                he_meta.get(
                    "box",
                    {},
                )
            )

            ttc = float(
                metrics[
                    "ttc_s"
                ]
            )

            writer.writerow({
                "condition":
                    args.condition,

                "probe_idx":
                    i,

                "carla_frame":
                    current_frame,

                "t_s":
                    float(
                        actor_frame[
                            "t_s"
                        ]
                    ),

                "ego_x":
                    float(
                        ego_loc.x
                    ),

                "ego_y":
                    float(
                        ego_loc.y
                    ),

                "ego_z":
                    float(
                        ego_loc.z
                    ),

                "ego_yaw":
                    float(
                        ego_tf.rotation.yaw
                    ),

                "ego_speed_mps":
                    speed_mps,

                "actor_x_sg_m":
                    float(
                        actor_frame[
                            "x_m"
                        ]
                    ),

                "actor_y_sg_m":
                    float(
                        actor_frame[
                            "y_m"
                        ]
                    ),

                "actor_yaw_sg_deg":
                    float(
                        actor_frame[
                            "yaw_deg"
                        ]
                    ),

                "actor_speed_mps":
                    float(
                        actor_frame[
                            "speed_mps"
                        ]
                    ),

                "actor_world_x":
                    float(
                        actor_tf.location.x
                    ),

                "actor_world_y":
                    float(
                        actor_tf.location.y
                    ),

                "actor_world_z":
                    float(
                        actor_tf.location.z
                    ),

                "actor_world_yaw":
                    float(
                        actor_tf.rotation.yaw
                    ),

                "bumper_gap_m":
                    metrics[
                        "bumper_gap_m"
                    ],

                "center_distance_m":
                    metrics[
                        "center_distance_m"
                    ],

                "lateral_distance_m":
                    metrics[
                        "lateral_distance_m"
                    ],

                "closing_speed_mps":
                    metrics[
                        "closing_speed_mps"
                    ],

                "ttc_s":
                    (
                        ttc
                        if math.isfinite(
                            ttc
                        )
                        else ""
                    ),

                "virtual_collision":
                    int(
                        metrics[
                            "virtual_collision"
                        ]
                    ),

                "route_index":
                    route_idx,

                "route_deviation_m":
                    route_deviation,

                "destination_distance_m":
                    destination_distance,
                "he_viewpoint_angle_deg":
                    (
                        he_meta.get(
                            "viewpoint_angle_deg",
                            "",
                        )
                        if args.condition == "he"
                        else ""
                    ),

                "he_sprite_width":
                    (
                        he_meta.get(
                            "sprite_width",
                            "",
                        )
                        if args.condition == "he"
                        else ""
                    ),

                "he_sprite_height":
                    (
                        he_meta.get(
                            "sprite_height",
                            "",
                        )
                        if args.condition == "he"
                        else ""
                    ),
                "command_name":
                    command_name(
                        nav[
                            "command"
                        ]
                    ),

                "command_value":
                    nav[
                        "command_value"
                    ],

                "target_x":
                    float(
                        nav[
                            "target_point"
                        ][0]
                    ),

                "target_y":
                    float(
                        nav[
                            "target_point"
                        ][1]
                    ),

                "tcp_steer":
                    tcp[
                        "steer"
                    ],

                "tcp_throttle":
                    tcp[
                        "throttle"
                    ],

                "tcp_brake":
                    tcp[
                        "brake"
                    ],

                "tcp_status_used":
                    tcp[
                        "status_used"
                    ],

                "tcp_status_next":
                    tcp[
                        "status_next"
                    ],

                "desired_speed":
                    tcp[
                        "desired_speed"
                    ],

                "pred_speed":
                    tcp[
                        "pred_speed"
                    ],

                "he_rendered":
                    int(
                        he_meta.get(
                            "rendered",
                            False,
                        )
                    ),

                "he_depth_m":
                    he_box.get(
                        "depth_m",
                        "",
                    ),

                "he_cx":
                    he_box.get(
                        "cx",
                        "",
                    ),

                "he_bottom_y":
                    he_box.get(
                        "bottom_y",
                        "",
                    ),

                "he_box_width":
                    he_box.get(
                        "box_width",
                        "",
                    ),

                "he_box_height":
                    he_box.get(
                        "box_height",
                        "",
                    ),

                "he_selected_angle":
                    (
                        he_meta.get(
                            "selected_angle"
                        )
                        if args.condition
                        == "he"
                        else ""
                    ),

                "he_angle_error_deg":
                    (
                        he_meta.get(
                            "angle_error_deg"
                        )
                        if args.condition
                        == "he"
                        else ""
                    ),

                "bootstrap":
                    int(
                        bootstrap
                    ),
            })

            # ------------------------------------------------
            # Exact image given to TCP
            # ------------------------------------------------

            input_writer.write(
                cv2.cvtColor(
                    tcp_rgb,
                    cv2.COLOR_RGB2BGR,
                )
            )

            # ------------------------------------------------
            # Human-readable debug video
            # ------------------------------------------------

            debug_frame = (
                make_debug_frame(
                    rgb=
                        tcp_rgb,

                    condition=
                        args.condition,

                    frame_idx=
                        i,

                    actor_frame=
                        actor_frame,

                    speed_mps=
                        speed_mps,

                    tcp=
                        tcp,

                    metrics=
                        metrics,

                    nav=
                        nav,

                    he_meta=
                        he_meta,
                )
            )

            debug_writer.write(
                debug_frame
            )

            completed_frames += 1

            # ------------------------------------------------
            # Console
            # ------------------------------------------------

            if (
                i < 10
                or
                i % 20 == 0
            ):

                ttc_value = (
                    metrics[
                        "ttc_s"
                    ]
                )

                ttc_text = (
                    f"{ttc_value:.2f}"
                    if math.isfinite(
                        ttc_value
                    )
                    else "inf"
                )

                print(
                    f"[{i:04d}] "
                    f"t={float(actor_frame['t_s']):5.2f} "
                    f"ego={speed_mps:4.2f} "
                    f"lead={float(actor_frame['speed_mps']):4.2f} "
                    f"gap={metrics['bumper_gap_m']:5.2f} "
                    f"TTC={ttc_text:>5s} | "
                    f"TCP "
                    f"S={tcp['steer']:+.3f} "
                    f"T={tcp['throttle']:.3f} "
                    f"B={tcp['brake']:.3f} "
                    f"| HE="
                    f"{int(he_meta.get('rendered', False))}"
                )

            # ------------------------------------------------
            # Route departure
            # ------------------------------------------------

            if (
                deviation_counter
                >=
                args.deviation_patience_frames
            ):

                print()
                print(
                    "[SAFETY] TCP departed route."
                )

                ego.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                    )
                )

                break

            # ------------------------------------------------
            # End of scenario
            # ------------------------------------------------

            if (
                i + 1
                >= len(
                    actor_frames
                )
            ):
                break

            # ------------------------------------------------
            # Advance physical CARLA adversary trajectory
            # BEFORE next synchronous tick.
            # ------------------------------------------------

            if (
                args.condition
                == "carla"
            ):

                next_actor_tf = (
                    sg_actor_to_world_transform(
                        ego0_tf=
                            ego0_tf,

                        actor_frame=
                            actor_frames[
                                i + 1
                            ],

                        world=
                            world,
                    )
                )

                adversary.set_transform(
                    next_actor_tf
                )

            # ------------------------------------------------
            # Advance closed-loop world
            # ------------------------------------------------

            current_frame = (
                world.tick()
            )

            current_image = (
                get_named_sensor_frame(
                    camera_queue,
                    current_frame,
                    "RGB camera",
                )
            )

            current_gnss = (
                get_named_sensor_frame(
                    gnss_queue,
                    current_frame,
                    "GNSS",
                )
            )

            current_imu = (
                get_named_sensor_frame(
                    imu_queue,
                    current_frame,
                    "IMU",
                )
            )

        # ====================================================
        # Complete
        # ====================================================

        csv_file.flush()

        print()
        print("=" * 78)
        print(
            "TCP <-> HE EXPERIMENT COMPLETE"
        )
        print("=" * 78)

        print(
            "condition:",
            args.condition,
        )

        print(
            "frames:",
            completed_frames,
        )

        print(
            "minimum bumper gap:",
            f"{min_gap:.3f} m",
        )

        print(
            "virtual collision:",
            any_virtual_collision,
        )

        if (
            first_brake_after_event
            is None
        ):
            print(
                "first TCP brake after event: none"
            )

        else:
            print(
                "first TCP brake after event:",
                "frame",
                first_brake_after_event[
                    "frame_idx"
                ],
                "t=",
                f"{first_brake_after_event['t_s']:.2f}s",
                "gap=",
                f"{first_brake_after_event['gap_m']:.2f}m",
            )

        print(
            "TCP input video:",
            input_video_path,
        )

        print(
            "debug video:",
            debug_video_path,
        )

        print(
            "CSV:",
            csv_path,
        )

        print("=" * 78)

    finally:

        print(
            "[cleanup]"
        )

        if csv_file is not None:
            csv_file.close()

        if input_writer is not None:
            input_writer.release()

        if debug_writer is not None:
            debug_writer.release()

        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

            camera.destroy()

        if gnss_sensor is not None:
            try:
                gnss_sensor.stop()
            except Exception:
                pass

            gnss_sensor.destroy()

        if imu_sensor is not None:
            try:
                imu_sensor.stop()
            except Exception:
                pass

            imu_sensor.destroy()

        if adversary is not None:
            adversary.destroy()

        if ego is not None:
            ego.destroy()

        world.apply_settings(
            original_settings
        )


if __name__ == "__main__":
    main()