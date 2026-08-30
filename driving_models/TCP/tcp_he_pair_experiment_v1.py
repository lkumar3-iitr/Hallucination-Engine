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
# Shared deterministic CARLA ego initialization
# ============================================================

COMMON_DIR = (
    HE_ROOT
    / "driving_models"
    / "common"
)

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(COMMON_DIR),
    )

from route_progress_metrics_v1 import (
    RouteProjector,
    compute_route_pair_metrics,
    route_virtual_collision,
)

from traffic_light_schedule_v1 import (
    TrafficLightScheduleExecutor,
)

from carla_ego_initialization import (
    canonicalize_ego_start,
)

from he_camera_renderer import (
    render_he_actor_view_matrix,
    load_view_matrix_sprite_bank,
)
from sprite_native_geometry_v1 import (
    load_sprite_native_geometry,
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
def ensure_dir(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def image_to_rgb_array(image):
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )
    array = array.reshape(
        (
            image.height,
            image.width,
            4,
        )
    )
    rgb = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return rgb


def decode_instance_segmentation(image):
    """
    Decode CARLA instance-segmentation raw BGRA image.

    raw_data byte order:
        B = channel 0
        G = channel 1
        R = channel 2

    CARLA encoding:
        R -> semantic class
        G/B -> instance identifier
    """
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )

    array = array.reshape(
        (
            image.height,
            image.width,
            4,
        )
    )

    blue = array[:, :, 0].astype(
        np.uint16
    )

    green = array[:, :, 1].astype(
        np.uint16
    )

    red = array[:, :, 2].astype(
        np.uint8
    )

    instance_id = (
        green
        + (blue << 8)
    )

    return (
        red,
        instance_id,
    )


def make_instance_camera_from_parent_rgb(
    world,
    ego,
    parent_rgb_camera,
):
    """
    Spawn an instance-segmentation camera with exactly the
    TCP-native camera intrinsics and relative mounting pose.

    TCP native camera:
        900x256
        FOV 100
        x=-1.5, y=0.0, z=2.0
    """

    camera_bp = (
        world
        .get_blueprint_library()
        .find(
            "sensor.camera.instance_segmentation"
        )
    )

    camera_bp.set_attribute(
        "image_size_x",
        parent_rgb_camera.attributes.get(
            "image_size_x",
            "900",
        ),
    )

    camera_bp.set_attribute(
        "image_size_y",
        parent_rgb_camera.attributes.get(
            "image_size_y",
            "256",
        ),
    )

    camera_bp.set_attribute(
        "fov",
        parent_rgb_camera.attributes.get(
            "fov",
            "100",
        ),
    )

    # IMPORTANT:
    # This transform is RELATIVE to the ego because
    # attach_to=ego is used.
    camera_tf = carla.Transform(
        carla.Location(
            x=-1.5,
            y=0.0,
            z=2.0,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        ),
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_tf,
        attach_to=ego,
    )

    return camera


def make_depth_camera_from_parent_rgb(
    world,
    ego,
    parent_rgb_camera,
):
    """
    Spawn a CARLA metric-depth camera exactly aligned with
    TCP's native RGB camera.

    TCP native camera:
        900x256
        FOV 100
        x=-1.5, y=0.0, z=2.0
        pitch=0, yaw=0, roll=0

    The depth camera is used ONLY for HE/world occlusion.
    It is not supplied to TCP and does not change the model.
    """

    camera_bp = (
        world
        .get_blueprint_library()
        .find(
            "sensor.camera.depth"
        )
    )

    camera_bp.set_attribute(
        "image_size_x",
        parent_rgb_camera.attributes.get(
            "image_size_x",
            "900",
        ),
    )

    camera_bp.set_attribute(
        "image_size_y",
        parent_rgb_camera.attributes.get(
            "image_size_y",
            "256",
        ),
    )

    camera_bp.set_attribute(
        "fov",
        parent_rgb_camera.attributes.get(
            "fov",
            "100",
        ),
    )

    # IMPORTANT:
    # Same RELATIVE transform as TCP RGB camera.
    camera_tf = carla.Transform(
        carla.Location(
            x=-1.5,
            y=0.0,
            z=2.0,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        ),
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_tf,
        attach_to=ego,
    )

    return camera


def carla_depth_image_to_m(
    depth_image,
):
    """
    Decode CARLA sensor.camera.depth raw BGRA image into
    metric forward-ray distance in metres.

    CARLA depth encoding:

        normalized =
            (R + 256*G + 256^2*B)
            / (256^3 - 1)

        depth_m =
            1000 * normalized

    raw_data is BGRA, therefore:
        B = channel 0
        G = channel 1
        R = channel 2
    """

    array = np.frombuffer(
        depth_image.raw_data,
        dtype=np.uint8,
    )

    array = array.reshape(
        (
            depth_image.height,
            depth_image.width,
            4,
        )
    )

    blue = (
        array[
            :,
            :,
            0,
        ]
        .astype(
            np.float32
        )
    )

    green = (
        array[
            :,
            :,
            1,
        ]
        .astype(
            np.float32
        )
    )

    red = (
        array[
            :,
            :,
            2,
        ]
        .astype(
            np.float32
        )
    )

    normalized = (
        red
        +
        green * 256.0
        +
        blue * 65536.0
    ) / 16777215.0

    depth_m = (
        normalized
        * 1000.0
    )

    return depth_m.astype(
        np.float32,
        copy=False,
    )


def mask_geometry_from_binary(mask_u8):
    """
    mask_u8: uint8 image, 0/255
    """
    ys, xs = np.where(mask_u8 > 0)

    if len(xs) == 0:
        return {
            "visible": False,
            "x1": None,
            "y1": None,
            "x2": None,
            "y2": None,
            "width": 0,
            "height": 0,
            "cx": None,
            "bottom_y": None,
            "area": 0,
        }

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    width = int(x2 - x1 + 1)
    height = int(y2 - y1 + 1)
    area = int((mask_u8 > 0).sum())

    return {
        "visible": True,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": width,
        "height": height,
        "cx": 0.5 * (x1 + x2),
        "bottom_y": float(y2),
        "area": area,
    }


def extract_vehicle_instance_mask_from_bbox(
    instance_image,
    projected_bbox,
    semantic_tags,
):
    """
    Same logic as record_carla_he_pair.py, adapted locally.

    We identify the target actor instance using:
      1. semantic tags
      2. projected bbox crop
      3. dominant instance id inside the crop
    """
    semantic_id, instance_id = decode_instance_segmentation(
        instance_image
    )

    height, width = semantic_id.shape

    target_semantic_tags = {
        int(tag)
        for tag in semantic_tags
    }

    if not target_semantic_tags:
        return (
            np.zeros(
                (height, width),
                dtype=np.uint8,
            ),
            {
                "found": False,
                "reason": "actor_has_no_semantic_tags",
            },
        )

    if not projected_bbox.get(
        "visible",
        False,
    ):
        return (
            np.zeros(
                (height, width),
                dtype=np.uint8,
            ),
            {
                "found": False,
                "reason": "projected_bbox_not_visible",
                "expected_semantic_tags": sorted(
                    target_semantic_tags
                ),
            },
        )

    x1 = max(
        0,
        int(np.floor(projected_bbox["x1"])),
    )
    y1 = max(
        0,
        int(np.floor(projected_bbox["y1"])),
    )
    x2 = min(
        width - 1,
        int(np.ceil(projected_bbox["x2"])),
    )
    y2 = min(
        height - 1,
        int(np.ceil(projected_bbox["y2"])),
    )

    if x2 < x1 or y2 < y1:
        return (
            np.zeros(
                (height, width),
                dtype=np.uint8,
            ),
            {
                "found": False,
                "reason": "invalid_projected_bbox_crop",
                "expected_semantic_tags": sorted(
                    target_semantic_tags
                ),
            },
        )

    crop_semantic = semantic_id[
        y1:y2 + 1,
        x1:x2 + 1,
    ]
    crop_instance = instance_id[
        y1:y2 + 1,
        x1:x2 + 1,
    ]

    candidate_pixels = np.isin(
        crop_semantic,
        list(target_semantic_tags),
    )

    if not candidate_pixels.any():
        return (
            np.zeros(
                (height, width),
                dtype=np.uint8,
            ),
            {
                "found": False,
                "reason": "no_target_semantic_pixels_in_projected_bbox",
                "expected_semantic_tags": sorted(
                    target_semantic_tags
                ),
                "bbox_crop": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
            },
        )

    candidate_instance_values = crop_instance[
        candidate_pixels
    ]

    unique_ids, counts = np.unique(
        candidate_instance_values,
        return_counts=True,
    )

    # remove instance id 0 if present
    keep = unique_ids != 0
    unique_ids = unique_ids[keep]
    counts = counts[keep]

    if len(unique_ids) == 0:
        return (
            np.zeros(
                (height, width),
                dtype=np.uint8,
            ),
            {
                "found": False,
                "reason": "only_background_instance_ids",
                "expected_semantic_tags": sorted(
                    target_semantic_tags
                ),
                "bbox_crop": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
            },
        )

    selected_instance_id = int(
        unique_ids[np.argmax(counts)]
    )

    full_mask = np.logical_and(
        np.isin(
            semantic_id,
            list(target_semantic_tags),
        ),
        instance_id == selected_instance_id,
    )

    mask_u8 = np.zeros(
        (height, width),
        dtype=np.uint8,
    )
    mask_u8[full_mask] = 255

    return (
        mask_u8,
        {
            "found": True,
            "method": "semantic_tag_plus_projected_bbox",
            "expected_semantic_tags": sorted(
                target_semantic_tags
            ),
            "selected_instance_id": selected_instance_id,
            "pixels_in_projected_bbox": int(
                candidate_pixels.sum()
            ),
            "mask_pixels": int(
                full_mask.sum()
            ),
            "bbox_crop": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
        },
    )


def write_binary_mask_png(mask_u8, out_path):
    ensure_dir(
        Path(out_path).parent
    )
    ok = cv2.imwrite(
        str(out_path),
        mask_u8,
    )
    if not ok:
        raise RuntimeError(
            f"Failed to write mask: {out_path}"
        )


def write_rgb_png(rgb, out_path):
    ensure_dir(
        Path(out_path).parent
    )
    bgr = rgb[:, :, ::-1]
    ok = cv2.imwrite(
        str(out_path),
        bgr,
    )
    if not ok:
        raise RuntimeError(
            f"Failed to write image: {out_path}"
        )
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
        "--route-metrics-csv",
        default=None,
        help=(
            "Optional inspected route.csv used for "
            "turn-aware route-progress gap/TTC metrics."
        ),
    )
    parser.add_argument(
        "--environment-json",
        default=None,
        help=(
            "Optional ScenarioGenerator environment sidecar "
            "containing deterministic traffic-light/weather events."
        ),
    )   
    parser.add_argument(
        "--he-bottom-y-offset-px",
        type=float,
        default=0.0,
        help=(
            "Optional HE sprite vertical placement correction "
            "in native-camera pixels. Negative moves HE upward."
        ),
    )

    parser.add_argument(
        "--he-scene-depth-occlusion",
        action="store_true",
        help=(
            "Enable CARLA/world-to-HE depth occlusion using a "
            "depth camera exactly aligned with TCP's native RGB camera."
        ),
    )

    parser.add_argument(
        "--he-scene-occlusion-margin-m",
        type=float,
        default=0.25,
        help=(
            "Depth safety margin in metres for CARLA/world-to-HE "
            "occlusion. Scene geometry must be this much closer "
            "than the HE actor before masking HE pixels."
        ),
    )

    parser.add_argument(
        "--dump-native-equivalence-v1",
        action="store_true",
        help=(
            "Dump native TCP RGB frames and actor masks for "
            "CARLA-vs-HE adversary equivalence analysis."
        ),
    )

    parser.add_argument(
        "--dump-native-rgb",
        action="store_true",
        help=(
            "When dumping native equivalence, also save the native TCP RGB "
            "frame PNG for each probe."
        ),
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
        "--view-matrix-csv",
        default=None,
        help=(
            "Production 4320 sprite-bank "
            "view_matrix.csv."
        ),
    )

    parser.add_argument(
        "--he-geometry-mode",
        choices=[
            "proxy",
            "sprite_native",
        ],
        default="proxy",
        help=(
            "HE visual geometry provider. "
            "'proxy' preserves the existing render-dimensions path. "
            "'sprite_native' derives visible width/height from "
            "sprite_geometry_v1.csv."
        ),
    )

    parser.add_argument(
        "--sprite-geometry-csv",
        default=None,
        help=(
            "sprite_geometry_v1.csv used when "
            "--he-geometry-mode sprite_native."
        ),
    )

    parser.add_argument(
        "--distance-selection-mode",
        choices=[
            "linear",
            "log",
            "inverse_depth",
        ],
        default="linear",
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

    # --------------------------------------------------------
    # Scenario camera metadata is intentionally ignored.
    #
    # ResolvedScenarioV2 describes model-independent physical
    # truth. TCP always uses its own validated native camera:
    #
    #     900 x 256
    #     FOV 100 deg
    #     x=-1.5, y=0.0, z=2.0
    #
    # Therefore the legacy camera block stored in the scenario
    # must not constrain TCP execution.
    # --------------------------------------------------------

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

    # ========================================================
    # Actor dimensions
    #
    # physical_dimensions:
    #     true physical footprint used for safety metrics.
    #
    # render_dimensions:
    #     visual proxy dimensions used only by HE projection.
    #
    # These must not be conflated. CARLA's full 3-D bounding
    # box does not necessarily equal the visible 2-D silhouette.
    # ========================================================

    physical_dimensions = (
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

    render_dimensions = (
        actor_info.get(
            "render_dimensions_m"
        )
        or
        physical_dimensions
    )

    print(
        "[physical dimensions]",
        physical_dimensions,
    )

    print(
        "[render dimensions]",
        render_dimensions,
    )
    # ========================================================
    # HE projection dimensions
    #
    # proxy:
    #     Preserve the existing visual-proxy projection.
    #
    # sprite_native:
    #     Use true physical actor geometry only for actor/camera
    #     projection, position, visibility and contact reference.
    #     Final visible width/height come from the sprite bank.
    # ========================================================

    if args.he_geometry_mode == "sprite_native":

        he_projection_dimensions = (
            physical_dimensions
        )

    else:

        he_projection_dimensions = (
            render_dimensions
        )

    print(
        "[HE geometry mode]",
        args.he_geometry_mode,
    )

    print(
        "[HE projection dimensions]",
        he_projection_dimensions,
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
    # Production HE 4320 view-matrix bank
    # ========================================================

    sprite_bank = None
    view_matrix = None
    sprite_cache = None
    sprite_geometry = None

    if args.condition == "he":

        if args.view_matrix_csv is None:
            raise RuntimeError(
                "--view-matrix-csv is required "
                "for HE condition."
            )

        view_matrix_csv = Path(
            args.view_matrix_csv
        ).resolve()

        sprite_bank = {
            "mode":
                "view_matrix",

            "view_matrix_csvs": [
                str(
                    view_matrix_csv
                )
            ],

            "target_height_m":
                0.75,

            # actor state comes from the actual TCP
            # camera/world geometry.
            "vertical_mode":
                "state_y",

            # Informational here because state_y is used.
            "camera_height_m":
                2.0,

            "distance_selection_mode":
                args.distance_selection_mode,
        }

        view_matrix = (
            load_view_matrix_sprite_bank(
                sprite_bank
            )
        )

        sprite_cache = (
            SpriteCache()
        )
        if args.he_geometry_mode == "sprite_native":

            if args.sprite_geometry_csv is None:

                raise RuntimeError(
                    "--sprite-geometry-csv is required "
                    "when --he-geometry-mode sprite_native."
                )

            sprite_geometry_csv = Path(
                args.sprite_geometry_csv
            ).resolve()

            sprite_geometry = (
                load_sprite_native_geometry(
                    sprite_geometry_csv
                )
            )

            print(
                "[HE sprite geometry]",
                sprite_geometry_csv,
            )

            print(
                "[HE geometry version]",
                sprite_geometry.geometry_version,
            )

            print(
                "[HE geometry alpha threshold]",
                sprite_geometry.geometry_alpha_threshold,
            )
        print(
            "[HE view matrix]",
            view_matrix_csv,
        )

        print(
            "[HE views]",
            len(
                view_matrix[
                    "records"
                ]
            ),
        )

        print(
            "[HE distance mode]",
            args.distance_selection_mode,
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

    # ========================================================
    # Native CARLA <-> HE adversary equivalence dumps
    # ========================================================

    native_equiv_dirs = None
    native_equiv_jsonl_path = None
    native_equiv_jsonl_fp = None

    if args.dump_native_equivalence_v1:

        native_equiv_root = (
            output_dir
            / "native_equivalence_v1"
            / args.condition
        )

        native_equiv_dirs = {
            "root":
                native_equiv_root,

            "rgb":
                native_equiv_root
                / "rgb",

            "masks_carla":
                native_equiv_root
                / "masks_carla",

            "masks_he":
                native_equiv_root
                / "masks_he",
        }

        for _path in native_equiv_dirs.values():

            ensure_dir(
                _path
            )

        native_equiv_jsonl_path = (
            native_equiv_root
            / "native_equivalence_rows.jsonl"
        )

        native_equiv_jsonl_fp = open(
            native_equiv_jsonl_path,
            "w",
            encoding="utf-8",
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
    depth_camera = None

    gnss_sensor = None
    imu_sensor = None

    depth_camera_queue = None

    instance_camera = None
    instance_camera_queue = None

    input_writer = None
    debug_writer = None
    csv_file = None
    traffic_light_executor = None
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
        # Deterministic ego initialization
        #
        # Every closed-loop model experiment must begin from
        # the same physically settled and canonical CARLA state.
        #
        # Important:
        # do this BEFORE spawning/listening to sensors so the
        # settling ticks do not accumulate stale sensor frames.
        # ====================================================

        (
            ego0_tf,
            ego0_state,
        ) = canonicalize_ego_start(
            world=world,
            ego=ego,
            nominal_spawn_tf=spawn_points[
                spawn_idx
            ],
            settle_ticks=30,
            hold_ticks=5,
        )

        print(
            "[init] canonical ego state ready"
        )
        # ====================================================
        # EXPERIMENT START BARRIER
        #
        # No TCP inference, TCP temporal state, or TCP-generated
        # vehicle control is permitted before this point.
        # ====================================================

        print(
            "[init] ========================================"
        )
        print(
            "[init] EXPERIMENT START BARRIER"
        )
        print(
            "[init] ego settled and canonicalized"
        )
        print(
            "[init] model history is empty"
        )
        print(
            "[init] ========================================"
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

        # ----------------------------------------------------
        # Optional TCP-native aligned depth camera.
        #
        # Used only for CARLA/world -> HE occlusion.
        # TCP itself still receives RGB only.
        # ----------------------------------------------------

        if (
            args.condition == "he"
            and
            args.he_scene_depth_occlusion
        ):

            depth_camera = (
                make_depth_camera_from_parent_rgb(
                    world=world,
                    ego=ego,
                    parent_rgb_camera=camera,
                )
            )

            print(
                "[HE scene occlusion] "
                "TCP-aligned depth camera enabled"
            )

            print(
                "[HE scene occlusion] margin:",
                f"{args.he_scene_occlusion_margin_m:.3f} m",
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

        if depth_camera is not None:

            depth_camera_queue = (
                queue.Queue()
            )

            depth_camera.listen(
                depth_camera_queue.put
            )

        gnss_sensor.listen(
            gnss_queue.put
        )

        imu_sensor.listen(
            imu_queue.put
        )
        # ====================================================
        # Native TCP instance-segmentation camera
        #
        # Diagnostic only. It does not affect TCP input.
        # ====================================================

        if (
            args.dump_native_equivalence_v1
            and
            args.condition == "carla"
        ):

            instance_camera = (
                make_instance_camera_from_parent_rgb(
                    world=world,
                    ego=ego,
                    parent_rgb_camera=camera,
                )
            )

            instance_camera_queue = (
                queue.Queue()
            )

            instance_camera.listen(
                instance_camera_queue.put
            )

            print(
                "[native equivalence] "
                "instance camera enabled"
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
        # Deterministic environment schedule
        # ====================================================

        if (
            args.environment_json
            is not None
        ):

            traffic_light_executor = (
                TrafficLightScheduleExecutor.from_json(
                    world=world,
                    carla_module=carla,
                    path=args.environment_json,
                )
            )

            traffic_light_executor.initialize()

            print(
                "[environment]",
                Path(
                    args.environment_json
                ).resolve(),
            )

            print(
                "[traffic-light initial state]",
                traffic_light_executor
                .primary_state_name(
                    0.0
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

        current_depth_image = None
        current_depth_m = None

        if depth_camera_queue is not None:

            current_depth_image = (
                get_named_sensor_frame(
                    depth_camera_queue,
                    current_frame,
                    "Depth camera",
                )
            )

            current_depth_m = (
                carla_depth_image_to_m(
                    current_depth_image
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
        current_instance_image = None

        if instance_camera_queue is not None:

            current_instance_image = (
                get_named_sensor_frame(
                    instance_camera_queue,
                    current_frame,
                    "instance camera",
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
            "traffic_light_state",
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

            # Turn-aware route metrics.
            "ego_route_progress_m",
            "actor_route_progress_m",

            "route_center_gap_m",
            "route_bumper_gap_m",

            "ego_route_lateral_m",
            "actor_route_lateral_m",
            "route_lateral_separation_m",

            "ego_route_speed_mps",
            "actor_route_speed_mps",
            "route_closing_speed_mps",

            "route_ttc_s",
            "route_virtual_collision",

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

            "he_scene_occlusion_enabled",
            "he_scene_occluded_fraction",
            "he_scene_occluded_pixels",
            "he_scene_remaining_pixels",

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

        # TCP temporal/model runtime state begins only after
        # physical CARLA initialization has completed.

        fusion = (
            TCPFusionState()
        )
        # ====================================================
        # Turn-aware route metrics
        # ====================================================

        route_projector = None

        route_metric_ego_segment = None
        route_metric_actor_segment = None

        min_route_gap = float(
            "inf"
        )

        any_route_virtual_collision = False

        ego_bb = (
            ego.bounding_box
        )

        ego_length_m = (
            2.0
            * float(
                ego_bb.extent.x
            )
        )

        ego_width_m = (
            2.0
            * float(
                ego_bb.extent.y
            )
        )

        if (
            args.route_metrics_csv
            is not None
        ):

            route_projector = (
                RouteProjector.from_csv(
                    args.route_metrics_csv
                )
            )

            print(
                "[route metrics]",
                Path(
                    args.route_metrics_csv
                ).resolve(),
            )

            print(
                "[route metrics length]",
                f"{route_projector.total_length_m:.2f} m",
            )

            print(
                "[ego physical dimensions]",
                f"L={ego_length_m:.3f} "
                f"W={ego_width_m:.3f}",
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
            # Current deterministic environment state
            #
            # The environment for this camera frame was applied
            # before the synchronous CARLA tick. Query the same
            # scenario timestamp here so CSV/logging records the
            # state actually associated with this frame.
            # ------------------------------------------------

            if (
                traffic_light_executor
                is not None
            ):

                traffic_light_state = (
                    traffic_light_executor
                    .primary_state_name(
                        float(
                            actor_frame[
                                "t_s"
                            ]
                        )
                    )
                )

            else:

                traffic_light_state = ""

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

            # ------------------------------------------------
            # Only experimental difference
            # ------------------------------------------------

            he_mask_u8 = None

            if args.condition == "he":
                tcp_rgb, he_meta = (
                    render_he_actor_view_matrix(
                        base_rgb=
                            base_rgb,

                        actor_tf=
                            actor_tf,

                        camera_tf=
                            current_image.transform,

                        dimensions=
                            he_projection_dimensions,

                        sprite_bank=
                            sprite_bank,

                        view_matrix=
                            view_matrix,

                        sprite_cache=
                            sprite_cache,

                        width=
                            900,

                        height=
                            256,

                        fov=
                            100.0,
                        bottom_y_offset_px=
                            args.he_bottom_y_offset_px,   
                        geometry_mode=
                            args.he_geometry_mode,

                        sprite_geometry=
                            sprite_geometry,

                        scene_depth_m=
                            current_depth_m,

                        scene_occlusion_margin_m=
                            args.he_scene_occlusion_margin_m,
                    )
                )
                # --------------------------------------------
                # Diagnostic HE alpha-mask reconstruction
                #
                # Render the identical actor onto black and
                # white backgrounds. From:
                #
                #   O = alpha * F + (1-alpha) * B
                #
                # white_render - black_render
                #     = 255 * (1-alpha)
                #
                # Therefore the actor alpha can be recovered
                # without modifying the frozen renderer.
                # --------------------------------------------

                if args.dump_native_equivalence_v1:

                    black_background = (
                        np.zeros_like(
                            base_rgb
                        )
                    )

                    white_background = (
                        np.full_like(
                            base_rgb,
                            255,
                        )
                    )

                    (
                        he_black,
                        _he_black_meta,
                    ) = (
                        render_he_actor_view_matrix(
                            base_rgb=
                                black_background,

                            actor_tf=
                                actor_tf,

                            camera_tf=
                                current_image.transform,

                            dimensions=
                                he_projection_dimensions,

                            sprite_bank=
                                sprite_bank,

                            view_matrix=
                                view_matrix,

                            sprite_cache=
                                sprite_cache,

                            width=
                                900,

                            height=
                                256,

                            fov=
                                100.0,
                            bottom_y_offset_px=
                                args.he_bottom_y_offset_px,
                            geometry_mode=
                                args.he_geometry_mode,

                            sprite_geometry=
                                sprite_geometry,

                            scene_depth_m=
                                current_depth_m,

                            scene_occlusion_margin_m=
                                args.he_scene_occlusion_margin_m,
                        )
                    )

                    (
                        he_white,
                        _he_white_meta,
                    ) = (
                        render_he_actor_view_matrix(
                            base_rgb=
                                white_background,

                            actor_tf=
                                actor_tf,

                            camera_tf=
                                current_image.transform,

                            dimensions=
                                he_projection_dimensions,

                            sprite_bank=
                                sprite_bank,

                            view_matrix=
                                view_matrix,

                            sprite_cache=
                                sprite_cache,

                            width=
                                900,

                            height=
                                256,

                            fov=
                                100.0,
                            bottom_y_offset_px=
                                args.he_bottom_y_offset_px,
                            geometry_mode=
                                args.he_geometry_mode,

                            sprite_geometry=
                                sprite_geometry,

                            scene_depth_m=
                                current_depth_m,

                            scene_occlusion_margin_m=
                                args.he_scene_occlusion_margin_m,
                        )
                    )

                    diff = (
                        he_white.astype(
                            np.float32
                        )
                        -
                        he_black.astype(
                            np.float32
                        )
                    )

                    transparency = (
                        np.mean(
                            diff,
                            axis=2,
                        )
                        / 255.0
                    )

                    alpha = (
                        1.0
                        -
                        transparency
                    )

                    alpha = np.clip(
                        alpha,
                        0.0,
                        1.0,
                    )

                    he_mask_u8 = np.rint(
                        alpha
                        * 255.0
                    ).astype(
                        np.uint8
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
                        physical_dimensions,

                    ego=
                        ego,
                )
            )
            # ------------------------------------------------
            # Turn-aware route safety metrics
            # ------------------------------------------------

            route_metrics = None
            route_collision = False

            if (
                route_projector
                is not None
            ):

                ego_velocity = (
                    ego.get_velocity()
                )

                route_metrics = (
                    compute_route_pair_metrics(
                        projector=
                            route_projector,

                        ego_x=
                            ego_loc.x,

                        ego_y=
                            ego_loc.y,

                        ego_vx=
                            ego_velocity.x,

                        ego_vy=
                            ego_velocity.y,

                        actor_x=
                            actor_tf.location.x,

                        actor_y=
                            actor_tf.location.y,

                        ego_length_m=
                            ego_length_m,

                        actor_length_m=
                            physical_dimensions[
                                "length_m"
                            ],

                        actor_route_speed_mps=
                            float(
                                actor_frame[
                                    "speed_mps"
                                ]
                            ),

                        previous_ego_segment_idx=
                            route_metric_ego_segment,

                        previous_actor_segment_idx=
                            route_metric_actor_segment,
                    )
                )

                route_metric_ego_segment = (
                    route_metrics
                    .ego
                    .segment_idx
                )

                route_metric_actor_segment = (
                    route_metrics
                    .actor
                    .segment_idx
                )

                route_collision = (
                    route_virtual_collision(
                        metrics=
                            route_metrics,

                        ego_length_m=
                            ego_length_m,

                        actor_length_m=
                            physical_dimensions[
                                "length_m"
                            ],

                        ego_width_m=
                            ego_width_m,

                        actor_width_m=
                            physical_dimensions[
                                "width_m"
                            ],
                    )
                )

                min_route_gap = min(
                    min_route_gap,
                    float(
                        route_metrics
                        .bumper_gap_m
                    ),
                )

                if route_collision:
                    any_route_virtual_collision = (
                        True
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
                        (
                            float(
                                route_metrics
                                .bumper_gap_m
                            )
                            if route_metrics is not None
                            else
                            float(
                                metrics[
                                    "bumper_gap_m"
                                ]
                            )
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

            he_scene_occlusion = (
                he_meta.get(
                    "scene_occlusion",
                    {},
                )
            )

            ttc = float(
                metrics[
                    "ttc_s"
                ]
            )
            # ------------------------------------------------
            # Native CARLA <-> HE adversary equivalence dump
            # ------------------------------------------------

            if args.dump_native_equivalence_v1:

                frame_tag = (
                    f"{i:06d}"
                )

                rgb_path = (
                    native_equiv_dirs[
                        "rgb"
                    ]
                    /
                    (
                        frame_tag
                        + ".png"
                    )
                )

                carla_mask_path = (
                    native_equiv_dirs[
                        "masks_carla"
                    ]
                    /
                    (
                        frame_tag
                        + ".png"
                    )
                )

                he_mask_path = (
                    native_equiv_dirs[
                        "masks_he"
                    ]
                    /
                    (
                        frame_tag
                        + ".png"
                    )
                )

                # Exact RGB image supplied to TCP.
                if args.dump_native_rgb:

                    write_rgb_png(
                        tcp_rgb,
                        rgb_path,
                    )

                native_row = {
                    "probe_idx":
                        int(i),

                    "carla_frame":
                        int(
                            current_frame
                        ),

                    "t_s":
                        float(
                            actor_frame[
                                "t_s"
                            ]
                        ),

                    "condition":
                        args.condition,

                    "rgb_path":
                        (
                            str(
                                rgb_path
                            )
                            if args.dump_native_rgb
                            else None
                        ),

                    "carla_mask_path":
                        None,

                    "he_mask_path":
                        None,

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

                    "bumper_gap_m":
                        float(
                            metrics[
                                "bumper_gap_m"
                            ]
                        ),

                    "ego_speed_mps":
                        float(
                            speed_mps
                        ),

                    "tcp_steer":
                        float(
                            tcp[
                                "steer"
                            ]
                        ),

                    "tcp_throttle":
                        float(
                            tcp[
                                "throttle"
                            ]
                        ),

                    "tcp_brake":
                        float(
                            tcp[
                                "brake"
                            ]
                        ),

                    "carla_mask_geometry":
                        None,

                    "carla_mask_info":
                        None,

                    "he_mask_geometry":
                        None,

                    "he_mask_info":
                        None,
                }

                # ============================================
                # CARLA condition
                # ============================================

                if args.condition == "carla":

                    carla_mask_u8 = np.zeros(
                        (
                            256,
                            900,
                        ),
                        dtype=np.uint8,
                    )

                    carla_mask_info = {
                        "found":
                            False,

                        "reason":
                            "instance_image_unavailable",
                    }

                    if (
                        current_instance_image
                        is not None
                        and
                        adversary is not None
                    ):

                        projected_bbox_native = (
                            project_virtual_actor(
                                actor_tf=
                                    actor_tf,

                                camera_tf=
                                    current_image.transform,

                                dimensions=
                                    physical_dimensions,

                                width=
                                    900,

                                height=
                                    256,

                                fov=
                                    100.0,
                            )
                        )

                        semantic_tags = list(
                            getattr(
                                adversary,
                                "semantic_tags",
                                [],
                            )
                        )

                        (
                            carla_mask_u8,
                            carla_mask_info,
                        ) = (
                            extract_vehicle_instance_mask_from_bbox(
                                instance_image=
                                    current_instance_image,

                                projected_bbox=
                                    projected_bbox_native,

                                semantic_tags=
                                    semantic_tags,
                            )
                        )

                    write_binary_mask_png(
                        carla_mask_u8,
                        carla_mask_path,
                    )

                    native_row[
                        "carla_mask_path"
                    ] = str(
                        carla_mask_path
                    )

                    native_row[
                        "carla_mask_geometry"
                    ] = (
                        mask_geometry_from_binary(
                            carla_mask_u8
                        )
                    )

                    native_row[
                        "carla_mask_info"
                    ] = (
                        carla_mask_info
                    )

                # ============================================
                # HE condition
                # ============================================

                else:

                    if he_mask_u8 is None:

                        he_mask_u8 = np.zeros(
                            (
                                256,
                                900,
                            ),
                            dtype=np.uint8,
                        )

                    write_binary_mask_png(
                        he_mask_u8,
                        he_mask_path,
                    )

                    native_row[
                        "he_mask_path"
                    ] = str(
                        he_mask_path
                    )

                    native_row[
                        "he_mask_geometry"
                    ] = (
                        mask_geometry_from_binary(
                            he_mask_u8
                        )
                    )

                    native_row[
                        "he_mask_info"
                    ] = {
                        "found":
                            bool(
                                he_meta.get(
                                    "rendered",
                                    False,
                                )
                            ),

                        "method":
                            (
                                "black_white_alpha_reconstruction"
                            ),

                        "selected_angle":
                            he_meta.get(
                                "selected_angle"
                            ),

                        "viewpoint_angle_deg":
                            he_meta.get(
                                "viewpoint_angle_deg"
                            ),

                        "query_distance_m":
                            he_meta.get(
                                "query_distance_m"
                            ),

                        "selected_distance_m":
                            he_meta.get(
                                "selected_distance_m"
                            ),

                        "query_elevation_deg":
                            he_meta.get(
                                "query_elevation_deg"
                            ),

                        "selected_elevation_deg":
                            he_meta.get(
                                "selected_elevation_deg"
                            ),

                        "sprite_path":
                            he_meta.get(
                                "sprite_path"
                            ),
                        "geometry_mode":
                            he_meta.get(
                                "geometry_mode"
                            ),

                        "geometry_version":
                            he_meta.get(
                                "geometry_version"
                            ),

                        "geometry_alpha_threshold":
                            he_meta.get(
                                "geometry_alpha_threshold"
                            ),

                        "target_box_width_px":
                            he_meta.get(
                                "target_box_width_px"
                            ),

                        "target_box_height_px":
                            he_meta.get(
                                "target_box_height_px"
                            ),

                        "sprite_native_geometry":
                            he_meta.get(
                                "sprite_native_geometry"
                            ),

                        "depth_m":
                            he_box.get(
                                "depth_m"
                            ),

                        "projected_cx_px":
                            he_box.get(
                                "cx"
                            ),

                        "projected_bottom_y_px":
                            he_box.get(
                                "bottom_y"
                            ),

                        "actor_reference_cx_px":
                            he_box.get(
                                "actor_reference_cx_px"
                            ),

                        "actor_reference_bottom_y_px":
                            he_box.get(
                                "actor_reference_bottom_y_px"
                            ),

                        "render_bottom_y_px":
                            he_meta.get(
                                "render_bottom_y"
                            ),

                        "projected_box_width_px":
                            he_box.get(
                                "box_width"
                            ),

                        "projected_box_height_px":
                            he_box.get(
                                "box_height"
                            ),

                        "bottom_y_offset_px":
                            he_meta.get(
                                "bottom_y_offset_px"
                            ),
                    }

                native_equiv_jsonl_fp.write(
                    json.dumps(
                        native_row
                    )
                    + "\n"
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
                "traffic_light_state":
                    traffic_light_state,
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

                # --------------------------------------------
                # Turn-aware route metrics
                # --------------------------------------------

                "ego_route_progress_m":
                    (
                        float(
                            route_metrics
                            .ego
                            .s_m
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "actor_route_progress_m":
                    (
                        float(
                            route_metrics
                            .actor
                            .s_m
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "route_center_gap_m":
                    (
                        float(
                            route_metrics
                            .center_gap_m
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "route_bumper_gap_m":
                    (
                        float(
                            route_metrics
                            .bumper_gap_m
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "ego_route_lateral_m":
                    (
                        float(
                            route_metrics
                            .ego
                            .lateral_m
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "actor_route_lateral_m":
                    (
                        float(
                            route_metrics
                            .actor
                            .lateral_m
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "route_lateral_separation_m":
                    (
                        float(
                            route_metrics
                            .lateral_separation_m
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "ego_route_speed_mps":
                    (
                        float(
                            route_metrics
                            .ego_route_speed_mps
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "actor_route_speed_mps":
                    (
                        float(
                            route_metrics
                            .actor_route_speed_mps
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "route_closing_speed_mps":
                    (
                        float(
                            route_metrics
                            .closing_speed_mps
                        )
                        if route_metrics
                        is not None
                        else ""
                    ),

                "route_ttc_s":
                    (
                        float(
                            route_metrics
                            .ttc_s
                        )
                        if (
                            route_metrics
                            is not None
                            and
                            math.isfinite(
                                route_metrics
                                .ttc_s
                            )
                        )
                        else ""
                    ),

                "route_virtual_collision":
                    (
                        int(
                            route_collision
                        )
                        if route_metrics
                        is not None
                        else ""
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

                "he_scene_occlusion_enabled":
                    int(
                        bool(
                            he_scene_occlusion.get(
                                "enabled",
                                False,
                            )
                        )
                    ),

                "he_scene_occluded_fraction":
                    he_scene_occlusion.get(
                        "occluded_fraction",
                        "",
                    ),

                "he_scene_occluded_pixels":
                    he_scene_occlusion.get(
                        "occluded_alpha_pixels",
                        "",
                    ),

                "he_scene_remaining_pixels":
                    he_scene_occlusion.get(
                        "remaining_alpha_pixels",
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
                    f"{int(he_meta.get('rendered', False))} "
                    f"OCC="
                    f"{100.0 * float(he_scene_occlusion.get('occluded_fraction', 0.0) or 0.0):5.1f}%"
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
            # Prepare deterministic environment for
            # camera frame i+1 BEFORE next synchronous tick.
            #
            # This must happen for BOTH:
            #
            #     CARLA condition
            #     HE condition
            #
            # because the traffic light is part of the common
            # physical/environment scenario.
            # ------------------------------------------------

            if (
                traffic_light_executor
                is not None
            ):

                traffic_light_executor.apply(
                    float(
                        actor_frames[
                            i + 1
                        ][
                            "t_s"
                        ]
                    )
                )

            # ------------------------------------------------
            # Advance physical CARLA adversary trajectory
            # BEFORE next synchronous tick.
            #
            # Only the CARLA condition has a physical
            # adversary actor.
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

            current_depth_image = None
            current_depth_m = None

            if depth_camera_queue is not None:

                current_depth_image = (
                    get_named_sensor_frame(
                        depth_camera_queue,
                        current_frame,
                        "Depth camera",
                    )
                )

                current_depth_m = (
                    carla_depth_image_to_m(
                        current_depth_image
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
            current_instance_image = None

            if instance_camera_queue is not None:

                current_instance_image = (
                    get_named_sensor_frame(
                        instance_camera_queue,
                        current_frame,
                        "instance camera",
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
            route_projector
            is not None
        ):

            print(
                "minimum route bumper gap:",
                f"{min_route_gap:.3f} m",
            )

            print(
                "route virtual collision:",
                any_route_virtual_collision,
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
        if (
            traffic_light_executor
            is not None
        ):

            try:
                traffic_light_executor.restore()

            except Exception as exc:
                print(
                    "[cleanup warning] "
                    "traffic-light restore:",
                    exc,
                )
        print(
            "[cleanup]"
        )

        if csv_file is not None:
            csv_file.close()
        if native_equiv_jsonl_fp is not None:

            native_equiv_jsonl_fp.close()
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

        if depth_camera is not None:
            try:
                depth_camera.stop()
            except Exception:
                pass

            depth_camera.destroy()

        if instance_camera is not None:

            try:
                instance_camera.stop()
            except Exception:
                pass

            instance_camera.destroy()
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