"""
he_camera_renderer.py

Model-independent Hallucination Engine camera renderer.

Purpose
-------
Project and render the same virtual actor into arbitrary CARLA
RGB camera viewpoints.

Unlike the original TCP experiment, camera geometry is explicit:

    width
    height
    FOV
    camera world transform

This allows the same scenario actor to be rendered consistently into:

    TCP:
        one camera

    NEAT:
        front
        left
        right

without modifying the frozen TCP renderer.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import carla


# ============================================================
# HE runtime modules
# ============================================================

THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]

HE_RUNTIME_DIR = (
    HE_ROOT
    / "HE_v_0.1"
)

if str(HE_RUNTIME_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(HE_RUNTIME_DIR),
    )


from run_he_temporal_compositor_v2 import (
    load_view_matrix_sprite_bank,
    select_view_matrix_sprite,
    warp_view_matrix_sprite_to_box_subpixel,
    alpha_composite_rgb,
)

# ============================================================
# HE runtime
# ============================================================

THIS_FILE = Path(__file__).resolve()

HE_ROOT = (
    THIS_FILE
    .parents[2]
)

HE_RUNTIME_DIR = (
    HE_ROOT
    / "HE_v_0.1"
)

if str(
    HE_RUNTIME_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            HE_RUNTIME_DIR
        ),
    )


from run_he_temporal_compositor_v1 import (
    SpriteCache,
    compute_viewpoint_sprite_angle,
    discover_available_sprite_angles,
    select_sprite,
)


from he_projection_geometry_v1 import (
    project_virtual_actor_center_depth_billboard,
)

# ============================================================
# Angle
# ============================================================

def normalize_angle_180(
    angle_deg,
):

    return (
        float(
            angle_deg
        )
        + 180.0
    ) % 360.0 - 180.0


# ============================================================
# Intrinsics
# ============================================================

def make_camera_intrinsic(
    width,
    height,
    fov_deg,
):

    focal = (
        float(
            width
        )
        /
        (
            2.0
            *
            math.tan(
                math.radians(
                    float(
                        fov_deg
                    )
                )
                / 2.0
            )
        )
    )

    k = np.identity(
        3,
        dtype=np.float64,
    )

    k[
        0,
        0,
    ] = focal

    k[
        1,
        1,
    ] = focal

    k[
        0,
        2,
    ] = (
        float(
            width
        )
        / 2.0
    )

    k[
        1,
        2,
    ] = (
        float(
            height
        )
        / 2.0
    )

    return k


def _transform_rotation_matrix(transform):
    """Return the CARLA local-to-parent 3-D rotation matrix."""
    return np.asarray(transform.get_matrix(), dtype=np.float64)[:3, :3]


def _transform_homogeneous_point(matrix, x, y):
    point = np.asarray([float(x), float(y), 1.0], dtype=np.float64)
    projected = np.asarray(matrix, dtype=np.float64) @ point
    if abs(float(projected[2])) <= 1e-8:
        return None
    return projected[:2] / projected[2]


def warp_view_matrix_sprite_camera_rotation(
    sprite_rgba,
    actor_tf,
    camera_tf,
    physical_bbox,
    sprite_info,
    capture_metadata,
    target_support_anchor,
    width,
    height,
    fov,
    silhouette_scale=1.0,
    alpha_threshold=10,
):
    """Reproject a centered far-bank capture into an arbitrary camera.

    The 4320 bank records a camera that looks directly at the actor. Runtime
    side cameras generally do not. A pure camera-rotation homography corrects
    that optical-axis difference without inventing a per-camera angle offset.
    Translation and elevation differences are approximated by the unique 2-D
    similarity that maps the same physical bbox support point and bbox center
    from the source capture into the runtime camera.
    """
    required_sprite = (
        "selected_angle",
        "selected_elevation_deg",
        "crop_x1_px",
        "crop_y1_px",
        "source_ground_anchor_x",
        "source_ground_anchor_y",
    )
    if any(sprite_info.get(name) is None for name in required_sprite):
        return None
    if target_support_anchor is None or not physical_bbox:
        return None

    source_width = float(capture_metadata.get("image_width_px", 0.0))
    source_height = float(capture_metadata.get("image_height_px", 0.0))
    source_fov = float(capture_metadata.get("fov_deg", 0.0))
    if source_width <= 0.0 or source_height <= 0.0 or source_fov <= 0.0:
        return None

    target_center = project_asset_physical_bbox_center(
        actor_tf=actor_tf,
        camera_tf=camera_tf,
        physical_bbox=physical_bbox,
        width=width,
        height=height,
        fov=fov,
    )
    if target_center is None:
        return None
    source_actor_tf = carla.Transform()
    source_camera_tf = carla.Transform(
        rotation=carla.Rotation(
            pitch=-float(sprite_info["selected_elevation_deg"]),
            yaw=float(sprite_info["selected_angle"]) - 180.0,
        )
    )
    source_camera_to_actor = (
        _transform_rotation_matrix(source_actor_tf).T
        @ _transform_rotation_matrix(source_camera_tf)
    )
    actor_to_target_camera = (
        _transform_rotation_matrix(camera_tf).T
        @ _transform_rotation_matrix(actor_tf)
    )
    ray_rotation = actor_to_target_camera @ source_camera_to_actor

    # CARLA camera rays are [forward, right, up], while image homogeneous
    # coordinates are [right, -up, forward].
    camera_to_image_axes = np.asarray([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)
    image_to_camera_axes = np.linalg.inv(camera_to_image_axes)
    source_k = make_camera_intrinsic(source_width, source_height, source_fov)
    target_k = make_camera_intrinsic(width, height, fov)
    crop_to_full = np.asarray([
        [1.0, 0.0, float(sprite_info["crop_x1_px"])],
        [0.0, 1.0, float(sprite_info["crop_y1_px"])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    rotation_h = (
        target_k
        @ camera_to_image_axes
        @ ray_rotation
        @ image_to_camera_axes
        @ np.linalg.inv(source_k)
        @ crop_to_full
    )

    source_support = _transform_homogeneous_point(
        rotation_h,
        sprite_info["source_ground_anchor_x"],
        sprite_info["source_ground_anchor_y"],
    )
    if source_support is None:
        return None
    source_center = _transform_homogeneous_point(
        rotation_h,
        float(capture_metadata["camera_cx_px"])
        - float(sprite_info["crop_x1_px"]),
        float(capture_metadata["camera_cy_px"])
        - float(sprite_info["crop_y1_px"]),
    )
    if source_center is None:
        return None

    source_vector = source_center - source_support
    target_support = np.asarray([
        float(target_support_anchor["u"]),
        float(target_support_anchor["v"]),
    ], dtype=np.float64)
    target_center_xy = np.asarray([
        float(target_center["u"]),
        float(target_center["v"]),
    ], dtype=np.float64)
    target_vector = target_center_xy - target_support
    denominator = float(np.dot(source_vector, source_vector))
    if denominator <= 1e-8:
        return None

    # The complex ratio target_vector/source_vector gives the unique 2-D
    # similarity that maps both the physical support point and bbox center.
    # Unlike radial scaling alone, this also accounts for a source elevation
    # sample that is near, but not identical to, the runtime camera elevation.
    similarity_a = float(np.dot(target_vector, source_vector) / denominator)
    similarity_b = float(
        (
            target_vector[1] * source_vector[0]
            - target_vector[0] * source_vector[1]
        ) / denominator
    )
    similarity_a *= float(silhouette_scale)
    similarity_b *= float(silhouette_scale)
    similarity = np.asarray([
        [similarity_a, -similarity_b],
        [similarity_b, similarity_a],
    ], dtype=np.float64)
    translation = target_support - similarity @ source_support
    anchor_similarity = np.asarray([
        [similarity[0, 0], similarity[0, 1], translation[0]],
        [similarity[1, 0], similarity[1, 1], translation[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    final_h = anchor_similarity @ rotation_h
    warped = cv2.warpPerspective(
        sprite_rgba,
        final_h,
        (int(width), int(height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha_y, alpha_x = np.where(warped[:, :, 3] > int(alpha_threshold))
    if len(alpha_x) == 0 or len(alpha_y) == 0:
        return None
    visible_bbox = {
        "x1": int(alpha_x.min()),
        "y1": int(alpha_y.min()),
        "x2": int(alpha_x.max()),
        "y2": int(alpha_y.max()),
    }
    resize_info = {
        "mode": "camera_rotation_homography",
        "fully_outside_frame": False,
        "paste": {
            "x1": 0,
            "y1": 0,
            "sprite_width": int(width),
            "sprite_height": int(height),
        },
        "float_x1": 0.0,
        "float_y1": 0.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "visible_bbox": visible_bbox,
    }
    return warped, resize_info, {
        "homography": final_h.tolist(),
        "similarity_scale": float(math.hypot(similarity_a, similarity_b)),
        "similarity_rotation_deg": float(
            math.degrees(math.atan2(similarity_b, similarity_a))
        ),
        "source_support_after_rotation_px": {
            "x": float(source_support[0]),
            "y": float(source_support[1]),
        },
        "source_center_after_rotation_px": {
            "x": float(source_center[0]),
            "y": float(source_center[1]),
        },
    }


def _linear_interpolation_bracket(values, query):
    ordered = sorted(float(value) for value in values)
    query = float(query)
    if query <= ordered[0]:
        return ordered[0], ordered[0], 0.0
    if query >= ordered[-1]:
        return ordered[-1], ordered[-1], 0.0
    for lower, upper in zip(ordered, ordered[1:]):
        if lower <= query <= upper:
            weight = (query - lower) / (upper - lower)
            return lower, upper, weight
    return ordered[-1], ordered[-1], 0.0


def _circular_interpolation_bracket(values, query):
    ordered = sorted(float(value) % 360.0 for value in values)
    query = float(query) % 360.0
    extended = ordered + [ordered[0] + 360.0]
    adjusted = query if query >= ordered[0] else query + 360.0
    for lower, upper in zip(extended, extended[1:]):
        if lower <= adjusted <= upper:
            weight = 0.0 if upper == lower else (adjusted - lower) / (upper - lower)
            return lower % 360.0, upper % 360.0, weight
    return ordered[0], ordered[0], 0.0


def _view_matrix_record_info(base_info, record, angle, distance, elevation):
    info = dict(base_info)
    info.update({
        "selected_angle": int(round(float(angle))) % 360,
        "selected_distance_m": float(distance),
        "selected_elevation_deg": float(elevation),
        "sprite_path": str(record["rgba_path"]),
        "crop_x1_px": record.get("crop_x1_px"),
        "crop_y1_px": record.get("crop_y1_px"),
        "source_ground_anchor_x": record.get("source_ground_anchor_x"),
        "source_ground_anchor_y": record.get("source_ground_anchor_y"),
        "bbox_center_distance_m": record.get("bbox_center_distance_m"),
    })
    return info


def continuous_view_matrix_samples(sprite_info, view_matrix):
    """Return trilinear angle/distance/elevation samples from a 4320 bank."""
    angle_lower, angle_upper, angle_weight = _circular_interpolation_bracket(
        view_matrix["angles"], sprite_info["relative_angle_deg"]
    )
    distance_lower, distance_upper, distance_weight = _linear_interpolation_bracket(
        view_matrix["distances"], sprite_info["query_distance_m"]
    )
    elevation_lower, elevation_upper, elevation_weight = _linear_interpolation_bracket(
        view_matrix["elevations"], sprite_info["query_elevation_deg"]
    )
    weighted = []
    for angle in {angle_lower, angle_upper}:
        aw = 1.0 if angle_lower == angle_upper else (
            1.0 - angle_weight if angle == angle_lower else angle_weight
        )
        for distance in {distance_lower, distance_upper}:
            dw = 1.0 if distance_lower == distance_upper else (
                1.0 - distance_weight if distance == distance_lower else distance_weight
            )
            for elevation in {elevation_lower, elevation_upper}:
                ew = 1.0 if elevation_lower == elevation_upper else (
                    1.0 - elevation_weight
                    if elevation == elevation_lower
                    else elevation_weight
                )
                key = (
                    int(round(float(angle))) % 360,
                    float(distance),
                    float(elevation),
                )
                record = view_matrix["index"].get(key)
                weight = aw * dw * ew
                if record is not None and weight > 0.0:
                    weighted.append((
                        weight,
                        _view_matrix_record_info(
                            sprite_info, record, angle, distance, elevation
                        ),
                    ))
    total = sum(weight for weight, unused in weighted)
    if total <= 0.0:
        return []
    return [(weight / total, info) for weight, info in weighted]


def warp_continuous_view_matrix_camera_rotation(
    sprite_info,
    view_matrix,
    sprite_cache,
    actor_tf,
    camera_tf,
    physical_bbox,
    target_support_anchor,
    width,
    height,
    fov,
    silhouette_scale=1.0,
    alpha_threshold=10,
):
    samples = continuous_view_matrix_samples(sprite_info, view_matrix)
    if not samples:
        return None
    rendered_samples = []
    for weight, sample_info in samples:
        sprite = sprite_cache.load_rgba(sample_info["sprite_path"])
        result = warp_view_matrix_sprite_camera_rotation(
            sprite_rgba=sprite,
            actor_tf=actor_tf,
            camera_tf=camera_tf,
            physical_bbox=physical_bbox,
            sprite_info=sample_info,
            capture_metadata=view_matrix.get("capture_metadata") or {},
            target_support_anchor=target_support_anchor,
            width=width,
            height=height,
            fov=fov,
            silhouette_scale=silhouette_scale,
            alpha_threshold=alpha_threshold,
        )
        if result is None:
            continue
        warped, unused_resize, sample_meta = result
        alpha_y, alpha_x = np.where(warped[:, :, 3] > int(alpha_threshold))
        if len(alpha_x) == 0 or len(alpha_y) == 0:
            continue
        rendered_samples.append({
            "weight": float(weight),
            "angle_deg": int(sample_info["selected_angle"]),
            "distance_m": float(sample_info["selected_distance_m"]),
            "elevation_deg": float(sample_info["selected_elevation_deg"]),
            "similarity_scale": sample_meta["similarity_scale"],
            "bbox": {
                "x1": float(alpha_x.min()),
                "y1": float(alpha_y.min()),
                "x2": float(alpha_x.max()),
                "y2": float(alpha_y.max()),
            },
            "warped": warped,
        })
    if not rendered_samples:
        return None

    total_weight = sum(sample["weight"] for sample in rendered_samples)
    target = {
        name: sum(
            sample["weight"] * sample["bbox"][name]
            for sample in rendered_samples
        ) / total_weight
        for name in ("x1", "y1", "x2", "y2")
    }
    selected_key = (
        int(sprite_info["selected_angle"]),
        float(sprite_info["selected_distance_m"]),
        float(sprite_info["selected_elevation_deg"]),
    )
    preferred = max(rendered_samples, key=lambda sample: sample["weight"])
    for sample in rendered_samples:
        sample_key = (
            sample["angle_deg"],
            sample["distance_m"],
            sample["elevation_deg"],
        )
        if sample_key == selected_key:
            preferred = sample
            break

    source_bbox = preferred["bbox"]
    warped, resize_info = warp_view_matrix_sprite_to_box_subpixel(
        sprite_rgba=preferred["warped"],
        frame_w=int(width),
        frame_h=int(height),
        target_cx=(target["x1"] + target["x2"]) / 2.0,
        target_bottom_y=target["y2"],
        target_box_w=max(1.0, target["x2"] - target["x1"] + 1.0),
        target_box_h=max(1.0, target["y2"] - target["y1"] + 1.0),
        anchor_x=(source_bbox["x1"] + source_bbox["x2"]) / 2.0,
        anchor_y=source_bbox["y2"],
        alpha_threshold=alpha_threshold,
        scale_mode="independent",
    )
    public_samples = [{
        key: value for key, value in sample.items()
        if key not in {"warped", "bbox"}
    } | {"bbox": sample["bbox"]} for sample in rendered_samples]
    return warped, resize_info, {
        "interpolation": "trilinear_geometry_nearest_appearance",
        "sample_count": len(rendered_samples),
        "selected_appearance": {
            "angle_deg": preferred["angle_deg"],
            "distance_m": preferred["distance_m"],
            "elevation_deg": preferred["elevation_deg"],
        },
        "target_bbox": target,
        "samples": public_samples,
    }


# ============================================================
# World -> image
# ============================================================

def project_world_point(
    point,
    world_to_camera,
    k,
):
    """
    CARLA camera coordinates:

        X = forward
        Y = right
        Z = up

    Standard image projection vector:

        [Y, -Z, X]
    """

    p = np.array(
        [
            float(
                point.x
            ),
            float(
                point.y
            ),
            float(
                point.z
            ),
            1.0,
        ],
        dtype=np.float64,
    )

    p_camera = (
        world_to_camera
        @ p
    )

    depth = float(
        p_camera[
            0
        ]
    )

    if depth <= 0.05:

        return None

    image_vector = (
        k
        @ np.array(
            [
                p_camera[
                    1
                ],

                -p_camera[
                    2
                ],

                p_camera[
                    0
                ],
            ],
            dtype=np.float64,
        )
    )

    u = float(
        image_vector[
            0
        ]
        /
        image_vector[
            2
        ]
    )

    v = float(
        image_vector[
            1
        ]
        /
        image_vector[
            2
        ]
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
                p_camera[
                    1
                ]
            ),

        "camera_up_m":
            float(
                p_camera[
                    2
                ]
            ),
    }


def project_asset_physical_bbox_center(
    actor_tf,
    camera_tf,
    physical_bbox,
    width,
    height,
    fov,
):
    if not physical_bbox:
        return None
    required = (
        "local_center_x_m",
        "local_center_y_m",
        "local_center_z_m",
    )
    if any(physical_bbox.get(key) is None for key in required):
        return None

    actor_to_world = np.asarray(actor_tf.get_matrix(), dtype=np.float64)
    center_world = actor_to_world @ np.asarray([
        float(physical_bbox["local_center_x_m"]),
        float(physical_bbox["local_center_y_m"]),
        float(physical_bbox["local_center_z_m"]),
        1.0,
    ], dtype=np.float64)
    return project_world_point(
        point=carla.Location(
            x=float(center_world[0]),
            y=float(center_world[1]),
            z=float(center_world[2]),
        ),
        world_to_camera=np.asarray(
            camera_tf.get_inverse_matrix(), dtype=np.float64
        ),
        k=make_camera_intrinsic(width, height, fov),
    )

def project_asset_physical_support_anchor(
    actor_tf,
    camera_tf,
    physical_bbox,
    width,
    height,
    fov,
):
    """
    Project the SAME physical bbox support-center definition used by
    the production asset generator.

    Production definition:
        1. construct all 8 physical CARLA bbox vertices,
        2. transform them into world coordinates,
        3. select the four vertices with the lowest world Z,
        4. average those four vertices,
        5. project that support center into the camera.

    This makes source and runtime anchor semantics identical.
    """

    if not physical_bbox:
        return None

    required = [
        "extent_x_m",
        "extent_y_m",
        "extent_z_m",
        "local_center_x_m",
        "local_center_y_m",
        "local_center_z_m",
    ]

    for key in required:
        if physical_bbox.get(key) is None:
            return None

    extent_x = float(
        physical_bbox["extent_x_m"]
    )

    extent_y = float(
        physical_bbox["extent_y_m"]
    )

    extent_z = float(
        physical_bbox["extent_z_m"]
    )

    bbox_tf = carla.Transform(
        carla.Location(
            x=float(
                physical_bbox[
                    "local_center_x_m"
                ]
            ),
            y=float(
                physical_bbox[
                    "local_center_y_m"
                ]
            ),
            z=float(
                physical_bbox[
                    "local_center_z_m"
                ]
            ),
        ),
        carla.Rotation(
            pitch=float(
                physical_bbox.get(
                    "local_rotation_pitch_deg",
                    0.0,
                )
            ),
            yaw=float(
                physical_bbox.get(
                    "local_rotation_yaw_deg",
                    0.0,
                )
            ),
            roll=float(
                physical_bbox.get(
                    "local_rotation_roll_deg",
                    0.0,
                )
            ),
        ),
    )

    actor_to_world = np.asarray(
        actor_tf.get_matrix(),
        dtype=np.float64,
    )

    bbox_to_actor = np.asarray(
        bbox_tf.get_matrix(),
        dtype=np.float64,
    )

    bbox_to_world = (
        actor_to_world
        @
        bbox_to_actor
    )

    world_vertices = []

    for local_x in (
        -extent_x,
        +extent_x,
    ):
        for local_y in (
            -extent_y,
            +extent_y,
        ):
            for local_z in (
                -extent_z,
                +extent_z,
            ):

                local_h = np.array(
                    [
                        float(local_x),
                        float(local_y),
                        float(local_z),
                        1.0,
                    ],
                    dtype=np.float64,
                )

                world_h = (
                    bbox_to_world
                    @
                    local_h
                )

                world_vertices.append(
                    np.asarray(
                        world_h[:3],
                        dtype=np.float64,
                    )
                )

    vertices_xyz = np.asarray(
        world_vertices,
        dtype=np.float64,
    )

    bottom_indices = np.argsort(
        vertices_xyz[:, 2]
    )[:4]

    support_xyz = np.mean(
        vertices_xyz[
            bottom_indices
        ],
        axis=0,
    )

    support_world = carla.Location(
        x=float(
            support_xyz[0]
        ),
        y=float(
            support_xyz[1]
        ),
        z=float(
            support_xyz[2]
        ),
    )

    k = make_camera_intrinsic(
        width=width,
        height=height,
        fov_deg=fov,
    )

    world_to_camera = np.asarray(
        camera_tf.get_inverse_matrix(),
        dtype=np.float64,
    )

    projected = project_world_point(
        point=support_world,
        world_to_camera=world_to_camera,
        k=k,
    )

    if projected is None:
        return None

    return {
        "u":
            float(
                projected["u"]
            ),

        "v":
            float(
                projected["v"]
            ),

        "depth_m":
            float(
                projected["depth"]
            ),

        "world_x_m":
            float(
                support_xyz[0]
            ),

        "world_y_m":
            float(
                support_xyz[1]
            ),

        "world_z_m":
            float(
                support_xyz[2]
            ),
    }
# ============================================================
# Actor projection
# ============================================================

def project_virtual_actor(
    actor_tf,
    camera_tf,
    dimensions,
    width,
    height,
    fov,
):
    """
    Project a metric actor for 2-D sprite rendering.

    Geometry model:
        - actor footprint is an oriented metric rectangle
        - horizontal extent comes from the projected footprint
        - vertical extent is evaluated at the nearest footprint
          support point to the camera

    This avoids fitting a flat sprite to the full projected
    envelope of a 3-D cuboid, which exaggerates height at
    close range.
    """

    actor_length = float(
        dimensions["length_m"]
    )

    actor_width = float(
        dimensions["width_m"]
    )

    actor_height = float(
        dimensions["height_m"]
    )

    if (
        actor_length <= 0.0
        or
        actor_width <= 0.0
        or
        actor_height <= 0.0
    ):
        return {
            "visible": False,
            "reason": "invalid_dimensions",
        }

    k = make_camera_intrinsic(
        width,
        height,
        fov,
    )

    world_to_camera = np.array(
        camera_tf.get_inverse_matrix(),
        dtype=np.float64,
    )

    actor_to_world = np.array(
        actor_tf.get_matrix(),
        dtype=np.float64,
    )

    world_to_actor = np.array(
        actor_tf.get_inverse_matrix(),
        dtype=np.float64,
    )

    half_length = (
        actor_length
        / 2.0
    )

    half_width = (
        actor_width
        / 2.0
    )

    # ========================================================
    # Actor-center projection
    # ========================================================

    actor_center = carla.Location(
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

    center_projection = (
        project_world_point(
            actor_center,
            world_to_camera,
            k,
        )
    )

    if center_projection is None:

        return {
            "visible": False,
            "reason": "actor_center_behind_camera",
        }

    depth = float(
        center_projection["depth"]
    )

    if depth <= 1.0:

        return {
            "visible": False,
            "reason": "too_close",
            "depth_m": depth,
        }

    # ========================================================
    # Camera position in ACTOR-LOCAL coordinates
    #
    # CARLA local axes:
    #   +x = forward
    #   +y = right
    #   +z = up
    # ========================================================

    camera_world_h = np.array(
        [
            float(
                camera_tf.location.x
            ),
            float(
                camera_tf.location.y
            ),
            float(
                camera_tf.location.z
            ),
            1.0,
        ],
        dtype=np.float64,
    )

    camera_local_h = (
        world_to_actor
        @
        camera_world_h
    )

    camera_local_x = float(
        camera_local_h[0]
    )

    camera_local_y = float(
        camera_local_h[1]
    )

    # ========================================================
    # Nearest footprint point to camera
    #
    # Clamp camera-local XY onto the oriented actor rectangle.
    #
    # Straight rear view:
    #     support_x = -L/2
    #     support_y ~= 0
    #
    # Straight front view:
    #     support_x = +L/2
    #
    # Side / oblique view is handled naturally.
    # ========================================================

    support_local_x = float(
        np.clip(
            camera_local_x,
            -half_length,
            +half_length,
        )
    )

    support_local_y = float(
        np.clip(
            camera_local_y,
            -half_width,
            +half_width,
        )
    )

    def local_to_world_location(
        local_x,
        local_y,
        local_z,
    ):

        local_h = np.array(
            [
                float(
                    local_x
                ),
                float(
                    local_y
                ),
                float(
                    local_z
                ),
                1.0,
            ],
            dtype=np.float64,
        )

        world_h = (
            actor_to_world
            @
            local_h
        )

        return carla.Location(
            x=float(
                world_h[0]
            ),
            y=float(
                world_h[1]
            ),
            z=float(
                world_h[2]
            ),
        )

    # ========================================================
    # Vertical support line
    #
    # Bottom and top now use the SAME XY and therefore the
    # same physical surface/depth.
    # ========================================================

    support_bottom = (
        local_to_world_location(
            support_local_x,
            support_local_y,
            0.0,
        )
    )

    support_top = (
        local_to_world_location(
            support_local_x,
            support_local_y,
            actor_height,
        )
    )

    p_support_bottom = (
        project_world_point(
            support_bottom,
            world_to_camera,
            k,
        )
    )

    p_support_top = (
        project_world_point(
            support_top,
            world_to_camera,
            k,
        )
    )

    if (
        p_support_bottom is None
        or
        p_support_top is None
    ):

        return {
            "visible": False,
            "reason": "support_plane_behind_camera",
        }

    support_depth = float(
        p_support_bottom[
            "depth"
        ]
    )

    if support_depth <= 1.0:

        return {
            "visible": False,
            "reason": "support_plane_too_close",
            "depth_m": depth,
            "support_depth_m": support_depth,
        }

    bottom_y = float(
        p_support_bottom[
            "v"
        ]
    )

    top_y = float(
        p_support_top[
            "v"
        ]
    )

    box_height = abs(
        bottom_y
        -
        top_y
    )

    # ========================================================
    # Horizontal projected footprint
    #
    # Project the four footprint corners at half vehicle
    # height. This captures perspective from length + width
    # without using the full cuboid's vertical envelope.
    # ========================================================

    footprint_projected = []

    footprint_depths = []

    reference_z = (
        actor_height
        * 0.5
    )

    for local_x in (
        -half_length,
        +half_length,
    ):

        for local_y in (
            -half_width,
            +half_width,
        ):

            world_point = (
                local_to_world_location(
                    local_x,
                    local_y,
                    reference_z,
                )
            )

            p = project_world_point(
                world_point,
                world_to_camera,
                k,
            )

            if p is not None:

                footprint_projected.append(
                    p
                )

                footprint_depths.append(
                    float(
                        p[
                            "depth"
                        ]
                    )
                )

    if len(
        footprint_projected
    ) != 4:

        return {
            "visible": False,
            "reason": "footprint_intersects_camera_plane",
            "depth_m": depth,
        }

    projected_u = np.array(
        [
            float(
                p["u"]
            )
            for p
            in footprint_projected
        ],
        dtype=np.float64,
    )

    x1 = float(
        np.min(
            projected_u
        )
    )

    x2 = float(
        np.max(
            projected_u
        )
    )

    box_width = float(
        x2
        -
        x1
    )

    cx = float(
        (
            x1
            +
            x2
        )
        / 2.0
    )

    y1 = float(
        bottom_y
        -
        box_height
    )

    y2 = float(
        bottom_y
    )

    nearest_depth = float(
        min(
            footprint_depths
        )
    )

    farthest_depth = float(
        max(
            footprint_depths
        )
    )

    # ========================================================
    # Visibility
    # ========================================================

    margin = 30.0

    visible = not (
        x2 < -margin
        or
        x1
        >
        float(
            width
        )
        + margin
        or
        y2 < -margin
        or
        y1
        >
        float(
            height
        )
        + margin
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
            bool(
                visible
            ),

        "cx":
            cx,

        "bottom_y":
            bottom_y,

        # Actor transform/reference origin projected directly
        # into the camera. Unlike cx/bottom_y above, these do
        # not depend on actor length/width/height.
        "actor_reference_cx_px":
            float(
                center_projection[
                    "u"
                ]
            ),

        "actor_reference_bottom_y_px":
            float(
                center_projection[
                    "v"
                ]
            ),

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

        # Actor-center depth retained for viewpoint selection.
        "depth_m":
            depth,

        "camera_right_m":
            float(
                center_projection[
                    "camera_right_m"
                ]
            ),
        "camera_up_m":
            float(
                center_projection[
                    "camera_up_m"
                ]
            ),
        "actor_relative_yaw_deg":
            actor_relative_yaw,

        # New geometry diagnostics.
        "support_depth_m":
            support_depth,

        "nearest_depth_m":
            nearest_depth,

        "farthest_depth_m":
            farthest_depth,

        "support_local_x_m":
            support_local_x,

        "support_local_y_m":
            support_local_y,

        "geometry_mode":
            "oriented_2p5d_support",
    }
# ============================================================
# Transparent sprite crop
# ============================================================

def trim_sprite_to_visible_alpha(
    sprite_rgba,
    alpha_threshold=2,
    padding_px=2,
):

    if (
        sprite_rgba.ndim
        != 3
        or
        sprite_rgba.shape[
            2
        ]
        != 4
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

    if (
        len(
            xs
        )
        == 0
        or
        len(
            ys
        )
        == 0
    ):

        return sprite_rgba

    h, w = (
        sprite_rgba.shape[
            :2
        ]
    )

    x1 = max(
        0,
        int(
            xs.min()
        )
        -
        int(
            padding_px
        ),
    )

    x2 = min(
        w,
        int(
            xs.max()
        )
        + 1
        +
        int(
            padding_px
        ),
    )

    y1 = max(
        0,
        int(
            ys.min()
        )
        -
        int(
            padding_px
        ),
    )

    y2 = min(
        h,
        int(
            ys.max()
        )
        + 1
        +
        int(
            padding_px
        ),
    )

    return (
        sprite_rgba[
            y1:y2,
            x1:x2,
        ]
        .copy()
    )


# ============================================================
# Sub-pixel compositor
# ============================================================

def alpha_composite_sprite_subpixel(
    frame_rgb,
    sprite_rgba,
    center_x,
    bottom_y,
    target_box_height,
    target_box_width=None,
    global_alpha=1.0,
):
    """
    Premultiplied-alpha subpixel sprite compositing.

    Supports independent horizontal and vertical scale.

    If target_box_width is None, behavior is backward-compatible
    with the previous isotropic height-driven scaling.
    """

    frame_h, frame_w = (
        frame_rgb.shape[
            :2
        ]
    )

    src_h, src_w = (
        sprite_rgba.shape[
            :2
        ]
    )

    if (
        src_h < 2
        or
        src_w < 2
    ):
        return (
            frame_rgb,
            {
                "rendered":
                    False,

                "reason":
                    "sprite_too_small",
            },
        )

    source_height_span = float(
        src_h - 1
    )

    source_width_span = float(
        src_w - 1
    )

    scale_y = (
        float(
            target_box_height
        )
        /
        source_height_span
    )

    if target_box_width is None:

        # Old behavior.
        scale_x = (
            scale_y
        )

    else:

        scale_x = (
            float(
                target_box_width
            )
            /
            source_width_span
        )

    if (
        scale_x <= 0.0
        or
        scale_y <= 0.0
    ):

        return (
            frame_rgb,
            {
                "rendered":
                    False,

                "reason":
                    "invalid_target_scale",
            },
        )

    src_anchor_x = (
        source_width_span
        / 2.0
    )

    src_anchor_y = (
        source_height_span
    )

    tx = (
        float(
            center_x
        )
        -
        scale_x
        *
        src_anchor_x
    )

    ty = (
        float(
            bottom_y
        )
        -
        scale_y
        *
        src_anchor_y
    )

    affine = np.array(
        [
            [
                scale_x,
                0.0,
                tx,
            ],
            [
                0.0,
                scale_y,
                ty,
            ],
        ],
        dtype=np.float32,
    )

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

    # Premultiplied RGB avoids dark/black interpolation halos.
    premult_rgb = (
        sprite[
            :,
            :,
            :3
        ]
        *
        alpha
    )

    premult_rgba = (
        np.concatenate(
            [
                premult_rgb,
                alpha,
            ],
            axis=2,
        )
    )

    warped = cv2.warpAffine(
        premult_rgba,
        affine,
        (
            frame_w,
            frame_h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=(
            cv2.BORDER_CONSTANT
        ),
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
        *
        float(
            global_alpha
        ),
        0.0,
        1.0,
    )

    warped_rgb = (
        warped_rgb
        *
        float(
            global_alpha
        )
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
        *
        (
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

    return (
        output,
        {
            "rendered":
                True,

            # Keep old metadata name for existing callers.
            # It now means vertical scale.
            "scale":
                float(
                    scale_y
                ),

            "scale_x":
                float(
                    scale_x
                ),

            "scale_y":
                float(
                    scale_y
                ),

            "target_width":
                float(
                    source_width_span
                    *
                    scale_x
                ),

            "target_height":
                float(
                    source_height_span
                    *
                    scale_y
                ),
        },
    )
# ============================================================
# Generic actor renderer
# ============================================================

def render_he_actor(
    base_rgb,
    actor_tf,
    camera_tf,
    dimensions,
    sprite_bank,
    available_angles,
    sprite_cache,
    width,
    height,
    fov,
    scene_depth_m=None,
    scene_occlusion_margin_m=0.25,
):

    frame = (
        base_rgb.copy()
    )

    box = project_virtual_actor(
        actor_tf=
            actor_tf,

        camera_tf=
            camera_tf,

        dimensions=
            dimensions,

        width=
            width,

        height=
            height,

        fov=
            fov,
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

        "viewpoint_angle_deg":
            None,
    }

    if not box.get(
        "visible",
        False,
    ):

        return (
            frame,
            meta,
        )

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

    sprite_info = select_sprite(
        sprite_bank=
            sprite_bank,

        relative_angle_deg=
            viewpoint_angle,

        available_angles=
            available_angles,
    )

    if not sprite_info[
        "exists"
    ]:

        return (
            frame,
            meta,
        )

    sprite_rgba = (
        sprite_cache.load_rgba(
            sprite_info[
                "sprite_path"
            ]
        )
    )

    sprite_rgba = (
        trim_sprite_to_visible_alpha(
            sprite_rgba
        )
    )

    frame, render_meta = (
        alpha_composite_sprite_subpixel(
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

            target_box_width=
                box[
                    "box_width"
                ],
        )
    )

    if not render_meta.get(
        "rendered",
        False,
    ):

        return (
            frame,
            meta,
        )

    meta.update({
        "rendered":
            True,

        "viewpoint_angle_deg":
            float(
                viewpoint_angle
            ),

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

        "sprite_width":
            float(
                render_meta[
                    "target_width"
                ]
            ),

        "sprite_height":
            float(
                render_meta[
                    "target_height"
                ]
            ),

        "subpixel_scale":
            float(
                render_meta[
                    "scale"
                ]
            ),
        "subpixel_scale_x":
            float(
                render_meta[
                    "scale_x"
                ]
            ),

        "subpixel_scale_y":
            float(
                render_meta[
                    "scale_y"
                ]
            ),
    })

    return (
        frame,
        meta,
    )

# ============================================================
# Scene-depth occlusion
# ============================================================

def apply_scene_depth_occlusion_to_sprite(
    sprite_rgba,
    scene_depth_m,
    paste_x1,
    paste_y1,
    actor_nearest_depth_m,
    margin_m=0.25,
):
    """
    Mask HE pixels that are hidden by closer CARLA/world geometry.

    Parameters
    ----------
    sprite_rgba:
        Local warped HE RGBA crop.

    scene_depth_m:
        Full-frame metric depth image aligned exactly with the
        model's native RGB camera.

    paste_x1, paste_y1:
        Top-left location where sprite_rgba will be composited
        into the native RGB frame.

    actor_nearest_depth_m:
        Conservative nearest physical depth of the HE actor.

    margin_m:
        Depth tolerance. Scene geometry must be at least this
        much closer than the HE actor before it occludes HE.

        Positive margin helps prevent road/contact-surface noise
        from incorrectly cutting away the bottom of the sprite.

    Returns
    -------
    masked_rgba, metadata
    """

    meta = {
        "enabled":
            False,

        "actor_nearest_depth_m":
            (
                float(
                    actor_nearest_depth_m
                )
                if actor_nearest_depth_m is not None
                else None
            ),

        "margin_m":
            float(
                margin_m
            ),

        "occlusion_threshold_m":
            None,

        "sprite_alpha_pixels_in_frame":
            0,

        "occluded_alpha_pixels":
            0,

        "remaining_alpha_pixels":
            0,

        "occluded_fraction":
            0.0,
    }

    if scene_depth_m is None:

        meta[
            "reason"
        ] = "scene_depth_not_provided"

        return (
            sprite_rgba,
            meta,
        )

    actor_depth = float(
        actor_nearest_depth_m
    )

    if (
        not np.isfinite(
            actor_depth
        )
        or
        actor_depth <= 0.0
    ):

        meta[
            "reason"
        ] = "invalid_actor_depth"

        return (
            sprite_rgba,
            meta,
        )

    scene_depth = np.asarray(
        scene_depth_m,
        dtype=np.float32,
    )

    if scene_depth.ndim != 2:

        raise ValueError(
            "scene_depth_m must be a 2-D metric depth image."
        )

    output = (
        sprite_rgba.copy()
    )

    sprite_h, sprite_w = (
        output.shape[
            :2
        ]
    )

    frame_h, frame_w = (
        scene_depth.shape
    )

    x1 = int(
        paste_x1
    )

    y1 = int(
        paste_y1
    )

    x2 = (
        x1
        + sprite_w
    )

    y2 = (
        y1
        + sprite_h
    )

    # --------------------------------------------------------
    # Intersection of local sprite crop with camera frame.
    # --------------------------------------------------------

    frame_x1 = max(
        0,
        x1,
    )

    frame_y1 = max(
        0,
        y1,
    )

    frame_x2 = min(
        frame_w,
        x2,
    )

    frame_y2 = min(
        frame_h,
        y2,
    )

    meta[
        "enabled"
    ] = True

    if (
        frame_x2 <= frame_x1
        or
        frame_y2 <= frame_y1
    ):

        meta[
            "reason"
        ] = "sprite_outside_depth_frame"

        return (
            output,
            meta,
        )

    # Corresponding region inside local sprite crop.
    sprite_x1 = (
        frame_x1
        - x1
    )

    sprite_y1 = (
        frame_y1
        - y1
    )

    sprite_x2 = (
        frame_x2
        - x1
    )

    sprite_y2 = (
        frame_y2
        - y1
    )

    depth_region = (
        scene_depth[
            frame_y1:frame_y2,
            frame_x1:frame_x2,
        ]
    )

    alpha_region = (
        output[
            sprite_y1:sprite_y2,
            sprite_x1:sprite_x2,
            3
        ]
    )

    sprite_pixels = (
        alpha_region
        > 0
    )

    margin_m = max(
        0.0,
        float(
            margin_m
        ),
    )

    occlusion_threshold_m = (
        actor_depth
        - margin_m
    )

    valid_scene_depth = (
        np.isfinite(
            depth_region
        )
        &
        (
            depth_region
            > 0.0
        )
    )

    # --------------------------------------------------------
    # Conservative occlusion rule:
    #
    # scene geometry must be clearly closer than the nearest
    # physical surface of the HE actor.
    #
    # Using nearest actor depth intentionally avoids treating
    # road pixels near the vehicle contact point as foreground.
    # --------------------------------------------------------

    foreground = (
        sprite_pixels
        &
        valid_scene_depth
        &
        (
            depth_region
            <
            occlusion_threshold_m
        )
    )

    original_alpha_pixels = int(
        np.count_nonzero(
            sprite_pixels
        )
    )

    occluded_alpha_pixels = int(
        np.count_nonzero(
            foreground
        )
    )

    # alpha_region is a view into output, so this modifies
    # the local warped RGBA crop directly.
    alpha_region[
        foreground
    ] = 0

    remaining_alpha_pixels = int(
        np.count_nonzero(
            alpha_region
            > 0
        )
    )

    if original_alpha_pixels > 0:

        occluded_fraction = (
            float(
                occluded_alpha_pixels
            )
            /
            float(
                original_alpha_pixels
            )
        )

    else:

        occluded_fraction = 0.0

    meta.update({
        "reason":
            "ok",

        "occlusion_threshold_m":
            float(
                occlusion_threshold_m
            ),

        "sprite_alpha_pixels_in_frame":
            original_alpha_pixels,

        "occluded_alpha_pixels":
            occluded_alpha_pixels,

        "remaining_alpha_pixels":
            remaining_alpha_pixels,

        "occluded_fraction":
            float(
                occluded_fraction
            ),
    })

    return (
        output,
        meta,
    )


# ============================================================
# Production 4320 view-matrix renderer
# ============================================================

def render_he_actor_view_matrix(
    base_rgb,
    actor_tf,
    camera_tf,
    dimensions,
    sprite_bank,
    view_matrix,
    sprite_cache,
    width,
    height,
    fov,
    bottom_y_offset_px=0.0,
    geometry_mode="proxy",
    projection_mode="oriented_2p5d_support",
    sprite_geometry=None,
    close_width_blend_near_m=5.10,
    close_width_blend_far_m=5.90,
    scene_depth_m=None,
    scene_occlusion_margin_m=0.25,
    silhouette_scale=1.0,
    warp_scale_mode="independent",
    viewpoint_lateral_sign=1.0,
    center_depth_alpha_bbox_anchor=False,
    camera_rotation_reprojection=False,
    camera_rotation_reprojection_min_width_fraction=0.20,
    camera_rotation_reprojection_min_bearing_deg=30.0,
):

    frame = (
        base_rgb.copy()
    )
    geometry_mode = (
        str(
            geometry_mode
        )
        .strip()
        .lower()
    )
    silhouette_scale = float(silhouette_scale)
    warp_scale_mode = str(warp_scale_mode).strip().lower()
    viewpoint_lateral_sign = float(viewpoint_lateral_sign)
    if viewpoint_lateral_sign not in {-1.0, 1.0}:
        raise ValueError(
            "viewpoint_lateral_sign must be -1.0 or 1.0, got "
            f"{viewpoint_lateral_sign}"
        )
    if warp_scale_mode not in {
        "independent",
        "uniform_height_preserve_aspect",
    }:
        raise ValueError(
            "warp_scale_mode must be 'independent' or "
            "'uniform_height_preserve_aspect', got "
            f"{warp_scale_mode!r}"
        )
    if not 0.90 <= silhouette_scale <= 1.10:
        raise ValueError(
            "silhouette_scale must be within [0.90, 1.10], got "
            f"{silhouette_scale}"
        )

    if geometry_mode not in {
        "proxy",
        "sprite_native",
        "sprite_native_width",
        "close_width_blend",
        "sprite_alpha_metric",
        "sprite_alpha_width_proxy_height",
    }:
        raise ValueError(
            "geometry_mode must be one of "
            "'proxy', 'sprite_native', "
            "'sprite_native_width', 'close_width_blend', or "
            "'sprite_alpha_metric', or "
            "'sprite_alpha_width_proxy_height', got "
            f"{geometry_mode!r}"
        )
    projection_mode = (
        str(projection_mode)
        .strip()
        .lower()
    )

    if projection_mode not in {
        "oriented_2p5d_support",
        "center_depth_billboard",
    }:
        raise ValueError(
            "projection_mode must be either "
            "'oriented_2p5d_support' or "
            "'center_depth_billboard', got "
            f"{projection_mode!r}"
        )

    # --------------------------------------------------------
    # Native-camera metric projection
    # --------------------------------------------------------

    if projection_mode == "center_depth_billboard":
        box = (
            project_virtual_actor_center_depth_billboard(
                actor_tf=actor_tf,
                camera_tf=camera_tf,
                dimensions=dimensions,
                width=width,
                height=height,
                fov=fov,
            )
        )
    else:
        box = project_virtual_actor(
            actor_tf=actor_tf,
            camera_tf=camera_tf,
            dimensions=dimensions,
            width=width,
            height=height,
            fov=fov,
        )

        if (
            not box.get("visible", False)
            and geometry_mode in {
                "sprite_alpha_metric",
                "sprite_alpha_width_proxy_height",
            }
            and box.get("reason") == "footprint_intersects_camera_plane"
        ):
            box = project_virtual_actor_center_depth_billboard(
                actor_tf=actor_tf,
                camera_tf=camera_tf,
                dimensions=dimensions,
                width=width,
                height=height,
                fov=fov,
            )
            if box.get("visible", False):
                box["visibility_fallback"] = "center_depth_billboard"

    meta = {
        "rendered":
            False,

        "box":
            box,

        "sprite_mode":
            "view_matrix",

        "selected_angle":
            None,

        "angle_error_deg":
            None,

        "viewpoint_angle_deg":
            None,

        "selected_distance_m":
            None,

        "query_distance_m":
            None,

        "selected_elevation_deg":
            None,

        "query_elevation_deg":
            None,

        "geometry_mode":
            geometry_mode,

        "projection_mode":
            projection_mode,

        "geometry_version":
            None,

        "geometry_alpha_threshold":
            None,

        "target_box_width_px":
            None,

        "target_box_height_px":
            None,

        "sprite_native_geometry":
            None,

        "sprite_geometry_query_distance_m":
            None,

        "sprite_geometry_depth_coordinate":
            None,

        "close_width_blend_near_m":
            None,

        "close_width_blend_far_m":
            None,

        "close_width_blend_weight_native":
            None,

        "close_width_proxy_width_px":
            None,

        "close_width_native_width_px":
            None,

        "close_width_final_width_px":
            None,

        "target_visible_bbox_unclipped":
            None,

        "rendered_alpha_bbox":
            None,

        "warp_scale_mode":
            warp_scale_mode,

        "viewpoint_lateral_sign":
            float(viewpoint_lateral_sign),

        "camera_rotation_reprojection":
            bool(camera_rotation_reprojection),

        "camera_rotation_reprojection_meta":
            None,

        "camera_rotation_reprojection_eligible":
            False,
    
    }

    if not box.get(
        "visible",
        False,
    ):
        return (
            frame,
            meta,
        )
    # --------------------------------------------------------
    # Optional camera-specific vertical placement calibration.
    #
    # Positive = move sprite downward.
    # Negative = move sprite upward.
    #
    # This changes only final image placement. It does NOT
    # modify actor/world geometry, depth, TTC, or safety metrics.
    # --------------------------------------------------------

    render_bottom_y = (
        float(
            box[
                "bottom_y"
            ]
        )
        +
        float(
            bottom_y_offset_px
        )
    )

    meta[
        "bottom_y_offset_px"
    ] = float(
        bottom_y_offset_px
    )

    meta[
        "render_bottom_y"
    ] = float(
        render_bottom_y
    )
    # --------------------------------------------------------
    # Camera-relative state expected by the production
    # view-matrix selector.
    # --------------------------------------------------------

    state = {
        "x_m":
            float(viewpoint_lateral_sign)
            *
            float(box["camera_right_m"]),

        "y_m":
            float(
                box[
                    "camera_up_m"
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

    # --------------------------------------------------------
    # Frozen 4320 selector:
    #
    # viewpoint x distance x elevation
    # --------------------------------------------------------

    sprite_info = (
        select_view_matrix_sprite(
            state=state,
            sprite_bank=sprite_bank,
            view_matrix=view_matrix,
        )
    )

    if not sprite_info.get(
        "exists",
        False,
    ):
        meta[
            "reason"
        ] = "sprite_not_found"

        return (
            frame,
            meta,
        )

    sprite_rgba = sprite_cache.load_rgba(
        sprite_info["sprite_path"]
    )
    source_alpha_y, source_alpha_x = np.where(
        sprite_rgba[:, :, 3] > 10
    )
    if len(source_alpha_x) == 0 or len(source_alpha_y) == 0:
        meta["reason"] = "empty_sprite_alpha"
        return frame, meta

    source_alpha_center_x = (
        float(source_alpha_x.min()) + float(source_alpha_x.max())
    ) / 2.0
    source_alpha_bottom_y = float(source_alpha_y.max())
    source_alpha_width_px = float(
        source_alpha_x.max() - source_alpha_x.min() + 1
    )
    source_alpha_height_px = float(
        source_alpha_y.max() - source_alpha_y.min() + 1
    )
    # ========================================================
    # Physical source -> physical target anchor
    #
    # Source:
    #     physical bbox support center stored by asset generator
    #
    # Target:
    #     same physical support center reconstructed from the
    #     virtual actor pose and projected into this native camera
    #
    # Width/height remain unchanged in this patch.
    # ========================================================

    physical_bbox = (
        view_matrix.get(
            "physical_bbox"
        )
    )

    target_support_anchor = (
        project_asset_physical_support_anchor(
            actor_tf=actor_tf,
            camera_tf=camera_tf,
            physical_bbox=physical_bbox,
            width=width,
            height=height,
            fov=fov,
        )
    )

    source_ground_anchor_x = (
        sprite_info.get(
            "source_ground_anchor_x"
        )
    )

    source_ground_anchor_y = (
        sprite_info.get(
            "source_ground_anchor_y"
        )
    )

    alpha_metric_target_anchor = None
    if geometry_mode == "sprite_alpha_metric":
        target_bbox_center = project_asset_physical_bbox_center(
            actor_tf=actor_tf,
            camera_tf=camera_tf,
            physical_bbox=physical_bbox,
            width=width,
            height=height,
            fov=fov,
        )
        capture_metadata = view_matrix.get("capture_metadata") or {}
        capture_fx = float(capture_metadata.get("camera_fx_px", 0.0))
        capture_cx = float(capture_metadata.get("camera_cx_px", 0.0))
        capture_cy = float(capture_metadata.get("camera_cy_px", 0.0))
        crop_x1 = sprite_info.get("crop_x1_px")
        crop_y1 = sprite_info.get("crop_y1_px")
        if (
            target_bbox_center is not None
            and capture_fx > 0.0
            and crop_x1 is not None
            and crop_y1 is not None
        ):
            runtime_fx = float(width) / (
                2.0 * math.tan(math.radians(float(fov)) / 2.0)
            )
            selected_distance = max(
                float(sprite_info["selected_distance_m"]), 1e-6
            )
            runtime_depth = max(float(target_bbox_center["depth"]), 1e-6)
            anchor_scale = (
                runtime_fx / capture_fx * selected_distance / runtime_depth
            )
            source_center_x = capture_cx - float(crop_x1)
            source_center_y = capture_cy - float(crop_y1)
            alpha_metric_target_anchor = {
                "x": float(target_bbox_center["u"])
                + (source_alpha_center_x - source_center_x) * anchor_scale,
                "y": float(target_bbox_center["v"])
                + (source_alpha_bottom_y - source_center_y) * anchor_scale,
            }

    use_physical_anchor = (
        (
            projection_mode != "center_depth_billboard"
            or geometry_mode in {
                "sprite_alpha_metric",
                "sprite_alpha_width_proxy_height",
            }
        )
        and
        target_support_anchor
        is not None
        and
        source_ground_anchor_x
        is not None
        and
        source_ground_anchor_y
        is not None
    )

    if alpha_metric_target_anchor is not None:
        render_target_x = float(alpha_metric_target_anchor["x"])
        render_target_y = (
            float(alpha_metric_target_anchor["y"])
            + float(bottom_y_offset_px)
        )
        render_source_anchor_x = source_alpha_center_x
        render_source_anchor_y = source_alpha_bottom_y
        anchor_mode = "sprite_alpha_bbox_center_reprojection"

    elif use_physical_anchor:

        render_target_x = float(
            target_support_anchor[
                "u"
            ]
        )

        render_target_y = (
            float(
                target_support_anchor[
                    "v"
                ]
            )
            +
            float(
                bottom_y_offset_px
            )
        )

        render_source_anchor_x = float(
            source_ground_anchor_x
        )

        render_source_anchor_y = float(
            source_ground_anchor_y
        )

        anchor_mode = (
            "physical_bbox_support_center"
        )

    elif (
        projection_mode == "center_depth_billboard"
        and bool(center_depth_alpha_bbox_anchor)
    ):
        render_target_x = float(box["cx"])
        render_target_y = float(render_bottom_y)
        render_source_anchor_x = source_alpha_center_x
        render_source_anchor_y = source_alpha_bottom_y
        anchor_mode = "center_depth_alpha_bbox_center"

    else:

        # Backward-compatible fallback for old banks.
        render_target_x = float(
            box["cx"]
        )

        render_target_y = float(
            render_bottom_y
        )

        render_source_anchor_x = float(
            sprite_info["anchor_x"]
        )

        render_source_anchor_y = float(
            sprite_info["anchor_y"]
        )

        anchor_mode = (
            "center_depth_alpha_bottom"
            if projection_mode == "center_depth_billboard"
            else "legacy_alpha_bottom"
        )
    if (
        os.environ.get("HE_DEBUG_ANCHOR") == "1"
        and
        float(box["depth_m"]) < 20.0
        and
        abs(float(box["camera_right_m"])) > 1.0
    ):
        print(
            "[HE-ANCHOR]",
            "mode=", anchor_mode,
            "depth=", round(float(box["depth_m"]), 3),
            "right=", round(float(box["camera_right_m"]), 3),
            "bearing=", round(
                math.degrees(
                    math.atan2(
                        float(box["camera_right_m"]),
                        float(box["depth_m"]),
                    )
                ),
                3,
            ),
            "rel_yaw=", round(
                float(
                    box[
                        "actor_relative_yaw_deg"
                    ]
                ),
                3,
            ),
            "angle=", round(
                float(sprite_info["relative_angle_deg"]),
                3,
            ),
            "selected=", int(
                sprite_info["selected_angle"]
            ),
            "box_cx=", round(
                float(box["cx"]),
                3,
            ),
            "actor_ref_cx=", round(
                float(box["actor_reference_cx_px"]),
                3,
            ),
            "target_x=", round(
                float(render_target_x),
                3,
            ),
            "source_anchor_x=", round(
                float(render_source_anchor_x),
                3,
            ),
            "physical_bbox=",
            physical_bbox is not None,
            "source_ground_anchor=",
            (
                source_ground_anchor_x is not None
                and
                source_ground_anchor_y is not None
            ),
        )
    # --------------------------------------------------------
    # Final visible geometry.
    #
    # proxy:
    #     Preserve the existing dimensions-based projection.
    #
    # sprite_native:
    #     Diagnostic full sprite-native geometry. Width and height
    #     both come from the sprite geometry field, using the legacy
    #     forward-depth coordinate retained for A/B comparison.
    #
    # sprite_native_width:
    #     Diagnostic width-only mode. Keep center-depth billboard
    #     placement and HEIGHT unchanged, but obtain WIDTH from the
    #     sprite-native geometry field using the production view-matrix
    #     radial camera-to-target distance.
    #
    # close_width_blend:
    #     Gate-1 production candidate. Keep proxy width outside the
    #     very-close regime, keep radial sprite-native width inside it,
    #     and use smoothstep between the two measured crossover bounds.
    #     HEIGHT, placement, anchor, visibility and support-depth logic
    #     remain unchanged.
    # --------------------------------------------------------

    target_box_w = float(
        box[
            "box_width"
        ]
    )

    target_box_h = float(
        box[
            "box_height"
        ]
    )

    warp_alpha_threshold = 10

    geometry_prediction = None

    if geometry_mode in {
        "sprite_alpha_metric",
        "sprite_alpha_width_proxy_height",
    }:
        capture_metadata = view_matrix.get("capture_metadata") or {}
        capture_fx = float(capture_metadata.get("camera_fx_px", 0.0))
        if capture_fx <= 0.0:
            capture_width = float(capture_metadata.get("image_width_px", 0.0))
            capture_fov = float(capture_metadata.get("fov_deg", 0.0))
            if capture_width <= 0.0 or capture_fov <= 0.0:
                meta["reason"] = "missing_capture_intrinsics"
                return frame, meta
            capture_fx = capture_width / (
                2.0 * math.tan(math.radians(capture_fov) / 2.0)
            )

        runtime_fx = float(width) / (
            2.0 * math.tan(math.radians(float(fov)) / 2.0)
        )
        # Pinhole image scale is inverse camera-forward depth. The selected
        # far-bank capture distance is radial because its camera points at the
        # actor, but the runtime camera can be strongly oblique (NEAT side
        # cameras). Using runtime radial distance shrinks the sprite by cos(bearing).
        query_distance = max(float(box["depth_m"]), 1e-6)
        selected_distance = max(float(sprite_info["selected_distance_m"]), 1e-6)
        alpha_metric_scale = (
            runtime_fx / capture_fx * selected_distance / query_distance
        )
        target_box_w = source_alpha_width_px * alpha_metric_scale
        if geometry_mode == "sprite_alpha_metric":
            target_box_h = source_alpha_height_px * alpha_metric_scale
        meta["geometry_version"] = (
            "sprite_alpha_metric_v1"
            if geometry_mode == "sprite_alpha_metric"
            else "sprite_alpha_width_proxy_height_v1"
        )
        meta["sprite_alpha_metric_scale"] = float(alpha_metric_scale)
        meta["sprite_alpha_metric_depth_coordinate"] = "camera_forward_depth"
        meta["source_alpha_width_px"] = source_alpha_width_px
        meta["source_alpha_height_px"] = source_alpha_height_px

    if geometry_mode in {
        "sprite_native",
        "sprite_native_width",
        "close_width_blend",
    }:

        if sprite_geometry is None:

            meta[
                "reason"
            ] = "sprite_geometry_not_provided"

            return (
                frame,
                meta,
            )

        if geometry_mode in {
            "sprite_native_width",
            "close_width_blend",
        }:
            geometry_query_distance_m = float(
                sprite_info[
                    "query_distance_m"
                ]
            )
            geometry_depth_coordinate = (
                "view_matrix_radial_distance"
            )
        else:
            # Retain the old full sprite-native behavior only as a
            # diagnostic A/B mode. Gate-1 production candidate is the
            # width-only radial-distance mode above.
            geometry_query_distance_m = float(
                box[
                    "depth_m"
                ]
            )
            geometry_depth_coordinate = (
                "camera_forward_center_depth"
            )

        geometry_prediction = (
            sprite_geometry.predict(
                viewpoint_angle_deg=
                    float(
                        sprite_info[
                            "relative_angle_deg"
                        ]
                    ),

                elevation_deg=
                    float(
                        sprite_info[
                            "query_elevation_deg"
                        ]
                    ),

                depth_m=
                    geometry_query_distance_m,

                target_width=
                    width,

                target_height=
                    height,

                target_fov=
                    fov,
            )
        )

        meta[
            "sprite_geometry_query_distance_m"
        ] = float(
            geometry_query_distance_m
        )

        meta[
            "sprite_geometry_depth_coordinate"
        ] = str(
            geometry_depth_coordinate
        )

        if not geometry_prediction.get(
            "valid",
            False,
        ):

            meta[
                "reason"
            ] = (
                "sprite_native_geometry_failed"
            )

            meta[
                "sprite_native_geometry"
            ] = geometry_prediction

            return (
                frame,
                meta,
            )

        native_width_px = float(
            geometry_prediction[
                "box_width_px"
            ]
        )

        proxy_width_px = float(
            box[
                "box_width"
            ]
        )

        if geometry_mode == "sprite_native":
            target_box_w = float(
                native_width_px
            )

            # Legacy diagnostic mode: replace height too.
            target_box_h = float(
                geometry_prediction[
                    "box_height_px"
                ]
            )

            # Full sprite-native mode uses the geometry table's alpha
            # threshold for both dimensions, preserving previous A/B.
            warp_alpha_threshold = int(
                geometry_prediction[
                    "geometry_alpha_threshold"
                ]
            )

        elif geometry_mode == "sprite_native_width":
            target_box_w = float(
                native_width_px
            )

            # Width-only diagnostic: preserve center-depth height.
            target_box_h = float(
                box[
                    "box_height"
                ]
            )

        elif geometry_mode == "close_width_blend":
            near_m = float(
                close_width_blend_near_m
            )
            far_m = float(
                close_width_blend_far_m
            )

            if not (
                near_m > 0.0
                and
                far_m > near_m
            ):
                raise ValueError(
                    "close_width_blend requires "
                    "0 < near_m < far_m, got "
                    f"{near_m}, {far_m}"
                )

            radial_m = float(
                geometry_query_distance_m
            )

            if radial_m <= near_m:
                native_weight = 1.0
            elif radial_m >= far_m:
                native_weight = 0.0
            else:
                # 0 at far bound -> 1 at near bound.
                linear_t = (
                    far_m
                    -
                    radial_m
                ) / (
                    far_m
                    -
                    near_m
                )

                # C1-continuous smoothstep. This prevents a visible
                # width-velocity jump when entering/leaving the close
                # regime.
                native_weight = (
                    linear_t
                    *
                    linear_t
                    *
                    (
                        3.0
                        -
                        2.0
                        *
                        linear_t
                    )
                )

            target_box_w = (
                (
                    1.0
                    -
                    native_weight
                )
                *
                proxy_width_px
                +
                native_weight
                *
                native_width_px
            )

            # Height stays fully frozen on center-depth geometry.
            target_box_h = float(
                box[
                    "box_height"
                ]
            )

            meta[
                "close_width_blend_near_m"
            ] = float(
                near_m
            )

            meta[
                "close_width_blend_far_m"
            ] = float(
                far_m
            )

            meta[
                "close_width_blend_weight_native"
            ] = float(
                native_weight
            )

            meta[
                "close_width_proxy_width_px"
            ] = float(
                proxy_width_px
            )

            meta[
                "close_width_native_width_px"
            ] = float(
                native_width_px
            )

            meta[
                "close_width_final_width_px"
            ] = float(
                target_box_w
            )

        meta[
            "geometry_version"
        ] = geometry_prediction.get(
            "geometry_version"
        )

        meta[
            "sprite_native_geometry"
        ] = geometry_prediction

    meta[
        "geometry_alpha_threshold"
    ] = int(
        warp_alpha_threshold
    )

    # Uniform calibration around the physical support anchor. The neutral
    # default preserves all accepted renderer behavior.
    target_box_w *= silhouette_scale
    target_box_h *= silhouette_scale
    meta["silhouette_scale"] = silhouette_scale

    meta[
        "target_box_width_px"
    ] = float(
        target_box_w
    )

    meta[
        "target_box_height_px"
    ] = float(
        target_box_h
    )

    # --------------------------------------------------------
    # Frozen stable subpixel renderer.
    #
    # Appearance:
    #     selected 4320 sprite
    #
    # Final geometry:
    #     native-camera metric projection above
    # --------------------------------------------------------

    rotation_reprojection = None
    optical_bearing_deg = abs(math.degrees(math.atan2(
        float(box["camera_right_m"]), float(box["depth_m"])
    )))
    target_width_fraction = float(target_box_w) / max(float(width), 1.0)
    rotation_reprojection_eligible = (
        target_width_fraction
        >= float(camera_rotation_reprojection_min_width_fraction)
        or optical_bearing_deg
        >= float(camera_rotation_reprojection_min_bearing_deg)
    )
    meta["camera_rotation_reprojection_eligible"] = bool(
        rotation_reprojection_eligible
    )
    meta["camera_rotation_reprojection_optical_bearing_deg"] = float(
        optical_bearing_deg
    )
    meta["camera_rotation_reprojection_target_width_fraction"] = float(
        target_width_fraction
    )
    if bool(camera_rotation_reprojection) and rotation_reprojection_eligible:
        rotation_reprojection = warp_continuous_view_matrix_camera_rotation(
            sprite_info=sprite_info,
            view_matrix=view_matrix,
            sprite_cache=sprite_cache,
            actor_tf=actor_tf,
            camera_tf=camera_tf,
            physical_bbox=physical_bbox,
            target_support_anchor=target_support_anchor,
            width=width,
            height=height,
            fov=fov,
            silhouette_scale=silhouette_scale,
            alpha_threshold=warp_alpha_threshold,
        )

    if rotation_reprojection is not None:
        warped_rgba, resize_info, reprojection_meta = rotation_reprojection
        meta["camera_rotation_reprojection_meta"] = reprojection_meta
        anchor_mode = "camera_rotation_physical_support"
        visible_bbox = resize_info["visible_bbox"]
        target_box_w = float(visible_bbox["x2"] - visible_bbox["x1"] + 1)
        target_box_h = float(visible_bbox["y2"] - visible_bbox["y1"] + 1)
        meta["target_box_width_px"] = target_box_w
        meta["target_box_height_px"] = target_box_h
    else:
        (
            warped_rgba,
            resize_info,
        ) = warp_view_matrix_sprite_to_box_subpixel(
            sprite_rgba=sprite_rgba,

            frame_w=width,
            frame_h=height,

            target_cx=
                render_target_x,

            target_bottom_y=
                render_target_y,

            target_box_w=
                target_box_w,

            target_box_h=
                target_box_h,

            anchor_x=
                render_source_anchor_x,

            anchor_y=
                render_source_anchor_y,

            alpha_threshold=
                warp_alpha_threshold,

            scale_mode=
                warp_scale_mode,
        )

    # --------------------------------------------------------
    # Analytic pre-clipping visible alpha bbox.
    #
    # target_box_w/h describe the requested visible silhouette
    # size, but the selected source anchor does not have to be
    # the alpha-bbox center/bottom. Derive the actual target
    # alpha bounds from the source visible bbox and the exact
    # subpixel transform so close-range QA compares like with like.
    # --------------------------------------------------------

    source_visible_bbox = (
        resize_info[
            "visible_bbox"
        ]
    )

    target_visible_x1 = (
        float(
            resize_info[
                "float_x1"
            ]
        )
        +
        float(
            source_visible_bbox[
                "x1"
            ]
        )
        *
        float(
            resize_info[
                "scale_x"
            ]
        )
    )

    target_visible_y1 = (
        float(
            resize_info[
                "float_y1"
            ]
        )
        +
        float(
            source_visible_bbox[
                "y1"
            ]
        )
        *
        float(
            resize_info[
                "scale_y"
            ]
        )
    )

    target_visible_x2 = (
        float(
            resize_info[
                "float_x1"
            ]
        )
        +
        (
            float(
                source_visible_bbox[
                    "x2"
                ]
            )
            +
            1.0
        )
        *
        float(
            resize_info[
                "scale_x"
            ]
        )
    )

    target_visible_y2 = (
        float(
            resize_info[
                "float_y1"
            ]
        )
        +
        (
            float(
                source_visible_bbox[
                    "y2"
                ]
            )
            +
            1.0
        )
        *
        float(
            resize_info[
                "scale_y"
            ]
        )
    )

    meta[
        "target_visible_bbox_unclipped"
    ] = {
        "x1":
            float(
                target_visible_x1
            ),

        "y1":
            float(
                target_visible_y1
            ),

        "x2":
            float(
                target_visible_x2
            ),

        "y2":
            float(
                target_visible_y2
            ),

        "width_px":
            float(
                target_visible_x2
                -
                target_visible_x1
            ),

        "height_px":
            float(
                target_visible_y2
                -
                target_visible_y1
            ),

        "center_x":
            float(
                (
                    target_visible_x1
                    +
                    target_visible_x2
                )
                /
                2.0
            ),

        "bottom_y":
            float(
                target_visible_y2
            ),
    }

    if resize_info.get(
        "fully_outside_frame",
        False,
    ):

        meta[
            "reason"
        ] = "fully_outside_frame"

        return (
            frame,
            meta,
        )

    paste = (
        resize_info[
            "paste"
        ]
    )

    # --------------------------------------------------------
    # Optional CARLA/world -> HE depth occlusion.
    #
    # This happens AFTER:
    #   - actor projection
    #   - sprite selection
    #   - sprite geometry
    #   - subpixel warp
    #
    # Therefore it cannot modify the frozen HE placement or
    # geometry calibration. It only removes HE pixels that are
    # physically hidden behind closer world geometry.
    # --------------------------------------------------------

    (
        warped_rgba,
        scene_occlusion_meta,
    ) = apply_scene_depth_occlusion_to_sprite(
        sprite_rgba=
            warped_rgba,

        scene_depth_m=
            scene_depth_m,

        paste_x1=
            paste[
                "x1"
            ],

        paste_y1=
            paste[
                "y1"
            ],

        actor_nearest_depth_m=
            box.get(
                "nearest_depth_m",
                box[
                    "depth_m"
                ],
            ),

        margin_m=
            scene_occlusion_margin_m,
    )

    meta[
        "scene_occlusion"
    ] = scene_occlusion_meta

    (
        frame,
        full_mask,
    ) = alpha_composite_rgb(
        frame_rgb=frame,
        sprite_rgba=warped_rgba,
        x1=int(
            paste[
                "x1"
            ]
        ),
        y1=int(
            paste[
                "y1"
            ]
        ),
        global_alpha=1.0,
    )

    alpha_binary = (
        full_mask
        >=
        int(
            warp_alpha_threshold
        )
    )

    if not np.any(
        alpha_binary
    ):
        alpha_binary = (
            full_mask > 0
        )

    alpha_ys, alpha_xs = np.where(
        alpha_binary
    )

    if (
        len(alpha_xs) > 0
        and
        len(alpha_ys) > 0
    ):
        alpha_x1 = int(
            alpha_xs.min()
        )
        alpha_x2 = int(
            alpha_xs.max()
        )
        alpha_y1 = int(
            alpha_ys.min()
        )
        alpha_y2 = int(
            alpha_ys.max()
        )

        meta[
            "rendered_alpha_bbox"
        ] = {
            "x1":
                alpha_x1,

            "y1":
                alpha_y1,

            "x2":
                alpha_x2,

            "y2":
                alpha_y2,

            "width_px":
                float(
                    alpha_x2
                    -
                    alpha_x1
                    +
                    1
                ),

            "height_px":
                float(
                    alpha_y2
                    -
                    alpha_y1
                    +
                    1
                ),

            "center_x":
                float(
                    (
                        alpha_x1
                        +
                        alpha_x2
                    )
                    /
                    2.0
                ),

            "bottom_y":
                float(
                    alpha_y2
                ),

            "alpha_threshold":
                int(
                    warp_alpha_threshold
                ),
        }

    rendered = bool(
        np.any(
            full_mask > 0
        )
    )
    meta[
        "anchor_mode"
    ] = anchor_mode

    meta[
        "source_anchor_x_px"
    ] = float(
        render_source_anchor_x
    )

    meta[
        "source_anchor_y_px"
    ] = float(
        render_source_anchor_y
    )

    meta[
        "target_anchor_x_px"
    ] = float(
        render_target_x
    )

    meta[
        "target_anchor_y_px"
    ] = float(
        render_target_y
    )

    meta[
        "legacy_source_anchor_x_px"
    ] = float(
        sprite_info[
            "anchor_x"
        ]
    )

    meta[
        "legacy_source_anchor_y_px"
    ] = float(
        sprite_info[
            "anchor_y"
        ]
    )

    meta[
        "target_physical_support"
    ] = (
        target_support_anchor
    )
    meta.update({
        "rendered":
            rendered,

        "viewpoint_angle_deg":
            float(
                sprite_info[
                    "relative_angle_deg"
                ]
            ),

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

        "query_distance_m":
            float(
                sprite_info[
                    "query_distance_m"
                ]
            ),

        "selected_distance_m":
            float(
                sprite_info[
                    "selected_distance_m"
                ]
            ),

        "query_elevation_deg":
            float(
                sprite_info[
                    "query_elevation_deg"
                ]
            ),

        "selected_elevation_deg":
            float(
                sprite_info[
                    "selected_elevation_deg"
                ]
            ),

        "distance_selection_mode":
            sprite_info[
                "distance_selection_mode"
            ],

        "sprite_path":
            sprite_info[
                "sprite_path"
            ],

        "render_transform_mode":
            resize_info[
                "mode"
            ],

        "sprite_width":
            int(
                paste[
                    "sprite_width"
                ]
            ),

        "sprite_height":
            int(
                paste[
                    "sprite_height"
                ]
            ),

        "paste_x1":
            int(
                paste[
                    "x1"
                ]
            ),

        "paste_y1":
            int(
                paste[
                    "y1"
                ]
            ),

        "resize_info":
            resize_info,
    })

    return (
        frame,
        meta,
    )
