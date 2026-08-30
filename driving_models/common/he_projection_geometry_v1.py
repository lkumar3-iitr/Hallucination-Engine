"""
he_projection_geometry_v1.py

Stable center-depth billboard projection for Hallucination Engine sprites.

Rendering placement:
    - anchor at the actor reference/base projection
    - scale using actor-center depth
    - use an orientation-dependent effective footprint width at center depth

Physical support / nearest / farthest depths are retained for diagnostics,
occlusion, depth ordering, TTC, etc., but they do not control sprite scale
or render visibility.
"""

from __future__ import annotations

import math

import carla
import numpy as np


def normalize_angle_180(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def make_camera_intrinsic(width, height, fov_deg):
    focal = (
        float(width)
        /
        (
            2.0
            *
            math.tan(
                math.radians(float(fov_deg))
                / 2.0
            )
        )
    )

    k = np.identity(3, dtype=np.float64)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = float(width) / 2.0
    k[1, 2] = float(height) / 2.0
    return k


def _world_to_camera_xyz(point, world_to_camera):
    p = np.array(
        [
            float(point.x),
            float(point.y),
            float(point.z),
            1.0,
        ],
        dtype=np.float64,
    )

    pc = world_to_camera @ p

    return {
        "depth": float(pc[0]),
        "camera_right_m": float(pc[1]),
        "camera_up_m": float(pc[2]),
    }


def _project_world_point(
    point,
    world_to_camera,
    k,
    near_plane_m=0.05,
):
    pc = _world_to_camera_xyz(
        point,
        world_to_camera,
    )

    depth = float(pc["depth"])

    if depth <= float(near_plane_m):
        return None

    image_vector = (
        k
        @ np.array(
            [
                float(pc["camera_right_m"]),
                -float(pc["camera_up_m"]),
                depth,
            ],
            dtype=np.float64,
        )
    )

    u = float(image_vector[0] / image_vector[2])
    v = float(image_vector[1] / image_vector[2])

    return {
        "u": u,
        "v": v,
        **pc,
    }


def project_virtual_actor_center_depth_billboard(
    actor_tf,
    camera_tf,
    dimensions,
    width,
    height,
    fov,
    near_plane_m=0.05,
    visibility_margin_px=30.0,
):
    """
    Stable billboard projection.

    The full sprite is anchored at the actor reference/base and scaled from
    actor-center depth. It does NOT use nearest/support depth for scale.

    Apparent metric width:
        W_eff = |W cos(phi)| + |L sin(phi)|

    where phi is actor heading relative to the camera-to-actor ray.
    """

    actor_length = float(dimensions["length_m"])
    actor_width = float(dimensions["width_m"])
    actor_height = float(dimensions["height_m"])

    if (
        actor_length <= 0.0
        or actor_width <= 0.0
        or actor_height <= 0.0
    ):
        return {
            "visible": False,
            "reason": "invalid_dimensions",
            "geometry_mode": "center_depth_billboard",
        }

    k = make_camera_intrinsic(
        width,
        height,
        fov,
    )

    focal_x = float(k[0, 0])

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

    actor_reference = carla.Location(
        x=float(actor_tf.location.x),
        y=float(actor_tf.location.y),
        z=float(actor_tf.location.z),
    )

    reference_projection = _project_world_point(
        actor_reference,
        world_to_camera,
        k,
        near_plane_m=near_plane_m,
    )

    if reference_projection is None:
        raw_reference = _world_to_camera_xyz(
            actor_reference,
            world_to_camera,
        )

        return {
            "visible": False,
            "reason": "actor_reference_behind_camera",
            "depth_m": float(raw_reference["depth"]),
            "camera_right_m": float(raw_reference["camera_right_m"]),
            "camera_up_m": float(raw_reference["camera_up_m"]),
            "geometry_mode": "center_depth_billboard",
        }

    depth = float(reference_projection["depth"])
    camera_right_m = float(reference_projection["camera_right_m"])
    camera_up_m = float(reference_projection["camera_up_m"])

    actor_reference_cx = float(reference_projection["u"])
    actor_reference_bottom_y = float(reference_projection["v"])

    actor_top = carla.Location(
        x=float(actor_tf.location.x),
        y=float(actor_tf.location.y),
        z=float(actor_tf.location.z) + actor_height,
    )

    top_projection = _project_world_point(
        actor_top,
        world_to_camera,
        k,
        near_plane_m=near_plane_m,
    )

    if top_projection is None:
        return {
            "visible": False,
            "reason": "actor_top_behind_camera",
            "depth_m": depth,
            "camera_right_m": camera_right_m,
            "camera_up_m": camera_up_m,
            "geometry_mode": "center_depth_billboard",
        }

    bottom_y = actor_reference_bottom_y
    top_y = float(top_projection["v"])
    box_height = abs(bottom_y - top_y)

    bearing_deg = math.degrees(
        math.atan2(
            camera_right_m,
            depth,
        )
    )

    actor_relative_yaw = normalize_angle_180(
        float(actor_tf.rotation.yaw)
        -
        float(camera_tf.rotation.yaw)
    )

    phi_deg = normalize_angle_180(
        actor_relative_yaw
        -
        bearing_deg
    )

    phi_rad = math.radians(phi_deg)

    effective_width_m = (
        abs(actor_width * math.cos(phi_rad))
        +
        abs(actor_length * math.sin(phi_rad))
    )

    box_width = (
        focal_x
        *
        effective_width_m
        /
        depth
    )

    cx = actor_reference_cx

    x1 = float(cx - box_width / 2.0)
    x2 = float(cx + box_width / 2.0)
    y1 = float(bottom_y - box_height)
    y2 = float(bottom_y)

    half_length = actor_length / 2.0
    half_width = actor_width / 2.0

    camera_world_h = np.array(
        [
            float(camera_tf.location.x),
            float(camera_tf.location.y),
            float(camera_tf.location.z),
            1.0,
        ],
        dtype=np.float64,
    )

    camera_local_h = world_to_actor @ camera_world_h

    camera_local_x = float(camera_local_h[0])
    camera_local_y = float(camera_local_h[1])

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

    def local_to_world_location(local_x, local_y, local_z):
        local_h = np.array(
            [
                float(local_x),
                float(local_y),
                float(local_z),
                1.0,
            ],
            dtype=np.float64,
        )

        world_h = actor_to_world @ local_h

        return carla.Location(
            x=float(world_h[0]),
            y=float(world_h[1]),
            z=float(world_h[2]),
        )

    support_bottom = local_to_world_location(
        support_local_x,
        support_local_y,
        0.0,
    )

    support_xyz = _world_to_camera_xyz(
        support_bottom,
        world_to_camera,
    )

    support_depth = float(support_xyz["depth"])

    footprint_depths = []
    reference_z = actor_height * 0.5

    for local_x in (-half_length, +half_length):
        for local_y in (-half_width, +half_width):
            world_point = local_to_world_location(
                local_x,
                local_y,
                reference_z,
            )

            xyz = _world_to_camera_xyz(
                world_point,
                world_to_camera,
            )

            footprint_depths.append(
                float(xyz["depth"])
            )

    nearest_depth = float(min(footprint_depths))
    farthest_depth = float(max(footprint_depths))

    margin = float(visibility_margin_px)

    visible = not (
        x2 < -margin
        or x1 > float(width) + margin
        or y2 < -margin
        or y1 > float(height) + margin
    )

    return {
        "visible": bool(visible),

        "cx": float(cx),
        "bottom_y": float(bottom_y),

        "actor_reference_cx_px": float(actor_reference_cx),
        "actor_reference_bottom_y_px": float(actor_reference_bottom_y),

        "box_width": float(box_width),
        "box_height": float(box_height),

        "x1": float(x1),
        "x2": float(x2),
        "y1": float(y1),
        "y2": float(y2),

        "depth_m": float(depth),

        "camera_right_m": float(camera_right_m),
        "camera_up_m": float(camera_up_m),
        "actor_relative_yaw_deg": float(actor_relative_yaw),

        "support_depth_m": float(support_depth),
        "nearest_depth_m": float(nearest_depth),
        "farthest_depth_m": float(farthest_depth),

        "support_local_x_m": float(support_local_x),
        "support_local_y_m": float(support_local_y),

        "bearing_deg": float(bearing_deg),
        "view_relative_yaw_deg": float(phi_deg),
        "effective_width_m": float(effective_width_m),
        "scale_depth_m": float(depth),

        "geometry_mode": "center_depth_billboard",
    }
