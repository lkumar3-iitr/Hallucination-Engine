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
    sprite_geometry=None,
    scene_depth_m=None,
    scene_occlusion_margin_m=0.25,
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

    if geometry_mode not in {
        "proxy",
        "sprite_native",
    }:
        raise ValueError(
            "geometry_mode must be either "
            "'proxy' or 'sprite_native', got "
            f"{geometry_mode!r}"
        )
    # --------------------------------------------------------
    # Native-camera metric projection
    # --------------------------------------------------------

    box = project_virtual_actor(
        actor_tf=actor_tf,
        camera_tf=camera_tf,
        dimensions=dimensions,
        width=width,
        height=height,
        fov=fov,
    )

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
            float(
                box[
                    "camera_right_m"
                ]
            ),

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

    # --------------------------------------------------------
    # Final visible geometry.
    #
    # proxy:
    #     Preserve the existing dimensions-based projection.
    #
    # sprite_native:
    #     Keep actor/camera projection for position, visibility
    #     and depth, but obtain visible width/height directly
    #     from the sprite bank's continuous geometry field.
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

    if geometry_mode == "sprite_native":

        if sprite_geometry is None:

            meta[
                "reason"
            ] = "sprite_geometry_not_provided"

            return (
                frame,
                meta,
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
                    float(
                        box[
                            "depth_m"
                        ]
                    ),

                target_width=
                    width,

                target_height=
                    height,

                target_fov=
                    fov,
            )
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

        target_box_w = float(
            geometry_prediction[
                "box_width_px"
            ]
        )

        target_box_h = float(
            geometry_prediction[
                "box_height_px"
            ]
        )

        warp_alpha_threshold = int(
            geometry_prediction[
                "geometry_alpha_threshold"
            ]
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

    sprite_rgba = (
        sprite_cache.load_rgba(
            sprite_info[
                "sprite_path"
            ]
        )
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

    (
        warped_rgba,
        resize_info,
    ) = warp_view_matrix_sprite_to_box_subpixel(
        sprite_rgba=sprite_rgba,

        frame_w=width,
        frame_h=height,

        target_cx=
            box[
                "cx"
            ],

        target_bottom_y=
            render_bottom_y,

        target_box_w=
            target_box_w,

        target_box_h=
            target_box_h,

        anchor_x=
            sprite_info[
                "anchor_x"
            ],

        anchor_y=
            sprite_info[
                "anchor_y"
            ],

        alpha_threshold=
            warp_alpha_threshold,
    )

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

    rendered = bool(
        np.any(
            full_mask > 0
        )
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