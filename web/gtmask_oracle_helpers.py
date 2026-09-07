"""
gtmask_oracle_helpers.py

GT-mask placement helpers for the HE forward-pass experiment.

Critical rule
-------------
CARLA GT mask NEVER chooses:
    - angle
    - source distance
    - source elevation

Those are selected by the CURRENT production function:
    select_view_matrix_sprite(...)

The GT mask is used only after sprite selection to fit:
    center_x
    bottom_y
    target_width
    target_height

The fixed selected sprite is rendered into the full camera frame, so
clipping at image boundaries happens naturally.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


def camera_relative_state(
    actor_tf,
    camera_tf,
    viewpoint_lateral_sign=1.0,
):
    """
    Build the same state consumed by the production 4320 selector.

    CARLA camera local axes:
        +x = forward
        +y = right
        +z = up
    """
    world_to_camera = np.asarray(
        camera_tf.get_inverse_matrix(),
        dtype=np.float64,
    )

    actor_world = np.asarray(
        [
            float(actor_tf.location.x),
            float(actor_tf.location.y),
            float(actor_tf.location.z),
            1.0,
        ],
        dtype=np.float64,
    )

    pc = world_to_camera @ actor_world

    actor_relative_yaw = (
        float(actor_tf.rotation.yaw)
        - float(camera_tf.rotation.yaw)
        + 180.0
    ) % 360.0 - 180.0

    return {
        "x_m": (
            float(viewpoint_lateral_sign)
            * float(pc[1])
        ),
        "y_m": float(pc[2]),
        "z_m": float(pc[0]),
        "yaw_deg": float(actor_relative_yaw),
    }


def select_sprite_with_production_logic(
    renderer,
    actor_tf,
    camera_tf,
    sprite_bank,
    view_matrix,
    viewpoint_lateral_sign=1.0,
):
    """
    Use the copied CURRENT production selector exactly.

    No GT information is passed to this function.
    """
    state = camera_relative_state(
        actor_tf=actor_tf,
        camera_tf=camera_tf,
        viewpoint_lateral_sign=viewpoint_lateral_sign,
    )

    sprite_info = renderer.select_view_matrix_sprite(
        state=state,
        sprite_bank=sprite_bank,
        view_matrix=view_matrix,
    )

    return state, sprite_info


def crop_rgba_to_alpha(
    rgba,
    alpha_threshold=10,
):
    alpha = rgba[:, :, 3]
    ys, xs = np.where(
        alpha > int(alpha_threshold)
    )

    if len(xs) == 0:
        raise RuntimeError(
            "Selected sprite has empty alpha."
        )

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    return rgba[
        y1:y2,
        x1:x2,
    ].copy()


def bbox_from_mask(mask):
    ys, xs = np.where(
        np.asarray(mask) > 0
    )

    if len(xs) == 0:
        return None

    return {
        "x1": int(xs.min()),
        "y1": int(ys.min()),
        "x2": int(xs.max()),
        "y2": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def border_flags(mask):
    m = np.asarray(mask) > 0
    h, w = m.shape[:2]

    return {
        "left": bool(np.any(m[:, 0])),
        "right": bool(np.any(m[:, w - 1])),
        "top": bool(np.any(m[0, :])),
        "bottom": bool(np.any(m[h - 1, :])),
    }


def resize_and_place_mask(
    source_mask,
    center_x,
    bottom_y,
    target_width,
    target_height,
    frame_width,
    frame_height,
):
    """
    Resize the full source silhouette to the proposed full-object box and
    then clip by the target image bounds.

    This is the key distinction from the old preview: a clipped visible
    bbox is NOT treated as the full object bbox.
    """
    tw = max(2, int(round(float(target_width))))
    th = max(2, int(round(float(target_height))))

    resized = cv2.resize(
        (source_mask > 0).astype(np.uint8),
        (tw, th),
        interpolation=cv2.INTER_NEAREST,
    )

    # center_x is the full sprite horizontal center.
    # bottom_y is the full sprite bottom.
    x1 = int(round(float(center_x) - 0.5 * tw))
    y1 = int(round(float(bottom_y) - th))
    x2 = x1 + tw
    y2 = y1 + th

    canvas = np.zeros(
        (int(frame_height), int(frame_width)),
        dtype=np.uint8,
    )

    fx1 = max(0, x1)
    fy1 = max(0, y1)
    fx2 = min(int(frame_width), x2)
    fy2 = min(int(frame_height), y2)

    if fx2 <= fx1 or fy2 <= fy1:
        return canvas

    sx1 = fx1 - x1
    sy1 = fy1 - y1
    sx2 = sx1 + (fx2 - fx1)
    sy2 = sy1 + (fy2 - fy1)

    canvas[
        fy1:fy2,
        fx1:fx2,
    ] = resized[
        sy1:sy2,
        sx1:sx2,
    ]

    return canvas


def resize_and_place_rgba(
    source_rgba,
    center_x,
    bottom_y,
    target_width,
    target_height,
    frame_width,
    frame_height,
):
    tw = max(2, int(round(float(target_width))))
    th = max(2, int(round(float(target_height))))

    resized = cv2.resize(
        source_rgba,
        (tw, th),
        interpolation=cv2.INTER_LINEAR,
    )

    x1 = int(round(float(center_x) - 0.5 * tw))
    y1 = int(round(float(bottom_y) - th))
    x2 = x1 + tw
    y2 = y1 + th

    canvas = np.zeros(
        (
            int(frame_height),
            int(frame_width),
            4,
        ),
        dtype=np.uint8,
    )

    fx1 = max(0, x1)
    fy1 = max(0, y1)
    fx2 = min(int(frame_width), x2)
    fy2 = min(int(frame_height), y2)

    if fx2 <= fx1 or fy2 <= fy1:
        return canvas

    sx1 = fx1 - x1
    sy1 = fy1 - y1
    sx2 = sx1 + (fx2 - fx1)
    sy2 = sy1 + (fy2 - fy1)

    canvas[
        fy1:fy2,
        fx1:fx2,
    ] = resized[
        sy1:sy2,
        sx1:sx2,
    ]

    return canvas


def mask_iou(a, b):
    aa = np.asarray(a) > 0
    bb = np.asarray(b) > 0

    inter = int(
        np.count_nonzero(
            aa & bb
        )
    )

    union = int(
        np.count_nonzero(
            aa | bb
        )
    )

    if union == 0:
        return 1.0

    return float(inter) / float(union)


def mask_dice(a, b):
    aa = np.asarray(a) > 0
    bb = np.asarray(b) > 0

    inter = int(
        np.count_nonzero(
            aa & bb
        )
    )

    total = int(
        np.count_nonzero(aa)
        + np.count_nonzero(bb)
    )

    if total == 0:
        return 1.0

    return (
        2.0
        * float(inter)
        / float(total)
    )


def initial_box_from_gt(
    target_mask,
    previous_fit=None,
):
    """
    Fully visible:
        initialize directly from the true visible bbox.

    Clipped:
        use visible edges + previous full-object box as initialization.
    """
    bbox = bbox_from_mask(
        target_mask
    )

    if bbox is None:
        return None

    border = border_flags(
        target_mask
    )

    if not any(
        border.values()
    ):
        return {
            "center_x":
                0.5
                * (
                    float(bbox["x1"])
                    + float(bbox["x2"])
                ),
            "bottom_y":
                float(bbox["y2"]),
            "target_width":
                float(bbox["width"]),
            "target_height":
                float(bbox["height"]),
            "bbox": bbox,
            "border": border,
        }

    if previous_fit is None:
        # First clipped frame without history. This is only a seed;
        # coordinate descent below can expand the off-screen box.
        width = float(
            bbox["width"]
        )
        height = float(
            bbox["height"]
        )
    else:
        # Object is approaching during this pass-by, so use a modest
        # extrapolated full size while never shrinking below visible extent.
        width = max(
            float(bbox["width"]),
            float(
                previous_fit[
                    "target_width"
                ]
            ) * 1.04,
        )

        height = max(
            float(bbox["height"]),
            float(
                previous_fit[
                    "target_height"
                ]
            ) * 1.04,
        )

    if (
        border["left"]
        and not border["right"]
    ):
        center_x = (
            float(bbox["x2"])
            - 0.5 * width
        )

    elif (
        border["right"]
        and not border["left"]
    ):
        center_x = (
            float(bbox["x1"])
            + 0.5 * width
        )

    elif previous_fit is not None:
        center_x = float(
            previous_fit[
                "center_x"
            ]
        )

    else:
        center_x = (
            0.5
            * (
                float(bbox["x1"])
                + float(bbox["x2"])
            )
        )

    if (
        border["bottom"]
        and not border["top"]
    ):
        # top edge visible, bottom clipped
        bottom_y = (
            float(bbox["y1"])
            + height
        )

    elif (
        border["top"]
        and not border["bottom"]
    ):
        # bottom edge visible
        bottom_y = float(
            bbox["y2"]
        )

    elif previous_fit is not None:
        bottom_y = float(
            previous_fit[
                "bottom_y"
            ]
        )

    else:
        bottom_y = float(
            bbox["y2"]
        )

    return {
        "center_x":
            float(center_x),
        "bottom_y":
            float(bottom_y),
        "target_width":
            float(width),
        "target_height":
            float(height),
        "bbox":
            bbox,
        "border":
            border,
    }


def continuity_penalty(
    params,
    previous_fit,
):
    if previous_fit is None:
        return 0.0

    cx, by, tw, th = [
        float(v)
        for v in params
    ]

    pw = max(
        1.0,
        float(
            previous_fit[
                "target_width"
            ]
        ),
    )

    ph = max(
        1.0,
        float(
            previous_fit[
                "target_height"
            ]
        ),
    )

    return float(
        abs(
            cx
            - float(
                previous_fit[
                    "center_x"
                ]
            )
        ) / pw
        +
        abs(
            by
            - float(
                previous_fit[
                    "bottom_y"
                ]
            )
        ) / ph
        +
        abs(
            math.log(
                max(2.0, tw)
                / pw
            )
        )
        +
        abs(
            math.log(
                max(2.0, th)
                / ph
            )
        )
    )


def evaluate_box(
    params,
    source_mask,
    target_mask,
    previous_fit,
    continuity_weight,
):
    cx, by, tw, th = [
        float(v)
        for v in params
    ]

    if (
        tw < 2.0
        or th < 2.0
        or tw > 10000.0
        or th > 10000.0
    ):
        return (
            1e9,
            0.0,
            0.0,
            None,
        )

    frame_h, frame_w = (
        target_mask.shape[:2]
    )

    candidate = (
        resize_and_place_mask(
            source_mask=source_mask,
            center_x=cx,
            bottom_y=by,
            target_width=tw,
            target_height=th,
            frame_width=frame_w,
            frame_height=frame_h,
        )
    )

    score_iou = mask_iou(
        candidate,
        target_mask,
    )

    score_dice = mask_dice(
        candidate,
        target_mask,
    )

    loss = (
        1.0
        - score_iou
        + float(
            continuity_weight
        )
        * continuity_penalty(
            params,
            previous_fit,
        )
    )

    return (
        float(loss),
        float(score_iou),
        float(score_dice),
        candidate,
    )


def fit_fixed_sprite_to_gt_mask(
    selected_rgba,
    target_mask,
    previous_fit=None,
    continuity_weight=0.015,
):
    """
    Fit x/y/width/height for ONE already-selected sprite.

    This score cannot change sprite identity.
    """
    source_rgba = crop_rgba_to_alpha(
        selected_rgba,
        alpha_threshold=10,
    )

    source_mask = (
        source_rgba[:, :, 3]
        > 10
    ).astype(np.uint8)

    target_mask = (
        np.asarray(target_mask)
        > 0
    ).astype(np.uint8)

    init = initial_box_from_gt(
        target_mask,
        previous_fit=previous_fit,
    )

    if init is None:
        return {
            "success": False,
            "reason": "empty_target_mask",
        }

    params = np.asarray(
        [
            init["center_x"],
            init["bottom_y"],
            max(
                2.0,
                init[
                    "target_width"
                ],
            ),
            max(
                2.0,
                init[
                    "target_height"
                ],
            ),
        ],
        dtype=np.float64,
    )

    (
        best_loss,
        best_iou,
        best_dice,
        best_mask,
    ) = evaluate_box(
        params=params,
        source_mask=source_mask,
        target_mask=target_mask,
        previous_fit=previous_fit,
        continuity_weight=continuity_weight,
    )

    # Coarse-to-fine coordinate descent.
    #
    # We deliberately optimize only 4 box variables.
    # Angle/distance/elevation remain frozen.
    for fraction in (
        0.12,
        0.06,
        0.03,
        0.015,
        0.0075,
    ):
        steps = np.asarray(
            [
                max(
                    1.0,
                    params[2]
                    * fraction,
                ),
                max(
                    1.0,
                    params[3]
                    * fraction,
                ),
                max(
                    1.0,
                    params[2]
                    * fraction,
                ),
                max(
                    1.0,
                    params[3]
                    * fraction,
                ),
            ],
            dtype=np.float64,
        )

        for _round in range(3):
            improved = False

            for axis in range(4):
                for sign in (
                    -1.0,
                    +1.0,
                ):
                    trial = (
                        params.copy()
                    )

                    trial[
                        axis
                    ] += (
                        sign
                        * steps[
                            axis
                        ]
                    )

                    (
                        loss,
                        score_iou,
                        score_dice,
                        candidate_mask,
                    ) = evaluate_box(
                        params=trial,
                        source_mask=source_mask,
                        target_mask=target_mask,
                        previous_fit=previous_fit,
                        continuity_weight=continuity_weight,
                    )

                    if (
                        loss
                        + 1e-9
                        <
                        best_loss
                    ):
                        params = trial
                        best_loss = loss
                        best_iou = score_iou
                        best_dice = score_dice
                        best_mask = candidate_mask
                        improved = True

            if not improved:
                break

    frame_h, frame_w = (
        target_mask.shape[:2]
    )

    warped_rgba = (
        resize_and_place_rgba(
            source_rgba=source_rgba,
            center_x=params[0],
            bottom_y=params[1],
            target_width=params[2],
            target_height=params[3],
            frame_width=frame_w,
            frame_height=frame_h,
        )
    )

    return {
        "success": True,
        "reason": "ok",
        "center_x": float(
            params[0]
        ),
        "bottom_y": float(
            params[1]
        ),
        "target_width": float(
            params[2]
        ),
        "target_height": float(
            params[3]
        ),
        "fit_iou": float(
            best_iou
        ),
        "fit_dice": float(
            best_dice
        ),
        "fit_loss": float(
            best_loss
        ),
        "border": init[
            "border"
        ],
        "visible_bbox": init[
            "bbox"
        ],
        "fitted_mask": best_mask,
        "warped_rgba": warped_rgba,
    }


def force_exact_gt_alpha(
    fitted_rgba,
    target_mask,
):
    """
    Strongest qualitative oracle:
        keep selected sprite RGB,
        force final alpha to exact CARLA target mask.

    This is visualization only.
    """
    out = (
        fitted_rgba.copy()
    )

    target = (
        np.asarray(
            target_mask
        )
        > 0
    ).astype(np.uint8)

    source_alpha = (
        out[:, :, 3]
    )

    missing_inside_gt = (
        (target > 0)
        &
        (
            source_alpha
            < 16
        )
    ).astype(np.uint8) * 255

    if np.any(
        missing_inside_gt
    ):
        rgb = (
            out[:, :, :3]
            .copy()
        )

        # Fill only occasional uncovered GT pixels from nearby sprite RGB.
        rgb = cv2.inpaint(
            rgb,
            missing_inside_gt,
            3.0,
            cv2.INPAINT_TELEA,
        )

        out[
            :,
            :,
            :3,
        ] = rgb

    out[
        :,
        :,
        3,
    ] = (
        target
        * 255
    ).astype(np.uint8)

    return out


def alpha_composite_rgb(
    background_rgb,
    foreground_rgba,
):
    fg = (
        foreground_rgba[
            :,
            :,
            :3,
        ]
        .astype(
            np.float32
        )
    )

    alpha = (
        foreground_rgba[
            :,
            :,
            3:4,
        ]
        .astype(
            np.float32
        )
        / 255.0
    )

    bg = (
        background_rgb
        .astype(
            np.float32
        )
    )

    out = (
        fg
        * alpha
        +
        bg
        * (
            1.0
            - alpha
        )
    )

    return np.clip(
        out,
        0.0,
        255.0,
    ).astype(np.uint8)
