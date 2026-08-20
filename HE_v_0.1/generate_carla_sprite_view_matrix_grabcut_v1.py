"""
generate_carla_sprite_view_matrix_grabcut_v1.py

Diagnostic / candidate production sprite generator.

Goal
----
Recover the clean appearance of the original HE sprites while
supporting the new camera-independent asset coordinates:

    azimuth
    distance
    elevation

Key design
----------
1. Vehicle is moved high above the CARLA map.
2. Physics is disabled.
3. Camera is positioned relative to the vehicle.
4. CARLA map geometry is removed before sprite capture.
5. No background subtraction.
6. Projected CARLA 3-D bbox defines the segmentation ROI.
7. CARLA semantic pixels are used only as trusted foreground seeds.
8. GrabCut determines the final visible RGB silhouette.
9. Only upper-body enclosed holes may be filled.
10. Underbody / wheel gaps remain transparent.

Default diagnostic:
    angle      = 293 deg
    distances  = 5, 10, 20 m
    elevations = 0, 10, 20 deg

Run from:
    D:\\HallucinationEngine\\HE_v_0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import random
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import carla

import generate_carla_360_rgba_fixed_camera as gen


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted generation run. "
            "Views already present in view_matrix.csv are skipped."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help=(
            "Deterministic RNG seed used by Python, NumPy, "
            "and OpenCV during sprite generation."
        ),
    )

    parser.add_argument(
        "--required-map",
        default="Town10HD_Opt",
        help=(
            "Exact CARLA map required for production generation."
        ),
    )
    parser.add_argument(
        "--vehicle-blueprint",
        default="vehicle.tesla.model3",
    )

    parser.add_argument(
        "--vehicle-name",
        default="tesla_grabcut_view_matrix",
    )

    parser.add_argument(
        "--color",
        default="0,0,255",
    )

    parser.add_argument(
        "--output-root",
        default="assets/sprite_bank_grabcut_test",
    )

    parser.add_argument(
        "--spawn-index",
        type=int,
        default=10,
    )

    # --------------------------------------------------------
    # Asset coordinates
    # --------------------------------------------------------

    parser.add_argument(
        "--angles",
        type=int,
        nargs="+",
        default=[
            293,
        ],
    )
    parser.add_argument(
        "--all-angles",
        action="store_true",
        help=(
            "Generate every integer azimuth from 0 through 359. "
            "Overrides --angles."
        ),
    )

    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help=(
            "Generate the complete production view matrix in one run: "
            "360 angles x distances [5,10,20] m x "
            "elevations [0,5,10,20] deg. "
            "Also disables contact-sheet generation."
        ),
    )

    parser.add_argument(
        "--skip-contact-sheets",
        action="store_true",
        help=(
            "Do not create the per-angle distance/elevation "
            "contact sheets. Useful for 360-degree QA runs."
        ),
    )
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=[
            5.0,
            10.0,
            20.0,
        ],
    )

    parser.add_argument(
        "--elevations",
        type=float,
        nargs="+",
        default=[
            0.0,
            5.0,
            10.0,
            20.0,
        ],
    )

    parser.add_argument(
        "--target-height",
        type=float,
        default=0.75,
        help=(
            "Vehicle-local height around which the viewpoint "
            "sphere is constructed."
        ),
    )

    # --------------------------------------------------------
    # High-altitude capture
    # --------------------------------------------------------

    parser.add_argument(
        "--capture-altitude",
        type=float,
        default=80.0,
        help=(
            "Absolute CARLA world Z used for the vehicle. "
            "This separates the asset from road/buildings."
        ),
    )

    # --------------------------------------------------------
    # Capture camera
    # --------------------------------------------------------

    parser.add_argument(
        "--image-width",
        type=int,
        default=1920,
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=1080,
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=100.0,
    )

    # --------------------------------------------------------
    # GrabCut
    # --------------------------------------------------------

    parser.add_argument(
        "--vehicle-semantic-tag",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--grabcut-iterations",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--bbox-expand-ratio",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--crop-margin-px",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--upper-hole-fraction",
        type=float,
        default=0.78,
        help=(
            "Only holes above this fraction of actor height "
            "may be filled. Prevents underbody/road filling."
        ),
    )

    # --------------------------------------------------------
    # Window handling
    # --------------------------------------------------------

    parser.add_argument(
        "--window-mode",
        choices=[
            "black",
            "darken",
            "keep",
        ],
        default="black",
    )

    parser.add_argument(
        "--window-black-value",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--window-darken-factor",
        type=float,
        default=0.25,
    )

    # Compatibility with imported generator helpers.
    parser.add_argument(
        "--fixed-delta-seconds",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--settle-ticks",
        type=int,
        default=2,
    )

    return parser.parse_args()


# ============================================================
# Viewpoint geometry
# ============================================================

def build_view_camera_transform(
    base_location,
    base_yaw_deg,
    distance_m,
    elevation_deg,
    target_height_m,
):
    """
    Build camera pose from generic relative viewpoint coordinates.

    distance_m:
        Euclidean camera-to-target distance.

    elevation_deg:
        Vertical viewing elevation above target centre.

    The horizontal azimuth dimension is produced by rotating the
    actor, preserving the existing HE sprite angle convention.
    """

    elevation_rad = math.radians(
        float(
            elevation_deg
        )
    )

    yaw_rad = math.radians(
        float(
            base_yaw_deg
        )
    )

    horizontal_distance = (
        float(
            distance_m
        )
        *
        math.cos(
            elevation_rad
        )
    )

    vertical_distance = (
        float(
            distance_m
        )
        *
        math.sin(
            elevation_rad
        )
    )

    forward_x = math.cos(
        yaw_rad
    )

    forward_y = math.sin(
        yaw_rad
    )

    target_location = carla.Location(
        x=float(
            base_location.x
        ),

        y=float(
            base_location.y
        ),

        z=(
            float(
                base_location.z
            )
            +
            float(
                target_height_m
            )
        ),
    )

    camera_location = carla.Location(
        x=(
            float(
                target_location.x
            )
            +
            horizontal_distance
            *
            forward_x
        ),

        y=(
            float(
                target_location.y
            )
            +
            horizontal_distance
            *
            forward_y
        ),

        z=(
            float(
                target_location.z
            )
            +
            vertical_distance
        ),
    )

    camera_rotation = gen.look_at_rotation(
        camera_location,
        target_location,
    )

    return carla.Transform(
        camera_location,
        camera_rotation,
    )


# ============================================================
# Mask utilities
# ============================================================

def fill_holes_binary(
    mask_u8,
):
    """
    Fill enclosed holes in a binary mask.

    Used only temporarily. The caller decides WHICH newly-filled
    holes are allowed into the final silhouette.
    """

    binary = (
        mask_u8 > 0
    ).astype(
        np.uint8
    ) * 255

    h, w = binary.shape[:2]

    flood = binary.copy()

    flood_mask = np.zeros(
        (
            h + 2,
            w + 2,
        ),
        dtype=np.uint8,
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (
            0,
            0,
        ),
        255,
    )

    flood_inv = cv2.bitwise_not(
        flood
    )

    return cv2.bitwise_or(
        binary,
        flood_inv,
    )


def keep_useful_components(
    mask_u8,
    semantic_seed_u8,
):
    """
    Retain only foreground components belonging to the actor.

    GrabCut can occasionally classify a disconnected background
    region as foreground, especially at high camera elevation.

    CARLA semantic vehicle pixels provide the trusted actor seed.

    A small adaptive dilation is used so legitimate thin actor
    structures such as mirrors remain eligible even if they are
    separated from the semantic seed by a few raster pixels.

    IMPORTANT:
        There is deliberately NO area-only fallback here.
        A large disconnected component is still background.
    """

    binary = (
        mask_u8 > 0
    ).astype(
        np.uint8
    )

    (
        count,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count <= 1:

        return (
            binary
            *
            255
        ).astype(
            np.uint8
        )

    seed_binary = (
        semantic_seed_u8 > 0
    ).astype(
        np.uint8
    ) * 255

    seed_ys, seed_xs = np.where(
        seed_binary > 0
    )

    if (
        len(seed_xs) == 0
        or
        len(seed_ys) == 0
    ):
        raise RuntimeError(
            "Semantic vehicle seed is empty."
        )

    seed_height = max(
        1,
        int(
            seed_ys.max()
            -
            seed_ys.min()
            +
            1
        ),
    )

    # Small scale-aware tolerance.
    radius = int(
        round(
            seed_height
            *
            0.015
        )
    )

    radius = max(
        2,
        min(
            radius,
            7,
        ),
    )

    kernel_size = (
        2
        *
        radius
        +
        1
    )

    seed_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    expanded_seed = cv2.dilate(
        seed_binary,
        seed_kernel,
        iterations=1,
    ) > 0

    selected_labels = set()

    # --------------------------------------------------------
    # Keep components connected to / extremely near trusted
    # semantic vehicle pixels.
    # --------------------------------------------------------

    for label in range(
        1,
        count,
    ):

        component = (
            labels
            ==
            label
        )

        touches_actor = bool(
            np.any(
                component
                &
                expanded_seed
            )
        )

        if touches_actor:

            selected_labels.add(
                int(
                    label
                )
            )

    # --------------------------------------------------------
    # Safety fallback:
    #
    # The primary vehicle component should always intersect
    # the semantic seed. If CARLA/GrabCut produces an unusual
    # raster case, choose the component with maximum semantic
    # overlap rather than choosing by size.
    # --------------------------------------------------------

    if not selected_labels:

        best_label = None
        best_overlap = 0

        seed_bool = (
            semantic_seed_u8 > 0
        )

        for label in range(
            1,
            count,
        ):

            overlap = int(
                np.sum(
                    (
                        labels
                        ==
                        label
                    )
                    &
                    seed_bool
                )
            )

            if overlap > best_overlap:

                best_overlap = overlap
                best_label = label

        if best_label is None:

            raise RuntimeError(
                "Could not identify actor component."
            )

        selected_labels.add(
            int(
                best_label
            )
        )

    out = np.zeros_like(
        mask_u8,
        dtype=np.uint8,
    )

    for label in selected_labels:

        out[
            labels
            ==
            label
        ] = 255

    return out


# ============================================================
# Semantic-seeded GrabCut
# ============================================================

def make_seeded_grabcut_alpha(
    rgb,
    tags,
    bbox,
    args,
):
    """
    Extract actor appearance using RGB GrabCut.

    Projected 3-D bbox:
        defines where the actor is expected.

    Semantic CARLA pixels:
        become definite foreground seeds.

    RGB GrabCut:
        recovers glass, windows, roof, mirrors and antialiased
        appearance that semantic labels may not represent well.
    """

    image_h, image_w = (
        rgb.shape[
            :2
        ]
    )

    (
        bx1,
        by1,
        bx2,
        by2,
    ) = [
        int(v)
        for v in bbox
    ]

    bw = max(
        1,
        bx2 - bx1 + 1,
    )

    bh = max(
        1,
        by2 - by1 + 1,
    )

    mx = (
        int(
            round(
                bw
                *
                float(
                    args.bbox_expand_ratio
                )
            )
        )
        +
        int(
            args.crop_margin_px
        )
    )

    my = (
        int(
            round(
                bh
                *
                float(
                    args.bbox_expand_ratio
                )
            )
        )
        +
        int(
            args.crop_margin_px
        )
    )

    x1e = max(
        0,
        bx1 - mx,
    )

    y1e = max(
        0,
        by1 - my,
    )

    x2e = min(
        image_w - 1,
        bx2 + mx,
    )

    y2e = min(
        image_h - 1,
        by2 + my,
    )

    crop_rgb = rgb[
        y1e:
        y2e + 1,

        x1e:
        x2e + 1,
    ].copy()

    crop_tags = tags[
        y1e:
        y2e + 1,

        x1e:
        x2e + 1,
    ].copy()

    crop_h, crop_w = (
        crop_rgb.shape[
            :2
        ]
    )

    if (
        crop_h < 5
        or
        crop_w < 5
    ):

        raise RuntimeError(
            "GrabCut crop is too small."
        )

    # ========================================================
    # GrabCut initialization
    # ========================================================

    gc_mask = np.full(
        (
            crop_h,
            crop_w,
        ),
        cv2.GC_BGD,
        dtype=np.uint8,
    )

    local_bx1 = max(
        0,
        bx1 - x1e,
    )

    local_by1 = max(
        0,
        by1 - y1e,
    )

    local_bx2 = min(
        crop_w - 1,
        bx2 - x1e,
    )

    local_by2 = min(
        crop_h - 1,
        by2 - y1e,
    )

    # Everything inside projected 3-D bbox is possible actor.
    gc_mask[
        local_by1:
        local_by2 + 1,

        local_bx1:
        local_bx2 + 1,
    ] = cv2.GC_PR_FGD

    semantic_seed = (
        crop_tags
        ==
        int(
            args.vehicle_semantic_tag
        )
    )

    # Trusted CARLA vehicle pixels.
    gc_mask[
        semantic_seed
    ] = cv2.GC_FGD

    # A very small dilation strengthens thin semantic seeds,
    # without defining the final mask.
    semantic_seed_u8 = (
        semantic_seed.astype(
            np.uint8
        )
        *
        255
    )

    seed_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            3,
            3,
        ),
    )

    semantic_near = cv2.dilate(
        semantic_seed_u8,
        seed_kernel,
        iterations=1,
    ) > 0

    probable_seed = (
        semantic_near
        &
        ~semantic_seed
    )

    gc_mask[
        probable_seed
    ] = cv2.GC_PR_FGD

    # Strong crop border = definite background.
    border = max(
        3,
        int(
            round(
                min(
                    crop_h,
                    crop_w,
                )
                *
                0.015
            )
        ),
    )

    gc_mask[
        :border,
        :
    ] = cv2.GC_BGD

    gc_mask[
        -border:,
        :
    ] = cv2.GC_BGD

    gc_mask[
        :,
        :border
    ] = cv2.GC_BGD

    gc_mask[
        :,
        -border:
    ] = cv2.GC_BGD

    # ========================================================
    # GrabCut
    # ========================================================

    bgd_model = np.zeros(
        (
            1,
            65,
        ),
        dtype=np.float64,
    )

    fgd_model = np.zeros(
        (
            1,
            65,
        ),
        dtype=np.float64,
    )

    crop_bgr = cv2.cvtColor(
        crop_rgb,
        cv2.COLOR_RGB2BGR,
    )

    cv2.grabCut(
        crop_bgr,
        gc_mask,
        None,
        bgd_model,
        fgd_model,
        int(
            args.grabcut_iterations
        ),
        cv2.GC_INIT_WITH_MASK,
    )

    foreground = np.where(
        (
            gc_mask
            ==
            cv2.GC_FGD
        )
        |
        (
            gc_mask
            ==
            cv2.GC_PR_FGD
        ),
        255,
        0,
    ).astype(
        np.uint8
    )

    # ========================================================
    # Minimal cleanup
    # ========================================================

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            3,
            3,
        ),
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    foreground = keep_useful_components(
        foreground,
        semantic_seed_u8,
    )

    # ========================================================
    # Constrain RGB-recovered foreground to the neighbourhood
    # of the trusted semantic actor mask.
    #
    # GrabCut can occasionally attach a thin background feature
    # to a wheel/body by only one or two pixels. Connected-
    # component filtering cannot remove it because it is then
    # technically part of the actor component.
    #
    # The semantic mask already contains the painted body and
    # wheels. GrabCut mainly needs freedom around its boundary
    # to recover:
    #   - mirrors
    #   - antialiasing
    #   - thin pillars
    #   - window / roof boundaries
    #
    # Enclosed glass interiors are recovered later by the
    # upper-body hole-filling stage.
    # ========================================================

    seed_ys, seed_xs = np.where(
        semantic_seed_u8 > 0
    )

    if (
        len(seed_xs) > 0
        and
        len(seed_ys) > 0
    ):

        seed_height = max(
            1,
            int(
                seed_ys.max()
                -
                seed_ys.min()
                +
                1
            ),
        )

        recovery_radius = int(
            round(
                seed_height
                *
                0.08
            )
        )

        recovery_radius = max(
            3,
            min(
                recovery_radius,
                24,
            ),
        )

        recovery_kernel_size = (
            2
            *
            recovery_radius
            +
            1
        )

        recovery_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    recovery_kernel_size,
                    recovery_kernel_size,
                ),
            )
        )

        semantic_envelope = cv2.dilate(
            semantic_seed_u8,
            recovery_kernel,
            iterations=1,
        )

        foreground = cv2.bitwise_and(
            foreground,
            semantic_envelope,
        )

    # ========================================================
    # Upper-body hole filling only
    # ========================================================

    ys, xs = np.where(
        foreground > 0
    )

    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):

        raise RuntimeError(
            "GrabCut produced empty foreground."
        )

    fy1 = int(
        ys.min()
    )

    fy2 = int(
        ys.max()
    )

    actor_h = max(
        1,
        fy2 - fy1 + 1,
    )

    upper_bottom = int(
        round(
            fy1
            +
            float(
                args.upper_hole_fraction
            )
            *
            actor_h
        )
    )

    upper_bottom = max(
        fy1,
        min(
            fy2,
            upper_bottom,
        ),
    )

    completely_filled = fill_holes_binary(
        foreground
    )

    new_holes = cv2.subtract(
        completely_filled,
        foreground,
    )

    upper_region = np.zeros_like(
        foreground,
        dtype=np.uint8,
    )

    upper_region[
        fy1:
        upper_bottom + 1,
        :
    ] = 255

    upper_holes = cv2.bitwise_and(
        new_holes,
        upper_region,
    )

    final_mask = cv2.bitwise_or(
        foreground,
        upper_holes,
    )

    # These newly-filled upper holes are the only pixels for
    # which we optionally replace the background colour.
    window_mask = (
        upper_holes.copy()
    )

    # Feather only the final boundary.
    alpha = cv2.GaussianBlur(
        final_mask,
        (
            3,
            3,
        ),
        0,
    )

    # ========================================================
    # Expand local result into full image
    # ========================================================

    full_alpha = np.zeros(
        (
            image_h,
            image_w,
        ),
        dtype=np.uint8,
    )

    full_window = np.zeros(
        (
            image_h,
            image_w,
        ),
        dtype=np.uint8,
    )

    full_body = np.zeros(
        (
            image_h,
            image_w,
        ),
        dtype=np.uint8,
    )

    full_alpha[
        y1e:
        y2e + 1,

        x1e:
        x2e + 1,
    ] = alpha

    full_window[
        y1e:
        y2e + 1,

        x1e:
        x2e + 1,
    ] = window_mask

    full_body[
        y1e:
        y2e + 1,

        x1e:
        x2e + 1,
    ] = foreground

    return (
        full_alpha,
        full_body,
        full_window,
    )


# ============================================================
# Contact sheet
# ============================================================

def safe_float_name(
    value,
):

    return (
        f"{float(value):05.1f}"
        .replace(
            ".",
            "p",
        )
    )
# ============================================================
# Production dataset bookkeeping
# ============================================================

def expected_view_paths(
    output_dir,
    angle,
    distance,
    elevation,
):
    """
    Return the three files that define one completed view.
    """

    angle = int(angle) % 360

    d_name = safe_float_name(
        distance
    )

    e_name = safe_float_name(
        elevation
    )

    stem = (
        f"d_{d_name}_"
        f"e_{e_name}_"
        f"angle_{angle:03d}"
    )

    output_dir = Path(
        output_dir
    )

    return {
        "rgba":
            output_dir
            /
            "rgba"
            /
            f"{stem}_rgba.png",

        "mask":
            output_dir
            /
            "mask"
            /
            f"{stem}_mask.png",

        "debug":
            output_dir
            /
            "debug"
            /
            f"{stem}_debug.png",
    }

def atomic_save_image(
    save_function,
    path,
    image,
):
    """
    Save an image to a temporary PNG and atomically replace
    the destination only after the write succeeds.
    """

    path = Path(
        path
    )

    temp_path = path.with_name(
        path.stem
        +
        ".tmp"
        +
        path.suffix
    )

    if temp_path.exists():
        temp_path.unlink()

    save_function(
        temp_path,
        image,
    )

    if not file_is_nonempty(
        temp_path
    ):

        raise RuntimeError(
            "Temporary image write failed: "
            f"{temp_path}"
        )

    os.replace(
        temp_path,
        path,
    )

def file_is_nonempty(
    path,
):
    path = Path(
        path
    )

    return (
        path.exists()
        and
        path.is_file()
        and
        path.stat().st_size > 0
    )


def build_generation_signature(
    args,
):
    """
    Configuration that must remain identical across --resume runs.
    """

    return {
        "schema_version":
            2,

        "vehicle_blueprint":
            str(
                args.vehicle_blueprint
            ),

        "color":
            str(
                args.color
            ),

        "required_map":
            str(
                args.required_map
            ),

        "image_width":
            int(
                args.image_width
            ),

        "image_height":
            int(
                args.image_height
            ),

        "fov_deg":
            float(
                args.fov
            ),

        "capture_altitude_m":
            float(
                args.capture_altitude
            ),

        "target_height_m":
            float(
                args.target_height
            ),

        "angles_deg": [
            int(v) % 360
            for v in args.angles
        ],

        "distances_m": [
            float(v)
            for v in args.distances
        ],

        "elevations_deg": [
            float(v)
            for v in args.elevations
        ],

        "vehicle_semantic_tag":
            int(
                args.vehicle_semantic_tag
            ),

        "grabcut_iterations":
            int(
                args.grabcut_iterations
            ),

        "bbox_expand_ratio":
            float(
                args.bbox_expand_ratio
            ),

        "crop_margin_px":
            int(
                args.crop_margin_px
            ),

        "upper_hole_fraction":
            float(
                args.upper_hole_fraction
            ),

        "window_mode":
            str(
                args.window_mode
            ),

        "window_black_value":
            int(
                args.window_black_value
            ),

        "window_darken_factor":
            float(
                args.window_darken_factor
            ),

        "fixed_delta_seconds":
            float(
                args.fixed_delta_seconds
            ),

        "settle_ticks":
            int(
                args.settle_ticks
            ),

        "seed":
            int(
                args.seed
            ),

        "weather":
            "ClearNoon",

        "clean_world":
            True,
    }


def atomic_write_json(
    path,
    data,
):
    path = Path(
        path
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

        f.flush()
        os.fsync(
            f.fileno()
        )

    temp_path.replace(
        path
    )


def output_has_existing_dataset(
    output_dir,
):
    output_dir = Path(
        output_dir
    )

    csv_path = (
        output_dir
        /
        "view_matrix.csv"
    )

    if csv_path.exists():
        return True

    for subdir in [
        "rgba",
        "mask",
        "debug",
    ]:

        folder = (
            output_dir
            /
            subdir
        )

        if (
            folder.exists()
            and
            any(
                folder.glob(
                    "*.png"
                )
            )
        ):
            return True

    return False


def initialize_or_validate_generation_config(
    output_dir,
    args,
):
    """
    Prevent different generation settings from being mixed into
    one sprite bank.
    """

    output_dir = Path(
        output_dir
    )

    config_path = (
        output_dir
        /
        "generation_config.json"
    )

    signature = (
        build_generation_signature(
            args
        )
    )

    dataset_exists = (
        output_has_existing_dataset(
            output_dir
        )
    )

    # Fresh generation must not silently reuse old sprites.
    if (
        not args.resume
        and
        dataset_exists
    ):

        raise RuntimeError(
            "Output directory already contains sprite-bank data:\n"
            f"  {output_dir}\n"
            "Delete it first or use --resume."
        )

    if config_path.exists():

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:

            existing = json.load(
                f
            )

        existing_signature = (
            existing.get(
                "generation"
            )
        )

        if (
            existing_signature
            !=
            signature
        ):

            raise RuntimeError(
                "Generation configuration does not match the "
                "existing sprite bank.\n\n"
                "Existing:\n"
                +
                json.dumps(
                    existing_signature,
                    indent=2,
                )
                +
                "\n\nRequested:\n"
                +
                json.dumps(
                    signature,
                    indent=2,
                )
            )

        print(
            "[GrabCutBank] generation_config.json matches."
        )

        return (
            config_path,
            existing,
        )

    # Existing sprites without a configuration fingerprint are
    # unsafe to resume as a production dataset.
    if (
        args.resume
        and
        dataset_exists
    ):

        raise RuntimeError(
            "Existing sprite-bank data was found but "
            "generation_config.json is missing. "
            "Delete the old output before starting the new "
            "production generation."
        )

    document = {
        "generation":
            signature,

        "runtime":
            None,
    }

    atomic_write_json(
        config_path,
        document,
    )

    print(
        "[GrabCutBank] Created:",
        config_path,
    )

    return (
        config_path,
        document,
    )


def validate_runtime_environment(
    config_path,
    config_document,
    client,
    world,
    args,
):
    """
    Validate CARLA map/version across resumed runs.
    """

    map_name = str(
        world.get_map().name
    )

    short_map_name = (
        map_name
        .replace("\\", "/")
        .split("/")[-1]
    )

    if (
        short_map_name
        !=
        str(
            args.required_map
        )
    ):

        raise RuntimeError(
            "Wrong CARLA map for sprite generation.\n"
            f"Required: {args.required_map}\n"
            f"Current : {short_map_name}"
        )

    runtime = {
        "map":
            short_map_name,

        "carla_client_version":
            str(
                client.get_client_version()
            ),

        "carla_server_version":
            str(
                client.get_server_version()
            ),
    }

    existing_runtime = (
        config_document.get(
            "runtime"
        )
    )

    if (
        existing_runtime is not None
        and
        existing_runtime != runtime
    ):

        raise RuntimeError(
            "CARLA runtime differs from the runtime used to "
            "start this sprite bank.\n\n"
            "Existing:\n"
            +
            json.dumps(
                existing_runtime,
                indent=2,
            )
            +
            "\n\nCurrent:\n"
            +
            json.dumps(
                runtime,
                indent=2,
            )
        )

    config_document[
        "runtime"
    ] = runtime

    atomic_write_json(
        config_path,
        config_document,
    )

    print(
        "[GrabCutBank] CARLA runtime validated:"
    )

    print(
        "[GrabCutBank]   map:",
        runtime["map"],
    )

    print(
        "[GrabCutBank]   client:",
        runtime[
            "carla_client_version"
        ],
    )

    print(
        "[GrabCutBank]   server:",
        runtime[
            "carla_server_version"
        ],
    )
def make_view_rng_seed(
    base_seed,
    angle,
    distance,
    elevation,
):
    """
    Stable deterministic RNG seed for one view.

    This makes a resumed generation produce the same sprite
    regardless of how many earlier views were skipped.
    """

    angle_i = int(
        angle
    ) % 360

    distance_mm = int(
        round(
            float(distance)
            *
            1000.0
        )
    )

    elevation_mdeg = int(
        round(
            float(elevation)
            *
            1000.0
        )
    )

    value = (
        int(base_seed)
        *
        1000003
        +
        angle_i
        *
        9176
        +
        distance_mm
        *
        131
        +
        elevation_mdeg
        *
        17
    )

    # OpenCV expects a normal signed integer seed.
    value = (
        value
        &
        0x7FFFFFFF
    )

    return int(
        value
    )

def make_view_key(
    angle,
    distance,
    elevation,
):
    """
    Stable identity for one sprite-bank view.
    """

    return (
        int(angle) % 360,
        round(float(distance), 6),
        round(float(elevation), 6),
    )


def load_checkpoint_records(
    csv_path,
    output_dir,
):
    """
    Load only genuinely completed views.

    A CSV row is considered complete only when its RGBA, mask,
    and debug PNG all exist and are non-empty.

    Duplicate CSV rows are collapsed by view key.
    """

    record_by_key = {}

    if not csv_path.exists():
        return [], set()

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            try:

                angle = int(
                    float(
                        row[
                            "angle_deg"
                        ]
                    )
                ) % 360

                distance = float(
                    row[
                        "distance_m"
                    ]
                )

                elevation = float(
                    row[
                        "elevation_deg"
                    ]
                )

                key = make_view_key(
                    angle,
                    distance,
                    elevation,
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):

                continue

            paths = expected_view_paths(
                output_dir=output_dir,
                angle=angle,
                distance=distance,
                elevation=elevation,
            )

            files_ok = all(
                file_is_nonempty(
                    path
                )
                for path
                in paths.values()
            )

            if not files_ok:

                print(
                    "[GrabCutBank] Resume: incomplete view "
                    "will be regenerated: "
                    f"a={angle:03d} "
                    f"d={distance:g} "
                    f"e={elevation:g}"
                )

                continue

            # Make the CSV path correct for the machine currently
            # performing generation.
            row[
                "rgba_path"
            ] = str(
                paths[
                    "rgba"
                ].resolve()
            )

            # If a crash created duplicate CSV rows, latest valid
            # row wins.
            record_by_key[
                key
            ] = row

    records = list(
        record_by_key.values()
    )

    completed_keys = set(
        record_by_key.keys()
    )

    return (
        records,
        completed_keys,
    )


def append_checkpoint_record(
    csv_path,
    record,
):
    """
    Append one completed sprite record immediately.

    This makes the CSV crash-safe: after each successful sprite,
    its metadata is already on disk.
    """

    write_header = (
        not csv_path.exists()
        or
        csv_path.stat().st_size == 0
    )

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                record.keys()
            ),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            record
        )

        # Force Python's userspace buffer to disk immediately.
        f.flush()

def atomic_write_records_csv(
    csv_path,
    records,
):
    """
    Atomically rewrite the canonical view_matrix.csv.
    """

    if not records:
        return

    csv_path = Path(
        csv_path
    )

    temp_path = csv_path.with_name(
        csv_path.stem
        +
        ".tmp"
        +
        csv_path.suffix
    )

    with open(
        temp_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                records[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            records
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        temp_path,
        csv_path,
    )


def validate_completed_dataset(
    output_dir,
    records,
    args,
):
    """
    Final production-bank integrity check.

    Returns records sorted deterministically by
    distance -> elevation -> angle.
    """

    expected_keys = set()

    for distance in args.distances:
        for elevation in args.elevations:
            for angle in args.angles:

                expected_keys.add(
                    make_view_key(
                        angle,
                        distance,
                        elevation,
                    )
                )

    record_by_key = {}

    for record in records:

        try:

            key = make_view_key(
                record[
                    "angle_deg"
                ],
                record[
                    "distance_m"
                ],
                record[
                    "elevation_deg"
                ],
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

        record_by_key[
            key
        ] = record

    actual_keys = set(
        record_by_key.keys()
    )

    missing_keys = sorted(
        expected_keys
        -
        actual_keys
    )

    extra_keys = sorted(
        actual_keys
        -
        expected_keys
    )

    missing_files = []

    for (
        angle,
        distance,
        elevation
    ) in sorted(
        expected_keys
    ):

        paths = expected_view_paths(
            output_dir=output_dir,
            angle=angle,
            distance=distance,
            elevation=elevation,
        )

        for kind, path in paths.items():

            if not file_is_nonempty(
                path
            ):

                missing_files.append(
                    (
                        angle,
                        distance,
                        elevation,
                        kind,
                        str(path),
                    )
                )

    print()
    print("=" * 80)
    print(
        "[GrabCutBank] FINAL DATASET VALIDATION"
    )
    print("=" * 80)

    print(
        "[GrabCutBank] expected views:",
        len(
            expected_keys
        ),
    )

    print(
        "[GrabCutBank] completed views:",
        len(
            actual_keys
        ),
    )

    print(
        "[GrabCutBank] missing views:",
        len(
            missing_keys
        ),
    )

    print(
        "[GrabCutBank] extra views:",
        len(
            extra_keys
        ),
    )

    print(
        "[GrabCutBank] missing files:",
        len(
            missing_files
        ),
    )

    if missing_keys:

        print(
            "[GrabCutBank] first missing views:",
            missing_keys[
                :20
            ],
        )

    if missing_files:

        print(
            "[GrabCutBank] first missing files:"
        )

        for item in missing_files[
            :20
        ]:

            print(
                "   ",
                item,
            )

    if (
        missing_keys
        or
        extra_keys
        or
        missing_files
    ):

        raise RuntimeError(
            "Sprite dataset validation failed. "
            "Run the same command again with --resume."
        )

    sorted_records = sorted(
        record_by_key.values(),
        key=lambda row: (
            float(
                row[
                    "distance_m"
                ]
            ),
            float(
                row[
                    "elevation_deg"
                ]
            ),
            int(
                float(
                    row[
                        "angle_deg"
                    ]
                )
            ),
        ),
    )

    print(
        "[GrabCutBank] DATASET VALIDATION PASSED"
    )
    print("=" * 80)

    return sorted_records

def make_contact_sheet(
    output_dir,
    records,
    angle,
    distances,
    elevations,
):

    cell_w = 520
    cell_h = 320
    label_h = 36

    sheet = Image.new(
        "RGB",
        (
            len(
                elevations
            )
            *
            cell_w,

            len(
                distances
            )
            *
            (
                cell_h
                +
                label_h
            ),
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    lookup = {}

    for record in records:

        if int(
            record[
                "angle_deg"
            ]
        ) != int(
            angle
        ):
            continue

        lookup[
            (
                float(
                    record[
                        "distance_m"
                    ]
                ),
                float(
                    record[
                        "elevation_deg"
                    ]
                ),
            )
        ] = record

    for r_idx, distance in enumerate(
        distances
    ):

        for c_idx, elevation in enumerate(
            elevations
        ):

            x0 = (
                c_idx
                *
                cell_w
            )

            y0 = (
                r_idx
                *
                (
                    cell_h
                    +
                    label_h
                )
            )

            label = (
                f"d={distance:g} m  "
                f"elev={elevation:g} deg"
            )

            draw.text(
                (
                    x0 + 8,
                    y0 + 8,
                ),
                label,
                fill="black",
            )

            record = lookup.get(
                (
                    float(
                        distance
                    ),
                    float(
                        elevation
                    ),
                )
            )

            if record is None:
                continue

            image = Image.open(
                record[
                    "rgba_path"
                ]
            ).convert(
                "RGBA"
            )

            white = Image.new(
                "RGBA",
                image.size,
                (
                    255,
                    255,
                    255,
                    255,
                ),
            )

            white.alpha_composite(
                image
            )

            thumb = ImageOps.contain(
                white.convert(
                    "RGB"
                ),
                (
                    cell_w - 20,
                    cell_h - 20,
                ),
            )

            px = (
                x0
                +
                (
                    cell_w
                    -
                    thumb.width
                )
                //
                2
            )

            py = (
                y0
                +
                label_h
                +
                (
                    cell_h
                    -
                    thumb.height
                )
                //
                2
            )

            sheet.paste(
                thumb,
                (
                    px,
                    py,
                ),
            )

    output_path = (
        output_dir
        /
        f"contact_sheet_angle_{int(angle):03d}.png"
    )

    sheet.save(
        output_path
    )

    print(
        "[GrabCutBank] Contact sheet:",
        output_path,
    )

# ============================================================
# Clean capture world
# ============================================================

def prepare_clean_capture_world(
    world,
    settle_ticks=8,
):
    """
    Prepare CARLA as a clean sprite-capture environment.

    The target vehicle is NOT spawned yet when this function runs.

    Strategy
    --------
    1. Require an *_Opt layered CARLA map.
    2. Hide all static EnvironmentObjects except Sky.
    3. Unload every optional CARLA map layer.
    4. Leave the world running and settle for several ticks.

    This removes buildings, vegetation, props, parked vehicles,
    road geometry, signs, poles, walls, etc. from the rendered
    capture environment while retaining a simple sky background.

    The operation is intentionally idempotent so --resume works
    when the generator is restarted against the same CARLA server.
    """

    map_name = str(
        world.get_map().name
    )

    short_map_name = (
        map_name
        .replace("\\", "/")
        .split("/")[-1]
    )

    print()
    print(
        "[GrabCutBank] Preparing clean capture world..."
    )

    print(
        "[GrabCutBank] map:",
        short_map_name,
    )

    # --------------------------------------------------------
    # We rely on CARLA layered-map control.
    # --------------------------------------------------------

    if "_Opt" not in short_map_name:

        raise RuntimeError(
            "Clean sprite capture requires a CARLA *_Opt map. "
            f"Current map is: {short_map_name}"
        )

    # --------------------------------------------------------
    # Hide all static environment geometry except the sky.
    #
    # This also handles objects belonging to CARLA's minimum
    # map layout that cannot be removed through MapLayer alone.
    # --------------------------------------------------------

    environment_objects = (
        world.get_environment_objects(
            carla.CityObjectLabel.Any
        )
    )

    hidden_object_ids = {
        int(obj.id)
        for obj in environment_objects
        if (
            obj.type
            !=
            carla.CityObjectLabel.Sky
        )
    }

    if hidden_object_ids:

        world.enable_environment_objects(
            hidden_object_ids,
            False,
        )

    print(
        "[GrabCutBank] hidden environment objects:",
        len(hidden_object_ids),
    )

    # --------------------------------------------------------
    # Remove all optional layered-map content:
    #
    # Buildings
    # Decals
    # Foliage
    # Ground
    # ParkedVehicles
    # Particles
    # Props
    # StreetLights
    # Walls
    # --------------------------------------------------------

    world.unload_map_layer(
        carla.MapLayer.All
    )

    print(
        "[GrabCutBank] optional map layers unloaded."
    )

    # Give Unreal a few synchronous frames to apply visibility
    # and streaming changes before spawning the target vehicle.
    for _ in range(
        max(
            1,
            int(settle_ticks),
        )
    ):
        world.tick()

    print(
        "[GrabCutBank] clean capture world ready."
    )
    print()

# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.full_dataset:

        args.angles = list(
            range(
                360
            )
        )

        args.distances = [
            5.0,
            10.0,
            20.0,
        ]

        args.elevations = [
            0.0,
            5.0,
            10.0,
            20.0,
        ]

        # 360 contact sheets are unnecessary for the
        # production generation run.
        args.skip_contact_sheets = True

    elif args.all_angles:

        args.angles = list(
            range(
                360
            )
        )
    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    cv2.setRNGSeed(
        int(
            args.seed
        )
    )

    print(
        "[GrabCutBank] RNG seed:",
        args.seed,
    )
    output_dir = (
        Path(
            args.output_root
        )
        /
        args.vehicle_name
    )

    rgba_dir = (
        output_dir
        /
        "rgba"
    )

    mask_dir = (
        output_dir
        /
        "mask"
    )

    debug_dir = (
        output_dir
        /
        "debug"
    )

    for folder in [
        rgba_dir,
        mask_dir,
        debug_dir,
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    world = None
    original_settings = None

    vehicle = None
    rgb_camera = None
    seg_camera = None

    rgb_queue = queue.Queue()
    seg_queue = queue.Queue()

    csv_path = (
        output_dir
        /
        "view_matrix.csv"
    )
    (
        generation_config_path,
        generation_config,
    ) = initialize_or_validate_generation_config(
        output_dir=output_dir,
        args=args,
    )

    records = []
    completed_keys = set()

    if args.resume:

        (
            records,
            completed_keys,
        ) = load_checkpoint_records(
            csv_path=csv_path,
            output_dir=output_dir,
        )

        print(
            "[GrabCutBank] Resume enabled:"
            f" {len(completed_keys)} completed views found."
        )

    else:

        # Preserve the original fresh-run behavior.
        # Existing metadata is discarded and regenerated.
        if csv_path.exists():
            csv_path.unlink()

    try:

        client = carla.Client(
            args.host,
            args.port,
        )

        client.set_timeout(
            float(
                args.timeout
            )
        )

        world = client.get_world()

        original_settings = (
            world.get_settings()
        )
        validate_runtime_environment(
            config_path=(
                generation_config_path
            ),
            config_document=(
                generation_config
            ),
            client=client,
            world=world,
            args=args,
        )
        print()
        print("=" * 80)
        print(
            "HE HIGH-ALTITUDE SEMANTIC-SEEDED GRABCUT TEST"
        )
        print("=" * 80)

        print(
            "[GrabCutBank] map:",
            world.get_map().name,
        )

        print(
            "[GrabCutBank] altitude:",
            args.capture_altitude,
        )

        print(
            "[GrabCutBank] angles:",
            args.angles,
        )

        print(
            "[GrabCutBank] distances:",
            args.distances,
        )

        print(
            "[GrabCutBank] elevations:",
            args.elevations,
        )

        gen.setup_synchronous_mode(
            world,
            args.fixed_delta_seconds,
        )
        world.set_weather(
            carla.WeatherParameters.ClearNoon
        )

        for _ in range(2):
            world.tick()

        print(
            "[GrabCutBank] weather: ClearNoon"
        )
        # Remove any vehicles / walkers / sensors left by an earlier run.
        gen.clear_existing_dynamic_actors(
            world
        )

        # Remove the CARLA town before spawning the sprite vehicle.
        prepare_clean_capture_world(
            world=world,
            settle_ticks=8,
        )

        blueprint_library = (
            world.get_blueprint_library()
        )

        vehicle, actual_bp = (
            gen.spawn_vehicle(
                world,
                blueprint_library,
                args,
            )
        )
        if (
            str(
                actual_bp
            )
            !=
            str(
                args.vehicle_blueprint
            )
        ):

            raise RuntimeError(
                "Wrong vehicle blueprint spawned.\n"
                f"Requested: {args.vehicle_blueprint}\n"
                f"Actual   : {actual_bp}"
            )

        actual_color = (
            vehicle.attributes.get(
                "color",
                None,
            )
        )

        print(
            "[GrabCutBank] vehicle blueprint:",
            actual_bp,
        )

        print(
            "[GrabCutBank] vehicle color:",
            actual_color,
        )

        if (
            actual_color is not None
            and
            str(
                actual_color
            )
            !=
            str(
                args.color
            )
        ):

            raise RuntimeError(
                "Vehicle color does not match requested color.\n"
                f"Requested: {args.color}\n"
                f"Actual   : {actual_color}"
            )
        vehicle.set_simulate_physics(
            False
        )

        spawn_tf = (
            vehicle.get_transform()
        )

        # ====================================================
        # Move complete capture setup high above the map
        # ====================================================

        base_location = carla.Location(
            x=float(
                spawn_tf.location.x
            ),

            y=float(
                spawn_tf.location.y
            ),

            z=float(
                args.capture_altitude
            ),
        )

        base_yaw = float(
            spawn_tf.rotation.yaw
        )

        vehicle.set_transform(
            carla.Transform(
                base_location,

                carla.Rotation(
                    pitch=0.0,
                    yaw=base_yaw,
                    roll=0.0,
                ),
            )
        )

        for _ in range(3):
            world.tick()

        first_camera_tf = (
            build_view_camera_transform(
                base_location=(
                    base_location
                ),

                base_yaw_deg=(
                    base_yaw
                ),

                distance_m=(
                    args.distances[
                        0
                    ]
                ),

                elevation_deg=(
                    args.elevations[
                        0
                    ]
                ),

                target_height_m=(
                    args.target_height
                ),
            )
        )

        rgb_camera, seg_camera = (
            gen.spawn_cameras(
                world=world,

                blueprint_library=(
                    blueprint_library
                ),

                camera_tf=(
                    first_camera_tf
                ),

                args=args,
            )
        )

        rgb_camera.listen(
            rgb_queue.put
        )

        seg_camera.listen(
            seg_queue.put
        )

        # Warm sensors.
        for _ in range(5):

            world.tick()

            gen.flush_queue(
                rgb_queue
            )

            gen.flush_queue(
                seg_queue
            )

        K = gen.build_camera_intrinsics(
            width=args.image_width,
            height=args.image_height,
            fov_degrees=args.fov,
        )

        total = (
            len(
                args.angles
            )
            *
            len(
                args.distances
            )
            *
            len(
                args.elevations
            )
        )

        capture_index = 0

        for distance_m in args.distances:

            for elevation_deg in args.elevations:

                camera_tf = (
                    build_view_camera_transform(
                        base_location=(
                            base_location
                        ),

                        base_yaw_deg=(
                            base_yaw
                        ),

                        distance_m=(
                            distance_m
                        ),

                        elevation_deg=(
                            elevation_deg
                        ),

                        target_height_m=(
                            args.target_height
                        ),
                    )
                )

                rgb_camera.set_transform(
                    camera_tf
                )

                seg_camera.set_transform(
                    camera_tf
                )

                gen.flush_queue(
                    rgb_queue
                )

                gen.flush_queue(
                    seg_queue
                )

                for _ in range(3):

                    world.tick()

                    gen.flush_queue(
                        rgb_queue
                    )

                    gen.flush_queue(
                        seg_queue
                    )

                for angle in args.angles:

                    capture_index += 1

                    angle = int(
                        angle
                    ) % 360

                    view_key = make_view_key(
                        angle,
                        distance_m,
                        elevation_deg,
                    )

                    if (
                        args.resume
                        and
                        view_key in completed_keys
                    ):

                        print(
                            "[GrabCutBank] "
                            f"{capture_index:04d}/{total:04d} "
                            f"SKIP "
                            f"a={angle:03d} "
                            f"d={distance_m:5.1f}m "
                            f"e={elevation_deg:5.1f}deg"
                        )

                        continue

                    yaw = gen.vehicle_yaw_for_angle(
                        base_yaw,
                        angle,
                    )

                    vehicle_tf = carla.Transform(
                        base_location,

                        carla.Rotation(
                            pitch=0.0,
                            yaw=float(
                                yaw
                            ),
                            roll=0.0,
                        ),
                    )

                    vehicle.set_transform(
                        vehicle_tf
                    )

                    gen.flush_queue(
                        rgb_queue
                    )

                    gen.flush_queue(
                        seg_queue
                    )

                    for _ in range(
                        args.settle_ticks
                    ):
                        world.tick()

                    frame = world.tick()

                    rgb_image = (
                        gen.wait_for_frame(
                            rgb_queue,
                            frame,
                        )
                    )

                    seg_image = (
                        gen.wait_for_frame(
                            seg_queue,
                            frame,
                        )
                    )

                    rgb = (
                        gen.carla_rgb_to_array(
                            rgb_image
                        )
                    )

                    tags = (
                        gen.carla_semantic_to_tags(
                            seg_image
                        )
                    )

                    bbox = (
                        gen.compute_actor_2d_bbox(
                            actor=vehicle,

                            camera_actor=(
                                rgb_camera
                            ),

                            K=K,

                            image_width=(
                                args.image_width
                            ),

                            image_height=(
                                args.image_height
                            ),
                        )
                    )

                    if bbox is None:

                        print(
                            "[GrabCutBank] "
                            f"{capture_index}/{total} "
                            "bbox not visible"
                        )

                        continue
                    view_rng_seed = (
                        make_view_rng_seed(
                            base_seed=(
                                args.seed
                            ),
                            angle=angle,
                            distance=(
                                distance_m
                            ),
                            elevation=(
                                elevation_deg
                            ),
                        )
                    )

                    random.seed(
                        view_rng_seed
                    )

                    np.random.seed(
                        view_rng_seed
                    )

                    cv2.setRNGSeed(
                        view_rng_seed
                    )
                    (
                        alpha,
                        body_mask,
                        window_mask,
                    ) = (
                        make_seeded_grabcut_alpha(
                            rgb=rgb,

                            tags=tags,

                            bbox=bbox,

                            args=args,
                        )
                    )

                    rgb_treated = (
                        gen.apply_window_treatment(
                            rgb=rgb,

                            window_mask=(
                                window_mask
                            ),

                            args=args,
                        )
                    )

                    (
                        crop_rgb,
                        crop_alpha,
                        crop_window,
                        rgba,
                        crop_box,
                        anchor,
                    ) = (
                        gen.crop_rgba_outputs(
                            rgb_treated=(
                                rgb_treated
                            ),

                            alpha=alpha,

                            window_mask=(
                                window_mask
                            ),

                            bbox=bbox,

                            args=args,
                        )
                    )

                    d_name = safe_float_name(
                        distance_m
                    )

                    e_name = safe_float_name(
                        elevation_deg
                    )

                    stem = (
                        f"d_{d_name}_"
                        f"e_{e_name}_"
                        f"angle_{angle:03d}"
                    )

                    rgba_path = (
                        rgba_dir
                        /
                        f"{stem}_rgba.png"
                    )

                    mask_path = (
                        mask_dir
                        /
                        f"{stem}_mask.png"
                    )

                    debug_path = (
                        debug_dir
                        /
                        f"{stem}_debug.png"
                    )

                    atomic_save_image(
                        save_function=(
                            gen.save_rgba
                        ),
                        path=rgba_path,
                        image=rgba,
                    )

                    atomic_save_image(
                        save_function=(
                            gen.save_mask
                        ),
                        path=mask_path,
                        image=crop_alpha,
                    )

                    debug = gen.draw_debug(
                        rgb=rgb,
                        bbox=bbox,
                        crop_box=crop_box,
                        alpha=alpha,
                        window_mask=window_mask,
                        angle=angle,
                        yaw=yaw,
                    )

                    atomic_save_image(
                        save_function=(
                            gen.save_rgb
                        ),
                        path=debug_path,
                        image=debug,
                    )

                    alpha_pixels = int(
                        np.sum(
                            crop_alpha > 10
                        )
                    )

                    record = {

                        "angle_deg":
                            angle,

                        "distance_m":
                            float(
                                distance_m
                            ),

                        "elevation_deg":
                            float(
                                elevation_deg
                            ),

                        "capture_altitude_m":
                            float(
                                args.capture_altitude
                            ),
                        "rng_seed":
                            int(
                                view_rng_seed
                            ),
                        "sprite_width_px":
                            int(
                                rgba.shape[
                                    1
                                ]
                            ),

                        "sprite_height_px":
                            int(
                                rgba.shape[
                                    0
                                ]
                            ),

                        "alpha_pixels":
                            alpha_pixels,

                        "anchor_x":
                            float(
                                anchor[
                                    "x"
                                ]
                            ),

                        "anchor_y":
                            float(
                                anchor[
                                    "y"
                                ]
                            ),

                        "rgba_path":
                            str(
                                rgba_path.resolve()
                            ),
                    }

                    records.append(
                        record
                    )

                    completed_keys.add(
                        make_view_key(
                            angle,
                            distance_m,
                            elevation_deg,
                        )
                    )

                    append_checkpoint_record(
                        csv_path,
                        record,
                    )

                    print(
                        "[GrabCutBank] "
                        f"{capture_index:02d}/{total:02d} "
                        f"a={angle:03d} "
                        f"d={distance_m:5.1f}m "
                        f"e={elevation_deg:5.1f}deg "
                        f"rgba="
                        f"{rgba.shape[1]}x"
                        f"{rgba.shape[0]} "
                        f"alpha="
                        f"{alpha_pixels}"
                    )



        records = validate_completed_dataset(
            output_dir=output_dir,
            records=records,
            args=args,
        )

        if records:

            atomic_write_records_csv(
                csv_path=csv_path,
                records=records,
            )
            if not args.skip_contact_sheets:
                for angle in args.angles:

                    make_contact_sheet(
                        output_dir=(
                            output_dir
                        ),

                        records=records,

                        angle=int(
                            angle
                        ) % 360,

                        distances=(
                            args.distances
                        ),

                        elevations=(
                            args.elevations
                        ),
                    )

        print()
        print("=" * 80)
        print(
            "[GrabCutBank] DONE"
        )

        print(
            "[GrabCutBank] output:",
            output_dir,
        )

        print("=" * 80)

    except Exception:

        traceback.print_exc()

        raise

    finally:

        print()
        print(
            "[GrabCutBank] Cleaning up..."
        )

        for sensor in [
            rgb_camera,
            seg_camera,
        ]:

            if sensor is not None:

                try:
                    sensor.stop()
                except Exception:
                    pass

        for actor in [
            rgb_camera,
            seg_camera,
            vehicle,
        ]:

            if actor is not None:

                try:
                    actor.destroy()
                except Exception:
                    pass

        if (
            world is not None
            and
            original_settings is not None
        ):

            try:

                gen.restore_world_settings(
                    world,
                    original_settings,
                )

            except Exception:
                pass

        print(
            "[GrabCutBank] Cleanup complete."
        )


if __name__ == "__main__":

    main()
