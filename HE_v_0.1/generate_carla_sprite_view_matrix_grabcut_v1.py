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
4. No road-only map-layer unloading.
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
import math
import queue
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
):
    """
    Load metadata from an earlier interrupted run.

    Returns
    -------
    records : list[dict]
        Existing CSV rows.

    completed_keys : set[tuple]
        (angle, distance, elevation) views already completed.
    """

    records = []
    completed_keys = set()

    if not csv_path.exists():
        return records, completed_keys

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:
                key = make_view_key(
                    row["angle_deg"],
                    row["distance_m"],
                    row["elevation_deg"],
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            records.append(row)
            completed_keys.add(key)

    return records, completed_keys


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
# Main
# ============================================================

def main():

    args = parse_args()
    if args.all_angles:

        args.angles = list(
            range(
                360
            )
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

    records = []
    completed_keys = set()

    if args.resume:

        (
            records,
            completed_keys,
        ) = load_checkpoint_records(
            csv_path
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

        gen.clear_existing_dynamic_actors(
            world
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

                    gen.save_rgba(
                        rgba_path,
                        rgba,
                    )

                    gen.save_mask(
                        mask_path,
                        crop_alpha,
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

                    gen.save_rgb(
                        debug_path,
                        debug,
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



        if records:

            with open(
                csv_path,
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