"""
generate_carla_sprite_view_matrix_v1.py

Small diagnostic sprite-bank generator.

Generates sprites as a function of:

    azimuth
    elevation
    distance

instead of assuming a single camera configuration.

Default diagnostic:
    azimuth   = 293 deg
    distance  = 5, 10, 20 m
    elevation = 0, 10, 20 deg

Only 9 sprites.

The viewpoint is defined around the vehicle visual centre:

    target_height = 0.75 m

which corresponds approximately to half of the current HE
visual height of 1.5 m.

Road-only capture:
    Uses CARLA environment-object visibility instead of
    unload_map_layer(), because map-layer streaming was causing
    simulator crashes on Town10HD_Opt.

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

# Reuse the already-working segmentation / alpha / crop utilities.
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
        default="vehicle_blue_sedan_view_matrix",
    )

    parser.add_argument(
        "--color",
        default="0,0,255",
    )

    parser.add_argument(
        "--output-root",
        default="assets/sprite_bank_view_matrix",
    )

    parser.add_argument(
        "--spawn-index",
        type=int,
        default=-1,
    )

    # --------------------------------------------------------
    # Viewpoint coordinates
    # --------------------------------------------------------

    parser.add_argument(
        "--angles",
        type=int,
        nargs="+",
        default=[
            293,
        ],
        help=(
            "Sprite-bank azimuth angles. "
            "293 is the problematic oblique-left viewpoint."
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
        help=(
            "Camera-to-target viewing distance in metres."
        ),
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
        help=(
            "Camera elevation above the vehicle visual centre."
        ),
    )

    parser.add_argument(
        "--target-height",
        type=float,
        default=0.75,
        help=(
            "Vehicle-local height used as the centre of "
            "the viewpoint sphere."
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
    # Alpha extraction
    # --------------------------------------------------------

    parser.add_argument(
        "--vehicle-semantic-tag",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--rgb-diff-threshold",
        type=int,
        default=12,
        help=(
            "Minimum per-pixel RGB change used to recover "
            "actor appearance missing from semantic segmentation."
        ),
    )

    parser.add_argument(
        "--rgb-recovery-upper-fraction",
        type=float,
        default=0.78,
        help=(
            "Fraction of the semantic actor height in which "
            "broad RGB-difference recovery is allowed."
        ),
    )
    parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--crop-margin-px",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--bbox-expand-ratio",
        type=float,
        default=0.10,
    )

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

    parser.add_argument(
        "--road-only-scene",
        action="store_true",
    )

    parser.add_argument(
        "--restore-scene",
        action="store_true",
        help=(
            "Restore hidden environment objects afterward. "
            "Normally leave this OFF and restart CARLA after "
            "the diagnostic."
        ),
    )

    return parser.parse_args()


# ============================================================
# Road-only environment visibility
# ============================================================

ROAD_LABEL_NAMES = {
    "Roads",
    "RoadLines",
    "Ground",
    "Terrain",
    "Sidewalks",
    "SideWalks",
}


def hide_nonroad_environment_objects(
    world,
):
    """
    Hide static environment objects without unloading map layers.

    This avoids the Town*_Opt map-layer streaming path that was
    causing CARLA to terminate after sprite-generation runs.
    """

    label_enum = (
        carla.CityObjectLabel
    )

    hidden_ids = set()

    print()
    print("=" * 78)
    print(
        "[ViewMatrix] Hiding non-road environment objects"
    )
    print("=" * 78)

    label_names = [
        name
        for name in dir(
            label_enum
        )
        if not name.startswith(
            "_"
        )
    ]

    for name in sorted(
        label_names
    ):

        if name in ROAD_LABEL_NAMES:
            continue

        if name in {
            "Any",
            "None",
        }:
            continue

        value = getattr(
            label_enum,
            name,
        )

        # Skip non-enum attributes/methods.
        if callable(
            value
        ):
            continue

        try:

            objects = (
                world.get_environment_objects(
                    value
                )
            )

        except Exception:
            continue

        if not objects:
            continue

        ids = {
            int(
                obj.id
            )
            for obj in objects
        }

        if not ids:
            continue

        hidden_ids.update(
            ids
        )

        print(
            f"[ViewMatrix] "
            f"{name:20s}: "
            f"{len(ids):6d}"
        )

    if hidden_ids:

        # Batch calls to avoid sending an excessively large
        # object-ID array in a single request.
        ids_list = list(
            hidden_ids
        )

        batch_size = 1000

        for start in range(
            0,
            len(
                ids_list
            ),
            batch_size,
        ):

            batch = ids_list[
                start:
                start
                +
                batch_size
            ]

            world.enable_environment_objects(
                batch,
                False,
            )

        for _ in range(
            3
        ):
            world.tick()

    print(
        "[ViewMatrix] Total hidden:",
        len(
            hidden_ids
        ),
    )

    return hidden_ids


def restore_environment_objects(
    world,
    hidden_ids,
):

    if not hidden_ids:
        return

    ids_list = list(
        hidden_ids
    )

    batch_size = 1000

    print(
        "[ViewMatrix] Restoring environment objects..."
    )

    for start in range(
        0,
        len(
            ids_list
        ),
        batch_size,
    ):

        batch = ids_list[
            start:
            start
            +
            batch_size
        ]

        world.enable_environment_objects(
            batch,
            True,
        )

    for _ in range(
        3
    ):
        world.tick()


# ============================================================
# Generic spherical viewpoint
# ============================================================

def build_view_camera_transform(
    base_location,
    base_yaw_deg,
    distance_m,
    elevation_deg,
    target_height_m,
):
    """
    Camera viewpoint around the actor.

    distance_m:
        camera-to-target Euclidean distance

    elevation_deg:
        elevation above actor visual centre

    elevation = 0:
        camera level with visual centre

    elevation > 0:
        camera above vehicle, looking downward
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

    camera_rotation = (
        gen.look_at_rotation(
            camera_location,
            target_location,
        )
    )

    return carla.Transform(
        camera_location,
        camera_rotation,
    )


# ============================================================
# Output helpers
# ============================================================

def safe_name_float(
    value,
):

    value = float(
        value
    )

    sign = (
        "m"
        if value < 0
        else ""
    )

    text = (
        f"{abs(value):05.1f}"
        .replace(
            ".",
            "p",
        )
    )

    return (
        sign
        +
        text
    )


def make_contact_sheet(
    output_dir,
    records,
    angle,
    distances,
    elevations,
):

    records_lookup = {
        (
            float(
                r[
                    "distance_m"
                ]
            ),
            float(
                r[
                    "elevation_deg"
                ]
            ),
        ):
            r
        for r in records
        if int(
            r[
                "angle_deg"
            ]
        )
        ==
        int(
            angle
        )
    }

    cell_w = 520
    cell_h = 330

    label_h = 40

    cols = len(
        elevations
    )

    rows = len(
        distances
    )

    sheet = Image.new(
        "RGB",
        (
            cols
            *
            cell_w,
            rows
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

    for row_idx, distance in enumerate(
        distances
    ):

        for col_idx, elevation in enumerate(
            elevations
        ):

            x0 = (
                col_idx
                *
                cell_w
            )

            y0 = (
                row_idx
                *
                (
                    cell_h
                    +
                    label_h
                )
            )

            record = records_lookup.get(
                (
                    float(
                        distance
                    ),
                    float(
                        elevation
                    ),
                )
            )

            label = (
                f"d={distance:g} m   "
                f"elev={elevation:g} deg"
            )

            draw.text(
                (
                    x0 + 10,
                    y0 + 10,
                ),
                label,
                fill="black",
            )

            if record is None:
                continue

            rgba_path = Path(
                record[
                    "rgba_path"
                ]
            )

            image = (
                Image.open(
                    rgba_path
                )
                .convert(
                    "RGBA"
                )
            )

            background = Image.new(
                "RGBA",
                image.size,
                "white",
            )

            background.alpha_composite(
                image
            )

            thumb = ImageOps.contain(
                background.convert(
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
        "[ViewMatrix] Contact sheet:",
        output_path,
    )
def capture_background_rgb(
    world,
    vehicle,
    rgb_queue,
    seg_queue,
):
    """
    Capture the static scene with the target actor hidden.

    Vehicle physics is disabled in this generator, so temporarily
    moving it far below the map is safe and deterministic.
    """

    original_tf = (
        vehicle.get_transform()
    )

    hidden_tf = carla.Transform(

        carla.Location(
            x=float(
                original_tf.location.x
            ),

            y=float(
                original_tf.location.y
            ),

            z=(
                float(
                    original_tf.location.z
                )
                -
                100.0
            ),
        ),

        original_tf.rotation,
    )

    vehicle.set_transform(
        hidden_tf
    )

    gen.flush_queue(
        rgb_queue
    )

    gen.flush_queue(
        seg_queue
    )

    # Settle hidden actor / render state.
    for _ in range(3):

        world.tick()

        gen.flush_queue(
            rgb_queue
        )

        gen.flush_queue(
            seg_queue
        )

    frame = world.tick()

    rgb_image = (
        gen.wait_for_frame(
            rgb_queue,
            frame,
        )
    )

    # Drain matching semantic frame.
    gen.wait_for_frame(
        seg_queue,
        frame,
    )

    background_rgb = (
        gen.carla_rgb_to_array(
            rgb_image
        )
    )

    vehicle.set_transform(
        original_tf
    )

    gen.flush_queue(
        rgb_queue
    )

    gen.flush_queue(
        seg_queue
    )

    for _ in range(2):

        world.tick()

        gen.flush_queue(
            rgb_queue
        )

        gen.flush_queue(
            seg_queue
        )

    return background_rgb

# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

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

    client = None
    world = None

    original_settings = None

    vehicle = None
    rgb_camera = None
    seg_camera = None

    hidden_ids = set()

    rgb_queue = queue.Queue()
    seg_queue = queue.Queue()

    records = []

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

        world = (
            client.get_world()
        )

        original_settings = (
            world.get_settings()
        )

        print()
        print("=" * 78)
        print(
            "CARLA VIEWPOINT SPRITE MATRIX V1"
        )
        print("=" * 78)

        print(
            "[ViewMatrix] Map:",
            world.get_map().name,
        )

        print(
            "[ViewMatrix] angles:",
            args.angles,
        )

        print(
            "[ViewMatrix] distances:",
            args.distances,
        )

        print(
            "[ViewMatrix] elevations:",
            args.elevations,
        )

        print(
            "[ViewMatrix] image:",
            f"{args.image_width}x{args.image_height}",
            f"FOV={args.fov}",
        )

        gen.setup_synchronous_mode(
            world,
            args.fixed_delta_seconds,
        )

        # Clear current dynamic actors/sensors first.
        gen.clear_existing_dynamic_actors(
            world
        )

        if args.road_only_scene:

            hidden_ids = (
                hide_nonroad_environment_objects(
                    world
                )
            )

        blueprint_library = (
            world.get_blueprint_library()
        )

        # Existing helper only needs:
        # vehicle_blueprint, color, spawn_index.
        vehicle, actual_bp = (
            gen.spawn_vehicle(
                world,
                blueprint_library,
                args,
            )
        )

        base_tf = (
            vehicle.get_transform()
        )

        base_location = carla.Location(
            x=float(
                base_tf.location.x
            ),

            y=float(
                base_tf.location.y
            ),

            z=float(
                base_tf.location.z
            ),
        )

        base_yaw = float(
            base_tf.rotation.yaw
        )

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
            lambda image:
                rgb_queue.put(
                    image
                )
        )

        seg_camera.listen(
            lambda image:
                seg_queue.put(
                    image
                )
        )

        for _ in range(
            10
        ):

            world.tick()

            gen.flush_queue(
                rgb_queue
            )

            gen.flush_queue(
                seg_queue
            )

        K = (
            gen.build_camera_intrinsics(
                width=(
                    args.image_width
                ),

                height=(
                    args.image_height
                ),

                fov_degrees=(
                    args.fov
                ),
            )
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

        print()
        print(
            "[ViewMatrix] Total captures:",
            total,
        )
        print()

        capture_index = 0

        for distance_m in (
            args.distances
        ):

            for elevation_deg in (
                args.elevations
            ):

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

                # Let sensor transforms settle.
                for _ in range(
                    3
                ):

                    world.tick()

                    gen.flush_queue(
                        rgb_queue
                    )

                    gen.flush_queue(
                        seg_queue
                    )
                background_rgb = (
                    capture_background_rgb(
                        world=world,

                        vehicle=vehicle,

                        rgb_queue=(
                            rgb_queue
                        ),

                        seg_queue=(
                            seg_queue
                        ),
                    )
                )
                for angle in (
                    args.angles
                ):

                    capture_index += 1

                    angle = (
                        int(
                            angle
                        )
                        %
                        360
                    )

                    yaw = (
                        gen.vehicle_yaw_for_angle(
                            base_yaw,
                            angle,
                        )
                    )

                    vehicle_tf = (
                        carla.Transform(
                            base_location,

                            carla.Rotation(
                                pitch=0.0,

                                yaw=float(
                                    yaw
                                ),

                                roll=0.0,
                            ),
                        )
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

                    frame = (
                        world.tick()
                    )

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
                            "[ViewMatrix] "
                            f"{capture_index}/{total} "
                            f"d={distance_m:g} "
                            f"e={elevation_deg:g} "
                            f"a={angle:03d}: "
                            "bbox not visible"
                        )

                        continue

                    (
                        alpha,
                        body_mask,
                        window_mask,
                    ) = (
                        gen.make_alpha_from_semantic_and_background(
                            rgb=rgb,

                            background_rgb=(
                                background_rgb
                            ),

                            tags=tags,

                            bbox=bbox,

                            args=args,
                        )
                    )

                    alpha_pixels = int(
                        np.sum(
                            alpha > 10
                        )
                    )

                    if (
                        alpha_pixels
                        <
                        args.min_mask_pixels
                    ):

                        print(
                            "[ViewMatrix] semantic mask "
                            "too small; using GrabCut"
                        )

                        alpha = (
                            gen.make_grabcut_alpha_from_bbox(
                                rgb=rgb,
                                bbox=bbox,
                                args=args,
                            )
                        )

                        body_mask = (
                            alpha.copy()
                        )

                        window_mask = (
                            np.zeros_like(
                                alpha
                            )
                        )

                    alpha_pixels = int(
                        np.sum(
                            alpha > 10
                        )
                    )

                    if (
                        alpha_pixels
                        <
                        args.min_mask_pixels
                    ):

                        print(
                            "[ViewMatrix] "
                            f"{capture_index}/{total} "
                            "alpha too small, skipping"
                        )

                        continue

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

                    distance_name = (
                        safe_name_float(
                            distance_m
                        )
                    )

                    elevation_name = (
                        safe_name_float(
                            elevation_deg
                        )
                    )

                    stem = (
                        f"d_{distance_name}_"
                        f"e_{elevation_name}_"
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

                    debug = (
                        gen.draw_debug(
                            rgb=rgb,

                            bbox=bbox,

                            crop_box=(
                                crop_box
                            ),

                            alpha=alpha,

                            window_mask=(
                                window_mask
                            ),

                            angle=angle,

                            yaw=yaw,
                        )
                    )

                    gen.save_rgb(
                        debug_path,
                        debug,
                    )

                    camera_actual_tf = (
                        rgb_camera
                        .get_transform()
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

                        "target_height_m":
                            float(
                                args.target_height
                            ),

                        "camera_x":
                            float(
                                camera_actual_tf
                                .location
                                .x
                            ),

                        "camera_y":
                            float(
                                camera_actual_tf
                                .location
                                .y
                            ),

                        "camera_z":
                            float(
                                camera_actual_tf
                                .location
                                .z
                            ),

                        "camera_pitch_deg":
                            float(
                                camera_actual_tf
                                .rotation
                                .pitch
                            ),

                        "camera_yaw_deg":
                            float(
                                camera_actual_tf
                                .rotation
                                .yaw
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

                    print(
                        "[ViewMatrix] "
                        f"{capture_index:02d}/{total:02d} "
                        f"a={angle:03d} "
                        f"d={distance_m:5.1f}m "
                        f"e={elevation_deg:5.1f}deg "
                        f"pitch="
                        f"{camera_actual_tf.rotation.pitch:+6.2f} "
                        f"rgba="
                        f"{rgba.shape[1]}x"
                        f"{rgba.shape[0]} "
                        f"alpha="
                        f"{alpha_pixels}"
                    )

        # ====================================================
        # Metadata
        # ====================================================

        csv_path = (
            output_dir
            /
            "view_matrix.csv"
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

            for angle in (
                args.angles
            ):

                make_contact_sheet(
                    output_dir=(
                        output_dir
                    ),

                    records=records,

                    angle=int(
                        angle
                    )
                    %
                    360,

                    distances=(
                        args.distances
                    ),

                    elevations=(
                        args.elevations
                    ),
                )

        print()
        print("=" * 78)
        print(
            "[ViewMatrix] DONE"
        )
        print(
            "[ViewMatrix] Output:",
            output_dir,
        )
        print(
            "[ViewMatrix] Metadata:",
            csv_path,
        )
        print("=" * 78)

    except Exception:

        traceback.print_exc()

        raise

    finally:

        print()
        print(
            "[ViewMatrix] Cleaning up..."
        )

        for sensor in [
            rgb_camera,
            seg_camera,
        ]:

            if sensor is None:
                continue

            try:
                sensor.stop()
            except Exception:
                pass

        for actor in [
            rgb_camera,
            seg_camera,
            vehicle,
        ]:

            if actor is None:
                continue

            try:
                actor.destroy()
            except Exception:
                pass

        # Environment-object hiding is deliberately not restored
        # unless explicitly requested. A CARLA restart is safer
        # for the current Town10HD_Opt workflow.
        if (
            world is not None
            and
            hidden_ids
        ):

            if args.restore_scene:

                try:

                    restore_environment_objects(
                        world,
                        hidden_ids,
                    )

                except Exception as exc:

                    print(
                        "[ViewMatrix] Restore warning:",
                        exc,
                    )

            else:

                print(
                    "[ViewMatrix] Non-road objects remain hidden."
                )

                print(
                    "[ViewMatrix] Restart CARLA to restore "
                    "the normal scene."
                )

        if (
            world is not None
            and
            original_settings is not None
        ):

            try:

                world.apply_settings(
                    original_settings
                )

            except Exception:
                pass

        print(
            "[ViewMatrix] Cleanup complete."
        )


if __name__ == "__main__":

    main()