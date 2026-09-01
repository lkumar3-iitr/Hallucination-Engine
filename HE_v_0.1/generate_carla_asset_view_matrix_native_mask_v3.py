"""
generate_carla_asset_view_matrix_native_mask_v2.py

Production CARLA 0.9.15 asset-bank generator for Hallucination Engine.

Mask source
-----------
CARLA native:
    sensor.camera.instance_segmentation

No GrabCut is used.

The RGB camera provides appearance; the CARLA instance-segmentation
camera provides the mask.

For EVERY view, the script captures:
    1. actor-present instance segmentation,
    2. the identical camera view with the SAME actor temporarily moved away.

The alpha mask is the pixel-wise difference between those two CARLA-native
annotation frames. No GrabCut, no semantic-class gating, no actor-ID mapping,
and no instance-ID calibration are used.

This is intentionally robust to CARLA 0.9.15 semantic/LOD inconsistencies
and to Unreal instance IDs that do not map cleanly to Python actor.id.

Supported logical asset classes:
    vehicle
    truck
    pedestrian

Production view matrix:
    360 azimuths
    elevations = [0, 5, 10, 20] deg

    vehicle / pedestrian:
        distances = [5, 10, 20] m
        total = 4320 views / asset

    bus:
        distances = [10, 15, 20, 25] m
        total = 5760 views / asset

Important geometry convention
-----------------------------
distance_m is the requested camera-to-physical-actor-center distance.

By default:
    --target-height auto
uses the exact CARLA physical bounding-box center, including the bbox
local X/Y/Z offset after actor rotation. Therefore distance_m is an exact
Euclidean camera -> physical bbox-center distance for production banks.

Every row also stores the measured physical bounding-box-center distance:
    bbox_center_distance_m

and physical forward-depth span:
    nearest_bbox_depth_m
    farthest_bbox_depth_m

This makes banks from differently sized vehicles, trucks and
pedestrians directly interpretable.

Outputs
-------
<output-root>/<asset-id>/
    rgba/
    rgb/
    mask/
    instance/
    debug/
    contact_sheets/
    view_matrix.csv
    generation_config.json
    asset_metadata.json
    qa_summary.json

Resume
------
Use --resume.  A generation fingerprint prevents incompatible settings
from being mixed into one bank.  A view is considered complete only if
all required image files exist and are non-empty.

CARLA process reuse
-------------------
The generator NEVER reloads the map internally and NEVER calls
unload_map_layer(). It validates the current map, removes dynamic actors,
and disables static environment objects (except Sky) on every invocation.
Disabling the same environment objects repeatedly is idempotent, so
Tesla -> Patrol -> truck -> pedestrian can be generated sequentially in
one running CARLA process without restarting or reloading Town10HD_Opt.

This lifecycle is intentional: repeated MapLayer.All unload/reload
operations were observed to destabilize the CARLA/UE process between
separate asset-bank runs.

Python:
    3.7+

CARLA:
    tested design target: 0.9.15
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import queue
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import carla

import generate_carla_360_rgba_fixed_camera as gen


# Requested close-range Cartesian grids can include poses where the complete
# physical bbox is outside the capture plane.  Those poses have no usable
# sprite and are intentionally absent from the runtime bank; they must not
# abort an otherwise resumable multi-hour capture.
INTENTIONALLY_NONPROJECTABLE_KEYS = set()


GENERATOR_SCHEMA_VERSION = 2
VIEW_SCHEMA_VERSION = 1
ASSET_METADATA_SCHEMA_VERSION = 1

MASK_SOURCE = "sensor.camera.instance_segmentation"
MASK_POSTPROCESSING = "per_view_native_instance_frame_difference"

# CARLA 0.9.14+ / 0.9.15 semantic IDs.
# We resolve "auto" from the logical asset class instead of guessing
# the dominant semantic class inside a projected 3-D bbox.  Guessing
# can select sky/background through transparent windows.
SEMANTIC_TAG_BY_ASSET_CLASS = {
    "vehicle": 14,      # Car / van
    "truck": 15,        # Truck
    "bus": 16,          # Bus
    "pedestrian": 12,   # Pedestrian
}


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
        "--required-map",
        default="Town10HD_Opt",
    )

    parser.add_argument(
        "--asset-id",
        required=True,
        help="Stable HE asset-bank identifier.",
    )

    parser.add_argument(
        "--asset-class",
        required=True,
        choices=[
            "vehicle",
            "truck",
            "bus",
            "pedestrian",
        ],
    )

    parser.add_argument(
        "--actor-blueprint",
        required=True,
        help=(
            "CARLA actor blueprint, e.g. vehicle.tesla.model3, "
            "vehicle.carlamotors.european_hgv, "
            "walker.pedestrian.0001."
        ),
    )

    parser.add_argument(
        "--semantic-tag",
        default="auto",
        help=(
            "CARLA semantic tag integer or 'auto'. With 'auto', "
            "CARLA 0.9.15 class mapping is used: vehicle=14, "
            "truck=15, bus=16, pedestrian=12."
        ),
    )

    parser.add_argument(
        "--color",
        default="0,0,255",
        help=(
            "Requested actor color when the blueprint supports a "
            "'color' attribute. Ignored for blueprints without color."
        ),
    )

    parser.add_argument(
        "--output-root",
        default="assets/sprite_bank_instance_v1",
    )

    parser.add_argument(
        "--spawn-index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    # --------------------------------------------------------
    # View matrix
    # --------------------------------------------------------

    parser.add_argument(
        "--angles",
        type=int,
        nargs="+",
        default=[
            0,
        ],
    )

    parser.add_argument(
        "--all-angles",
        action="store_true",
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
        "--full-dataset",
        action="store_true",
        help=(
            "Production matrix: 360 angles x [5,10,20] m x "
            "[0,5,10,20] deg = 4320 views."
        ),
    )

    parser.add_argument(
        "--skip-contact-sheets",
        action="store_true",
    )

    # --------------------------------------------------------
    # Capture geometry
    # --------------------------------------------------------

    parser.add_argument(
        "--target-height",
        default="auto",
        help=(
            "'auto' uses actor.bounding_box.location.z. "
            "A numeric value may be supplied explicitly."
        ),
    )

    parser.add_argument(
        "--capture-altitude",
        type=float,
        default=80.0,
    )

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

    parser.add_argument(
        "--crop-margin-px",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--projected-bbox-tolerance-px",
        type=float,
        default=8.0,
        help=(
            "Small tolerance for the conservative projected CARLA physical "
            "bbox extending beyond the capture image. The exact overflow is "
            "still stored in metadata. Actual mask/image-edge QA remains "
            "independent and is not relaxed."
        ),
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
        "--background-settle-ticks",
        type=int,
        default=2,
        help=(
            "Ticks after temporarily moving the actor out of view before "
            "capturing the per-view CARLA annotation background."
        ),
    )

    parser.add_argument(
        "--hide-environment-objects",
        action="store_true",
        help=(
            "Optionally disable static environment objects except Sky. "
            "Not required for mask correctness because the mask is derived "
            "from actor-present vs actor-absent CARLA annotation frames."
        ),
    )

    args = parser.parse_args()

    if args.full_dataset:

        args.angles = list(
            range(
                360
            )
        )

        if args.asset_class == "bus":

            args.distances = [
                10.0,
                15.0,
                20.0,
                25.0,
            ]

        else:

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

        args.skip_contact_sheets = True

    elif args.all_angles:

        args.angles = list(
            range(
                360
            )
        )

    args.angles = [
        int(v) % 360
        for v in args.angles
    ]

    # Preserve order while removing duplicates.
    args.angles = list(
        dict.fromkeys(
            args.angles
        )
    )

    args.distances = [
        float(v)
        for v in args.distances
    ]

    args.elevations = [
        float(v)
        for v in args.elevations
    ]

    if not args.angles:
        raise RuntimeError(
            "No angles requested."
        )

    if not args.distances:
        raise RuntimeError(
            "No distances requested."
        )

    if not args.elevations:
        raise RuntimeError(
            "No elevations requested."
        )

    for value in args.distances:

        if value <= 0.0:

            raise RuntimeError(
                "Distances must be positive."
            )

    return args


# ============================================================
# Generic utilities
# ============================================================

def safe_float_name(
    value,
):

    return (
        "{:05.1f}".format(
            float(
                value
            )
        )
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

    return (
        int(
            angle
        ) % 360,
        round(
            float(
                distance
            ),
            6,
        ),
        round(
            float(
                elevation
            ),
            6,
        ),
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


def atomic_write_json(
    path,
    data,
):

    path = Path(
        path
    )

    temp_path = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    with open(
        str(
            temp_path
        ),
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

    os.replace(
        str(
            temp_path
        ),
        str(
            path
        ),
    )


def atomic_save_pil(
    path,
    image,
):

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

    image.save(
        str(
            temp_path
        )
    )

    if not file_is_nonempty(
        temp_path
    ):

        raise RuntimeError(
            "Temporary image write failed: {}".format(
                temp_path
            )
        )

    os.replace(
        str(
            temp_path
        ),
        str(
            path
        ),
    )


def save_rgb(
    path,
    rgb,
):

    atomic_save_pil(
        path,
        Image.fromarray(
            rgb.astype(
                np.uint8
            ),
            mode="RGB",
        ),
    )


def save_mask(
    path,
    mask,
):

    atomic_save_pil(
        path,
        Image.fromarray(
            mask.astype(
                np.uint8
            ),
            mode="L",
        ),
    )


def save_rgba(
    path,
    rgba,
):

    atomic_save_pil(
        path,
        Image.fromarray(
            rgba.astype(
                np.uint8
            ),
            mode="RGBA",
        ),
    )


def atomic_write_records_csv(
    csv_path,
    records,
):

    csv_path = Path(
        csv_path
    )

    temp_path = csv_path.with_suffix(
        csv_path.suffix
        +
        ".tmp"
    )

    if not records:

        if temp_path.exists():
            temp_path.unlink()

        with open(
            str(
                temp_path
            ),
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            f.write(
                ""
            )

            f.flush()
            os.fsync(
                f.fileno()
            )

        os.replace(
            str(
                temp_path
            ),
            str(
                csv_path
            ),
        )

        return

    fieldnames = []

    for record in records:

        for key in record.keys():

            if key not in fieldnames:

                fieldnames.append(
                    key
                )

    with open(
        str(
            temp_path
        ),
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:

            writer.writerow(
                record
            )

        f.flush()
        os.fsync(
            f.fileno()
        )

    os.replace(
        str(
            temp_path
        ),
        str(
            csv_path
        ),
    )


def normalize_angle_deg(
    angle_deg,
):

    return (
        (
            float(
                angle_deg
            )
            +
            180.0
        )
        %
        360.0
        -
        180.0
    )


def actor_yaw_for_angle(
    base_yaw_deg,
    angle_deg,
):
    """
    Preserve the existing HE asset-bank convention:

        angle 000 -> relative yaw   0
        angle 090 -> relative yaw -90
        angle 180 -> relative yaw 180
        angle 270 -> relative yaw +90
    """

    return float(
        base_yaw_deg
    ) - float(
        angle_deg
    )


def parse_semantic_tag_request(
    value,
    asset_class,
):

    text = str(
        value
    ).strip()

    if text.lower() == "auto":

        if (
            asset_class
            not in
            SEMANTIC_TAG_BY_ASSET_CLASS
        ):

            raise RuntimeError(
                "No automatic semantic tag mapping for asset class: {}".format(
                    asset_class
                )
            )

        return (
            "auto_asset_class",
            int(
                SEMANTIC_TAG_BY_ASSET_CLASS[
                    asset_class
                ]
            ),
        )

    try:

        return (
            "explicit",
            int(
                text
            ),
        )

    except Exception:

        raise RuntimeError(
            "--semantic-tag must be an integer or 'auto'."
        )

def resolve_target_height(
    actor,
    request,
):

    text = str(
        request
    ).strip()

    if text.lower() == "auto":

        return (
            "auto_bbox_center_z",
            float(
                actor.bounding_box.location.z
            ),
        )

    try:

        return (
            "explicit",
            float(
                text
            ),
        )

    except Exception:

        raise RuntimeError(
            "--target-height must be a number or 'auto'."
        )


# ============================================================
# Paths / view completeness
# ============================================================

def expected_view_paths(
    output_dir,
    angle,
    distance,
    elevation,
):

    angle = int(
        angle
    ) % 360

    d_name = safe_float_name(
        distance
    )

    e_name = safe_float_name(
        elevation
    )

    stem = (
        "d_{}_e_{}_angle_{:03d}".format(
            d_name,
            e_name,
            angle,
        )
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
            "{}_rgba.png".format(
                stem
            ),

        "rgb":
            output_dir
            /
            "rgb"
            /
            "{}_rgb.png".format(
                stem
            ),

        "mask":
            output_dir
            /
            "mask"
            /
            "{}_mask.png".format(
                stem
            ),

        "instance":
            output_dir
            /
            "instance"
            /
            "{}_instance.png".format(
                stem
            ),

        "debug":
            output_dir
            /
            "debug"
            /
            "{}_debug.png".format(
                stem
            ),
    }


def output_has_existing_dataset(
    output_dir,
):

    output_dir = Path(
        output_dir
    )

    if (
        output_dir
        /
        "view_matrix.csv"
    ).exists():

        return True

    for subdir in [
        "rgba",
        "rgb",
        "mask",
        "instance",
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


# ============================================================
# CARLA geometry
# ============================================================

def actor_bbox_center_world_location(
    actor,
):
    """
    Return the exact physical CARLA bounding-box center in world coordinates.

    Use CARLA's own transformed world vertices and average them. This avoids
    any ambiguity about matrix layout, local bbox offsets, or bbox rotation.
    """

    actor_tf = actor.get_transform()

    vertices = list(
        actor.bounding_box.get_world_vertices(
            actor_tf
        )
    )

    if not vertices:

        raise RuntimeError(
            "Actor bounding box returned no world vertices."
        )

    return carla.Location(
        x=float(
            sum(v.x for v in vertices)
            /
            len(vertices)
        ),
        y=float(
            sum(v.y for v in vertices)
            /
            len(vertices)
        ),
        z=float(
            sum(v.z for v in vertices)
            /
            len(vertices)
        ),
    )


def resolve_capture_target_location(
    actor,
    base_location,
    target_height_mode,
    resolved_target_height_m,
    actor_transform=None,
):
    """
    Production convention:

    --target-height auto
        target = exact physical CARLA bbox center (X/Y/Z).

    IMPORTANT:
        When actor_transform is supplied, use that requested transform
        directly instead of actor.get_transform(). This avoids a one-tick
        CARLA transform lag after actor.set_transform().

    explicit target height
        preserves the legacy actor-origin X/Y + requested Z offset.
    """

    if str(target_height_mode) == "auto_bbox_center_z":

        if actor_transform is None:

            actor_transform = actor.get_transform()

        vertices = list(
            actor.bounding_box.get_world_vertices(
                actor_transform
            )
        )

        if not vertices:

            raise RuntimeError(
                "Actor bounding box returned no world vertices."
            )

        return carla.Location(
            x=float(
                sum(v.x for v in vertices)
                /
                len(vertices)
            ),
            y=float(
                sum(v.y for v in vertices)
                /
                len(vertices)
            ),
            z=float(
                sum(v.z for v in vertices)
                /
                len(vertices)
            ),
        )

    return carla.Location(
        x=float(base_location.x),
        y=float(base_location.y),
        z=(
            float(base_location.z)
            +
            float(resolved_target_height_m)
        ),
    )

def build_view_camera_transform_from_target(
    target_location,
    base_yaw_deg,
    distance_m,
    elevation_deg,
):
    """
    Place the camera at an exact Euclidean distance from target_location.
    """

    elevation_rad = math.radians(
        float(elevation_deg)
    )

    yaw_rad = math.radians(
        float(base_yaw_deg)
    )

    horizontal_distance = (
        float(distance_m)
        *
        math.cos(elevation_rad)
    )

    vertical_distance = (
        float(distance_m)
        *
        math.sin(elevation_rad)
    )

    camera_location = carla.Location(
        x=(
            float(target_location.x)
            +
            horizontal_distance
            *
            math.cos(yaw_rad)
        ),
        y=(
            float(target_location.y)
            +
            horizontal_distance
            *
            math.sin(yaw_rad)
        ),
        z=(
            float(target_location.z)
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


def build_view_camera_transform(
    base_location,
    base_yaw_deg,
    distance_m,
    elevation_deg,
    target_height_m,
):

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
            math.cos(
                yaw_rad
            )
        ),
        y=(
            float(
                target_location.y
            )
            +
            horizontal_distance
            *
            math.sin(
                yaw_rad
            )
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


def world_location_to_camera_xyz(
    world_location,
    camera_actor,
):

    camera_tf = (
        camera_actor.get_transform()
    )

    world_to_camera = np.asarray(
        camera_tf.get_inverse_matrix(),
        dtype=np.float64,
    )

    point_world = np.asarray(
        [
            float(
                world_location.x
            ),
            float(
                world_location.y
            ),
            float(
                world_location.z
            ),
            1.0,
        ],
        dtype=np.float64,
    )

    point_camera = (
        world_to_camera
        @
        point_world
    )

    return (
        float(
            point_camera[
                0
            ]
        ),
        float(
            point_camera[
                1
            ]
        ),
        float(
            point_camera[
                2
            ]
        ),
    )


def project_camera_xyz(
    forward_m,
    right_m,
    up_m,
    K,
):

    if float(
        forward_m
    ) <= 0.001:

        return (
            None,
            None,
        )

    u = (
        float(
            K[
                0,
                0
            ]
        )
        *
        float(
            right_m
        )
        /
        float(
            forward_m
        )
        +
        float(
            K[
                0,
                2
            ]
        )
    )

    v = (
        float(
            K[
                1,
                1
            ]
        )
        *
        (
            -float(
                up_m
            )
        )
        /
        float(
            forward_m
        )
        +
        float(
            K[
                1,
                2
            ]
        )
    )

    return (
        float(
            u
        ),
        float(
            v
        ),
    )


def compute_view_geometry_metadata(
    actor,
    camera_actor,
    K,
    bbox_2d,
    crop_box,
    crop_mask,
    anchor,
    image_width,
    image_height,
):

    actor_tf = (
        actor.get_transform()
    )

    camera_tf = (
        camera_actor.get_transform()
    )

    bbox3d = (
        actor.bounding_box
    )

    vertices = list(
        bbox3d.get_world_vertices(
            actor_tf
        )
    )

    vertices_xyz = np.asarray(
        [
            [
                float(v.x),
                float(v.y),
                float(v.z),
            ]
            for v in vertices
        ],
        dtype=np.float64,
    )

    center_xyz = np.mean(
        vertices_xyz,
        axis=0,
    )

    center_world = carla.Location(
        x=float(
            center_xyz[
                0
            ]
        ),
        y=float(
            center_xyz[
                1
            ]
        ),
        z=float(
            center_xyz[
                2
            ]
        ),
    )

    bottom_indices = np.argsort(
        vertices_xyz[
            :,
            2
        ]
    )[
        :4
    ]

    support_xyz = np.mean(
        vertices_xyz[
            bottom_indices
        ],
        axis=0,
    )

    support_world = carla.Location(
        x=float(
            support_xyz[
                0
            ]
        ),
        y=float(
            support_xyz[
                1
            ]
        ),
        z=float(
            support_xyz[
                2
            ]
        ),
    )

    (
        center_forward,
        center_right,
        center_up,
    ) = world_location_to_camera_xyz(
        center_world,
        camera_actor,
    )

    bbox_center_distance = math.sqrt(
        center_forward
        *
        center_forward
        +
        center_right
        *
        center_right
        +
        center_up
        *
        center_up
    )

    (
        support_forward,
        support_right,
        support_up,
    ) = world_location_to_camera_xyz(
        support_world,
        camera_actor,
    )

    bbox_depths = []

    for vertex in vertices:

        (
            vertex_forward,
            _vertex_right,
            _vertex_up,
        ) = world_location_to_camera_xyz(
            vertex,
            camera_actor,
        )

        bbox_depths.append(
            float(
                vertex_forward
            )
        )

    nearest_depth = min(
        bbox_depths
    )

    farthest_depth = max(
        bbox_depths
    )

    (
        projected_ground_x,
        projected_ground_y,
    ) = project_camera_xyz(
        support_forward,
        support_right,
        support_up,
        K,
    )

    bbox_x1 = int(
        bbox_2d[
            0
        ]
    )

    bbox_y1 = int(
        bbox_2d[
            1
        ]
    )

    bbox_x2 = int(
        bbox_2d[
            2
        ]
    )

    bbox_y2 = int(
        bbox_2d[
            3
        ]
    )

    crop_x1 = int(
        crop_box[
            0
        ]
    )

    crop_y1 = int(
        crop_box[
            1
        ]
    )

    crop_x2 = int(
        crop_box[
            2
        ]
    )

    crop_y2 = int(
        crop_box[
            3
        ]
    )

    binary = (
        crop_mask > 0
    ).astype(
        np.uint8
    )

    ys, xs = np.where(
        binary > 0
    )

    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):

        raise RuntimeError(
            "Empty alpha mask while computing view metadata."
        )

    alpha_area = int(
        len(
            xs
        )
    )

    centroid_x = float(
        np.mean(
            xs
        )
    )

    centroid_y = float(
        np.mean(
            ys
        )
    )

    visible_x1 = int(
        xs.min()
    )

    visible_x2 = int(
        xs.max()
    )

    visible_y1 = int(
        ys.min()
    )

    visible_y2 = int(
        ys.max()
    )

    visible_width = (
        visible_x2
        -
        visible_x1
        +
        1
    )

    visible_height = (
        visible_y2
        -
        visible_y1
        +
        1
    )

    bottom_y = int(
        visible_y2
    )

    bottom_xs = xs[
        ys
        ==
        bottom_y
    ]

    bottom_x = float(
        np.mean(
            bottom_xs
        )
    )

    (
        component_count,
        _labels,
        _stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    component_count = max(
        0,
        int(
            component_count
        )
        -
        1,
    )

    return {
        "actor_world_x_m":
            float(
                actor_tf.location.x
            ),

        "actor_world_y_m":
            float(
                actor_tf.location.y
            ),

        "actor_world_z_m":
            float(
                actor_tf.location.z
            ),

        "actor_pitch_deg":
            float(
                actor_tf.rotation.pitch
            ),

        "actor_yaw_deg":
            float(
                actor_tf.rotation.yaw
            ),

        "actor_roll_deg":
            float(
                actor_tf.rotation.roll
            ),

        "camera_world_x_m":
            float(
                camera_tf.location.x
            ),

        "camera_world_y_m":
            float(
                camera_tf.location.y
            ),

        "camera_world_z_m":
            float(
                camera_tf.location.z
            ),

        "camera_pitch_deg":
            float(
                camera_tf.rotation.pitch
            ),

        "camera_yaw_deg":
            float(
                camera_tf.rotation.yaw
            ),

        "camera_roll_deg":
            float(
                camera_tf.rotation.roll
            ),

        "relative_actor_yaw_deg":
            float(
                normalize_angle_deg(
                    actor_tf.rotation.yaw
                    -
                    camera_tf.rotation.yaw
                )
            ),

        "bbox_center_distance_m":
            float(
                bbox_center_distance
            ),

        "actor_depth_m":
            float(
                center_forward
            ),

        "camera_forward_distance_m":
            float(
                center_forward
            ),

        "camera_right_offset_m":
            float(
                center_right
            ),

        "camera_up_offset_m":
            float(
                center_up
            ),

        "support_depth_m":
            float(
                support_forward
            ),

        "nearest_bbox_depth_m":
            float(
                nearest_depth
            ),

        "farthest_bbox_depth_m":
            float(
                farthest_depth
            ),

        "projected_bbox_x1_px":
            bbox_x1,

        "projected_bbox_y1_px":
            bbox_y1,

        "projected_bbox_x2_px":
            bbox_x2,

        "projected_bbox_y2_px":
            bbox_y2,

        "projected_bbox_width_px":
            int(
                bbox_x2
                -
                bbox_x1
                +
                1
            ),

        "projected_bbox_height_px":
            int(
                bbox_y2
                -
                bbox_y1
                +
                1
            ),

        "projected_bbox_center_x_px":
            float(
                (
                    bbox_x1
                    +
                    bbox_x2
                )
                /
                2.0
            ),

        "projected_bbox_bottom_y_px":
            float(
                bbox_y2
            ),

        "projected_ground_anchor_x_px":
            (
                float(
                    projected_ground_x
                )
                if projected_ground_x is not None
                else ""
            ),

        "projected_ground_anchor_y_px":
            (
                float(
                    projected_ground_y
                )
                if projected_ground_y is not None
                else ""
            ),

        "crop_x1_px":
            crop_x1,

        "crop_y1_px":
            crop_y1,

        "crop_x2_px":
            crop_x2,

        "crop_y2_px":
            crop_y2,

        "crop_width_px":
            int(
                crop_mask.shape[
                    1
                ]
            ),

        "crop_height_px":
            int(
                crop_mask.shape[
                    0
                ]
            ),

        "alpha_area_px":
            alpha_area,

        "alpha_centroid_x_crop_px":
            centroid_x,

        "alpha_centroid_y_crop_px":
            centroid_y,

        "alpha_centroid_x_full_px":
            float(
                crop_x1
                +
                centroid_x
            ),

        "alpha_centroid_y_full_px":
            float(
                crop_y1
                +
                centroid_y
            ),

        "visible_x1_crop_px":
            int(
                visible_x1
            ),

        "visible_y1_crop_px":
            int(
                visible_y1
            ),

        "visible_x2_crop_px":
            int(
                visible_x2
            ),

        "visible_y2_crop_px":
            int(
                visible_y2
            ),

        "visible_x1_full_px":
            int(
                crop_x1
                +
                visible_x1
            ),

        "visible_y1_full_px":
            int(
                crop_y1
                +
                visible_y1
            ),

        "visible_x2_full_px":
            int(
                crop_x1
                +
                visible_x2
            ),

        "visible_y2_full_px":
            int(
                crop_y1
                +
                visible_y2
            ),

        "visible_width_px":
            int(
                visible_width
            ),

        "visible_height_px":
            int(
                visible_height
            ),

        "alpha_bottom_anchor_x_crop_px":
            float(
                bottom_x
            ),

        "alpha_bottom_anchor_y_crop_px":
            float(
                bottom_y
            ),

        "alpha_bottom_anchor_x_full_px":
            float(
                crop_x1
                +
                bottom_x
            ),

        "alpha_bottom_anchor_y_full_px":
            float(
                crop_y1
                +
                bottom_y
            ),

        "he_anchor_x_crop_px":
            float(
                anchor[
                    "x"
                ]
            ),

        "he_anchor_y_crop_px":
            float(
                anchor[
                    "y"
                ]
            ),

        "he_anchor_x_full_px":
            float(
                crop_x1
                +
                float(
                    anchor[
                        "x"
                    ]
                )
            ),

        "he_anchor_y_full_px":
            float(
                crop_y1
                +
                float(
                    anchor[
                        "y"
                    ]
                )
            ),

        "connected_component_count":
            int(
                component_count
            ),

        "mask_touches_crop_left":
            bool(
                np.any(
                    binary[
                        :,
                        0
                    ] > 0
                )
            ),

        "mask_touches_crop_right":
            bool(
                np.any(
                    binary[
                        :,
                        -1
                    ] > 0
                )
            ),

        "mask_touches_crop_top":
            bool(
                np.any(
                    binary[
                        0,
                        :
                    ] > 0
                )
            ),

        "mask_touches_crop_bottom":
            bool(
                np.any(
                    binary[
                        -1,
                        :
                    ] > 0
                )
            ),

        "crop_touches_image_left":
            bool(
                crop_x1 <= 0
            ),

        "crop_touches_image_right":
            bool(
                crop_x2
                >=
                int(
                    image_width
                )
                -
                1
            ),

        "crop_touches_image_top":
            bool(
                crop_y1 <= 0
            ),

        "crop_touches_image_bottom":
            bool(
                crop_y2
                >=
                int(
                    image_height
                )
                -
                1
            ),
    }


# ============================================================
# QA
# ============================================================

def build_view_qa(
    geometry,
):

    flags = []
    warnings = []

    alpha_area = int(
        geometry[
            "alpha_area_px"
        ]
    )

    visible_width = int(
        geometry[
            "visible_width_px"
        ]
    )

    visible_height = int(
        geometry[
            "visible_height_px"
        ]
    )

    component_count = int(
        geometry[
            "connected_component_count"
        ]
    )

    nearest_depth = float(
        geometry[
            "nearest_bbox_depth_m"
        ]
    )

    if alpha_area <= 0:

        flags.append(
            "empty_mask"
        )

    if alpha_area < 100:

        flags.append(
            "tiny_mask"
        )

    if visible_width < 3:

        flags.append(
            "tiny_width"
        )

    if visible_height < 3:

        flags.append(
            "tiny_height"
        )

    if component_count == 0:

        flags.append(
            "no_component"
        )

    if component_count > 1:

        warnings.append(
            "disconnected_components"
        )

    if nearest_depth <= 0.05:

        flags.append(
            "camera_intersects_bbox"
        )

    if any(
        bool(
            geometry[
                key
            ]
        )
        for key in [
            "mask_touches_crop_left",
            "mask_touches_crop_right",
            "mask_touches_crop_top",
            "mask_touches_crop_bottom",
        ]
    ):

        flags.append(
            "mask_touches_crop_edge"
        )

    if any(
        bool(
            geometry[
                key
            ]
        )
        for key in [
            "crop_touches_image_left",
            "crop_touches_image_right",
            "crop_touches_image_top",
            "crop_touches_image_bottom",
        ]
    ):

        flags.append(
            "crop_touches_capture_edge"
        )

    aspect_ratio = (
        float(
            visible_width
        )
        /
        max(
            1.0,
            float(
                visible_height
            ),
        )
    )

    if (
        aspect_ratio < 0.05
        or
        aspect_ratio > 12.0
    ):

        flags.append(
            "extreme_aspect_ratio"
        )

    return {
        "qa_pass":
            len(
                flags
            ) == 0,

        "qa_flag_count":
            int(
                len(
                    flags
                )
            ),

        "qa_flags":
            "|".join(
                flags
            ),

        "qa_warning_count":
            int(
                len(
                    warnings
                )
            ),

        "qa_warnings":
            "|".join(
                warnings
            ),

        "visible_aspect_ratio":
            float(
                aspect_ratio
            ),
    }


# ============================================================
# Clean capture world
# ============================================================

def prepare_clean_capture_world(
    world,
    hide_environment_objects=False,
    settle_ticks=4,
):
    """
    Prepare a stable capture world without unloading CARLA map layers.

    For the native frame-difference mask, static environment objects do not
    contaminate alpha because the camera is identical in the actor-present
    and actor-absent annotation frames.

    By default we therefore avoid the expensive 60k+ environment-object
    disable operation. It can still be requested explicitly for diagnostics.
    """

    print()

    if hide_environment_objects:

        print(
            "[NativeAssetBank] Disabling static environment objects "
            "except Sky..."
        )

        try:

            environment_objects = list(
                world.get_environment_objects(
                    carla.CityObjectLabel.Any
                )
            )

            hidden_ids = {
                int(
                    obj.id
                )
                for obj in environment_objects
                if obj.type
                !=
                carla.CityObjectLabel.Sky
            }

            if hidden_ids:

                world.enable_environment_objects(
                    hidden_ids,
                    False,
                )

            print(
                "[NativeAssetBank] hidden environment objects:",
                len(
                    hidden_ids
                ),
            )

        except Exception as exc:

            print(
                "[NativeAssetBank] environment-object hiding warning:",
                exc,
            )

    else:

        print(
            "[NativeAssetBank] Static environment left loaded; "
            "native annotation frame-difference isolates the actor."
        )

    for _ in range(
        max(
            1,
            int(
                settle_ticks
            ),
        )
    ):

        world.tick()

    print(
        "[NativeAssetBank] capture world ready; no map layers unloaded."
    )


# ============================================================
# Actor / sensors
# ============================================================

def spawn_target_actor(
    world,
    blueprint_library,
    args,
):

    blueprint = (
        blueprint_library.find(
            str(
                args.actor_blueprint
            )
        )
    )

    if blueprint.has_attribute(
        "color"
    ):

        blueprint.set_attribute(
            "color",
            str(
                args.color
            ),
        )

    if blueprint.has_attribute(
        "role_name"
    ):

        blueprint.set_attribute(
            "role_name",
            "he_asset_target",
        )

    spawn_points = (
        world.get_map()
        .get_spawn_points()
    )

    if not spawn_points:

        raise RuntimeError(
            "CARLA map has no spawn points."
        )

    reference_tf = spawn_points[
        int(
            args.spawn_index
        )
        %
        len(
            spawn_points
        )
    ]

    high_location = carla.Location(
        x=float(
            reference_tf.location.x
        ),
        y=float(
            reference_tf.location.y
        ),
        z=float(
            args.capture_altitude
        ),
    )

    high_tf = carla.Transform(
        high_location,
        carla.Rotation(
            pitch=0.0,
            yaw=float(
                reference_tf.rotation.yaw
            ),
            roll=0.0,
        ),
    )

    actor = world.try_spawn_actor(
        blueprint,
        high_tf,
    )

    if actor is None:

        # Fallback: spawn at normal point then immediately move high.
        actor = world.try_spawn_actor(
            blueprint,
            reference_tf,
        )

        if actor is None:

            raise RuntimeError(
                "Could not spawn actor blueprint: {}".format(
                    args.actor_blueprint
                )
            )

        try:

            actor.set_simulate_physics(
                False
            )

        except Exception:

            pass

        actor.set_transform(
            high_tf
        )

    try:

        actor.set_simulate_physics(
            False
        )

    except Exception:

        pass

    return (
        actor,
        high_location,
        float(
            reference_tf.rotation.yaw
        ),
    )


def configure_camera_blueprint(
    blueprint,
    args,
):

    blueprint.set_attribute(
        "image_size_x",
        str(
            int(
                args.image_width
            )
        ),
    )

    blueprint.set_attribute(
        "image_size_y",
        str(
            int(
                args.image_height
            )
        ),
    )

    blueprint.set_attribute(
        "fov",
        str(
            float(
                args.fov
            )
        ),
    )

    if blueprint.has_attribute(
        "sensor_tick"
    ):

        blueprint.set_attribute(
            "sensor_tick",
            "0.0",
        )


def spawn_cameras(
    world,
    blueprint_library,
    camera_tf,
    args,
):

    rgb_bp = (
        blueprint_library.find(
            "sensor.camera.rgb"
        )
    )

    instance_bp = (
        blueprint_library.find(
            MASK_SOURCE
        )
    )

    configure_camera_blueprint(
        rgb_bp,
        args,
    )

    configure_camera_blueprint(
        instance_bp,
        args,
    )

    rgb_camera = world.spawn_actor(
        rgb_bp,
        camera_tf,
    )

    try:

        instance_camera = world.spawn_actor(
            instance_bp,
            camera_tf,
        )

    except Exception:

        try:

            rgb_camera.destroy()

        except Exception:

            pass

        raise

    return (
        rgb_camera,
        instance_camera,
    )


# ============================================================
# Instance segmentation
# ============================================================

def decode_instance_image(
    image,
):
    """
    CARLA raw image layout is BGRA.

    For instance segmentation:
        R -> semantic tag
        G/B -> instance identifier bytes
    """

    raw = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )

    raw = raw.reshape(
        (
            image.height,
            image.width,
            4,
        )
    )

    b = raw[
        :,
        :,
        0
    ].copy()

    g = raw[
        :,
        :,
        1
    ].copy()

    r = raw[
        :,
        :,
        2
    ].copy()

    instance_rgb = np.stack(
        [
            r,
            g,
            b,
        ],
        axis=2,
    )

    instance_id = (
        g.astype(
            np.uint32
        )
        *
        np.uint32(
            256
        )
        +
        b.astype(
            np.uint32
        )
    )

    return (
        r,
        instance_id,
        instance_rgb,
    )


def compute_actor_projected_bbox_native(
    actor,
    camera_actor,
    K,
    image_width,
    image_height,
):
    """
    Project all physical CARLA bbox vertices without the aggressive
    out-of-image filtering used by the older helper.

    Returns:
        clipped_bbox, projection_metadata

    clipped_bbox is suitable for the existing geometry metadata function.
    projection_metadata keeps the UNCLIPPED physical projection as well,
    so we can explicitly detect incomplete source captures.
    """

    actor_tf = actor.get_transform()

    vertices = list(
        actor.bounding_box.get_world_vertices(
            actor_tf
        )
    )

    projected = []

    for vertex in vertices:

        result = gen.project_carla_world_to_image(
            vertex,
            camera_actor,
            K,
        )

        if result is None:
            continue

        u, v, depth = result

        projected.append(
            (
                float(
                    u
                ),
                float(
                    v
                ),
                float(
                    depth
                ),
            )
        )

    if len(
        projected
    ) < 2:

        return (
            None,
            {
                "projected_vertex_count":
                    int(
                        len(
                            projected
                        )
                    ),
            },
        )

    xs = [
        p[
            0
        ]
        for p in projected
    ]

    ys = [
        p[
            1
        ]
        for p in projected
    ]

    raw_x1 = float(
        min(
            xs
        )
    )

    raw_y1 = float(
        min(
            ys
        )
    )

    raw_x2 = float(
        max(
            xs
        )
    )

    raw_y2 = float(
        max(
            ys
        )
    )

    clipped_x1 = int(
        max(
            0.0,
            min(
                float(
                    image_width
                    -
                    1
                ),
                raw_x1,
            ),
        )
    )

    clipped_y1 = int(
        max(
            0.0,
            min(
                float(
                    image_height
                    -
                    1
                ),
                raw_y1,
            ),
        )
    )

    clipped_x2 = int(
        max(
            0.0,
            min(
                float(
                    image_width
                    -
                    1
                ),
                raw_x2,
            ),
        )
    )

    clipped_y2 = int(
        max(
            0.0,
            min(
                float(
                    image_height
                    -
                    1
                ),
                raw_y2,
            ),
        )
    )

    if (
        clipped_x2 <= clipped_x1
        or
        clipped_y2 <= clipped_y1
    ):

        clipped_bbox = None

    else:

        clipped_bbox = [
            int(
                clipped_x1
            ),
            int(
                clipped_y1
            ),
            int(
                clipped_x2
            ),
            int(
                clipped_y2
            ),
        ]

    overflow_left_px = max(
        0.0,
        -float(
            raw_x1
        ),
    )

    overflow_top_px = max(
        0.0,
        -float(
            raw_y1
        ),
    )

    overflow_right_px = max(
        0.0,
        float(
            raw_x2
        )
        -
        float(
            image_width
            -
            1
        ),
    )

    overflow_bottom_px = max(
        0.0,
        float(
            raw_y2
        )
        -
        float(
            image_height
            -
            1
        ),
    )

    max_overflow_px = max(
        overflow_left_px,
        overflow_top_px,
        overflow_right_px,
        overflow_bottom_px,
    )

    fully_inside = (
        max_overflow_px
        <=
        0.0
    )

    metadata = {
        "projected_vertex_count":
            int(
                len(
                    projected
                )
            ),

        "projected_bbox_raw_x1_px":
            float(
                raw_x1
            ),

        "projected_bbox_raw_y1_px":
            float(
                raw_y1
            ),

        "projected_bbox_raw_x2_px":
            float(
                raw_x2
            ),

        "projected_bbox_raw_y2_px":
            float(
                raw_y2
            ),

        "projected_bbox_raw_width_px":
            float(
                raw_x2
                -
                raw_x1
            ),

        "projected_bbox_raw_height_px":
            float(
                raw_y2
                -
                raw_y1
            ),

        "projected_bbox_fully_inside_capture":
            bool(
                fully_inside
            ),
        "projected_bbox_overflow_left_px":
            float(
                overflow_left_px
            ),

        "projected_bbox_overflow_right_px":
            float(
                overflow_right_px
            ),

        "projected_bbox_overflow_top_px":
            float(
                overflow_top_px
            ),

        "projected_bbox_overflow_bottom_px":
            float(
                overflow_bottom_px
            ),

        "projected_bbox_max_overflow_px":
            float(
                max_overflow_px
            ),
    }

    return (
        clipped_bbox,
        metadata,
    )


def build_native_frame_difference_mask(
    present_instance_rgb,
    background_instance_rgb,
    present_instance_id=None,
):
    """
    Build the CARLA-native actor silhouette from the difference between
    actor-present and actor-absent instance-segmentation frames.

    CARLA 0.9.15 occasionally produces isolated annotation changes
    unrelated to the target actor. In production banks we observed
    exactly-one-pixel foreign instance IDs capable of expanding a sprite
    crop by hundreds of pixels.

    The frame difference remains the primary mask source. Instance IDs
    are used only to remove negligible foreign annotation noise.
    """

    if (
        present_instance_rgb.shape
        !=
        background_instance_rgb.shape
    ):
        raise RuntimeError(
            "Present/background instance frames have different shapes."
        )

    changed = np.any(
        present_instance_rgb
        !=
        background_instance_rgb,
        axis=2,
    )

    # --------------------------------------------------------
    # Filter negligible foreign non-zero instance IDs.
    # --------------------------------------------------------

    if present_instance_id is not None:

        changed_nonzero_ids = present_instance_id[
            changed
            &
            (
                present_instance_id > 0
            )
        ]

        if changed_nonzero_ids.size > 0:

            unique_ids, counts = np.unique(
                changed_nonzero_ids,
                return_counts=True,
            )

            order = np.argsort(
                counts
            )[::-1]

            unique_ids = unique_ids[
                order
            ]

            counts = counts[
                order
            ]

            dominant_count = int(
                counts[0]
            )

            # Keep substantial secondary IDs in case a valid actor is
            # represented using multiple CARLA instance IDs.
            minimum_valid_count = max(
                4,
                int(
                    round(
                        dominant_count * 0.01
                    )
                ),
            )

            accepted_ids = np.asarray(
                [
                    int(instance_value)
                    for instance_value, count
                    in zip(
                        unique_ids.tolist(),
                        counts.tolist(),
                    )
                    if int(count)
                    >= minimum_valid_count
                ],
                dtype=present_instance_id.dtype,
            )

            # Preserve zero-ID changed pixels because some CARLA assets
            # can expose incomplete/unhelpful instance bytes.
            changed = (
                changed
                &
                (
                    np.isin(
                        present_instance_id,
                        accepted_ids,
                    )
                    |
                    (
                        present_instance_id == 0
                    )
                )
            )

    # --------------------------------------------------------
    # Remove isolated one-pixel annotation flicker.
    #
    # We deliberately do NOT keep only the largest component because
    # valid vehicles/buses may have multiple disconnected components.
    # --------------------------------------------------------

    binary = changed.astype(
        np.uint8
    )

    (
        component_count,
        labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(
        binary
    )

    for label in range(
        1,
        int(component_count),
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= 2:

            cleaned[
                labels == label
            ] = 1

    if np.any(
        cleaned > 0
    ):
        changed = (
            cleaned > 0
        )

    mask = (
        changed.astype(
            np.uint8
        )
        *
        255
    )

    if not np.any(
        mask > 0
    ):
        raise RuntimeError(
            "CARLA native frame-difference produced an empty target mask."
        )

    return mask


def summarize_instance_annotation_on_mask(
    semantic_tags,
    instance_id,
    mask,
):
    """
    Keep CARLA semantic/instance information as metadata only.
    It is never used to decide the alpha mask.
    """

    target = (
        mask > 0
    )

    observed_tags, observed_tag_counts = np.unique(
        semantic_tags[
            target
        ],
        return_counts=True,
    )

    semantic_summary = {
        int(
            tag
        ):
            int(
                count
            )
        for (
            tag,
            count,
        )
        in zip(
            observed_tags.tolist(),
            observed_tag_counts.tolist(),
        )
    }

    observed_ids, observed_id_counts = np.unique(
        instance_id[
            target
        ],
        return_counts=True,
    )

    instance_summary = [
        (
            int(
                instance_value
            ),
            int(
                count
            ),
        )
        for (
            instance_value,
            count,
        )
        in zip(
            observed_ids.tolist(),
            observed_id_counts.tolist(),
        )
    ]

    instance_summary.sort(
        key=lambda item: item[
            1
        ],
        reverse=True,
    )

    return (
        semantic_summary,
        instance_summary,
    )




# ============================================================
# Crop / sprite
# ============================================================

def make_sprite_crop(
    rgb,
    mask,
    instance_rgb,
    margin_px,
):

    ys, xs = np.where(
        mask > 0
    )

    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):

        raise RuntimeError(
            "Cannot crop empty mask."
        )

    image_h, image_w = (
        mask.shape[
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
            margin_px
        ),
    )

    y1 = max(
        0,
        int(
            ys.min()
        )
        -
        int(
            margin_px
        ),
    )

    x2 = min(
        image_w
        -
        1,
        int(
            xs.max()
        )
        +
        int(
            margin_px
        ),
    )

    y2 = min(
        image_h
        -
        1,
        int(
            ys.max()
        )
        +
        int(
            margin_px
        ),
    )

    crop_rgb = rgb[
        y1:
        y2 + 1,
        x1:
        x2 + 1,
    ].copy()

    crop_mask = mask[
        y1:
        y2 + 1,
        x1:
        x2 + 1,
    ].copy()

    crop_instance = instance_rgb[
        y1:
        y2 + 1,
        x1:
        x2 + 1,
    ].copy()

    rgba = np.dstack(
        [
            crop_rgb,
            crop_mask,
        ]
    ).astype(
        np.uint8
    )

    crop_ys, crop_xs = np.where(
        crop_mask > 0
    )

    bottom_y = int(
        crop_ys.max()
    )

    bottom_xs = crop_xs[
        crop_ys
        ==
        bottom_y
    ]

    anchor = {
        "x":
            float(
                np.mean(
                    bottom_xs
                )
            ),

        "y":
            float(
                bottom_y
            ),
    }

    return (
        crop_rgb,
        crop_mask,
        crop_instance,
        rgba,
        [
            int(
                x1
            ),
            int(
                y1
            ),
            int(
                x2
            ),
            int(
                y2
            ),
        ],
        anchor,
    )


def make_debug_image(
    rgb,
    bbox,
    mask,
    angle,
    distance,
    elevation,
    semantic_tag,
    selected_ids,
):

    debug = (
        rgb.copy()
    )

    (
        x1,
        y1,
        x2,
        y2,
    ) = [
        int(v)
        for v in bbox
    ]

    cv2.rectangle(
        debug,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (
            255,
            0,
            0,
        ),
        2,
    )

    binary = (
        mask > 0
    ).astype(
        np.uint8
    )

    contour_result = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(
        contour_result
    ) == 2:

        contours = contour_result[
            0
        ]

    else:

        contours = contour_result[
            1
        ]

    cv2.drawContours(
        debug,
        contours,
        -1,
        (
            0,
            255,
            0,
        ),
        2,
    )

    text = (
        "a={:03d} d={:.1f} e={:.1f} tag={} ids={}".format(
            int(
                angle
            ),
            float(
                distance
            ),
            float(
                elevation
            ),
            int(
                semantic_tag
            ),
            ",".join(
                str(v)
                for v in selected_ids[
                    :8
                ]
            ),
        )
    )

    cv2.putText(
        debug,
        text,
        (
            20,
            35,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return debug


# ============================================================
# Generation fingerprint / metadata
# ============================================================

def build_generation_signature(
    args,
    resolved_target_height_m,
    target_height_mode,
):

    semantic_mode, explicit_semantic_tag = (
        parse_semantic_tag_request(
            args.semantic_tag,
            args.asset_class,
        )
    )

    return {
        "generator_schema_version":
            GENERATOR_SCHEMA_VERSION,

        "mask_source":
            MASK_SOURCE,

        "mask_postprocessing":
            MASK_POSTPROCESSING,

        "mask_extraction":
            "per_view_actor_present_vs_actor_moved_native_annotation_difference",

        "background_settle_ticks":
            int(
                args.background_settle_ticks
            ),

        "hide_environment_objects":
            bool(
                args.hide_environment_objects
            ),

        "asset_id":
            str(
                args.asset_id
            ),

        "asset_class":
            str(
                args.asset_class
            ),

        "actor_blueprint":
            str(
                args.actor_blueprint
            ),

        "semantic_tag_mode":
            semantic_mode,

        "semantic_tag_request":
            str(
                args.semantic_tag
            ),

        "resolved_semantic_tag":
            int(
                explicit_semantic_tag
            ),

        "color_request":
            str(
                args.color
            ),

        "required_map":
            str(
                args.required_map
            ),

        "world_lifecycle":
            "reuse_current_world_clean_idempotently",

        "image_width_px":
            int(
                args.image_width
            ),

        "image_height_px":
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

        "target_height_mode":
            str(
                target_height_mode
            ),

        "distance_reference":
            (
                "exact_physical_bbox_center"
                if str(target_height_mode) == "auto_bbox_center_z"
                else "legacy_actor_origin_xy_explicit_height"
            ),

        "target_height_request":
            str(
                args.target_height
            ),

        "resolved_target_height_m":
            float(
                resolved_target_height_m
            ),

        "angles_deg":
            [
                int(v) % 360
                for v in args.angles
            ],

        "distances_m":
            [
                float(v)
                for v in args.distances
            ],

        "elevations_deg":
            [
                float(v)
                for v in args.elevations
            ],

        "crop_margin_px":
            int(
                args.crop_margin_px
            ),

        "projected_bbox_tolerance_px":
            float(
                args.projected_bbox_tolerance_px
            ),

        "fixed_delta_seconds":
            float(
                args.fixed_delta_seconds
            ),

        "settle_ticks":
            int(
                args.settle_ticks
            ),

        "clean_world":
            bool(
                args.hide_environment_objects
            ),

        "clean_world_strategy":
            (
                "disable_environment_objects_except_sky_no_map_layer_unload"
                if args.hide_environment_objects
                else "high_altitude_capture_native_annotation_difference"
            ),

        "weather":
            "ClearNoon",
    }


def initialize_or_validate_generation_config(
    output_dir,
    args,
    resolved_target_height_m,
    target_height_mode,
    runtime,
):

    output_dir = Path(
        output_dir
    )

    path = (
        output_dir
        /
        "generation_config.json"
    )

    signature = build_generation_signature(
        args,
        resolved_target_height_m,
        target_height_mode,
    )

    dataset_exists = output_has_existing_dataset(
        output_dir
    )

    if (
        dataset_exists
        and
        not args.resume
    ):

        raise RuntimeError(
            "Output already contains asset-bank data:\n"
            "  {}\n"
            "Delete it first or use --resume.".format(
                output_dir
            )
        )

    if path.exists():

        with open(
            str(
                path
            ),
            "r",
            encoding="utf-8",
        ) as f:

            existing = json.load(
                f
            )

        if (
            existing.get(
                "generation"
            )
            !=
            signature
        ):

            raise RuntimeError(
                "Generation configuration mismatch.\n\n"
                "Existing:\n{}\n\nRequested:\n{}".format(
                    json.dumps(
                        existing.get(
                            "generation"
                        ),
                        indent=2,
                    ),
                    json.dumps(
                        signature,
                        indent=2,
                    ),
                )
            )

        if (
            existing.get(
                "runtime"
            )
            !=
            runtime
        ):

            raise RuntimeError(
                "CARLA runtime mismatch.\n\n"
                "Existing:\n{}\n\nCurrent:\n{}".format(
                    json.dumps(
                        existing.get(
                            "runtime"
                        ),
                        indent=2,
                    ),
                    json.dumps(
                        runtime,
                        indent=2,
                    ),
                )
            )

        print(
            "[NativeAssetBank] generation_config.json matches."
        )

        return path

    document = {
        "generation":
            signature,

        "runtime":
            runtime,
    }

    atomic_write_json(
        path,
        document,
    )

    print(
        "[NativeAssetBank] Created:",
        path,
    )

    return path


def build_asset_metadata(
    actor,
    args,
    actual_color,
    resolved_target_height_m,
    target_height_mode,
    runtime,
    K,
    resolved_semantic_tag=None,
):

    bbox = (
        actor.bounding_box
    )

    bbox_rotation = getattr(
        bbox,
        "rotation",
        carla.Rotation(),
    )

    return {
        "schema_version":
            ASSET_METADATA_SCHEMA_VERSION,

        "asset_id":
            str(
                args.asset_id
            ),

        "asset_class":
            str(
                args.asset_class
            ),

        "carla_blueprint":
            str(
                actor.type_id
            ),

        "semantic_tag_request":
            str(
                args.semantic_tag
            ),

        "resolved_semantic_tag":
            (
                int(
                    resolved_semantic_tag
                )
                if resolved_semantic_tag is not None
                else None
            ),

        "actor_semantic_tags":
            [
                int(v)
                for v in getattr(
                    actor,
                    "semantic_tags",
                    []
                )
            ],

        "color_request":
            str(
                args.color
            ),

        "actual_color":
            (
                str(
                    actual_color
                )
                if actual_color is not None
                else None
            ),

        "mask":
            {
                "source":
                    MASK_SOURCE,

                "postprocessing":
                    MASK_POSTPROCESSING,

                "extraction":
                    "per_view_actor_present_vs_actor_moved_native_annotation_difference",
            },

        "physical_bbox":
            {
                "extent_x_m":
                    float(
                        bbox.extent.x
                    ),

                "extent_y_m":
                    float(
                        bbox.extent.y
                    ),

                "extent_z_m":
                    float(
                        bbox.extent.z
                    ),

                "size_x_m":
                    float(
                        2.0
                        *
                        bbox.extent.x
                    ),

                "size_y_m":
                    float(
                        2.0
                        *
                        bbox.extent.y
                    ),

                "size_z_m":
                    float(
                        2.0
                        *
                        bbox.extent.z
                    ),

                "length_m":
                    float(
                        2.0
                        *
                        bbox.extent.x
                    ),

                "width_m":
                    float(
                        2.0
                        *
                        bbox.extent.y
                    ),

                "height_m":
                    float(
                        2.0
                        *
                        bbox.extent.z
                    ),

                "local_center_x_m":
                    float(
                        bbox.location.x
                    ),

                "local_center_y_m":
                    float(
                        bbox.location.y
                    ),

                "local_center_z_m":
                    float(
                        bbox.location.z
                    ),

                "local_bottom_z_m":
                    float(
                        bbox.location.z
                        -
                        bbox.extent.z
                    ),

                "local_top_z_m":
                    float(
                        bbox.location.z
                        +
                        bbox.extent.z
                    ),

                "local_rotation_pitch_deg":
                    float(
                        bbox_rotation.pitch
                    ),

                "local_rotation_yaw_deg":
                    float(
                        bbox_rotation.yaw
                    ),

                "local_rotation_roll_deg":
                    float(
                        bbox_rotation.roll
                    ),
            },

        "capture":
            {
                "image_width_px":
                    int(
                        args.image_width
                    ),

                "image_height_px":
                    int(
                        args.image_height
                    ),

                "fov_deg":
                    float(
                        args.fov
                    ),

                "camera_fx_px":
                    float(
                        K[
                            0,
                            0
                        ]
                    ),

                "camera_fy_px":
                    float(
                        K[
                            1,
                            1
                        ]
                    ),

                "camera_cx_px":
                    float(
                        K[
                            0,
                            2
                        ]
                    ),

                "camera_cy_px":
                    float(
                        K[
                            1,
                            2
                        ]
                    ),

                "capture_altitude_m":
                    float(
                        args.capture_altitude
                    ),

                "target_height_mode":
                    str(
                        target_height_mode
                    ),

                "distance_reference":
                    (
                        "exact_physical_bbox_center"
                        if str(target_height_mode) == "auto_bbox_center_z"
                        else "legacy_actor_origin_xy_explicit_height"
                    ),

                "target_height_request":
                    str(
                        args.target_height
                    ),

                "resolved_target_height_m":
                    float(
                        resolved_target_height_m
                    ),

                "angles_deg":
                    [
                        int(v) % 360
                        for v in args.angles
                    ],

                "distances_m":
                    [
                        float(v)
                        for v in args.distances
                    ],

                "elevations_deg":
                    [
                        float(v)
                        for v in args.elevations
                    ],
            },

        "runtime":
            runtime,
    }


# ============================================================
# Resume
# ============================================================

def load_checkpoint_records(
    csv_path,
    output_dir,
):

    csv_path = Path(
        csv_path
    )

    output_dir = Path(
        output_dir
    )

    if not csv_path.exists():

        return (
            [],
            set(),
        )

    records = []
    completed_keys = set()

    with open(
        str(
            csv_path
        ),
        "r",
        encoding="utf-8",
        newline="",
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

            except Exception:

                continue

            paths = expected_view_paths(
                output_dir,
                angle,
                distance,
                elevation,
            )

            if not all(
                file_is_nonempty(
                    path
                )
                for path in paths.values()
            ):

                continue

            row[
                "rgba_path"
            ] = str(
                paths[
                    "rgba"
                ].resolve()
            )

            row[
                "rgb_path"
            ] = str(
                paths[
                    "rgb"
                ].resolve()
            )

            row[
                "mask_path"
            ] = str(
                paths[
                    "mask"
                ].resolve()
            )

            row[
                "instance_path"
            ] = str(
                paths[
                    "instance"
                ].resolve()
            )

            row[
                "debug_path"
            ] = str(
                paths[
                    "debug"
                ].resolve()
            )

            row[
                "rgba_relpath"
            ] = paths[
                "rgba"
            ].relative_to(
                output_dir
            ).as_posix()

            row[
                "rgb_relpath"
            ] = paths[
                "rgb"
            ].relative_to(
                output_dir
            ).as_posix()

            row[
                "mask_relpath"
            ] = paths[
                "mask"
            ].relative_to(
                output_dir
            ).as_posix()

            row[
                "instance_relpath"
            ] = paths[
                "instance"
            ].relative_to(
                output_dir
            ).as_posix()

            row[
                "debug_relpath"
            ] = paths[
                "debug"
            ].relative_to(
                output_dir
            ).as_posix()

            key = make_view_key(
                angle,
                distance,
                elevation,
            )

            records.append(
                row
            )

            completed_keys.add(
                key
            )

    return (
        records,
        completed_keys,
    )


# ============================================================
# Contact sheets / validation
# ============================================================

def make_contact_sheet(
    output_dir,
    records,
    angle,
    distances,
    elevations,
):

    output_dir = Path(
        output_dir
    )

    contact_dir = (
        output_dir
        /
        "contact_sheets"
    )

    contact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            float(
                record[
                    "angle_deg"
                ]
            )
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

    for row_index, distance in enumerate(
        distances
    ):

        for column_index, elevation in enumerate(
            elevations
        ):

            x0 = (
                column_index
                *
                cell_w
            )

            y0 = (
                row_index
                *
                (
                    cell_h
                    +
                    label_h
                )
            )

            draw.text(
                (
                    x0 + 8,
                    y0 + 8,
                ),
                "d={:g} m  elev={:g} deg".format(
                    distance,
                    elevation,
                ),
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
                    cell_w
                    -
                    20,
                    cell_h
                    -
                    20,
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
        contact_dir
        /
        "contact_sheet_angle_{:03d}.png".format(
            int(
                angle
            ) % 360
        )
    )

    atomic_save_pil(
        output_path,
        sheet,
    )

    print(
        "[NativeAssetBank] Contact sheet:",
        output_path,
    )


def final_validate_dataset(
    output_dir,
    args,
    records,
):

    output_dir = Path(
        output_dir
    )

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

    expected_keys.difference_update(
        INTENTIONALLY_NONPROJECTABLE_KEYS
    )

    record_map = {}

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

            record_map[
                key
            ] = record

        except Exception:

            pass

    actual_keys = set(
        record_map.keys()
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
            output_dir,
            angle,
            distance,
            elevation,
        )

        for name, path in paths.items():

            if not file_is_nonempty(
                path
            ):

                missing_files.append(
                    (
                        angle,
                        distance,
                        elevation,
                        name,
                        str(
                            path
                        ),
                    )
                )

    sorted_records = [
        record_map[
            key
        ]
        for key in sorted(
            expected_keys
        )
        if key in record_map
    ]

    print()
    print("=" * 80)
    print(
        "[NativeAssetBank] FINAL DATASET VALIDATION"
    )
    print("=" * 80)

    print(
        "[NativeAssetBank] expected views:",
        len(
            expected_keys
        ),
    )

    print(
        "[NativeAssetBank] intentionally non-projectable views:",
        len(INTENTIONALLY_NONPROJECTABLE_KEYS),
    )

    print(
        "[NativeAssetBank] completed views:",
        len(
            actual_keys
        ),
    )

    print(
        "[NativeAssetBank] missing views:",
        len(
            missing_keys
        ),
    )

    print(
        "[NativeAssetBank] extra views:",
        len(
            extra_keys
        ),
    )

    print(
        "[NativeAssetBank] missing files:",
        len(
            missing_files
        ),
    )

    if (
        missing_keys
        or
        extra_keys
        or
        missing_files
    ):

        raise RuntimeError(
            "Final dataset validation failed."
        )

    print(
        "[NativeAssetBank] DATASET VALIDATION PASSED"
    )
    print("=" * 80)

    return sorted_records


def write_qa_summary(
    output_dir,
    records,
):

    failed = [
        row
        for row in records
        if str(
            row.get(
                "qa_pass",
                ""
            )
        ).lower()
        not in [
            "true",
            "1",
        ]
    ]

    flag_counts = Counter()
    warning_counts = Counter()

    for row in records:

        flags = str(
            row.get(
                "qa_flags",
                ""
            )
        ).strip()

        if flags:

            for flag in flags.split(
                "|"
            ):

                if flag:

                    flag_counts[
                        flag
                    ] += 1

        warnings = str(
            row.get(
                "qa_warnings",
                ""
            )
        ).strip()

        if warnings:

            for warning in warnings.split(
                "|"
            ):

                if warning:

                    warning_counts[
                        warning
                    ] += 1

    summary = {
        "views":
            int(
                len(
                    records
                )
            ),

        "qa_passed":
            int(
                len(
                    records
                )
                -
                len(
                    failed
                )
            ),

        "qa_flagged":
            int(
                len(
                    failed
                )
            ),

        "flag_counts":
            dict(
                flag_counts
            ),

        "warning_counts":
            dict(
                warning_counts
            ),
    }

    atomic_write_json(
        Path(
            output_dir
        )
        /
        "qa_summary.json",
        summary,
    )

    print(
        "[NativeAssetBank] QA:",
        summary[
            "qa_passed"
        ],
        "passed,",
        summary[
            "qa_flagged"
        ],
        "flagged, warnings=",
        summary[
            "warning_counts"
        ],
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    semantic_mode, explicit_semantic_tag = (
        parse_semantic_tag_request(
            args.semantic_tag,
            args.asset_class,
        )
    )

    output_dir = (
        Path(
            args.output_root
        )
        /
        str(
            args.asset_id
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for subdir in [
        "rgba",
        "rgb",
        "mask",
        "instance",
        "debug",
        "contact_sheets",
    ]:

        (
            output_dir
            /
            subdir
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    csv_path = (
        output_dir
        /
        "view_matrix.csv"
    )

    asset_metadata_path = (
        output_dir
        /
        "asset_metadata.json"
    )

    world = None
    original_settings = None

    actor = None
    rgb_camera = None
    instance_camera = None

    rgb_queue = queue.Queue()
    instance_queue = queue.Queue()

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

        # IMPORTANT:
        # Do not load/reload the CARLA world between asset banks.
        #
        # Repeated Town10HD_Opt reloads after unloading map layers can be
        # expensive and, in practice, can destabilize/crash the UE server.
        # The asset generator does not need a fresh episode: it removes
        # dynamic actors and clean-world preparation is intentionally
        # idempotent. Therefore one CARLA process can generate many banks
        # sequentially.
        world = client.get_world()

        print(
            "[NativeAssetBank] Reusing current CARLA world "
            "(no map reload)."
        )

        map_full_name = str(
            world.get_map().name
        )

        map_short_name = (
            map_full_name
            .replace(
                "\\",
                "/",
            )
            .split(
                "/"
            )[
                -1
            ]
        )

        if (
            map_short_name
            !=
            str(
                args.required_map
            )
        ):

            raise RuntimeError(
                "Wrong CARLA map.\n"
                "Required: {}\n"
                "Current : {}".format(
                    args.required_map,
                    map_short_name,
                )
            )

        runtime = {
            "map_short_name":
                map_short_name,

            "map_full_name":
                map_full_name,

            "carla_client_version":
                str(
                    client.get_client_version()
                ),

            "carla_server_version":
                str(
                    client.get_server_version()
                ),
        }

        original_settings = (
            world.get_settings()
        )

        print()
        print("=" * 80)
        print(
            "HE PRODUCTION CARLA-NATIVE ASSET BANK"
        )
        print("=" * 80)

        print(
            "[NativeAssetBank] asset:",
            args.asset_id,
        )

        print(
            "[NativeAssetBank] class:",
            args.asset_class,
        )

        print(
            "[NativeAssetBank] blueprint:",
            args.actor_blueprint,
        )

        print(
            "[NativeAssetBank] requested views:",
            (
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
            ),
        )

        print(
            "[NativeAssetBank] CARLA:",
            runtime[
                "carla_client_version"
            ],
            "/",
            runtime[
                "carla_server_version"
            ],
        )

        print(
            "[NativeAssetBank] map:",
            runtime[
                "map_full_name"
            ],
        )

        gen.setup_synchronous_mode(
            world,
            args.fixed_delta_seconds,
        )

        world.set_weather(
            carla.WeatherParameters.ClearNoon
        )

        for _ in range(
            2
        ):

            world.tick()

        gen.clear_existing_dynamic_actors(
            world
        )

        prepare_clean_capture_world(
            world,
            hide_environment_objects=(
                args.hide_environment_objects
            ),
            settle_ticks=4,
        )

        blueprint_library = (
            world.get_blueprint_library()
        )

        (
            actor,
            base_location,
            base_yaw,
        ) = spawn_target_actor(
            world,
            blueprint_library,
            args,
        )

        actual_color = actor.attributes.get(
            "color",
            None,
        )

        (
            target_height_mode,
            resolved_target_height_m,
        ) = resolve_target_height(
            actor,
            args.target_height,
        )

        print(
            "[NativeAssetBank] target height:",
            target_height_mode,
            "=",
            resolved_target_height_m,
            "m",
        )

        initialize_or_validate_generation_config(
            output_dir=output_dir,
            args=args,
            resolved_target_height_m=(
                resolved_target_height_m
            ),
            target_height_mode=(
                target_height_mode
            ),
            runtime=runtime,
        )

        K = gen.build_camera_intrinsics(
            width=args.image_width,
            height=args.image_height,
            fov_degrees=args.fov,
        )

        # Existing metadata may contain the auto-detected semantic tag.
        # "auto" is resolved deterministically from --asset-class.
        # For CARLA 0.9.15: vehicle=14, truck=15, bus=16, pedestrian=12.
        resolved_semantic_tag = int(
            explicit_semantic_tag
        )

        if (
            args.resume
            and
            asset_metadata_path.exists()
        ):

            with open(
                str(
                    asset_metadata_path
                ),
                "r",
                encoding="utf-8",
            ) as f:

                existing_asset_metadata = json.load(
                    f
                )

            existing_resolved_tag = (
                existing_asset_metadata.get(
                    "resolved_semantic_tag"
                )
            )

            if (
                resolved_semantic_tag is None
                and
                existing_resolved_tag is not None
            ):

                resolved_semantic_tag = int(
                    existing_resolved_tag
                )

            elif (
                resolved_semantic_tag is not None
                and
                existing_resolved_tag is not None
                and
                int(
                    existing_resolved_tag
                )
                !=
                int(
                    resolved_semantic_tag
                )
            ):

                raise RuntimeError(
                    "Resolved semantic tag differs from existing asset metadata."
                )

        asset_metadata = build_asset_metadata(
            actor=actor,
            args=args,
            actual_color=actual_color,
            resolved_target_height_m=(
                resolved_target_height_m
            ),
            target_height_mode=(
                target_height_mode
            ),
            runtime=runtime,
            K=K,
            resolved_semantic_tag=(
                resolved_semantic_tag
            ),
        )

        atomic_write_json(
            asset_metadata_path,
            asset_metadata,
        )

        print(
            "[NativeAssetBank] physical bbox:"
        )

        print(
            "[NativeAssetBank]   length:",
            asset_metadata[
                "physical_bbox"
            ][
                "length_m"
            ],
            "m",
        )

        print(
            "[NativeAssetBank]   width :",
            asset_metadata[
                "physical_bbox"
            ][
                "width_m"
            ],
            "m",
        )

        print(
            "[NativeAssetBank]   height:",
            asset_metadata[
                "physical_bbox"
            ][
                "height_m"
            ],
            "m",
        )

        (
            records,
            completed_keys,
        ) = load_checkpoint_records(
            csv_path,
            output_dir,
        )

        if args.resume:

            print(
                "[NativeAssetBank] completed views loaded:",
                len(
                    completed_keys
                ),
            )

        calibration_angle = int(
            args.angles[
                0
            ]
        ) % 360

        calibration_actor_yaw = actor_yaw_for_angle(
            base_yaw,
            calibration_angle,
        )

        calibration_actor_tf = carla.Transform(
            base_location,
            carla.Rotation(
                pitch=0.0,
                yaw=float(
                    calibration_actor_yaw
                ),
                roll=0.0,
            ),
        )

        actor.set_transform(
            calibration_actor_tf
        )

        calibration_target_location = resolve_capture_target_location(
            actor=actor,
            base_location=base_location,
            target_height_mode=(
                target_height_mode
            ),
            resolved_target_height_m=(
                resolved_target_height_m
            ),
            actor_transform=(
                calibration_actor_tf
            ),
        )

        first_camera_tf = build_view_camera_transform_from_target(
            target_location=(
                calibration_target_location
            ),
            base_yaw_deg=base_yaw,
            distance_m=args.distances[
                0
            ],
            elevation_deg=args.elevations[
                0
            ],
        )

        (
            rgb_camera,
            instance_camera,
        ) = spawn_cameras(
            world,
            blueprint_library,
            first_camera_tf,
            args,
        )

        rgb_camera.listen(
            rgb_queue.put
        )

        instance_camera.listen(
            instance_queue.put
        )

        for _ in range(
            5
        ):

            world.tick()

            gen.flush_queue(
                rgb_queue
            )

            gen.flush_queue(
                instance_queue
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

        parking_location = carla.Location(
            x=float(
                base_location.x
            )
            +
            500.0,
            y=float(
                base_location.y
            )
            +
            500.0,
            z=float(
                base_location.z
            )
            +
            200.0,
        )

        for distance_m in args.distances:

            for elevation_deg in args.elevations:

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
                            "[NativeAssetBank] "
                            "{:04d}/{:04d} SKIP "
                            "a={:03d} d={:5.1f}m e={:5.1f}deg".format(
                                capture_index,
                                total,
                                angle,
                                distance_m,
                                elevation_deg,
                            )
                        )

                        continue

                    actor_yaw = actor_yaw_for_angle(
                        base_yaw,
                        angle,
                    )

                    current_actor_tf = carla.Transform(
                        base_location,
                        carla.Rotation(
                            pitch=0.0,
                            yaw=float(
                                actor_yaw
                            ),
                            roll=0.0,
                        ),
                    )

                    actor.set_transform(
                        current_actor_tf
                    )

                    capture_target_location = resolve_capture_target_location(
                        actor=actor,
                        base_location=base_location,
                        target_height_mode=(
                            target_height_mode
                        ),
                        resolved_target_height_m=(
                            resolved_target_height_m
                        ),
                        actor_transform=(
                            current_actor_tf
                        ),
                    )

                    camera_tf = build_view_camera_transform_from_target(
                        target_location=(
                            capture_target_location
                        ),
                        base_yaw_deg=base_yaw,
                        distance_m=distance_m,
                        elevation_deg=elevation_deg,
                    )

                    rgb_camera.set_transform(
                        camera_tf
                    )

                    instance_camera.set_transform(
                        camera_tf
                    )

                    gen.flush_queue(
                        rgb_queue
                    )

                    gen.flush_queue(
                        instance_queue
                    )

                    for _ in range(
                        max(
                            1,
                            int(
                                args.settle_ticks
                            ),
                        )
                    ):

                        world.tick()

                        gen.flush_queue(
                            rgb_queue
                        )

                        gen.flush_queue(
                            instance_queue
                        )

                    # ------------------------------------------------
                    # ACTOR-PRESENT capture
                    # ------------------------------------------------

                    present_frame = world.tick()

                    rgb_image = gen.wait_for_frame(
                        rgb_queue,
                        present_frame,
                    )

                    present_instance_image = gen.wait_for_frame(
                        instance_queue,
                        present_frame,
                    )

                    rgb = gen.carla_rgb_to_array(
                        rgb_image
                    )

                    (
                        present_semantic_tags,
                        present_instance_id,
                        present_instance_rgb,
                    ) = decode_instance_image(
                        present_instance_image
                    )

                    (
                        bbox,
                        projection_metadata,
                    ) = compute_actor_projected_bbox_native(
                        actor=actor,
                        camera_actor=rgb_camera,
                        K=K,
                        image_width=args.image_width,
                        image_height=args.image_height,
                    )

                    if bbox is None:

                        INTENTIONALLY_NONPROJECTABLE_KEYS.add(view_key)
                        print(
                            "[NativeAssetBank] "
                            "{:04d}/{:04d} OFFSCREEN "
                            "a={:03d} d={:5.1f}m e={:5.1f}deg".format(
                                capture_index,
                                total,
                                angle,
                                distance_m,
                                elevation_deg,
                            )
                        )
                        continue

                    # ------------------------------------------------
                    # ACTOR-ABSENT annotation capture.
                    #
                    # Same camera. Same static CARLA world. Only the
                    # target actor is moved out of view.
                    # ------------------------------------------------

                    actor.set_transform(
                        carla.Transform(
                            parking_location,
                            current_actor_tf.rotation,
                        )
                    )

                    gen.flush_queue(
                        rgb_queue
                    )

                    gen.flush_queue(
                        instance_queue
                    )

                    for _ in range(
                        max(
                            1,
                            int(
                                args.background_settle_ticks
                            ),
                        )
                    ):

                        world.tick()

                        gen.flush_queue(
                            rgb_queue
                        )

                        gen.flush_queue(
                            instance_queue
                        )

                    background_frame = world.tick()

                    _background_rgb_image = gen.wait_for_frame(
                        rgb_queue,
                        background_frame,
                    )

                    background_instance_image = gen.wait_for_frame(
                        instance_queue,
                        background_frame,
                    )

                    (
                        _background_semantic_tags,
                        _background_instance_id,
                        background_instance_rgb,
                    ) = decode_instance_image(
                        background_instance_image
                    )

                    # Restore actor immediately. Geometry metadata below
                    # therefore describes the real requested view.
                    actor.set_transform(
                        current_actor_tf
                    )

                    # After restoring the actor, re-apply the actor-present pose
                    # to the camera/sensors and settle briefly before computing any
                    # geometric metadata. This guarantees that projected bbox / depth
                    # measurements correspond to the requested view, not to the parked
                    # background-capture pose.
                    rgb_camera.set_transform(
                        camera_tf
                    )

                    instance_camera.set_transform(
                        camera_tf
                    )

                    gen.flush_queue(
                        rgb_queue
                    )

                    gen.flush_queue(
                        instance_queue
                    )

                    for _ in range(
                        max(
                            1,
                            int(
                                args.settle_ticks
                            ),
                        )
                    ):

                        world.tick()

                        gen.flush_queue(
                            rgb_queue
                        )

                        gen.flush_queue(
                            instance_queue
                        )

                    # Freeze actor-present geometry NOW, before any later step can
                    # disturb the actor/camera pose. These values are the physical
                    # metadata that HE will use downstream.
                    (
                        present_bbox,
                        present_projection_metadata,
                    ) = compute_actor_projected_bbox_native(
                        actor=actor,
                        camera_actor=rgb_camera,
                        K=K,
                        image_width=args.image_width,
                        image_height=args.image_height,
                    )

                    if present_bbox is None:

                        raise RuntimeError(
                            "Physical actor bbox cannot be re-projected after actor restoration: "
                            "a={} d={} e={}; projection={}".format(
                                angle,
                                distance_m,
                                elevation_deg,
                                present_projection_metadata,
                            )
                        )

                    mask = build_native_frame_difference_mask(
                        present_instance_rgb=(
                            present_instance_rgb
                        ),
                        background_instance_rgb=(
                            background_instance_rgb
                        ),
                        present_instance_id=(
                            present_instance_id
                        ),
                    )

                    (
                        observed_semantic_summary,
                        instance_id_counts,
                    ) = summarize_instance_annotation_on_mask(
                        semantic_tags=(
                            present_semantic_tags
                        ),
                        instance_id=(
                            present_instance_id
                        ),
                        mask=mask,
                    )

                    selected_ids = [
                        int(
                            instance_value
                        )
                        for (
                            instance_value,
                            _count,
                        )
                        in instance_id_counts
                        if int(
                            instance_value
                        )
                        !=
                        0
                    ]

                    (
                        crop_rgb,
                        crop_mask,
                        crop_instance,
                        rgba,
                        crop_box,
                        anchor,
                    ) = make_sprite_crop(
                        rgb=rgb,
                        mask=mask,
                        instance_rgb=(
                            present_instance_rgb
                        ),
                        margin_px=args.crop_margin_px,
                    )

                    geometry = compute_view_geometry_metadata(
                        actor=actor,
                        camera_actor=rgb_camera,
                        K=K,
                        bbox_2d=present_bbox,
                        crop_box=crop_box,
                        crop_mask=crop_mask,
                        anchor=anchor,
                        image_width=args.image_width,
                        image_height=args.image_height,
                    )

                    geometry.update(
                        present_projection_metadata
                    )

                    # Separate physical projection reference from actual
                    # visible alpha dimensions for downstream HE rendering.
                    geometry[
                        "he_reference_projected_width_px"
                    ] = float(
                        projection_metadata[
                            "projected_bbox_raw_width_px"
                        ]
                    )

                    geometry[
                        "he_reference_projected_height_px"
                    ] = float(
                        projection_metadata[
                            "projected_bbox_raw_height_px"
                        ]
                    )

                    qa = build_view_qa(
                        geometry
                    )

                    qa_warning_list = [
                        value
                        for value in str(
                            qa.get(
                                "qa_warnings",
                                ""
                            )
                        ).split(
                            "|"
                        )
                        if value
                    ]

                    projected_bbox_overflow_px = float(
                        projection_metadata[
                            "projected_bbox_max_overflow_px"
                        ]
                    )

                    if (
                        projected_bbox_overflow_px
                        >
                        float(
                            args.projected_bbox_tolerance_px
                        )
                    ):

                        qa[
                            "qa_pass"
                        ] = False

                        qa_flag_list = [
                            value
                            for value in str(
                                qa.get(
                                    "qa_flags",
                                    ""
                                )
                            ).split(
                                "|"
                            )
                            if value
                        ]

                        qa_flag_list.append(
                            "projected_bbox_outside_capture"
                        )

                        qa[
                            "qa_flags"
                        ] = "|".join(
                            dict.fromkeys(
                                qa_flag_list
                            )
                        )

                        qa[
                            "qa_flag_count"
                        ] = int(
                            len(
                                [
                                    value
                                    for value in qa[
                                        "qa_flags"
                                    ].split(
                                        "|"
                                    )
                                    if value
                                ]
                            )
                        )
                    if (
                        projected_bbox_overflow_px > 0.0
                        and
                        projected_bbox_overflow_px
                        <=
                        float(
                            args.projected_bbox_tolerance_px
                        )
                    ):

                        qa_warning_list.append(
                            "minor_projected_bbox_overflow"
                        )
                    qa[
                        "qa_warning_count"
                    ] = int(
                        len(
                            qa_warning_list
                        )
                    )

                    qa[
                        "qa_warnings"
                    ] = "|".join(
                        qa_warning_list
                    )

                    paths = expected_view_paths(
                        output_dir,
                        angle,
                        distance_m,
                        elevation_deg,
                    )

                    debug = make_debug_image(
                        rgb=rgb,
                        bbox=bbox,
                        mask=mask,
                        angle=angle,
                        distance=distance_m,
                        elevation=elevation_deg,
                        semantic_tag=(
                            resolved_semantic_tag
                        ),
                        selected_ids=selected_ids,
                    )

                    save_rgba(
                        paths[
                            "rgba"
                        ],
                        rgba,
                    )

                    save_rgb(
                        paths[
                            "rgb"
                        ],
                        crop_rgb,
                    )

                    save_mask(
                        paths[
                            "mask"
                        ],
                        crop_mask,
                    )

                    save_rgb(
                        paths[
                            "instance"
                        ],
                        crop_instance,
                    )

                    save_rgb(
                        paths[
                            "debug"
                        ],
                        debug,
                    )

                    id_summary = ";".join(
                        "{}:{}".format(
                            instance_value,
                            count,
                        )
                        for (
                            instance_value,
                            count,
                        )
                        in instance_id_counts
                    )

                    semantic_summary_text = ";".join(
                        "{}:{}".format(
                            tag,
                            count,
                        )
                        for (
                            tag,
                            count,
                        )
                        in sorted(
                            observed_semantic_summary.items()
                        )
                    )

                    record = {
                        "schema_version":
                            VIEW_SCHEMA_VERSION,

                        "generation_status":
                            "complete",

                        "asset_id":
                            str(
                                args.asset_id
                            ),

                        "asset_class":
                            str(
                                args.asset_class
                            ),

                        "carla_blueprint":
                            str(
                                actor.type_id
                            ),

                        "carla_actor_id":
                            int(
                                actor.id
                            ),

                        "semantic_tag_expected":
                            int(
                                resolved_semantic_tag
                            ),

                        "actor_semantic_tags":
                            ",".join(
                                str(
                                    int(v)
                                )
                                for v in getattr(
                                    actor,
                                    "semantic_tags",
                                    []
                                )
                            ),

                        "observed_semantic_tags":
                            semantic_summary_text,

                        "mask_source":
                            MASK_SOURCE,

                        "mask_method":
                            "per_view_native_instance_frame_difference",

                        "view_index":
                            int(
                                capture_index
                            ),

                        "angle_deg":
                            int(
                                angle
                            ),

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

                        "actor_location_x_m":
                            float(
                                current_actor_tf.location.x
                            ),

                        "actor_location_y_m":
                            float(
                                current_actor_tf.location.y
                            ),

                        "actor_location_z_m":
                            float(
                                current_actor_tf.location.z
                            ),

                        "actor_pitch_deg":
                            float(
                                current_actor_tf.rotation.pitch
                            ),

                        "actor_yaw_deg":
                            float(
                                current_actor_tf.rotation.yaw
                            ),

                        "actor_roll_deg":
                            float(
                                current_actor_tf.rotation.roll
                            ),

                        "camera_location_x_m":
                            float(
                                camera_tf.location.x
                            ),

                        "camera_location_y_m":
                            float(
                                camera_tf.location.y
                            ),

                        "camera_location_z_m":
                            float(
                                camera_tf.location.z
                            ),

                        "camera_pitch_deg":
                            float(
                                camera_tf.rotation.pitch
                            ),

                        "camera_yaw_deg":
                            float(
                                camera_tf.rotation.yaw
                            ),

                        "camera_roll_deg":
                            float(
                                camera_tf.rotation.roll
                            ),

                        "target_height_m":
                            float(
                                resolved_target_height_m
                            ),

                        "image_width_px":
                            int(
                                args.image_width
                            ),

                        "image_height_px":
                            int(
                                args.image_height
                            ),

                        "fov_deg":
                            float(
                                args.fov
                            ),

                        "camera_fx_px":
                            float(
                                K[
                                    0,
                                    0
                                ]
                            ),

                        "camera_fy_px":
                            float(
                                K[
                                    1,
                                    1
                                ]
                            ),

                        "camera_cx_px":
                            float(
                                K[
                                    0,
                                    2
                                ]
                            ),

                        "camera_cy_px":
                            float(
                                K[
                                    1,
                                    2
                                ]
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

                        "instance_id_count":
                            int(
                                len(
                                    selected_ids
                                )
                            ),

                        "instance_ids":
                            ",".join(
                                str(v)
                                for v in selected_ids
                            ),

                        "instance_id_pixel_counts":
                            id_summary,

                        "native_difference_pixels":
                            int(
                                np.sum(
                                    mask > 0
                                )
                            ),

                        "rgba_path":
                            str(
                                paths[
                                    "rgba"
                                ].resolve()
                            ),

                        "rgb_path":
                            str(
                                paths[
                                    "rgb"
                                ].resolve()
                            ),

                        "mask_path":
                            str(
                                paths[
                                    "mask"
                                ].resolve()
                            ),

                        "instance_path":
                            str(
                                paths[
                                    "instance"
                                ].resolve()
                            ),

                        "debug_path":
                            str(
                                paths[
                                    "debug"
                                ].resolve()
                            ),

                        "rgba_relpath":
                            paths[
                                "rgba"
                            ].relative_to(
                                output_dir
                            ).as_posix(),

                        "rgb_relpath":
                            paths[
                                "rgb"
                            ].relative_to(
                                output_dir
                            ).as_posix(),

                        "mask_relpath":
                            paths[
                                "mask"
                            ].relative_to(
                                output_dir
                            ).as_posix(),

                        "instance_relpath":
                            paths[
                                "instance"
                            ].relative_to(
                                output_dir
                            ).as_posix(),

                        "debug_relpath":
                            paths[
                                "debug"
                            ].relative_to(
                                output_dir
                            ).as_posix(),
                    }

                    record.update(
                        geometry
                    )

                    record.update(
                        qa
                    )

                    records = [
                        r
                        for r in records
                        if make_view_key(
                            r[
                                "angle_deg"
                            ],
                            r[
                                "distance_m"
                            ],
                            r[
                                "elevation_deg"
                            ],
                        )
                        !=
                        view_key
                    ]

                    records.append(
                        record
                    )

                    completed_keys.add(
                        view_key
                    )

                    atomic_write_records_csv(
                        csv_path,
                        records,
                    )

                    print(
                        "[NativeAssetBank] "
                        "{:04d}/{:04d} "
                        "a={:03d} d={:5.1f}m e={:5.1f}deg "
                        "size={}x{} alpha={} "
                        "center={:.3f} near={:.3f} far={:.3f} "
                        "qa={}".format(
                            capture_index,
                            total,
                            angle,
                            distance_m,
                            elevation_deg,
                            rgba.shape[
                                1
                            ],
                            rgba.shape[
                                0
                            ],
                            geometry[
                                "alpha_area_px"
                            ],
                            geometry[
                                "bbox_center_distance_m"
                            ],
                            geometry[
                                "nearest_bbox_depth_m"
                            ],
                            geometry[
                                "farthest_bbox_depth_m"
                            ],
                            qa[
                                "qa_flags"
                            ]
                            if qa[
                                "qa_flags"
                            ]
                            else "PASS",
                        )
                    )

        records = final_validate_dataset(
            output_dir,
            args,
            records,
        )

        atomic_write_records_csv(
            csv_path,
            records,
        )

        write_qa_summary(
            output_dir,
            records,
        )

        # Final metadata write guarantees resolved semantic tag is stored.
        asset_metadata = build_asset_metadata(
            actor=actor,
            args=args,
            actual_color=actual_color,
            resolved_target_height_m=(
                resolved_target_height_m
            ),
            target_height_mode=(
                target_height_mode
            ),
            runtime=runtime,
            K=K,
            resolved_semantic_tag=(
                resolved_semantic_tag
            ),
        )

        atomic_write_json(
            asset_metadata_path,
            asset_metadata,
        )

        if not args.skip_contact_sheets:

            for angle in args.angles:

                make_contact_sheet(
                    output_dir=output_dir,
                    records=records,
                    angle=angle,
                    distances=args.distances,
                    elevations=args.elevations,
                )

        print()
        print("=" * 80)
        print(
            "[NativeAssetBank] DONE"
        )
        print(
            "[NativeAssetBank] views:",
            len(
                records
            ),
        )
        print(
            "[NativeAssetBank] output:",
            output_dir,
        )
        print("=" * 80)

    except Exception:

        traceback.print_exc()
        raise

    finally:

        print()
        print(
            "[NativeAssetBank] Cleaning up..."
        )

        for sensor in [
            rgb_camera,
            instance_camera,
        ]:

            if sensor is not None:

                try:

                    sensor.stop()

                except Exception:

                    pass

        for item in [
            rgb_camera,
            instance_camera,
            actor,
        ]:

            if item is not None:

                try:

                    item.destroy()

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
            "[NativeAssetBank] Cleanup complete."
        )


if __name__ == "__main__":

    main()
