"""
compare_carla_he_vehicle_geometry_v1.py

Static pixel-geometry calibration:

    physical CARLA Tesla Model 3
                vs
    HE Tesla sprite rendered with the CURRENT HE renderer

Purpose
-------
Determine whether the HE vehicle appears systematically:

    - smaller / larger,
    - farther / closer,
    - higher / lower on the image,
    - horizontally shifted,

than the physical CARLA vehicle when both represent the same
3-D actor pose.

IMPORTANT
---------
This script does NOT run NEAT.
This script does NOT use the old HE placement lookup table.
This script calls the exact current:

    render_he_actor()

used by the NEAT HE experiment.

Default camera:
    NEAT front camera
    x = 1.3 m
    y = 0.0 m
    z = 2.3 m
    yaw = 0 deg
    400 x 300
    FOV = 100 deg

Default tested camera depths:
    5, 6, 8, 10, 15, 20, 25, 30, 40, 50 m

Outputs:
    measurements.csv

    images/
        depth_005p0_carla.png
        depth_005p0_he.png
        depth_005p0_comparison.png
        depth_005p0_overlay.png
        ...

Run from:

    D:\\HallucinationEngine

Example:

python driving_models\\common\\compare_carla_he_vehicle_geometry_v1.py ^
  --town Town10HD_Opt ^
  --spawn-index 10 ^
  --carla-pythonapi E:\\Carla\\Carla_0.9.15\\PythonAPI\\carla

"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Repository paths
# ============================================================

THIS_FILE = Path(__file__).resolve()

HE_ROOT = (
    THIS_FILE
    .parents[2]
)

COMMON_DIR = (
    HE_ROOT
    / "driving_models"
    / "common"
)

HE_RUNTIME_DIR = (
    HE_ROOT
    / "HE_v_0.1"
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
    / "common"
    / "outputs"
    / "carla_he_vehicle_geometry_multiview_v1"
)

MULTIVIEW_POSES = [
    {
        "name": "rear_center",
        "lateral_m": 0.0,
        "relative_yaw_deg": 0.0,
    },
    {
        "name": "rear_left",
        "lateral_m": -2.0,
        "relative_yaw_deg": 0.0,
    },
    {
        "name": "rear_right",
        "lateral_m": 2.0,
        "relative_yaw_deg": 0.0,
    },
    {
        "name": "oblique_left",
        "lateral_m": -2.0,
        "relative_yaw_deg": 45.0,
    },
    {
        "name": "oblique_right",
        "lateral_m": 2.0,
        "relative_yaw_deg": -45.0,
    },
    {
        "name": "side_left",
        "lateral_m": -3.0,
        "relative_yaw_deg": 90.0,
    },
    {
        "name": "side_right",
        "lateral_m": 3.0,
        "relative_yaw_deg": -90.0,
    },
]

# CARLA 0.9.15 semantic segmentation:
# Car = semantic tag 14.
#
# Raw CARLA image format is BGRA.
# Semantic tag is encoded in the RED channel,
# therefore ndarray channel index = 2.
VEHICLE_SEMANTIC_TAG = 14


# ============================================================
# Arguments
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Compare physical CARLA Tesla apparent geometry "
            "against current HE Tesla rendering."
        )
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
        "--timeout",
        type=float,
        default=20.0,
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
        "--carla-pythonapi",
        default="",
    )

    parser.add_argument(
        "--sprite-root",
        default=str(
            DEFAULT_SPRITE_ROOT
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=100.0,
    )

    # NEAT native front camera.
    parser.add_argument(
        "--camera-x",
        type=float,
        default=1.3,
    )

    parser.add_argument(
        "--camera-y",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--camera-z",
        type=float,
        default=2.3,
    )

    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--camera-yaw",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--camera-roll",
        type=float,
        default=0.0,
    )

    # Use the CURRENT scenario dimensions first.
    #
    # We deliberately do NOT replace these with the CARLA
    # Tesla bounding-box values yet. We want to measure the
    # current HE implementation exactly as it is.
    parser.add_argument(
        "--he-length-m",
        type=float,
        default=4.2,
    )

    parser.add_argument(
        "--he-width-m",
        type=float,
        default=1.8,
    )

    parser.add_argument(
        "--he-height-m",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=[
            5.0,
            6.0,
            8.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
            50.0,
        ],
        help=(
            "Requested longitudinal camera depths in metres."
        ),
    )

    parser.add_argument(
        "--settle-ticks",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--canonical-settle-ticks",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--pose-distances",
        type=float,
        nargs="+",
        default=[
            5.0,
            8.0,
            10.0,
            15.0,
            20.0,
        ],
        help=(
            "Camera-depth values used for the "
            "representative multi-view calibration."
        ),
    )

    return parser


# ============================================================
# CARLA import
# ============================================================

def import_carla_module(
    carla_pythonapi,
):
    """
    Import the CARLA module already installed in the active
    Python environment.

    IMPORTANT:
    Do NOT manually add CARLA's old Windows py3.7 .egg to
    sys.path.  The he_neat environment uses Python 3.10 and
    already has a compatible CARLA 0.9.15 installation.

    --carla-pythonapi is retained for command-line compatibility
    with the other experiment runners, but this calibration
    script does not need the legacy dist egg.
    """

    try:

        import carla

    except ImportError as exc:

        raise RuntimeError(
            "Could not import CARLA from the active Python "
            "environment. Verify with:\n"
            "python -c \"import carla; print(carla.__file__)\""
        ) from exc

    print(
        "[CARLA Python module]",
        getattr(
            carla,
            "__file__",
            "<unknown>",
        ),
    )

    return carla

# ============================================================
# Sensor utilities
# ============================================================

def get_sensor_frame(
    sensor_queue,
    frame_id,
    sensor_name,
    timeout=5.0,
):

    while True:

        try:

            data = sensor_queue.get(
                timeout=timeout
            )

        except queue.Empty as exc:

            raise RuntimeError(
                f"Timeout waiting for "
                f"{sensor_name} "
                f"frame {frame_id}"
            ) from exc

        if data.frame < frame_id:
            continue

        if data.frame > frame_id:

            raise RuntimeError(
                f"{sensor_name} skipped "
                f"requested frame {frame_id}; "
                f"received {data.frame}"
            )

        return data


def carla_image_to_rgb(
    image,
):

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

    # CARLA raw image = BGRA.
    bgr = array[
        :,
        :,
        :3,
    ]

    rgb = bgr[
        :,
        :,
        ::-1,
    ].copy()

    return rgb


def semantic_vehicle_mask(
    image,
):

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

    # BGRA raw layout:
    # index 2 = RED channel.
    semantic_tag = array[
        :,
        :,
        2,
    ]

    mask = (
        semantic_tag
        ==
        VEHICLE_SEMANTIC_TAG
    )

    return mask


# ============================================================
# Mask measurements
# ============================================================

def measurement_from_mask(
    mask,
):

    ys, xs = np.where(
        mask
    )

    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):

        return None

    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    )

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    )

    width_px = (
        x2
        -
        x1
        +
        1
    )

    height_px = (
        y2
        -
        y1
        +
        1
    )

    return {
        "x1":
            x1,

        "y1":
            y1,

        "x2":
            x2,

        "y2":
            y2,

        "width_px":
            float(
                width_px
            ),

        "height_px":
            float(
                height_px
            ),

        "center_x":
            float(
                (
                    x1
                    +
                    x2
                )
                / 2.0
            ),

        "center_y":
            float(
                (
                    y1
                    +
                    y2
                )
                / 2.0
            ),

        "bottom_y":
            float(
                y2
            ),

        "area_px":
            int(
                len(
                    xs
                )
            ),
    }


def select_target_vehicle_component(
    raw_vehicle_mask,
    expected_box,
):

    binary = (
        raw_vehicle_mask
        .astype(
            np.uint8
        )
    )

    (
        component_count,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if component_count <= 1:

        return (
            np.zeros_like(
                raw_vehicle_mask,
                dtype=bool,
            ),
            None,
        )

    expected_cx = float(
        expected_box[
            "cx"
        ]
    )

    expected_bottom = float(
        expected_box[
            "bottom_y"
        ]
    )

    expected_h = max(
        float(
            expected_box[
                "box_height"
            ]
        ),
        1.0,
    )

    expected_w = max(
        float(
            expected_box[
                "box_width"
            ]
        ),
        1.0,
    )

    best_id = None
    best_score = float(
        "inf"
    )

    for component_id in range(
        1,
        component_count,
    ):

        x = int(
            stats[
                component_id,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                component_id,
                cv2.CC_STAT_TOP,
            ]
        )

        w = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        h = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < 3:
            continue

        component_cx = (
            x
            +
            (
                w - 1
            )
            / 2.0
        )

        component_bottom = (
            y
            +
            h
            -
            1
        )

        dx = (
            abs(
                component_cx
                -
                expected_cx
            )
            /
            max(
                expected_w,
                5.0,
            )
        )

        db = (
            abs(
                component_bottom
                -
                expected_bottom
            )
            /
            max(
                expected_h,
                5.0,
            )
        )

        dh = (
            abs(
                float(
                    h
                )
                -
                expected_h
            )
            /
            max(
                expected_h,
                5.0,
            )
        )

        score = (
            dx
            +
            db
            +
            0.20
            *
            dh
        )

        if score < best_score:

            best_score = (
                score
            )

            best_id = (
                component_id
            )

    if best_id is None:

        return (
            np.zeros_like(
                raw_vehicle_mask,
                dtype=bool,
            ),
            None,
        )

    selected_mask = (
        labels
        ==
        best_id
    )

    measurement = (
        measurement_from_mask(
            selected_mask
        )
    )

    return (
        selected_mask,
        measurement,
    )


# ============================================================
# HE alpha-mask reconstruction
# ============================================================

def build_he_alpha_mask(
    meta,
    sprite_bank,
    sprite_cache,
    trim_sprite_to_visible_alpha,
    width,
    height,
):

    if not meta.get(
        "rendered",
        False,
    ):

        return np.zeros(
            (
                height,
                width,
            ),
            dtype=bool,
        )

    selected_angle = int(
        meta[
            "selected_angle"
        ]
    )

    sprite_path = (
        Path(
            sprite_bank[
                "root"
            ]
        )
        /
        sprite_bank[
            "rgba_dir"
        ]
        /
        sprite_bank[
            "angle_format"
        ].format(
            angle=
                selected_angle
        )
    )

    sprite_rgba = (
        sprite_cache.load_rgba(
            str(
                sprite_path
            )
        )
    )

    sprite_rgba = (
        trim_sprite_to_visible_alpha(
            sprite_rgba
        )
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

        return np.zeros(
            (
                height,
                width,
            ),
            dtype=bool,
        )

    box = meta[
        "box"
    ]

    scale_x = float(
        meta.get(
            "subpixel_scale_x",
            meta[
                "subpixel_scale"
            ],
        )
    )

    scale_y = float(
        meta.get(
            "subpixel_scale_y",
            meta[
                "subpixel_scale"
            ],
        )
    )

    src_anchor_x = (
        float(
            src_w - 1
        )
        /
        2.0
    )

    src_anchor_y = float(
        src_h - 1
    )

    tx = (
        float(
            box[
                "cx"
            ]
        )
        -
        scale_x
        *
        src_anchor_x
    )

    ty = (
        float(
            box[
                "bottom_y"
            ]
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

    source_alpha = (
        sprite_rgba[
            :,
            :,
            3,
        ]
        .astype(
            np.float32
        )
        /
        255.0
    )

    warped_alpha = (
        cv2.warpAffine(
            source_alpha,
            affine,
            (
                width,
                height,
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=(
                cv2.BORDER_CONSTANT
            ),
            borderValue=0.0,
        )
    )

    # Same basic alpha significance threshold used by
    # trim_sprite_to_visible_alpha().
    threshold = (
        2.0
        /
        255.0
    )

    return (
        warped_alpha
        >
        threshold
    )


# ============================================================
# Visualization
# ============================================================

def annotated_bgr(
    rgb,
    measurement,
    label,
):

    image = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    if measurement is not None:

        x1 = int(
            measurement[
                "x1"
            ]
        )

        y1 = int(
            measurement[
                "y1"
            ]
        )

        x2 = int(
            measurement[
                "x2"
            ]
        )

        y2 = int(
            measurement[
                "y2"
            ]
        )

        cv2.rectangle(
            image,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                0,
            ),
            1,
        )

        text = (
            f"{label} "
            f"{measurement['width_px']:.0f}x"
            f"{measurement['height_px']:.0f}"
        )

    else:

        text = (
            f"{label}: no mask"
        )

    cv2.putText(
        image,
        text,
        (
            8,
            18,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    return image


def safe_ratio(
    numerator,
    denominator,
):

    if (
        numerator is None
        or
        denominator is None
        or
        abs(
            denominator
        )
        < 1e-9
    ):

        return float(
            "nan"
        )

    return (
        float(
            numerator
        )
        /
        float(
            denominator
        )
    )


def fmt_float(
    value,
    digits=3,
):

    if value is None:
        return "nan"

    try:

        if math.isnan(
            float(
                value
            )
        ):

            return "nan"

    except Exception:
        pass

    return (
        f"{float(value):.{digits}f}"
    )


# ============================================================
# Actor transform
# ============================================================

def actor_transform_at_camera_depth(
    carla,
    camera_tf,
    ground_z,
    depth_m,
    lateral_m=0.0,
    relative_yaw_deg=0.0,
):
    """
    Construct actor pose using camera-relative metric coordinates.

    depth_m:
        forward distance from camera to actor centre

    lateral_m:
        positive = camera right
        negative = camera left

    relative_yaw_deg:
        actor yaw relative to camera yaw
    """

    forward = (
        camera_tf
        .get_forward_vector()
    )

    right = (
        camera_tf
        .get_right_vector()
    )

    location = carla.Location(
        x=(
            float(
                camera_tf.location.x
            )
            +
            float(
                depth_m
            )
            *
            float(
                forward.x
            )
            +
            float(
                lateral_m
            )
            *
            float(
                right.x
            )
        ),

        y=(
            float(
                camera_tf.location.y
            )
            +
            float(
                depth_m
            )
            *
            float(
                forward.y
            )
            +
            float(
                lateral_m
            )
            *
            float(
                right.y
            )
        ),

        z=float(
            ground_z
        ),
    )

    rotation = carla.Rotation(
        pitch=0.0,

        yaw=(
            float(
                camera_tf.rotation.yaw
            )
            +
            float(
                relative_yaw_deg
            )
        ),

        roll=0.0,
    )

    return carla.Transform(
        location,
        rotation,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = (
        build_parser()
        .parse_args()
    )

    carla = (
        import_carla_module(
            args.carla_pythonapi
        )
    )

    # HE renderer runtime paths.
    for path in [
        COMMON_DIR,
        HE_RUNTIME_DIR,
    ]:

        if str(
            path
        ) not in sys.path:

            sys.path.insert(
                0,
                str(
                    path
                ),
            )

    from he_camera_renderer import (
        SpriteCache,
        discover_available_sprite_angles,
        render_he_actor,
        trim_sprite_to_visible_alpha,
    )

    output_root = Path(
        args.output_root
    ).resolve()

    images_dir = (
        output_root
        /
        "images"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_root
        /
        "measurements.csv"
    )

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

    available_angles = (
        discover_available_sprite_angles(
            sprite_bank
        )
    )

    if not available_angles:

        raise RuntimeError(
            "No HE sprite angles found in: "
            f"{sprite_bank['root']}"
        )

    sprite_cache = (
        SpriteCache()
    )

    he_dimensions = {
        "length_m":
            float(
                args.he_length_m
            ),

        "width_m":
            float(
                args.he_width_m
            ),

        "height_m":
            float(
                args.he_height_m
            ),
    }

    print()
    print("=" * 92)
    print(
        "CARLA <-> HE STATIC VEHICLE GEOMETRY CALIBRATION"
    )
    print("=" * 92)

    print(
        "[town]",
        args.town,
    )

    print(
        "[camera]",
        f"{args.width}x{args.height}",
        f"FOV={args.fov}",
    )

    print(
        "[camera mount]",
        f"x={args.camera_x}",
        f"y={args.camera_y}",
        f"z={args.camera_z}",
        f"yaw={args.camera_yaw}",
    )

    print(
        "[HE dimensions]",
        he_dimensions,
    )

    print(
        "[sprite root]",
        sprite_bank[
            "root"
        ],
    )

    print(
        "[sprite angles]",
        len(
            available_angles
        ),
    )

    print(
        "[distances]",
        args.distances,
    )

    print("=" * 92)
    print()

    client = carla.Client(
        args.host,
        args.port,
    )

    client.set_timeout(
        float(
            args.timeout
        )
    )

    world = None

    ego = None
    adversary = None

    rgb_camera = None
    semantic_camera = None

    actors = []

    original_settings = None

    q_rgb = queue.Queue()
    q_sem = queue.Queue()

    results = []

    try:

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

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            0.05
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

        if not spawn_points:

            raise RuntimeError(
                "Town contains no spawn points."
            )

        spawn_idx = int(
            args.spawn_index
        )

        if (
            spawn_idx < 0
            or
            spawn_idx
            >= len(
                spawn_points
            )
        ):

            raise RuntimeError(
                f"Invalid spawn index "
                f"{spawn_idx}; "
                f"town has "
                f"{len(spawn_points)} "
                f"spawn points."
            )

        bp_lib = (
            world
            .get_blueprint_library()
        )

        # ====================================================
        # Ego
        # ====================================================

        ego_bp = bp_lib.find(
            "vehicle.tesla.model3"
        )

        if ego_bp.has_attribute(
            "role_name"
        ):

            ego_bp.set_attribute(
                "role_name",
                "hero",
            )

        nominal_spawn_tf = (
            spawn_points[
                spawn_idx
            ]
        )

        ego = world.try_spawn_actor(
            ego_bp,
            nominal_spawn_tf,
        )

        if ego is None:

            raise RuntimeError(
                "Could not spawn ego."
            )

        actors.append(
            ego
        )

        hold = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=False,
        )

        zero_vector = (
            carla.Vector3D(
                0.0,
                0.0,
                0.0,
            )
        )

        # ====================================================
        # Physical settling
        # ====================================================

        print(
            "[init] physical ego settling..."
        )

        for _ in range(
            int(
                args.settle_ticks
            )
        ):

            ego.apply_control(
                hold
            )

            world.tick()

        settled_tf = (
            ego.get_transform()
        )

        print(
            "[init settled] "
            f"x={settled_tf.location.x:.6f} "
            f"y={settled_tf.location.y:.6f} "
            f"z={settled_tf.location.z:.6f} "
            f"yaw={settled_tf.rotation.yaw:.4f}"
        )

        print(
            "[init nominal] "
            f"x={nominal_spawn_tf.location.x:.6f} "
            f"y={nominal_spawn_tf.location.y:.6f} "
            f"z={nominal_spawn_tf.location.z:.6f} "
            f"yaw={nominal_spawn_tf.rotation.yaw:.4f}"
        )

        canonical_tf = carla.Transform(

            carla.Location(
                x=(
                    nominal_spawn_tf
                    .location
                    .x
                ),

                y=(
                    nominal_spawn_tf
                    .location
                    .y
                ),

                z=(
                    settled_tf
                    .location
                    .z
                ),
            ),

            carla.Rotation(
                pitch=(
                    settled_tf
                    .rotation
                    .pitch
                ),

                yaw=(
                    nominal_spawn_tf
                    .rotation
                    .yaw
                ),

                roll=(
                    settled_tf
                    .rotation
                    .roll
                ),
            ),
        )

        ego.set_transform(
            canonical_tf
        )

        ego.set_target_velocity(
            zero_vector
        )

        ego.set_target_angular_velocity(
            zero_vector
        )

        for _ in range(
            int(
                args.canonical_settle_ticks
            )
        ):

            ego.apply_control(
                hold
            )

            world.tick()

        ego.set_target_velocity(
            zero_vector
        )

        ego.set_target_angular_velocity(
            zero_vector
        )

        ego.apply_control(
            hold
        )

        ego0_tf = (
            ego.get_transform()
        )

        print(
            "[experiment ego] "
            f"x={ego0_tf.location.x:.6f} "
            f"y={ego0_tf.location.y:.6f} "
            f"z={ego0_tf.location.z:.6f} "
            f"yaw={ego0_tf.rotation.yaw:.4f}"
        )

        # ====================================================
        # NEAT native FRONT camera
        # ====================================================

        camera_relative_tf = (
            carla.Transform(

                carla.Location(
                    x=float(
                        args.camera_x
                    ),

                    y=float(
                        args.camera_y
                    ),

                    z=float(
                        args.camera_z
                    ),
                ),

                carla.Rotation(
                    pitch=float(
                        args.camera_pitch
                    ),

                    yaw=float(
                        args.camera_yaw
                    ),

                    roll=float(
                        args.camera_roll
                    ),
                ),
            )
        )

        rgb_bp = bp_lib.find(
            "sensor.camera.rgb"
        )

        rgb_bp.set_attribute(
            "image_size_x",
            str(
                args.width
            ),
        )

        rgb_bp.set_attribute(
            "image_size_y",
            str(
                args.height
            ),
        )

        rgb_bp.set_attribute(
            "fov",
            str(
                args.fov
            ),
        )

        semantic_bp = bp_lib.find(
            "sensor.camera.semantic_segmentation"
        )

        semantic_bp.set_attribute(
            "image_size_x",
            str(
                args.width
            ),
        )

        semantic_bp.set_attribute(
            "image_size_y",
            str(
                args.height
            ),
        )

        semantic_bp.set_attribute(
            "fov",
            str(
                args.fov
            ),
        )

        rgb_camera = world.spawn_actor(
            rgb_bp,
            camera_relative_tf,
            attach_to=ego,
            attachment_type=(
                carla.AttachmentType.Rigid
            ),
        )

        semantic_camera = (
            world.spawn_actor(
                semantic_bp,
                camera_relative_tf,
                attach_to=ego,
                attachment_type=(
                    carla.AttachmentType.Rigid
                ),
            )
        )

        actors.extend(
            [
                rgb_camera,
                semantic_camera,
            ]
        )

        rgb_camera.listen(
            q_rgb.put
        )

        semantic_camera.listen(
            q_sem.put
        )

        # Camera warm-up.
        for _ in range(3):

            frame = world.tick()

            get_sensor_frame(
                q_rgb,
                frame,
                "RGB",
            )

            get_sensor_frame(
                q_sem,
                frame,
                "semantic",
            )

        camera_tf = (
            rgb_camera
            .get_transform()
        )

        print(
            "[camera world] "
            f"x={camera_tf.location.x:.6f} "
            f"y={camera_tf.location.y:.6f} "
            f"z={camera_tf.location.z:.6f} "
            f"yaw={camera_tf.rotation.yaw:.4f}"
        )

        # ====================================================
        # Physical reference Tesla
        # ====================================================

        adv_bp = bp_lib.find(
            "vehicle.tesla.model3"
        )

        if adv_bp.has_attribute(
            "role_name"
        ):

            adv_bp.set_attribute(
                "role_name",
                "geometry_reference",
            )

        if adv_bp.has_attribute(
            "color"
        ):

            # Match current HE experiment convention.
            adv_bp.set_attribute(
                "color",
                "0,0,255",
            )

        # Spawn safely somewhere away from the ego,
        # then disable physics and reposition precisely.
        adv_spawn_idx = min(
            45,
            len(
                spawn_points
            )
            -
            1,
        )

        adversary = (
            world.try_spawn_actor(
                adv_bp,
                spawn_points[
                    adv_spawn_idx
                ],
            )
        )

        if adversary is None:

            # Fallback to last spawn.
            adversary = (
                world.try_spawn_actor(
                    adv_bp,
                    spawn_points[
                        -1
                    ],
                )
            )

        if adversary is None:

            raise RuntimeError(
                "Could not spawn physical "
                "Tesla reference actor."
            )

        actors.append(
            adversary
        )

        adversary.set_simulate_physics(
            False
        )

        bbox = (
            adversary.bounding_box
        )

        carla_length = (
            2.0
            *
            float(
                bbox.extent.x
            )
        )

        carla_width = (
            2.0
            *
            float(
                bbox.extent.y
            )
        )

        carla_height = (
            2.0
            *
            float(
                bbox.extent.z
            )
        )

        print()
        print(
            "[physical Tesla bbox] "
            f"L={carla_length:.3f} "
            f"W={carla_width:.3f} "
            f"H={carla_height:.3f}"
        )

        print(
            "[HE dimensions] "
            f"L={args.he_length_m:.3f} "
            f"W={args.he_width_m:.3f} "
            f"H={args.he_height_m:.3f}"
        )

        print()

        # ====================================================
        # Calibration distances
        # ====================================================

        test_cases = []

        for pose in MULTIVIEW_POSES:

            for requested_depth in (
                args.pose_distances
            ):

                test_cases.append({
                    "pose_name":
                        str(
                            pose[
                                "name"
                            ]
                        ),

                    "depth_m":
                        float(
                            requested_depth
                        ),

                    "lateral_m":
                        float(
                            pose[
                                "lateral_m"
                            ]
                        ),

                    "relative_yaw_deg":
                        float(
                            pose[
                                "relative_yaw_deg"
                            ]
                        ),
                })

        print()
        print(
            "[multi-view cases]",
            len(
                test_cases
            ),
        )
        print()

        for case in test_cases:

            pose_name = (
                case[
                    "pose_name"
                ]
            )

            requested_depth = float(
                case[
                    "depth_m"
                ]
            )

            lateral_m = float(
                case[
                    "lateral_m"
                ]
            )

            relative_yaw_deg = float(
                case[
                    "relative_yaw_deg"
                ]
            )

            camera_tf = (
                rgb_camera
                .get_transform()
            )

            target_tf = (
                actor_transform_at_camera_depth(
                    carla=carla,

                    camera_tf=
                        camera_tf,

                    ground_z=(
                        ego0_tf
                        .location
                        .z
                    ),

                    depth_m=
                        requested_depth,

                    lateral_m=
                        lateral_m,

                    relative_yaw_deg=
                        relative_yaw_deg,
                )
            )

            # --------------------------------------------
            # Physical CARLA frame
            # --------------------------------------------

            adversary.set_transform(
                target_tf
            )

            frame = world.tick()

            physical_rgb_image = (
                get_sensor_frame(
                    q_rgb,
                    frame,
                    "RGB physical",
                )
            )

            physical_sem_image = (
                get_sensor_frame(
                    q_sem,
                    frame,
                    "semantic physical",
                )
            )

            physical_rgb = (
                carla_image_to_rgb(
                    physical_rgb_image
                )
            )

            physical_vehicle_mask = (
                semantic_vehicle_mask(
                    physical_sem_image
                )
            )

            # --------------------------------------------
            # Capture clean background with target hidden
            # --------------------------------------------

            camera_tf_now = (
                rgb_camera
                .get_transform()
            )

            forward = (
                camera_tf_now
                .get_forward_vector()
            )

            hide_tf = carla.Transform(

                carla.Location(
                    x=(
                        camera_tf_now
                        .location
                        .x
                        -
                        100.0
                        *
                        forward.x
                    ),

                    y=(
                        camera_tf_now
                        .location
                        .y
                        -
                        100.0
                        *
                        forward.y
                    ),

                    z=(
                        ego0_tf
                        .location
                        .z
                    ),
                ),

                carla.Rotation(
                    yaw=(
                        ego0_tf
                        .rotation
                        .yaw
                    ),
                ),
            )

            adversary.set_transform(
                hide_tf
            )

            frame = world.tick()

            base_rgb_image = (
                get_sensor_frame(
                    q_rgb,
                    frame,
                    "RGB background",
                )
            )

            # Drain matching semantic frame as well.
            background_sem_image = (
                get_sensor_frame(
                    q_sem,
                    frame,
                    "semantic background",
                )
            )

            background_vehicle_mask = (
                semantic_vehicle_mask(
                    background_sem_image
                )
            )

            base_rgb = (
                carla_image_to_rgb(
                    base_rgb_image
                )
            )

            # --------------------------------------------
            # HE rendering at IDENTICAL target transform
            # --------------------------------------------

            camera_tf_for_he = (
                rgb_camera
                .get_transform()
            )

            he_rgb, he_meta = (
                render_he_actor(
                    base_rgb=base_rgb,
                    actor_tf=target_tf,
                    camera_tf=(
                        camera_tf_for_he
                    ),
                    dimensions=(
                        he_dimensions
                    ),
                    sprite_bank=(
                        sprite_bank
                    ),
                    available_angles=(
                        available_angles
                    ),
                    sprite_cache=(
                        sprite_cache
                    ),
                    width=int(
                        args.width
                    ),
                    height=int(
                        args.height
                    ),
                    fov=float(
                        args.fov
                    ),
                )
            )

            if not he_meta.get(
                "rendered",
                False,
            ):

                print(
                    f"[{requested_depth:5.1f} m] "
                    "HE not rendered:",
                    he_meta,
                )

                continue

            theoretical_box = (
                he_meta[
                    "box"
                ]
            )

            # ------------------------------------------------
            # Isolate ONLY the inserted physical Tesla.
            #
            # The semantic image contains every vehicle,
            # including potentially visible parts of the ego.
            #
            # We capture the exact same scene again after
            # moving the Tesla behind the camera, then subtract
            # the persistent vehicle pixels.
            # ------------------------------------------------

            target_difference_mask = (
                physical_vehicle_mask
                &
                ~background_vehicle_mask
            )

            # Small connected-component cleanup.
            (
                physical_mask,
                physical_measurement,
            ) = (
                select_target_vehicle_component(
                    target_difference_mask,
                    theoretical_box,
                )
            )
            print(
                f"[mask {pose_name:14s} "
                f"{requested_depth:5.1f} m] "
                f"physical="
                f"{int(physical_vehicle_mask.sum())} "
                f"background="
                f"{int(background_vehicle_mask.sum())} "
                f"target="
                f"{int(target_difference_mask.sum())}"
            )

            he_mask = (
                build_he_alpha_mask(
                    meta=he_meta,
                    sprite_bank=(
                        sprite_bank
                    ),
                    sprite_cache=(
                        sprite_cache
                    ),
                    trim_sprite_to_visible_alpha=(
                        trim_sprite_to_visible_alpha
                    ),
                    width=int(
                        args.width
                    ),
                    height=int(
                        args.height
                    ),
                )
            )

            he_measurement = (
                measurement_from_mask(
                    he_mask
                )
            )

            if physical_measurement is None:

                print(
                    f"[{requested_depth:5.1f} m] "
                    "Could not isolate physical Tesla."
                )

                continue

            if he_measurement is None:

                print(
                    f"[{requested_depth:5.1f} m] "
                    "Could not measure HE alpha mask."
                )

                continue

            actual_depth = float(
                theoretical_box[
                    "depth_m"
                ]
            )

            carla_w = (
                physical_measurement[
                    "width_px"
                ]
            )

            carla_h = (
                physical_measurement[
                    "height_px"
                ]
            )

            he_w = (
                he_measurement[
                    "width_px"
                ]
            )

            he_h = (
                he_measurement[
                    "height_px"
                ]
            )

            width_ratio = (
                safe_ratio(
                    he_w,
                    carla_w,
                )
            )

            height_ratio = (
                safe_ratio(
                    he_h,
                    carla_h,
                )
            )

            area_ratio = (
                safe_ratio(
                    he_measurement[
                        "area_px"
                    ],
                    physical_measurement[
                        "area_px"
                    ],
                )
            )

            center_x_error = (
                he_measurement[
                    "center_x"
                ]
                -
                physical_measurement[
                    "center_x"
                ]
            )

            bottom_y_error = (
                he_measurement[
                    "bottom_y"
                ]
                -
                physical_measurement[
                    "bottom_y"
                ]
            )

            # If HE width is smaller, this quantity becomes
            # larger than true depth: HE visually resembles
            # a farther-away vehicle.
            apparent_depth_width = (
                actual_depth
                *
                safe_ratio(
                    carla_w,
                    he_w,
                )
            )

            apparent_depth_error = (
                apparent_depth_width
                -
                actual_depth
            )

            row = {

                "pose_name":
                    pose_name,

                "lateral_offset_m":
                    lateral_m,

                "relative_yaw_deg":
                    relative_yaw_deg,
                "requested_depth_m":
                    requested_depth,

                "actual_camera_depth_m":
                    actual_depth,

                "selected_sprite_angle_deg":
                    int(
                        he_meta[
                            "selected_angle"
                        ]
                    ),

                "viewpoint_angle_deg":
                    float(
                        he_meta[
                            "viewpoint_angle_deg"
                        ]
                    ),

                "theoretical_he_box_height_px":
                    float(
                        theoretical_box[
                            "box_height"
                        ]
                    ),

                "theoretical_he_box_width_px":
                    float(
                        theoretical_box[
                            "box_width"
                        ]
                    ),

                "carla_width_px":
                    carla_w,

                "carla_height_px":
                    carla_h,

                "carla_center_x_px":
                    physical_measurement[
                        "center_x"
                    ],

                "carla_bottom_y_px":
                    physical_measurement[
                        "bottom_y"
                    ],

                "carla_area_px":
                    physical_measurement[
                        "area_px"
                    ],

                "he_width_px":
                    he_w,

                "he_height_px":
                    he_h,

                "he_center_x_px":
                    he_measurement[
                        "center_x"
                    ],

                "he_bottom_y_px":
                    he_measurement[
                        "bottom_y"
                    ],

                "he_area_px":
                    he_measurement[
                        "area_px"
                    ],

                "width_ratio_he_over_carla":
                    width_ratio,

                "height_ratio_he_over_carla":
                    height_ratio,

                "area_ratio_he_over_carla":
                    area_ratio,

                "center_x_error_px":
                    center_x_error,

                "bottom_y_error_px":
                    bottom_y_error,

                "apparent_depth_from_width_m":
                    apparent_depth_width,

                "apparent_depth_error_m":
                    apparent_depth_error,
            }

            results.append(
                row
            )

            print(
                f"[{pose_name:14s}] "
                f"[{requested_depth:5.1f} m] "
                f"[{requested_depth:5.1f} m] "
                f"depth={actual_depth:6.2f} | "
                f"CARLA="
                f"{carla_w:5.1f}x"
                f"{carla_h:5.1f} | "
                f"HE="
                f"{he_w:5.1f}x"
                f"{he_h:5.1f} | "
                f"Wratio={width_ratio:6.3f} "
                f"Hratio={height_ratio:6.3f} | "
                f"bottomΔ="
                f"{bottom_y_error:+6.2f}px | "
                f"apparentΔ="
                f"{apparent_depth_error:+6.2f}m"
            )

            # --------------------------------------------
            # Save diagnostics
            # --------------------------------------------

            depth_name = (
                f"{requested_depth:05.1f}"
                .replace(
                    ".",
                    "p",
                )
            )
            case_name = (
                f"{pose_name}_depth_{depth_name}"
            )

            carla_annotated = (
                annotated_bgr(
                    physical_rgb,
                    physical_measurement,
                    "CARLA",
                )
            )

            he_annotated = (
                annotated_bgr(
                    he_rgb,
                    he_measurement,
                    "HE",
                )
            )

            comparison = np.hstack(
                [
                    carla_annotated,
                    he_annotated,
                ]
            )

            overlay = (
                cv2.addWeighted(
                    carla_annotated,
                    0.5,
                    he_annotated,
                    0.5,
                    0.0,
                )
            )

            cv2.imwrite(
                str(
                    images_dir
                    /
                    f"{case_name}_carla.png"
                ),
                carla_annotated,
            )

            cv2.imwrite(
                str(
                    images_dir
                    /
                    f"depth_{case_name}_he.png"
                ),
                he_annotated,
            )

            cv2.imwrite(
                str(
                    images_dir
                    /
                    f"depth_{case_name}_comparison.png"
                ),
                comparison,
            )

            cv2.imwrite(
                str(
                    images_dir
                    /
                    f"depth_{case_name}_overlay.png"
                ),
                overlay,
            )

            cv2.imwrite(
                str(
                    images_dir
                    /
                    f"depth_{case_name}_carla_mask.png"
                ),
                (
                    physical_mask
                    .astype(
                        np.uint8
                    )
                    *
                    255
                ),
            )

            cv2.imwrite(
                str(
                    images_dir
                    /
                    f"depth_{case_name}_he_mask.png"
                ),
                (
                    he_mask
                    .astype(
                        np.uint8
                    )
                    *
                    255
                ),
            )

        # ====================================================
        # CSV
        # ====================================================

        if not results:

            raise RuntimeError(
                "Calibration produced no valid measurements."
            )

        fieldnames = list(
            results[
                0
            ].keys()
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                results
            )

        # ====================================================
        # Aggregate summary
        # ====================================================

        width_ratios = np.array(
            [
                r[
                    "width_ratio_he_over_carla"
                ]
                for r in results
            ],
            dtype=np.float64,
        )

        height_ratios = np.array(
            [
                r[
                    "height_ratio_he_over_carla"
                ]
                for r in results
            ],
            dtype=np.float64,
        )

        bottom_errors = np.array(
            [
                r[
                    "bottom_y_error_px"
                ]
                for r in results
            ],
            dtype=np.float64,
        )

        apparent_errors = np.array(
            [
                r[
                    "apparent_depth_error_m"
                ]
                for r in results
            ],
            dtype=np.float64,
        )

        print()
        print("=" * 92)
        print(
            "CALIBRATION SUMMARY"
        )
        print("=" * 92)

        print(
            "Measurements:",
            len(
                results
            ),
        )

        print(
            "Mean HE/CARLA width ratio :",
            f"{np.nanmean(width_ratios):.4f}",
        )

        print(
            "Mean HE/CARLA height ratio:",
            f"{np.nanmean(height_ratios):.4f}",
        )

        print(
            "Mean bottom-y error        :",
            f"{np.nanmean(bottom_errors):+.3f} px",
        )

        print(
            "Mean apparent-depth error  :",
            f"{np.nanmean(apparent_errors):+.3f} m",
        )

        print()
        print("-" * 92)
        print("SUMMARY BY VIEW")
        print("-" * 92)

        for pose in MULTIVIEW_POSES:

            pose_name = (
                pose[
                    "name"
                ]
            )

            pose_rows = [
                r
                for r in results
                if r[
                    "pose_name"
                ]
                ==
                pose_name
            ]

            if not pose_rows:
                continue

            wr = np.array(
                [
                    r[
                        "width_ratio_he_over_carla"
                    ]
                    for r in pose_rows
                ],
                dtype=np.float64,
            )

            hr = np.array(
                [
                    r[
                        "height_ratio_he_over_carla"
                    ]
                    for r in pose_rows
                ],
                dtype=np.float64,
            )

            by = np.array(
                [
                    r[
                        "bottom_y_error_px"
                    ]
                    for r in pose_rows
                ],
                dtype=np.float64,
            )

            cx = np.array(
                [
                    r[
                        "center_x_error_px"
                    ]
                    for r in pose_rows
                ],
                dtype=np.float64,
            )

            print(
                f"{pose_name:16s} "
                f"W={np.nanmean(wr):.3f}  "
                f"H={np.nanmean(hr):.3f}  "
                f"bottomΔ={np.nanmean(by):+.2f}px  "
                f"cxΔ={np.nanmean(cx):+.2f}px"
            )
        print(
            "CSV:",
            csv_path,
        )

        print(
            "Images:",
            images_dir,
        )

        print("=" * 92)

    finally:

        print(
            "[cleanup]"
        )

        for sensor in [
            rgb_camera,
            semantic_camera,
        ]:

            if sensor is not None:

                try:
                    sensor.stop()
                except Exception:
                    pass

        for actor in reversed(
            actors
        ):

            if actor is None:
                continue

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

                world.apply_settings(
                    original_settings
                )

            except Exception:
                pass


if __name__ == "__main__":

    main()