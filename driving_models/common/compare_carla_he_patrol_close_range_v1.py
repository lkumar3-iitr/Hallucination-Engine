#!/usr/bin/env python3
"""
compare_carla_he_patrol_close_range_v1.py

Deterministic close-range geometry calibration for the production HE renderer.

Purpose
-------
Isolate the close-range Hallucination Engine geometry from TCP/NEAT control.

For a single production asset (default: Nissan Patrol), place the *physical*
CARLA actor and the HE actor at the exact same camera-relative poses:

    depth = 20, 15, 12, 10, 8, 7, 6, 5, 4, 3.5, 3.0 m
    lateral = -3.0 m
    relative yaw = 0 deg

At each pose this script measures:

    Physical CARLA semantic mask:
        center_x
        bottom_y
        visible width
        visible height
        visible area

    Current HE projection:
        actor-center depth
        support depth
        nearest/farthest footprint depth
        projected box center
        projected bottom_y
        projected width/height

    Production 4320 selector:
        viewpoint angle
        selected angle
        query/selected distance
        query/selected elevation

It also saves physical, HE, and overlay images and measurements.csv.

IMPORTANT
---------
- No TCP or NEAT model is loaded.
- No driving-model control is used.
- No HE geometry equation is changed.
- This calls the current shared production render_he_actor_view_matrix().
- The physical CARLA blueprint and production physical dimensions are resolved
  from he_asset_registry_v1.py + the production asset bank metadata.

Default camera is TCP native:
    900 x 256
    FOV 100
    x=-1.5, y=0, z=2.0

Run from:
    D:\\HallucinationEngine

Example:
python driving_models\\common\\compare_carla_he_patrol_close_range_v1.py ^
  --asset-root D:\\HallucinationEngine-asset\\HE_v_0.1\\assets\\sprite_bank_native_production
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


THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]
COMMON_DIR = HE_ROOT / "driving_models" / "common"

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from carla_ego_initialization import canonicalize_ego_start
from he_asset_registry_v1 import HEAssetRegistry, DEFAULT_MANIFEST
from he_camera_renderer import (
    SpriteCache,
    load_view_matrix_sprite_bank,
    project_virtual_actor,
    render_he_actor_view_matrix,
)
from sprite_native_geometry_v1 import (
    load_sprite_native_geometry,
)


VEHICLE_SEMANTIC_TAG = 14


def build_parser():
    parser = argparse.ArgumentParser(
        description="Deterministic CARLA-vs-HE Patrol close-range geometry test."
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--spawn-index", type=int, default=10)

    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--asset-key", default="vehicle.passenger_02")

    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fov", type=float, default=100.0)

    # TCP native camera.
    parser.add_argument("--camera-x", type=float, default=-1.5)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=2.0)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)

    parser.add_argument("--lateral-m", type=float, default=-3.0)
    parser.add_argument("--relative-yaw-deg", type=float, default=0.0)

    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=[20.0, 15.0, 12.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.5, 3.0],
    )

    parser.add_argument(
        "--distance-selection-mode",
        choices=["linear", "log", "inverse_depth"],
        default="linear",
    )

    parser.add_argument(
        "--projection-mode",
        choices=[
            "oriented_2p5d_support",
            "center_depth_billboard",
        ],
        default="oriented_2p5d_support",
    )

    parser.add_argument(
        "--geometry-mode",
        choices=[
            "proxy",
            "sprite_native",
            "sprite_native_width",
            "close_width_blend",
        ],
        default="proxy",
    )

    parser.add_argument(
        "--sprite-geometry-csv",
        default=None,
        help=(
            "Self-contained sprite_geometry_v1.csv. Required when "
            "--geometry-mode sprite_native, sprite_native_width, "
            "or close_width_blend."
        ),
    )

    parser.add_argument(
        "--close-width-blend-near-m",
        type=float,
        default=5.10,
        help=(
            "Radial distance at/below which close_width_blend uses "
            "100% sprite-native width."
        ),
    )

    parser.add_argument(
        "--close-width-blend-far-m",
        type=float,
        default=5.90,
        help=(
            "Radial distance at/above which close_width_blend uses "
            "100% center-depth proxy width."
        ),
    )

    parser.add_argument("--he-bottom-y-offset-px", type=float, default=0.0)

    parser.add_argument("--physics-settle-ticks", type=int, default=30)
    parser.add_argument("--canonical-hold-ticks", type=int, default=5)

    parser.add_argument(
        "--output-root",
        default=str(
            HE_ROOT
            / "driving_models"
            / "common"
            / "outputs"
            / "carla_he_patrol_close_range_v1"
        ),
    )

    return parser


def get_sensor_frame(q, frame_id, name, timeout=5.0):
    while True:
        try:
            data = q.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError(
                f"Timeout waiting for {name} frame {frame_id}"
            ) from exc

        if data.frame < frame_id:
            continue

        if data.frame > frame_id:
            raise RuntimeError(
                f"{name} skipped requested frame {frame_id}; received {data.frame}"
            )

        return data


def carla_image_to_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1].copy()


def semantic_vehicle_mask(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    # CARLA semantic segmentation raw image is BGRA.
    # Semantic tag is in red => index 2.
    return array[:, :, 2] == VEHICLE_SEMANTIC_TAG


def instance_change_vehicle_mask(
    physical_instance_image,
    background_instance_image,
    physical_semantic_image,
):
    # Isolate the inserted physical reference vehicle by comparing
    # CARLA instance-segmentation identities between:
    #   A. Patrol at the test pose
    #   B. Patrol hidden behind the camera
    #
    # CARLA raw instance image is BGRA. The instance code is carried
    # in B/G. We only need identity change, not mapping to actor.id.

    physical = np.frombuffer(
        physical_instance_image.raw_data,
        dtype=np.uint8,
    ).reshape(
        (
            physical_instance_image.height,
            physical_instance_image.width,
            4,
        )
    )

    background = np.frombuffer(
        background_instance_image.raw_data,
        dtype=np.uint8,
    ).reshape(
        (
            background_instance_image.height,
            background_instance_image.width,
            4,
        )
    )

    instance_changed = (
        (physical[:, :, 0] != background[:, :, 0])
        |
        (physical[:, :, 1] != background[:, :, 1])
    )

    vehicle_pixels = semantic_vehicle_mask(
        physical_semantic_image
    )

    return (
        instance_changed
        &
        vehicle_pixels
    )


def measurement_from_mask(mask):
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width_px": float(x2 - x1 + 1),
        "height_px": float(y2 - y1 + 1),
        "center_x": float((x1 + x2) / 2.0),
        "bottom_y": float(y2),
        "area_px": int(len(xs)),
    }


def largest_component(mask):
    binary = mask.astype(np.uint8)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count <= 1:
        return np.zeros_like(mask, dtype=bool)

    best_id = None
    best_area = -1

    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_id = component_id

    if best_id is None:
        return np.zeros_like(mask, dtype=bool)

    return labels == best_id


def he_difference_mask(he_rgb, base_rgb, threshold=3):
    """
    Secondary visual measurement only.

    The authoritative HE geometry remains he_meta['box'].
    This mask helps show the actually visible inserted raster.
    """
    diff = np.max(
        np.abs(
            he_rgb.astype(np.int16)
            - base_rgb.astype(np.int16)
        ),
        axis=2,
    )

    mask = diff > int(threshold)

    # Tiny cleanup to suppress isolated compression/sensor differences.
    if np.any(mask):
        kernel = np.ones((2, 2), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        )
        mask = mask_u8 > 0

    return largest_component(mask)


def actor_transform_at_camera_depth(
    carla,
    camera_tf,
    ground_z,
    depth_m,
    lateral_m,
    relative_yaw_deg,
):
    forward = camera_tf.get_forward_vector()
    right = camera_tf.get_right_vector()

    location = carla.Location(
        x=(
            float(camera_tf.location.x)
            + float(depth_m) * float(forward.x)
            + float(lateral_m) * float(right.x)
        ),
        y=(
            float(camera_tf.location.y)
            + float(depth_m) * float(forward.y)
            + float(lateral_m) * float(right.y)
        ),
        z=float(ground_z),
    )

    rotation = carla.Rotation(
        pitch=0.0,
        yaw=(
            float(camera_tf.rotation.yaw)
            + float(relative_yaw_deg)
        ),
        roll=0.0,
    )

    return carla.Transform(location, rotation)


def annotate(rgb, measurement=None, box=None, title=""):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if measurement is not None:
        cv2.rectangle(
            bgr,
            (int(measurement["x1"]), int(measurement["y1"])),
            (int(measurement["x2"]), int(measurement["y2"])),
            (0, 255, 0),
            1,
        )

    if box is not None and all(
        k in box for k in ("x1", "y1", "x2", "y2")
    ):
        cv2.rectangle(
            bgr,
            (int(round(box["x1"])), int(round(box["y1"]))),
            (int(round(box["x2"])), int(round(box["y2"]))),
            (0, 0, 255),
            1,
        )

    if title:
        cv2.putText(
            bgr,
            title,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return bgr


def safe_float(d, key, default=float("nan")):
    try:
        return float(d[key])
    except Exception:
        return float(default)


def safe_ratio(a, b):
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return float("nan")

    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-9:
        return float("nan")

    return a / b


def fstr(value, digits=2):
    try:
        value = float(value)
        if not np.isfinite(value):
            return "nan"
        return f"{value:.{digits}f}"
    except Exception:
        return "nan"


def main():
    args = build_parser().parse_args()

    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "Could not import CARLA from the active Python environment."
        ) from exc

    output_root = Path(args.output_root).resolve()
    images_dir = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "measurements.csv"

    registry = HEAssetRegistry(
        asset_root=args.asset_root,
        manifest_path=args.manifest,
    )

    asset = registry.resolve(args.asset_key)

    print()
    print("=" * 92)
    print("CARLA <-> HE PATROL CLOSE-RANGE GEOMETRY V1")
    print("=" * 92)
    print("[asset key]       ", asset.key)
    print("[asset id]        ", asset.asset_id)
    print("[CARLA blueprint] ", asset.carla_blueprint)
    print("[view matrix]     ", asset.view_matrix_csv)
    print("[camera]          ", f"{args.width}x{args.height} FOV={args.fov}")
    print(
        "[camera mount]    ",
        f"x={args.camera_x} y={args.camera_y} z={args.camera_z}",
    )
    print(
        "[pose sweep]      ",
        f"lateral={args.lateral_m} m yaw={args.relative_yaw_deg} deg",
    )
    print("[depths]          ", args.depths)
    print("[projection mode]  ", args.projection_mode)
    print("[geometry mode]    ", args.geometry_mode)

    if asset.physical_bbox is not None:
        dimensions = {
            "length_m": float(asset.physical_bbox.length_m),
            "width_m": float(asset.physical_bbox.width_m),
            "height_m": float(asset.physical_bbox.height_m),
        }
        print("[bank physical bbox]", dimensions)
    else:
        dimensions = None
        print("[bank physical bbox] unavailable; will use live CARLA bbox")

    sprite_bank = {
        "mode": "view_matrix",
        "view_matrix_csvs": [str(asset.view_matrix_csv)],
        "target_height_m": 0.75,
        "vertical_mode": "state_y",
        "camera_height_m": float(args.camera_z),
        "distance_selection_mode": args.distance_selection_mode,
    }

    view_matrix = load_view_matrix_sprite_bank(sprite_bank)
    sprite_cache = SpriteCache()

    sprite_geometry = None

    if args.geometry_mode in {
        "sprite_native",
        "sprite_native_width",
        "close_width_blend",
    }:
        if args.sprite_geometry_csv is None:
            raise RuntimeError(
                "--sprite-geometry-csv is required when "
                "--geometry-mode sprite_native, "
                "sprite_native_width, or close_width_blend."
            )

        sprite_geometry = load_sprite_native_geometry(
            Path(
                args.sprite_geometry_csv
            )
        )

        print(
            "[sprite geometry]  ",
            Path(
                args.sprite_geometry_csv
            ).resolve(),
        )
        print(
            "[geometry version] ",
            sprite_geometry.geometry_version,
        )
        print(
            "[alpha threshold]  ",
            sprite_geometry.geometry_alpha_threshold,
        )

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(float(args.timeout))

    world = None
    original_settings = None

    ego = None
    adversary = None
    rgb_camera = None
    semantic_camera = None
    instance_camera = None

    actors = []
    q_rgb = queue.Queue()
    q_sem = queue.Queue()
    q_inst = queue.Queue()

    results = []

    try:
        print("[CARLA] loading:", args.town)
        world = client.load_world(args.town)

        original_settings = world.get_settings()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()

        if not spawn_points:
            raise RuntimeError("Town contains no spawn points.")

        spawn_idx = int(args.spawn_index) % len(spawn_points)
        bp_lib = world.get_blueprint_library()

        # ----------------------------------------------------
        # Fixed ego/camera platform
        # ----------------------------------------------------
        ego_bp = bp_lib.find("vehicle.tesla.model3")
        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "hero")

        ego = world.try_spawn_actor(
            ego_bp,
            spawn_points[spawn_idx],
        )

        if ego is None:
            raise RuntimeError("Could not spawn fixed camera ego.")

        actors.append(ego)

        ego0_tf, _ego0_state = canonicalize_ego_start(
            world=world,
            ego=ego,
            nominal_spawn_tf=spawn_points[spawn_idx],
            settle_ticks=int(args.physics_settle_ticks),
            hold_ticks=int(args.canonical_hold_ticks),
        )

        camera_relative_tf = carla.Transform(
            carla.Location(
                x=float(args.camera_x),
                y=float(args.camera_y),
                z=float(args.camera_z),
            ),
            carla.Rotation(
                pitch=float(args.camera_pitch),
                yaw=float(args.camera_yaw),
                roll=float(args.camera_roll),
            ),
        )

        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(args.width))
        rgb_bp.set_attribute("image_size_y", str(args.height))
        rgb_bp.set_attribute("fov", str(args.fov))

        sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", str(args.width))
        sem_bp.set_attribute("image_size_y", str(args.height))
        sem_bp.set_attribute("fov", str(args.fov))

        inst_bp = bp_lib.find("sensor.camera.instance_segmentation")
        inst_bp.set_attribute("image_size_x", str(args.width))
        inst_bp.set_attribute("image_size_y", str(args.height))
        inst_bp.set_attribute("fov", str(args.fov))

        rgb_camera = world.spawn_actor(
            rgb_bp,
            camera_relative_tf,
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )

        semantic_camera = world.spawn_actor(
            sem_bp,
            camera_relative_tf,
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )

        instance_camera = world.spawn_actor(
            inst_bp,
            camera_relative_tf,
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )

        actors.extend(
            [
                rgb_camera,
                semantic_camera,
                instance_camera,
            ]
        )

        rgb_camera.listen(q_rgb.put)
        semantic_camera.listen(q_sem.put)
        instance_camera.listen(q_inst.put)

        # Warm up sensors.
        for _ in range(3):
            frame = world.tick()
            get_sensor_frame(q_rgb, frame, "RGB warmup")
            get_sensor_frame(q_sem, frame, "semantic warmup")
            get_sensor_frame(q_inst, frame, "instance warmup")

        camera_tf = rgb_camera.get_transform()

        print(
            "[camera world]     ",
            f"x={camera_tf.location.x:.3f}",
            f"y={camera_tf.location.y:.3f}",
            f"z={camera_tf.location.z:.3f}",
            f"yaw={camera_tf.rotation.yaw:.3f}",
        )

        # ----------------------------------------------------
        # Physical reference actor resolved from production asset
        # ----------------------------------------------------
        try:
            adv_bp = bp_lib.find(asset.carla_blueprint)
        except Exception as exc:
            patrol_matches = [
                bp.id
                for bp in bp_lib.filter("*patrol*")
            ]
            raise RuntimeError(
                "Could not find production asset CARLA blueprint "
                f"{asset.carla_blueprint!r}. Available patrol-like "
                f"blueprints: {patrol_matches}"
            ) from exc

        if adv_bp.has_attribute("role_name"):
            adv_bp.set_attribute("role_name", "geometry_reference")

        # Spawn somewhere valid, then disable physics and teleport.
        adversary = None

        preferred_indices = [
            min(45, len(spawn_points) - 1),
            len(spawn_points) - 1,
            max(0, spawn_idx - 10),
            (spawn_idx + 20) % len(spawn_points),
        ]

        for idx in preferred_indices:
            adversary = world.try_spawn_actor(
                adv_bp,
                spawn_points[idx],
            )
            if adversary is not None:
                break

        if adversary is None:
            raise RuntimeError(
                f"Could not spawn physical reference actor {asset.carla_blueprint!r}."
            )

        actors.append(adversary)
        adversary.set_simulate_physics(False)

        live_bbox = adversary.bounding_box

        live_bbox_location = {
            "x_m": float(live_bbox.location.x),
            "y_m": float(live_bbox.location.y),
            "z_m": float(live_bbox.location.z),
        }

        live_dimensions = {
            "length_m": 2.0 * float(live_bbox.extent.x),
            "width_m": 2.0 * float(live_bbox.extent.y),
            "height_m": 2.0 * float(live_bbox.extent.z),
        }

        if dimensions is None:
            dimensions = dict(live_dimensions)

        print("[live CARLA bbox] ", live_dimensions)
        print("[live bbox offset]", live_bbox_location)
        print("[HE dimensions]   ", dimensions)

        bbox_delta = {
            k: float(dimensions[k]) - float(live_dimensions[k])
            for k in dimensions
        }
        print("[bbox delta HE-CARLA]", bbox_delta)
        print()

        # ----------------------------------------------------
        # Pose sweep
        # ----------------------------------------------------
        for requested_depth in args.depths:
            requested_depth = float(requested_depth)

            camera_tf = rgb_camera.get_transform()

            target_tf = actor_transform_at_camera_depth(
                carla=carla,
                camera_tf=camera_tf,
                ground_z=float(ego0_tf.location.z),
                depth_m=requested_depth,
                lateral_m=float(args.lateral_m),
                relative_yaw_deg=float(args.relative_yaw_deg),
            )

            # Physical frame.
            adversary.set_transform(target_tf)

            frame = world.tick()

            physical_rgb_image = get_sensor_frame(
                q_rgb,
                frame,
                "RGB physical",
            )
            physical_sem_image = get_sensor_frame(
                q_sem,
                frame,
                "semantic physical",
            )

            physical_inst_image = get_sensor_frame(
                q_inst,
                frame,
                "instance physical",
            )

            physical_rgb = carla_image_to_rgb(
                physical_rgb_image
            )
            physical_vehicle_mask = semantic_vehicle_mask(
                physical_sem_image
            )

            # Capture the same scene without reference actor.
            camera_tf_now = rgb_camera.get_transform()
            forward = camera_tf_now.get_forward_vector()

            hide_tf = carla.Transform(
                carla.Location(
                    x=(
                        float(camera_tf_now.location.x)
                        - 100.0 * float(forward.x)
                    ),
                    y=(
                        float(camera_tf_now.location.y)
                        - 100.0 * float(forward.y)
                    ),
                    z=float(ego0_tf.location.z),
                ),
                carla.Rotation(
                    yaw=float(ego0_tf.rotation.yaw),
                ),
            )

            adversary.set_transform(hide_tf)

            frame = world.tick()

            base_rgb_image = get_sensor_frame(
                q_rgb,
                frame,
                "RGB background",
            )
            background_sem_image = get_sensor_frame(
                q_sem,
                frame,
                "semantic background",
            )

            background_inst_image = get_sensor_frame(
                q_inst,
                frame,
                "instance background",
            )

            base_rgb = carla_image_to_rgb(base_rgb_image)

            # Authoritative physical target mask. Unlike semantic-class
            # subtraction, this survives overlap with the ego vehicle.
            physical_target_mask = instance_change_vehicle_mask(
                physical_instance_image=physical_inst_image,
                background_instance_image=background_inst_image,
                physical_semantic_image=physical_sem_image,
            )

            physical_target_mask = largest_component(
                physical_target_mask
            )
            physical_measurement = measurement_from_mask(
                physical_target_mask
            )

            # Current authoritative HE projection.
            camera_tf_for_he = rgb_camera.get_transform()

            projected_box = project_virtual_actor(
                actor_tf=target_tf,
                camera_tf=camera_tf_for_he,
                dimensions=dimensions,
                width=int(args.width),
                height=int(args.height),
                fov=float(args.fov),
            )

            he_rgb, he_meta = render_he_actor_view_matrix(
                base_rgb=base_rgb,
                actor_tf=target_tf,
                camera_tf=camera_tf_for_he,
                dimensions=dimensions,
                sprite_bank=sprite_bank,
                view_matrix=view_matrix,
                sprite_cache=sprite_cache,
                width=int(args.width),
                height=int(args.height),
                fov=float(args.fov),
                bottom_y_offset_px=float(
                    args.he_bottom_y_offset_px
                ),
                geometry_mode=args.geometry_mode,
                projection_mode=args.projection_mode,
                sprite_geometry=sprite_geometry,
                close_width_blend_near_m=float(
                    args.close_width_blend_near_m
                ),
                close_width_blend_far_m=float(
                    args.close_width_blend_far_m
                ),
            )

            # Prefer the production compositor's exact post-clipping
            # alpha bbox. RGB differencing can miss sprite pixels whose
            # color resembles the background.
            exact_alpha_bbox = he_meta.get(
                "rendered_alpha_bbox"
            )

            if exact_alpha_bbox is not None:
                he_visual_measurement = dict(
                    exact_alpha_bbox
                )
            else:
                he_visual_mask = he_difference_mask(
                    he_rgb,
                    base_rgb,
                )
                he_visual_measurement = measurement_from_mask(
                    he_visual_mask
                )

            box = he_meta.get("box", projected_box) or {}

            target_visible_bbox = he_meta.get(
                "target_visible_bbox_unclipped"
            )

            if target_visible_bbox is None:
                target_visible_bbox = (
                    box
                    if all(
                        key in box
                        for key in (
                            "x1",
                            "y1",
                            "x2",
                            "y2",
                        )
                    )
                    else None
                )

            center_depth = safe_float(box, "depth_m")
            support_depth = safe_float(box, "support_depth_m")
            nearest_depth = safe_float(box, "nearest_depth_m")
            farthest_depth = safe_float(box, "farthest_depth_m")

            he_box_w = safe_float(box, "box_width")
            he_box_h = safe_float(box, "box_height")
            he_box_cx = safe_float(box, "cx")

            if "render_bottom_y" in he_meta:
                he_box_bottom = safe_float(
                    he_meta,
                    "render_bottom_y",
                )
            else:
                he_box_bottom = safe_float(box, "bottom_y")

            physical_w = (
                physical_measurement["width_px"]
                if physical_measurement is not None
                else float("nan")
            )
            physical_h = (
                physical_measurement["height_px"]
                if physical_measurement is not None
                else float("nan")
            )
            physical_cx = (
                physical_measurement["center_x"]
                if physical_measurement is not None
                else float("nan")
            )
            physical_bottom = (
                physical_measurement["bottom_y"]
                if physical_measurement is not None
                else float("nan")
            )

            he_visual_w = (
                he_visual_measurement["width_px"]
                if he_visual_measurement is not None
                else float("nan")
            )
            he_visual_h = (
                he_visual_measurement["height_px"]
                if he_visual_measurement is not None
                else float("nan")
            )

            row = {
                "requested_center_depth_m": requested_depth,
                "lateral_m": float(args.lateral_m),
                "relative_yaw_deg": float(args.relative_yaw_deg),

                "asset_key": asset.key,
                "asset_id": asset.asset_id,
                "carla_blueprint": asset.carla_blueprint,

                "he_length_m": float(dimensions["length_m"]),
                "he_width_m": float(dimensions["width_m"]),
                "he_height_m": float(dimensions["height_m"]),

                "carla_length_m": float(live_dimensions["length_m"]),
                "carla_width_m": float(live_dimensions["width_m"]),
                "carla_height_m": float(live_dimensions["height_m"]),

                "physical_visible": int(physical_measurement is not None),
                "physical_cx_px": physical_cx,
                "physical_bottom_y_px": physical_bottom,
                "physical_width_px": physical_w,
                "physical_height_px": physical_h,
                "physical_area_px": (
                    physical_measurement["area_px"]
                    if physical_measurement is not None
                    else 0
                ),

                "he_rendered": int(bool(he_meta.get("rendered", False))),
                "he_reason": he_meta.get(
                    "reason",
                    box.get("reason", ""),
                ),

                "he_center_depth_m": center_depth,
                "he_support_depth_m": support_depth,
                "he_nearest_depth_m": nearest_depth,
                "he_farthest_depth_m": farthest_depth,

                "center_minus_support_m": (
                    center_depth - support_depth
                    if np.isfinite(center_depth)
                    and np.isfinite(support_depth)
                    else float("nan")
                ),
                "center_over_support": safe_ratio(
                    center_depth,
                    support_depth,
                ),

                "he_box_cx_px": he_box_cx,
                "he_box_bottom_y_px": he_box_bottom,
                "he_box_width_px": he_box_w,
                "he_box_height_px": he_box_h,

                "he_visual_width_px": he_visual_w,
                "he_visual_height_px": he_visual_h,

                "he_box_over_carla_width": safe_ratio(
                    he_box_w,
                    physical_w,
                ),
                "he_box_over_carla_height": safe_ratio(
                    he_box_h,
                    physical_h,
                ),
                "he_visual_over_carla_width": safe_ratio(
                    he_visual_w,
                    physical_w,
                ),
                "he_visual_over_carla_height": safe_ratio(
                    he_visual_h,
                    physical_h,
                ),

                "he_box_cx_error_px": (
                    he_box_cx - physical_cx
                    if np.isfinite(he_box_cx)
                    and np.isfinite(physical_cx)
                    else float("nan")
                ),
                "he_box_bottom_y_error_px": (
                    he_box_bottom - physical_bottom
                    if np.isfinite(he_box_bottom)
                    and np.isfinite(physical_bottom)
                    else float("nan")
                ),

                "viewpoint_angle_deg": he_meta.get(
                    "viewpoint_angle_deg",
                    float("nan"),
                ),
                "selected_angle_deg": he_meta.get(
                    "selected_angle",
                    float("nan"),
                ),
                "query_distance_m": he_meta.get(
                    "query_distance_m",
                    float("nan"),
                ),
                "selected_distance_m": he_meta.get(
                    "selected_distance_m",
                    float("nan"),
                ),
                "query_elevation_deg": he_meta.get(
                    "query_elevation_deg",
                    float("nan"),
                ),
                "selected_elevation_deg": he_meta.get(
                    "selected_elevation_deg",
                    float("nan"),
                ),

                "geometry_mode":
                    he_meta.get(
                        "geometry_mode",
                        args.geometry_mode,
                    ),

                "geometry_version":
                    he_meta.get(
                        "geometry_version"
                    ),

                "geometry_alpha_threshold":
                    he_meta.get(
                        "geometry_alpha_threshold"
                    ),

                "sprite_geometry_query_distance_m":
                    he_meta.get(
                        "sprite_geometry_query_distance_m",
                        float("nan"),
                    ),

                "sprite_geometry_depth_coordinate":
                    he_meta.get(
                        "sprite_geometry_depth_coordinate"
                    ),

                "render_target_width_px":
                    he_meta.get(
                        "target_box_width_px",
                        float("nan"),
                    ),

                "close_width_blend_near_m":
                    he_meta.get(
                        "close_width_blend_near_m",
                        float("nan"),
                    ),

                "close_width_blend_far_m":
                    he_meta.get(
                        "close_width_blend_far_m",
                        float("nan"),
                    ),

                "close_width_blend_weight_native":
                    he_meta.get(
                        "close_width_blend_weight_native",
                        float("nan"),
                    ),

                "close_width_proxy_width_px":
                    he_meta.get(
                        "close_width_proxy_width_px",
                        float("nan"),
                    ),

                "close_width_native_width_px":
                    he_meta.get(
                        "close_width_native_width_px",
                        float("nan"),
                    ),

                "close_width_final_width_px":
                    he_meta.get(
                        "close_width_final_width_px",
                        float("nan"),
                    ),

                "render_target_height_px":
                    he_meta.get(
                        "target_box_height_px",
                        float("nan"),
                    ),

                "sprite_geometry_predicted_width_px":
                    (
                        (
                            he_meta.get(
                                "sprite_native_geometry"
                            )
                            or
                            {}
                        ).get(
                            "box_width_px",
                            float("nan"),
                        )
                    ),

                "sprite_geometry_predicted_height_px":
                    (
                        (
                            he_meta.get(
                                "sprite_native_geometry"
                            )
                            or
                            {}
                        ).get(
                            "box_height_px",
                            float("nan"),
                        )
                    ),

                "target_visible_x1_px":
                    (
                        target_visible_bbox.get(
                            "x1"
                        )
                        if target_visible_bbox is not None
                        else float("nan")
                    ),

                "target_visible_y1_px":
                    (
                        target_visible_bbox.get(
                            "y1"
                        )
                        if target_visible_bbox is not None
                        else float("nan")
                    ),

                "target_visible_x2_px":
                    (
                        target_visible_bbox.get(
                            "x2"
                        )
                        if target_visible_bbox is not None
                        else float("nan")
                    ),

                "target_visible_y2_px":
                    (
                        target_visible_bbox.get(
                            "y2"
                        )
                        if target_visible_bbox is not None
                        else float("nan")
                    ),

                "target_visible_width_px":
                    (
                        target_visible_bbox.get(
                            "width_px",
                            float("nan"),
                        )
                        if target_visible_bbox is not None
                        else float("nan")
                    ),

                "target_visible_height_px":
                    (
                        target_visible_bbox.get(
                            "height_px",
                            float("nan"),
                        )
                        if target_visible_bbox is not None
                        else float("nan")
                    ),

                "source_anchor_x_px":
                    he_meta.get(
                        "source_anchor_x_px",
                        float("nan"),
                    ),

                "source_anchor_y_px":
                    he_meta.get(
                        "source_anchor_y_px",
                        float("nan"),
                    ),

                "target_anchor_x_px":
                    he_meta.get(
                        "target_anchor_x_px",
                        float("nan"),
                    ),

                "target_anchor_y_px":
                    he_meta.get(
                        "target_anchor_y_px",
                        float("nan"),
                    ),

                "anchor_mode":
                    he_meta.get(
                        "anchor_mode"
                    ),
            }

            results.append(row)

            depth_tag = (
                f"{requested_depth:05.2f}"
                .replace(".", "p")
            )

            physical_annotated = annotate(
                physical_rgb,
                measurement=physical_measurement,
                title=(
                    f"CARLA z={requested_depth:.2f} "
                    f"{fstr(physical_w,0)}x{fstr(physical_h,0)}"
                ),
            )

            he_annotated = annotate(
                he_rgb,
                measurement=he_visual_measurement,
                box=target_visible_bbox,
                title=(
                    f"HE zc={fstr(center_depth)} "
                    f"zs={fstr(support_depth)} "
                    f"box={fstr(he_box_w,0)}x{fstr(he_box_h,0)}"
                ),
            )

            overlay_rgb = (
                0.5 * physical_rgb.astype(np.float32)
                + 0.5 * he_rgb.astype(np.float32)
            )
            overlay_rgb = np.clip(
                overlay_rgb,
                0,
                255,
            ).astype(np.uint8)

            overlay_annotated = annotate(
                overlay_rgb,
                measurement=physical_measurement,
                box=target_visible_bbox,
                title="overlay: green=CARLA mask, red=HE target box",
            )

            comparison = np.hstack(
                [
                    physical_annotated,
                    he_annotated,
                    overlay_annotated,
                ]
            )

            cv2.imwrite(
                str(
                    images_dir
                    / f"depth_{depth_tag}_carla.png"
                ),
                physical_annotated,
            )
            cv2.imwrite(
                str(
                    images_dir
                    / f"depth_{depth_tag}_he.png"
                ),
                he_annotated,
            )
            cv2.imwrite(
                str(
                    images_dir
                    / f"depth_{depth_tag}_overlay.png"
                ),
                overlay_annotated,
            )
            cv2.imwrite(
                str(
                    images_dir
                    / f"depth_{depth_tag}_comparison.png"
                ),
                comparison,
            )

            print(
                f"[z={requested_depth:5.2f}] "
                f"CARLA={fstr(physical_w,0)}x{fstr(physical_h,0)} "
                f"HEbox={fstr(he_box_w,0)}x{fstr(he_box_h,0)} "
                f"Hratio={fstr(row['he_box_over_carla_height'],3)} "
                f"center={fstr(center_depth)} "
                f"support={fstr(support_depth)} "
                f"near={fstr(nearest_depth)} "
                f"center/support={fstr(row['center_over_support'],3)} "
                f"view={fstr(row['viewpoint_angle_deg'],1)} "
                f"A={fstr(row['selected_angle_deg'],0)} "
                f"E={fstr(row['query_elevation_deg'],1)}"
                f"->{fstr(row['selected_elevation_deg'],0)} "
                f"rendered={row['he_rendered']} "
                f"reason={row['he_reason']}"
            )

        if results:
            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=list(results[0].keys()),
                )
                writer.writeheader()
                writer.writerows(results)

        print()
        print("=" * 92)
        print("CLOSE-RANGE GEOMETRY SWEEP COMPLETE")
        print("=" * 92)
        print("[CSV]   ", csv_path)
        print("[images]", images_dir)
        print()
        print("Key columns to inspect:")
        print("  he_center_depth_m")
        print("  he_support_depth_m")
        print("  he_nearest_depth_m")
        print("  center_over_support")
        print("  he_box_over_carla_height")
        print("  he_box_over_carla_width")
        print("=" * 92)

    finally:
        # Stop sensors first.
        for sensor in (
            rgb_camera,
            semantic_camera,
            instance_camera,
        ):
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass

        for actor in reversed(actors):
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass

        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass


if __name__ == "__main__":
    main()
