#!/usr/bin/env python3
"""
generate_carla_360_rgba_fixed_camera.py

Generate 360-degree RGBA vehicle sprites in CARLA.

Design:
  - Does NOT load/change map.
  - Uses current/default CARLA world.
  - Fixed RGB + semantic camera.
  - Vehicle rotates in place.
  - Uses projected 3D bbox to isolate target car.
  - Uses semantic vehicle mask inside bbox.
  - Fills holes so windows are included in alpha.
  - Window holes are painted black/dark/tinted.
  - Saves RGBA, RGB crop, alpha mask, debug, metadata.

Angle convention:
  angle_000 = rear view
  angle_090 = side view
  angle_180 = front view, same as old front-facing sprite
  angle_270 = opposite side view

First test:
  python generate_carla_360_rgba_fixed_camera.py ^
    --vehicle-blueprint vehicle.tesla.model3 ^
    --vehicle-name vehicle_blue_sedan_rgba_test ^
    --angle-step 30 ^
    --output-root assets\\sprite_bank_rgba ^
    --overwrite

Final:
  python generate_carla_360_rgba_fixed_camera.py ^
    --vehicle-blueprint vehicle.tesla.model3 ^
    --vehicle-name vehicle_blue_sedan ^
    --angle-step 1 ^
    --output-root assets\\sprite_bank_rgba ^
    --overwrite
"""

import argparse
import csv
import json
import math
import queue
import random
import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import carla


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-mask-pixels", type=int, default=500)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)

    parser.add_argument("--vehicle-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--vehicle-name", default="vehicle_blue_sedan")
    parser.add_argument("--color", default="0,0,255")

    parser.add_argument("--output-root", default="assets/sprite_bank_rgba")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=55.0)

    parser.add_argument("--camera-distance", type=float, default=10.0)
    parser.add_argument("--camera-height", type=float, default=1.35)
    parser.add_argument("--target-height", type=float, default=1.0)

    parser.add_argument("--angle-step", type=int, default=1)

    parser.add_argument(
        "--spawn-index",
        type=int,
        default=-1,
        help="-1 means random valid spawn point."
    )

    parser.add_argument("--crop-margin-px", type=int, default=30)
    parser.add_argument("--bbox-expand-ratio", type=float, default=0.10)

    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--settle-ticks", type=int, default=2)

    parser.add_argument(
        "--vehicle-semantic-tag",
        type=int,
        default=14,
        help="CARLA semantic tag for Car in CARLA 0.9.15."
    )

    parser.add_argument(
        "--window-mode",
        choices=["black", "darken", "keep"],
        default="black",
        help="How to treat transparent/glass window holes."
    )

    parser.add_argument(
        "--window-black-value",
        type=int,
        default=18,
        help="RGB value for black windows when --window-mode black."
    )

    parser.add_argument(
        "--window-darken-factor",
        type=float,
        default=0.25,
        help="Multiplier when --window-mode darken."
    )

    parser.add_argument(
        "--no-clear-existing-actors",
        action="store_true",
        help="Do not clear existing vehicles/walkers/sensors."
    )

    parser.add_argument(
        "--save-all-debug",
        action="store_true",
        help="Save debug for every angle. Otherwise every 30 degrees."
    )
    parser.add_argument(
        "--road-only-scene",
        action="store_true",
        help=(
            "Temporarily unload optional Town*_Opt map layers "
            "during sprite capture so the target vehicle cannot "
            "be occluded by buildings, foliage, props, parked "
            "vehicles, street lights, or walls."
        )
    )
    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================
def get_sprite_capture_layers():
    """
    Optional CARLA map layers that should not be visible while
    generating isolated vehicle sprites.

    IMPORTANT:
    Ground is intentionally NOT removed because it contains the
    road/ground surface we want to preserve.
    """

    names = [
        "Buildings",
        "Decals",
        "Foliage",
        "ParkedVehicles",
        "Particles",
        "Props",
        "StreetLights",
        "Walls",
    ]

    layers = []

    for name in names:

        if hasattr(
            carla.MapLayer,
            name
        ):
            layers.append(
                (
                    name,
                    getattr(
                        carla.MapLayer,
                        name
                    ),
                )
            )

    return layers


def unload_sprite_capture_layers(
    world,
):
    """
    Hide non-road optional layers in an optimized CARLA map.

    Intended for Town*_Opt maps.
    """

    unloaded = []

    print()
    print(
        "[RGBABank] Enabling road-only capture scene..."
    )

    for (
        name,
        layer,
    ) in get_sprite_capture_layers():

        try:

            world.unload_map_layer(
                layer
            )

            unloaded.append(
                (
                    name,
                    layer,
                )
            )

            print(
                f"[RGBABank] unloaded layer: {name}"
            )

        except Exception as e:

            print(
                f"[RGBABank] could not unload {name}: {e}"
            )

    # Give CARLA time to process layer changes.
    for _ in range(5):
        world.tick()

    return unloaded


def restore_sprite_capture_layers(
    world,
    unloaded_layers,
):
    """
    Restore layers removed for sprite generation.
    """

    if not unloaded_layers:
        return

    print()
    print(
        "[RGBABank] Restoring map layers..."
    )

    for (
        name,
        layer,
    ) in unloaded_layers:

        try:

            world.load_map_layer(
                layer
            )

            print(
                f"[RGBABank] restored layer: {name}"
            )

        except Exception as e:

            print(
                f"[RGBABank] could not restore {name}: {e}"
            )

    for _ in range(5):
        world.tick()

def make_grabcut_alpha_from_bbox(rgb, bbox, args):
    """
    Fallback/main alpha extraction using RGB + projected bbox.
    This is closer to the old front-facing sprite method.
    """
    image_h, image_w = rgb.shape[:2]

    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * args.bbox_expand_ratio) + args.crop_margin_px
    my = int(bh * args.bbox_expand_ratio) + args.crop_margin_px

    x1e = max(0, x1 - mx)
    y1e = max(0, y1 - my)
    x2e = min(image_w - 1, x2 + mx)
    y2e = min(image_h - 1, y2 + my)

    crop_rgb = rgb[y1e:y2e + 1, x1e:x2e + 1].copy()

    if crop_rgb.size == 0:
        raise RuntimeError("Empty GrabCut crop.")

    crop_h, crop_w = crop_rgb.shape[:2]

    mask = np.full((crop_h, crop_w), cv2.GC_PR_BGD, dtype=np.uint8)

    # probable foreground: inner region
    fg_margin_x = max(3, int(crop_w * 0.12))
    fg_margin_y = max(3, int(crop_h * 0.10))

    mask[
        fg_margin_y:crop_h - fg_margin_y,
        fg_margin_x:crop_w - fg_margin_x
    ] = cv2.GC_PR_FGD

    # strong background border
    border = max(4, int(min(crop_w, crop_h) * 0.04))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    cv2.grabCut(
        crop_bgr,
        mask,
        None,
        bgd_model,
        fgd_model,
        7,
        cv2.GC_INIT_WITH_MASK
    )

    crop_alpha = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Clean crop alpha.
    kernel = np.ones((5, 5), dtype=np.uint8)
    crop_alpha = cv2.morphologyEx(crop_alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    crop_alpha = cv2.morphologyEx(crop_alpha, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest component.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (crop_alpha > 0).astype(np.uint8),
        connectivity=8
    )

    if num_labels > 1:
        largest_label = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]

        for label in range(2, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_label = label

        crop_alpha = (labels == largest_label).astype(np.uint8) * 255

    crop_alpha = cv2.GaussianBlur(crop_alpha, (3, 3), 0)

    full_alpha = np.zeros((image_h, image_w), dtype=np.uint8)
    full_alpha[y1e:y2e + 1, x1e:x2e + 1] = crop_alpha

    return full_alpha

def safe_destroy(actor):
    if actor is None:
        return

    try:
        if actor.type_id.startswith("sensor."):
            actor.stop()
    except Exception:
        pass

    try:
        actor.destroy()
    except Exception:
        pass


def clear_existing_dynamic_actors(world):
    patterns = ["vehicle.*", "walker.*", "sensor.*"]
    actors = []

    for pattern in patterns:
        actors.extend(list(world.get_actors().filter(pattern)))

    print(f"[RGBABank] Clearing {len(actors)} existing vehicles/walkers/sensors...")

    for actor in actors:
        safe_destroy(actor)

    world.tick()


def setup_synchronous_mode(world, fixed_delta_seconds):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)


def restore_world_settings(world, original_settings):
    if world is not None and original_settings is not None:
        world.apply_settings(original_settings)


def flush_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def wait_for_frame(q, expected_frame, timeout=5.0):
    while True:
        image = q.get(timeout=timeout)
        if image.frame >= expected_frame:
            return image


def ensure_dirs(output_dir, overwrite):
    output_dir = Path(output_dir)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    rgba_dir = output_dir / "rgba"
    rgb_dir = output_dir / "rgb"
    mask_dir = output_dir / "mask"
    window_dir = output_dir / "window_mask"
    debug_dir = output_dir / "debug"

    rgba_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    window_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    return rgba_dir, rgb_dir, mask_dir, window_dir, debug_dir


# ============================================================
# Image conversion
# ============================================================

def carla_rgb_to_array(image):
    """
    CARLA RGB raw buffer is BGRA.
    Return RGB array.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]
    return rgb.copy()


def carla_semantic_to_tags(image):
    """
    CARLA semantic raw buffer is BGRA.
    Semantic class id is usually red channel, index 2.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    tags = arr[:, :, 2]
    return tags.copy()


# ============================================================
# Camera geometry
# ============================================================

def build_camera_intrinsics(width, height, fov_degrees):
    fov = np.deg2rad(fov_degrees)
    focal = width / (2.0 * np.tan(fov / 2.0))

    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K


def project_carla_world_to_image(world_location, camera_actor, K):
    camera_tf = camera_actor.get_transform()
    world_2_camera = np.array(camera_tf.get_inverse_matrix())

    point_world = np.array([
        world_location.x,
        world_location.y,
        world_location.z,
        1.0
    ])

    point_camera = world_2_camera @ point_world

    # CARLA camera: x forward, y right, z up
    # OpenCV: x right, y down, z forward
    x_cv = point_camera[1]
    y_cv = -point_camera[2]
    z_cv = point_camera[0]

    if z_cv <= 0.1:
        return None

    point_img = K @ np.array([x_cv, y_cv, z_cv])

    u = point_img[0] / point_img[2]
    v = point_img[1] / point_img[2]

    return float(u), float(v), float(z_cv)


def compute_actor_2d_bbox(actor, camera_actor, K, image_width, image_height):
    bbox = actor.bounding_box
    actor_tf = actor.get_transform()
    vertices = bbox.get_world_vertices(actor_tf)

    projected = []

    for vertex in vertices:
        p = project_carla_world_to_image(vertex, camera_actor, K)

        if p is None:
            continue

        u, v, _ = p

        if u < -image_width or u > 2 * image_width:
            continue

        if v < -image_height or v > 2 * image_height:
            continue

        projected.append((u, v))

    if len(projected) < 2:
        return None

    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]

    x1 = max(0, min(image_width - 1, min(xs)))
    y1 = max(0, min(image_height - 1, min(ys)))
    x2 = max(0, min(image_width - 1, max(xs)))
    y2 = max(0, min(image_height - 1, max(ys)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [int(x1), int(y1), int(x2), int(y2)]


def look_at_rotation(camera_location, target_location):
    dx = target_location.x - camera_location.x
    dy = target_location.y - camera_location.y
    dz = target_location.z - camera_location.z

    yaw = np.rad2deg(np.arctan2(dy, dx))
    horizontal_dist = np.sqrt(dx * dx + dy * dy)
    pitch = np.rad2deg(np.arctan2(dz, horizontal_dist))

    return carla.Rotation(
        pitch=float(pitch),
        yaw=float(yaw),
        roll=0.0
    )


def build_fixed_front_camera_transform(vehicle, camera_distance, camera_height, target_height):
    vehicle_tf = vehicle.get_transform()
    vehicle_loc = vehicle_tf.location
    vehicle_yaw = vehicle_tf.rotation.yaw

    yaw_rad = np.deg2rad(vehicle_yaw)

    forward_x = np.cos(yaw_rad)
    forward_y = np.sin(yaw_rad)

    camera_location = carla.Location(
        x=vehicle_loc.x + camera_distance * forward_x,
        y=vehicle_loc.y + camera_distance * forward_y,
        z=vehicle_loc.z + camera_height
    )

    target_location = carla.Location(
        x=vehicle_loc.x,
        y=vehicle_loc.y,
        z=vehicle_loc.z + target_height
    )

    camera_rotation = look_at_rotation(camera_location, target_location)

    return carla.Transform(camera_location, camera_rotation)


def vehicle_yaw_for_angle(base_yaw, angle_deg):
    """
    Fixed camera.

    angle_180 = front view:
      vehicle yaw = base_yaw

    angle_000 = rear view:
      vehicle yaw = base_yaw + 180
    """
    return base_yaw + (180.0 - float(angle_deg))


# ============================================================
# CARLA setup
# ============================================================

def find_vehicle_blueprint(blueprint_library, blueprint_id):
    try:
        return blueprint_library.find(blueprint_id)
    except RuntimeError:
        matches = blueprint_library.filter(blueprint_id)

        if len(matches) > 0:
            return matches[0]

        vehicles = blueprint_library.filter("vehicle.*")

        if len(vehicles) == 0:
            raise RuntimeError("No vehicle blueprints found.")

        print("[RGBABank] Warning: requested vehicle not found. Using:", vehicles[0].id)
        return vehicles[0]


def spawn_vehicle(world, blueprint_library, args):
    vehicle_bp = find_vehicle_blueprint(blueprint_library, args.vehicle_blueprint)

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", args.color)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "he_fixed_camera_rgba_target")

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points found.")

    if args.spawn_index >= 0:
        candidates = [spawn_points[min(args.spawn_index, len(spawn_points) - 1)]]
    else:
        candidates = list(spawn_points)
        random.shuffle(candidates)

    for sp in candidates:
        tf = carla.Transform(
            sp.location,
            carla.Rotation(
                pitch=0.0,
                yaw=sp.rotation.yaw,
                roll=0.0
            )
        )

        vehicle = world.try_spawn_actor(vehicle_bp, tf)

        if vehicle is not None:
            vehicle.set_autopilot(False)
            vehicle.set_simulate_physics(False)

            world.tick()

            print("[RGBABank] Spawned vehicle:", vehicle_bp.id)
            print("[RGBABank] Location:", vehicle.get_location())
            print("[RGBABank] Base yaw:", vehicle.get_transform().rotation.yaw)

            return vehicle, vehicle_bp.id

    raise RuntimeError("Could not spawn target vehicle. Try a different --spawn-index.")


def spawn_cameras(world, blueprint_library, camera_tf, args):
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(args.image_width))
    rgb_bp.set_attribute("image_size_y", str(args.image_height))
    rgb_bp.set_attribute("fov", str(args.fov))
    rgb_bp.set_attribute("sensor_tick", "0.0")

    seg_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    seg_bp.set_attribute("image_size_x", str(args.image_width))
    seg_bp.set_attribute("image_size_y", str(args.image_height))
    seg_bp.set_attribute("fov", str(args.fov))
    seg_bp.set_attribute("sensor_tick", "0.0")

    rgb_camera = world.spawn_actor(rgb_bp, camera_tf)
    seg_camera = world.spawn_actor(seg_bp, camera_tf)

    print("[RGBABank] Fixed camera location:", camera_tf.location)
    print("[RGBABank] Fixed camera rotation:", camera_tf.rotation)

    return rgb_camera, seg_camera


# ============================================================
# Mask logic
# ============================================================

def expand_bbox(bbox, image_width, image_height, expand_ratio, extra_margin_px):
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * expand_ratio) + extra_margin_px
    my = int(bh * expand_ratio) + extra_margin_px

    x1e = max(0, x1 - mx)
    y1e = max(0, y1 - my)
    x2e = min(image_width - 1, x2 + mx)
    y2e = min(image_height - 1, y2 + my)

    return [x1e, y1e, x2e, y2e]

def get_visible_mask_bbox(mask_u8):
    """
    Return the tight visible bounding box of a binary/alpha mask.

    Returns:
        (x1, y1, x2, y2)

    or None if the mask contains no visible pixels.
    """

    ys, xs = np.where(
        mask_u8 > 0
    )

    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):
        return None

    return (
        int(
            xs.min()
        ),
        int(
            ys.min()
        ),
        int(
            xs.max()
        ),
        int(
            ys.max()
        ),
    )
def keep_largest_component(mask_u8):
    binary = (mask_u8 > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return mask_u8

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for label in range(2, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area > largest_area:
            largest_area = area
            largest_label = label

    return (labels == largest_label).astype(np.uint8) * 255


def fill_holes(mask_u8):
    """
    Fill holes in the vehicle body mask.
    This makes window/glass holes part of the alpha silhouette.
    """
    binary = (mask_u8 > 0).astype(np.uint8) * 255

    h, w = binary.shape[:2]
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = binary | flood_inv

    return filled


def clean_body_mask(mask_u8):
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255

    kernel = np.ones((5, 5), dtype=np.uint8)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    mask_u8 = keep_largest_component(mask_u8)

    return mask_u8


def feather_alpha(mask_u8):
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    mask_u8 = cv2.GaussianBlur(mask_u8, (3, 3), 0)
    return mask_u8


def make_alpha_and_window_mask(tags, bbox, args):
    """
    original_body_mask:
      semantic vehicle pixels only.

    filled_alpha_mask:
      semantic mask with holes filled, so windows become part of alpha.

    window_mask:
      newly filled hole regions, likely windshield/side windows.
    """
    bbox_exp = expand_bbox(
        bbox=bbox,
        image_width=args.image_width,
        image_height=args.image_height,
        expand_ratio=args.bbox_expand_ratio,
        extra_margin_px=args.crop_margin_px
    )

    x1, y1, x2, y2 = bbox_exp

    full_body = np.zeros((args.image_height, args.image_width), dtype=np.uint8)

    roi_tags = tags[y1:y2 + 1, x1:x2 + 1]
    roi_vehicle = (roi_tags == args.vehicle_semantic_tag).astype(np.uint8) * 255

    full_body[y1:y2 + 1, x1:x2 + 1] = roi_vehicle

    body_clean = clean_body_mask(full_body)
    filled = fill_holes(body_clean)

    window_mask = cv2.subtract(filled, body_clean)

    # Remove tiny noisy filled holes.
    window_mask = keep_largest_components_by_area(window_mask, min_area=80, max_components=12)

    alpha = feather_alpha(filled)

    return alpha, body_clean, window_mask


def keep_largest_components_by_area(mask_u8, min_area=80, max_components=12):
    binary = (mask_u8 > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return mask_u8

    components = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            components.append((area, label))

    components.sort(reverse=True)
    selected = components[:max_components]

    out = np.zeros_like(mask_u8)

    for _, label in selected:
        out[labels == label] = 255

    return out

def keep_components_touching_seed(
    candidate_mask,
    seed_mask,
):
    """
    Keep candidate connected components that touch the trusted
    semantic actor mask.

    This rejects most background-difference noise while allowing
    RGB-derived glass/roof/mirror regions to join the actor.
    """

    candidate = (
        candidate_mask > 0
    ).astype(
        np.uint8
    )

    seed = (
        seed_mask > 0
    )

    (
        num_labels,
        labels,
        _,
        _,
    ) = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )

    if num_labels <= 1:
        return (
            candidate
            *
            255
        ).astype(
            np.uint8
        )

    touching_labels = np.unique(
        labels[
            seed
        ]
    )

    touching_labels = {
        int(label)
        for label in touching_labels
        if int(label) != 0
    }

    out = np.zeros_like(
        candidate_mask,
        dtype=np.uint8,
    )

    for label in touching_labels:

        out[
            labels == label
        ] = 255

    return out


def make_alpha_from_semantic_and_background(
    rgb,
    background_rgb,
    tags,
    bbox,
    args,
):
    """
    Hybrid actor alpha extraction.

    Trusted source:
        CARLA semantic vehicle mask.

    Recovery source:
        RGB difference between actor-present and clean-background
        frames, but ONLY in the upper vehicle region where glass,
        roof and pillars are commonly missed.

    Important:
        RGB recovery is hard-clipped to the projected 3-D actor
        bounding box so vehicle shadows cannot become part of
        the sprite.
    """

    image_h, image_w = rgb.shape[:2]

    # ========================================================
    # Exact projected actor ROI
    # ========================================================

    bx1, by1, bx2, by2 = [
        int(v)
        for v in bbox
    ]

    # Tiny safety margin only.
    pad = max(
        2,
        int(
            round(
                min(
                    bx2 - bx1 + 1,
                    by2 - by1 + 1,
                )
                * 0.01
            )
        ),
    )

    rx1 = max(
        0,
        bx1 - pad,
    )

    ry1 = max(
        0,
        by1 - pad,
    )

    rx2 = min(
        image_w - 1,
        bx2 + pad,
    )

    ry2 = min(
        image_h - 1,
        by2 + pad,
    )

    actor_roi = np.zeros(
        (
            image_h,
            image_w,
        ),
        dtype=np.uint8,
    )

    actor_roi[
        ry1:
        ry2 + 1,
        rx1:
        rx2 + 1,
    ] = 255

    # ========================================================
    # Trusted semantic body
    # ========================================================

    semantic_body = np.zeros(
        (
            image_h,
            image_w,
        ),
        dtype=np.uint8,
    )

    roi_tags = tags[
        ry1:
        ry2 + 1,
        rx1:
        rx2 + 1,
    ]

    semantic_body[
        ry1:
        ry2 + 1,
        rx1:
        rx2 + 1,
    ] = (
        roi_tags
        ==
        int(
            args.vehicle_semantic_tag
        )
    ).astype(
        np.uint8
    ) * 255

    semantic_body = clean_body_mask(
        semantic_body
    )

    semantic_bbox = get_visible_mask_bbox(
        semantic_body
    )

    if semantic_bbox is None:

        alpha = make_grabcut_alpha_from_bbox(
            rgb=rgb,
            bbox=bbox,
            args=args,
        )

        return (
            alpha,
            alpha.copy(),
            np.zeros_like(
                alpha
            ),
        )

    sx1, sy1, sx2, sy2 = semantic_bbox

    semantic_height = max(
        1,
        sy2 - sy1 + 1,
    )

    # ========================================================
    # RGB change mask
    # ========================================================

    diff = np.abs(
        rgb.astype(
            np.int16
        )
        -
        background_rgb.astype(
            np.int16
        )
    )

    # Require meaningful RGB difference.
    diff_strength = np.max(
        diff,
        axis=2,
    )

    diff_threshold = int(
        getattr(
            args,
            "rgb_diff_threshold",
            12,
        )
    )

    diff_mask = (
        diff_strength
        >=
        diff_threshold
    ).astype(
        np.uint8
    ) * 255

    # HARD actor-bbox clipping.
    #
    # This is the key shadow fix.
    diff_mask = cv2.bitwise_and(
        diff_mask,
        actor_roi,
    )

    # ========================================================
    # Recover ONLY upper-body regions
    # ========================================================

    upper_fraction = float(
        getattr(
            args,
            "rgb_recovery_upper_fraction",
            0.72,
        )
    )

    upper_bottom = int(
        round(
            sy1
            +
            semantic_height
            *
            upper_fraction
        )
    )

    upper_bottom = max(
        sy1,
        min(
            sy2,
            upper_bottom,
        ),
    )

    upper_region = np.zeros(
        (
            image_h,
            image_w,
        ),
        dtype=np.uint8,
    )

    upper_region[
        max(
            ry1,
            sy1 - pad
        ):
        min(
            ry2,
            upper_bottom
        )
        + 1,

        rx1:
        rx2 + 1,
    ] = 255

    rgb_recovery = cv2.bitwise_and(
        diff_mask,
        upper_region,
    )

    # ========================================================
    # Join semantic + recovered upper body
    # ========================================================

    candidate = cv2.bitwise_or(
        semantic_body,
        rgb_recovery,
    )

    # Small closing only to bridge 1–2 px gaps.
    bridge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            3,
            3,
        ),
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        bridge_kernel,
        iterations=1,
    )

    candidate = keep_components_touching_seed(
        candidate,
        semantic_body,
    )

    # ========================================================
    # Final alpha
    #
    # IMPORTANT:
    # Do NOT globally fill every enclosed hole.
    #
    # A global fill can incorrectly include:
    #   - road between the wheels
    #   - underbody gaps
    #   - wheel-arch background
    #
    # We only retain newly-filled holes in the upper body,
    # where windshield / windows / panoramic roof occur.
    # ========================================================

    globally_filled = fill_holes(
        candidate
    )

    newly_filled = cv2.subtract(
        globally_filled,
        candidate,
    )

    upper_hole_fill = cv2.bitwise_and(
        newly_filled,
        upper_region,
    )

    final_filled = cv2.bitwise_or(
        candidate,
        upper_hole_fill,
    )

    final_filled = cv2.bitwise_and(
        final_filled,
        actor_roi,
    )

    alpha = feather_alpha(
        final_filled
    )

    # ========================================================
    # Window mask
    #
    # IMPORTANT:
    # Derive windows ONLY from semantic enclosed holes.
    #
    # RGB-recovered roof/glass pixels must NOT automatically
    # become black windows. That caused the large black blocks
    # in the previous experiment.
    # ========================================================

    semantic_filled = fill_holes(
        semantic_body
    )

    semantic_holes = cv2.subtract(
        semantic_filled,
        semantic_body,
    )
    # Window holes are valid only in the upper vehicle region.
    # This prevents underbody/wheel gaps from being treated as
    # windows.
    semantic_holes = cv2.bitwise_and(
        semantic_holes,
        upper_region,
    )
    filled_area = int(
        np.sum(
            final_filled > 0
        )
    )

    min_window_area = max(
        2,
        int(
            round(
                filled_area
                *
                0.00005
            )
        ),
    )

    window_mask = keep_largest_components_by_area(
        semantic_holes,
        min_area=min_window_area,
        max_components=32,
    )

    window_mask = cv2.bitwise_and(
        window_mask,
        final_filled,
    )

    return (
        alpha,
        semantic_body,
        window_mask,
    )

def apply_window_treatment(rgb, window_mask, args):
    out = rgb.copy()
    mask = window_mask > 0

    if args.window_mode == "keep":
        return out

    if args.window_mode == "black":
        v = int(np.clip(args.window_black_value, 0, 255))
        out[mask] = np.array([v, v, v], dtype=np.uint8)
        return out

    if args.window_mode == "darken":
        factor = float(np.clip(args.window_darken_factor, 0.0, 1.0))
        out[mask] = (out[mask].astype(np.float32) * factor).astype(np.uint8)
        return out

    return out


# ============================================================
# Save / visualization
# ============================================================

def crop_rgba_outputs(rgb_treated, alpha, window_mask, bbox, args):
    ys, xs = np.where(alpha > 10)

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Empty alpha mask.")

    image_h, image_w = alpha.shape[:2]

    x1 = max(0, int(xs.min()) - args.crop_margin_px)
    y1 = max(0, int(ys.min()) - args.crop_margin_px)
    x2 = min(image_w - 1, int(xs.max()) + args.crop_margin_px)
    y2 = min(image_h - 1, int(ys.max()) + args.crop_margin_px)

    crop_rgb = rgb_treated[y1:y2 + 1, x1:x2 + 1].copy()
    crop_alpha = alpha[y1:y2 + 1, x1:x2 + 1].copy()
    crop_window = window_mask[y1:y2 + 1, x1:x2 + 1].copy()

    rgba = np.dstack([crop_rgb, crop_alpha])

    # Remove invisible RGB background.
    rgba[crop_alpha <= 5, :3] = 0

    crop_ys, crop_xs = np.where(crop_alpha > 10)
    anchor_x = float((crop_xs.min() + crop_xs.max()) / 2.0)
    anchor_y = float(crop_ys.max())

    crop_box = [x1, y1, x2, y2]

    return crop_rgb, crop_alpha, crop_window, rgba, crop_box, {"x": anchor_x, "y": anchor_y}


def save_rgb(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_mask(path, mask):
    cv2.imwrite(str(path), mask)


def save_rgba(path, rgba):
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), bgra)


def draw_debug(rgb, bbox, crop_box, alpha, window_mask, angle, yaw):
    debug = rgb.copy()

    # Alpha overlay green.
    overlay = np.zeros_like(debug)
    overlay[:, :, 1] = alpha
    debug = cv2.addWeighted(debug, 0.75, overlay, 0.25, 0)

    # Window overlay red.
    win = window_mask > 0
    debug[win] = (0.5 * debug[win] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(
        debug,
        f"angle={angle:03d} yaw={yaw:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return debug


def make_contact_sheet(image_paths, output_path, title=None, thumb_size=(280, 170), cols=4):
    if not image_paths:
        return

    rows = math.ceil(len(image_paths) / cols)
    label_h = 28
    title_h = 34 if title else 0

    sheet_w = cols * thumb_size[0]
    sheet_h = title_h + rows * (thumb_size[1] + label_h)

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    if title:
        draw.text((10, 8), title, fill="black")

    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert("RGBA")

        bg = Image.new("RGBA", img.size, "white")
        bg.alpha_composite(img)

        thumb = ImageOps.contain(bg.convert("RGB"), thumb_size)

        col = idx % cols
        row = idx // cols

        x0 = col * thumb_size[0]
        y0 = title_h + row * (thumb_size[1] + label_h)

        px = x0 + (thumb_size[0] - thumb.width) // 2
        py = y0 + (thumb_size[1] - thumb.height) // 2

        sheet.paste(thumb, (px, py))
        draw.text((x0 + 8, y0 + thumb_size[1] + 5), Path(path).stem, fill="black")

    sheet.save(output_path)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_root) / args.vehicle_name
    rgba_dir, rgb_dir, mask_dir, window_dir, debug_dir = ensure_dirs(
        output_dir,
        overwrite=args.overwrite
    )

    client = None
    world = None
    original_settings = None
    actors = []

    rgb_queue = queue.Queue()
    seg_queue = queue.Queue()
    unloaded_map_layers = []
    metadata_rows = []
    contact_rgba_paths = []
    contact_debug_paths = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        world = client.get_world()
        original_settings = world.get_settings()

        print("[RGBABank] Using current/default CARLA world.")
        print("[RGBABank] No map/town will be loaded or changed.")
        print("[RGBABank] Current map:", world.get_map().name)

        setup_synchronous_mode(world, args.fixed_delta_seconds)
        if args.road_only_scene:

            unloaded_map_layers = (
                unload_sprite_capture_layers(
                    world
                )
            )
        if not args.no_clear_existing_actors:
            clear_existing_dynamic_actors(world)

        blueprint_library = world.get_blueprint_library()

        vehicle, actual_bp = spawn_vehicle(world, blueprint_library, args)
        actors.append(vehicle)

        base_tf = vehicle.get_transform()
        base_location = carla.Location(
            x=base_tf.location.x,
            y=base_tf.location.y,
            z=base_tf.location.z
        )
        base_yaw = float(base_tf.rotation.yaw)

        fixed_camera_tf = build_fixed_front_camera_transform(
            vehicle=vehicle,
            camera_distance=args.camera_distance,
            camera_height=args.camera_height,
            target_height=args.target_height
        )

        rgb_camera, seg_camera = spawn_cameras(
            world=world,
            blueprint_library=blueprint_library,
            camera_tf=fixed_camera_tf,
            args=args
        )

        actors.extend([rgb_camera, seg_camera])

        rgb_camera.listen(lambda image: rgb_queue.put(image))
        seg_camera.listen(lambda image: seg_queue.put(image))

        for _ in range(20):
            world.tick()
            flush_queue(rgb_queue)
            flush_queue(seg_queue)

        K = build_camera_intrinsics(
            width=args.image_width,
            height=args.image_height,
            fov_degrees=args.fov
        )

        angles = list(range(0, 360, args.angle_step))
        debug_angles = set(range(0, 360, 30))

        for angle in angles:
            yaw = vehicle_yaw_for_angle(base_yaw, angle)

            vehicle_tf = carla.Transform(
                base_location,
                carla.Rotation(
                    pitch=0.0,
                    yaw=float(yaw),
                    roll=0.0
                )
            )

            vehicle.set_transform(vehicle_tf)

            flush_queue(rgb_queue)
            flush_queue(seg_queue)

            for _ in range(args.settle_ticks):
                world.tick()

            frame = world.tick()

            rgb_image = wait_for_frame(rgb_queue, frame)
            seg_image = wait_for_frame(seg_queue, frame)

            rgb = carla_rgb_to_array(rgb_image)
            tags = carla_semantic_to_tags(seg_image)

            bbox = compute_actor_2d_bbox(
                actor=vehicle,
                camera_actor=rgb_camera,
                K=K,
                image_width=args.image_width,
                image_height=args.image_height
            )

            if bbox is None:
                print(f"[RGBABank] Warning: angle {angle:03d}, bbox not visible. Skipping.")
                continue

            # Try semantic alpha first.
            alpha, body_mask, window_mask = make_alpha_and_window_mask(
                tags=tags,
                bbox=bbox,
                args=args
            )

            alpha_pixels = int(np.sum(alpha > 10))

            # If semantic mask fails, use old-style RGB GrabCut from projected bbox.
            if alpha_pixels < args.min_mask_pixels:
                print(
                    f"[RGBABank] angle {angle:03d}: semantic alpha too small "
                    f"({alpha_pixels}), using GrabCut fallback."
                )

                alpha = make_grabcut_alpha_from_bbox(
                    rgb=rgb,
                    bbox=bbox,
                    args=args
                )

                body_mask = alpha.copy()
                window_mask = np.zeros_like(alpha)

            alpha_pixels = int(np.sum(alpha > 10))

            if alpha_pixels < args.min_mask_pixels:
                print(f"[RGBABank] Warning: angle {angle:03d}, alpha still too small. Skipping.")
                continue

            rgb_treated = apply_window_treatment(
                rgb=rgb,
                window_mask=window_mask,
                args=args
            )

            try:
                crop_rgb, crop_alpha, crop_window, rgba, crop_box, anchor = crop_rgba_outputs(
                    rgb_treated=rgb_treated,
                    alpha=alpha,
                    window_mask=window_mask,
                    bbox=bbox,
                    args=args
                )
            except Exception as e:
                print(f"[RGBABank] Warning: angle {angle:03d}, crop failed: {e}")
                continue

            angle_name = f"angle_{angle:03d}"

            rgba_path = rgba_dir / f"{angle_name}_rgba.png"
            rgb_path = rgb_dir / f"{angle_name}.png"
            mask_path = mask_dir / f"{angle_name}_mask.png"
            window_path = window_dir / f"{angle_name}_window_mask.png"
            debug_path = debug_dir / f"{angle_name}_debug.png"

            save_rgba(rgba_path, rgba)
            save_rgb(rgb_path, crop_rgb)
            save_mask(mask_path, crop_alpha)
            save_mask(window_path, crop_window)

            if args.save_all_debug or angle in debug_angles:
                debug = draw_debug(
                    rgb=rgb_treated,
                    bbox=bbox,
                    crop_box=crop_box,
                    alpha=alpha,
                    window_mask=window_mask,
                    angle=angle,
                    yaw=yaw
                )
                save_rgb(debug_path, debug)
                contact_debug_paths.append(debug_path)

            if angle % 30 == 0:
                contact_rgba_paths.append(rgba_path)
                print(
                    f"[RGBABank] Saved angle {angle:03d} | "
                    f"rgba={rgba.shape[1]}x{rgba.shape[0]} | "
                    f"alpha_pixels={int(np.sum(crop_alpha > 10))} | "
                    f"window_pixels={int(np.sum(crop_window > 10))}"
                )

            metadata_rows.append({
                "angle": int(angle),
                "vehicle_yaw": float(yaw),
                "rgba": str(rgba_path.relative_to(output_dir)),
                "rgb": str(rgb_path.relative_to(output_dir)),
                "mask": str(mask_path.relative_to(output_dir)),
                "window_mask": str(window_path.relative_to(output_dir)),
                "debug": str(debug_path.relative_to(output_dir)),
                "bbox": bbox,
                "crop_box": crop_box,
                "anchor_point": anchor,
                "sprite_width": int(rgba.shape[1]),
                "sprite_height": int(rgba.shape[0]),
                "alpha_pixels": int(np.sum(crop_alpha > 10)),
                "window_pixels": int(np.sum(crop_window > 10))
            })

        metadata = {
            "schema": "he_fixed_camera_rotating_vehicle_rgba_v1",
            "world_map": world.get_map().name,
            "vehicle_name": args.vehicle_name,
            "vehicle_blueprint_requested": args.vehicle_blueprint,
            "vehicle_blueprint_actual": actual_bp,
            "vehicle_color": args.color,
            "angle_convention": {
                "0": "rear_view",
                "90": "side_view",
                "180": "front_view_same_as_old_front_sprite",
                "270": "opposite_side_view"
            },
            "window_treatment": {
                "mode": args.window_mode,
                "black_value": args.window_black_value,
                "darken_factor": args.window_darken_factor
            },
            "capture_settings": {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "fov": args.fov,
                "camera_distance": args.camera_distance,
                "camera_height": args.camera_height,
                "target_height": args.target_height,
                "angle_step": args.angle_step,
                "crop_margin_px": args.crop_margin_px,
                "bbox_expand_ratio": args.bbox_expand_ratio,
                "spawn_index": args.spawn_index,
                "base_yaw": base_yaw
            },
            "camera_transform": {
                "location": {
                    "x": float(fixed_camera_tf.location.x),
                    "y": float(fixed_camera_tf.location.y),
                    "z": float(fixed_camera_tf.location.z)
                },
                "rotation": {
                    "pitch": float(fixed_camera_tf.rotation.pitch),
                    "yaw": float(fixed_camera_tf.rotation.yaw),
                    "roll": float(fixed_camera_tf.rotation.roll)
                }
            },
            "angles": metadata_rows
        }

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(output_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "angle",
                "vehicle_yaw",
                "sprite_width",
                "sprite_height",
                "alpha_pixels",
                "window_pixels",
                "bbox",
                "crop_box"
            ])

            for row in metadata_rows:
                writer.writerow([
                    row["angle"],
                    f"{row['vehicle_yaw']:.3f}",
                    row["sprite_width"],
                    row["sprite_height"],
                    row["alpha_pixels"],
                    row["window_pixels"],
                    row["bbox"],
                    row["crop_box"]
                ])

        make_contact_sheet(
            contact_rgba_paths,
            output_dir / "contact_sheet_rgba_30deg.png",
            title="Fixed camera, rotating vehicle | RGBA with black windows"
        )

        make_contact_sheet(
            contact_debug_paths,
            output_dir / "contact_sheet_debug_30deg.png",
            title="RGBA debug | green=alpha red=window holes"
        )

        print("\n[RGBABank] Done.")
        print("[RGBABank] Output:", output_dir)
        print("[RGBABank] Check:")
        print("  contact_sheet_rgba_30deg.png")
        print("  contact_sheet_debug_30deg.png")

    except Exception:
        traceback.print_exc()

    finally:
        print("[RGBABank] Cleaning up...")

        for actor in actors:
            safe_destroy(actor)
        if unloaded_map_layers:

            print(
                "[RGBABank] Map layers left unloaded intentionally."
            )

            print(
                "[RGBABank] Restart CARLA after sprite generation "
                "to restore the normal map."
            )
        restore_world_settings(world, original_settings)

        print("[RGBABank] Done cleanup.")


if __name__ == "__main__":
    main()