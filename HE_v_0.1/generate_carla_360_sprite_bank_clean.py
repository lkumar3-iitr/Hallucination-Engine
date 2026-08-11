#!/usr/bin/env python3
"""
generate_carla_360_sprite_bank_clean.py

Generate a 360-degree vehicle sprite bank in CARLA.

Design:
  - Does NOT load/change map.
  - Uses current/default CARLA world.
  - Clears existing vehicles/walkers/sensors by default.
  - Spawns one target vehicle.
  - Keeps vehicle fixed.
  - Moves RGB + semantic camera around vehicle on a perfect circle.
  - Camera always looks at same vehicle center.
  - Uses projected 3D bbox to isolate the target region.
  - Uses semantic segmentation mask inside projected bbox.
  - Falls back to GrabCut if semantic mask fails.
  - Saves RGB crop, alpha mask, RGBA sprite, debug images, metadata.

Angle convention:
  angle_000 = rear view of vehicle
  angle_090 = side view
  angle_180 = front view of vehicle
  angle_270 = opposite side view

Example 30-degree test:
  python generate_carla_360_sprite_bank_clean.py ^
    --vehicle-blueprint vehicle.tesla.model3 ^
    --vehicle-name vehicle_blue_sedan_test ^
    --color 0,0,255 ^
    --angle-step 30 ^
    --output-root assets\\sprite_bank ^
    --overwrite

Final 1-degree:
  python generate_carla_360_sprite_bank_clean.py ^
    --vehicle-blueprint vehicle.tesla.model3 ^
    --vehicle-name vehicle_blue_sedan ^
    --color 0,0,255 ^
    --angle-step 1 ^
    --output-root assets\\sprite_bank ^
    --overwrite
"""

import argparse
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
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)

    parser.add_argument("--vehicle-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--vehicle-name", default="vehicle_blue_sedan")
    parser.add_argument("--color", default="0,0,255")

    parser.add_argument("--output-root", default="assets/sprite_bank")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=55.0)

    parser.add_argument("--camera-distance", type=float, default=11.5)
    parser.add_argument("--camera-height", type=float, default=1.35)
    parser.add_argument("--target-height", type=float, default=1.0)

    parser.add_argument("--angle-step", type=int, default=1)

    parser.add_argument("--spawn-index", type=int, default=-1,
                        help="-1 means random valid spawn point.")

    parser.add_argument("--crop-margin-px", type=int, default=25)
    parser.add_argument("--bbox-expand-ratio", type=float, default=0.10)
    parser.add_argument("--min-mask-pixels", type=int, default=500)

    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--settle-ticks", type=int, default=2)
    parser.add_argument("--grabcut-iters", type=int, default=7)

    parser.add_argument("--vehicle-semantic-tag", type=int, default=10,
                        help="CARLA semantic tag for vehicles is usually 10.")

    parser.add_argument("--no-clear-existing-actors", action="store_true",
                        help="Do not destroy existing vehicles/walkers/sensors.")

    parser.add_argument("--save-all-debug", action="store_true")

    parser.add_argument("--capture-z-offset", type=float, default=0.0,
                        help="Keep 0 for road capture. Use high value only for isolated/sky capture.")

    return parser.parse_args()


# ============================================================
# General utilities
# ============================================================

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
    """
    Clear vehicles, walkers, and old sensors from the current world.
    This avoids another car entering the projected bbox or GrabCut region.
    """
    patterns = [
        "vehicle.*",
        "walker.*",
        "sensor.*",
    ]

    actors_to_destroy = []

    for pattern in patterns:
        actors_to_destroy.extend(list(world.get_actors().filter(pattern)))

    print(f"[SpriteBank] Clearing {len(actors_to_destroy)} existing dynamic actors...")

    for actor in actors_to_destroy:
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
    """
    Wait until sensor gives frame >= expected_frame.
    """
    while True:
        data = q.get(timeout=timeout)

        if data.frame >= expected_frame:
            return data


def ensure_output_dirs(output_dir, overwrite):
    output_dir = Path(output_dir)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    rgba_dir = output_dir / "rgba"
    rgb_dir = output_dir / "rgb"
    mask_dir = output_dir / "mask"
    debug_dir = output_dir / "debug"

    rgba_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    return rgba_dir, rgb_dir, mask_dir, debug_dir


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
    CARLA semantic segmentation raw labels are normally in red channel.
    Raw buffer is BGRA, so red channel is index 2.
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
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return K


def project_carla_world_to_image(world_location, camera_actor, K):
    """
    CARLA camera coordinates:
        x = forward
        y = right
        z = up

    OpenCV camera coordinates:
        x = right
        y = down
        z = forward

    Conversion:
        [x_cv, y_cv, z_cv] = [y_carla, -z_carla, x_carla]
    """
    camera_transform = camera_actor.get_transform()
    world_2_camera = np.array(camera_transform.get_inverse_matrix())

    point_world = np.array([
        world_location.x,
        world_location.y,
        world_location.z,
        1.0
    ])

    point_camera = world_2_camera @ point_world

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
    """
    Project CARLA actor 3D bounding box into image.
    """
    bbox = actor.bounding_box
    actor_transform = actor.get_transform()

    vertices = bbox.get_world_vertices(actor_transform)

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


def camera_transform_for_angle(vehicle, angle_deg, camera_distance, camera_height, target_height):
    """
    Perfect circular camera orbit around fixed vehicle.

    angle_180 = camera in front of vehicle, same concept as old front sprite.
    angle_000 = camera behind vehicle.
    """
    vehicle_tf = vehicle.get_transform()
    vehicle_loc = vehicle_tf.location
    vehicle_yaw = vehicle_tf.rotation.yaw

    # Same convention as previous working version:
    # angle_180 = camera along vehicle forward direction.
    relative_deg = angle_deg - 180.0
    theta = np.deg2rad(vehicle_yaw + relative_deg)

    camera_location = carla.Location(
        x=vehicle_loc.x + camera_distance * np.cos(theta),
        y=vehicle_loc.y + camera_distance * np.sin(theta),
        z=vehicle_loc.z + camera_height,
    )

    target_location = carla.Location(
        x=vehicle_loc.x,
        y=vehicle_loc.y,
        z=vehicle_loc.z + target_height,
    )

    camera_rotation = look_at_rotation(camera_location, target_location)

    return carla.Transform(camera_location, camera_rotation)


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

        print(f"[SpriteBank] Warning: could not find {blueprint_id}, falling back to vehicle.*")
        all_vehicles = blueprint_library.filter("vehicle.*")

        if len(all_vehicles) == 0:
            raise RuntimeError("No vehicle blueprints found.")

        return all_vehicles[0]


def spawn_target_vehicle(world, blueprint_library, args):
    vehicle_bp = find_vehicle_blueprint(blueprint_library, args.vehicle_blueprint)

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", args.color)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "he_sprite_target")

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points found in current map.")

    if args.spawn_index >= 0:
        selected = spawn_points[min(args.spawn_index, len(spawn_points) - 1)]
        candidate_spawn_points = [selected]
    else:
        candidate_spawn_points = list(spawn_points)
        random.shuffle(candidate_spawn_points)

    for sp in candidate_spawn_points:
        loc = carla.Location(
            x=sp.location.x,
            y=sp.location.y,
            z=sp.location.z + args.capture_z_offset
        )

        tf = carla.Transform(
            loc,
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

            print("[SpriteBank] Spawned target vehicle:", vehicle_bp.id)
            print("[SpriteBank] Vehicle location:", vehicle.get_location())
            print("[SpriteBank] Vehicle yaw:", vehicle.get_transform().rotation.yaw)

            return vehicle, vehicle_bp.id

    raise RuntimeError("Could not spawn target vehicle. Try different --spawn-index.")


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

    return rgb_camera, seg_camera


# ============================================================
# Mask extraction
# ============================================================

def expand_bbox(bbox, image_width, image_height, expand_ratio):
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    mx = int(box_w * expand_ratio)
    my = int(box_h * expand_ratio)

    x1e = max(0, x1 - mx)
    y1e = max(0, y1 - my)
    x2e = min(image_width - 1, x2 + mx)
    y2e = min(image_height - 1, y2 + my)

    return [x1e, y1e, x2e, y2e]


def keep_largest_component(mask_u8):
    """
    Input: uint8 mask, 0 or 255.
    Output: uint8 mask, 0 or 255.
    """
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

    cleaned = (labels == largest_label).astype(np.uint8) * 255

    return cleaned


def fill_holes(mask_u8):
    """
    Fill holes inside the vehicle silhouette.
    This helps keep windows/glass inside the alpha region.
    """
    binary = (mask_u8 > 0).astype(np.uint8) * 255

    h, w = binary.shape[:2]
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = binary | flood_inv

    return filled


def clean_alpha_mask(mask_u8):
    """
    Clean silhouette mask.
    """
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255

    kernel = np.ones((5, 5), dtype=np.uint8)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    mask_u8 = keep_largest_component(mask_u8)
    mask_u8 = fill_holes(mask_u8)

    # Slight edge feathering.
    mask_u8 = cv2.GaussianBlur(mask_u8, (3, 3), 0)

    return mask_u8


def make_semantic_mask_inside_bbox(tags, bbox, vehicle_tag, image_width, image_height, expand_ratio):
    """
    Use CARLA semantic segmentation, but only inside projected actor bbox.
    This avoids picking other vehicles elsewhere in the scene.
    """
    x1, y1, x2, y2 = expand_bbox(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        expand_ratio=expand_ratio
    )

    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    roi_tags = tags[y1:y2 + 1, x1:x2 + 1]
    roi_mask = (roi_tags == vehicle_tag).astype(np.uint8) * 255

    full_mask[y1:y2 + 1, x1:x2 + 1] = roi_mask

    full_mask = clean_alpha_mask(full_mask)

    return full_mask


def make_grabcut_mask_from_bbox(rgb, bbox, bbox_expand_ratio, grabcut_iters):
    """
    Fallback mask extraction using GrabCut.
    """
    image_h, image_w = rgb.shape[:2]

    x1, y1, x2, y2 = expand_bbox(
        bbox=bbox,
        image_width=image_w,
        image_height=image_h,
        expand_ratio=bbox_expand_ratio
    )

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 5 or box_h <= 5:
        raise RuntimeError("BBox too small for GrabCut.")

    crop_rgb = rgb[y1:y2 + 1, x1:x2 + 1].copy()

    if crop_rgb.size == 0:
        raise RuntimeError("Empty crop for GrabCut.")

    crop_h, crop_w = crop_rgb.shape[:2]

    mask = np.full((crop_h, crop_w), cv2.GC_PR_BGD, dtype=np.uint8)

    fg_margin_x = max(3, int(crop_w * 0.12))
    fg_margin_y = max(3, int(crop_h * 0.10))

    mask[
        fg_margin_y:crop_h - fg_margin_y,
        fg_margin_x:crop_w - fg_margin_x
    ] = cv2.GC_PR_FGD

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
        grabcut_iters,
        cv2.GC_INIT_WITH_MASK
    )

    mask_fg = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    full_mask[y1:y2 + 1, x1:x2 + 1] = mask_fg

    full_mask = clean_alpha_mask(full_mask)

    return full_mask


# ============================================================
# Crop/save
# ============================================================

def crop_outputs(rgb, mask_u8, crop_margin_px):
    ys, xs = np.where(mask_u8 > 10)

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Mask is empty. Could not crop sprite.")

    image_h, image_w = mask_u8.shape[:2]

    x1 = max(0, int(xs.min()) - crop_margin_px)
    y1 = max(0, int(ys.min()) - crop_margin_px)
    x2 = min(image_w - 1, int(xs.max()) + crop_margin_px)
    y2 = min(image_h - 1, int(ys.max()) + crop_margin_px)

    crop_rgb = rgb[y1:y2 + 1, x1:x2 + 1].copy()
    crop_alpha = mask_u8[y1:y2 + 1, x1:x2 + 1].copy()

    rgba = np.dstack([crop_rgb, crop_alpha])

    crop_ys, crop_xs = np.where(crop_alpha > 10)

    anchor_x = float((crop_xs.min() + crop_xs.max()) / 2.0)
    anchor_y = float(crop_ys.max())

    crop_box = [x1, y1, x2, y2]

    return crop_rgb, crop_alpha, rgba, crop_box, {"x": anchor_x, "y": anchor_y}


def save_rgba_png(path, rgba):
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), bgra)


def save_rgb_png(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_mask_png(path, mask):
    cv2.imwrite(str(path), mask)


def save_debug_image(path, rgb, mask, crop_box, bbox_2d=None, angle_deg=None, method=None):
    debug = rgb.copy()

    if bbox_2d is not None:
        bx1, by1, bx2, by2 = bbox_2d
        cv2.rectangle(debug, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

    x1, y1, x2, y2 = crop_box
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)

    mask_color = np.zeros_like(debug)
    mask_color[:, :, 1] = mask

    debug = cv2.addWeighted(debug, 0.75, mask_color, 0.25, 0)

    label = ""

    if angle_deg is not None:
        label += f"angle={angle_deg:03d}"

    if method is not None:
        label += f" method={method}"

    if label:
        cv2.putText(
            debug,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    debug_bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), debug_bgr)


def make_contact_sheet(rgba_dir, output_path):
    selected_angles = list(range(0, 360, 30))

    thumb_w = 280
    thumb_h = 190
    label_h = 26

    cols = 4
    rows = math.ceil(len(selected_angles) / cols)

    sheet = Image.new(
        "RGB",
        (cols * thumb_w, rows * (thumb_h + label_h)),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for idx, angle in enumerate(selected_angles):
        rgba_path = rgba_dir / f"angle_{angle:03d}_rgba.png"

        if not rgba_path.exists():
            continue

        sprite = Image.open(rgba_path).convert("RGBA")
        white_bg = Image.new("RGBA", sprite.size, "white")
        white_bg.alpha_composite(sprite)

        thumb = ImageOps.contain(white_bg.convert("RGB"), (thumb_w, thumb_h))

        col = idx % cols
        row = idx // cols

        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)

        paste_x = x0 + (thumb_w - thumb.width) // 2
        paste_y = y0 + (thumb_h - thumb.height) // 2

        sheet.paste(thumb, (paste_x, paste_y))
        draw.text((x0 + 8, y0 + thumb_h + 5), f"{angle:03d} deg", fill="black")

    sheet.save(output_path)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_root) / args.vehicle_name
    rgba_dir, rgb_dir, mask_dir, debug_dir = ensure_output_dirs(
        output_dir,
        overwrite=args.overwrite
    )

    client = None
    world = None
    original_settings = None
    actors = []

    rgb_queue = queue.Queue()
    seg_queue = queue.Queue()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        world = client.get_world()
        original_settings = world.get_settings()

        print("[SpriteBank] Using current/default CARLA world.")
        print("[SpriteBank] No map/town will be loaded or changed.")
        print("[SpriteBank] Current map:", world.get_map().name)

        setup_synchronous_mode(world, args.fixed_delta_seconds)

        if not args.no_clear_existing_actors:
            clear_existing_dynamic_actors(world)

        blueprint_library = world.get_blueprint_library()

        vehicle, actual_vehicle_bp = spawn_target_vehicle(world, blueprint_library, args)
        actors.append(vehicle)

        vehicle_bb = vehicle.bounding_box.extent

        K = build_camera_intrinsics(
            width=args.image_width,
            height=args.image_height,
            fov_degrees=args.fov
        )

        initial_camera_tf = camera_transform_for_angle(
            vehicle=vehicle,
            angle_deg=180,
            camera_distance=args.camera_distance,
            camera_height=args.camera_height,
            target_height=args.target_height
        )

        rgb_camera, seg_camera = spawn_cameras(
            world=world,
            blueprint_library=blueprint_library,
            camera_tf=initial_camera_tf,
            args=args
        )

        actors.extend([rgb_camera, seg_camera])

        rgb_camera.listen(lambda image: rgb_queue.put(image))
        seg_camera.listen(lambda image: seg_queue.put(image))

        # Warm-up sensors.
        for _ in range(20):
            world.tick()
            flush_queue(rgb_queue)
            flush_queue(seg_queue)

        metadata = {
            "schema": "he_sprite_bank_clean_v1",
            "world_map": world.get_map().name,
            "vehicle_name": args.vehicle_name,
            "vehicle_blueprint_requested": args.vehicle_blueprint,
            "vehicle_blueprint_actual": actual_vehicle_bp,
            "vehicle_color": args.color,
            "angle_step_deg": args.angle_step,
            "angle_convention": {
                "0": "rear_view",
                "90": "side_view",
                "180": "front_view",
                "270": "opposite_side_view"
            },
            "capture_settings": {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "fov_deg": args.fov,
                "camera_distance_m": args.camera_distance,
                "camera_height_m": args.camera_height,
                "target_height_m": args.target_height,
                "crop_margin_px": args.crop_margin_px,
                "bbox_expand_ratio": args.bbox_expand_ratio,
                "spawn_index": args.spawn_index,
                "fixed_delta_seconds": args.fixed_delta_seconds,
                "settle_ticks": args.settle_ticks,
                "capture_z_offset": args.capture_z_offset,
                "clear_existing_actors": not args.no_clear_existing_actors
            },
            "vehicle_dimensions": {
                "length_m": float(2.0 * vehicle_bb.x),
                "width_m": float(2.0 * vehicle_bb.y),
                "height_m": float(2.0 * vehicle_bb.z)
            },
            "paths": {
                "rgba": "rgba",
                "rgb": "rgb",
                "mask": "mask",
                "debug": "debug"
            },
            "angles": {}
        }

        debug_angles = set(range(0, 360, 30))

        for angle in range(0, 360, args.angle_step):
            camera_tf = camera_transform_for_angle(
                vehicle=vehicle,
                angle_deg=angle,
                camera_distance=args.camera_distance,
                camera_height=args.camera_height,
                target_height=args.target_height
            )

            rgb_camera.set_transform(camera_tf)
            seg_camera.set_transform(camera_tf)

            flush_queue(rgb_queue)
            flush_queue(seg_queue)

            for _ in range(args.settle_ticks):
                world.tick()

            frame = world.tick()

            rgb_image = wait_for_frame(rgb_queue, frame)
            seg_image = wait_for_frame(seg_queue, frame)

            rgb = carla_rgb_to_array(rgb_image)
            tags = carla_semantic_to_tags(seg_image)

            bbox_2d = compute_actor_2d_bbox(
                actor=vehicle,
                camera_actor=rgb_camera,
                K=K,
                image_width=args.image_width,
                image_height=args.image_height
            )

            if bbox_2d is None:
                print(f"[SpriteBank] Warning: angle {angle:03d}, bbox not visible. Skipping.")
                continue

            method_used = "semantic_bbox"

            mask = make_semantic_mask_inside_bbox(
                tags=tags,
                bbox=bbox_2d,
                vehicle_tag=args.vehicle_semantic_tag,
                image_width=args.image_width,
                image_height=args.image_height,
                expand_ratio=args.bbox_expand_ratio
            )

            mask_pixels = int(np.sum(mask > 10))

            if mask_pixels < args.min_mask_pixels:
                method_used = "grabcut_fallback"

                try:
                    mask = make_grabcut_mask_from_bbox(
                        rgb=rgb,
                        bbox=bbox_2d,
                        bbox_expand_ratio=args.bbox_expand_ratio,
                        grabcut_iters=args.grabcut_iters
                    )
                    mask_pixels = int(np.sum(mask > 10))
                except Exception as e:
                    print(f"[SpriteBank] Warning: angle {angle:03d}, GrabCut failed: {e}")
                    continue

            if mask_pixels < args.min_mask_pixels:
                print(
                    f"[SpriteBank] Warning: angle {angle:03d}, "
                    f"mask too small after {method_used}. pixels={mask_pixels}. Skipping."
                )
                continue

            try:
                crop_rgb, crop_mask, rgba, crop_box, anchor = crop_outputs(
                    rgb=rgb,
                    mask_u8=mask,
                    crop_margin_px=args.crop_margin_px
                )
            except Exception as e:
                print(f"[SpriteBank] Warning: angle {angle:03d}, crop failed: {e}")
                continue

            angle_name = f"angle_{angle:03d}"

            rgba_path = rgba_dir / f"{angle_name}_rgba.png"
            rgb_path = rgb_dir / f"{angle_name}.png"
            mask_path = mask_dir / f"{angle_name}_mask.png"

            save_rgba_png(rgba_path, rgba)
            save_rgb_png(rgb_path, crop_rgb)
            save_mask_png(mask_path, crop_mask)

            if args.save_all_debug or angle in debug_angles:
                debug_path = debug_dir / f"{angle_name}_debug.png"
                save_debug_image(
                    path=debug_path,
                    rgb=rgb,
                    mask=mask,
                    crop_box=crop_box,
                    bbox_2d=bbox_2d,
                    angle_deg=angle,
                    method=method_used
                )

            metadata["angles"][f"{angle:03d}"] = {
                "angle_deg": int(angle),
                "rgba": str(rgba_path.relative_to(output_dir)),
                "rgb": str(rgb_path.relative_to(output_dir)),
                "mask": str(mask_path.relative_to(output_dir)),
                "bbox_2d": bbox_2d,
                "crop_box": crop_box,
                "anchor_point": anchor,
                "sprite_width": int(rgba.shape[1]),
                "sprite_height": int(rgba.shape[0]),
                "mask_pixels": int(mask_pixels),
                "method_used": method_used,
                "camera_transform": {
                    "location": {
                        "x": float(camera_tf.location.x),
                        "y": float(camera_tf.location.y),
                        "z": float(camera_tf.location.z)
                    },
                    "rotation": {
                        "pitch": float(camera_tf.rotation.pitch),
                        "yaw": float(camera_tf.rotation.yaw),
                        "roll": float(camera_tf.rotation.roll)
                    }
                }
            }

            if angle % 30 == 0:
                print(
                    f"[SpriteBank] Saved angle {angle:03d} | "
                    f"sprite={rgba.shape[1]}x{rgba.shape[0]} | "
                    f"mask_pixels={mask_pixels} | "
                    f"method={method_used}"
                )

        metadata_path = output_dir / "metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        contact_sheet_path = output_dir / "contact_sheet.png"
        make_contact_sheet(rgba_dir, contact_sheet_path)

        saved_count = len(metadata["angles"])

        print("\n[SpriteBank] Done.")
        print("[SpriteBank] Saved angles:", saved_count)
        print("[SpriteBank] Output:", output_dir)
        print("[SpriteBank] Contact sheet:", contact_sheet_path)
        print("[SpriteBank] Metadata:", metadata_path)
        print("[SpriteBank] Debug folder:", debug_dir)

        if saved_count < int(360 / args.angle_step):
            print("[SpriteBank] Warning: some angles were skipped. Check debug folder/log.")

    except Exception:
        traceback.print_exc()

    finally:
        print("[SpriteBank] Cleaning up...")

        for actor in actors:
            safe_destroy(actor)

        restore_world_settings(world, original_settings)

        print("[SpriteBank] Done cleanup.")


if __name__ == "__main__":
    main()