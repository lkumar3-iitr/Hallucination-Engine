#!/usr/bin/env python3
"""
Generate 360-degree CARLA vehicle sprite bank using the same idea as the old
front-facing sprite generator:

- use current/default CARLA world
- spawn one vehicle at a valid spawn point
- keep vehicle fixed
- move RGB camera around vehicle every N degrees
- camera always looks at vehicle center
- project actor 3D bounding box to image
- use GrabCut initialized from projected bbox
- save RGBA sprite, RGB crop, mask, debug image, metadata, contact sheet

Angle convention:
    angle_000 = rear view of vehicle
    angle_180 = front view of vehicle

This means angle_180 should look similar to your old front-facing sprite.
"""

import argparse
import json
import math
import os
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
# Utilities
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

    parser.add_argument("--camera-distance", type=float, default=10.0)
    parser.add_argument("--camera-height", type=float, default=1.35)
    parser.add_argument("--target-height", type=float, default=1.0)

    parser.add_argument("--angle-step", type=int, default=1)
    parser.add_argument("--spawn-index", type=int, default=-1,
                        help="-1 means random valid spawn point.")

    parser.add_argument("--crop-margin-px", type=int, default=15)
    parser.add_argument("--min-mask-pixels", type=int, default=500)

    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--grabcut-iters", type=int, default=7)

    return parser.parse_args()


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


def flush_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def setup_synchronous_mode(world, fixed_delta_seconds):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)


def restore_world_settings(world, original_settings):
    if world is not None and original_settings is not None:
        world.apply_settings(original_settings)


def ensure_output_dirs(root, overwrite=False):
    root = Path(root)

    if root.exists() and overwrite:
        shutil.rmtree(root)

    rgba_dir = root / "rgba"
    rgb_dir = root / "rgb"
    mask_dir = root / "mask"
    debug_dir = root / "debug"

    rgba_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    return rgba_dir, rgb_dir, mask_dir, debug_dir


def carla_rgb_to_array(image):
    """
    CARLA raw RGB camera buffer is BGRA.
    Return RGB numpy array.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]
    return rgb.copy()


def get_latest_sensor_frame(world, q, ticks=2):
    flush_queue(q)

    latest = None

    for _ in range(ticks):
        world.tick()
        try:
            latest = q.get(timeout=3.0)
        except queue.Empty:
            print("[SpriteBank] Warning: no sensor frame received.")

    if latest is None:
        raise RuntimeError("No sensor frame received.")

    return latest


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
    CARLA camera coords:
        x = forward
        y = right
        z = up

    OpenCV coords:
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
    angle_180 = old front-facing camera concept:
        camera in front of vehicle, looking at vehicle.

    angle_000:
        camera behind vehicle, looking at rear.
    """

    vehicle_transform = vehicle.get_transform()
    vehicle_loc = vehicle_transform.location
    vehicle_yaw = vehicle_transform.rotation.yaw

    # In the old front-facing script:
    #   camera = vehicle_loc + forward * distance
    # That should correspond to angle_180.
    relative_deg = angle_deg - 180.0
    theta = np.deg2rad(vehicle_yaw + relative_deg)

    dir_x = np.cos(theta)
    dir_y = np.sin(theta)

    camera_location = carla.Location(
        x=vehicle_loc.x + camera_distance * dir_x,
        y=vehicle_loc.y + camera_distance * dir_y,
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

def spawn_vehicle(world, blueprint_library, args):
    vehicle_bp = blueprint_library.find(args.vehicle_blueprint)

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", args.color)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "he_sprite_target")

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points found.")

    if args.spawn_index >= 0:
        spawn_points = [spawn_points[min(args.spawn_index, len(spawn_points) - 1)]]
    else:
        random.shuffle(spawn_points)

    for sp in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)

        if vehicle is not None:
            vehicle.set_autopilot(False)
            vehicle.set_simulate_physics(False)

            world.tick()

            print("[SpriteBank] Spawned vehicle:", args.vehicle_blueprint)
            print("[SpriteBank] Vehicle location:", vehicle.get_location())
            print("[SpriteBank] Vehicle yaw:", vehicle.get_transform().rotation.yaw)

            return vehicle

    raise RuntimeError("Could not spawn vehicle.")


def spawn_rgb_camera(world, blueprint_library, camera_transform, args):
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(args.image_width))
    rgb_bp.set_attribute("image_size_y", str(args.image_height))
    rgb_bp.set_attribute("fov", str(args.fov))
    rgb_bp.set_attribute("sensor_tick", "0.0")

    rgb_camera = world.spawn_actor(rgb_bp, camera_transform)

    return rgb_camera


# ============================================================
# GrabCut mask
# ============================================================

def clean_mask(mask):
    mask = mask.astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return mask

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for label in range(2, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area > largest_area:
            largest_area = area
            largest_label = label

    cleaned = (labels == largest_label).astype(np.uint8) * 255
    cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)

    return cleaned


def make_mask_from_bbox_grabcut(rgb, bbox_2d, grabcut_iters):
    image_h, image_w = rgb.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in bbox_2d]

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 5 or box_h <= 5:
        raise RuntimeError("BBox too small for GrabCut.")

    margin_x = int(box_w * 0.06)
    margin_y = int(box_h * 0.06)

    x1e = max(0, x1 - margin_x)
    y1e = max(0, y1 - margin_y)
    x2e = min(image_w - 1, x2 + margin_x)
    y2e = min(image_h - 1, y2 + margin_y)

    crop_rgb = rgb[y1e:y2e + 1, x1e:x2e + 1].copy()

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

    # Avoid full road strip leakage at bottom.
    bottom_cut = int(crop_h * 0.98)
    mask_fg[bottom_cut:, :] = 0

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_fg)

    if num_labels > 1:
        crop_cx = crop_w / 2.0
        crop_cy = crop_h / 2.0

        best_label = None
        best_score = -1e9

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            if area < 300:
                continue

            cx, cy = centroids[label]
            dist = abs(cx - crop_cx) + 0.5 * abs(cy - crop_cy)
            score = area - 4.0 * dist

            if score > best_score:
                best_score = score
                best_label = label

        if best_label is not None:
            mask_fg = (labels == best_label).astype(np.uint8) * 255

    mask_fg = cv2.GaussianBlur(mask_fg, (3, 3), 0)

    full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    full_mask[y1e:y2e + 1, x1e:x2e + 1] = mask_fg

    return full_mask


# ============================================================
# Crop/save
# ============================================================

def crop_outputs(rgb, mask, crop_margin_px):
    ys, xs = np.where(mask > 10)

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Mask is empty. Could not crop sprite.")

    h, w = mask.shape[:2]

    x1 = max(0, int(xs.min()) - crop_margin_px)
    y1 = max(0, int(ys.min()) - crop_margin_px)
    x2 = min(w - 1, int(xs.max()) + crop_margin_px)
    y2 = min(h - 1, int(ys.max()) + crop_margin_px)

    crop_rgb = rgb[y1:y2 + 1, x1:x2 + 1].copy()
    crop_alpha = mask[y1:y2 + 1, x1:x2 + 1].copy()

    rgba = np.dstack([crop_rgb, crop_alpha])

    crop_ys, crop_xs = np.where(crop_alpha > 10)

    anchor_x = float((crop_xs.min() + crop_xs.max()) / 2.0)
    anchor_y = float(crop_ys.max())

    return crop_rgb, crop_alpha, rgba, [x1, y1, x2, y2], {"x": anchor_x, "y": anchor_y}


def save_rgba_png(path, rgba):
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), bgra)


def save_rgb_png(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_mask_png(path, mask):
    cv2.imwrite(str(path), mask)


def save_debug_image(path, rgb, mask, crop_box, bbox_2d=None, angle_deg=None):
    debug = rgb.copy()

    if bbox_2d is not None:
        bx1, by1, bx2, by2 = bbox_2d
        cv2.rectangle(debug, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

    x1, y1, x2, y2 = crop_box
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)

    mask_color = np.zeros_like(debug)
    mask_color[:, :, 1] = mask

    debug = cv2.addWeighted(debug, 0.75, mask_color, 0.25, 0)

    if angle_deg is not None:
        cv2.putText(
            debug,
            f"angle={angle_deg:03d}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
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

    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
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

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        world = client.get_world()
        original_settings = world.get_settings()

        print("[SpriteBank] Using current/default CARLA world. No map will be changed.")
        print("[SpriteBank] Current map:", world.get_map().name)

        setup_synchronous_mode(world, args.fixed_delta_seconds)
        blueprint_library = world.get_blueprint_library()

        vehicle = spawn_vehicle(world, blueprint_library, args)
        actors.append(vehicle)

        K = build_camera_intrinsics(
            width=args.image_width,
            height=args.image_height,
            fov_degrees=args.fov
        )

        initial_tf = camera_transform_for_angle(
            vehicle=vehicle,
            angle_deg=180,
            camera_distance=args.camera_distance,
            camera_height=args.camera_height,
            target_height=args.target_height
        )

        rgb_camera = spawn_rgb_camera(world, blueprint_library, initial_tf, args)
        actors.append(rgb_camera)

        rgb_camera.listen(lambda image: rgb_queue.put(image))

        # Warm-up
        for _ in range(20):
            world.tick()
            flush_queue(rgb_queue)

        vehicle_bb = vehicle.bounding_box.extent

        metadata = {
            "schema": "he_sprite_bank_grabcut_v1",
            "world_map": world.get_map().name,
            "vehicle_name": args.vehicle_name,
            "vehicle_blueprint": args.vehicle_blueprint,
            "vehicle_color": args.color,
            "angle_step_deg": args.angle_step,
            "angle_convention": {
                "0": "rear_view",
                "180": "front_view",
                "note": "angle_180 is generated using the same front-camera idea as the old one-sprite script."
            },
            "capture_settings": {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "fov_deg": args.fov,
                "camera_distance_m": args.camera_distance,
                "camera_height_m": args.camera_height,
                "target_height_m": args.target_height,
                "crop_margin_px": args.crop_margin_px,
                "grabcut_iters": args.grabcut_iters,
                "spawn_index": args.spawn_index,
                "fixed_delta_seconds": args.fixed_delta_seconds
            },
            "vehicle_dimensions": {
                "length_m": float(2.0 * vehicle_bb.x),
                "width_m": float(2.0 * vehicle_bb.y),
                "height_m": float(2.0 * vehicle_bb.z)
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

            # Let camera transform settle, then capture
            world.tick()
            rgb_image = get_latest_sensor_frame(world, rgb_queue, ticks=2)
            rgb = carla_rgb_to_array(rgb_image)

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

            try:
                mask = make_mask_from_bbox_grabcut(
                    rgb=rgb,
                    bbox_2d=bbox_2d,
                    grabcut_iters=args.grabcut_iters
                )

                if int(np.sum(mask > 0)) < args.min_mask_pixels:
                    print(f"[SpriteBank] Warning: angle {angle:03d}, mask too small. Skipping.")
                    continue

                mask = clean_mask(mask)

                crop_rgb, crop_mask, rgba, crop_box, anchor = crop_outputs(
                    rgb=rgb,
                    mask=mask,
                    crop_margin_px=args.crop_margin_px
                )

            except Exception as e:
                print(f"[SpriteBank] Warning: angle {angle:03d}, extraction failed: {e}")
                continue

            angle_name = f"angle_{angle:03d}"

            rgba_path = rgba_dir / f"{angle_name}_rgba.png"
            rgb_path = rgb_dir / f"{angle_name}.png"
            mask_path = mask_dir / f"{angle_name}_mask.png"

            save_rgba_png(rgba_path, rgba)
            save_rgb_png(rgb_path, crop_rgb)
            save_mask_png(mask_path, crop_mask)

            if angle in debug_angles:
                debug_path = debug_dir / f"{angle_name}_debug.png"
                save_debug_image(
                    debug_path,
                    rgb=rgb,
                    mask=mask,
                    crop_box=crop_box,
                    bbox_2d=bbox_2d,
                    angle_deg=angle
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
                "mask_pixels": int(np.sum(crop_mask > 0)),
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
                    f"mask_pixels={int(np.sum(crop_mask > 0))}"
                )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        contact_sheet_path = output_dir / "contact_sheet.png"
        make_contact_sheet(rgba_dir, contact_sheet_path)

        print("\n[SpriteBank] Done.")
        print("[SpriteBank] Output:", output_dir)
        print("[SpriteBank] Contact sheet:", contact_sheet_path)
        print("[SpriteBank] Metadata:", metadata_path)
        print("[SpriteBank] Debug folder:", debug_dir)

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