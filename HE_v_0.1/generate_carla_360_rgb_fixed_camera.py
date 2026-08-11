#!/usr/bin/env python3
"""
generate_carla_360_rgb_fixed_camera.py

Generate 360-degree RGB sprite views in CARLA.

Main idea:
  - Do NOT load/change map.
  - Use current/default CARLA world.
  - Spawn one vehicle.
  - Spawn one fixed RGB camera in front of the vehicle.
  - Keep camera fixed.
  - Rotate only the vehicle.
  - Save full RGB, bbox crop RGB, debug RGB, metadata, contact sheets.

Angle convention:
  angle_180 = front view, same as old front-facing sprite.
  angle_000 = rear view.
  angle_090 / 270 = side views.

First test:
  python generate_carla_360_rgb_fixed_camera.py ^
    --vehicle-blueprint vehicle.tesla.model3 ^
    --vehicle-name vehicle_blue_sedan_test ^
    --angle-step 30 ^
    --output-root assets\\sprite_bank_rgb ^
    --overwrite

Final RGB bank:
  python generate_carla_360_rgb_fixed_camera.py ^
    --vehicle-blueprint vehicle.tesla.model3 ^
    --vehicle-name vehicle_blue_sedan ^
    --angle-step 1 ^
    --output-root assets\\sprite_bank_rgb ^
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

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)

    parser.add_argument("--vehicle-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--vehicle-name", default="vehicle_blue_sedan")
    parser.add_argument("--color", default="0,0,255")

    parser.add_argument("--output-root", default="assets/sprite_bank_rgb")
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

    parser.add_argument("--crop-margin-px", type=int, default=35)
    parser.add_argument("--bbox-expand-ratio", type=float, default=0.12)

    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--settle-ticks", type=int, default=2)

    parser.add_argument(
        "--no-clear-existing-actors",
        action="store_true",
        help="Do not clear existing vehicles/walkers/sensors."
    )

    parser.add_argument(
        "--save-all-debug",
        action="store_true",
        help="Save debug image for every angle. Otherwise saves every 30 degrees."
    )

    return parser.parse_args()


# ============================================================
# Utilities
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
    patterns = ["vehicle.*", "walker.*", "sensor.*"]
    actors = []

    for pattern in patterns:
        actors.extend(list(world.get_actors().filter(pattern)))

    print(f"[RGBBank] Clearing {len(actors)} existing vehicles/walkers/sensors...")

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


def carla_rgb_to_array(image):
    """
    CARLA raw RGB camera buffer is BGRA.
    Return RGB numpy array.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]
    return rgb.copy()


def ensure_dirs(output_dir, overwrite):
    output_dir = Path(output_dir)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    full_dir = output_dir / "rgb_full"
    crop_dir = output_dir / "rgb_crop"
    debug_dir = output_dir / "debug"

    full_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    return full_dir, crop_dir, debug_dir


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
    camera_tf = camera_actor.get_transform()
    world_2_camera = np.array(camera_tf.get_inverse_matrix())

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
    """
    Fixed front-camera placement.

    This matches the old front-facing sprite idea:
      camera = vehicle location + vehicle forward vector * distance
      camera looks at vehicle center.
    """
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
    Camera is fixed in front of the base vehicle.

    angle_180 should be front view:
      vehicle yaw = base_yaw

    angle_000 should be rear view:
      vehicle yaw = base_yaw + 180

    Therefore:
      yaw = base_yaw + (180 - angle)
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

        print("[RGBBank] Warning: requested vehicle not found. Using:", vehicles[0].id)
        return vehicles[0]


def spawn_vehicle(world, blueprint_library, args):
    vehicle_bp = find_vehicle_blueprint(blueprint_library, args.vehicle_blueprint)

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", args.color)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "he_fixed_camera_sprite_target")

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

            print("[RGBBank] Spawned vehicle:", vehicle_bp.id)
            print("[RGBBank] Location:", vehicle.get_location())
            print("[RGBBank] Base yaw:", vehicle.get_transform().rotation.yaw)

            return vehicle, vehicle_bp.id

    raise RuntimeError("Could not spawn target vehicle. Try a different --spawn-index.")


def spawn_rgb_camera(world, blueprint_library, camera_tf, args):
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(args.image_width))
    rgb_bp.set_attribute("image_size_y", str(args.image_height))
    rgb_bp.set_attribute("fov", str(args.fov))
    rgb_bp.set_attribute("sensor_tick", "0.0")

    camera = world.spawn_actor(rgb_bp, camera_tf)

    print("[RGBBank] Fixed camera location:", camera_tf.location)
    print("[RGBBank] Fixed camera rotation:", camera_tf.rotation)

    return camera


# ============================================================
# Save / visualization
# ============================================================

def save_rgb(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


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


def crop_from_bbox(rgb, bbox, args):
    crop_box = expand_bbox(
        bbox=bbox,
        image_width=args.image_width,
        image_height=args.image_height,
        expand_ratio=args.bbox_expand_ratio,
        extra_margin_px=args.crop_margin_px
    )

    x1, y1, x2, y2 = crop_box
    crop = rgb[y1:y2 + 1, x1:x2 + 1].copy()

    return crop, crop_box


def draw_debug(rgb, bbox, crop_box, angle, vehicle_yaw):
    debug = rgb.copy()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(
        debug,
        f"angle={angle:03d} yaw={vehicle_yaw:.1f}",
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
        img = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(img, thumb_size)

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
    full_dir, crop_dir, debug_dir = ensure_dirs(output_dir, overwrite=args.overwrite)

    client = None
    world = None
    original_settings = None
    actors = []

    rgb_queue = queue.Queue()

    metadata_rows = []
    contact_full_paths = []
    contact_crop_paths = []
    contact_debug_paths = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        world = client.get_world()
        original_settings = world.get_settings()

        print("[RGBBank] Using current/default CARLA world.")
        print("[RGBBank] No map/town will be loaded or changed.")
        print("[RGBBank] Current map:", world.get_map().name)

        setup_synchronous_mode(world, args.fixed_delta_seconds)

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

        rgb_camera = spawn_rgb_camera(
            world=world,
            blueprint_library=blueprint_library,
            camera_tf=fixed_camera_tf,
            args=args
        )

        actors.append(rgb_camera)

        rgb_camera.listen(lambda image: rgb_queue.put(image))

        # Warm-up
        for _ in range(20):
            world.tick()
            flush_queue(rgb_queue)

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

            for _ in range(args.settle_ticks):
                world.tick()

            frame = world.tick()
            image = wait_for_frame(rgb_queue, frame)
            rgb = carla_rgb_to_array(image)

            bbox = compute_actor_2d_bbox(
                actor=vehicle,
                camera_actor=rgb_camera,
                K=K,
                image_width=args.image_width,
                image_height=args.image_height
            )

            if bbox is None:
                print(f"[RGBBank] Warning: angle {angle:03d}, bbox not visible. Skipping.")
                continue

            crop, crop_box = crop_from_bbox(rgb, bbox, args)

            angle_name = f"angle_{angle:03d}"

            full_path = full_dir / f"{angle_name}_full.png"
            crop_path = crop_dir / f"{angle_name}.png"
            debug_path = debug_dir / f"{angle_name}_debug.png"

            save_rgb(full_path, rgb)
            save_rgb(crop_path, crop)

            if args.save_all_debug or angle in debug_angles:
                debug = draw_debug(
                    rgb=rgb,
                    bbox=bbox,
                    crop_box=crop_box,
                    angle=angle,
                    vehicle_yaw=yaw
                )
                save_rgb(debug_path, debug)
                contact_debug_paths.append(debug_path)

            if angle % 30 == 0:
                contact_full_paths.append(full_path)
                contact_crop_paths.append(crop_path)
                print(
                    f"[RGBBank] Saved angle {angle:03d} | "
                    f"crop={crop.shape[1]}x{crop.shape[0]} | "
                    f"yaw={yaw:.1f}"
                )

            metadata_rows.append({
                "angle": int(angle),
                "vehicle_yaw": float(yaw),
                "full_rgb": str(full_path.relative_to(output_dir)),
                "crop_rgb": str(crop_path.relative_to(output_dir)),
                "debug": str(debug_path.relative_to(output_dir)),
                "bbox": bbox,
                "crop_box": crop_box,
                "crop_width": int(crop.shape[1]),
                "crop_height": int(crop.shape[0])
            })

        metadata = {
            "schema": "he_fixed_camera_rotating_vehicle_rgb_v1",
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
            writer.writerow(["angle", "vehicle_yaw", "crop_width", "crop_height", "bbox", "crop_box"])

            for row in metadata_rows:
                writer.writerow([
                    row["angle"],
                    f"{row['vehicle_yaw']:.3f}",
                    row["crop_width"],
                    row["crop_height"],
                    row["bbox"],
                    row["crop_box"]
                ])

        make_contact_sheet(
            contact_full_paths,
            output_dir / "contact_sheet_full_30deg.png",
            title="Fixed camera, rotating vehicle | full RGB"
        )

        make_contact_sheet(
            contact_crop_paths,
            output_dir / "contact_sheet_crop_30deg.png",
            title="Fixed camera, rotating vehicle | bbox crop RGB"
        )

        make_contact_sheet(
            contact_debug_paths,
            output_dir / "contact_sheet_debug_30deg.png",
            title="Fixed camera, rotating vehicle | debug"
        )

        print("\n[RGBBank] Done.")
        print("[RGBBank] Output:", output_dir)
        print("[RGBBank] Check:")
        print("  contact_sheet_full_30deg.png")
        print("  contact_sheet_crop_30deg.png")
        print("  contact_sheet_debug_30deg.png")

    except Exception:
        traceback.print_exc()

    finally:
        print("[RGBBank] Cleaning up...")

        for actor in actors:
            safe_destroy(actor)

        restore_world_settings(world, original_settings)

        print("[RGBBank] Done cleanup.")


if __name__ == "__main__":
    main()