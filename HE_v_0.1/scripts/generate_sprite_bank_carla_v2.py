#!/usr/bin/env python3
"""
Generate a CARLA vehicle sprite bank using orbit-camera capture.

Important design:
  - Uses the currently loaded CARLA world.
  - Does NOT load/change town.
  - Vehicle is placed high above the current map to isolate the mask.
  - Camera orbits around the fixed vehicle.
  - Camera always looks at the vehicle center.
  - Saves RGB crop, binary mask crop, RGBA sprite, metadata, and contact sheet.

Angle convention:
  angle_000 = rear view of vehicle
  angle_180 = front view of vehicle

Example:
  python scripts/generate_sprite_bank_carla_v2.py ^
    --vehicle-filter vehicle.tesla.model3 ^
    --vehicle-name vehicle_blue_sedan ^
    --color 0,0,255 ^
    --angle-step 1 ^
    --output-root assets/sprite_bank ^
    --overwrite
"""

import argparse
import json
import math
import queue
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import carla
except ImportError as exc:
    raise RuntimeError(
        "Could not import CARLA Python API. "
        "Make sure CARLA PythonAPI is in PYTHONPATH."
    ) from exc

try:
    import cv2
except ImportError:
    cv2 = None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)

    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--vehicle-name", default="vehicle_blue_sedan")
    parser.add_argument("--color", default="0,0,255")

    parser.add_argument("--output-root", default="assets/sprite_bank")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=55.0)

    parser.add_argument("--camera-distance", type=float, default=7.0)
    parser.add_argument("--camera-height", type=float, default=1.6)
    parser.add_argument("--target-height", type=float, default=0.9)

    parser.add_argument("--angle-step", type=int, default=1)

    parser.add_argument("--vehicle-tag", type=int, default=10)
    parser.add_argument("--padding-ratio", type=float, default=0.18)
    parser.add_argument("--min-mask-pixels", type=int, default=200)

    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--capture-z-offset", type=float, default=80.0)

    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)

    parser.add_argument(
        "--vehicle-yaw",
        type=float,
        default=None,
        help="If omitted, uses the yaw of the selected spawn point."
    )

    return parser.parse_args()


def ensure_dirs(output_root: Path, overwrite: bool):
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)

    rgb_dir = output_root / "rgb"
    mask_dir = output_root / "mask"
    rgba_dir = output_root / "rgba"
    debug_dir = output_root / "debug"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgba_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    return rgb_dir, mask_dir, rgba_dir, debug_dir


def find_vehicle_blueprint(blueprint_library, vehicle_filter: str):
    matches = blueprint_library.filter(vehicle_filter)

    if len(matches) == 0:
        print(f"[WARN] No match for {vehicle_filter}. Falling back to vehicle.*")
        matches = blueprint_library.filter("vehicle.*")

    if len(matches) == 0:
        raise RuntimeError("No vehicle blueprint found.")

    return matches[0]


def image_to_rgb_array(image):
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    raw = raw.reshape((image.height, image.width, 4))

    # CARLA camera raw format: BGRA
    rgb = raw[:, :, [2, 1, 0]].copy()
    return rgb


def semantic_to_tag_array(image):
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    raw = raw.reshape((image.height, image.width, 4))

    # CARLA semantic raw labels are usually stored in red channel.
    tags = raw[:, :, 2].copy()
    return tags


def drain_queue(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def wait_for_sensor_frame(q, expected_frame, timeout=5.0):
    start = time.time()

    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for sensor frame {expected_frame}")

        data = q.get(timeout=timeout)

        if data.frame >= expected_frame:
            return data


def keep_largest_component(mask):
    if cv2 is None:
        return mask

    mask_u8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8
    )

    if num_labels <= 1:
        return mask

    # label 0 = background
    component_areas = stats[1:, cv2.CC_STAT_AREA]

    if len(component_areas) == 0:
        return mask

    largest_label = int(np.argmax(component_areas)) + 1

    return labels == largest_label


def clean_mask(mask):
    if cv2 is None:
        return mask

    mask_u8 = mask.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    return mask_u8 > 0


def crop_with_mask(rgb, mask, padding_ratio, min_mask_pixels):
    mask_pixels = int(mask.sum())

    if mask_pixels < min_mask_pixels:
        return None

    ys, xs = np.where(mask)

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    img_h, img_w = mask.shape[:2]

    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(img_w, x2 + pad_x)
    y2p = min(img_h, y2 + pad_y)

    rgb_crop = rgb[y1p:y2p, x1p:x2p].copy()
    mask_crop = mask[y1p:y2p, x1p:x2p].copy()

    crop_ys, crop_xs = np.where(mask_crop)

    anchor_x = float((crop_xs.min() + crop_xs.max()) / 2.0)
    anchor_y = float(crop_ys.max())

    return {
        "rgb_crop": rgb_crop,
        "mask_crop": mask_crop,
        "bbox_original": [x1, y1, x2, y2],
        "bbox_padded": [x1p, y1p, x2p, y2p],
        "anchor_point": {
            "x": anchor_x,
            "y": anchor_y
        },
        "mask_pixels": mask_pixels
    }


def save_sprite(angle, crop, rgb_dir, mask_dir, rgba_dir):
    angle_name = f"angle_{angle:03d}"

    rgb_crop = crop["rgb_crop"]
    mask_crop = crop["mask_crop"]

    rgb_path = rgb_dir / f"{angle_name}.png"
    mask_path = mask_dir / f"{angle_name}_mask.png"
    rgba_path = rgba_dir / f"{angle_name}_rgba.png"

    Image.fromarray(rgb_crop).save(rgb_path)

    mask_u8 = mask_crop.astype(np.uint8) * 255
    Image.fromarray(mask_u8, mode="L").save(mask_path)

    rgba = np.zeros(
        (rgb_crop.shape[0], rgb_crop.shape[1], 4),
        dtype=np.uint8
    )

    rgba[:, :, :3] = rgb_crop
    rgba[:, :, 3] = mask_u8

    # Remove invisible background color values.
    rgba[mask_u8 == 0, :3] = 0

    Image.fromarray(rgba, mode="RGBA").save(rgba_path)

    return rgb_path, mask_path, rgba_path


def save_debug(angle, rgb, mask, debug_dir):
    angle_name = f"angle_{angle:03d}"

    Image.fromarray(rgb).save(debug_dir / f"{angle_name}_full_rgb.png")

    mask_u8 = mask.astype(np.uint8) * 255
    Image.fromarray(mask_u8, mode="L").save(debug_dir / f"{angle_name}_full_mask.png")

    overlay = rgb.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(debug_dir / f"{angle_name}_overlay.png")


def make_contact_sheet(rgba_dir: Path, output_path: Path):
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

        bg = Image.new("RGBA", sprite.size, "white")
        bg.alpha_composite(sprite)

        thumb = ImageOps.contain(bg.convert("RGB"), (thumb_w, thumb_h))

        col = idx % cols
        row = idx // cols

        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)

        paste_x = x0 + (thumb_w - thumb.width) // 2
        paste_y = y0 + (thumb_h - thumb.height) // 2

        sheet.paste(thumb, (paste_x, paste_y))
        draw.text((x0 + 8, y0 + thumb_h + 5), f"{angle:03d} deg", fill="black")

    sheet.save(output_path)


def look_at_rotation(source: carla.Location, target: carla.Location):
    dx = target.x - source.x
    dy = target.y - source.y
    dz = target.z - source.z

    yaw = math.degrees(math.atan2(dy, dx))

    horizontal_dist = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horizontal_dist))

    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)


def orbit_camera_transform(
    target_location: carla.Location,
    vehicle_yaw_deg: float,
    view_angle_deg: float,
    camera_distance: float,
    camera_height: float
):
    """
    angle_000 = camera behind vehicle looking at rear
    angle_180 = camera in front of vehicle looking at front
    """

    azimuth_deg = vehicle_yaw_deg + view_angle_deg + 180.0
    azimuth_rad = math.radians(azimuth_deg)

    cam_x = target_location.x + camera_distance * math.cos(azimuth_rad)
    cam_y = target_location.y + camera_distance * math.sin(azimuth_rad)
    cam_z = target_location.z + camera_height

    cam_location = carla.Location(x=cam_x, y=cam_y, z=cam_z)
    cam_rotation = look_at_rotation(cam_location, target_location)

    return carla.Transform(cam_location, cam_rotation)


def main():
    args = parse_args()

    output_root = Path(args.output_root) / args.vehicle_name
    rgb_dir, mask_dir, rgba_dir, debug_dir = ensure_dirs(
        output_root,
        overwrite=args.overwrite
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    original_settings = world.get_settings()

    actors_to_destroy = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        vehicle_bp = find_vehicle_blueprint(
            blueprint_library,
            args.vehicle_filter
        )

        if vehicle_bp.has_attribute("color"):
            vehicle_bp.set_attribute("color", args.color)

        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "he_sprite_target")

        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            raise RuntimeError("No spawn points found in current CARLA map.")

        spawn_index = min(args.spawn_index, len(spawn_points) - 1)
        spawn_tf = spawn_points[spawn_index]

        base_yaw = spawn_tf.rotation.yaw if args.vehicle_yaw is None else args.vehicle_yaw

        vehicle_location = carla.Location(
            x=spawn_tf.location.x,
            y=spawn_tf.location.y,
            z=spawn_tf.location.z + args.capture_z_offset
        )

        vehicle_transform = carla.Transform(
            vehicle_location,
            carla.Rotation(pitch=0.0, yaw=base_yaw, roll=0.0)
        )

        print("[INFO] Using current CARLA world. No town/map will be loaded.")
        print(f"[INFO] Vehicle blueprint: {vehicle_bp.id}")
        print(f"[INFO] Vehicle yaw: {base_yaw:.2f}")
        print(f"[INFO] Capture location: x={vehicle_location.x:.2f}, "
              f"y={vehicle_location.y:.2f}, z={vehicle_location.z:.2f}")

        vehicle = world.try_spawn_actor(vehicle_bp, vehicle_transform)

        if vehicle is None:
            raise RuntimeError(
                "Could not spawn vehicle at high capture location. "
                "Try --spawn-index 1 or a different --capture-z-offset."
            )

        actors_to_destroy.append(vehicle)
        vehicle.set_simulate_physics(False)

        # Tick once so bounding box and transform settle.
        world.tick()

        vehicle_bb = vehicle.bounding_box.extent

        vehicle_dims = {
            "length_m": float(2.0 * vehicle_bb.x),
            "width_m": float(2.0 * vehicle_bb.y),
            "height_m": float(2.0 * vehicle_bb.z)
        }

        # Target point around vehicle body center.
        target_location = carla.Location(
            x=vehicle_location.x,
            y=vehicle_location.y,
            z=vehicle_location.z + args.target_height
        )

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

        initial_camera_tf = orbit_camera_transform(
            target_location=target_location,
            vehicle_yaw_deg=base_yaw,
            view_angle_deg=0.0,
            camera_distance=args.camera_distance,
            camera_height=args.camera_height
        )

        rgb_camera = world.spawn_actor(rgb_bp, initial_camera_tf)
        seg_camera = world.spawn_actor(seg_bp, initial_camera_tf)

        actors_to_destroy.extend([rgb_camera, seg_camera])

        rgb_q = queue.Queue()
        seg_q = queue.Queue()

        rgb_camera.listen(rgb_q.put)
        seg_camera.listen(seg_q.put)

        for _ in range(5):
            world.tick()
            time.sleep(0.01)

        drain_queue(rgb_q)
        drain_queue(seg_q)

        metadata = {
            "schema": "he_sprite_bank_v2",
            "world_map": world.get_map().name,
            "vehicle_name": args.vehicle_name,
            "vehicle_model": vehicle_bp.id,
            "vehicle_color": args.color,
            "angle_step_deg": args.angle_step,
            "angle_convention": {
                "0": "rear_view",
                "180": "front_view"
            },
            "capture_mode": "orbit_camera_fixed_vehicle_high_altitude",
            "capture_settings": {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "fov_deg": args.fov,
                "camera_distance_m": args.camera_distance,
                "camera_height_m": args.camera_height,
                "target_height_m": args.target_height,
                "capture_z_offset_m": args.capture_z_offset,
                "spawn_index": spawn_index,
                "vehicle_yaw_deg": base_yaw,
                "fixed_delta_seconds": args.fixed_delta_seconds
            },
            "semantic_mask": {
                "vehicle_tag": args.vehicle_tag,
                "note": "Mask is extracted from semantic segmentation raw red channel."
            },
            "vehicle_dimensions": vehicle_dims,
            "paths": {
                "rgb_dir": "rgb",
                "mask_dir": "mask",
                "rgba_dir": "rgba",
                "debug_dir": "debug"
            },
            "angles": {}
        }

        angles = list(range(0, 360, args.angle_step))
        debug_angles = set([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])

        for angle in angles:
            cam_tf = orbit_camera_transform(
                target_location=target_location,
                vehicle_yaw_deg=base_yaw,
                view_angle_deg=float(angle),
                camera_distance=args.camera_distance,
                camera_height=args.camera_height
            )

            rgb_camera.set_transform(cam_tf)
            seg_camera.set_transform(cam_tf)

            # One settle tick, one capture tick.
            world.tick()
            frame = world.tick()

            rgb_image = wait_for_sensor_frame(rgb_q, frame)
            seg_image = wait_for_sensor_frame(seg_q, frame)

            rgb = image_to_rgb_array(rgb_image)
            tags = semantic_to_tag_array(seg_image)

            mask = tags == args.vehicle_tag
            mask = keep_largest_component(mask)
            mask = clean_mask(mask)

            if angle in debug_angles:
                save_debug(angle, rgb, mask, debug_dir)

            crop = crop_with_mask(
                rgb=rgb,
                mask=mask,
                padding_ratio=args.padding_ratio,
                min_mask_pixels=args.min_mask_pixels
            )

            if crop is None:
                print(f"[WARN] Angle {angle:03d}: empty or too-small mask. Skipping.")
                continue

            rgb_path, mask_path, rgba_path = save_sprite(
                angle=angle,
                crop=crop,
                rgb_dir=rgb_dir,
                mask_dir=mask_dir,
                rgba_dir=rgba_dir
            )

            sprite_h, sprite_w = crop["mask_crop"].shape[:2]

            metadata["angles"][f"{angle:03d}"] = {
                "angle_deg": int(angle),
                "camera_transform": {
                    "location": {
                        "x": float(cam_tf.location.x),
                        "y": float(cam_tf.location.y),
                        "z": float(cam_tf.location.z)
                    },
                    "rotation": {
                        "pitch": float(cam_tf.rotation.pitch),
                        "yaw": float(cam_tf.rotation.yaw),
                        "roll": float(cam_tf.rotation.roll)
                    }
                },
                "rgb": str(rgb_path.relative_to(output_root)),
                "mask": str(mask_path.relative_to(output_root)),
                "rgba": str(rgba_path.relative_to(output_root)),
                "sprite_width": int(sprite_w),
                "sprite_height": int(sprite_h),
                "bbox_original_image": crop["bbox_original"],
                "bbox_padded_image": crop["bbox_padded"],
                "anchor_point": crop["anchor_point"],
                "mask_pixels": int(crop["mask_pixels"])
            }

            if angle % 30 == 0:
                print(f"[INFO] Saved angle {angle:03d} | "
                      f"sprite={sprite_w}x{sprite_h} | "
                      f"mask_pixels={crop['mask_pixels']}")

        metadata_path = output_root / "metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        contact_sheet_path = output_root / "contact_sheet.png"
        make_contact_sheet(rgba_dir, contact_sheet_path)

        print("\n[DONE] Sprite bank generated.")
        print(f"Output folder:  {output_root}")
        print(f"Metadata:       {metadata_path}")
        print(f"Contact sheet:  {contact_sheet_path}")
        print(f"Debug folder:   {debug_dir}")

    finally:
        print("[INFO] Cleaning up generated CARLA actors.")

        for actor in actors_to_destroy:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass

        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()