#!/usr/bin/env python3
"""
Generate a 360-degree CARLA sprite bank for one vehicle.

Outputs:
  sprite_bank/<vehicle_name>/
    rgb/angle_000.png ...
    mask/angle_000_mask.png ...
    rgba/angle_000_rgba.png ...
    metadata.json
    contact_sheet.png

Angle convention:
  angle_000 = rear view
  angle_090 = left-side-ish view
  angle_180 = front view
  angle_270 = right-side-ish view

Run after starting CARLA server.
"""

import argparse
import json
import math
import queue
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import carla
except ImportError as exc:
    raise RuntimeError(
        "Could not import CARLA Python API. Make sure CARLA's PythonAPI/carla "
        "is available in PYTHONPATH."
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

    parser.add_argument("--town", default=None,
                        help="Optional CARLA town to load, e.g., Town05. If omitted, current world is used.")

    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--vehicle-name", default="vehicle_blue_sedan")
    parser.add_argument("--color", default="0,0,255",
                        help="CARLA color string. Example: 0,0,255 for blue.")

    parser.add_argument("--output-root", default="assets/sprite_bank")

    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=70.0)

    parser.add_argument("--camera-distance", type=float, default=12.0)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--camera-pitch", type=float, default=0.0)

    parser.add_argument("--angle-step", type=int, default=1)
    parser.add_argument("--vehicle-tag", type=int, default=10,
                        help="CARLA semantic tag for vehicles is usually 10.")

    parser.add_argument("--padding-ratio", type=float, default=0.15)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)

    return parser.parse_args()


def ensure_dirs(root: Path):
    rgb_dir = root / "rgb"
    mask_dir = root / "mask"
    rgba_dir = root / "rgba"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgba_dir.mkdir(parents=True, exist_ok=True)

    return rgb_dir, mask_dir, rgba_dir


def find_vehicle_blueprint(blueprint_library, vehicle_filter: str):
    matches = blueprint_library.filter(vehicle_filter)
    if len(matches) == 0:
        print(f"[WARN] No blueprint matched {vehicle_filter}. Falling back to vehicle.*")
        matches = blueprint_library.filter("vehicle.*")

    if len(matches) == 0:
        raise RuntimeError("No vehicle blueprint found in this CARLA world.")

    return matches[0]


def image_to_rgb_array(image):
    """
    CARLA raw camera image is BGRA.
    Return RGB uint8 array.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, [2, 1, 0]]
    return rgb


def semantic_to_tag_array(image):
    """
    CARLA semantic segmentation tag is stored in red channel.
    Raw buffer is BGRA, so red channel is index 2.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    tags = array[:, :, 2]
    return tags


def keep_largest_component(mask):
    """
    Keeps largest connected component in the mask.
    Useful in case background/static vehicles are also visible.
    """
    if cv2 is None:
        return mask

    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels <= 1:
        return mask

    # label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1

    return labels == largest_label


def crop_with_mask(rgb, mask, padding_ratio):
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    h, w = mask.shape[:2]

    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x)
    y2p = min(h, y2 + pad_y)

    rgb_crop = rgb[y1p:y2p, x1p:x2p]
    mask_crop = mask[y1p:y2p, x1p:x2p]

    # Anchor = bottom-center of actual vehicle mask inside cropped image
    cys, cxs = np.where(mask_crop)
    anchor_x = float((cxs.min() + cxs.max()) / 2.0)
    anchor_y = float(cys.max())

    result = {
        "rgb_crop": rgb_crop,
        "mask_crop": mask_crop,
        "bbox_original": [int(x1), int(y1), int(x2), int(y2)],
        "bbox_padded": [int(x1p), int(y1p), int(x2p), int(y2p)],
        "anchor_point": {
            "x": anchor_x,
            "y": anchor_y
        }
    }

    return result


def save_sprite_files(angle, crop_result, rgb_dir, mask_dir, rgba_dir):
    angle_name = f"angle_{angle:03d}"

    rgb_crop = crop_result["rgb_crop"]
    mask_crop = crop_result["mask_crop"]

    rgb_path = rgb_dir / f"{angle_name}.png"
    mask_path = mask_dir / f"{angle_name}_mask.png"
    rgba_path = rgba_dir / f"{angle_name}_rgba.png"

    rgb_img = Image.fromarray(rgb_crop)
    mask_img = Image.fromarray((mask_crop.astype(np.uint8) * 255), mode="L")

    rgba = np.zeros((rgb_crop.shape[0], rgb_crop.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_crop
    rgba[:, :, 3] = mask_crop.astype(np.uint8) * 255
    rgba_img = Image.fromarray(rgba, mode="RGBA")

    rgb_img.save(rgb_path)
    mask_img.save(mask_path)
    rgba_img.save(rgba_path)

    return rgb_path, mask_path, rgba_path


def get_sensor_data(sensor_queue, expected_frame, timeout=5.0):
    """
    Discard old frames until expected frame is reached.
    """
    while True:
        data = sensor_queue.get(timeout=timeout)
        if data.frame == expected_frame:
            return data
        if data.frame > expected_frame:
            return data


def make_contact_sheet(rgba_dir: Path, output_path: Path):
    selected_angles = list(range(0, 360, 30))
    thumb_w, thumb_h = 260, 180
    label_h = 24

    cols = 4
    rows = math.ceil(len(selected_angles) / cols)

    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, angle in enumerate(selected_angles):
        path = rgba_dir / f"angle_{angle:03d}_rgba.png"
        if not path.exists():
            continue

        img = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", img.size, "white")
        bg.alpha_composite(img)

        thumb = ImageOps.contain(bg.convert("RGB"), (thumb_w, thumb_h))

        col = idx % cols
        row = idx // cols

        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)

        paste_x = x0 + (thumb_w - thumb.width) // 2
        paste_y = y0 + (thumb_h - thumb.height) // 2

        sheet.paste(thumb, (paste_x, paste_y))
        draw.text((x0 + 8, y0 + thumb_h + 4), f"{angle:03d} deg", fill="black")

    sheet.save(output_path)


def main():
    args = parse_args()

    output_root = Path(args.output_root) / args.vehicle_name
    rgb_dir, mask_dir, rgba_dir = ensure_dirs(output_root)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    if args.town:
        print(f"[INFO] Loading world: {args.town}")
        world = client.load_world(args.town)
    else:
        world = client.get_world()

    original_settings = world.get_settings()

    actors_to_destroy = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        vehicle_bp = find_vehicle_blueprint(blueprint_library, args.vehicle_filter)

        if vehicle_bp.has_attribute("color"):
            vehicle_bp.set_attribute("color", args.color)

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in this map.")

        spawn_index = min(args.spawn_index, len(spawn_points) - 1)
        base_transform = spawn_points[spawn_index]

        base_location = base_transform.location
        base_yaw = base_transform.rotation.yaw

        print(f"[INFO] Vehicle blueprint: {vehicle_bp.id}")
        print(f"[INFO] Spawn index: {spawn_index}")
        print(f"[INFO] Base yaw: {base_yaw:.2f}")

        vehicle = world.try_spawn_actor(vehicle_bp, base_transform)
        if vehicle is None:
            raise RuntimeError("Could not spawn vehicle. Try a different --spawn-index.")

        actors_to_destroy.append(vehicle)
        vehicle.set_simulate_physics(False)

        # Camera fixed behind the vehicle for yaw=0.
        theta = math.radians(base_yaw)
        forward_x = math.cos(theta)
        forward_y = math.sin(theta)

        cam_location = carla.Location(
            x=base_location.x - forward_x * args.camera_distance,
            y=base_location.y - forward_y * args.camera_distance,
            z=base_location.z + args.camera_height
        )

        cam_rotation = carla.Rotation(
            pitch=args.camera_pitch,
            yaw=base_yaw,
            roll=0.0
        )

        camera_transform = carla.Transform(cam_location, cam_rotation)

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

        rgb_camera = world.spawn_actor(rgb_bp, camera_transform)
        seg_camera = world.spawn_actor(seg_bp, camera_transform)

        actors_to_destroy.extend([rgb_camera, seg_camera])

        rgb_queue = queue.Queue()
        seg_queue = queue.Queue()

        rgb_camera.listen(rgb_queue.put)
        seg_camera.listen(seg_queue.put)

        # Warmup ticks
        for _ in range(5):
            world.tick()
            time.sleep(0.01)

        bb = vehicle.bounding_box.extent
        vehicle_dims = {
            "length_m": float(2.0 * bb.x),
            "width_m": float(2.0 * bb.y),
            "height_m": float(2.0 * bb.z)
        }

        metadata = {
            "schema": "he_sprite_bank_v1",
            "vehicle_name": args.vehicle_name,
            "vehicle_model": vehicle_bp.id,
            "vehicle_color": args.color,
            "angle_step_deg": args.angle_step,
            "angle_convention": {
                "0": "rear_view",
                "90": "left_side_view",
                "180": "front_view",
                "270": "right_side_view"
            },
            "capture_settings": {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "fov_deg": args.fov,
                "camera_distance_m": args.camera_distance,
                "camera_height_m": args.camera_height,
                "camera_pitch_deg": args.camera_pitch,
                "spawn_index": spawn_index,
                "town": args.town,
                "fixed_delta_seconds": args.fixed_delta_seconds
            },
            "semantic_mask": {
                "vehicle_tag": args.vehicle_tag,
                "tag_channel": "red"
            },
            "vehicle_dimensions": vehicle_dims,
            "paths": {
                "rgb_dir": "rgb",
                "mask_dir": "mask",
                "rgba_dir": "rgba"
            },
            "angles": {}
        }

        angles = list(range(0, 360, args.angle_step))

        for angle in angles:
            yaw = base_yaw + angle

            vehicle_transform = carla.Transform(
                base_location,
                carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
            )

            vehicle.set_transform(vehicle_transform)

            frame = world.tick()

            rgb_image = get_sensor_data(rgb_queue, frame)
            seg_image = get_sensor_data(seg_queue, frame)

            rgb = image_to_rgb_array(rgb_image)
            tags = semantic_to_tag_array(seg_image)

            mask = tags == args.vehicle_tag
            mask = keep_largest_component(mask)

            crop_result = crop_with_mask(rgb, mask, args.padding_ratio)

            if crop_result is None:
                print(f"[WARN] Empty mask at angle {angle:03d}. Skipping.")
                continue

            rgb_path, mask_path, rgba_path = save_sprite_files(
                angle,
                crop_result,
                rgb_dir,
                mask_dir,
                rgba_dir
            )

            h, w = crop_result["mask_crop"].shape[:2]

            metadata["angles"][f"{angle:03d}"] = {
                "angle_deg": angle,
                "vehicle_world_yaw_deg": float(yaw),
                "rgb": str(rgb_path.relative_to(output_root)),
                "mask": str(mask_path.relative_to(output_root)),
                "rgba": str(rgba_path.relative_to(output_root)),
                "sprite_width": int(w),
                "sprite_height": int(h),
                "bbox_original_image": crop_result["bbox_original"],
                "bbox_padded_image": crop_result["bbox_padded"],
                "anchor_point": crop_result["anchor_point"]
            }

            if angle % 30 == 0:
                print(f"[INFO] Saved angle {angle:03d}")

        metadata_path = output_root / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        contact_sheet_path = output_root / "contact_sheet.png"
        make_contact_sheet(rgba_dir, contact_sheet_path)

        print("\n[DONE] Sprite bank generated:")
        print(f"  Output folder: {output_root}")
        print(f"  Metadata:      {metadata_path}")
        print(f"  Contact sheet: {contact_sheet_path}")

    finally:
        print("[INFO] Cleaning up CARLA actors.")
        for actor in actors_to_destroy:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass

        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()