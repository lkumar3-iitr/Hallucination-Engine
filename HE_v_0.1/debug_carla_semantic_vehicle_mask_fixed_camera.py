#!/usr/bin/env python3
"""
debug_carla_semantic_vehicle_mask_fixed_camera.py

Debug CARLA semantic segmentation channel/tag for fixed-camera rotating-vehicle sprite generation.

This script:
  - Does NOT load/change map.
  - Uses current/default CARLA world.
  - Spawns one vehicle.
  - Spawns fixed RGB + semantic cameras.
  - Rotates only the vehicle.
  - Captures selected angles.
  - Saves RGB, bbox debug, all semantic channels, vehicle-tag masks, best mask, filled alpha, and RGBA preview.

Goal:
  Find which raw channel/tag combination gives a clean vehicle mask.

Example:
  python debug_carla_semantic_vehicle_mask_fixed_camera.py ^
    --vehicle-blueprint vehicle.tesla.model3 ^
    --vehicle-name vehicle_blue_sedan_semantic_debug ^
    --angles 0,90,180,270 ^
    --output-root assets\\semantic_debug ^
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
    parser.add_argument("--vehicle-name", default="vehicle_blue_sedan_semantic_debug")
    parser.add_argument("--color", default="0,0,255")

    parser.add_argument("--output-root", default="assets/semantic_debug")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=55.0)

    parser.add_argument("--camera-distance", type=float, default=10.0)
    parser.add_argument("--camera-height", type=float, default=1.35)
    parser.add_argument("--target-height", type=float, default=1.0)

    parser.add_argument("--spawn-index", type=int, default=-1)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--settle-ticks", type=int, default=2)

    parser.add_argument("--bbox-expand-ratio", type=float, default=0.10)
    parser.add_argument("--crop-margin-px", type=int, default=30)

    parser.add_argument(
        "--angles",
        default="0,90,180,270",
        help="Comma-separated angles to debug, e.g. 0,90,180,270 or 180"
    )

    parser.add_argument(
        "--candidate-tags",
        default="10,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28",
        help="Semantic tag values to test. Default includes vehicle tag 10 and nearby possibilities."
    )

    parser.add_argument("--window-black-value", type=int, default=20)

    parser.add_argument("--no-clear-existing-actors", action="store_true")

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

    print(f"[SemDebug] Clearing {len(actors)} existing vehicles/walkers/sensors...")

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


def save_rgb(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_gray(path, gray):
    cv2.imwrite(str(path), gray)


def save_rgba(path, rgba):
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), bgra)


# ============================================================
# Image conversion
# ============================================================

def carla_rgb_to_array(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]
    return rgb.copy()


def carla_raw_bgra(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr.copy()


# ============================================================
# Geometry
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

    camera_location = carla.Location(
        x=vehicle_loc.x + camera_distance * np.cos(yaw_rad),
        y=vehicle_loc.y + camera_distance * np.sin(yaw_rad),
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
    return base_yaw + (180.0 - float(angle_deg))


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


# ============================================================
# Spawn
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

        print("[SemDebug] Warning: requested vehicle not found. Using:", vehicles[0].id)
        return vehicles[0]


def spawn_vehicle(world, blueprint_library, args):
    vehicle_bp = find_vehicle_blueprint(blueprint_library, args.vehicle_blueprint)

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", args.color)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "he_semantic_debug_target")

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
            carla.Rotation(pitch=0.0, yaw=sp.rotation.yaw, roll=0.0)
        )

        vehicle = world.try_spawn_actor(vehicle_bp, tf)

        if vehicle is not None:
            vehicle.set_autopilot(False)
            vehicle.set_simulate_physics(False)
            world.tick()

            print("[SemDebug] Spawned vehicle:", vehicle_bp.id)
            print("[SemDebug] Location:", vehicle.get_location())
            print("[SemDebug] Base yaw:", vehicle.get_transform().rotation.yaw)

            return vehicle, vehicle_bp.id

    raise RuntimeError("Could not spawn target vehicle.")


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

    print("[SemDebug] Fixed camera location:", camera_tf.location)
    print("[SemDebug] Fixed camera rotation:", camera_tf.rotation)

    return rgb_camera, seg_camera


# ============================================================
# Mask processing
# ============================================================

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
    binary = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = binary.shape[:2]

    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)

    return binary | flood_inv


def clean_mask(mask_u8):
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    mask_u8 = keep_largest_component(mask_u8)

    return mask_u8


def make_rgba_preview(rgb, alpha, window_black_value=20):
    filled = fill_holes(alpha)
    window_mask = cv2.subtract(filled, alpha)

    rgb2 = rgb.copy()
    win = window_mask > 0
    v = int(np.clip(window_black_value, 0, 255))
    rgb2[win] = np.array([v, v, v], dtype=np.uint8)

    alpha2 = cv2.GaussianBlur(filled, (3, 3), 0)

    rgba = np.dstack([rgb2, alpha2])
    rgba[alpha2 <= 5, :3] = 0

    return rgba, window_mask, alpha2


def crop_by_alpha(rgb_or_rgba, alpha, crop_margin_px):
    ys, xs = np.where(alpha > 10)

    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = alpha.shape[:2]

    x1 = max(0, int(xs.min()) - crop_margin_px)
    y1 = max(0, int(ys.min()) - crop_margin_px)
    x2 = min(w - 1, int(xs.max()) + crop_margin_px)
    y2 = min(h - 1, int(ys.max()) + crop_margin_px)

    return rgb_or_rgba[y1:y2 + 1, x1:x2 + 1].copy(), [x1, y1, x2, y2]


def draw_bbox_debug(rgb, bbox, crop_bbox, angle):
    out = rgb.copy()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if crop_bbox is not None:
        x1, y1, x2, y2 = crop_bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(
        out,
        f"angle={angle:03d}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return out


def score_mask(mask, bbox, args):
    """
    Score a candidate semantic channel/tag mask inside expanded bbox.
    """
    if bbox is None:
        return 0

    x1, y1, x2, y2 = expand_bbox(
        bbox,
        args.image_width,
        args.image_height,
        args.bbox_expand_ratio,
        args.crop_margin_px
    )

    roi = mask[y1:y2 + 1, x1:x2 + 1]
    pixels = int(np.sum(roi > 0))

    return pixels


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_root) / args.vehicle_name

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    angles = [int(x.strip()) for x in args.angles.split(",") if x.strip()]
    candidate_tags = [int(x.strip()) for x in args.candidate_tags.split(",") if x.strip()]

    client = None
    world = None
    original_settings = None
    actors = []

    rgb_queue = queue.Queue()
    seg_queue = queue.Queue()

    results = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        world = client.get_world()
        original_settings = world.get_settings()

        print("[SemDebug] Using current/default CARLA world.")
        print("[SemDebug] No map/town will be loaded or changed.")
        print("[SemDebug] Current map:", world.get_map().name)

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

        camera_tf = build_fixed_front_camera_transform(
            vehicle,
            args.camera_distance,
            args.camera_height,
            args.target_height
        )

        rgb_camera, seg_camera = spawn_cameras(world, blueprint_library, camera_tf, args)
        actors.extend([rgb_camera, seg_camera])

        rgb_camera.listen(lambda image: rgb_queue.put(image))
        seg_camera.listen(lambda image: seg_queue.put(image))

        for _ in range(20):
            world.tick()
            flush_queue(rgb_queue)
            flush_queue(seg_queue)

        K = build_camera_intrinsics(args.image_width, args.image_height, args.fov)

        for angle in angles:
            angle_dir = output_dir / f"angle_{angle:03d}"
            angle_dir.mkdir(parents=True, exist_ok=True)

            yaw = vehicle_yaw_for_angle(base_yaw, angle)

            vehicle_tf = carla.Transform(
                base_location,
                carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0)
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
            seg_raw = carla_raw_bgra(seg_image)

            bbox = compute_actor_2d_bbox(
                vehicle,
                rgb_camera,
                K,
                args.image_width,
                args.image_height
            )

            if bbox is None:
                print(f"[SemDebug] angle {angle:03d}: bbox not visible.")
                continue

            x1e, y1e, x2e, y2e = expand_bbox(
                bbox,
                args.image_width,
                args.image_height,
                args.bbox_expand_ratio,
                args.crop_margin_px
            )

            save_rgb(angle_dir / f"angle_{angle:03d}_rgb.png", rgb)

            bbox_debug = draw_bbox_debug(rgb, bbox, [x1e, y1e, x2e, y2e], angle)
            save_rgb(angle_dir / f"angle_{angle:03d}_bbox_debug.png", bbox_debug)

            best = {
                "score": -1,
                "channel": None,
                "tag": None,
                "mask": None
            }

            channel_stats = []

            for ch in range(4):
                ch_img = seg_raw[:, :, ch]
                save_gray(angle_dir / f"angle_{angle:03d}_seg_ch{ch}.png", ch_img)

                # Save unique values in ROI for this channel.
                roi_ch = ch_img[y1e:y2e + 1, x1e:x2e + 1]
                unique_vals, counts = np.unique(roi_ch, return_counts=True)

                top_pairs = sorted(
                    [(int(v), int(c)) for v, c in zip(unique_vals, counts)],
                    key=lambda x: x[1],
                    reverse=True
                )[:20]

                for tag in candidate_tags:
                    raw_mask = (ch_img == tag).astype(np.uint8) * 255

                    # Restrict to expanded bbox.
                    restricted = np.zeros_like(raw_mask)
                    restricted[y1e:y2e + 1, x1e:x2e + 1] = raw_mask[y1e:y2e + 1, x1e:x2e + 1]

                    cleaned = clean_mask(restricted)
                    score = score_mask(cleaned, bbox, args)

                    if score > best["score"]:
                        best = {
                            "score": score,
                            "channel": ch,
                            "tag": tag,
                            "mask": cleaned
                        }

                channel_stats.append({
                    "channel": ch,
                    "top_values_in_bbox_roi": top_pairs
                })

            if best["mask"] is None or best["score"] <= 0:
                print(f"[SemDebug] angle {angle:03d}: no semantic mask found.")
                continue

            save_gray(
                angle_dir / f"angle_{angle:03d}_best_mask_ch{best['channel']}_tag{best['tag']}.png",
                best["mask"]
            )

            rgba_preview, window_mask, alpha_filled = make_rgba_preview(
                rgb,
                best["mask"],
                window_black_value=args.window_black_value
            )

            save_gray(angle_dir / f"angle_{angle:03d}_filled_alpha.png", alpha_filled)
            save_gray(angle_dir / f"angle_{angle:03d}_window_mask.png", window_mask)

            cropped_rgba = crop_by_alpha(rgba_preview, alpha_filled, args.crop_margin_px)

            if cropped_rgba is not None:
                crop_img, crop_box = cropped_rgba
                save_rgba(angle_dir / f"angle_{angle:03d}_rgba_preview_crop.png", crop_img)
            else:
                crop_box = None

            # Debug overlay.
            overlay = rgb.copy()
            m = alpha_filled > 0
            overlay[m] = (0.7 * overlay[m] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
            wmask = window_mask > 0
            overlay[wmask] = (0.5 * overlay[wmask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            save_rgb(angle_dir / f"angle_{angle:03d}_alpha_window_overlay.png", overlay)

            print(
                f"[SemDebug] angle {angle:03d}: "
                f"best channel={best['channel']} tag={best['tag']} pixels={best['score']} "
                f"crop_box={crop_box}"
            )

            results.append({
                "angle": angle,
                "vehicle_yaw": yaw,
                "bbox": bbox,
                "expanded_bbox": [x1e, y1e, x2e, y2e],
                "best_channel": best["channel"],
                "best_tag": best["tag"],
                "best_score_pixels": int(best["score"]),
                "channel_stats": channel_stats,
                "crop_box": crop_box
            })

        with open(output_dir / "semantic_debug_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "world_map": world.get_map().name,
                    "vehicle_blueprint_requested": args.vehicle_blueprint,
                    "vehicle_blueprint_actual": actual_bp,
                    "base_yaw": base_yaw,
                    "angles": results
                },
                f,
                indent=2
            )

        print("\n[SemDebug] Done.")
        print("[SemDebug] Output:", output_dir)
        print("[SemDebug] Open each angle folder and check:")
        print("  *_seg_ch0.png ... *_seg_ch3.png")
        print("  *_best_mask_*.png")
        print("  *_rgba_preview_crop.png")
        print("  *_alpha_window_overlay.png")

    except Exception:
        traceback.print_exc()

    finally:
        print("[SemDebug] Cleaning up...")

        for actor in actors:
            safe_destroy(actor)

        restore_world_settings(world, original_settings)

        print("[SemDebug] Done cleanup.")


if __name__ == "__main__":
    main()