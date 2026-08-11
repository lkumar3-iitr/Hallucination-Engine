#!/usr/bin/env python3
"""
scan_carla_sprite_rgb_candidates.py

RGB-only candidate scanner for CARLA sprite-bank generation.

Purpose:
  - Find a clean spawn point for 360-degree sprite capture.
  - Does NOT load or change the map.
  - Uses current/default CARLA world.
  - Clears dynamic actors by default: vehicles, walkers, sensors.
  - Spawns one target vehicle at many candidate spawn points.
  - Rotates/orbits RGB camera around the fixed vehicle.
  - Saves full RGB views and bbox-cropped RGB views.
  - Saves contact sheets for visual inspection.

No RGBA.
No mask.
No GrabCut.
No semantic segmentation.

Angle convention:
  angle_000 = rear view
  angle_090 = side view
  angle_180 = front view
  angle_270 = opposite side view

Recommended first run:
  python scan_carla_sprite_rgb_candidates.py ^
    --vehicle-blueprint vehicle.tesla.model3 ^
    --vehicle-name vehicle_blue_sedan ^
    --candidate-count 30 ^
    --angle-step 30 ^
    --camera-distance 8.0 ^
    --fov 65 ^
    --output-root assets\\sprite_rgb_candidates ^
    --overwrite
"""

import argparse
import csv
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

    parser.add_argument("--output-root", default="assets/sprite_rgb_candidates")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=65.0)

    parser.add_argument("--camera-distance", type=float, default=8.0)
    parser.add_argument("--camera-height", type=float, default=1.35)
    parser.add_argument("--target-height", type=float, default=1.0)

    parser.add_argument("--angle-step", type=int, default=30)

    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument(
        "--spawn-indices",
        default=None,
        help="Optional comma-separated spawn indices, e.g. 0,5,12. Overrides candidate-count/start-index."
    )

    parser.add_argument("--crop-margin-px", type=int, default=35)
    parser.add_argument("--bbox-expand-ratio", type=float, default=0.12)

    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--settle-ticks", type=int, default=2)

    parser.add_argument(
        "--no-clear-existing-actors",
        action="store_true",
        help="Do not destroy existing vehicles/walkers/sensors."
    )

    return parser.parse_args()


# ============================================================
# CARLA utilities
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
    patterns = [
        "vehicle.*",
        "walker.*",
        "sensor.*",
    ]

    actors = []

    for pattern in patterns:
        actors.extend(list(world.get_actors().filter(pattern)))

    print(f"[RGBScan] Clearing {len(actors)} existing vehicles/walkers/sensors...")

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


# ============================================================
# Geometry
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

    OpenCV camera coords:
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
    Perfect circular orbit around fixed vehicle.

    angle_180 = front-facing camera, same direction as your old front-sprite code.
    angle_000 = rear view.
    """
    vehicle_tf = vehicle.get_transform()
    vehicle_loc = vehicle_tf.location
    vehicle_yaw = vehicle_tf.rotation.yaw

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
# Spawn helpers
# ============================================================

def find_vehicle_blueprint(blueprint_library, blueprint_id):
    try:
        return blueprint_library.find(blueprint_id)
    except RuntimeError:
        matches = blueprint_library.filter(blueprint_id)

        if len(matches) > 0:
            return matches[0]

        all_vehicles = blueprint_library.filter("vehicle.*")

        if len(all_vehicles) == 0:
            raise RuntimeError("No vehicle blueprints found.")

        print("[RGBScan] Warning: requested vehicle not found. Using:", all_vehicles[0].id)
        return all_vehicles[0]


def spawn_vehicle_at_index(world, blueprint_library, args, spawn_index):
    spawn_points = world.get_map().get_spawn_points()

    if spawn_index < 0 or spawn_index >= len(spawn_points):
        raise RuntimeError(f"Invalid spawn_index={spawn_index}. Total spawn points={len(spawn_points)}")

    vehicle_bp = find_vehicle_blueprint(blueprint_library, args.vehicle_blueprint)

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", args.color)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "he_rgb_candidate_target")

    sp = spawn_points[spawn_index]

    tf = carla.Transform(
        sp.location,
        carla.Rotation(
            pitch=0.0,
            yaw=sp.rotation.yaw,
            roll=0.0
        )
    )

    vehicle = world.try_spawn_actor(vehicle_bp, tf)

    if vehicle is None:
        return None, vehicle_bp.id

    vehicle.set_autopilot(False)
    vehicle.set_simulate_physics(False)

    world.tick()

    return vehicle, vehicle_bp.id


def spawn_rgb_camera(world, blueprint_library, camera_tf, args):
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(args.image_width))
    rgb_bp.set_attribute("image_size_y", str(args.image_height))
    rgb_bp.set_attribute("fov", str(args.fov))
    rgb_bp.set_attribute("sensor_tick", "0.0")

    camera = world.spawn_actor(rgb_bp, camera_tf)
    return camera


# ============================================================
# Saving / visualization
# ============================================================

def save_rgb(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def draw_bbox(rgb, bbox, label=None):
    out = rgb.copy()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if label:
        cv2.putText(
            out,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return out


def crop_from_bbox(rgb, bbox, args):
    if bbox is None:
        return None, None

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


def candidate_score(angle_stats):
    """
    Simple automatic score:
      - prefer all angles visible
      - penalize bbox touching image edge
      - penalize large bbox-area variation
    This does not detect all poles/occlusions, so visual inspection is still needed.
    """
    visible = [s for s in angle_stats if s["bbox_visible"]]

    if not visible:
        return -1e9

    visible_count = len(visible)
    edge_penalty = sum(1 for s in visible if s["touches_edge"])

    areas = [s["bbox_area"] for s in visible if s["bbox_area"] > 0]

    if len(areas) >= 2:
        area_mean = float(np.mean(areas))
        area_std = float(np.std(areas))
        variation = area_std / max(area_mean, 1.0)
    else:
        variation = 1.0

    score = visible_count * 100.0 - edge_penalty * 20.0 - variation * 40.0
    return float(score)


def bbox_touches_edge(bbox, image_width, image_height, margin=8):
    if bbox is None:
        return True

    x1, y1, x2, y2 = bbox

    return (
        x1 <= margin or
        y1 <= margin or
        x2 >= image_width - 1 - margin or
        y2 >= image_height - 1 - margin
    )


# ============================================================
# Candidate scanning
# ============================================================

def parse_spawn_indices(args, total_spawn_points):
    if args.spawn_indices:
        indices = []

        for item in args.spawn_indices.split(","):
            item = item.strip()
            if not item:
                continue
            indices.append(int(item))

        indices = [i for i in indices if 0 <= i < total_spawn_points]
        return indices

    all_indices = list(range(total_spawn_points))

    if args.shuffle:
        random.shuffle(all_indices)
        return all_indices[:args.candidate_count]

    indices = []
    i = args.start_index

    while i < total_spawn_points and len(indices) < args.candidate_count:
        indices.append(i)
        i += args.stride

    return indices


def scan_one_candidate(world, blueprint_library, args, spawn_index, output_root, K):
    candidate_name = f"candidate_{spawn_index:03d}"
    cand_dir = output_root / candidate_name

    full_dir = cand_dir / "full"
    crop_dir = cand_dir / "crop"
    debug_dir = cand_dir / "debug"

    full_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    actors = []
    rgb_queue = queue.Queue()

    stats = {
        "spawn_index": spawn_index,
        "candidate_dir": str(cand_dir),
        "vehicle_spawned": False,
        "vehicle_blueprint": None,
        "vehicle_location": None,
        "vehicle_yaw": None,
        "score": -1e9,
        "visible_angles": 0,
        "total_angles": 0,
        "angle_stats": []
    }

    try:
        vehicle, vehicle_bp_id = spawn_vehicle_at_index(
            world=world,
            blueprint_library=blueprint_library,
            args=args,
            spawn_index=spawn_index
        )

        if vehicle is None:
            print(f"[RGBScan] candidate {spawn_index:03d}: could not spawn vehicle.")
            return stats

        actors.append(vehicle)

        vehicle_tf = vehicle.get_transform()

        stats["vehicle_spawned"] = True
        stats["vehicle_blueprint"] = vehicle_bp_id
        stats["vehicle_location"] = {
            "x": float(vehicle_tf.location.x),
            "y": float(vehicle_tf.location.y),
            "z": float(vehicle_tf.location.z)
        }
        stats["vehicle_yaw"] = float(vehicle_tf.rotation.yaw)

        initial_tf = camera_transform_for_angle(
            vehicle=vehicle,
            angle_deg=180,
            camera_distance=args.camera_distance,
            camera_height=args.camera_height,
            target_height=args.target_height
        )

        rgb_camera = spawn_rgb_camera(
            world=world,
            blueprint_library=blueprint_library,
            camera_tf=initial_tf,
            args=args
        )

        actors.append(rgb_camera)

        rgb_camera.listen(lambda image: rgb_queue.put(image))

        # Warm-up
        for _ in range(5):
            world.tick()
            flush_queue(rgb_queue)

        full_paths = []
        crop_paths = []
        debug_paths = []

        angles = list(range(0, 360, args.angle_step))
        stats["total_angles"] = len(angles)

        for angle in angles:
            camera_tf = camera_transform_for_angle(
                vehicle=vehicle,
                angle_deg=angle,
                camera_distance=args.camera_distance,
                camera_height=args.camera_height,
                target_height=args.target_height
            )

            rgb_camera.set_transform(camera_tf)

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

            angle_name = f"angle_{angle:03d}"

            full_path = full_dir / f"{angle_name}.png"
            save_rgb(full_path, rgb)
            full_paths.append(full_path)

            bbox_visible = bbox is not None

            crop_path = None
            crop_box = None
            bbox_area = 0

            if bbox_visible:
                x1, y1, x2, y2 = bbox
                bbox_area = int((x2 - x1) * (y2 - y1))

                crop, crop_box = crop_from_bbox(rgb, bbox, args)

                if crop is not None and crop.size > 0:
                    crop_path = crop_dir / f"{angle_name}_crop.png"
                    save_rgb(crop_path, crop)
                    crop_paths.append(crop_path)

            debug = draw_bbox(
                rgb=rgb,
                bbox=bbox,
                label=f"spawn={spawn_index} angle={angle:03d}"
            )

            debug_path = debug_dir / f"{angle_name}_debug.png"
            save_rgb(debug_path, debug)
            debug_paths.append(debug_path)

            touches_edge = bbox_touches_edge(bbox, args.image_width, args.image_height)

            angle_stat = {
                "angle": int(angle),
                "bbox_visible": bool(bbox_visible),
                "bbox": bbox,
                "crop_box": crop_box,
                "bbox_area": int(bbox_area),
                "touches_edge": bool(touches_edge),
                "full_path": str(full_path.relative_to(cand_dir)),
                "crop_path": str(crop_path.relative_to(cand_dir)) if crop_path else None,
                "debug_path": str(debug_path.relative_to(cand_dir))
            }

            stats["angle_stats"].append(angle_stat)

        stats["visible_angles"] = sum(1 for s in stats["angle_stats"] if s["bbox_visible"])
        stats["score"] = candidate_score(stats["angle_stats"])

        make_contact_sheet(
            image_paths=full_paths,
            output_path=cand_dir / "contact_sheet_full.png",
            title=f"{candidate_name} | full RGB | score={stats['score']:.1f}"
        )

        make_contact_sheet(
            image_paths=crop_paths,
            output_path=cand_dir / "contact_sheet_crop.png",
            title=f"{candidate_name} | bbox crops | score={stats['score']:.1f}"
        )

        make_contact_sheet(
            image_paths=debug_paths,
            output_path=cand_dir / "contact_sheet_debug.png",
            title=f"{candidate_name} | bbox debug | score={stats['score']:.1f}"
        )

        print(
            f"[RGBScan] candidate {spawn_index:03d}: "
            f"visible={stats['visible_angles']}/{stats['total_angles']} "
            f"score={stats['score']:.1f} "
            f"dir={cand_dir}"
        )

        return stats

    finally:
        for actor in actors:
            safe_destroy(actor)

        world.tick()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    output_root = Path(args.output_root) / args.vehicle_name

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    client = None
    world = None
    original_settings = None

    all_stats = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        world = client.get_world()
        original_settings = world.get_settings()

        print("[RGBScan] Using current/default CARLA world.")
        print("[RGBScan] No map/town will be loaded or changed.")
        print("[RGBScan] Current map:", world.get_map().name)

        setup_synchronous_mode(world, args.fixed_delta_seconds)

        if not args.no_clear_existing_actors:
            clear_existing_dynamic_actors(world)

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        total_spawn_points = len(spawn_points)

        if total_spawn_points == 0:
            raise RuntimeError("No spawn points found in current map.")

        spawn_indices = parse_spawn_indices(args, total_spawn_points)

        print(f"[RGBScan] Total spawn points in map: {total_spawn_points}")
        print(f"[RGBScan] Scanning {len(spawn_indices)} candidates:", spawn_indices)

        K = build_camera_intrinsics(
            width=args.image_width,
            height=args.image_height,
            fov_degrees=args.fov
        )

        for spawn_index in spawn_indices:
            stats = scan_one_candidate(
                world=world,
                blueprint_library=blueprint_library,
                args=args,
                spawn_index=spawn_index,
                output_root=output_root,
                K=K
            )

            all_stats.append(stats)

        summary_json = output_root / "summary.json"

        with open(summary_json, "w", encoding="utf-8") as f:
            import json
            json.dump(
                {
                    "world_map": world.get_map().name,
                    "vehicle_blueprint": args.vehicle_blueprint,
                    "vehicle_name": args.vehicle_name,
                    "camera_distance": args.camera_distance,
                    "camera_height": args.camera_height,
                    "target_height": args.target_height,
                    "fov": args.fov,
                    "image_width": args.image_width,
                    "image_height": args.image_height,
                    "angle_step": args.angle_step,
                    "candidates": all_stats
                },
                f,
                indent=2
            )

        summary_csv = output_root / "summary.csv"

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "spawn_index",
                "vehicle_spawned",
                "visible_angles",
                "total_angles",
                "score",
                "candidate_dir",
                "vehicle_x",
                "vehicle_y",
                "vehicle_z",
                "vehicle_yaw"
            ])

            for s in all_stats:
                loc = s.get("vehicle_location") or {}
                writer.writerow([
                    s.get("spawn_index"),
                    s.get("vehicle_spawned"),
                    s.get("visible_angles"),
                    s.get("total_angles"),
                    f"{s.get('score', -1e9):.2f}",
                    s.get("candidate_dir"),
                    loc.get("x"),
                    loc.get("y"),
                    loc.get("z"),
                    s.get("vehicle_yaw")
                ])

        ranked = sorted(
            all_stats,
            key=lambda x: x.get("score", -1e9),
            reverse=True
        )

        print("\n[RGBScan] Done.")
        print("[RGBScan] Output:", output_root)
        print("[RGBScan] Summary JSON:", summary_json)
        print("[RGBScan] Summary CSV:", summary_csv)

        print("\n[RGBScan] Top candidates by automatic score:")
        for item in ranked[:10]:
            print(
                f"  spawn_index={item['spawn_index']:03d} | "
                f"visible={item['visible_angles']}/{item['total_angles']} | "
                f"score={item['score']:.1f} | "
                f"{item['candidate_dir']}"
            )

        print("\n[RGBScan] Now visually inspect contact_sheet_full.png and contact_sheet_crop.png.")
        print("[RGBScan] Pick the spawn_index where all angles are clean and no pole/object blocks the camera.")

    except Exception:
        traceback.print_exc()

    finally:
        restore_world_settings(world, original_settings)
        print("[RGBScan] Restored world settings.")


if __name__ == "__main__":
    main()