import argparse
import json
import os
import random
from pathlib import Path
import queue
import carla
import cv2
import numpy as np
import yaml

from projection_utils import (
    build_camera_intrinsics,
    compute_2d_bbox_from_projected_points,
    compute_relative_state,
    get_bbox_world_corners,
    project_world_points_to_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect CARLA-supervised bbox placement dataset for HEPlacementModel v1."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1_straight.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of samples from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()

def clear_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

def get_synced_camera_image(world, image_queue, target_frame, timeout=2.0):
    """
    Wait for the camera image whose frame matches target_frame.

    In CARLA synchronous mode:
      frame = world.tick()
      camera image should eventually arrive with image.frame == frame
    """
    while True:
        image = image_queue.get(timeout=timeout)

        if image.frame == target_frame:
            return image_to_numpy(image), image.frame

        # Discard old frames.
        if image.frame < target_frame:
            continue

        # If camera jumped ahead, return it, but this should rarely happen.
        if image.frame > target_frame:
            return image_to_numpy(image), image.frame
        
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_blueprint(world, filter_name, preferred=None):
    bp_lib = world.get_blueprint_library()
    candidates = bp_lib.filter(filter_name)

    if not candidates:
        raise RuntimeError(f"No blueprint found for filter: {filter_name}")

    if preferred is not None:
        preferred_matches = [bp for bp in candidates if preferred in bp.id]
        if preferred_matches:
            return preferred_matches[0]

    return random.choice(candidates)


def set_camera_blueprint_attributes(camera_bp, width, height, fov):
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", str(fov))
    return camera_bp


def image_to_numpy(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return rgb.copy()


def draw_debug_bbox(rgb, bbox_result, text=None):
    img = rgb.copy()

    if bbox_result["visible"] == 1:
        x1 = int(round(bbox_result["x_min"]))
        y1 = int(round(bbox_result["y_min"]))
        x2 = int(round(bbox_result["x_max"]))
        y2 = int(round(bbox_result["y_max"]))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx = int(round(bbox_result["center_x"]))
        by = int(round(bbox_result["bottom_y"]))
        cv2.circle(img, (cx, by), 5, (0, 0, 255), -1)

    if text:
        y = 25
        for line in text.split("\n"):
            cv2.putText(
                img,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 25

    return img


def set_synchronous_mode(world, fixed_delta_seconds):
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    return original_settings


def restore_world(world, original_settings):
    if original_settings is not None:
        world.apply_settings(original_settings)


def destroy_actors(actors):
    for actor in actors:
        if actor is not None and actor.is_alive:
            try:
                actor.destroy()
            except Exception:
                pass


def sample_relative_state(cfg_sampling):
    rel_x = random.uniform(sampling_cfg_float(cfg_sampling, "rel_x_min"), sampling_cfg_float(cfg_sampling, "rel_x_max"))
    rel_z = random.uniform(sampling_cfg_float(cfg_sampling, "rel_z_min"), sampling_cfg_float(cfg_sampling, "rel_z_max"))
    rel_yaw = random.uniform(sampling_cfg_float(cfg_sampling, "rel_yaw_min"), sampling_cfg_float(cfg_sampling, "rel_yaw_max"))
    return rel_x, rel_z, rel_yaw


def sampling_cfg_float(cfg, key):
    return float(cfg[key])


def relative_to_world_transform(
    world_map,
    ego_transform,
    rel_x,
    rel_z,
    rel_yaw,
):
    """
    Road-safe adversary placement.

    Instead of placing the adversary at arbitrary ego-relative x/y,
    this function:
      1. finds the ego road waypoint,
      2. moves forward along the road by rel_z,
      3. optionally shifts to a neighboring valid driving lane based on rel_x,
      4. uses the final driving-lane waypoint transform.

    This prevents vehicles from spawning on sidewalks, trees, poles, or buildings.
    """

    ego_wp = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if ego_wp is None:
        return None

    # Move forward along the road by rel_z meters.
    forward_candidates = ego_wp.next(float(rel_z))

    if not forward_candidates:
        return None

    base_wp = forward_candidates[0]

    # Approximate lane choice using rel_x.
    # CARLA lane width is usually around 3.5 m.
    lane_width = max(float(base_wp.lane_width), 3.0)

    lane_shift = int(round(float(rel_x) / lane_width))

    target_wp = base_wp

    if lane_shift > 0:
        # Move right lane by lane.
        for _ in range(lane_shift):
            next_wp = target_wp.get_right_lane()
            if (
                next_wp is not None
                and next_wp.lane_type == carla.LaneType.Driving
            ):
                target_wp = next_wp
            else:
                break

    elif lane_shift < 0:
        # Move left lane by lane.
        for _ in range(abs(lane_shift)):
            next_wp = target_wp.get_left_lane()
            if (
                next_wp is not None
                and next_wp.lane_type == carla.LaneType.Driving
            ):
                target_wp = next_wp
            else:
                break

    wp_transform = target_wp.transform

    # Use road/lane center position.
    loc = wp_transform.location

    # Use road yaw as base yaw, then add adversary relative yaw.
    road_yaw = wp_transform.rotation.yaw
    adv_yaw = road_yaw + float(rel_yaw)

    return carla.Transform(
        carla.Location(
            x=float(loc.x),
            y=float(loc.y),
            z=float(loc.z + 0.05),
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=float(adv_yaw),
            roll=0.0,
        ),
    )

def choose_spawn_transform(world, index=None):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in current CARLA map.")

    if index is not None:
        return spawn_points[int(index) % len(spawn_points)]

    return random.choice(spawn_points)

def wait_for_camera_frame(world, latest_image, min_frame, max_ticks=10):
    """
    In synchronous mode, wait until the camera callback has received
    an image frame at least as new as min_frame.
    """
    for _ in range(max_ticks):
        if latest_image["frame"] is not None and latest_image["frame"] >= min_frame:
            return latest_image["rgb"], latest_image["frame"]

        frame = world.tick()

        if latest_image["frame"] is not None and latest_image["frame"] >= frame:
            return latest_image["rgb"], latest_image["frame"]

    return latest_image["rgb"], latest_image["frame"]

def get_different_spawn_transform(world, ego_spawn):
    spawn_points = world.get_map().get_spawn_points()

    ego_loc = ego_spawn.location
    candidates = []

    for sp in spawn_points:
        d = sp.location.distance(ego_loc)
        if d > 30.0:
            candidates.append(sp)

    if candidates:
        return random.choice(candidates)

    return random.choice(spawn_points)

def semantic_to_labels(image):
    """
    Convert CARLA semantic segmentation raw image to label map.

    CARLA usually stores semantic tag in the red channel of BGRA raw data.
    But to debug channel issues, we return all three channels too.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    # BGRA layout:
    # B = 0, G = 1, R = 2, A = 3
    labels_r = array[:, :, 2].copy()
    labels_g = array[:, :, 1].copy()
    labels_b = array[:, :, 0].copy()

    return {
        "r": labels_r,
        "g": labels_g,
        "b": labels_b,
    }


def get_synced_sensor_image(world, sensor_queue, target_frame, converter, timeout=2.0):
    """
    Get sensor data matching target_frame.
    Works for RGB and semantic cameras.
    """
    while True:
        image = sensor_queue.get(timeout=timeout)

        if image.frame == target_frame:
            return converter(image), image.frame

        if image.frame < target_frame:
            continue

        # Rare case: sensor jumped ahead.
        return converter(image), image.frame


def compute_vehicle_pixel_ratio_in_bbox(
    semantic_labels,
    bbox_result,
    vehicle_label=10,
):
    """
    Compute how much of the projected bbox is actually vehicle pixels.

    semantic_labels can be:
      - single label map, or
      - dict with r/g/b maps.

    We use the max ratio across channels to handle CARLA version differences.
    """
    if bbox_result["visible"] != 1:
        return 0.0

    if isinstance(semantic_labels, dict):
        label_maps = semantic_labels
    else:
        label_maps = {"single": semantic_labels}

    best_ratio = 0.0

    for _, label_map in label_maps.items():
        h, w = label_map.shape[:2]

        x1 = int(max(0, min(w - 1, round(bbox_result["x_min"]))))
        y1 = int(max(0, min(h - 1, round(bbox_result["y_min"]))))
        x2 = int(max(0, min(w - 1, round(bbox_result["x_max"]))))
        y2 = int(max(0, min(h - 1, round(bbox_result["y_max"]))))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = label_map[y1:y2 + 1, x1:x2 + 1]
        total = crop.size

        if total <= 0:
            continue

        vehicle_pixels = np.sum(crop == int(vehicle_label))
        ratio = float(vehicle_pixels) / float(total)

        best_ratio = max(best_ratio, ratio)

    return best_ratio

def passes_quality_filter(
    bbox_result,
    semantic_labels,
    width,
    height,
    quality_cfg,
):
    """
    Return True only for clean, useful training samples.
    """
    if not quality_cfg.get("enabled", True):
        return True, {
            "passed": True,
            "reason": "filter_disabled",
            "vehicle_pixel_ratio": None,
        }

    if bbox_result["visible"] != 1:
        return False, {
            "passed": False,
            "reason": "not_visible",
            "vehicle_pixel_ratio": 0.0,
        }

    box_w = float(bbox_result["box_width"])
    box_h = float(bbox_result["box_height"])

    if box_w < float(quality_cfg.get("min_clean_box_width_px", 8)):
        return False, {
            "passed": False,
            "reason": "box_too_narrow",
            "vehicle_pixel_ratio": 0.0,
        }

    if box_h < float(quality_cfg.get("min_clean_box_height_px", 8)):
        return False, {
            "passed": False,
            "reason": "box_too_short",
            "vehicle_pixel_ratio": 0.0,
        }

    if quality_cfg.get("require_fully_inside", True):
        margin = float(quality_cfg.get("edge_margin_px", 8))

        # Use unclipped bbox to reject partially outside image samples.
        ux1 = float(bbox_result["unclipped_x_min"])
        uy1 = float(bbox_result["unclipped_y_min"])
        ux2 = float(bbox_result["unclipped_x_max"])
        uy2 = float(bbox_result["unclipped_y_max"])

        fully_inside = (
            ux1 >= margin
            and uy1 >= margin
            and ux2 <= float(width - 1 - margin)
            and uy2 <= float(height - 1 - margin)
        )

        if not fully_inside:
            return False, {
                "passed": False,
                "reason": "bbox_clipped_or_near_edge",
                "vehicle_pixel_ratio": 0.0,
            }

    min_ratio = float(quality_cfg.get("min_vehicle_pixel_ratio", 0.25))

    # If min_vehicle_pixel_ratio <= 0, disable semantic occlusion filtering.
    # This is useful because CARLA semantic label extraction may vary by version.
    if min_ratio <= 0.0:
        return True, {
            "passed": True,
            "reason": "clean_geometry_only",
            "vehicle_pixel_ratio": None,
        }

    vehicle_label = int(quality_cfg.get("vehicle_semantic_label", 10))
    vehicle_ratio = compute_vehicle_pixel_ratio_in_bbox(
        semantic_labels,
        bbox_result,
        vehicle_label=vehicle_label,
    )

    if vehicle_ratio < min_ratio:
        return False, {
            "passed": False,
            "reason": "occluded_or_not_enough_vehicle_pixels",
            "vehicle_pixel_ratio": vehicle_ratio,
        }

    return True, {
        "passed": True,
        "reason": "clean",
        "vehicle_pixel_ratio": vehicle_ratio,
    }

def main():
    args = parse_args()
    cfg = load_config(args.config)

    random.seed(args.seed)
    np.random.seed(args.seed)

    carla_cfg = cfg["carla"]
    camera_cfg = cfg["camera"]
    sampling_cfg = cfg["sampling"]
    visibility_cfg = cfg["visibility"]
    output_cfg = cfg["output"]
    quality_cfg = cfg.get("quality_filter", {"enabled": False})

    num_samples = args.num_samples or int(sampling_cfg["num_samples"])

    dataset_dir = output_cfg["dataset_dir"]
    debug_dir = os.path.join(dataset_dir, "debug_images")
    labels_path = os.path.join(dataset_dir, "labels.jsonl")
    meta_path = os.path.join(dataset_dir, "metadata.json")

    ensure_dir(dataset_dir)

    save_debug_images = bool(output_cfg.get("save_debug_images", True))
    debug_image_limit = int(output_cfg.get("debug_image_limit", 100))

    if save_debug_images:
        ensure_dir(debug_dir)

    client = carla.Client(carla_cfg["host"], int(carla_cfg["port"]))
    client.set_timeout(20.0)

    print(f"[INFO] Loading world: {carla_cfg['town']}")
    world = client.load_world(carla_cfg["town"])

    original_settings = None
    actors = []
    camera = None
    semseg_camera = None

    try:
        original_settings = set_synchronous_mode(
            world,
            float(carla_cfg.get("fixed_delta_seconds", 0.05)),
        )

        blueprint_library = world.get_blueprint_library()

        ego_bp = get_blueprint(world, "vehicle.*", preferred="vehicle.tesla.model3")
        adv_bp = get_blueprint(world, "vehicle.*", preferred="vehicle.audi.tt")

        # Use a fixed/random valid road spawn for ego.
        ego_spawn_index = carla_cfg.get("ego_spawn_index", None)
        ego_spawn = choose_spawn_transform(world, ego_spawn_index)

        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        actors.append(ego)
        ego.set_autopilot(False)
        ego.set_simulate_physics(False)

        # Spawn adversary ONCE at a valid spawn point.
        adv_spawn = get_different_spawn_transform(world, ego_spawn)
        adv = world.try_spawn_actor(adv_bp, adv_spawn)
        if adv is None:
            raise RuntimeError("Failed to spawn adversary vehicle even at a valid spawn point.")

        actors.append(adv)
        adv.set_autopilot(False)
        adv.set_simulate_physics(True)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        width = int(camera_cfg["width"])
        height = int(camera_cfg["height"])
        fov = float(camera_cfg["fov"])

        set_camera_blueprint_attributes(camera_bp, width, height, fov)

        camera_transform = carla.Transform(
            carla.Location(
                x=float(camera_cfg.get("mount_x", 1.5)),
                y=float(camera_cfg.get("mount_y", 0.0)),
                z=float(camera_cfg.get("mount_z", 1.6)),
            ),
            carla.Rotation(
                pitch=float(camera_cfg.get("pitch", 0.0)),
                yaw=float(camera_cfg.get("yaw", 0.0)),
                roll=float(camera_cfg.get("roll", 0.0)),
            ),
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=ego,
        )

        actors.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        # Semantic segmentation camera for dataset quality filtering only.
        semseg_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        set_camera_blueprint_attributes(semseg_bp, width, height, fov)

        semseg_camera = world.spawn_actor(
            semseg_bp,
            camera_transform,
            attach_to=ego,
        )

        actors.append(semseg_camera)

        semseg_queue = queue.Queue()
        semseg_camera.listen(semseg_queue.put)

        K = build_camera_intrinsics(width, height, fov)

        metadata = {
            "project": "HEPlacementModel",
            "version": "v1_straight",
            "town": carla_cfg["town"],
            "num_samples_requested": num_samples,
            "camera": {
                "width": width,
                "height": height,
                "fov": fov,
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "mount_x": float(camera_cfg.get("mount_x", 1.5)),
                "mount_y": float(camera_cfg.get("mount_y", 0.0)),
                "mount_z": float(camera_cfg.get("mount_z", 1.6)),
                "pitch": float(camera_cfg.get("pitch", 0.0)),
                "yaw": float(camera_cfg.get("yaw", 0.0)),
                "roll": float(camera_cfg.get("roll", 0.0)),
            },
            "sampling": sampling_cfg,
            "visibility": visibility_cfg,
            "ego_spawn_index": ego_spawn_index,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[INFO] Saving labels to: {labels_path}")

        visible_count = 0
        invisible_count = 0

        # Warm-up ticks.
        for _ in range(5):
            frame = world.tick()
            try:
                _ = get_synced_camera_image(world, image_queue, frame, timeout=2.0)
            except Exception:
                pass

        clear_queue(image_queue)
        clear_queue(semseg_queue)

        with open(labels_path, "w", encoding="utf-8") as f:
            sample_id = 0
            attempts = 0
            max_attempts = int(num_samples * quality_cfg.get("max_attempts_multiplier", 10))

            rejected_count = 0
            rejected_reasons = {}

            while sample_id < num_samples and attempts < max_attempts:
                attempts += 1
                rel_x, rel_z, rel_yaw = sample_relative_state(sampling_cfg)

                ego_transform = ego.get_transform()

                new_adv_transform = relative_to_world_transform(
                    world.get_map(),
                    ego_transform,
                    rel_x=rel_x,
                    rel_z=rel_z,
                    rel_yaw=rel_yaw,
                )

                if new_adv_transform is None:
                    invisible_count += 1
                    continue

                # Teleport instead of re-spawning.
                adv.set_transform(new_adv_transform)
                adv.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                adv.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

                # Very important:
                # Clear old camera frames before ticking. Otherwise debug image and bbox become offset by one sample.
                clear_queue(image_queue)
                clear_queue(semseg_queue)
                # First tick applies the new transform. Discard this image to avoid render lag.
                frame_apply = world.tick()
                try:
                    _ = get_synced_camera_image(world, image_queue, frame_apply, timeout=2.0)
                except Exception:
                    pass

                # Second tick gives a clean image with the adversary at the current sampled transform.
                clear_queue(image_queue)
                clear_queue(semseg_queue)
                frame_capture = world.tick()

                try:
                    rgb, camera_frame = get_synced_sensor_image(
                        world,
                        image_queue,
                        target_frame=frame_capture,
                        converter=image_to_numpy,
                        timeout=2.0,
                    )

                    semantic_labels, semantic_frame = get_synced_sensor_image(
                        world,
                        semseg_queue,
                        target_frame=frame_capture,
                        converter=semantic_to_labels,
                        timeout=2.0,
                    )

                except Exception:
                    invisible_count += 1
                    continue

                camera_world_transform = camera.get_transform()

                corners_world = get_bbox_world_corners(adv)
                points_2d, valid_depth = project_world_points_to_image(
                    corners_world,
                    camera_world_transform,
                    K,
                )

                bbox_result = compute_2d_bbox_from_projected_points(
                    points_2d,
                    valid_depth,
                    image_width=width,
                    image_height=height,
                    min_bbox_width_px=float(visibility_cfg.get("min_bbox_width_px", 2)),
                    min_bbox_height_px=float(visibility_cfg.get("min_bbox_height_px", 2)),
                )
                passed_quality, quality_info = passes_quality_filter(
                    bbox_result=bbox_result,
                    semantic_labels=semantic_labels,
                    width=width,
                    height=height,
                    quality_cfg=quality_cfg,
                )

                if not passed_quality:
                    rejected_count += 1
                    reason = quality_info["reason"]
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                    continue
                rel_state_actual = compute_relative_state(
                    ego_transform,
                    adv.get_transform(),
                )

                if bbox_result["visible"] == 1:
                    visible_count += 1
                else:
                    invisible_count += 1

                adv_transform_actual = adv.get_transform()

                row = {
                    "sample_id": sample_id,
                    "frame": int(camera_frame),
                    "quality": quality_info,
                    "town": carla_cfg["town"],
                    "ego": {
                        "x": float(ego_transform.location.x),
                        "y": float(ego_transform.location.y),
                        "z": float(ego_transform.location.z),
                        "pitch": float(ego_transform.rotation.pitch),
                        "yaw": float(ego_transform.rotation.yaw),
                        "roll": float(ego_transform.rotation.roll),
                    },
                    "adversary": {
                        "x": float(adv_transform_actual.location.x),
                        "y": float(adv_transform_actual.location.y),
                        "z": float(adv_transform_actual.location.z),
                        "pitch": float(adv_transform_actual.rotation.pitch),
                        "yaw": float(adv_transform_actual.rotation.yaw),
                        "roll": float(adv_transform_actual.rotation.roll),
                        "blueprint": adv_bp.id,
                    },
                    "relative_state": {
                        "rel_x": float(rel_state_actual["rel_x"]),
                        "rel_z": float(rel_state_actual["rel_z"]),
                        "rel_y": float(rel_state_actual["rel_y"]),
                        "rel_yaw": float(rel_state_actual["rel_yaw"]),
                    },
                    "camera": {
                        "width": width,
                        "height": height,
                        "fov": fov,
                        "fx": float(K[0, 0]),
                        "fy": float(K[1, 1]),
                        "cx": float(K[0, 2]),
                        "cy": float(K[1, 2]),
                        "mount_x": float(camera_cfg.get("mount_x", 1.5)),
                        "mount_y": float(camera_cfg.get("mount_y", 0.0)),
                        "mount_z": float(camera_cfg.get("mount_z", 1.6)),
                        "pitch": float(camera_cfg.get("pitch", 0.0)),
                        "yaw": float(camera_cfg.get("yaw", 0.0)),
                        "roll": float(camera_cfg.get("roll", 0.0)),
                    },
                    "target": bbox_result,
                }

                f.write(json.dumps(row) + "\n")

                if save_debug_images and sample_id < debug_image_limit:
                    text = (
                        f"id={sample_id} visible={bbox_result['visible']}\n"
                        f"rel_x={rel_state_actual['rel_x']:.2f}, "
                        f"rel_z={rel_state_actual['rel_z']:.2f}, "
                        f"rel_yaw={rel_state_actual['rel_yaw']:.1f}\n"
                        f"cx={bbox_result['center_x']:.1f}, "
                        f"bottom_y={bbox_result['bottom_y']:.1f}, "
                        f"w={bbox_result['box_width']:.1f}, "
                        f"h={bbox_result['box_height']:.1f}"
                    )

                    debug_img = draw_debug_bbox(rgb, bbox_result, text=text)
                    debug_path = os.path.join(
                        debug_dir,
                        f"sample_{sample_id:06d}.png",
                    )
                    cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                sample_id += 1
                if sample_id % 50 == 0:
                    print(
                        f"[INFO] accepted={sample_id}/{num_samples} "
                        f"attempts={attempts} "
                        f"visible={visible_count} "
                        f"rejected={rejected_count} "
                        f"last_quality={quality_info}"
                    )

        print("[DONE] Dataset collection finished.")
        print(f"[DONE] labels: {labels_path}")
        print(f"[DONE] metadata: {meta_path}")
        print(f"[DONE] visible={visible_count}, invisible={invisible_count}")
        print(f"[DONE] rejected={rejected_count}")
        print(f"[DONE] rejected_reasons={rejected_reasons}")

    finally:
        print("[INFO] Cleaning up actors and restoring world settings.")
        if semseg_camera is not None:
            try:
                semseg_camera.stop()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

        destroy_actors(actors)
        restore_world(world, original_settings)


if __name__ == "__main__":
    main()