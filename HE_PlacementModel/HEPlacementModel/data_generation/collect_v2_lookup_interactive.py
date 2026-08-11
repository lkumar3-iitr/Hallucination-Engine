import argparse
import json
import math
import os
import queue
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np
import pygame

from projection_utils import (
    build_camera_intrinsics,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town10HD_Opt")

    parser.add_argument("--output-dir", type=str, default="dataset/v2_lookup_interactive")

    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)

    parser.add_argument("--camera-mount-x", type=float, default=1.5)
    parser.add_argument("--camera-mount-y", type=float, default=0.0)
    parser.add_argument("--camera-mount-z", type=float, default=1.6)
    parser.add_argument("--camera-pitch", type=float, default=0.0)

    # Small lateral offset only.
    parser.add_argument("--rel-x-min", type=float, default=-3.0)
    parser.add_argument("--rel-x-max", type=float, default=3.0)
    parser.add_argument("--rel-x-step", type=float, default=0.5)

    parser.add_argument("--rel-z-min", type=float, default=0.0)
    parser.add_argument("--rel-z-max", type=float, default=100.0)
    parser.add_argument("--rel-z-step", type=float, default=0.5)

    parser.add_argument("--yaw-min", type=float, default=0.0)
    parser.add_argument("--yaw-max", type=float, default=359.0)
    parser.add_argument("--yaw-step", type=float, default=1.0)

    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)

    parser.add_argument("--debug-images", action="store_true")
    parser.add_argument("--debug-image-limit", type=int, default=100)

    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def frange(start, stop, step):
    values = []
    x = float(start)
    stop = float(stop)
    step = float(step)

    while x <= stop + 1e-6:
        values.append(round(x, 6))
        x += step

    return values


def set_sync_mode(world, fixed_delta_seconds):
    old_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    return old_settings


def restore_settings(world, old_settings):
    if old_settings is not None:
        world.apply_settings(old_settings)


def get_blueprint(world, preferred_id, fallback_filter="vehicle.*"):
    bp_lib = world.get_blueprint_library()

    try:
        return bp_lib.find(preferred_id)
    except Exception:
        candidates = bp_lib.filter(fallback_filter)
        if not candidates:
            raise RuntimeError(f"No blueprint found for {fallback_filter}")
        return candidates[0]


def set_camera_attrs(camera_bp, width, height, fov):
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", str(fov))


def image_to_numpy(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return rgb.copy()

def make_camera_intrinsic(width, height, fov_deg):
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    k = np.identity(3)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    return k


def project_world_point_to_image_pair_style(world_point, world_to_camera, k):
    """
    Same projection convention as record_carla_he_pair.py.

    CARLA camera coords:
      x = forward
      y = right
      z = up

    Image projection:
      [u, v, w] = K * [y, -z, x]
    """
    p = np.array([world_point.x, world_point.y, world_point.z, 1.0])
    p_cam = world_to_camera @ p

    depth = float(p_cam[0])
    if depth <= 0.05:
        return None

    p_img = k @ np.array([p_cam[1], -p_cam[2], p_cam[0]])
    u = float(p_img[0] / p_img[2])
    v = float(p_img[1] / p_img[2])

    return u, v, depth


def compute_actor_2d_bbox_pair_style(actor, camera, width, height, fov):
    """
    Same bbox definition as record_carla_he_pair.py, but converted to the
    lookup collector's target format.
    """
    k = make_camera_intrinsic(width, height, fov)
    world_to_camera = np.array(camera.get_transform().get_inverse_matrix())

    bb = actor.bounding_box
    vertices = bb.get_world_vertices(actor.get_transform())

    points = []
    depths = []

    for v in vertices:
        proj = project_world_point_to_image_pair_style(v, world_to_camera, k)
        if proj is None:
            continue

        u, vv, d = proj
        points.append((u, vv))
        depths.append(d)

    if len(points) < 2:
        return {
            "visible": 0,
            "center_x": 0.0,
            "bottom_y": 0.0,
            "box_width": 0.0,
            "box_height": 0.0,
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 0.0,
            "y_max": 0.0,
            "unclipped_x_min": None,
            "unclipped_y_min": None,
            "unclipped_x_max": None,
            "unclipped_y_max": None,
            "reason": "bbox_behind_camera",
        }

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    raw_x1 = float(min(xs))
    raw_y1 = float(min(ys))
    raw_x2 = float(max(xs))
    raw_y2 = float(max(ys))

    x1 = max(0.0, raw_x1)
    y1 = max(0.0, raw_y1)
    x2 = min(float(width - 1), raw_x2)
    y2 = min(float(height - 1), raw_y2)

    if x2 <= 0 or x1 >= width or y2 <= 0 or y1 >= height:
        return {
            "visible": 0,
            "center_x": 0.0,
            "bottom_y": 0.0,
            "box_width": 0.0,
            "box_height": 0.0,
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 0.0,
            "y_max": 0.0,
            "unclipped_x_min": raw_x1,
            "unclipped_y_min": raw_y1,
            "unclipped_x_max": raw_x2,
            "unclipped_y_max": raw_y2,
            "reason": "bbox_outside_image",
        }

    if x2 <= x1 or y2 <= y1:
        return {
            "visible": 0,
            "center_x": 0.0,
            "bottom_y": 0.0,
            "box_width": 0.0,
            "box_height": 0.0,
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 0.0,
            "y_max": 0.0,
            "unclipped_x_min": raw_x1,
            "unclipped_y_min": raw_y1,
            "unclipped_x_max": raw_x2,
            "unclipped_y_max": raw_y2,
            "reason": "bbox_invalid",
        }

    return {
        "visible": 1,
        "center_x": float((x1 + x2) / 2.0),
        "bottom_y": float(y2),
        "box_width": float(x2 - x1),
        "box_height": float(y2 - y1),
        "x_min": float(x1),
        "y_min": float(y1),
        "x_max": float(x2),
        "y_max": float(y2),
        "unclipped_x_min": raw_x1,
        "unclipped_y_min": raw_y1,
        "unclipped_x_max": raw_x2,
        "unclipped_y_max": raw_y2,
        "depth_m": float(np.mean(depths)),
    }

def get_synced_image(world, image_queue, target_frame, timeout=2.0):
    while True:
        image = image_queue.get(timeout=timeout)

        if image.frame == target_frame:
            return image_to_numpy(image), image.frame

        if image.frame < target_frame:
            continue

        return image_to_numpy(image), image.frame


def clear_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def destroy_actors(actors):
    for actor in actors:
        if actor is not None and actor.is_alive:
            try:
                actor.destroy()
            except Exception:
                pass


def normalize_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def relative_to_world_transform(world, ego_transform, rel_x, rel_z, rel_yaw, z_lift=0.05):
    """
    Ego convention:
      rel_x positive = right
      rel_z positive = forward
      rel_yaw = adversary yaw relative to ego yaw

    Uses ego-relative x/z for geometry, then snaps z to nearest road waypoint
    to avoid floating/sinking vehicles.
    """
    ego_loc = ego_transform.location
    ego_yaw_deg = ego_transform.rotation.yaw
    ego_yaw = math.radians(ego_yaw_deg)

    forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float32)
    right = np.array(
        [math.cos(ego_yaw + math.pi / 2.0), math.sin(ego_yaw + math.pi / 2.0)],
        dtype=np.float32,
    )

    ego_xy = np.array([ego_loc.x, ego_loc.y], dtype=np.float32)
    adv_xy = ego_xy + float(rel_z) * forward + float(rel_x) * right

    adv_yaw = normalize_angle_deg(ego_yaw_deg + float(rel_yaw))

    loc = carla.Location(
        x=float(adv_xy[0]),
        y=float(adv_xy[1]),
        z=float(ego_loc.z + z_lift),
    )

    try:
        wp = world.get_map().get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is not None:
            loc.z = float(wp.transform.location.z + z_lift)
    except Exception:
        pass

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=float(adv_yaw),
            roll=0.0,
        ),
    )
def camera_relative_to_world_transform(world, camera_transform, rel_x, rel_z, rel_yaw, z_lift=0.05):
    """
    Camera-relative lookup-table placement.

    This is the correct convention for HE runtime placement:
      rel_x   = lateral/right offset from camera
      rel_z   = forward depth from camera
      rel_yaw = adversary yaw relative to camera yaw

    The actor x/y position is generated from the camera transform.
    The actor z height is snapped to the CARLA road waypoint.
    """
    cam_loc = camera_transform.location
    cam_yaw_deg = camera_transform.rotation.yaw
    cam_yaw = math.radians(cam_yaw_deg)

    forward = np.array(
        [math.cos(cam_yaw), math.sin(cam_yaw)],
        dtype=np.float32,
    )

    right = np.array(
        [math.cos(cam_yaw + math.pi / 2.0), math.sin(cam_yaw + math.pi / 2.0)],
        dtype=np.float32,
    )

    cam_xy = np.array([cam_loc.x, cam_loc.y], dtype=np.float32)
    adv_xy = cam_xy + float(rel_z) * forward + float(rel_x) * right

    adv_yaw = normalize_angle_deg(cam_yaw_deg + float(rel_yaw))

    loc = carla.Location(
        x=float(adv_xy[0]),
        y=float(adv_xy[1]),
        z=float(cam_loc.z),
    )

    try:
        wp = world.get_map().get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is not None:
            loc.z = float(wp.transform.location.z + z_lift)
    except Exception:
        loc.z = float(cam_loc.z)

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=float(adv_yaw),
            roll=0.0,
        ),
    )

def compute_camera_relative_state(camera_transform, adv_transform):
    """
    Compute adversary state in camera-local coordinates.

    Output convention:
      rel_x = lateral/right relative to camera
      rel_z = forward/depth relative to camera
      rel_y = vertical relative to camera
      rel_yaw = adversary yaw relative to camera yaw
    """
    cam_loc = camera_transform.location
    adv_loc = adv_transform.location

    dx = adv_loc.x - cam_loc.x
    dy = adv_loc.y - cam_loc.y
    dz = adv_loc.z - cam_loc.z

    cam_yaw = math.radians(camera_transform.rotation.yaw)

    forward_x = math.cos(cam_yaw)
    forward_y = math.sin(cam_yaw)

    right_x = -math.sin(cam_yaw)
    right_y = math.cos(cam_yaw)

    rel_z = dx * forward_x + dy * forward_y
    rel_x = dx * right_x + dy * right_y
    rel_y = dz

    rel_yaw = normalize_angle_deg(
        adv_transform.rotation.yaw - camera_transform.rotation.yaw
    )

    return {
        "rel_x": float(rel_x),
        "rel_z": float(rel_z),
        "rel_y": float(rel_y),
        "rel_yaw": float(rel_yaw),
    }

def draw_debug_bbox(rgb, bbox_result, text):
    img = rgb.copy()

    if bbox_result["visible"] == 1:
        x1 = int(round(bbox_result["x_min"]))
        y1 = int(round(bbox_result["y_min"]))
        x2 = int(round(bbox_result["x_max"]))
        y2 = int(round(bbox_result["y_max"]))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx = int(round(bbox_result["center_x"]))
        by = int(round(bbox_result["bottom_y"]))
        cv2.circle(img, (cx, by), 4, (0, 0, 255), -1)

    y = 24
    for line in text.split("\n"):
        cv2.putText(
            img,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24

    return img


def manual_place_ego(world, ego, spectator):
    """
    User controls ego with pygame:
      W/S = throttle/brake/reverse
      A/D = steer
      SPACE = brake
      ENTER = freeze placement and start dataset generation
      ESC = quit
    """
    pygame.init()
    pygame.display.set_caption("HEPlacementModel v2: place ego, press ENTER to generate")
    screen = pygame.display.set_mode((720, 240))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    reverse = False
    print()
    print("=" * 80)
    print("[MANUAL CONTROL]")
    print("Use W/A/S/D to place ego vehicle on a clean straight road.")
    print("SPACE = brake")
    print("R = toggle reverse")
    print("ENTER = lock ego pose and start dataset generation")
    print("ESC = quit")
    print("=" * 80)
    print()

    while True:
        throttle = 0.0
        brake = 0.0
        steer = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                    world.tick()
                    pygame.quit()
                    print("[INFO] ENTER pressed. Ego pose locked.")
                    return ego.get_transform()

                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise KeyboardInterrupt

                if event.key == pygame.K_r:
                    reverse = not reverse
                    print(f"[INFO] reverse = {reverse}")

        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            throttle = 0.45

        if keys[pygame.K_s]:
            brake = 0.6

        if keys[pygame.K_SPACE]:
            brake = 1.0

        if keys[pygame.K_a]:
            steer = -0.45

        if keys[pygame.K_d]:
            steer = 0.45

        ego.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=steer,
                reverse=reverse,
            )
        )

        frame = world.tick()

        # Spectator follows ego from behind.
        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        yaw = math.radians(ego_tf.rotation.yaw)

        behind = carla.Location(
            x=ego_loc.x - 8.0 * math.cos(yaw),
            y=ego_loc.y - 8.0 * math.sin(yaw),
            z=ego_loc.z + 5.0,
        )

        spectator.set_transform(
            carla.Transform(
                behind,
                carla.Rotation(
                    pitch=-25.0,
                    yaw=ego_tf.rotation.yaw,
                    roll=0.0,
                ),
            )
        )

        screen.fill((20, 20, 20))

        lines = [
            "W/A/S/D: drive ego to clean road     ENTER: start generation",
            "SPACE: brake     R: reverse toggle     ESC: quit",
            f"reverse={reverse}  frame={frame}",
            f"x={ego_loc.x:.2f}, y={ego_loc.y:.2f}, z={ego_loc.z:.2f}, yaw={ego_tf.rotation.yaw:.2f}",
        ]

        y = 20
        for line in lines:
            surf = font.render(line, True, (230, 230, 230))
            screen.blit(surf, (20, y))
            y += 32

        pygame.display.flip()
        clock.tick(30)

def add_skip(skip_reasons, reason):
    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug_images"
    ensure_dir(output_dir)

    if args.debug_images:
        ensure_dir(debug_dir)

    labels_path = output_dir / "labels.jsonl"
    metadata_path = output_dir / "metadata.json"

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print(f"[INFO] Loading world: {args.town}")
    world = client.load_world(args.town)

    old_settings = None
    actors = []
    camera = None

    try:
        old_settings = set_sync_mode(world, args.fixed_delta_seconds)

        blueprint_library = world.get_blueprint_library()

        ego_bp = get_blueprint(world, "vehicle.tesla.model3")
        adv_bp = get_blueprint(world, "vehicle.audi.tt")

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found.")

        ego_spawn = spawn_points[0]
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        actors.append(ego)
        ego.set_autopilot(False)

        spectator = world.get_spectator()

        # Let user drive ego to a clean place.
        ego_transform_locked = manual_place_ego(world, ego, spectator)

        # Freeze ego.
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        ego.set_simulate_physics(False)
        ego.set_transform(ego_transform_locked)

        # Attach RGB camera after ego pose is fixed.
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        set_camera_attrs(camera_bp, args.image_width, args.image_height, args.fov)

        camera_transform = carla.Transform(
            carla.Location(
                x=args.camera_mount_x,
                y=args.camera_mount_y,
                z=args.camera_mount_z,
            ),
            carla.Rotation(
                pitch=args.camera_pitch,
                yaw=0.0,
                roll=0.0,
            ),
        )

        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego)
        actors.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # Spawn adversary once, then teleport it for every grid point.
        # Spawn adversary once at a valid CARLA spawn point, then teleport it for every grid point.
        adv = None

        for sp in spawn_points:
            # Avoid spawning directly on top of ego.
            dist = sp.location.distance(ego_transform_locked.location)

            if dist < 20.0:
                continue

            adv = world.try_spawn_actor(adv_bp, sp)

            if adv is not None:
                print(f"[INFO] Adversary spawned at valid spawn point, distance from ego = {dist:.2f} m")
                break

        if adv is None:
            # Last fallback: try all spawn points without distance check.
            for sp in spawn_points:
                adv = world.try_spawn_actor(adv_bp, sp)
                if adv is not None:
                    print("[WARN] Adversary spawned using fallback spawn point.")
                    break

        if adv is None:
            raise RuntimeError("Failed to spawn adversary vehicle at any valid CARLA spawn point.")

        actors.append(adv)
        adv.set_autopilot(False)
        adv.set_simulate_physics(False)

        K = build_camera_intrinsics(args.image_width, args.image_height, args.fov)

        rel_x_values = frange(args.rel_x_min, args.rel_x_max, args.rel_x_step)
        rel_z_values = frange(args.rel_z_min, args.rel_z_max, args.rel_z_step)
        yaw_values = frange(args.yaw_min, args.yaw_max, args.yaw_step)

        total = len(rel_x_values) * len(rel_z_values) * len(yaw_values)

        metadata = {
            "project": "HEPlacementModel",
            "version": "v2_lookup_interactive",
            "description": "Deterministic CARLA-calibrated bbox lookup dataset. User manually places ego first.",
            "town": args.town,
            "ego_locked_transform": {
                "x": float(ego_transform_locked.location.x),
                "y": float(ego_transform_locked.location.y),
                "z": float(ego_transform_locked.location.z),
                "pitch": float(ego_transform_locked.rotation.pitch),
                "yaw": float(ego_transform_locked.rotation.yaw),
                "roll": float(ego_transform_locked.rotation.roll),
            },
            "camera": {
                "width": args.image_width,
                "height": args.image_height,
                "fov": args.fov,
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "mount_x": args.camera_mount_x,
                "mount_y": args.camera_mount_y,
                "mount_z": args.camera_mount_z,
                "pitch": args.camera_pitch,
                "yaw": 0.0,
                "roll": 0.0,
            },
            "grid": {
                "rel_x_values": rel_x_values,
                "rel_z_values": rel_z_values,
                "yaw_values": yaw_values,
                "total_grid_points": total,
            },
            "convention": {
                "rel_x": "lateral offset in camera frame, positive right",
                "rel_z": "forward distance/depth in camera frame, positive forward",
                "rel_yaw": "adversary yaw relative to camera yaw, degrees",
            },
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print()
        print("=" * 80)
        print("[DATASET GENERATION START]")
        print(f"[INFO] Output labels: {labels_path}")
        print(f"[INFO] Total grid points: {total}")
        print(f"[INFO] rel_x count: {len(rel_x_values)}")
        print(f"[INFO] rel_z count: {len(rel_z_values)}")
        print(f"[INFO] yaw count: {len(yaw_values)}")
        print("=" * 80)
        print()

        # Camera warmup.
        for _ in range(5):
            frame = world.tick()
            clear_queue(image_queue)

        sample_id = 0
        visible_count = 0
        invisible_count = 0
        skipped_count = 0
        skip_reasons = {}

        with open(labels_path, "w", encoding="utf-8") as fout:
            for rel_z in rel_z_values:
                for rel_x in rel_x_values:
                    for rel_yaw in yaw_values:
                        try:
                            camera_world_transform_for_placement = camera.get_transform()

                            adv_tf = camera_relative_to_world_transform(
                                world,
                                camera_world_transform_for_placement,
                                rel_x=rel_x,
                                rel_z=rel_z,
                                rel_yaw=rel_yaw,
                                z_lift=0.05,
                            )

                            if adv_tf is None:
                                skipped_count += 1
                                add_skip(skip_reasons, "transform_none")
                                continue

                            adv.set_transform(adv_tf)
                            adv.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                            adv.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

                            # Apply transform.
                            clear_queue(image_queue)
                            world.tick()

                            # Capture frame after transform is applied.
                            clear_queue(image_queue)
                            frame_capture = world.tick()

                            rgb = None
                            camera_frame = int(frame_capture)

                            if args.debug_images and sample_id < args.debug_image_limit:
                                try:
                                    rgb, camera_frame = get_synced_image(
                                        world,
                                        image_queue,
                                        frame_capture,
                                        timeout=2.0,
                                    )
                                except Exception:
                                    rgb = None
                                    camera_frame = int(frame_capture)

                            camera_world_transform = camera.get_transform()
                            adv_transform_actual = adv.get_transform()

                            camera_rel_state = compute_camera_relative_state(
                                camera_transform=camera_world_transform,
                                adv_transform=adv_transform_actual,
                            )
                            bbox_result = compute_actor_2d_bbox_pair_style(
                                actor=adv,
                                camera=camera,
                                width=args.image_width,
                                height=args.image_height,
                                fov=args.fov,
                            )

                        except Exception as e:
                            skipped_count += 1
                            add_skip(skip_reasons, type(e).__name__)
                            continue

                        if bbox_result["visible"] == 1:
                            visible_count += 1
                        else:
                            invisible_count += 1

                        row = {
                            "sample_id": sample_id,
                            "frame": camera_frame,
                            "relative_state": {
                                "rel_x": float(camera_rel_state["rel_x"]),
                                "rel_z": float(camera_rel_state["rel_z"]),
                                "rel_y": float(camera_rel_state["rel_y"]),
                                "rel_yaw": float(camera_rel_state["rel_yaw"]),
                            },
                            "requested_camera_relative_state": {
                                "rel_x": float(rel_x),
                                "rel_z": float(rel_z),
                                "rel_yaw": float(rel_yaw),
                            },
                            "camera": metadata["camera"],
                            "target": {
                                "center_x": float(bbox_result["center_x"]),
                                "bottom_y": float(bbox_result["bottom_y"]),
                                "box_width": float(bbox_result["box_width"]),
                                "box_height": float(bbox_result["box_height"]),
                                "visible": int(bbox_result["visible"]),
                                "x_min": float(bbox_result["x_min"]),
                                "y_min": float(bbox_result["y_min"]),
                                "x_max": float(bbox_result["x_max"]),
                                "y_max": float(bbox_result["y_max"]),
                                "unclipped_x_min": bbox_result["unclipped_x_min"],
                                "unclipped_y_min": bbox_result["unclipped_y_min"],
                                "unclipped_x_max": bbox_result["unclipped_x_max"],
                                "unclipped_y_max": bbox_result["unclipped_y_max"],
                            },
                        }

                        fout.write(json.dumps(row) + "\n")

                        if args.debug_images and rgb is not None and sample_id < args.debug_image_limit:
                            text = (
                                f"id={sample_id}\n"
                                f"rel_x={rel_x:.2f}, rel_z={rel_z:.2f}, yaw={rel_yaw:.1f}\n"
                                f"cx={bbox_result['center_x']:.1f}, by={bbox_result['bottom_y']:.1f}, "
                                f"w={bbox_result['box_width']:.1f}, h={bbox_result['box_height']:.1f}"
                            )
                            dbg = draw_debug_bbox(rgb, bbox_result, text)
                            cv2.imwrite(
                                str(debug_dir / f"sample_{sample_id:06d}.png"),
                                cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR),
                            )

                        if sample_id % 1000 == 0:
                            print(
                                f"[INFO] sample={sample_id}/{total} "
                                f"rel_z={rel_z:.1f} rel_x={rel_x:.1f} yaw={rel_yaw:.1f} "
                                f"visible={visible_count} invisible={invisible_count} skipped={skipped_count}"
                            )

                        sample_id += 1

        print()
        print("[DONE] v2 lookup dataset generation complete.")
        print(f"[DONE] labels: {labels_path}")
        print(f"[DONE] metadata: {metadata_path}")
        print(f"[DONE] total_written={sample_id}, visible={visible_count}, invisible={invisible_count}, skipped={skipped_count}")
        print(f"[DONE] skip_reasons={skip_reasons}")

    finally:
        print("[INFO] Cleaning up.")
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

        destroy_actors(actors)
        restore_settings(world, old_settings)


if __name__ == "__main__":
    main()