#!/usr/bin/env python3
"""
record_carla_dashcam_video.py

Record clean CARLA dashcam videos for HE scenario testing.

Purpose:
  - Generate controlled background videos from CARLA.
  - These videos will later be used by the HE temporal compositor.
  - The HE compositor will NOT use CARLA world information.

Default:
  - Loads Town10HD_Opt.
  - Spawns ego vehicle.
  - Records front dashcam RGB video.

Example:
  python record_carla_dashcam_video.py ^
    --town Town10HD_Opt ^
    --output-dir carla_recordings\\carla_straight_001 ^
    --duration-sec 10 ^
    --fps 30 ^
    --autopilot ^
    --overwrite

Fixed throttle example:
  python record_carla_dashcam_video.py ^
    --town Town10HD_Opt ^
    --output-dir carla_recordings\\carla_fixed_throttle_001 ^
    --duration-sec 10 ^
    --fps 30 ^
    --throttle 0.35 ^
    --steer 0.0 ^
    --overwrite
"""

import argparse
import json
import queue
import random
import shutil
import time
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
    parser.add_argument("--timeout", type=float, default=20.0)

    parser.add_argument("--town", default="Town10HD_Opt",
                        help="Default HD map for video generation.")

    parser.add_argument("--output-dir", default="carla_recordings/carla_dashcam_001")
    parser.add_argument("--output-video-name", default="road_video_carla_001.mp4")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--image-width", type=int, default=848)
    parser.add_argument("--image-height", type=int, default=478)
    parser.add_argument("--fov", type=float, default=70.0)

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)

    parser.add_argument("--vehicle-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--spawn-index", type=int, default=-1)

    parser.add_argument("--autopilot", action="store_true",
                        help="Use CARLA autopilot for ego vehicle.")

    parser.add_argument("--throttle", type=float, default=0.35,
                        help="Used only when --autopilot is not set.")

    parser.add_argument("--steer", type=float, default=0.0,
                        help="Used only when --autopilot is not set.")

    parser.add_argument("--brake", type=float, default=0.0,
                        help="Used only when --autopilot is not set.")

    parser.add_argument("--camera-x", type=float, default=1.6)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=1.45)
    parser.add_argument("--camera-pitch", type=float, default=-4.0)

    parser.add_argument("--save-frames", action="store_true")
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

    print(f"[Recorder] Clearing {len(actors)} vehicles/walkers/sensors...")

    for actor in actors:
        safe_destroy(actor)

    world.tick()


def setup_sync(world, fps):
    settings = world.get_settings()
    original = settings

    new_settings = carla.WorldSettings(
        no_rendering_mode=False,
        synchronous_mode=True,
        fixed_delta_seconds=1.0 / float(fps)
    )

    world.apply_settings(new_settings)

    return original


def restore_settings(world, original_settings):
    if world is not None and original_settings is not None:
        world.apply_settings(original_settings)


def flush_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def wait_for_image(q, expected_frame, timeout=5.0):
    while True:
        image = q.get(timeout=timeout)
        if image.frame >= expected_frame:
            return image


def carla_rgb_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))

    # CARLA raw is BGRA.
    bgr = arr[:, :, :3].copy()

    return bgr


def build_intrinsics(width, height, fov_deg):
    fov = np.deg2rad(float(fov_deg))
    fx = width / (2.0 * np.tan(fov / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy)
    }


# ============================================================
# Spawn
# ============================================================

def find_blueprint(bp_lib, bp_id):
    try:
        return bp_lib.find(bp_id)
    except RuntimeError:
        matches = bp_lib.filter(bp_id)
        if len(matches) > 0:
            return matches[0]

        vehicles = bp_lib.filter("vehicle.*")
        if len(vehicles) == 0:
            raise RuntimeError("No vehicle blueprints found.")

        print("[Recorder] Warning: requested vehicle not found. Using:", vehicles[0].id)
        return vehicles[0]


def spawn_ego_vehicle(world, bp_lib, args):
    vehicle_bp = find_blueprint(bp_lib, args.vehicle_blueprint)

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "ego_he_video_recorder")

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", "0,0,255")

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points found.")

    if args.spawn_index >= 0:
        candidates = [spawn_points[min(args.spawn_index, len(spawn_points) - 1)]]
    else:
        candidates = list(spawn_points)
        random.shuffle(candidates)

    for sp in candidates:
        ego = world.try_spawn_actor(vehicle_bp, sp)

        if ego is not None:
            print("[Recorder] Spawned ego:", vehicle_bp.id)
            print("[Recorder] Spawn transform:", sp)
            return ego, vehicle_bp.id, sp

    raise RuntimeError("Could not spawn ego vehicle.")


def spawn_camera(world, bp_lib, ego, args):
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.image_width))
    cam_bp.set_attribute("image_size_y", str(args.image_height))
    cam_bp.set_attribute("fov", str(args.fov))
    cam_bp.set_attribute("sensor_tick", "0.0")

    cam_tf = carla.Transform(
        carla.Location(
            x=args.camera_x,
            y=args.camera_y,
            z=args.camera_z
        ),
        carla.Rotation(
            pitch=args.camera_pitch,
            yaw=0.0,
            roll=0.0
        )
    )

    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    print("[Recorder] Spawned RGB camera.")
    print("[Recorder] Camera transform relative to ego:", cam_tf)

    return camera, cam_tf


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"

    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = output_dir / args.output_video_name

    client = None
    world = None
    original_settings = None
    actors = []

    image_queue = queue.Queue()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        print("[Recorder] Loading HD town:", args.town)
        world = client.load_world(args.town)

        original_settings = setup_sync(world, args.fps)

        if not args.no_clear_existing_actors:
            clear_existing_dynamic_actors(world)

        bp_lib = world.get_blueprint_library()

        ego, ego_bp_id, spawn_tf = spawn_ego_vehicle(world, bp_lib, args)
        actors.append(ego)

        camera, camera_tf = spawn_camera(world, bp_lib, ego, args)
        actors.append(camera)

        camera.listen(lambda image: image_queue.put(image))

        # Allow sensors to initialize.
        for _ in range(20):
            world.tick()
            flush_queue(image_queue)

        if args.autopilot:
            ego.set_autopilot(True)
            print("[Recorder] Ego driving mode: autopilot")
        else:
            ego.set_autopilot(False)
            print("[Recorder] Ego driving mode: fixed control")
            print("[Recorder] throttle:", args.throttle, "steer:", args.steer, "brake:", args.brake)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            float(args.fps),
            (args.image_width, args.image_height)
        )

        total_frames = int(round(args.duration_sec * args.fps))

        frame_log = []

        print("[Recorder] Recording frames:", total_frames)
        print("[Recorder] Output video:", output_video_path)

        for frame_idx in range(total_frames):
            if not args.autopilot:
                control = carla.VehicleControl(
                    throttle=float(args.throttle),
                    steer=float(args.steer),
                    brake=float(args.brake),
                    hand_brake=False,
                    reverse=False
                )
                ego.apply_control(control)

            frame = world.tick()
            image = wait_for_image(image_queue, frame)

            bgr = carla_rgb_to_bgr(image)

            writer.write(bgr)

            if args.save_frames:
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.png"), bgr)

            ego_tf = ego.get_transform()
            ego_vel = ego.get_velocity()

            frame_log.append({
                "frame_idx": frame_idx,
                "carla_frame": int(frame),
                "ego_location": {
                    "x": float(ego_tf.location.x),
                    "y": float(ego_tf.location.y),
                    "z": float(ego_tf.location.z)
                },
                "ego_rotation": {
                    "pitch": float(ego_tf.rotation.pitch),
                    "yaw": float(ego_tf.rotation.yaw),
                    "roll": float(ego_tf.rotation.roll)
                },
                "ego_velocity": {
                    "x": float(ego_vel.x),
                    "y": float(ego_vel.y),
                    "z": float(ego_vel.z)
                }
            })

            if frame_idx % 30 == 0:
                print(f"[Recorder] frame {frame_idx}/{total_frames}")

        writer.release()

        camera_info = {
            "town": args.town,
            "output_video": str(output_video_path),
            "image_width": args.image_width,
            "image_height": args.image_height,
            "fps": args.fps,
            "duration_sec": args.duration_sec,
            "fov": args.fov,
            "intrinsics": build_intrinsics(args.image_width, args.image_height, args.fov),
            "ego_vehicle_blueprint": ego_bp_id,
            "spawn_index": args.spawn_index,
            "spawn_transform": {
                "location": {
                    "x": float(spawn_tf.location.x),
                    "y": float(spawn_tf.location.y),
                    "z": float(spawn_tf.location.z)
                },
                "rotation": {
                    "pitch": float(spawn_tf.rotation.pitch),
                    "yaw": float(spawn_tf.rotation.yaw),
                    "roll": float(spawn_tf.rotation.roll)
                }
            },
            "camera_transform_relative_to_ego": {
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
            },
            "driving": {
                "autopilot": bool(args.autopilot),
                "throttle": float(args.throttle),
                "steer": float(args.steer),
                "brake": float(args.brake)
            },
            "frames": frame_log
        }

        with open(output_dir / "camera_info.json", "w", encoding="utf-8") as f:
            json.dump(camera_info, f, indent=2)

        print("\n[Recorder] Done.")
        print("[Recorder] Video      :", output_video_path)
        print("[Recorder] Camera info:", output_dir / "camera_info.json")

    finally:
        print("[Recorder] Cleaning up...")

        for actor in actors:
            safe_destroy(actor)

        restore_settings(world, original_settings)

        print("[Recorder] Done cleanup.")


if __name__ == "__main__":
    main()