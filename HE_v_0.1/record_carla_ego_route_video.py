import argparse
import queue
import random
import shutil
import time
from pathlib import Path
import json
import carla
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town10HD_Opt")

    parser.add_argument("--output", type=str, default="recordings/ego_autopilot_spawn0_900.mp4")
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)

    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=1.6)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)

    parser.add_argument("--spawn-index", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ego-vehicle-filter", type=str, default="vehicle.tesla.model3")

    parser.add_argument("--autopilot-speed-diff", type=float, default=0.0,
                        help="Traffic Manager speed difference percentage. Negative is faster, positive is slower.")

    parser.add_argument("--ignore-lights-percentage", type=float, default=0.0)
    parser.add_argument("--ignore-signs-percentage", type=float, default=0.0)

    parser.add_argument("--traffic", action="store_true",
                        help="Spawn background vehicles and walkers.")
    parser.add_argument("--num-vehicles", type=int, default=30)
    parser.add_argument("--num-walkers", type=int, default=40)

    parser.add_argument("--tm-port", type=int, default=8000)

    parser.add_argument("--weather", type=str, default="clear",
                        choices=["clear", "cloudy", "wet", "sunset", "soft_rain", "hard_rain"])

    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frames-dir", type=str, default=None)
    parser.add_argument(
        "--save-ego-pose",
        action="store_true",
        help="Save ego pose, velocity, and camera pose per recorded frame as JSONL.",
    )

    parser.add_argument(
        "--ego-pose-output",
        type=str,
        default=None,
        help="Optional output path for ego pose JSONL. Default: same video name with _ego_pose.jsonl.",
    )
    parser.add_argument("--warmup-frames", type=int, default=30)

    return parser.parse_args()

def carla_transform_to_dict(transform):
    loc = transform.location
    rot = transform.rotation

    return {
        "x": float(loc.x),
        "y": float(loc.y),
        "z": float(loc.z),
        "pitch": float(rot.pitch),
        "yaw": float(rot.yaw),
        "roll": float(rot.roll),
    }


def carla_vector_to_dict(vec):
    return {
        "x": float(vec.x),
        "y": float(vec.y),
        "z": float(vec.z),
    }


def get_speed_mps(vec):
    return float(np.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z))


def image_to_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return rgb.copy()


def get_synced_image(image_queue, target_frame, timeout=3.0):
    while True:
        image = image_queue.get(timeout=timeout)

        if image.frame == target_frame:
            return image_to_rgb(image)

        if image.frame < target_frame:
            continue

        # If the sensor skipped ahead, return latest available image.
        return image_to_rgb(image)


def clear_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def destroy_actors(actors):
    for actor in actors:
        if actor is not None and actor.is_alive:
            try:
                actor.destroy()
            except Exception:
                pass


def apply_weather(world, weather_name):
    if weather_name == "clear":
        weather = carla.WeatherParameters.ClearNoon
    elif weather_name == "cloudy":
        weather = carla.WeatherParameters.CloudyNoon
    elif weather_name == "wet":
        weather = carla.WeatherParameters.WetNoon
    elif weather_name == "sunset":
        weather = carla.WeatherParameters.ClearSunset
    elif weather_name == "soft_rain":
        weather = carla.WeatherParameters.SoftRainNoon
    elif weather_name == "hard_rain":
        weather = carla.WeatherParameters.HardRainNoon
    else:
        weather = carla.WeatherParameters.ClearNoon

    world.set_weather(weather)


def get_vehicle_blueprint(bp_lib, vehicle_filter):
    candidates = bp_lib.filter(vehicle_filter)

    if len(candidates) == 0:
        print(f"[WARN] No vehicle found for filter={vehicle_filter}. Falling back to vehicle.tesla.model3")
        candidates = bp_lib.filter("vehicle.tesla.model3")

    if len(candidates) == 0:
        raise RuntimeError("Could not find ego vehicle blueprint.")

    bp = candidates[0]

    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")

    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", colors[0])

    return bp


def spawn_background_vehicles(world, bp_lib, traffic_manager, spawn_points, ego_spawn_index, num_vehicles, seed):
    rng = random.Random(seed)
    vehicle_actors = []

    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute("number_of_wheels")) == 4]

    if not vehicle_bps:
        print("[WARN] No background vehicle blueprints found.")
        return vehicle_actors

    indices = list(range(len(spawn_points)))
    rng.shuffle(indices)

    for idx in indices:
        if len(vehicle_actors) >= num_vehicles:
            break

        if idx == ego_spawn_index:
            continue

        bp = rng.choice(vehicle_bps)

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")

        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", rng.choice(colors))

        actor = world.try_spawn_actor(bp, spawn_points[idx])
        if actor is None:
            continue

        vehicle_actors.append(actor)
        actor.set_autopilot(True, traffic_manager.get_port())

        traffic_manager.auto_lane_change(actor, True)
        traffic_manager.distance_to_leading_vehicle(actor, 2.5)

    return vehicle_actors


def spawn_background_walkers(world, bp_lib, num_walkers, seed):
    rng = random.Random(seed)
    walker_actors = []
    controller_actors = []

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")

    if not walker_bps:
        print("[WARN] No walker blueprints found.")
        return walker_actors, controller_actors

    spawn_locations = []

    for _ in range(num_walkers * 3):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_locations.append(carla.Transform(loc))

        if len(spawn_locations) >= num_walkers:
            break

    for transform in spawn_locations:
        bp = rng.choice(walker_bps)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(bp, transform)
        if walker is None:
            continue

        walker_actors.append(walker)

        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is not None:
            controller_actors.append(controller)

    world.tick()

    for controller in controller_actors:
        try:
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(1.0 + rng.random() * 1.2)
        except Exception:
            pass

    return walker_actors, controller_actors


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.ego_pose_output is None:
        ego_pose_output_path = output_path.with_suffix("")
        ego_pose_output_path = Path(str(ego_pose_output_path) + "_ego_pose.jsonl")
    else:
        ego_pose_output_path = Path(args.ego_pose_output)

    if args.save_ego_pose:
        ego_pose_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.frames_dir is None:
        frames_dir = output_path.with_suffix("")
        frames_dir = Path(str(frames_dir) + "_frames")
    else:
        frames_dir = Path(args.frames_dir)

    if args.save_frames:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        ensure_dir(frames_dir)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    print(f"[INFO] Loading world: {args.town}")
    world = client.load_world(args.town)

    old_settings = world.get_settings()

    actors = []
    walker_controllers = []
    ego = None
    camera = None
    writer = None
    ego_pose_file = None

    try:
        apply_weather(world, args.weather)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(args.fps)
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found.")

        spawn_index = int(args.spawn_index) % len(spawn_points)
        spawn = spawn_points[spawn_index]

        ego_bp = get_vehicle_blueprint(bp_lib, args.ego_vehicle_filter)

        ego = world.try_spawn_actor(ego_bp, spawn)

        if ego is None:
            raise RuntimeError(f"Failed to spawn ego vehicle at spawn-index={spawn_index}")

        actors.append(ego)

        ego.set_autopilot(True, traffic_manager.get_port())

        traffic_manager.auto_lane_change(ego, True)
        traffic_manager.distance_to_leading_vehicle(ego, 2.5)
        traffic_manager.vehicle_percentage_speed_difference(ego, float(args.autopilot_speed_diff))
        traffic_manager.ignore_lights_percentage(ego, float(args.ignore_lights_percentage))
        traffic_manager.ignore_signs_percentage(ego, float(args.ignore_signs_percentage))

        if args.traffic:
            bg_vehicles = spawn_background_vehicles(
                world=world,
                bp_lib=bp_lib,
                traffic_manager=traffic_manager,
                spawn_points=spawn_points,
                ego_spawn_index=spawn_index,
                num_vehicles=int(args.num_vehicles),
                seed=args.seed + 1,
            )
            actors.extend(bg_vehicles)
            print(f"[INFO] Spawned background vehicles: {len(bg_vehicles)}")

            walkers, controllers = spawn_background_walkers(
                world=world,
                bp_lib=bp_lib,
                num_walkers=int(args.num_walkers),
                seed=args.seed + 2,
            )
            actors.extend(walkers)
            walker_controllers.extend(controllers)
            actors.extend(controllers)
            print(f"[INFO] Spawned walkers: {len(walkers)}")
            print(f"[INFO] Spawned walker controllers: {len(controllers)}")

        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("fov", str(args.fov))

        camera_transform = carla.Transform(
            carla.Location(
                x=float(args.camera_x),
                y=float(args.camera_y),
                z=float(args.camera_z),
            ),
            carla.Rotation(
                pitch=float(args.camera_pitch),
                yaw=float(args.camera_yaw),
                roll=float(args.camera_roll),
            ),
        )

        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego)
        actors.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(args.fps),
            (int(args.width), int(args.height)),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_path}")

        if args.save_ego_pose:
            ego_pose_file = open(ego_pose_output_path, "w", encoding="utf-8")
            print("[INFO] Ego pose JSONL:", ego_pose_output_path)
        
        print("[INFO] Recording started")
        print("[INFO] Output:", output_path)
        print("[INFO] Frames:", args.frames)
        print("[INFO] FPS:", args.fps)
        print("[INFO] Town:", args.town)
        print("[INFO] Spawn index:", spawn_index)
        print("[INFO] Ego vehicle:", ego.type_id)
        print("[INFO] Autopilot speed diff:", args.autopilot_speed_diff)
        print("[INFO] Traffic:", args.traffic)
        print("[INFO] Weather:", args.weather)
        print(
            "[INFO] Camera: "
            f"{args.width}x{args.height}, fov={args.fov}, "
            f"x={args.camera_x}, y={args.camera_y}, z={args.camera_z}, "
            f"pitch={args.camera_pitch}, yaw={args.camera_yaw}, roll={args.camera_roll}"
        )

        # Warmup so autopilot starts moving and camera queue stabilizes.
        for _ in range(int(args.warmup_frames)):
            clear_queue(image_queue)
            frame = world.tick()
            try:
                _ = get_synced_image(image_queue, frame, timeout=3.0)
            except Exception:
                pass

        for i in range(int(args.frames)):
            clear_queue(image_queue)
            frame = world.tick()
            rgb = get_synced_image(image_queue, frame, timeout=3.0)

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            if args.save_ego_pose and ego_pose_file is not None:
                ego_tf = ego.get_transform()
                ego_vel = ego.get_velocity()
                ego_acc = ego.get_acceleration()
                camera_tf = camera.get_transform()

                pose_record = {
                    "recorded_frame_idx": int(i),
                    "carla_frame": int(frame),
                    "t_s": float(i) / float(args.fps),
                    "ego_transform": carla_transform_to_dict(ego_tf),
                    "ego_velocity": carla_vector_to_dict(ego_vel),
                    "ego_acceleration": carla_vector_to_dict(ego_acc),
                    "ego_speed_mps": get_speed_mps(ego_vel),
                    "camera_transform": carla_transform_to_dict(camera_tf),
                    "camera_mount": {
                        "x": float(args.camera_x),
                        "y": float(args.camera_y),
                        "z": float(args.camera_z),
                        "pitch": float(args.camera_pitch),
                        "yaw": float(args.camera_yaw),
                        "roll": float(args.camera_roll),
                    },
                    "camera": {
                        "width": int(args.width),
                        "height": int(args.height),
                        "fov": float(args.fov),
                    },
                }

                ego_pose_file.write(json.dumps(pose_record) + "\n")
            if args.save_frames:
                cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), bgr)

            if i % 50 == 0:
                velocity = ego.get_velocity()
                speed = 3.6 * np.sqrt(
                    velocity.x * velocity.x +
                    velocity.y * velocity.y +
                    velocity.z * velocity.z
                )
                print(f"[INFO] frame {i}/{args.frames} | ego speed {speed:.2f} km/h")

        print("[DONE] Recording complete:", output_path)

    finally:
        print("[INFO] Cleaning up")

        if writer is not None:
            writer.release()
        if ego_pose_file is not None:
            ego_pose_file.close()
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

        for controller in walker_controllers:
            if controller is not None and controller.is_alive:
                try:
                    controller.stop()
                except Exception:
                    pass

        # Disable autopilot before destroy.
        for actor in actors:
            if actor is not None and actor.is_alive and actor.type_id.startswith("vehicle."):
                try:
                    actor.set_autopilot(False)
                except Exception:
                    pass

        destroy_actors(actors)

        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass

        world.apply_settings(old_settings)


if __name__ == "__main__":
    main()