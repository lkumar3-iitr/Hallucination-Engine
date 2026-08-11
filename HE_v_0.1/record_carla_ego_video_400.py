import argparse
import queue
from pathlib import Path

import carla
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town10HD_Opt")

    parser.add_argument("--output", type=str, default="recordings/ego_straight_400.mp4")
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--fps", type=float, default=20.0)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)

    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=1.6)
    parser.add_argument("--camera-pitch", type=float, default=0.0)

    parser.add_argument("--ego-speed", type=float, default=0.0)
    parser.add_argument("--spawn-index", type=int, default=0)

    return parser.parse_args()


def image_to_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return rgb.copy()


def get_synced_image(image_queue, target_frame, timeout=2.0):
    while True:
        image = image_queue.get(timeout=timeout)

        if image.frame == target_frame:
            return image_to_rgb(image)

        if image.frame < target_frame:
            continue

        return image_to_rgb(image)


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


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print(f"[INFO] Loading world: {args.town}")
    world = client.load_world(args.town)

    old_settings = world.get_settings()
    actors = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(args.fps)
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()

        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        camera_bp = bp_lib.find("sensor.camera.rgb")

        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("fov", str(args.fov))

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found.")

        spawn = spawn_points[int(args.spawn_index) % len(spawn_points)]

        ego = world.try_spawn_actor(vehicle_bp, spawn)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        actors.append(ego)
        ego.set_autopilot(False)

        camera_transform = carla.Transform(
            carla.Location(
                x=float(args.camera_x),
                y=float(args.camera_y),
                z=float(args.camera_z),
            ),
            carla.Rotation(
                pitch=float(args.camera_pitch),
                yaw=0.0,
                roll=0.0,
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

        print("[INFO] Recording started")
        print("[INFO] Output:", output_path)
        print("[INFO] Frames:", args.frames)
        print("[INFO] Camera: "
              f"{args.width}x{args.height}, fov={args.fov}, "
              f"x={args.camera_x}, y={args.camera_y}, z={args.camera_z}, pitch={args.camera_pitch}")

        # Warmup
        for _ in range(10):
            clear_queue(image_queue)
            frame = world.tick()
            try:
                _ = get_synced_image(image_queue, frame, timeout=2.0)
            except Exception:
                pass

        for i in range(int(args.frames)):
            if args.ego_speed > 0:
                ego.apply_control(
                    carla.VehicleControl(
                        throttle=min(0.5, args.ego_speed / 20.0),
                        steer=0.0,
                        brake=0.0,
                    )
                )
            else:
                ego.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                    )
                )

            clear_queue(image_queue)
            frame = world.tick()
            rgb = get_synced_image(image_queue, frame, timeout=2.0)

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

            if i % 50 == 0:
                print(f"[INFO] frame {i}/{args.frames}")

        writer.release()

        print("[DONE] Recording complete:", output_path)

    finally:
        print("[INFO] Cleaning up")
        for actor in actors:
            if actor is not None and actor.type_id.startswith("sensor."):
                try:
                    actor.stop()
                except Exception:
                    pass

        destroy_actors(actors)
        world.apply_settings(old_settings)


if __name__ == "__main__":
    main()