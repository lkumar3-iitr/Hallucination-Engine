"""Independent complete sweeps with physical target and paired clean backgrounds."""
import argparse
import json
import queue

import carla
import cv2
import numpy as np

try:
    from .capture_validation import as_dict, receive
except ImportError:
    from capture_validation import as_dict, receive
from he_renderer.evaluation.evaluate import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(ROOT / "artifacts/complete_sweeps_v1"))
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--semantic-tag", type=int, required=True)
    parser.add_argument("--color", default="0,0,255")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--camera-height-above-actor-m", type=float, default=1.55)
    parser.add_argument("--lateral-m", type=float, default=3.5)
    parser.add_argument("--x-start-m", type=float, default=-18.6)
    parser.add_argument("--x-stop-m", type=float, default=18.6)
    parser.add_argument("--x-step-m", type=float, default=0.3)
    parser.add_argument("--camera-yaws", type=float, nargs="+", default=[0.0, -60.0, -90.0])
    args = parser.parse_args()
    from pathlib import Path
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.width < 1 or args.height < 1 or args.x_step_m <= 0:
        parser.error("width, height, and x-step-m must be positive")
    if args.x_stop_m < args.x_start_m:
        parser.error("x-stop-m must be at least x-start-m")
    x_values = np.arange(
        args.x_start_m,
        args.x_stop_m + args.x_step_m * 0.5,
        args.x_step_m,
    )
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(20)
    world = client.get_world()
    if any(a.type_id.startswith(("vehicle.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("Use an idle CARLA server")
    original_settings = world.get_settings()
    owned, writers = [], []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = .05
        world.apply_settings(settings)
        bp = world.get_blueprint_library().find(args.blueprint)
        if bp.has_attribute("color"):
            bp.set_attribute("color", args.color)
        target_tf = carla.Transform(carla.Location(x=0, y=0, z=80))
        hidden_tf = carla.Transform(carla.Location(x=0, y=0, z=200))
        target = world.spawn_actor(bp, target_tf)
        owned.append(target)
        if hasattr(target, "set_simulate_physics"):
            target.set_simulate_physics(False)
        sensors, queues = [], []
        for kind in ("rgb", "instance_segmentation"):
            bp = world.get_blueprint_library().find("sensor.camera."+kind)
            for key, value in {
                "image_size_x": str(args.width),
                "image_size_y": str(args.height),
                "fov": str(args.fov),
                "sensor_tick": "0",
            }.items():
                bp.set_attribute(key, value)
            sensor = world.spawn_actor(bp, carla.Transform(carla.Location(z=85)))
            owned.append(sensor)
            q = queue.Queue()
            sensor.listen(q.put)
            sensors.append(sensor)
            queues.append(q)
        def settle():
            for _ in range(3):
                frame = world.tick()
                images = [receive(q, frame) for q in queues]
            return frame, images
        instance_key = None
        for yaw in args.camera_yaws:
            yaw_name = f"{yaw:g}".replace("-", "m").replace(".", "p")
            directory = output / f"yaw_{yaw_name}"
            (directory / "masks").mkdir(parents=True)
            setup = {"schema": "he_renderer_complete_passby_v1", "fps": 20,
                     "frame_count": int(len(x_values)),
                     "asset_id": args.asset_id, "carla_blueprint": args.blueprint,
                     "semantic_tag": args.semantic_tag,
                     "camera": {"width": args.width, "height": args.height, "fov_deg": args.fov},
                     "yaw_deg": yaw, "lateral_m": args.lateral_m,
                     "camera_height_above_actor_m": args.camera_height_above_actor_m,
                     "range": {"x_start_m": args.x_start_m, "x_stop_m": args.x_stop_m,
                               "x_step_m": args.x_step_m}, "town": world.get_map().name,
                     "clean_background": "target moved to z=200; same camera and matched frame per stream"}
            (directory / "setup.json").write_text(json.dumps(setup, indent=2))
            frame_size = (args.width, args.height)
            physical_writer = cv2.VideoWriter(str(directory / "carla_reference.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, frame_size)
            background_writer = cv2.VideoWriter(str(directory / "background_only.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, frame_size)
            writers.extend([physical_writer, background_writer])
            if not physical_writer.isOpened() or not background_writer.isOpened():
                raise RuntimeError("Video writer failed")
            with (directory / "frames.jsonl").open("w") as log:
                for index, camera_x in enumerate(x_values):
                    camera = carla.Transform(carla.Location(
                        x=float(camera_x), y=args.lateral_m,
                        z=80.0 + args.camera_height_above_actor_m),
                                             carla.Rotation(yaw=yaw))
                    for sensor in sensors:
                        sensor.set_transform(camera)
                    target.set_transform(target_tf)
                    if instance_key is None:
                        # Let newly spawned vehicle render assets finish initialization.
                        for _ in range(10):
                            settle()
                    physical_frame, (rgb, annotation) = settle()
                    tf = as_dict(target.get_transform())
                    camera_dict = as_dict(sensors[0].get_transform())
                    ids = annotation[:, :, 0].astype(np.uint16)+256*annotation[:, :, 1].astype(np.uint16)
                    target_semantic = annotation[:, :, 2] == args.semantic_tag
                    if instance_key is None:
                        keys, counts = np.unique(ids[target_semantic], return_counts=True)
                        if not len(keys) or counts.max()/counts.sum() < .99:
                            raise RuntimeError("Cannot calibrate target key")
                        instance_key = int(keys[np.argmax(counts)])
                    mask = (target_semantic & (ids == instance_key)).astype(np.uint8)*255
                    target.set_transform(hidden_tf)
                    background_frame, (background, hidden_annotation) = settle()
                    hidden_ids = hidden_annotation[:, :, 0].astype(np.uint16)+256*hidden_annotation[:, :, 1].astype(np.uint16)
                    if np.any((hidden_annotation[:, :, 2] == args.semantic_tag) &
                              (hidden_ids == instance_key)):
                        raise RuntimeError("Target remained in clean background")
                    name = f"frame_{index:06d}.png"
                    cv2.imwrite(str(directory / "masks" / name), mask)
                    physical_writer.write(rgb[:, :, :3])
                    background_writer.write(background[:, :, :3])
                    log.write(json.dumps({"scenario_frame": index, "mask_path": "masks/"+name,
                                          "actor_transform": tf, "camera_transform": camera_dict,
                                          "carla_frame": physical_frame, "background_frame": background_frame,
                                          "target_instance_key": instance_key})+"\n")
                    log.flush()
                    if index % 25 == 0:
                        print(f"yaw {yaw} frame {index}/{len(x_values)-1} GT pixels {np.count_nonzero(mask)}", flush=True)
            physical_writer.release()
            background_writer.release()
            (directory / "COMPLETE.json").write_text(json.dumps({"frames": len(x_values)}))
    finally:
        for writer in writers:
            writer.release()
        for actor in reversed(owned):
            if actor.type_id.startswith("sensor."):
                actor.stop()
            actor.destroy()
        world.apply_settings(original_settings)
    (output / "COMPLETE.json").write_text(json.dumps({
        "sweeps": len(args.camera_yaws),
        "frames": len(args.camera_yaws) * len(x_values),
    }))
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()
