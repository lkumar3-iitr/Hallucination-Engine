"""Independent complete sweeps with physical target and paired clean backgrounds."""
import argparse
import json
import queue

import carla
import cv2
import numpy as np

from capture_validation import as_dict, receive
from he_renderer.evaluation.evaluate import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(ROOT / "artifacts/complete_sweeps_v1"))
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--semantic-tag", type=int, required=True)
    parser.add_argument("--color", default="0,0,255")
    args = parser.parse_args()
    from pathlib import Path
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    client = carla.Client("127.0.0.1", 2000)
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
            for key, value in {"image_size_x": "1280", "image_size_y": "720", "fov": "90", "sensor_tick": "0"}.items():
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
        for yaw in (0, -60, -90):
            directory = output / f"yaw_{yaw}"
            (directory / "masks").mkdir(parents=True)
            setup = {"schema": "he_renderer_complete_passby_v1", "fps": 20, "frame_count": 125,
                     "asset_id": args.asset_id, "carla_blueprint": args.blueprint,
                     "semantic_tag": args.semantic_tag,
                     "camera": {"width": 1280, "height": 720, "fov_deg": 90},
                     "yaw_deg": yaw, "lateral_m": 3.5, "camera_height_above_actor_m": 1.55,
                     "range": "camera x=-18.6 to +18.6 m, step .3 m", "town": world.get_map().name,
                     "clean_background": "target moved to z=200; same camera and matched frame per stream"}
            (directory / "setup.json").write_text(json.dumps(setup, indent=2))
            physical_writer = cv2.VideoWriter(str(directory / "carla_reference.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280, 720))
            background_writer = cv2.VideoWriter(str(directory / "background_only.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280, 720))
            writers.extend([physical_writer, background_writer])
            if not physical_writer.isOpened() or not background_writer.isOpened():
                raise RuntimeError("Video writer failed")
            with (directory / "frames.jsonl").open("w") as log:
                for index in range(125):
                    camera = carla.Transform(carla.Location(x=-18.6+index*.3, y=3.5, z=81.55),
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
                        print(f"yaw {yaw} frame {index}/124 GT pixels {np.count_nonzero(mask)}", flush=True)
            physical_writer.release()
            background_writer.release()
            (directory / "COMPLETE.json").write_text(json.dumps({"frames": 125}))
    finally:
        for writer in writers:
            writer.release()
        for actor in reversed(owned):
            if actor.type_id.startswith("sensor."):
                actor.stop()
            actor.destroy()
        world.apply_settings(original_settings)
    (output / "COMPLETE.json").write_text(json.dumps({"sweeps": 3, "frames": 375}))
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()
