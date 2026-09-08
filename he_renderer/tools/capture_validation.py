"""Capture independent CARLA poses; restore world settings and destroy only owned actors."""
from __future__ import annotations

import argparse
import itertools
import json
import queue
from pathlib import Path

import carla
import cv2
import numpy as np

from he_renderer.evaluation.evaluate import DEFAULT_BANK, ROOT


def as_dict(tf):
    return {**{k: float(getattr(tf.location, k)) for k in ("x", "y", "z")},
            **{k: float(getattr(tf.rotation, k)) for k in ("pitch", "yaw", "roll")}}


def receive(q, frame):
    while True:
        image = q.get(timeout=15)
        if image.frame == frame:
            return np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4).copy()
        if image.frame > frame:
            raise RuntimeError("Sensor frame advanced beyond requested capture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/carla_independent_v1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite capture: {args.output}")
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(15)
    world = client.get_world()
    if any(a.type_id.startswith(("vehicle.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("CARLA has existing vehicles/sensors; use an idle server")
    original = world.get_settings()
    owned = []
    writer = None
    args.output.mkdir(parents=True)
    (args.output / "masks").mkdir()
    (args.output / "rgb").mkdir()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = .05
        world.apply_settings(settings)
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("color", "0,0,255")
        actor = world.spawn_actor(bp, carla.Transform(carla.Location(z=80)))
        owned.append(actor)
        actor.set_simulate_physics(False)
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
        # Actor get_transform reads the client snapshot, which is stale until a tick.
        tick = world.tick()
        for q in queues:
            receive(q, tick)
        center = actor.get_transform().transform(actor.bounding_box.location)
        center = np.array([center.x, center.y, center.z])
        poses = list(itertools.product([13, 57, 101, 147, 193, 237, 281, 327],
                                       [4, 7, 14], [7, 16], [-18, 18]))
        setup = {"schema": "he_renderer_independent_selection_validation_v1", "fps": 20,
                 "frame_count": len(poses), "town": world.get_map().name,
                 "camera": {"width": 1280, "height": 720, "fov_deg": 90},
                 "mask_semantics": "dominant vehicle instance key in isolated single-vehicle capture; key logged per frame",
                 "purpose": "Held-out poses; never used to reconstruct hull or fit selector",
                 "blueprint": actor.type_id, "actor_id": actor.id}
        (args.output / "setup.json").write_text(json.dumps(setup, indent=2))
        writer = cv2.VideoWriter(str(args.output / "carla_reference.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError("Video writer failed")
        with (args.output / "frames.jsonl").open("w") as log:
            target_instance_key = None
            for index, (bearing, distance, elevation, offaxis) in enumerate(poses):
                az, el = np.deg2rad([bearing, elevation])
                xyz = center + distance*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
                camera = carla.Transform(carla.Location(*map(float, xyz)),
                                         carla.Rotation(pitch=-elevation, yaw=bearing-180+offaxis))
                for sensor in sensors:
                    sensor.set_transform(camera)
                # Settle transform propagation and exposure; consume matched sensor frames.
                for _ in range(3):
                    tick = world.tick()
                    images = [receive(q, tick) for q in queues]
                rgb, instance = images
                ids = instance[:, :, 1].astype(np.uint16)*256 + instance[:, :, 0].astype(np.uint16)
                vehicle_pixels = instance[:, :, 2] == 14
                keys, counts = np.unique(ids[vehicle_pixels], return_counts=True)
                if not len(keys):
                    cv2.imwrite(str(args.output / f"failed_instance_{index}.png"), instance)
                    cv2.imwrite(str(args.output / f"failed_rgb_{index}.png"), rgb)
                    raise RuntimeError(f"No vehicle pixels at pose {index}")
                if target_instance_key is None:
                    if counts.max()/counts.sum() < .999:
                        raise RuntimeError("Initial view cannot calibrate a unique target instance")
                    target_instance_key = int(keys[np.argmax(counts)])
                instance_key = target_instance_key
                purity = float(counts[keys == instance_key].sum()/counts.sum())
                mask = (vehicle_pixels & (ids == instance_key)).astype(np.uint8)*255
                if not mask.any():
                    raise RuntimeError(f"Calibrated target instance missing at pose {index}")
                name = f"frame_{index:06d}.png"
                cv2.imwrite(str(args.output / "masks" / name), mask)
                cv2.imwrite(str(args.output / "rgb" / name), rgb[:, :, :3])
                writer.write(rgb[:, :, :3])
                row = {"scenario_frame": index, "carla_frame": tick, "mask_path": "masks/"+name,
                       "instance_key": instance_key, "instance_purity": purity,
                       "actor_transform": as_dict(actor.get_transform()),
                       "camera_transform": as_dict(sensors[0].get_transform()),
                       "design_pose": {"bearing": bearing, "distance": distance,
                                       "elevation": elevation, "offaxis": offaxis}}
                log.write(json.dumps(row)+"\n")
                log.flush()
                if index % 12 == 0:
                    print(f"capture {index+1}/{len(poses)}, target pixels={np.count_nonzero(mask)}", flush=True)
    finally:
        if writer is not None:
            writer.release()
        for item in reversed(owned):
            if item.type_id.startswith("sensor."):
                item.stop()
            item.destroy()
        world.apply_settings(original)
    (args.output / "COMPLETE.json").write_text(json.dumps({"frames": len(poses)}))
    print(f"Captured {len(poses)} independent poses: {args.output}", flush=True)


if __name__ == "__main__":
    main()
