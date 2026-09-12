"""Capture level, horizontally aimed actor views into an isolated ASTRA bank."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import queue
from pathlib import Path

import carla
import cv2
import numpy as np


def receive(q, frame):
    while True:
        image = q.get(timeout=30)
        if image.frame == frame:
            return np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4).copy()
        if image.frame > frame:
            raise RuntimeError("Sensor advanced beyond requested frame")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--forward-min", type=float, default=-12)
    parser.add_argument("--forward-max", type=float, default=12)
    parser.add_argument("--forward-step", type=float, default=.5)
    parser.add_argument("--right", type=float, nargs="+", default=[-4.5, -3.5, 3.5, 4.5])
    parser.add_argument("--up", type=float, nargs="+", default=[-.1726465702, .5749115236])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--blueprint", default="vehicle.mitsubishi.fusorosa")
    parser.add_argument("--semantic-tag", type=int, default=16)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fov", type=float, default=150.)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or not 0 < args.fov < 180:
        parser.error("Capture dimensions must be positive and FOV must be between 0 and 180")
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if root not in output.parents:
        parser.error("Output must remain inside astra")
    config = dict(forward_min=args.forward_min, forward_max=args.forward_max,
                  forward_step=args.forward_step, right=args.right, up=args.up,
                  width=args.width, height=args.height, fov=args.fov, aim="bbox_angular_midpoint_v1",
                  blueprint=args.blueprint)
    if args.semantic_tag != 16:
        config["semantic_tag"] = args.semantic_tag
    if args.forward_step <= 0 or args.forward_max < args.forward_min:
        parser.error("Invalid forward range")
    if output.exists():
        if not args.resume or json.loads((output/"config.json").read_text()) != config:
            raise ValueError("Existing output requires matching --resume configuration")
    else:
        (output/"rgba").mkdir(parents=True)
        (output/"config.json").write_text(json.dumps(config, indent=2))
    rows = []
    csv_path = output/"view_matrix.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    if any(not (output/row["rgba_relpath"]).is_file() for row in rows):
        raise ValueError("Resume CSV references missing images; preserve and repair the bank first")
    completed = {int(row["view_index"]) for row in rows}
    if len(completed) != len(rows):
        raise ValueError("Resume CSV contains duplicate capture indices")
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(30)
    world = client.get_world()
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("Use an idle CARLA server")
    original, owned = world.get_settings(), []
    try:
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, .05
        world.apply_settings(settings)
        bp = world.get_blueprint_library().find(config["blueprint"])
        if bp.has_attribute("color"):
            bp.set_attribute("color", "0,0,255")
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "true")
        actor = world.spawn_actor(bp, carla.Transform(carla.Location(z=80)))
        owned.append(actor)
        actor.set_simulate_physics(False)
        bbox = actor.bounding_box
        center = np.array([bbox.location.x, bbox.location.y, bbox.location.z])
        extents = np.array([bbox.extent.x, bbox.extent.y, bbox.extent.z])
        (output/"geometry.json").write_text(json.dumps({"center": center.tolist(), "extents": extents.tolist()}))
        sensors, queues = [], []
        for kind in ("rgb", "instance_segmentation"):
            bp = world.get_blueprint_library().find("sensor.camera."+kind)
            for key, value in {"image_size_x": config["width"], "image_size_y": config["height"],
                               "fov": config["fov"], "sensor_tick": 0}.items():
                bp.set_attribute(key, str(value))
            if kind == "rgb" and bp.has_attribute("motion_blur_intensity"):
                bp.set_attribute("motion_blur_intensity", "0")
            sensor = world.spawn_actor(bp, carla.Transform(carla.Location(z=85)))
            owned.append(sensor)
            q = queue.Queue()
            sensor.listen(q.put)
            sensors.append(sensor)
            queues.append(q)
        for _ in range(12):
            frame = world.tick()
            for q in queues:
                receive(q, frame)
        values = np.arange(args.forward_min, args.forward_max+args.forward_step*.1, args.forward_step)
        nodes = list(itertools.product(values, args.right, args.up))
        target_key = None
        for index, (forward, right, up) in enumerate(nodes):
            if index in completed:
                continue
            offset = center-np.array([forward, right, up])
            if np.all(np.abs(offset-center) < extents+.1):
                raise ValueError("Capture camera intersects expanded actor bbox")
            corners = np.array(list(itertools.product((-1, 1), repeat=3))) * extents + center
            rays = corners-offset
            center_angle = math.atan2(right, forward)
            angles = center_angle + (np.arctan2(rays[:, 1], rays[:, 0])-center_angle+math.pi) % (2*math.pi)-math.pi
            yaw = float((angles.min()+angles.max())/2)
            if math.degrees(angles.max()-angles.min()) >= config["fov"]-2:
                raise ValueError(f"Insufficient horizontal capture coverage at node {index}")
            camera = carla.Transform(carla.Location(x=float(offset[0]), y=float(offset[1]), z=float(80+offset[2])),
                                     carla.Rotation(yaw=math.degrees(yaw)))
            for sensor in sensors:
                sensor.set_transform(camera)
            for _ in range(3):
                frame = world.tick()
                images = [receive(q, frame) for q in queues]
            rgb, annotation = images
            mask = annotation[:, :, 2] == args.semantic_tag
            # Pin the target once: static map vehicles can share its semantic tag.
            ids = annotation[:, :, 0].astype(np.uint16)+256*annotation[:, :, 1].astype(np.uint16)
            if target_key is None:
                world_center = np.asarray(actor.get_transform().get_matrix()) @ np.r_[center, 1.]
                p = np.asarray(sensors[0].get_transform().get_inverse_matrix()) @ world_center
                focal = config["width"]/(2*math.tan(math.radians(config["fov"])/2))
                u = int(round(config["width"]/2+focal*p[1]/p[0]))
                v = int(round(config["height"]/2-focal*p[2]/p[0]))
                window = np.zeros_like(mask)
                window[max(0,v-8):v+9, max(0,u-8):u+9] = True
                keys, counts = np.unique(ids[mask & window], return_counts=True)
                if not len(keys) or counts.max()/counts.sum() < .95:
                    raise ValueError(f"Ambiguous initial instance calibration at node {index}")
                target_key = int(keys[np.argmax(counts)])
            mask &= ids == target_key
            if mask.sum() < 100:
                raise ValueError(f"Ambiguous/empty actor mask at node {index}")
            ys, xs = np.where(mask)
            if xs.min() == 0 or xs.max() == config["width"]-1 or ys.min() == 0 or ys.max() == config["height"]-1:
                raise ValueError(f"Capture clipped at node {index}; increase field of view")
            x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
            rgba = np.dstack((rgb[y0:y1, x0:x1, :3], mask[y0:y1, x0:x1].astype(np.uint8)*255))
            name = f"rgba/view_{index:05d}.png"
            if not cv2.imwrite(str(output/name), rgba):
                raise IOError(name)
            focal = config["width"]/(2*math.tan(math.radians(config["fov"])/2))
            row = dict(view_index=index, rgba_relpath=name, close_forward_m=float(forward),
                       close_right_m=float(right), close_target_up_m=float(up), camera_fx_px=focal,
                       camera_fy_px=focal, camera_cx_px=config["width"]/2, camera_cy_px=config["height"]/2,
                       crop_x1_px=int(x0), crop_y1_px=int(y0), alpha_area_px=int(mask.sum()))
            for prefix, tf in (("actor", actor.get_transform()), ("camera", sensors[0].get_transform())):
                for axis in "xyz":
                    row[f"{prefix}_location_{axis}_m"] = float(getattr(tf.location, axis))
                for axis in ("yaw", "pitch", "roll"):
                    row[f"{prefix}_{axis}_deg"] = float(getattr(tf.rotation, axis))
            first_row = not rows
            rows.append(row)
            with csv_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                if first_row:
                    writer.writeheader()
                writer.writerow(row)
            if index % 10 == 0:
                print(f"capture {index+1}/{len(nodes)} pixels={mask.sum()}", flush=True)
        (output/"COMPLETE.json").write_text(json.dumps({"frames": len(nodes)}))
    finally:
        for actor in reversed(owned):
            try:
                if actor.type_id.startswith("sensor."):
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        world.apply_settings(original)


if __name__ == "__main__":
    main()
