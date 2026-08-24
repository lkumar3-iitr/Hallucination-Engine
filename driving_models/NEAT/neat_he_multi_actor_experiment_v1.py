"""
neat_he_multi_actor_experiment_v1.py

HE-only multi-actor closed-loop smoke test for pretrained NEAT.

Purpose
-------
Validate that the same model-independent ScenarioGenerator runtime and
HEMultiActorCompositorV1 already used by TCP can drive NEAT without
changing NEAT itself.

NEAT remains native:
    front : 400x300, FOV 100, x=1.3, y=0.0, z=2.3, yaw   0 deg
    left  : 400x300, FOV 100, x=1.3, y=0.0, z=2.3, yaw -60 deg
    right : 400x300, FOV 100, x=1.3, y=0.0, z=2.3, yaw +60 deg

No physical scenario adversaries are spawned in this V1 runner.
"""

from __future__ import annotations

import argparse
import csv
import queue
import sys
from pathlib import Path

import carla
import cv2
import numpy as np
import torch

from neat_carla_0915_closed_loop import (
    HE_ROOT,
    DEFAULT_NEAT_ROOT,
    DEFAULT_CHECKPOINT,
    import_carla_agents,
    load_neat,
    carla_image_to_rgb,
    make_camera,
    make_gnss,
    make_imu,
    get_sensor_frame,
    distance_2d,
    build_route,
    closest_route_index,
    route_length,
    build_neat_gps_plan,
    get_neat_navigation,
    get_speed_mps,
    run_neat,
)

COMMON_DIR = HE_ROOT / "driving_models" / "common"

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from carla_ego_initialization import canonicalize_ego_start
from he_asset_registry_v1 import DEFAULT_MANIFEST
from he_multi_actor_compositor_v1 import HEMultiActorCompositorV1
from scenario_execution_runtime_v1 import (
    ExecutionWorldOrigin,
    load_execution_runtime,
)

DEFAULT_RESOLVED = (
    HE_ROOT
    / "ScenarioGenerator"
    / "outputs"
    / "v2_resolved"
    / "multi_actor_smoke_001.resolved_v2.json"
)

DEFAULT_OUTPUT_ROOT = (
    HE_ROOT
    / "driving_models"
    / "NEAT"
    / "outputs"
    / "neat_he_multi_actor_v1"
)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_execution_origin(carla_map, ego0_tf):
    waypoint = carla_map.get_waypoint(
        ego0_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if waypoint is None:
        raise RuntimeError(
            "Could not resolve driving waypoint for canonical ego start."
        )

    return ExecutionWorldOrigin(
        x_m=float(ego0_tf.location.x),
        y_m=float(ego0_tf.location.y),
        z_m=float(waypoint.transform.location.z),
        yaw_deg=float(ego0_tf.rotation.yaw),
    )


def make_bootstrap_result():
    return {
        "steer": 0.0,
        "throttle": 0.0,
        "brake": 0.0,
        "red_light_occ": 0,
        "metadata": {
            "desired_speed": float("nan"),
            "angle": float("nan"),
            "angle_last": float("nan"),
            "angle_target": float("nan"),
            "angle_final": float("nan"),
        },
    }


def crop_neat_rgb(rgb, crop=256):
    h, w = rgb.shape[:2]
    y1 = h // 2 - crop // 2
    x1 = w // 2 - crop // 2
    return rgb[y1:y1 + crop, x1:x1 + crop].copy()


def make_video_writer(path, fps):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (768, 256),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")

    return writer


def write_neat_input_frame(writer, front_rgb, left_rgb, right_rgb):
    composite = np.concatenate(
        [
            crop_neat_rgb(front_rgb),
            crop_neat_rgb(left_rgb),
            crop_neat_rgb(right_rgb),
        ],
        axis=1,
    )
    writer.write(composite[:, :, ::-1])


def rendered_rows(composite):
    return [row for row in composite.actor_results if row.rendered]


def nearest_rendered_depth(composite):
    rows = rendered_rows(composite)
    if not rows:
        return None
    return min(float(row.distance_forward_m) for row in rows)


def rendered_actor_ids(composite):
    return [row.actor_id for row in rendered_rows(composite)]


def command_label(command):
    name = getattr(command, "name", None)
    if name is not None:
        return str(name)
    return str(command)


def command_value(command):
    value = getattr(command, "value", None)
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    return float("nan")


def scalar_float(value, default=float("nan")):
    if value is None:
        return float(default)

    if torch.is_tensor(value):
        try:
            return float(value.detach().cpu().reshape(-1)[0].item())
        except Exception:
            return float(default)

    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])
    except Exception:
        try:
            return float(value)
        except Exception:
            return float(default)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "HE-only multi-actor NEAT closed-loop smoke test. "
            "NEAT is unchanged; all scenario actors are virtual HE actors "
            "rendered independently into NEAT's native cameras."
        )
    )

    parser.add_argument("--resolved", default=str(DEFAULT_RESOLVED))
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--distance-selection-mode",
        choices=["linear", "log", "inverse_depth"],
        default="linear",
    )
    parser.add_argument("--he-bottom-y-offset-px", type=float, default=0.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=-1)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--spawn-index", type=int, default=10)
    parser.add_argument("--destination-index", type=int, default=-1)
    parser.add_argument("--desired-route-distance-m", type=float, default=120.0)
    parser.add_argument("--route-sampling-resolution", type=float, default=2.0)
    parser.add_argument("--carla-pythonapi", default=None)
    parser.add_argument("--physics-settle-ticks", type=int, default=30)
    parser.add_argument("--canonical-hold-ticks", type=int, default=5)
    parser.add_argument("--destination-tolerance-m", type=float, default=3.0)
    parser.add_argument("--max-route-deviation-m", type=float, default=8.0)
    parser.add_argument("--deviation-patience-frames", type=int, default=10)

    parser.add_argument("--neat-root", default=str(DEFAULT_NEAT_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--debug-every", type=int, default=20)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    args = parser.parse_args()

    resolved_path = Path(args.resolved).resolve()
    asset_root = Path(args.asset_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    if device.type == "cuda":
        print("[GPU]", torch.cuda.get_device_name(0))

    GlobalRoutePlanner = import_carla_agents(args.carla_pythonapi)

    (
        net,
        config,
        RoutePlanner,
        plan_grid,
        light_grid,
    ) = load_neat(
        args.neat_root,
        args.checkpoint,
        device,
    )

    print("[NEAT] cameras:", config.num_camera)
    print("[NEAT] seq_len:", config.seq_len)
    print("[NEAT] pred_len:", config.pred_len)

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(20.0)

    print("[CARLA] loading:", args.town)
    world = client.load_world(args.town)
    original_settings = world.get_settings()

    actors = []
    csv_fp = None
    video_writer = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 20.0
        world.apply_settings(settings)

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()

        if not spawn_points:
            raise RuntimeError("No CARLA spawn points.")

        spawn_idx = int(args.spawn_index) % len(spawn_points)

        grp = GlobalRoutePlanner(
            carla_map,
            float(args.route_sampling_resolution),
        )

        route, destination_idx = build_route(
            grp,
            spawn_points,
            spawn_idx,
            int(args.destination_index),
            float(args.desired_route_distance_m),
        )

        destination = route[-1][0].transform.location

        print("[route] start:", spawn_idx)
        print("[route] destination:", destination_idx)
        print("[route] points:", len(route))
        print("[route] length:", f"{route_length(route):.1f} m")

        ego_bp = world.get_blueprint_library().filter(
            "vehicle.tesla.model3"
        )[0]

        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "hero")

        ego = world.try_spawn_actor(
            ego_bp,
            spawn_points[spawn_idx],
        )

        if ego is None:
            raise RuntimeError("Could not spawn ego.")

        actors.append(ego)

        gps_plan = build_neat_gps_plan(carla_map, route)
        route_planner = RoutePlanner(4.0, 50.0)
        route_planner.set_route(gps_plan, gps=True)

        print("[NEAT planner] RoutePlanner(4.0, 50.0)")

        camera_front = make_camera(world, ego, yaw=0.0)
        camera_left = make_camera(world, ego, yaw=-60.0)
        camera_right = make_camera(world, ego, yaw=60.0)
        gnss = make_gnss(world, ego)
        imu = make_imu(world, ego)

        actors.extend(
            [
                camera_front,
                camera_left,
                camera_right,
                gnss,
                imu,
            ]
        )

        q_front = queue.Queue()
        q_left = queue.Queue()
        q_right = queue.Queue()
        q_gnss = queue.Queue()
        q_imu = queue.Queue()

        camera_front.listen(q_front.put)
        camera_left.listen(q_left.put)
        camera_right.listen(q_right.put)
        gnss.listen(q_gnss.put)
        imu.listen(q_imu.put)

        ego0_tf, _ego0_state = canonicalize_ego_start(
            world=world,
            ego=ego,
            nominal_spawn_tf=spawn_points[spawn_idx],
            settle_ticks=int(args.physics_settle_ticks),
            hold_ticks=int(args.canonical_hold_ticks),
        )

        print("[init] EXPERIMENT START BARRIER")

        origin = build_execution_origin(
            carla_map=carla_map,
            ego0_tf=ego0_tf,
        )

        runtime = load_execution_runtime(
            resolved_json=str(resolved_path),
            asset_root=asset_root,
            origin=origin,
            manifest_path=manifest_path,
        )

        summary = runtime.summary()
        fps = float(summary["fps"])

        if abs(fps - 20.0) > 1e-6:
            raise RuntimeError(
                "This NEAT multi-actor smoke test currently expects 20 Hz."
            )

        total_scenario_frames = (
            int(round(float(summary["duration_s"]) * fps)) + 1
        )

        start_frame = max(0, int(args.start_frame))
        end_frame = total_scenario_frames - 1

        if int(args.max_frames) > 0:
            end_frame = min(
                end_frame,
                start_frame + int(args.max_frames) - 1,
            )

        if end_frame < start_frame:
            raise RuntimeError("No scenario frames selected.")

        print("[runtime]", summary)
        print("[scenario frames]", f"{start_frame}..{end_frame}")

        compositor = HEMultiActorCompositorV1(
            distance_selection_mode=args.distance_selection_mode,
            bottom_y_offset_px=float(args.he_bottom_y_offset_px),
        )

        current_frame = world.tick()

        current_front = get_sensor_frame(
            q_front, current_frame, "front"
        )
        current_left = get_sensor_frame(
            q_left, current_frame, "left"
        )
        current_right = get_sensor_frame(
            q_right, current_frame, "right"
        )
        current_gnss = get_sensor_frame(
            q_gnss, current_frame, "GNSS"
        )
        current_imu = get_sensor_frame(
            q_imu, current_frame, "IMU"
        )

        scenario_id = str(summary["scenario_id"])
        output_dir = ensure_dir(output_root / scenario_id)

        csv_path = output_dir / f"{scenario_id}_he_neat.csv"
        video_path = output_dir / f"{scenario_id}_he_neat_inputs.mp4"

        csv_fields = [
            "scenario_frame",
            "carla_frame",
            "t_s",
            "ego_x",
            "ego_y",
            "ego_z",
            "ego_yaw",
            "ego_speed_mps",
            "route_index",
            "route_deviation_m",
            "destination_distance_m",
            "command_name",
            "command_value",
            "target_x",
            "target_y",
            "neat_steer",
            "neat_throttle",
            "neat_brake",
            "red_light_occ",
            "bootstrap",
            "active_actor_count",
            "front_rendered_count",
            "left_rendered_count",
            "right_rendered_count",
            "front_rendered_ids",
            "left_rendered_ids",
            "right_rendered_ids",
            "front_nearest_depth_m",
            "left_nearest_depth_m",
            "right_nearest_depth_m",
        ]

        csv_fp = open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(csv_fp, fieldnames=csv_fields)
        writer.writeheader()

        if args.save_video:
            video_writer = make_video_writer(video_path, fps)

        route_idx = 0
        deviation_counter = 0
        completed_frames = 0
        max_brake = 0.0

        print()
        print("=" * 78)
        print("NEAT + HE MULTI-ACTOR CLOSED-LOOP V1")
        print("=" * 78)
        print("scenario :", scenario_id)
        print("NEAT cams: 3 x 400x300 FOV100; yaws 0/-60/+60")
        print("actors   : runtime-driven")
        print("condition: HE only")
        print("=" * 78)
        print()

        for scenario_frame in range(start_frame, end_frame + 1):
            t_s = float(scenario_frame) / fps

            base_front = carla_image_to_rgb(current_front)
            base_left = carla_image_to_rgb(current_left)
            base_right = carla_image_to_rgb(current_right)

            active_actors = list(
                runtime.active_actors(scenario_frame)
            )

            front = compositor.render(
                base_rgb=base_front,
                camera_tf=current_front.transform,
                active_actors=active_actors,
                width=400,
                height=300,
                fov=100.0,
            )

            left = compositor.render(
                base_rgb=base_left,
                camera_tf=current_left.transform,
                active_actors=active_actors,
                width=400,
                height=300,
                fov=100.0,
            )

            right = compositor.render(
                base_rgb=base_right,
                camera_tf=current_right.transform,
                active_actors=active_actors,
                width=400,
                height=300,
                fov=100.0,
            )

            speed_mps = get_speed_mps(ego)

            nav = get_neat_navigation(
                route_planner,
                current_gnss,
                current_imu,
            )

            bootstrap = (
                scenario_frame - start_frame
            ) < int(config.seq_len)

            if bootstrap:
                result = make_bootstrap_result()
            else:
                result = run_neat(
                    net=net,
                    config=config,
                    plan_grid=plan_grid,
                    light_grid=light_grid,
                    rgb_front=front.rgb,
                    rgb_left=left.rgb,
                    rgb_right=right.rgb,
                    speed_mps=speed_mps,
                    target_xy=nav["target_point"],
                    device=device,
                )

            ego.apply_control(
                carla.VehicleControl(
                    steer=float(result["steer"]),
                    throttle=float(result["throttle"]),
                    brake=float(result["brake"]),
                    hand_brake=False,
                    manual_gear_shift=False,
                )
            )

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location

            route_idx, route_deviation = closest_route_index(
                route,
                ego_loc,
                route_idx,
            )

            destination_distance = distance_2d(
                ego_loc,
                destination,
            )

            if route_deviation > float(args.max_route_deviation_m):
                deviation_counter += 1
            else:
                deviation_counter = 0

            front_ids = rendered_actor_ids(front)
            left_ids = rendered_actor_ids(left)
            right_ids = rendered_actor_ids(right)

            front_depth = nearest_rendered_depth(front)
            left_depth = nearest_rendered_depth(left)
            right_depth = nearest_rendered_depth(right)

            writer.writerow(
                {
                    "scenario_frame": scenario_frame,
                    "carla_frame": current_frame,
                    "t_s": f"{t_s:.6f}",
                    "ego_x": f"{ego_loc.x:.6f}",
                    "ego_y": f"{ego_loc.y:.6f}",
                    "ego_z": f"{ego_loc.z:.6f}",
                    "ego_yaw": f"{ego_tf.rotation.yaw:.6f}",
                    "ego_speed_mps": f"{speed_mps:.6f}",
                    "route_index": route_idx,
                    "route_deviation_m": f"{route_deviation:.6f}",
                    "destination_distance_m": f"{destination_distance:.6f}",
                    "command_name": command_label(nav["command"]),
                    "command_value": command_value(nav["command"]),
                    "target_x": f"{float(nav['target_point'][0]):.6f}",
                    "target_y": f"{float(nav['target_point'][1]):.6f}",
                    "neat_steer": f"{float(result['steer']):.6f}",
                    "neat_throttle": f"{float(result['throttle']):.6f}",
                    "neat_brake": f"{float(result['brake']):.6f}",
                    "red_light_occ": int(result["red_light_occ"]),
                    "bootstrap": int(bool(bootstrap)),
                    "active_actor_count": len(active_actors),
                    "front_rendered_count": len(front_ids),
                    "left_rendered_count": len(left_ids),
                    "right_rendered_count": len(right_ids),
                    "front_rendered_ids": "|".join(front_ids),
                    "left_rendered_ids": "|".join(left_ids),
                    "right_rendered_ids": "|".join(right_ids),
                    "front_nearest_depth_m": (
                        "" if front_depth is None else f"{front_depth:.6f}"
                    ),
                    "left_nearest_depth_m": (
                        "" if left_depth is None else f"{left_depth:.6f}"
                    ),
                    "right_nearest_depth_m": (
                        "" if right_depth is None else f"{right_depth:.6f}"
                    ),
                }
            )

            csv_fp.flush()

            if video_writer is not None:
                write_neat_input_frame(
                    video_writer,
                    front.rgb,
                    left.rgb,
                    right.rgb,
                )

            completed_frames += 1
            max_brake = max(max_brake, float(result["brake"]))

            debug_every = int(args.debug_every)

            if (
                scenario_frame == start_frame
                or scenario_frame == end_frame
                or (
                    debug_every > 0
                    and scenario_frame % debug_every == 0
                )
            ):
                print(
                    f"[frame {scenario_frame:04d}] "
                    f"speed={speed_mps:.2f} "
                    f"active={len(active_actors)} "
                    f"rendered(F/L/R)="
                    f"{len(front_ids)}/{len(left_ids)}/{len(right_ids)} "
                    f"NEAT=(S={float(result['steer']):+.3f},"
                    f"T={float(result['throttle']):.3f},"
                    f"B={float(result['brake']):.3f}) "
                    f"route_dev={route_deviation:.2f}"
                )

            if deviation_counter >= int(args.deviation_patience_frames):
                print(
                    "[stop] route deviation exceeded limit for",
                    deviation_counter,
                    "consecutive frames",
                )
                break

            if destination_distance <= float(args.destination_tolerance_m):
                print(
                    "[stop] destination reached:",
                    f"{destination_distance:.2f} m",
                )
                break

            if scenario_frame >= end_frame:
                break

            current_frame = world.tick()

            current_front = get_sensor_frame(
                q_front, current_frame, "front"
            )
            current_left = get_sensor_frame(
                q_left, current_frame, "left"
            )
            current_right = get_sensor_frame(
                q_right, current_frame, "right"
            )
            current_gnss = get_sensor_frame(
                q_gnss, current_frame, "GNSS"
            )
            current_imu = get_sensor_frame(
                q_imu, current_frame, "IMU"
            )

        print()
        print("=" * 78)
        print("NEAT + HE MULTI-ACTOR RUN COMPLETE")
        print("=" * 78)
        print("completed frames:", completed_frames)
        print("max NEAT brake:", f"{max_brake:.3f}")
        print("CSV:", csv_path)

        if args.save_video:
            print("input video:", video_path)

        print()
        print("No physical scenario adversaries were spawned.")
        print("=" * 78)

    finally:
        if csv_fp is not None:
            csv_fp.close()

        if video_writer is not None:
            video_writer.release()

        for actor in reversed(actors):
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
