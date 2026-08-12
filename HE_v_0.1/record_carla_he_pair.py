import argparse
import json
import math
import queue
from pathlib import Path

import cv2
import numpy as np

try:
    import carla
except ImportError:
    raise RuntimeError("Could not import carla. Run this inside your CARLA Python environment.")


# ============================================================
# Basic math / transform helpers
# ============================================================

def yaw_forward_right(yaw_deg):
    yaw = math.radians(float(yaw_deg))
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return forward, right


def normalize_angle_180(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def carla_transform_to_dict(tf):
    return {
        "x": float(tf.location.x),
        "y": float(tf.location.y),
        "z": float(tf.location.z),
        "pitch": float(tf.rotation.pitch),
        "yaw": float(tf.rotation.yaw),
        "roll": float(tf.rotation.roll),
    }


def local_ego_initial_to_world_transform(
    ego0_tf,
    local_x_m,
    local_z_m,
    local_y_m=0.0,
    local_yaw_deg=0.0,
    world=None,
    snap_to_road=True,
    z_lift=0.05,
):
    """
    Convert our HE/ScenarioGenerator local convention to CARLA world.

    local_z_m = forward/depth from ego initial pose
    local_x_m = lateral/right axis from ego initial pose
    local_y_m = vertical offset
    local_yaw_deg = heading relative to ego initial yaw

    If world is provided, snap the resulting x/y position to the nearest
    CARLA road waypoint height. This avoids floating adversary vehicles.
    """

    ego0_x = float(ego0_tf.location.x)
    ego0_y = float(ego0_tf.location.y)
    ego0_z = float(ego0_tf.location.z)
    ego0_yaw = float(ego0_tf.rotation.yaw)

    forward, right = yaw_forward_right(ego0_yaw)

    world_xy = (
        np.array([ego0_x, ego0_y], dtype=np.float64)
        + forward * float(local_z_m)
        + right * float(local_x_m)
    )

    world_z = ego0_z + float(local_y_m)
    world_yaw = ego0_yaw + float(local_yaw_deg)

    loc = carla.Location(
        x=float(world_xy[0]),
        y=float(world_xy[1]),
        z=float(world_z),
    )

    if world is not None and snap_to_road:
        waypoint = world.get_map().get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is not None:
            # Preserve the requested x/y position.
            # Only use the road waypoint to correct height.
            loc.z = float(waypoint.transform.location.z + float(z_lift))

    return carla.Transform(
        loc,
        carla.Rotation(
            pitch=0.0,
            yaw=float(world_yaw),
            roll=0.0,
        ),
    )

def make_camera_intrinsic(width, height, fov_deg):
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    k = np.identity(3)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    return k


def project_world_point_to_image(world_point, world_to_camera, k):
    """
    CARLA camera coordinates:
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


def compute_actor_2d_bbox(actor, camera_transform, width, height, fov):
    """
    Project actor 3D bounding box into the camera image.
    Returns visible bbox in pixel coordinates.
    """

    k = make_camera_intrinsic(width, height, fov)
    world_to_camera = np.array(
        camera_transform.get_inverse_matrix()
    )

    bb = actor.bounding_box
    vertices = bb.get_world_vertices(actor.get_transform())

    points = []
    depths = []

    for v in vertices:
        proj = project_world_point_to_image(v, world_to_camera, k)
        if proj is None:
            continue

        u, vv, d = proj
        points.append((u, vv))
        depths.append(d)

    if len(points) < 2:
        return {
            "visible": False,
            "reason": "bbox_behind_camera",
        }

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width - 1), max(xs))
    y2 = min(float(height - 1), max(ys))

    if x2 <= 0 or x1 >= width or y2 <= 0 or y1 >= height:
        return {
            "visible": False,
            "reason": "bbox_outside_image",
            "raw_x1": float(min(xs)),
            "raw_y1": float(min(ys)),
            "raw_x2": float(max(xs)),
            "raw_y2": float(max(ys)),
            "depth_m": float(np.mean(depths)),
        }

    if x2 <= x1 or y2 <= y1:
        return {
            "visible": False,
            "reason": "bbox_invalid",
        }

    return {
        "visible": True,
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "cx": float((x1 + x2) / 2.0),
        "cy": float((y1 + y2) / 2.0),
        "bottom_y": float(y2),
        "box_width": float(x2 - x1),
        "box_height": float(y2 - y1),
        "depth_m": float(np.mean(depths)),
    }


# ============================================================
# Trajectory helpers
# ============================================================

def generate_scripted_ego_poses(spawn_tf, frames, fps, speed_mps, motion,
                                turn_start_s, turn_duration_s, turn_yaw_deg):
    """
    Generate deterministic ego poses.

    motion:
      straight
      right_turn
      left_turn

    This is intentionally scripted so background and real-adversary videos
    have exactly the same ego trajectory.
    """

    dt = 1.0 / float(fps)

    x = float(spawn_tf.location.x)
    y = float(spawn_tf.location.y)
    z = float(spawn_tf.location.z)
    yaw = float(spawn_tf.rotation.yaw)

    pitch = float(spawn_tf.rotation.pitch)
    roll = float(spawn_tf.rotation.roll)

    if motion == "straight":
        total_yaw_change = 0.0
    elif motion == "right_turn":
        total_yaw_change = abs(float(turn_yaw_deg))
    elif motion == "left_turn":
        total_yaw_change = -abs(float(turn_yaw_deg))
    else:
        raise ValueError(f"Unknown ego motion: {motion}")

    turn_start_f = int(round(turn_start_s * fps))
    turn_end_f = int(round((turn_start_s + turn_duration_s) * fps))

    poses = []

    for frame_idx in range(frames):
        poses.append(
            carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
            )
        )

        # Smooth constant yaw-rate during turn window.
        if turn_start_f <= frame_idx < turn_end_f and turn_duration_s > 0:
            yaw_rate_deg_per_s = total_yaw_change / float(turn_duration_s)
        else:
            yaw_rate_deg_per_s = 0.0

        yaw += yaw_rate_deg_per_s * dt

        forward, _ = yaw_forward_right(yaw)
        x += float(speed_mps) * dt * forward[0]
        y += float(speed_mps) * dt * forward[1]

    return poses


def adversary_local_state_at_time(args, t_s):
    """
    Returns local state in ego-initial coordinates:
      x_m, y_m, z_m, yaw_deg
    """

    if args.scenario == "static":
        return {
            "x_m": float(args.lane_y),
            "y_m": 0.0,
            "z_m": float(args.start_distance),
            "yaw_deg": 0.0,
        }

    if args.scenario == "oncoming":
        return {
            "x_m": float(args.lane_y),
            "y_m": 0.0,
            "z_m": float(args.start_distance) - float(args.adv_speed) * float(t_s),
            "yaw_deg": 180.0,
        }

    if args.scenario == "following":
        return {
            "x_m": float(args.lane_y),
            "y_m": 0.0,
            "z_m": float(args.start_distance) + float(args.adv_speed) * float(t_s),
            "yaw_deg": 0.0,
        }

    if args.scenario == "cut_in":
        cut_start = float(args.cut_start_s)
        cut_duration = max(0.01, float(args.cut_duration_s))
        alpha = (float(t_s) - cut_start) / cut_duration
        alpha = max(0.0, min(1.0, alpha))

        # Smoothstep lateral transition.
        s = alpha * alpha * (3.0 - 2.0 * alpha)

        x0 = float(args.lane_y)
        x1 = float(args.target_lane_y)
        x = x0 + (x1 - x0) * s

        z = float(args.start_distance) + float(args.adv_speed) * float(t_s)

        # Approximate heading from dx/dz during cut-in.
        if 0.0 < alpha < 1.0:
            dx = x1 - x0
            dz = max(0.01, float(args.adv_speed) * cut_duration)
            yaw_deg = math.degrees(math.atan2(dx, dz))
        else:
            yaw_deg = 0.0

        return {
            "x_m": float(x),
            "y_m": 0.0,
            "z_m": float(z),
            "yaw_deg": float(yaw_deg),
        }

    raise ValueError(f"Unknown scenario: {args.scenario}")


# ============================================================
# CARLA helpers
# ============================================================

def setup_world(client, town, fps):
    if town:
        world = client.load_world(town)
    else:
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / float(fps)
    world.apply_settings(settings)

    return world


def restore_world(world, original_settings):
    world.apply_settings(original_settings)


def get_blueprint(world, filter_name):
    bps = world.get_blueprint_library().filter(filter_name)
    if not bps:
        raise RuntimeError(f"No blueprint found for filter: {filter_name}")
    return bps[0]


def spawn_actor_safe(world, blueprint, transform):
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        # Try raising a little if spawn is blocked by ground collision.
        transform.location.z += 0.5
        actor = world.try_spawn_actor(blueprint, transform)

    if actor is None:
        raise RuntimeError(f"Could not spawn actor at {transform}")

    return actor

def settle_ego_on_road(world, ego, settle_frames=30):
    """
    Let the ego vehicle settle naturally onto the CARLA road before
    switching to deterministic scripted motion.

    This is important because CARLA spawn points may start the vehicle
    slightly above the road. HEPlacementModel v2 was calibrated using
    an ego vehicle that had physically settled before the camera pose
    was locked.
    """

    ego.set_simulate_physics(True)
    ego.set_autopilot(False)

    # Hold the vehicle stationary while gravity settles it.
    ego.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=False,
        )
    )

    for _ in range(int(settle_frames)):
        world.tick()

    settled_tf = ego.get_transform()

    print(
        "[INFO] Ego settled transform: "
        f"x={settled_tf.location.x:.6f}, "
        f"y={settled_tf.location.y:.6f}, "
        f"z={settled_tf.location.z:.6f}, "
        f"pitch={settled_tf.rotation.pitch:.6f}, "
        f"yaw={settled_tf.rotation.yaw:.6f}, "
        f"roll={settled_tf.rotation.roll:.6f}"
    )

    # From this point onward the pair recorder controls the pose itself.
    ego.set_simulate_physics(False)
    ego.set_transform(settled_tf)

    return settled_tf

def make_rgb_camera(world, ego, args):
    camera_bp = get_blueprint(world, "sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", str(args.fov))

    camera_tf = carla.Transform(
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

    camera = world.spawn_actor(camera_bp, camera_tf, attach_to=ego)
    return camera


def image_to_rgb_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return rgb

def get_image_for_carla_frame(image_queue, target_frame, timeout=5.0):
    """
    Return the RGB sensor measurement belonging exactly to target_frame.

    In synchronous CARLA execution we want:
        world frame N
        RGB image frame N
        image.transform from frame N

    Any stale sensor measurements are discarded. A future-frame image
    indicates a synchronization problem and aborts the recording.
    """
    while True:
        image = image_queue.get(timeout=timeout)

        if int(image.frame) < int(target_frame):
            # Stale measurement left in the queue.
            continue

        if int(image.frame) > int(target_frame):
            raise RuntimeError(
                f"RGB sensor skipped target CARLA frame {target_frame}; "
                f"received frame {image.frame}"
            )

        return image

# ============================================================
# Main recording
# ============================================================

def run(args):
    output_root = Path(args.output_root)
    pair_dir = output_root / args.pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "background":
        video_path = pair_dir / "background.mp4"
        ego_pose_path = pair_dir / "background_ego_pose.jsonl"
        adv_pose_path = None
        bbox_path = None
    else:
        video_path = pair_dir / "real_adversary.mp4"
        ego_pose_path = pair_dir / "real_ego_pose.jsonl"
        adv_pose_path = pair_dir / "real_adversary_pose.jsonl"
        bbox_path = pair_dir / "real_bbox.jsonl"

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    original_settings = world.get_settings()

    actors = []
    writer = None
    ego_pose_file = None
    adv_pose_file = None
    bbox_file = None

    try:
        world = setup_world(client, args.town, args.fps)

        blueprint_library = world.get_blueprint_library()

        vehicle_bp = get_blueprint(world, args.ego_vehicle_filter)
        adv_bp = get_blueprint(world, args.adversary_vehicle_filter)

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found.")

        if args.spawn_index < 0 or args.spawn_index >= len(spawn_points):
            raise RuntimeError(
                f"spawn_index {args.spawn_index} out of range. "
                f"Available: 0 to {len(spawn_points)-1}"
            )

        ego0_tf = spawn_points[args.spawn_index]

        ego = spawn_actor_safe(world, vehicle_bp, ego0_tf)
        actors.append(ego)

        # IMPORTANT:
        # CARLA spawn points may place the vehicle reference frame above the
        # actual road surface. HEPlacementModel v2 was calibrated after the
        # Tesla ego had physically settled on the road, so reproduce that
        # convention here before locking deterministic motion.
        ego0_tf = settle_ego_on_road(
            world=world,
            ego=ego,
            settle_frames=args.ego_settle_frames,
        )

        camera = make_rgb_camera(world, ego, args)
        actors.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        adversary = None
        if args.mode == "real_adversary":
            adv_state0 = adversary_local_state_at_time(args, 0.0)
            adv_tf0 = local_ego_initial_to_world_transform(
                ego0_tf=ego0_tf,
                local_x_m=adv_state0["x_m"],
                local_y_m=adv_state0["y_m"],
                local_z_m=adv_state0["z_m"],
                local_yaw_deg=adv_state0["yaw_deg"],
                world=world,
                snap_to_road=True,
                z_lift=0.05,
            )

            adversary = spawn_actor_safe(world, adv_bp, adv_tf0)
            adversary.set_simulate_physics(False)
            actors.append(adversary)

        ego_poses = generate_scripted_ego_poses(
            spawn_tf=ego0_tf,
            frames=args.frames,
            fps=args.fps,
            speed_mps=args.ego_speed,
            motion=args.ego_motion,
            turn_start_s=args.turn_start_s,
            turn_duration_s=args.turn_duration_s,
            turn_yaw_deg=args.turn_yaw_deg,
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            float(args.fps),
            (int(args.width), int(args.height)),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")

        ego_pose_file = open(ego_pose_path, "w", encoding="utf-8")

        if adv_pose_path is not None:
            adv_pose_file = open(adv_pose_path, "w", encoding="utf-8")

        if bbox_path is not None:
            bbox_file = open(bbox_path, "w", encoding="utf-8")

        print("[INFO] Pair dir:", pair_dir)
        print("[INFO] Mode:", args.mode)
        print("[INFO] Video:", video_path)
        print("[INFO] Ego pose:", ego_pose_path)
        if adv_pose_path:
            print("[INFO] Adversary pose:", adv_pose_path)
        if bbox_path:
            print("[INFO] Real bbox:", bbox_path)

        # Warmup.
        for _ in range(args.warmup_frames):
            warmup_frame = world.tick()

            try:
                get_image_for_carla_frame(
                    image_queue,
                    target_frame=warmup_frame,
                    timeout=2.0,
                )
            except queue.Empty:
                pass

        for frame_idx in range(args.frames):
            t_s = float(frame_idx) / float(args.fps)

            ego_tf = ego_poses[frame_idx]
            ego.set_transform(ego_tf)

            if adversary is not None:
                adv_state = adversary_local_state_at_time(args, t_s)
                adv_tf = local_ego_initial_to_world_transform(
                    ego0_tf=ego0_tf,
                    local_x_m=adv_state["x_m"],
                    local_y_m=adv_state["y_m"],
                    local_z_m=adv_state["z_m"],
                    local_yaw_deg=adv_state["yaw_deg"],
                    world=world,
                    snap_to_road=True,
                    z_lift=0.05,
                )
                adversary.set_transform(adv_tf)

            carla_frame = world.tick()

            image = get_image_for_carla_frame(
                image_queue,
                target_frame=carla_frame,
                timeout=5.0,
            )
            rgb = image_to_rgb_array(image)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

            # Sensor transform corresponding exactly to image.frame.
            camera_tf = image.transform

            ego_record = {
                "recorded_frame_idx": int(frame_idx),
                "carla_frame": int(image.frame),
                "t_s": float(t_s),
                "ego_transform": carla_transform_to_dict(ego_tf),
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
            ego_pose_file.write(json.dumps(ego_record) + "\n")

            if adversary is not None:
                adv_tf_current = adversary.get_transform()

                adv_record = {
                    "recorded_frame_idx": int(frame_idx),
                    "carla_frame": int(image.frame),
                    "t_s": float(t_s),
                    "adversary_id": "adv_001",
                    "scenario": args.scenario,
                    "local_state_ego_initial": adv_state,
                    "adversary_transform": carla_transform_to_dict(adv_tf_current),
                }
                adv_pose_file.write(json.dumps(adv_record) + "\n")

                bbox = compute_actor_2d_bbox(
                    actor=adversary,
                    camera_transform=camera_tf,
                    width=args.width,
                    height=args.height,
                    fov=args.fov,
                )

                bbox_record = {
                    "recorded_frame_idx": int(frame_idx),
                    "carla_frame": int(image.frame),
                    "t_s": float(t_s),
                    "adversary_id": "adv_001",
                    "bbox": bbox,
                }
                bbox_file.write(json.dumps(bbox_record) + "\n")

            if frame_idx % 30 == 0:
                print(f"[INFO] frame {frame_idx}/{args.frames}")

        print("[DONE] Recording complete.")

    finally:
        if writer is not None:
            writer.release()

        if ego_pose_file is not None:
            ego_pose_file.close()

        if adv_pose_file is not None:
            adv_pose_file.close()

        if bbox_file is not None:
            bbox_file.close()

        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception:
                pass

        try:
            restore_world(world, original_settings)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)

    parser.add_argument(
        "--mode",
        required=True,
        choices=["background", "real_adversary"],
    )

    parser.add_argument("--pair-name", required=True)
    parser.add_argument("--output-root", default="recordings/he_pairs")

    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)

    parser.add_argument("--spawn-index", type=int, default=10)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument(
        "--ego-settle-frames",
        type=int,
        default=30,
        help="CARLA ticks used to let ego settle onto the road before deterministic recording.",
    )
    parser.add_argument("--ego-vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--adversary-vehicle-filter", default="vehicle.tesla.model3")

    parser.add_argument(
        "--ego-motion",
        choices=["straight", "right_turn", "left_turn"],
        default="straight",
    )
    parser.add_argument("--ego-speed", type=float, default=5.0)
    parser.add_argument("--turn-start-s", type=float, default=2.0)
    parser.add_argument("--turn-duration-s", type=float, default=4.0)
    parser.add_argument("--turn-yaw-deg", type=float, default=45.0)

    # Fixed HEPlacementModel v2 camera convention.
    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=1.6)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)

    parser.add_argument(
        "--scenario",
        choices=["static", "oncoming", "following", "cut_in"],
        default="oncoming",
    )

    parser.add_argument("--lane-y", type=float, default=-3.5)
    parser.add_argument("--target-lane-y", type=float, default=0.0)
    parser.add_argument("--start-distance", type=float, default=60.0)
    parser.add_argument("--adv-speed", type=float, default=8.0)
    parser.add_argument("--cut-start-s", type=float, default=1.5)
    parser.add_argument("--cut-duration-s", type=float, default=3.0)

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())