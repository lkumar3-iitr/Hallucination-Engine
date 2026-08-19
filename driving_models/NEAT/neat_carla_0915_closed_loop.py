"""
neat_carla_0915_closed_loop.py

Native NEAT closed-loop baseline adapted to CARLA 0.9.15.

Purpose
-------
1. Spawn ego vehicle in CARLA 0.9.15.
2. Build an explicit CARLA route.
3. Attach NEAT's original three RGB cameras:
      front : yaw   0 deg
      left  : yaw -60 deg
      right : yaw +60 deg
      400x300, FOV 100
      x=1.3, y=0.0, z=2.3
4. Attach GNSS + IMU.
5. Use NEAT's original RoutePlanner(4.0, 50.0).
6. Use NEAT's original 256x256 center crop.
7. Run official pretrained NEAT encoder/decoder.
8. Use NEAT's original waypoint planner + PID controller.
9. Apply only NEAT controls to the ego vehicle.

No BasicAgent / autopilot controls the ego.
No adversary.
No HE rendering.

This is the clean CARLA baseline probe.
"""

import argparse
import csv
import json
import math
import queue
import sys
from pathlib import Path

import carla
import numpy as np
import torch
from PIL import Image, ImageDraw


# ============================================================
# Paths
# ============================================================

THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]

DEFAULT_NEAT_ROOT = (
    HE_ROOT
    / "external_models"
    / "neat"
)

DEFAULT_CHECKPOINT = (
    DEFAULT_NEAT_ROOT
    / "model_ckpt"
    / "neat"
)


# ============================================================
# CARLA agents import
# ============================================================

def import_carla_agents(pythonapi_hint=None):

    def _load():
        from agents.navigation.global_route_planner import (
            GlobalRoutePlanner
        )

        return GlobalRoutePlanner

    try:
        return _load()

    except ModuleNotFoundError:
        pass

    candidates = []

    if pythonapi_hint:
        candidates.append(
            Path(pythonapi_hint)
        )

    for path in candidates:

        path = path.resolve()

        # Example:
        #
        # E:\Carla\Carla_0.9.15\PythonAPI\carla
        #
        # contains:
        #
        # agents\
        #

        if (path / "agents").is_dir():

            sys.path.insert(
                0,
                str(path),
            )

        elif (
            path
            / "carla"
            / "agents"
        ).is_dir():

            sys.path.insert(
                0,
                str(
                    path
                    / "carla"
                ),
            )

        else:
            continue

        try:
            return _load()

        except ModuleNotFoundError:
            pass

    raise RuntimeError(
        "Could not import CARLA agents.\n"
        "Pass:\n"
        "--carla-pythonapi "
        "E:\\Carla\\Carla_0.9.15\\PythonAPI\\carla"
    )


# ============================================================
# NEAT imports
# ============================================================

def import_neat(neat_root):

    neat_root = Path(
        neat_root
    ).resolve()

    if str(neat_root) not in sys.path:

        sys.path.insert(
            0,
            str(neat_root),
        )

    leaderboard_root = (
        neat_root
        / "leaderboard"
    )

    if str(leaderboard_root) not in sys.path:

        sys.path.insert(
            0,
            str(leaderboard_root),
        )

    from neat.architectures import AttentionField
    from neat.config import GlobalConfig

    from team_code.planner import RoutePlanner

    return (
        AttentionField,
        GlobalConfig,
        RoutePlanner,
    )


# ============================================================
# NEAT model
# ============================================================

def load_neat(
    neat_root,
    checkpoint_dir,
    device,
):

    (
        AttentionField,
        GlobalConfig,
        RoutePlanner,
    ) = import_neat(
        neat_root
    )

    config = GlobalConfig()

    if config.seq_len != 1:
        raise RuntimeError(
            "This probe currently assumes "
            "NEAT seq_len == 1."
        )

    net = AttentionField(
        config,
        device,
    )

    encoder_path = (
        Path(checkpoint_dir)
        / "best_encoder.pth"
    )

    decoder_path = (
        Path(checkpoint_dir)
        / "best_decoder.pth"
    )

    print(
        "[NEAT] encoder:",
        encoder_path,
    )

    print(
        "[NEAT] decoder:",
        decoder_path,
    )

    encoder_state = torch.load(
        encoder_path,
        map_location="cpu",
        weights_only=True,
    )

    decoder_state = torch.load(
        decoder_path,
        map_location="cpu",
        weights_only=True,
    )

    encoder_result = (
        net.encoder.load_state_dict(
            encoder_state,
            strict=True,
        )
    )

    decoder_result = (
        net.decoder.load_state_dict(
            decoder_state,
            strict=True,
        )
    )

    print(
        "[NEAT] encoder:",
        encoder_result,
    )

    print(
        "[NEAT] decoder:",
        decoder_result,
    )

    net = net.to(
        device
    )

    net.eval()

    plan_grid = net.create_plan_grid(
        config.plan_scale,
        config.plan_points,
        1,
    )

    light_grid = net.create_light_grid(
        config.light_x_steps,
        config.light_y_steps,
        1,
    )

    return (
        net,
        config,
        RoutePlanner,
        plan_grid,
        light_grid,
    )


# ============================================================
# Image processing
# ============================================================

def carla_image_to_rgb(image):

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )

    array = array.reshape(
        (
            image.height,
            image.width,
            4,
        )
    )

    # CARLA raw image:
    # BGRA
    #
    # Convert to RGB.

    rgb = (
        array[
            :,
            :,
            :3,
        ][:, :, ::-1]
        .copy()
    )

    return rgb


def prepare_neat_image(
    rgb,
    crop=256,
):

    height, width = (
        rgb.shape[:2]
    )

    start_y = (
        height // 2
        - crop // 2
    )

    start_x = (
        width // 2
        - crop // 2
    )

    cropped = rgb[
        start_y:
        start_y + crop,

        start_x:
        start_x + crop,
    ]

    chw = np.transpose(
        cropped,
        (
            2,
            0,
            1,
        ),
    ).copy()

    return chw


def rgb_to_neat_tensor(
    rgb,
    device,
    crop=256,
):

    array = prepare_neat_image(
        rgb,
        crop=crop,
    )

    tensor = torch.from_numpy(
        array
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    # IMPORTANT:
    #
    # Do NOT divide by 255 here.
    #
    # This reproduces the original NEAT
    # leaderboard agent preprocessing.

    return tensor


# ============================================================
# CARLA sensors
# ============================================================

def make_camera(
    world,
    ego,
    yaw,
):

    bp = (
        world
        .get_blueprint_library()
        .find(
            "sensor.camera.rgb"
        )
    )

    bp.set_attribute(
        "image_size_x",
        "400",
    )

    bp.set_attribute(
        "image_size_y",
        "300",
    )

    bp.set_attribute(
        "fov",
        "100",
    )

    transform = carla.Transform(
        carla.Location(
            x=1.3,
            y=0.0,
            z=2.3,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=float(yaw),
            roll=0.0,
        ),
    )

    return world.spawn_actor(
        bp,
        transform,
        attach_to=ego,
    )


def make_gnss(
    world,
    ego,
):

    bp = (
        world
        .get_blueprint_library()
        .find(
            "sensor.other.gnss"
        )
    )

    transform = carla.Transform(
        carla.Location(
            x=0.0,
            y=0.0,
            z=0.0,
        )
    )

    return world.spawn_actor(
        bp,
        transform,
        attach_to=ego,
    )


def make_imu(
    world,
    ego,
):

    bp = (
        world
        .get_blueprint_library()
        .find(
            "sensor.other.imu"
        )
    )

    transform = carla.Transform(
        carla.Location(
            x=0.0,
            y=0.0,
            z=0.0,
        )
    )

    return world.spawn_actor(
        bp,
        transform,
        attach_to=ego,
    )


def get_sensor_frame(
    sensor_queue,
    target_frame,
    name,
    timeout=10.0,
):

    while True:

        try:
            data = sensor_queue.get(
                timeout=timeout
            )

        except queue.Empty:

            raise RuntimeError(
                f"{name} did not produce "
                f"frame {target_frame}."
            )

        if data.frame == target_frame:
            return data

        if data.frame < target_frame:
            continue

        raise RuntimeError(
            f"{name} skipped CARLA "
            f"frame {target_frame}; "
            f"got {data.frame}."
        )


# ============================================================
# Route helpers
# ============================================================

def distance_2d(
    a,
    b,
):

    return math.hypot(
        float(a.x)
        - float(b.x),

        float(a.y)
        - float(b.y),
    )


def route_length(route):

    total = 0.0

    for i in range(
        len(route) - 1
    ):

        a = (
            route[i][0]
            .transform
            .location
        )

        b = (
            route[i + 1][0]
            .transform
            .location
        )

        total += distance_2d(
            a,
            b,
        )

    return total


def build_route(
    grp,
    spawn_points,
    start_idx,
    destination_idx,
    desired_m,
):

    start = (
        spawn_points[
            start_idx
        ].location
    )

    # --------------------------------------------------------
    # Explicit destination
    # --------------------------------------------------------

    if destination_idx >= 0:

        destination_idx %= (
            len(spawn_points)
        )

        if (
            destination_idx
            == start_idx
        ):
            raise ValueError(
                "Destination equals "
                "start spawn."
            )

        route = grp.trace_route(
            start,
            spawn_points[
                destination_idx
            ].location,
        )

        if len(route) < 2:
            raise RuntimeError(
                "No usable CARLA route."
            )

        return (
            route,
            destination_idx,
        )

    # --------------------------------------------------------
    # Automatic destination
    # --------------------------------------------------------

    candidates = []

    for idx, tf in enumerate(
        spawn_points
    ):

        if idx == start_idx:
            continue

        euclidean = distance_2d(
            start,
            tf.location,
        )

        if euclidean >= 35.0:

            candidates.append(
                (
                    abs(
                        euclidean
                        - desired_m
                    ),
                    idx,
                )
            )

    for _, idx in sorted(
        candidates
    )[:30]:

        route = grp.trace_route(
            start,
            spawn_points[
                idx
            ].location,
        )

        if len(route) < 2:
            continue

        length_m = route_length(
            route
        )

        if (
            70.0
            <= length_m
            <= 220.0
        ):

            return (
                route,
                idx,
            )

    raise RuntimeError(
        "Could not automatically "
        "select a route."
    )


def closest_route_index(
    route,
    location,
    previous_idx,
):

    start = max(
        0,
        previous_idx - 5,
    )

    end = min(
        len(route),
        previous_idx + 100,
    )

    best_idx = start
    best_distance = float(
        "inf"
    )

    for idx in range(
        start,
        end,
    ):

        point = (
            route[idx][0]
            .transform
            .location
        )

        d = distance_2d(
            point,
            location,
        )

        if d < best_distance:

            best_distance = d
            best_idx = idx

    return (
        best_idx,
        best_distance,
    )


# ============================================================
# CARLA route -> NEAT GPS route
# ============================================================

def build_neat_gps_plan(
    carla_map,
    route,
):

    plan = []

    for waypoint, road_option in route:

        geo = (
            carla_map
            .transform_to_geolocation(
                waypoint
                .transform
                .location
            )
        )

        gps_point = {
            "lat":
                float(
                    geo.latitude
                ),

            "lon":
                float(
                    geo.longitude
                ),
        }

        plan.append(
            (
                gps_point,
                road_option,
            )
        )

    return plan


# ============================================================
# Original NEAT navigation processing
# ============================================================

def get_neat_navigation(
    route_planner,
    gnss,
    imu,
):

    gps_raw = np.array(
        [
            float(
                gnss.latitude
            ),
            float(
                gnss.longitude
            ),
        ],
        dtype=np.float64,
    )

    position = (
        (
            gps_raw
            - route_planner.mean
        )
        * route_planner.scale
    )

    (
        next_wp,
        next_cmd,
    ) = route_planner.run_step(
        position
    )

    compass = float(
        imu.compass
    )

    if math.isnan(
        compass
    ):
        compass = 0.0

    theta = (
        compass
        + np.pi / 2.0
    )

    rotation = np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta),
            ],
        ],
        dtype=np.float64,
    )

    command_vector = np.array(
        [
            next_wp[0]
            - position[0],

            next_wp[1]
            - position[1],
        ],
        dtype=np.float64,
    )

    target_point = (
        rotation.T
        .dot(
            command_vector
        )
    )

    return {
        "position":
            position,

        "next_wp":
            next_wp,

        "command":
            next_cmd,

        "target_point":
            target_point,

        "compass":
            compass,
    }


# ============================================================
# Vehicle speed
# ============================================================

def get_speed_mps(vehicle):

    v = vehicle.get_velocity()

    return math.sqrt(
        v.x * v.x
        + v.y * v.y
        + v.z * v.z
    )


# ============================================================
# NEAT inference
# ============================================================

@torch.no_grad()
def run_neat(
    net,
    config,
    plan_grid,
    light_grid,
    rgb_front,
    rgb_left,
    rgb_right,
    speed_mps,
    target_xy,
    device,
):

    images = [
        rgb_to_neat_tensor(
            rgb_front,
            device,
            crop=config.crop,
        ),

        rgb_to_neat_tensor(
            rgb_left,
            device,
            crop=config.crop,
        ),

        rgb_to_neat_tensor(
            rgb_right,
            device,
            crop=config.crop,
        ),
    ]

    velocity = torch.tensor(
        [
            float(
                speed_mps
            )
        ],
        dtype=torch.float32,
        device=device,
    )

    # Original NEAT agent:
    #
    # target_point =
    # [
    #   FloatTensor([x]),
    #   FloatTensor([y])
    # ]
    #
    # then torch.stack(...)
    #
    # shape = (2, 1)

    target_point = torch.tensor(
        [
            [
                float(
                    target_xy[0]
                )
            ],
            [
                float(
                    target_xy[1]
                )
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    encoding = net.encoder(
        images,
        velocity,
    )

    (
        predicted_waypoints,
        red_light_occ,
    ) = net.plan(
        target_point,
        encoding,
        plan_grid,
        light_grid,
        config.plan_points,
        config.plan_iters,
    )

    future_waypoints = (
        predicted_waypoints[
            :,
            config.seq_len:
        ]
    )

    (
        steer,
        throttle,
        brake,
        metadata,
    ) = net.control_pid(
        future_waypoints,
        velocity,
        target_point,
        red_light_occ,
    )

    steer = float(
        steer
    )

    throttle = float(
        throttle
    )

    brake = float(
        brake
    )

    # Original neat_agent.py
    # post-processing.

    if brake < 0.05:
        brake = 0.0

    if throttle > brake:
        brake = 0.0

    if torch.is_tensor(
        red_light_occ
    ):
        red_light_value = int(
            red_light_occ
            .detach()
            .cpu()
            .item()
        )
    else:
        red_light_value = int(
            red_light_occ
        )

    return {
        "steer":
            steer,

        "throttle":
            throttle,

        "brake":
            brake,

        "red_light_occ":
            red_light_value,

        "metadata":
            metadata,

        "predicted_waypoints":
            predicted_waypoints
            .detach()
            .cpu()
            .numpy(),
    }


# ============================================================
# Debug image
# ============================================================

def save_debug_image(
    path,
    rgb_left,
    rgb_front,
    rgb_right,
    frame_idx,
    speed_mps,
    target,
    control,
):

    composite = np.concatenate(
        [
            rgb_left,
            rgb_front,
            rgb_right,
        ],
        axis=1,
    )

    image = Image.fromarray(
        composite
    )

    draw = ImageDraw.Draw(
        image
    )

    text = (
        f"frame={frame_idx}   "
        f"speed={speed_mps:.2f} m/s   "
        f"target=({target[0]:+.2f},"
        f"{target[1]:+.2f})   "
        f"S={control['steer']:+.3f}   "
        f"T={control['throttle']:.3f}   "
        f"B={control['brake']:.3f}"
    )

    draw.rectangle(
        (
            0,
            0,
            1200,
            24,
        ),
        fill=(
            0,
            0,
            0,
        ),
    )

    draw.text(
        (
            5,
            5,
        ),
        text,
        fill=(
            255,
            255,
            255,
        ),
    )

    image.save(
        path
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--town",
        default="Town10HD_Opt",
    )

    parser.add_argument(
        "--spawn-index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--destination-index",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--desired-route-distance-m",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--route-sampling-resolution",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--carla-pythonapi",
        default=None,
    )

    parser.add_argument(
        "--neat-root",
        default=str(
            DEFAULT_NEAT_ROOT
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=str(
            DEFAULT_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--destination-tolerance-m",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--max-route-deviation-m",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--deviation-patience-frames",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--debug-every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            HE_ROOT
            / "driving_models"
            / "NEAT"
            / "outputs"
            / "closed_loop_probe"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    print(
        "[GPU]",
        torch.cuda.get_device_name(0)
    )

    # ========================================================
    # Imports
    # ========================================================

    GlobalRoutePlanner = (
        import_carla_agents(
            args.carla_pythonapi
        )
    )

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

    print(
        "[NEAT] cameras:",
        config.num_camera,
    )

    print(
        "[NEAT] seq_len:",
        config.seq_len,
    )

    print(
        "[NEAT] pred_len:",
        config.pred_len,
    )

    # ========================================================
    # Outputs
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    debug_dir = (
        output_dir
        / "debug"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "telemetry.csv"
    )

    # ========================================================
    # CARLA
    # ========================================================

    client = carla.Client(
        args.host,
        args.port,
    )

    client.set_timeout(
        20.0
    )

    print(
        "[CARLA] loading:",
        args.town,
    )

    world = client.load_world(
        args.town
    )

    original_settings = (
        world.get_settings()
    )

    actors = []

    csv_file = None

    try:

        # ====================================================
        # Synchronous mode
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            1.0
            / float(
                args.fps
            )
        )

        world.apply_settings(
            settings
        )

        # ====================================================
        # Route
        # ====================================================

        carla_map = (
            world.get_map()
        )

        spawn_points = (
            carla_map
            .get_spawn_points()
        )

        spawn_idx = (
            args.spawn_index
            % len(
                spawn_points
            )
        )

        grp = GlobalRoutePlanner(
            carla_map,
            float(
                args.route_sampling_resolution
            ),
        )

        (
            route,
            destination_idx,
        ) = build_route(
            grp,
            spawn_points,
            spawn_idx,
            args.destination_index,
            args.desired_route_distance_m,
        )

        destination = (
            route[-1][0]
            .transform
            .location
        )

        print(
            "[route] start:",
            spawn_idx,
        )

        print(
            "[route] destination:",
            destination_idx,
        )

        print(
            "[route] points:",
            len(route),
        )

        print(
            "[route] length:",
            f"{route_length(route):.1f} m",
        )

        # ====================================================
        # Ego
        # ====================================================

        ego_bp = (
            world
            .get_blueprint_library()
            .filter(
                "vehicle.tesla.model3"
            )[0]
        )

        if ego_bp.has_attribute(
            "role_name"
        ):

            ego_bp.set_attribute(
                "role_name",
                "hero",
            )

        ego = world.try_spawn_actor(
            ego_bp,
            spawn_points[
                spawn_idx
            ],
        )

        if ego is None:

            raise RuntimeError(
                "Could not spawn ego."
            )

        actors.append(
            ego
        )

        # ====================================================
        # Original NEAT route planner
        # ====================================================

        gps_plan = build_neat_gps_plan(
            carla_map,
            route,
        )

        route_planner = RoutePlanner(
            4.0,
            50.0,
        )

        route_planner.set_route(
            gps_plan,
            gps=True,
        )

        print(
            "[NEAT planner]",
            "RoutePlanner(4.0, 50.0)",
        )

        # ====================================================
        # Sensors
        # ====================================================

        camera_front = make_camera(
            world,
            ego,
            yaw=0.0,
        )

        camera_left = make_camera(
            world,
            ego,
            yaw=-60.0,
        )

        camera_right = make_camera(
            world,
            ego,
            yaw=60.0,
        )

        gnss = make_gnss(
            world,
            ego,
        )

        imu = make_imu(
            world,
            ego,
        )

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

        camera_front.listen(
            q_front.put
        )

        camera_left.listen(
            q_left.put
        )

        camera_right.listen(
            q_right.put
        )

        gnss.listen(
            q_gnss.put
        )

        imu.listen(
            q_imu.put
        )

        # ====================================================
        # Stabilize ego
        # ====================================================

        hold = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
        )

        print(
            "[init] stabilizing..."
        )

        for _ in range(5):

            ego.apply_control(
                hold
            )

            frame = world.tick()

            get_sensor_frame(
                q_front,
                frame,
                "front",
            )

            get_sensor_frame(
                q_left,
                frame,
                "left",
            )

            get_sensor_frame(
                q_right,
                frame,
                "right",
            )

            get_sensor_frame(
                q_gnss,
                frame,
                "GNSS",
            )

            get_sensor_frame(
                q_imu,
                frame,
                "IMU",
            )

        # Reset exactly to requested spawn.

        ego.set_transform(
            spawn_points[
                spawn_idx
            ]
        )

        ego.set_target_velocity(
            carla.Vector3D()
        )

        ego.set_target_angular_velocity(
            carla.Vector3D()
        )

        ego.apply_control(
            hold
        )

        # ====================================================
        # CSV
        # ====================================================

        csv_file = open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        fields = [
            "probe_idx",
            "carla_frame",

            "ego_x",
            "ego_y",
            "ego_z",
            "ego_yaw",

            "speed_mps",

            "route_index",
            "route_deviation_m",
            "destination_distance_m",

            "gps_lat",
            "gps_lon",
            "compass",

            "target_x",
            "target_y",

            "steer",
            "throttle",
            "brake",

            "red_light_occ",

            "desired_speed",
            "angle",
            "angle_last",
            "angle_target",
            "angle_final",

            "bootstrap",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
        )

        writer.writeheader()

        # ====================================================
        # First observation
        # ====================================================

        frame = world.tick()

        current_front = get_sensor_frame(
            q_front,
            frame,
            "front",
        )

        current_left = get_sensor_frame(
            q_left,
            frame,
            "left",
        )

        current_right = get_sensor_frame(
            q_right,
            frame,
            "right",
        )

        current_gnss = get_sensor_frame(
            q_gnss,
            frame,
            "GNSS",
        )

        current_imu = get_sensor_frame(
            q_imu,
            frame,
            "IMU",
        )

        route_idx = 0
        bad_deviation_frames = 0

        # ====================================================
        # Closed loop
        # ====================================================

        for i in range(
            args.frames
        ):

            ego_tf = ego.get_transform()

            ego_location = (
                ego_tf.location
            )

            speed_mps = get_speed_mps(
                ego
            )

            (
                route_idx,
                route_deviation,
            ) = closest_route_index(
                route,
                ego_location,
                route_idx,
            )

            destination_distance = (
                distance_2d(
                    ego_location,
                    destination,
                )
            )

            nav = get_neat_navigation(
                route_planner,
                current_gnss,
                current_imu,
            )

            rgb_front = carla_image_to_rgb(
                current_front
            )

            rgb_left = carla_image_to_rgb(
                current_left
            )

            rgb_right = carla_image_to_rgb(
                current_right
            )

            bootstrap = (
                i < config.seq_len
            )

            if bootstrap:

                result = {
                    "steer":
                        0.0,

                    "throttle":
                        0.0,

                    "brake":
                        0.0,

                    "red_light_occ":
                        0,

                    "metadata": {
                        "desired_speed":
                            float("nan"),

                        "angle":
                            float("nan"),

                        "angle_last":
                            float("nan"),

                        "angle_target":
                            float("nan"),

                        "angle_final":
                            float("nan"),
                    },
                }

            else:

                result = run_neat(
                    net=net,
                    config=config,
                    plan_grid=plan_grid,
                    light_grid=light_grid,

                    rgb_front=rgb_front,
                    rgb_left=rgb_left,
                    rgb_right=rgb_right,

                    speed_mps=speed_mps,

                    target_xy=(
                        nav[
                            "target_point"
                        ]
                    ),

                    device=device,
                )

            metadata = (
                result[
                    "metadata"
                ]
            )

            control = carla.VehicleControl(
                steer=float(
                    result[
                        "steer"
                    ]
                ),

                throttle=float(
                    result[
                        "throttle"
                    ]
                ),

                brake=float(
                    result[
                        "brake"
                    ]
                ),
            )

            writer.writerow({
                "probe_idx":
                    i,

                "carla_frame":
                    frame,

                "ego_x":
                    ego_location.x,

                "ego_y":
                    ego_location.y,

                "ego_z":
                    ego_location.z,

                "ego_yaw":
                    ego_tf.rotation.yaw,

                "speed_mps":
                    speed_mps,

                "route_index":
                    route_idx,

                "route_deviation_m":
                    route_deviation,

                "destination_distance_m":
                    destination_distance,

                "gps_lat":
                    current_gnss.latitude,

                "gps_lon":
                    current_gnss.longitude,

                "compass":
                    nav[
                        "compass"
                    ],

                "target_x":
                    nav[
                        "target_point"
                    ][0],

                "target_y":
                    nav[
                        "target_point"
                    ][1],

                "steer":
                    result[
                        "steer"
                    ],

                "throttle":
                    result[
                        "throttle"
                    ],

                "brake":
                    result[
                        "brake"
                    ],

                "red_light_occ":
                    result[
                        "red_light_occ"
                    ],

                "desired_speed":
                    metadata.get(
                        "desired_speed",
                        float("nan"),
                    ),

                "angle":
                    metadata.get(
                        "angle",
                        float("nan"),
                    ),

                "angle_last":
                    metadata.get(
                        "angle_last",
                        float("nan"),
                    ),

                "angle_target":
                    metadata.get(
                        "angle_target",
                        float("nan"),
                    ),

                "angle_final":
                    metadata.get(
                        "angle_final",
                        float("nan"),
                    ),

                "bootstrap":
                    int(
                        bootstrap
                    ),
            })

            csv_file.flush()

            if (
                args.debug_every > 0
                and i % args.debug_every == 0
            ):

                debug_path = (
                    debug_dir
                    / f"{i:05d}.png"
                )

                save_debug_image(
                    debug_path,
                    rgb_left,
                    rgb_front,
                    rgb_right,
                    i,
                    speed_mps,
                    nav[
                        "target_point"
                    ],
                    result,
                )

            if (
                i < 10
                or i % 20 == 0
            ):

                print(
                    f"[{i:04d}] "
                    f"speed={speed_mps:5.2f} "
                    f"target=("
                    f"{nav['target_point'][0]:+.2f},"
                    f"{nav['target_point'][1]:+.2f}) "
                    f"S={result['steer']:+.3f} "
                    f"T={result['throttle']:.3f} "
                    f"B={result['brake']:.3f} "
                    f"RL={result['red_light_occ']} "
                    f"dest={destination_distance:.1f} "
                    f"dev={route_deviation:.2f}"
                )

            # -----------------------------------------------
            # Destination reached
            # -----------------------------------------------

            if (
                destination_distance
                <= args.destination_tolerance_m
            ):

                print(
                    "[DONE] destination reached."
                )

                break

            # -----------------------------------------------
            # Route deviation safety
            # -----------------------------------------------

            if (
                route_deviation
                > args.max_route_deviation_m
            ):

                bad_deviation_frames += 1

            else:

                bad_deviation_frames = 0

            if (
                bad_deviation_frames
                >= args.deviation_patience_frames
            ):

                print(
                    "[STOP] route deviation exceeded."
                )

                break

            # -----------------------------------------------
            # Apply NEAT control
            # -----------------------------------------------

            ego.apply_control(
                control
            )

            # -----------------------------------------------
            # Next synchronous frame
            # -----------------------------------------------

            frame = world.tick()

            current_front = get_sensor_frame(
                q_front,
                frame,
                "front",
            )

            current_left = get_sensor_frame(
                q_left,
                frame,
                "left",
            )

            current_right = get_sensor_frame(
                q_right,
                frame,
                "right",
            )

            current_gnss = get_sensor_frame(
                q_gnss,
                frame,
                "GNSS",
            )

            current_imu = get_sensor_frame(
                q_imu,
                frame,
                "IMU",
            )

    finally:

        if csv_file is not None:
            csv_file.close()

        for actor in reversed(
            actors
        ):

            try:
                actor.destroy()

            except Exception:
                pass

        try:
            world.apply_settings(
                original_settings
            )

        except Exception:
            pass

    print()
    print(
        "[output]",
        csv_path,
    )

    print(
        "[debug]",
        debug_dir,
    )


if __name__ == "__main__":
    main()