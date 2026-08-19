"""
tcp_carla_0915_closed_loop.py

Native TCP closed-loop baseline on CARLA 0.9.15.

This script:

1. Spawns an ego vehicle.
2. Builds an explicit CARLA route.
3. Attaches TCP's native RGB camera:
       900x256
       FOV 100
       x=-1.5, y=0, z=2.0
4. Attaches GNSS + IMU.
5. Uses TCP's original RoutePlanner(4.0, 50.0).
6. Runs the pretrained TCP model.
7. Applies TCP steer/throttle/brake to the ego.
8. Records:
       - annotated MP4
       - CSV telemetry
9. Stops at the destination or on excessive route deviation.

BasicAgent / TrafficManager do NOT control the ego.

The only controller is TCP.
"""

import argparse
import csv
import math
import queue
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import carla


# ============================================================
# Reuse validated TCP / route-probe utilities
# ============================================================

from tcp_carla_0915_probe import (
    HE_ROOT,
    DEFAULT_TCP_ROOT,
    DEFAULT_CHECKPOINT,
    carla_image_to_rgb,
    get_sensor_frame,
    get_vehicle_speed_mps,
    load_tcp_model,
    make_camera,
    prepare_tcp_rgb_native,
)

from tcp_carla_0915_route_probe import (
    TCPFusionState,
    build_route,
    closest_route_index,
    command_name,
    distance_2d,
    import_carla_agents,
    route_length,
    run_tcp_route,
)


# ============================================================
# TCP original RoutePlanner
# ============================================================

def import_tcp_route_planner(tcp_root):
    """
    TCP repository layout:

        TCP/
            leaderboard/
                team_code/
                    planner.py

    planner.py contains the RoutePlanner used by the
    original TCP leaderboard agent.
    """

    tcp_root = Path(
        tcp_root
    ).resolve()

    leaderboard_root = (
        tcp_root
        / "leaderboard"
    )

    planner_file = (
        leaderboard_root
        / "team_code"
        / "planner.py"
    )

    if not planner_file.exists():
        raise FileNotFoundError(
            "Could not find TCP planner:\n"
            f"{planner_file}"
        )

    if str(
        leaderboard_root
    ) not in sys.path:

        sys.path.insert(
            0,
            str(
                leaderboard_root
            ),
        )

    from team_code.planner import RoutePlanner

    return RoutePlanner


# ============================================================
# Additional CARLA sensors
# ============================================================

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

    # IMPORTANT:
    # Leave sensor_tick at CARLA's default 0.0.
    # This gives one GNSS measurement per simulation step
    # in our synchronous 20 Hz runner.

    transform = carla.Transform(
        carla.Location(
            x=0.0,
            y=0.0,
            z=0.0,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        ),
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

    # IMPORTANT:
    # Leave sensor_tick at CARLA's default 0.0.
    # We want one IMU sample for every synchronous
    # simulation step.

    transform = carla.Transform(
        carla.Location(
            x=0.0,
            y=0.0,
            z=0.0,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        ),
    )

    return world.spawn_actor(
        bp,
        transform,
        attach_to=ego,
    )

def get_named_sensor_frame(
    sensor_queue,
    target_frame,
    sensor_name,
    timeout=10.0,
):
    """
    Retrieve exactly target_frame from a sensor queue.

    Gives a useful error identifying the sensor if CARLA
    fails to produce the frame.
    """

    while True:

        try:
            data = sensor_queue.get(
                timeout=timeout
            )

        except queue.Empty:
            raise RuntimeError(
                f"{sensor_name} did not produce "
                f"CARLA frame {target_frame} "
                f"within {timeout:.1f} seconds."
            )

        if data.frame == target_frame:
            return data

        if data.frame < target_frame:
            continue

        raise RuntimeError(
            f"{sensor_name} skipped requested "
            f"CARLA frame {target_frame}; "
            f"received future frame {data.frame}."
        )

# ============================================================
# CARLA route -> TCP GPS route
# ============================================================

def build_tcp_gps_plan(
    carla_map,
    route,
):
    """
    Convert:

        [(carla.Waypoint, RoadOption), ...]

    into the GPS representation expected by TCP RoutePlanner:

        [({"lat": ..., "lon": ...}, RoadOption), ...]

    CARLA Map.transform_to_geolocation() ensures that
    route coordinates and GNSS measurements share the same
    geographical coordinate system.
    """

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
            "lat": float(
                geo.latitude
            ),
            "lon": float(
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
# TCP navigation input
# ============================================================

def get_tcp_navigation(
    route_planner,
    gnss,
    imu,
):
    """
    Reproduce TCP agent navigation processing.

    Original TCP:

        gps = (gps - mean) * scale

        next_wp, next_cmd =
            RoutePlanner.run_step(gps)

        theta = compass + pi/2

        target_point =
            R.T @ (next_wp - gps)
    """

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

    local_command_point = np.array(
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
            local_command_point
        )
    )

    cmd_value = int(
        getattr(
            next_cmd,
            "value",
            4,
        )
    )

    if cmd_value < 0:
        cmd_value = 4

    if not (
        1 <= cmd_value <= 6
    ):
        cmd_value = 4

    return {
        "position":
            position,

        "next_wp":
            next_wp,

        "command":
            next_cmd,

        "command_value":
            cmd_value,

        "target_point":
            target_point,

        "compass":
            compass,
    }


# ============================================================
# Video
# ============================================================

def make_video_frame(
    rgb,
    frame_idx,
    speed_mps,
    nav,
    tcp,
    destination_distance,
    route_deviation,
    bootstrap=False,
):
    """
    Produce annotated TCP camera frame.

    The image fed into TCP remains untouched.

    Annotation is applied only to the copy written to MP4.
    """

    frame = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    # Dark transparent HUD.
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (900, 102),
        (0, 0, 0),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        0.58,
        frame,
        0.42,
        0.0,
    )

    target = (
        nav[
            "target_point"
        ]
    )

    cmd = command_name(
        nav[
            "command"
        ]
    )

    if bootstrap:

        control_line = (
            "TCP bootstrap frame "
            "| control = 0"
        )

    else:

        control_line = (
            f"TCP  "
            f"S={tcp['steer']:+.3f}  "
            f"T={tcp['throttle']:.3f}  "
            f"B={tcp['brake']:.3f}  "
            f"status={tcp['status_used']}"
        )

    desired = (
        tcp.get(
            "desired_speed",
            float("nan"),
        )
    )

    lines = [
        (
            f"Frame {frame_idx:05d}  "
            f"Speed {speed_mps:.2f} m/s  "
            f"({speed_mps * 3.6:.1f} km/h)"
        ),

        (
            f"Command {cmd}  "
            f"Target "
            f"({target[0]:+.2f}, "
            f"{target[1]:+.2f})"
        ),

        control_line,

        (
            f"Desired {desired:.2f} m/s  "
            f"Dest {destination_distance:.1f} m  "
            f"RouteDev {route_deviation:.2f} m"
        ),
    ]

    y = 20

    for line in lines:

        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 24

    return frame


# ============================================================
# Bootstrap TCP result
# ============================================================

def make_bootstrap_result():
    """
    Original TCP agent outputs zero control on its first
    sequence initialization step.
    """

    return {
        "steer":
            0.0,

        "throttle":
            0.0,

        "brake":
            0.0,

        "status_used":
            0,

        "status_next":
            0,

        "steer_ctrl":
            0.0,

        "throttle_ctrl":
            0.0,

        "brake_ctrl":
            0.0,

        "steer_traj":
            0.0,

        "throttle_traj":
            0.0,

        "brake_traj":
            0.0,

        "pred_speed":
            float("nan"),

        "desired_speed":
            float("nan"),

        "angle_target":
            float("nan"),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # CARLA
    # --------------------------------------------------------

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
        help=(
            "Negative = automatically "
            "choose destination."
        ),
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

    # --------------------------------------------------------
    # TCP
    # --------------------------------------------------------

    parser.add_argument(
        "--tcp-root",
        default=str(
            DEFAULT_TCP_ROOT
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

    # --------------------------------------------------------
    # Termination / safety
    # --------------------------------------------------------

    parser.add_argument(
        "--destination-tolerance-m",
        type=float,
        default=2.5,
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

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    parser.add_argument(
        "--output-csv",
        default=str(
            HE_ROOT
            / "driving_models"
            / "TCP"
            / "outputs"
            / "tcp_closed_loop.csv"
        ),
    )

    parser.add_argument(
        "--video-output",
        default=str(
            HE_ROOT
            / "driving_models"
            / "TCP"
            / "outputs"
            / "tcp_closed_loop.mp4"
        ),
    )

    parser.add_argument(
        "--no-video-overlay",
        action="store_true",
        help=(
            "Write clean TCP camera "
            "frames without HUD."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Imports requiring external paths
    # ========================================================

    (
        _BasicAgent,
        GlobalRoutePlanner,
    ) = import_carla_agents(
        args.carla_pythonapi
    )

    tcp_root = Path(
        args.tcp_root
    ).resolve()

    RoutePlanner = (
        import_tcp_route_planner(
            tcp_root
        )
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        args.device
    )

    if device.type == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested "
                "but unavailable."
            )

        print(
            "[GPU]",
            torch.cuda.get_device_name(
                0
            ),
        )

    # ========================================================
    # TCP model
    # ========================================================

    checkpoint = Path(
        args.checkpoint
    ).resolve()

    print(
        "[TCP root]",
        tcp_root,
    )

    print(
        "[checkpoint]",
        checkpoint,
    )

    net, config = load_tcp_model(
        tcp_root=tcp_root,
        checkpoint_path=checkpoint,
        device=device,
    )

    print(
        "[TCP] pred_len:",
        config.pred_len,
    )

    # ========================================================
    # Outputs
    # ========================================================

    csv_path = Path(
        args.output_csv
    )

    video_path = Path(
        args.video_output
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # CARLA connection
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

    ego = None
    camera = None
    gnss_sensor = None
    imu_sensor = None
    video_writer = None
    csv_file = None

    try:

        # ====================================================
        # Synchronous mode
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = (
            True
        )

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
        # Route / ego
        # ====================================================

        carla_map = (
            world.get_map()
        )

        spawn_points = (
            carla_map
            .get_spawn_points()
        )

        if not spawn_points:
            raise RuntimeError(
                "No CARLA spawn points."
            )

        spawn_idx = (
            args.spawn_index
            % len(
                spawn_points
            )
        )

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
                f"Could not spawn ego "
                f"at index {spawn_idx}."
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

        destination_location = (
            route[-1][0]
            .transform
            .location
        )

        print(
            "[CARLA] ego:",
            ego.id,
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
        # TCP original RoutePlanner
        # ====================================================

        tcp_gps_plan = (
            build_tcp_gps_plan(
                carla_map,
                route,
            )
        )

        tcp_route_planner = (
            RoutePlanner(
                4.0,
                50.0,
            )
        )

        tcp_route_planner.set_route(
            tcp_gps_plan,
            gps=True,
        )

        print(
            "[TCP planner]",
            "RoutePlanner(4.0, 50.0)",
        )

        # ====================================================
        # Sensors
        # ====================================================

        camera = make_camera(
            world=world,
            ego=ego,

            width=900,
            height=256,
            fov=100,

            x=-1.5,
            y=0.0,
            z=2.0,

            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        )

        gnss_sensor = make_gnss(
            world,
            ego,
        )

        imu_sensor = make_imu(
            world,
            ego,
        )

        camera_queue = (
            queue.Queue()
        )

        gnss_queue = (
            queue.Queue()
        )

        imu_queue = (
            queue.Queue()
        )

        camera.listen(
            camera_queue.put
        )

        gnss_sensor.listen(
            gnss_queue.put
        )

        imu_sensor.listen(
            imu_queue.put
        )

        # ====================================================
        # Stabilize spawn
        # ====================================================

        hold_control = (
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )

        print(
            "[init] stabilizing ego..."
        )

        for _ in range(5):

            ego.apply_control(
                hold_control
            )

            frame = world.tick()

            get_sensor_frame(
                camera_queue,
                frame,
            )

            get_sensor_frame(
                gnss_queue,
                frame,
            )

            get_sensor_frame(
                imu_queue,
                frame,
            )

        # Put the vehicle back exactly on
        # the requested spawn pose after physics settles.

        ego.set_transform(
            spawn_points[
                spawn_idx
            ]
        )

        ego.set_target_velocity(
            carla.Vector3D(
                x=0.0,
                y=0.0,
                z=0.0,
            )
        )

        ego.set_target_angular_velocity(
            carla.Vector3D(
                x=0.0,
                y=0.0,
                z=0.0,
            )
        )

        ego.apply_control(
            hold_control
        )

        current_frame = (
            world.tick()
        )

        current_image = (
            get_named_sensor_frame(
                camera_queue,
                current_frame,
                "RGB camera",
            )
        )

        current_gnss = (
            get_named_sensor_frame(
                gnss_queue,
                current_frame,
                "GNSS",
            )
        )

        current_imu = (
            get_named_sensor_frame(
                imu_queue,
                current_frame,
                "IMU",
            )
        )

        # ====================================================
        # Video writer
        # ====================================================

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        video_writer = (
            cv2.VideoWriter(
                str(
                    video_path
                ),
                fourcc,
                float(
                    args.fps
                ),
                (
                    900,
                    256,
                ),
            )
        )

        if not video_writer.isOpened():

            raise RuntimeError(
                "Could not open MP4 writer:\n"
                f"{video_path}"
            )

        # ====================================================
        # CSV
        # ====================================================

        csv_fields = [
            "probe_idx",
            "carla_frame",

            "x",
            "y",
            "z",
            "yaw",

            "speed_mps",
            "speed_kmh",

            "route_index",
            "route_deviation_m",
            "destination_distance_m",

            "command_name",
            "command_value",

            "gps_lat",
            "gps_lon",
            "compass",

            "target_x",
            "target_y",

            "tcp_steer",
            "tcp_throttle",
            "tcp_brake",

            "tcp_status_used",
            "tcp_status_next",

            "steer_ctrl",
            "throttle_ctrl",
            "brake_ctrl",

            "steer_traj",
            "throttle_traj",
            "brake_traj",

            "pred_speed",
            "desired_speed",
            "angle_target",

            "bootstrap",
        ]

        csv_file = open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        # ====================================================
        # TCP controller state
        # ====================================================

        fusion = (
            TCPFusionState()
        )

        route_idx = 0

        deviation_counter = 0

        completed_frames = 0

        # ====================================================
        # Start banner
        # ====================================================

        print()
        print("=" * 78)
        print(
            "TCP NATIVE CLOSED-LOOP BASELINE"
        )
        print("=" * 78)

        print(
            "ego controller : TCP"
        )

        print(
            "camera         : "
            "900x256 FOV100 "
            "x=-1.5 z=2.0"
        )

        print(
            "navigation     : "
            "TCP RoutePlanner(4.0, 50.0)"
        )

        print(
            "video          :",
            video_path,
        )

        print(
            "telemetry      :",
            csv_path,
        )

        print("=" * 78)
        print()

        # ====================================================
        # Main closed-loop
        # ====================================================

        for i in range(
            args.frames
        ):

            # ------------------------------------------------
            # Current synchronized sensor frame
            # ------------------------------------------------

            rgb = (
                prepare_tcp_rgb_native(
                    carla_image_to_rgb(
                        current_image
                    )
                )
            )

            speed_mps = (
                get_vehicle_speed_mps(
                    ego
                )
            )

            nav = (
                get_tcp_navigation(
                    tcp_route_planner,
                    current_gnss,
                    current_imu,
                )
            )

            # ------------------------------------------------
            # Original TCP has seq_len=1 and returns zero
            # control on its first initialization frame.
            # ------------------------------------------------

            bootstrap = (
                i == 0
            )

            if bootstrap:

                tcp = (
                    make_bootstrap_result()
                )

            else:

                tcp = run_tcp_route(
                    net,
                    rgb,
                    speed_mps,
                    nav[
                        "target_point"
                    ],
                    nav[
                        "command_value"
                    ],
                    device,
                    fusion,
                )

            # ------------------------------------------------
            # TCP is THE controller now.
            # ------------------------------------------------

            control = (
                carla.VehicleControl(
                    steer=float(
                        tcp[
                            "steer"
                        ]
                    ),

                    throttle=float(
                        tcp[
                            "throttle"
                        ]
                    ),

                    brake=float(
                        tcp[
                            "brake"
                        ]
                    ),

                    hand_brake=False,

                    manual_gear_shift=False,
                )
            )

            ego.apply_control(
                control
            )

            # ------------------------------------------------
            # Ego state
            # ------------------------------------------------

            transform = (
                ego.get_transform()
            )

            location = (
                transform.location
            )

            route_idx = (
                closest_route_index(
                    route,
                    location,
                    route_idx,
                )
            )

            route_location = (
                route[
                    route_idx
                ][0]
                .transform
                .location
            )

            route_deviation = (
                distance_2d(
                    location,
                    route_location,
                )
            )

            destination_distance = (
                distance_2d(
                    location,
                    destination_location,
                )
            )

            # ------------------------------------------------
            # Route deviation safety monitor
            # ------------------------------------------------

            if (
                route_deviation
                >
                args.max_route_deviation_m
            ):

                deviation_counter += 1

            else:

                deviation_counter = 0

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            writer.writerow({
                "probe_idx":
                    i,

                "carla_frame":
                    current_frame,

                "x":
                    float(
                        location.x
                    ),

                "y":
                    float(
                        location.y
                    ),

                "z":
                    float(
                        location.z
                    ),

                "yaw":
                    float(
                        transform.rotation.yaw
                    ),

                "speed_mps":
                    speed_mps,

                "speed_kmh":
                    speed_mps * 3.6,

                "route_index":
                    route_idx,

                "route_deviation_m":
                    route_deviation,

                "destination_distance_m":
                    destination_distance,

                "command_name":
                    command_name(
                        nav[
                            "command"
                        ]
                    ),

                "command_value":
                    nav[
                        "command_value"
                    ],

                "gps_lat":
                    float(
                        current_gnss.latitude
                    ),

                "gps_lon":
                    float(
                        current_gnss.longitude
                    ),

                "compass":
                    nav[
                        "compass"
                    ],

                "target_x":
                    float(
                        nav[
                            "target_point"
                        ][0]
                    ),

                "target_y":
                    float(
                        nav[
                            "target_point"
                        ][1]
                    ),

                "tcp_steer":
                    tcp[
                        "steer"
                    ],

                "tcp_throttle":
                    tcp[
                        "throttle"
                    ],

                "tcp_brake":
                    tcp[
                        "brake"
                    ],

                "tcp_status_used":
                    tcp[
                        "status_used"
                    ],

                "tcp_status_next":
                    tcp[
                        "status_next"
                    ],

                "steer_ctrl":
                    tcp[
                        "steer_ctrl"
                    ],

                "throttle_ctrl":
                    tcp[
                        "throttle_ctrl"
                    ],

                "brake_ctrl":
                    tcp[
                        "brake_ctrl"
                    ],

                "steer_traj":
                    tcp[
                        "steer_traj"
                    ],

                "throttle_traj":
                    tcp[
                        "throttle_traj"
                    ],

                "brake_traj":
                    tcp[
                        "brake_traj"
                    ],

                "pred_speed":
                    tcp[
                        "pred_speed"
                    ],

                "desired_speed":
                    tcp[
                        "desired_speed"
                    ],

                "angle_target":
                    tcp[
                        "angle_target"
                    ],

                "bootstrap":
                    int(
                        bootstrap
                    ),
            })

            # ------------------------------------------------
            # MP4 frame
            # ------------------------------------------------

            if args.no_video_overlay:

                video_frame = (
                    cv2.cvtColor(
                        rgb,
                        cv2.COLOR_RGB2BGR,
                    )
                )

            else:

                video_frame = (
                    make_video_frame(
                        rgb=rgb,
                        frame_idx=i,
                        speed_mps=speed_mps,
                        nav=nav,
                        tcp=tcp,
                        destination_distance=(
                            destination_distance
                        ),
                        route_deviation=(
                            route_deviation
                        ),
                        bootstrap=bootstrap,
                    )
                )

            video_writer.write(
                video_frame
            )

            completed_frames += 1

            # ------------------------------------------------
            # Console
            # ------------------------------------------------

            if (
                i < 10
                or i % 20 == 0
            ):

                target = (
                    nav[
                        "target_point"
                    ]
                )

                print(
                    f"[{i:04d}] "
                    f"v={speed_mps:5.2f} "
                    f"cmd="
                    f"{command_name(nav['command']):>10s} "
                    f"tp=("
                    f"{target[0]:+5.2f},"
                    f"{target[1]:+5.2f}) | "
                    f"TCP "
                    f"S={tcp['steer']:+.3f} "
                    f"T={tcp['throttle']:.3f} "
                    f"B={tcp['brake']:.3f} "
                    f"status="
                    f"{tcp['status_used']} | "
                    f"dest="
                    f"{destination_distance:5.1f}m "
                    f"dev="
                    f"{route_deviation:4.2f}m"
                )

            # ------------------------------------------------
            # Destination reached
            # ------------------------------------------------

            if (
                destination_distance
                <=
                args.destination_tolerance_m
            ):

                print()
                print(
                    "[route] destination reached"
                )

                ego.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                    )
                )

                break

            # ------------------------------------------------
            # Safety: TCP departed route
            # ------------------------------------------------

            if (
                deviation_counter
                >=
                args.deviation_patience_frames
            ):

                print()
                print(
                    "[SAFETY] Route deviation exceeded "
                    f"{args.max_route_deviation_m:.1f} m "
                    f"for {deviation_counter} frames."
                )

                print(
                    "[SAFETY] Applying emergency brake "
                    "and terminating run."
                )

                ego.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                    )
                )

                # Record one final stopped simulation tick.
                world.tick()

                break

            # ------------------------------------------------
            # Advance closed-loop simulation
            # ------------------------------------------------

            current_frame = (
                world.tick()
            )

            current_image = (
                get_sensor_frame(
                    camera_queue,
                    current_frame,
                )
            )

            current_gnss = (
                get_sensor_frame(
                    gnss_queue,
                    current_frame,
                )
            )

            current_imu = (
                get_sensor_frame(
                    imu_queue,
                    current_frame,
                )
            )

        # ====================================================
        # Complete
        # ====================================================

        csv_file.flush()

        print()
        print("=" * 78)
        print(
            "TCP CLOSED-LOOP RUN COMPLETE"
        )
        print(
            "frames:",
            completed_frames,
        )
        print(
            "video:",
            video_path,
        )
        print(
            "csv:",
            csv_path,
        )
        print("=" * 78)

    finally:

        print(
            "[cleanup]"
        )

        if csv_file is not None:
            csv_file.close()

        if video_writer is not None:
            video_writer.release()

        if camera is not None:

            try:
                camera.stop()
            except Exception:
                pass

            camera.destroy()

        if gnss_sensor is not None:

            try:
                gnss_sensor.stop()
            except Exception:
                pass

            gnss_sensor.destroy()

        if imu_sensor is not None:

            try:
                imu_sensor.stop()
            except Exception:
                pass

            imu_sensor.destroy()

        if ego is not None:
            ego.destroy()

        world.apply_settings(
            original_settings
        )


if __name__ == "__main__":
    main()