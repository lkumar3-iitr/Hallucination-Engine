"""
tcp_carla_0915_route_probe.py

Route-aware TCP observation probe for CARLA 0.9.15.

CARLA BasicAgent drives an explicit GlobalRoutePlanner route.
TCP sees its native 900x256 camera and the same route command/target point,
but TCP controls are NOT applied.
"""

import argparse
import csv
import math
import os
import queue
import sys
from collections import deque
from pathlib import Path

import carla
import numpy as np
import torch

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
    rgb_to_tcp_tensor,
)


# ============================================================
# CARLA agents import
# ============================================================

def import_carla_agents(pythonapi_hint=None):
    def _load():
        from agents.navigation.basic_agent import BasicAgent
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        return BasicAgent, GlobalRoutePlanner

    try:
        return _load()
    except ModuleNotFoundError:
        pass

    candidates = []

    if pythonapi_hint:
        candidates.append(
            Path(pythonapi_hint)
        )

    carla_root = os.environ.get(
        "CARLA_ROOT"
    )

    if carla_root:
        root = Path(carla_root)

        candidates += [
            root / "PythonAPI" / "carla",
            root / "PythonAPI",
            root,
        ]

    for base in (
        Path.cwd(),
        HE_ROOT,
        HE_ROOT.parent,
    ):
        candidates += [
            base / "PythonAPI" / "carla",
            base / "PythonAPI",
            base / "carla",
        ]

    for p in candidates:
        p = p.expanduser().resolve()

        if (p / "agents").is_dir():
            sys.path.insert(
                0,
                str(p),
            )

        elif (
            p
            / "carla"
            / "agents"
        ).is_dir():
            sys.path.insert(
                0,
                str(p / "carla"),
            )

        else:
            continue

        try:
            return _load()

        except ModuleNotFoundError:
            pass

    raise RuntimeError(
        "Could not import CARLA's agents package.\n"
        "Pass, for example:\n"
        "  --carla-pythonapi "
        "D:\\CARLA_0.9.15\\PythonAPI\\carla"
    )


# ============================================================
# Route helpers
# ============================================================

def distance_2d(a, b):
    return math.hypot(
        float(a.x) - float(b.x),
        float(a.y) - float(b.y),
    )


def route_length(route):
    return sum(
        distance_2d(
            route[i][0]
            .transform
            .location,

            route[i + 1][0]
            .transform
            .location,
        )
        for i in range(
            len(route) - 1
        )
    )


def build_route(
    grp,
    spawn_points,
    start_idx,
    destination_idx,
    desired_m,
):
    start = (
        spawn_points[start_idx]
        .location
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
                "Destination equals start spawn."
            )

        route = grp.trace_route(
            start,
            spawn_points[
                destination_idx
            ].location,
        )

        if len(route) < 2:
            raise RuntimeError(
                "GlobalRoutePlanner returned "
                "no usable route."
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

        d = distance_2d(
            start,
            tf.location,
        )

        if d >= 35.0:
            candidates.append(
                (
                    abs(
                        d
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
        "Could not auto-select a useful "
        "destination. "
        "Pass --destination-index explicitly."
    )


def closest_route_index(
    route,
    vehicle_location,
    previous_idx,
):
    """
    Track progress along the route.

    Search slightly behind previous_idx as well,
    because the vehicle may sit between sampled waypoints.
    """

    start = max(
        0,
        previous_idx - 5,
    )

    end = min(
        len(route),
        previous_idx + 100,
    )

    best_idx = start
    best_d = float("inf")

    for idx in range(
        start,
        end,
    ):
        route_location = (
            route[idx][0]
            .transform
            .location
        )

        d = distance_2d(
            route_location,
            vehicle_location,
        )

        if d < best_d:
            best_idx = idx
            best_d = d

    return best_idx


def target_route_index(
    route,
    current_idx,
    lookahead_m,
):
    """
    Move forward along the route until approximately
    lookahead_m metres have accumulated.
    """

    idx = current_idx
    dist = 0.0

    while (
        idx + 1 < len(route)
        and dist < lookahead_m
    ):
        a = (
            route[idx][0]
            .transform
            .location
        )

        b = (
            route[idx + 1][0]
            .transform
            .location
        )

        dist += distance_2d(
            a,
            b,
        )

        idx += 1

    return idx, dist


def world_to_tcp_target(
    ego_transform,
    target_location,
):
    """
    Convert world point -> TCP target-point convention.

    CARLA ego-local:
        +forward = forward
        +right   = right

    TCP PID convention:
        target[0] = lateral / right
        target[1] = negative forward

    Therefore a target straight ahead should
    be approximately:

        [0, -distance]
    """

    ego = (
        ego_transform.location
    )

    dx = (
        float(target_location.x)
        - float(ego.x)
    )

    dy = (
        float(target_location.y)
        - float(ego.y)
    )

    yaw = math.radians(
        float(
            ego_transform
            .rotation
            .yaw
        )
    )

    c = math.cos(yaw)
    s = math.sin(yaw)

    forward = (
        dx * c
        + dy * s
    )

    right = (
        -dx * s
        + dy * c
    )

    return np.array(
        [
            right,
            -forward,
        ],
        dtype=np.float32,
    )


def command_value(command):
    value = int(
        getattr(
            command,
            "value",
            4,
        )
    )

    # TCP maps invalid / VOID commands
    # back to LANEFOLLOW.
    if not (
        1 <= value <= 6
    ):
        value = 4

    return value


def command_name(command):
    return str(
        getattr(
            command,
            "name",
            command,
        )
    )


# ============================================================
# TCP route-conditioned state
# ============================================================

def make_tcp_state(
    speed_mps,
    target_xy,
    cmd_value,
    device,
):
    speed = torch.tensor(
        [
            [
                float(
                    speed_mps
                )
                / 12.0
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    target = torch.tensor(
        [
            [
                float(
                    target_xy[0]
                ),
                float(
                    target_xy[1]
                ),
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    cmd = torch.zeros(
        (1, 6),
        dtype=torch.float32,
        device=device,
    )

    cmd[
        0,
        cmd_value - 1,
    ] = 1.0

    state = torch.cat(
        [
            speed,
            target,
            cmd,
        ],
        dim=1,
    )

    return (
        state,
        target,
    )


# ============================================================
# TCP official control fusion state
# ============================================================

class TCPFusionState:
    """
    Reproduces the agent-level steering-history logic
    used by the original TCP agent.
    """

    def __init__(self):
        self.status = 0

        self.last_steers = deque(
            maxlen=20
        )

    def fuse(
        self,
        ctrl,
        traj,
    ):
        alpha = 0.3

        status_used = (
            self.status
        )

        # ----------------------------------------------------
        # Normal mode:
        #   more trajectory branch
        # ----------------------------------------------------

        if status_used == 0:
            w_ctrl = alpha
            w_traj = (
                1.0 - alpha
            )

        # ----------------------------------------------------
        # Turning mode:
        #   more direct-control branch
        # ----------------------------------------------------

        else:
            w_ctrl = (
                1.0 - alpha
            )
            w_traj = alpha

        steer = np.clip(
            w_ctrl * ctrl[0]
            + w_traj * traj[0],
            -1.0,
            1.0,
        )

        throttle = np.clip(
            w_ctrl * ctrl[1]
            + w_traj * traj[1],
            0.0,
            0.75,
        )

        brake = np.clip(
            w_ctrl * ctrl[2]
            + w_traj * traj[2],
            0.0,
            1.0,
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

        # ----------------------------------------------------
        # Original TCP final post-processing
        # ----------------------------------------------------

        if brake > 0.5:
            throttle = 0.0

        # ----------------------------------------------------
        # Update turning status for NEXT frame
        # ----------------------------------------------------

        self.last_steers.append(
            abs(steer)
        )

        num_turning = sum(
            s > 0.10
            for s in self.last_steers
        )

        self.status = (
            1
            if num_turning > 10
            else 0
        )

        return (
            steer,
            throttle,
            brake,
            status_used,
            self.status,
        )


# ============================================================
# TCP inference
# ============================================================

@torch.no_grad()
def run_tcp_route(
    net,
    rgb,
    speed_mps,
    target_xy,
    cmd_value,
    device,
    fusion,
):
    image = rgb_to_tcp_tensor(
        rgb,
        device,
    )

    state, target = (
        make_tcp_state(
            speed_mps,
            target_xy,
            cmd_value,
            device,
        )
    )

    velocity = torch.tensor(
        [
            float(
                speed_mps
            )
        ],
        dtype=torch.float32,
        device=device,
    )

    pred = net(
        image,
        state,
        target,
    )

    # --------------------------------------------------------
    # TCP direct-control branch
    # --------------------------------------------------------

    (
        steer_c,
        throttle_c,
        brake_c,
        _,
    ) = net.process_action(
        pred,
        cmd_value,
        velocity,
        target,
    )

    # --------------------------------------------------------
    # TCP trajectory/PID branch
    # --------------------------------------------------------

    (
        steer_t,
        throttle_t,
        brake_t,
        meta_t,
    ) = net.control_pid(
        pred[
            "pred_wp"
        ].clone(),

        velocity,

        target.clone(),
    )

    # Same cleanup as original TCP agent.
    if brake_t < 0.05:
        brake_t = 0.0

    if (
        throttle_t
        > brake_t
    ):
        brake_t = 0.0

    ctrl = (
        float(steer_c),
        float(throttle_c),
        float(brake_c),
    )

    traj = (
        float(steer_t),
        float(throttle_t),
        float(brake_t),
    )

    (
        steer,
        throttle,
        brake,
        status_used,
        status_next,
    ) = fusion.fuse(
        ctrl,
        traj,
    )

    pred_speed = float(
        pred[
            "pred_speed"
        ]
        .detach()
        .cpu()
        .reshape(-1)[0]
    )

    return {
        "steer":
            steer,

        "throttle":
            throttle,

        "brake":
            brake,

        "status_used":
            status_used,

        "status_next":
            status_next,

        "steer_ctrl":
            ctrl[0],

        "throttle_ctrl":
            ctrl[1],

        "brake_ctrl":
            ctrl[2],

        "steer_traj":
            traj[0],

        "throttle_traj":
            traj[1],

        "brake_traj":
            traj[2],

        "pred_speed":
            pred_speed,

        "desired_speed":
            float(
                meta_t.get(
                    "desired_speed",
                    float("nan"),
                )
            ),

        "angle_target":
            float(
                meta_t.get(
                    "angle_target",
                    float("nan"),
                )
            ),
    }


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
        help=(
            "Negative means choose "
            "destination automatically."
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
        "--tcp-target-distance-m",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--driver-speed-kmh",
        type=float,
        default=30.0,
    )

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

    parser.add_argument(
        "--carla-pythonapi",
        default=None,
    )

    parser.add_argument(
        "--output",
        default=str(
            HE_ROOT
            / "driving_models"
            / "TCP"
            / "outputs"
            / "tcp_route_probe.csv"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CARLA navigation agents
    # --------------------------------------------------------

    (
        BasicAgent,
        GlobalRoutePlanner,
    ) = import_carla_agents(
        args.carla_pythonapi
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        args.device
    )

    if device.type == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        print(
            "[GPU]",
            torch.cuda.get_device_name(
                0
            ),
        )

    # --------------------------------------------------------
    # TCP
    # --------------------------------------------------------

    tcp_root = Path(
        args.tcp_root
    ).resolve()

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

    # --------------------------------------------------------
    # CARLA
    # --------------------------------------------------------

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
    csv_file = None

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        # ----------------------------------------------------
        # Synchronous CARLA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Spawn ego
        # ----------------------------------------------------

        carla_map = (
            world.get_map()
        )

        spawn_points = (
            carla_map
            .get_spawn_points()
        )

        if not spawn_points:
            raise RuntimeError(
                "No spawn points available."
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

        # ----------------------------------------------------
        # Build explicit route
        # ----------------------------------------------------

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

        print(
            "[CARLA] ego:",
            ego.id,
            "spawn:",
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



        # ----------------------------------------------------
        # TCP native camera
        # ----------------------------------------------------

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

        sensor_queue = (
            queue.Queue()
        )

        camera.listen(
            sensor_queue.put
        )

        # ----------------------------------------------------
        # Warm up sensor while keeping ego stationary.
        #
        # Important:
        # A newly spawned CARLA vehicle can physically settle
        # during the first few ticks. If BasicAgent is initialized
        # before this happens, its first route waypoint may end up
        # behind the ego.
        # ----------------------------------------------------

        hold_control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
        )

        for _ in range(5):

            ego.apply_control(
                hold_control
            )

            frame = world.tick()

            get_sensor_frame(
                sensor_queue,
                frame,
            )


        # ----------------------------------------------------
        # Restore the exact requested spawn pose after physics
        # settling and explicitly remove any residual motion.
        # ----------------------------------------------------

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

        # One synchronized frame after the reset so camera and
        # vehicle state agree.
        frame = world.tick()

        get_sensor_frame(
            sensor_queue,
            frame,
        )


        # ----------------------------------------------------
        # Create BasicAgent only AFTER ego initialization is
        # completely stable.
        # ----------------------------------------------------

        driver = BasicAgent(
            ego,

            target_speed=float(
                args.driver_speed_kmh
            ),

            map_inst=carla_map,

            grp_inst=grp,
        )

        # Diagnostic route probe only:
        # do not stop at traffic lights.
        driver.ignore_traffic_lights(
            True
        )

        driver.set_global_plan(
            route
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        fields = [
            "probe_idx",
            "carla_frame",

            "route_index",
            "target_route_index",

            "command_name",
            "command_value",

            "target_x",
            "target_y",
            "target_distance_m",

            "speed_mps",

            "driver_steer",
            "driver_throttle",
            "driver_brake",

            "tcp_steer",
            "tcp_throttle",
            "tcp_brake",
            "tcp_pred_speed",

            "tcp_status_used",
            "tcp_status_next",

            "steer_ctrl",
            "throttle_ctrl",
            "brake_ctrl",

            "steer_traj",
            "throttle_traj",
            "brake_traj",

            "desired_speed",
            "angle_target",
        ]

        csv_file = open(
            output,
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
        )

        writer.writeheader()

        # ----------------------------------------------------
        # TCP agent-level state
        # ----------------------------------------------------

        route_idx = 0

        fusion = (
            TCPFusionState()
        )

        completed = 0

        print()
        print("=" * 78)
        print(
            "TCP ROUTE-AWARE OBSERVATION PROBE"
        )
        print("=" * 78)

        print(
            "driver      : CARLA BasicAgent"
        )

        print(
            "TCP camera  : "
            "900x256 FOV100 x=-1.5 z=2.0"
        )

        print(
            "TCP controls: NOT applied"
        )

        print(
            "target      :",
            f"{args.tcp_target_distance_m:.1f} m "
            "route lookahead",
        )

        print("=" * 78)

        # ----------------------------------------------------
        # Main loop
        # ----------------------------------------------------

        for i in range(
            args.frames
        ):
            if driver.done():

                print(
                    "[route] destination reached"
                )

                break

            # -----------------------------------------------
            # Independent CARLA driver
            # -----------------------------------------------

            driver_control = (
                driver.run_step()
            )

            ego.apply_control(
                driver_control
            )

            # -----------------------------------------------
            # Advance one synchronous frame
            # -----------------------------------------------

            carla_frame = (
                world.tick()
            )

            image = get_sensor_frame(
                sensor_queue,
                carla_frame,
            )

            rgb = (
                prepare_tcp_rgb_native(
                    carla_image_to_rgb(
                        image
                    )
                )
            )

            # -----------------------------------------------
            # Locate ego on shared route
            # -----------------------------------------------

            ego_transform = (
                ego.get_transform()
            )

            route_idx = (
                closest_route_index(
                    route,
                    ego_transform.location,
                    route_idx,
                )
            )

            (
                target_idx,
                target_dist,
            ) = target_route_index(
                route,
                route_idx,
                args.tcp_target_distance_m,
            )

            (
                target_wp,
                road_option,
            ) = route[
                target_idx
            ]

            # -----------------------------------------------
            # Route point -> TCP local target
            # -----------------------------------------------

            target_xy = (
                world_to_tcp_target(
                    ego_transform,
                    target_wp
                    .transform
                    .location,
                )
            )

            cmd_value = (
                command_value(
                    road_option
                )
            )

            speed = (
                get_vehicle_speed_mps(
                    ego
                )
            )

            # -----------------------------------------------
            # TCP observation only
            # -----------------------------------------------

            tcp = run_tcp_route(
                net,
                rgb,
                speed,
                target_xy,
                cmd_value,
                device,
                fusion,
            )

            # -----------------------------------------------
            # Log
            # -----------------------------------------------

            writer.writerow({
                "probe_idx":
                    i,

                "carla_frame":
                    carla_frame,

                "route_index":
                    route_idx,

                "target_route_index":
                    target_idx,

                "command_name":
                    command_name(
                        road_option
                    ),

                "command_value":
                    cmd_value,

                "target_x":
                    float(
                        target_xy[0]
                    ),

                "target_y":
                    float(
                        target_xy[1]
                    ),

                "target_distance_m":
                    target_dist,

                "speed_mps":
                    speed,

                "driver_steer":
                    float(
                        driver_control.steer
                    ),

                "driver_throttle":
                    float(
                        driver_control.throttle
                    ),

                "driver_brake":
                    float(
                        driver_control.brake
                    ),

                "tcp_steer":
                    tcp["steer"],

                "tcp_throttle":
                    tcp["throttle"],

                "tcp_brake":
                    tcp["brake"],

                "tcp_pred_speed":
                    tcp["pred_speed"],

                "tcp_status_used":
                    tcp["status_used"],

                "tcp_status_next":
                    tcp["status_next"],

                "steer_ctrl":
                    tcp["steer_ctrl"],

                "throttle_ctrl":
                    tcp["throttle_ctrl"],

                "brake_ctrl":
                    tcp["brake_ctrl"],

                "steer_traj":
                    tcp["steer_traj"],

                "throttle_traj":
                    tcp["throttle_traj"],

                "brake_traj":
                    tcp["brake_traj"],

                "desired_speed":
                    tcp["desired_speed"],

                "angle_target":
                    tcp["angle_target"],
            })

            completed += 1

            # -----------------------------------------------
            # Console summary
            # -----------------------------------------------

            if (
                i < 10
                or i % 20 == 0
                or i
                == args.frames - 1
            ):
                print(
                    f"[{i:04d}] "
                    f"speed={speed:5.2f} "
                    f"cmd="
                    f"{command_name(road_option):>10s} "
                    f"tp=("
                    f"{target_xy[0]:+5.2f},"
                    f"{target_xy[1]:+5.2f}) | "
                    f"DRV "
                    f"S={driver_control.steer:+.3f} "
                    f"T={driver_control.throttle:.3f} "
                    f"B={driver_control.brake:.3f} | "
                    f"TCP "
                    f"S={tcp['steer']:+.3f} "
                    f"T={tcp['throttle']:.3f} "
                    f"B={tcp['brake']:.3f} "
                    f"status="
                    f"{tcp['status_used']}"
                )

        # ----------------------------------------------------
        # Finish
        # ----------------------------------------------------

        csv_file.close()
        csv_file = None

        print()
        print("=" * 78)
        print(
            "ROUTE PROBE COMPLETE"
        )

        print(
            "frames:",
            completed,
        )

        print(
            "output:",
            output,
        )

        print("=" * 78)

    finally:
        print(
            "[cleanup]"
        )

        if csv_file is not None:
            csv_file.close()

        if camera is not None:
            camera.stop()
            camera.destroy()

        if ego is not None:
            ego.destroy()

        world.apply_settings(
            original_settings
        )


if __name__ == "__main__":
    main()