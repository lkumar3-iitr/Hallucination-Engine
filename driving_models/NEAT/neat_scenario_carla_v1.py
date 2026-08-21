"""
neat_scenario_carla_v1.py

CARLA-only ScenarioGenerator execution for pretrained NEAT.

Stage 1 of the NEAT <-> HE experiment.

Important
---------
The resolved scenario is MODEL-INDEPENDENT physical truth.

We reuse exactly:

    ScenarioGenerator/outputs/v2_resolved/
        tcp_lead_brake_001.resolved_v2.json

The old "camera" block inside that resolved artifact is NOT used here.
It is legacy execution metadata from the original TCP integration.

NEAT uses its own validated native camera configuration:

    front : yaw   0 deg
    left  : yaw -60 deg
    right : yaw +60 deg

    400x300
    FOV 100
    x=1.3
    y=0.0
    z=2.3

The physical adversary trajectory itself is unchanged.

Condition implemented here:

    CARLA:
        physical adversary actor

HE rendering will be added only after this baseline is validated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
from pathlib import Path

import carla
import numpy as np
import torch

from PIL import Image, ImageDraw


# ============================================================
# Reuse validated NEAT/CARLA integration
# ============================================================

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


# ============================================================
# Defaults
# ============================================================

DEFAULT_RESOLVED = (
    HE_ROOT
    / "ScenarioGenerator"
    / "outputs"
    / "v2_resolved"
    / "tcp_lead_brake_001.resolved_v2.json"
)

DEFAULT_OUTPUT_ROOT = (
    HE_ROOT
    / "driving_models"
    / "NEAT"
    / "outputs"
    / "scenario_carla_v1"
)


# ============================================================
# JSON
# ============================================================

def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# Resolved Scenario v2
# ============================================================

def load_resolved_scenario(
    path: Path,
    actor_id: str,
):

    data = load_json(
        path
    )

    if (
        data.get("schema_version")
        != "2.0-resolved"
    ):

        raise ValueError(
            "Expected resolved "
            "ScenarioSchema v2."
        )

    actors = {
        row["actor_id"]: row
        for row
        in data.get(
            "actors",
            [],
        )
    }

    if actor_id not in actors:

        raise KeyError(
            f"Actor {actor_id!r} "
            "not found in scenario."
        )

    actor_info = (
        actors[
            actor_id
        ]
    )

    frames = [
        row
        for row
        in data.get(
            "actor_frames",
            [],
        )
        if (
            row["actor_id"]
            == actor_id
        )
    ]

    frames = sorted(
        frames,
        key=lambda row:
            int(
                row[
                    "frame_idx"
                ]
            ),
    )

    if not frames:

        raise RuntimeError(
            "No resolved actor frames."
        )

    frame_ids = [
        int(
            row[
                "frame_idx"
            ]
        )
        for row in frames
    ]

    expected = list(
        range(
            frame_ids[0],
            frame_ids[-1] + 1,
        )
    )

    if frame_ids != expected:

        raise RuntimeError(
            "Actor trajectory is "
            "not frame-contiguous."
        )

    return (
        data,
        actor_info,
        frames,
    )


# ============================================================
# Coordinate helpers
# ============================================================

def yaw_forward_right(
    yaw_deg,
):

    yaw = math.radians(
        float(
            yaw_deg
        )
    )

    forward = np.array(
        [
            math.cos(yaw),
            math.sin(yaw),
        ],
        dtype=np.float64,
    )

    right = np.array(
        [
            -math.sin(yaw),
            math.cos(yaw),
        ],
        dtype=np.float64,
    )

    return (
        forward,
        right,
    )


# ============================================================
# ScenarioGenerator -> CARLA world
# ============================================================

def sg_actor_to_world_transform(
    ego0_tf,
    actor_frame,
    world,
):
    """
    ScenarioGenerator v2:

        +x   = forward
        +y   = left
        +yaw = CCW / left

    Mapping used by the validated TCP scenario runner:

        world forward = +SG x
        world right   = -SG y

        CARLA world yaw =
            initial ego yaw - SG yaw
    """

    forward, right = (
        yaw_forward_right(
            ego0_tf
            .rotation
            .yaw
        )
    )

    sg_forward = float(
        actor_frame[
            "x_m"
        ]
    )

    sg_left = float(
        actor_frame[
            "y_m"
        ]
    )

    origin = np.array(
        [
            float(
                ego0_tf
                .location
                .x
            ),

            float(
                ego0_tf
                .location
                .y
            ),
        ],
        dtype=np.float64,
    )

    world_xy = (
        origin
        + forward
        * sg_forward
        - right
        * sg_left
    )

    location = carla.Location(
        x=float(
            world_xy[0]
        ),

        y=float(
            world_xy[1]
        ),

        z=float(
            ego0_tf
            .location
            .z
        ),
    )

    # --------------------------------------------------------
    # Keep vehicle on road height.
    # --------------------------------------------------------

    waypoint = (
        world
        .get_map()
        .get_waypoint(
            location,

            project_to_road=True,

            lane_type=(
                carla.LaneType.Driving
            ),
        )
    )

    if waypoint is not None:

        location.z = float(
            waypoint
            .transform
            .location
            .z
            + 0.05
        )

    world_yaw = (
        float(
            ego0_tf
            .rotation
            .yaw
        )
        -
        float(
            actor_frame[
                "yaw_deg"
            ]
        )
    )

    return carla.Transform(
        location,

        carla.Rotation(
            pitch=0.0,
            yaw=world_yaw,
            roll=0.0,
        ),
    )


# ============================================================
# Shared physical/safety metrics
# ============================================================

def compute_virtual_metrics(
    ego_tf,
    ego0_tf,
    ego_speed_mps,
    actor_frame,
    dimensions,
    ego,
):

    forward, right = (
        yaw_forward_right(
            ego0_tf
            .rotation
            .yaw
        )
    )

    delta = np.array(
        [
            float(
                ego_tf.location.x
                -
                ego0_tf.location.x
            ),

            float(
                ego_tf.location.y
                -
                ego0_tf.location.y
            ),
        ],
        dtype=np.float64,
    )

    ego_progress = float(
        np.dot(
            delta,
            forward,
        )
    )

    ego_right = float(
        np.dot(
            delta,
            right,
        )
    )

    # SG +y = left.
    ego_left = (
        -ego_right
    )

    actor_progress = float(
        actor_frame[
            "x_m"
        ]
    )

    actor_left = float(
        actor_frame[
            "y_m"
        ]
    )

    longitudinal_center_distance = (
        actor_progress
        -
        ego_progress
    )

    lateral_distance = abs(
        actor_left
        -
        ego_left
    )

    ego_length = float(
        ego
        .bounding_box
        .extent
        .x
        * 2.0
    )

    ego_width = float(
        ego
        .bounding_box
        .extent
        .y
        * 2.0
    )

    actor_length = float(
        dimensions[
            "length_m"
        ]
    )

    actor_width = float(
        dimensions[
            "width_m"
        ]
    )

    half_length_sum = (
        0.5
        *
        (
            ego_length
            +
            actor_length
        )
    )

    half_width_sum = (
        0.5
        *
        (
            ego_width
            +
            actor_width
        )
    )

    bumper_gap = (
        longitudinal_center_distance
        -
        half_length_sum
    )

    actor_speed = float(
        actor_frame[
            "speed_mps"
        ]
    )

    closing_speed = (
        float(
            ego_speed_mps
        )
        -
        actor_speed
    )

    if (
        bumper_gap > 0.0
        and
        closing_speed > 1e-3
    ):

        ttc = (
            bumper_gap
            /
            closing_speed
        )

    else:

        ttc = float(
            "inf"
        )

    virtual_collision = (
        abs(
            longitudinal_center_distance
        )
        <=
        half_length_sum

        and

        lateral_distance
        <=
        half_width_sum
    )

    return {
        "ego_progress_m":
            ego_progress,

        "ego_left_m":
            ego_left,

        "actor_progress_m":
            actor_progress,

        "actor_left_m":
            actor_left,

        "center_distance_m":
            longitudinal_center_distance,

        "lateral_distance_m":
            lateral_distance,

        "bumper_gap_m":
            bumper_gap,

        "closing_speed_mps":
            closing_speed,

        "ttc_s":
            ttc,

        "virtual_collision":
            bool(
                virtual_collision
            ),
    }


# ============================================================
# Exact NEAT crop for visualization
# ============================================================

def crop_neat_rgb(
    rgb,
    crop=256,
):

    h, w = (
        rgb.shape[
            :2
        ]
    )

    y1 = (
        h // 2
        -
        crop // 2
    )

    x1 = (
        w // 2
        -
        crop // 2
    )

    return (
        rgb[
            y1:y1 + crop,
            x1:x1 + crop,
        ]
        .copy()
    )


# ============================================================
# Debug image
# ============================================================

def save_debug_image(
    path,
    rgb_front,
    rgb_left,
    rgb_right,
    frame_idx,
    actor_frame,
    speed_mps,
    result,
    metrics,
    nav,
):

    crop_front = crop_neat_rgb(
        rgb_front
    )

    crop_left = crop_neat_rgb(
        rgb_left
    )

    crop_right = crop_neat_rgb(
        rgb_right
    )

    # Same ordering used by run_neat:
    #
    # front | left | right

    composite = np.concatenate(
        [
            crop_front,
            crop_left,
            crop_right,
        ],
        axis=1,
    )

    # Add HUD ABOVE image,
    # never into model input.

    hud_h = 76

    canvas = np.zeros(
        (
            256 + hud_h,
            768,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        hud_h:,
        :,
    ] = composite

    image = Image.fromarray(
        canvas
    )

    draw = ImageDraw.Draw(
        image
    )

    ttc = float(
        metrics[
            "ttc_s"
        ]
    )

    if math.isfinite(
        ttc
    ):

        ttc_text = (
            f"{ttc:.2f}s"
        )

    else:

        ttc_text = "inf"

    target = (
        nav[
            "target_point"
        ]
    )

    line1 = (
        f"NEAT / CARLA   "
        f"frame={frame_idx:04d}   "
        f"t={float(actor_frame['t_s']):.2f}s   "
        f"ego={speed_mps:.2f}m/s   "
        f"lead={float(actor_frame['speed_mps']):.2f}m/s"
    )

    line2 = (
        f"gap={metrics['bumper_gap_m']:.2f}m   "
        f"TTC={ttc_text}   "
        f"target=({target[0]:+.2f},{target[1]:+.2f})   "
        f"RL={result['red_light_occ']}"
    )

    line3 = (
        f"S={result['steer']:+.3f}   "
        f"T={result['throttle']:.3f}   "
        f"B={result['brake']:.3f}   "
        f"views: FRONT | LEFT | RIGHT"
    )

    draw.text(
        (6, 5),
        line1,
        fill=(
            255,
            255,
            255,
        ),
    )

    draw.text(
        (6, 27),
        line2,
        fill=(
            255,
            255,
            255,
        ),
    )

    draw.text(
        (6, 49),
        line3,
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
# Bootstrap result
# ============================================================

def make_bootstrap_result():

    return {
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


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Scenario
    # --------------------------------------------------------

    parser.add_argument(
        "--resolved",
        default=str(
            DEFAULT_RESOLVED
        ),
    )

    parser.add_argument(
        "--actor-id",
        default="adv_lead",
    )

    parser.add_argument(
        "--event-start-s",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
    )

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
        "--carla-pythonapi",
        default=None,
    )

    # --------------------------------------------------------
    # NEAT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

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
        "--debug-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Scenario
    # ========================================================

    resolved_path = Path(
        args.resolved
    ).resolve()

    (
        scenario,
        actor_info,
        actor_frames,
    ) = load_resolved_scenario(
        resolved_path,
        args.actor_id,
    )

    fps = float(
        scenario[
            "fps"
        ]
    )

    if abs(
        fps - 20.0
    ) > 1e-6:

        raise RuntimeError(
            "Current NEAT integration "
            "expects 20 Hz."
        )

    if args.max_frames > 0:

        actor_frames = (
            actor_frames[
                :args.max_frames
            ]
        )

    dimensions = (
        actor_info.get(
            "dimensions_m"
        )
        or
        {
            "length_m":
                4.2,

            "width_m":
                1.8,

            "height_m":
                1.5,
        }
    )

    blueprint_name = (
        actor_info.get(
            "asset_key"
        )
    )

    if (
        blueprint_name is None
        or
        not blueprint_name.startswith(
            "vehicle."
        )
    ):

        blueprint_name = (
            "vehicle.tesla.model3"
        )

    print()
    print("=" * 78)
    print(
        "NEAT SCENARIO / CARLA V1"
    )
    print("=" * 78)

    print(
        "[scenario]",
        scenario[
            "scenario_id"
        ],
    )

    print(
        "[resolved]",
        resolved_path,
    )

    print(
        "[actor]",
        args.actor_id,
    )

    print(
        "[blueprint]",
        blueprint_name,
    )

    print(
        "[frames]",
        len(
            actor_frames
        ),
    )

    print(
        "[fps]",
        fps,
    )

    print(
        "[scenario camera]",
        "IGNORED - NEAT uses native cameras"
    )

    print("=" * 78)
    print()

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and
        not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    if device.type == "cuda":

        print(
            "[GPU]",
            torch.cuda.get_device_name(
                0
            ),
        )

    # ========================================================
    # CARLA navigation
    # ========================================================

    GlobalRoutePlanner = (
        import_carla_agents(
            args.carla_pythonapi
        )
    )

    # ========================================================
    # NEAT
    # ========================================================

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
        "[NEAT]",
        f"{config.num_camera} cameras",
    )

    # ========================================================
    # Output
    # ========================================================

    output_dir = (
        Path(
            args.output_root
        )
        /
        scenario[
            "scenario_id"
        ]
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
        /
        (
            scenario[
                "scenario_id"
            ]
            + "_carla.csv"
        )
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

    actors = []

    csv_file = None

    try:

        # ====================================================
        # Synchronous 20 Hz
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = (
            True
        )

        settings.fixed_delta_seconds = (
            1.0
            /
            fps
        )

        world.apply_settings(
            settings
        )

        # ====================================================
        # Map / spawn
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
            %
            len(
                spawn_points
            )
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
        # Route
        # ====================================================

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
            len(
                route
            ),
        )

        print(
            "[route] length:",
            f"{route_length(route):.1f} m",
        )

        # ====================================================
        # Original NEAT route planner
        # ====================================================

        gps_plan = (
            build_neat_gps_plan(
                carla_map,
                route,
            )
        )

        route_planner = RoutePlanner(
            4.0,
            50.0,
        )

        route_planner.set_route(
            gps_plan,
            gps=True,
        )

        # ====================================================
        # NEAT native sensors
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

        gnss_sensor = make_gnss(
            world,
            ego,
        )

        imu_sensor = make_imu(
            world,
            ego,
        )

        actors.extend(
            [
                camera_front,
                camera_left,
                camera_right,
                gnss_sensor,
                imu_sensor,
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

        gnss_sensor.listen(
            q_gnss.put
        )

        imu_sensor.listen(
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
            "[init] stabilizing ego..."
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

        # ----------------------------------------------------
        # Reset ego to exactly common scenario origin.
        # ----------------------------------------------------

        ego.set_transform(
            spawn_points[
                spawn_idx
            ]
        )

        ego.set_target_velocity(
            carla.Vector3D(
                0.0,
                0.0,
                0.0,
            )
        )

        ego.set_target_angular_velocity(
            carla.Vector3D(
                0.0,
                0.0,
                0.0,
            )
        )

        ego.apply_control(
            hold
        )

        ego0_tf = (
            ego.get_transform()
        )

        # ====================================================
        # Physical adversary
        # ====================================================

        actor_tf0 = (
            sg_actor_to_world_transform(
                ego0_tf,
                actor_frames[0],
                world,
            )
        )

        adv_bp = (
            world
            .get_blueprint_library()
            .find(
                blueprint_name
            )
        )

        if adv_bp.has_attribute(
            "role_name"
        ):

            adv_bp.set_attribute(
                "role_name",
                "scenario_adversary",
            )

        # Keep same blue vehicle convention as TCP/HE pair.

        if adv_bp.has_attribute(
            "color"
        ):

            adv_bp.set_attribute(
                "color",
                "0,0,255",
            )

        adversary = world.try_spawn_actor(
            adv_bp,
            actor_tf0,
        )

        if adversary is None:

            raise RuntimeError(
                "Could not spawn "
                "scenario adversary."
            )

        actors.append(
            adversary
        )

        # Exact scenario trajectory controls its pose.
        # No TrafficManager / physics.

        adversary.set_simulate_physics(
            False
        )

        print(
            "[adversary] spawned:",
            adversary.id,
        )

        bb = (
            adversary.bounding_box
        )

        print(
            "[adversary bbox]",
            "L=%.3f W=%.3f H=%.3f"
            %
            (
                2.0
                * float(
                    bb.extent.x
                ),

                2.0
                * float(
                    bb.extent.y
                ),

                2.0
                * float(
                    bb.extent.z
                ),
            ),
        )

        # ====================================================
        # First synchronized experimental frame
        # ====================================================

        current_frame = (
            world.tick()
        )

        current_front = (
            get_sensor_frame(
                q_front,
                current_frame,
                "front",
            )
        )

        current_left = (
            get_sensor_frame(
                q_left,
                current_frame,
                "left",
            )
        )

        current_right = (
            get_sensor_frame(
                q_right,
                current_frame,
                "right",
            )
        )

        current_gnss = (
            get_sensor_frame(
                q_gnss,
                current_frame,
                "GNSS",
            )
        )

        current_imu = (
            get_sensor_frame(
                q_imu,
                current_frame,
                "IMU",
            )
        )

        # ====================================================
        # CSV
        # ====================================================

        csv_fields = [
            "condition",
            "scenario_id",
            "model_name",

            "probe_idx",
            "carla_frame",
            "t_s",

            "ego_x",
            "ego_y",
            "ego_z",
            "ego_yaw",
            "ego_speed_mps",

            "actor_x_sg_m",
            "actor_y_sg_m",
            "actor_yaw_sg_deg",
            "actor_speed_mps",

            "actor_world_x",
            "actor_world_y",
            "actor_world_z",
            "actor_world_yaw",

            "bumper_gap_m",
            "center_distance_m",
            "lateral_distance_m",
            "closing_speed_mps",
            "ttc_s",
            "virtual_collision",

            "route_index",
            "route_deviation_m",
            "destination_distance_m",

            "command_name",
            "command_value",

            "target_x",
            "target_y",

            "model_steer",
            "model_throttle",
            "model_brake",

            "desired_speed",
            "red_light_occ",

            "angle",
            "angle_last",
            "angle_target",
            "angle_final",

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
        # Runtime
        # ====================================================

        route_idx = 0
        deviation_counter = 0

        min_gap = float(
            "inf"
        )

        any_collision = False

        first_brake_after_event = (
            None
        )

        completed_frames = 0

        print()
        print("=" * 78)
        print(
            "NEAT / CARLA LEAD-BRAKE EXPERIMENT"
        )
        print("=" * 78)
        print(
            "scenario:",
            scenario[
                "scenario_id"
            ],
        )
        print(
            "model: NEAT"
        )
        print(
            "condition: carla"
        )
        print(
            "frames:",
            len(
                actor_frames
            ),
        )
        print("=" * 78)
        print()

        # ====================================================
        # Scenario loop
        # ====================================================

        for i, actor_frame in enumerate(
            actor_frames
        ):

            # ------------------------------------------------
            # Scenario truth for THIS observation.
            # ------------------------------------------------

            actor_tf = (
                sg_actor_to_world_transform(
                    ego0_tf,
                    actor_frame,
                    world,
                )
            )

            # ------------------------------------------------
            # Three CARLA observations
            # ------------------------------------------------

            rgb_front = (
                carla_image_to_rgb(
                    current_front
                )
            )

            rgb_left = (
                carla_image_to_rgb(
                    current_left
                )
            )

            rgb_right = (
                carla_image_to_rgb(
                    current_right
                )
            )

            # ------------------------------------------------
            # Ego / navigation
            # ------------------------------------------------

            speed_mps = (
                get_speed_mps(
                    ego
                )
            )

            nav = (
                get_neat_navigation(
                    route_planner,
                    current_gnss,
                    current_imu,
                )
            )

            bootstrap = (
                i < config.seq_len
            )

            if bootstrap:

                result = (
                    make_bootstrap_result()
                )

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

            # ------------------------------------------------
            # NEAT controls ego.
            # ------------------------------------------------

            ego.apply_control(
                carla.VehicleControl(
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

                    hand_brake=False,
                    manual_gear_shift=False,
                )
            )

            # ------------------------------------------------
            # Current ego state
            # ------------------------------------------------

            ego_tf = (
                ego.get_transform()
            )

            ego_loc = (
                ego_tf.location
            )

            (
                route_idx,
                route_deviation,
            ) = closest_route_index(
                route,
                ego_loc,
                route_idx,
            )

            destination_distance = (
                distance_2d(
                    ego_loc,
                    destination,
                )
            )

            # ------------------------------------------------
            # Shared safety metrics
            # ------------------------------------------------

            metrics = (
                compute_virtual_metrics(
                    ego_tf=ego_tf,
                    ego0_tf=ego0_tf,

                    ego_speed_mps=
                        speed_mps,

                    actor_frame=
                        actor_frame,

                    dimensions=
                        dimensions,

                    ego=ego,
                )
            )

            min_gap = min(
                min_gap,
                float(
                    metrics[
                        "bumper_gap_m"
                    ]
                ),
            )

            if metrics[
                "virtual_collision"
            ]:

                any_collision = True

            if (
                first_brake_after_event
                is None

                and

                float(
                    actor_frame[
                        "t_s"
                    ]
                )
                >=
                args.event_start_s

                and

                float(
                    result[
                        "brake"
                    ]
                )
                >= 0.5
            ):

                first_brake_after_event = {
                    "frame_idx":
                        i,

                    "t_s":
                        float(
                            actor_frame[
                                "t_s"
                            ]
                        ),

                    "gap_m":
                        float(
                            metrics[
                                "bumper_gap_m"
                            ]
                        ),
                }

            # ------------------------------------------------
            # Route departure
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
            # Metadata
            # ------------------------------------------------

            metadata = (
                result[
                    "metadata"
                ]
            )

            command = (
                nav[
                    "command"
                ]
            )

            command_name = str(
                getattr(
                    command,
                    "name",
                    command,
                )
            )

            command_value = int(
                getattr(
                    command,
                    "value",
                    4,
                )
            )

            ttc = float(
                metrics[
                    "ttc_s"
                ]
            )

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            writer.writerow({
                "condition":
                    "carla",

                "scenario_id":
                    scenario[
                        "scenario_id"
                    ],

                "model_name":
                    "NEAT",

                "probe_idx":
                    i,

                "carla_frame":
                    current_frame,

                "t_s":
                    float(
                        actor_frame[
                            "t_s"
                        ]
                    ),

                "ego_x":
                    float(
                        ego_loc.x
                    ),

                "ego_y":
                    float(
                        ego_loc.y
                    ),

                "ego_z":
                    float(
                        ego_loc.z
                    ),

                "ego_yaw":
                    float(
                        ego_tf
                        .rotation
                        .yaw
                    ),

                "ego_speed_mps":
                    float(
                        speed_mps
                    ),

                "actor_x_sg_m":
                    float(
                        actor_frame[
                            "x_m"
                        ]
                    ),

                "actor_y_sg_m":
                    float(
                        actor_frame[
                            "y_m"
                        ]
                    ),

                "actor_yaw_sg_deg":
                    float(
                        actor_frame[
                            "yaw_deg"
                        ]
                    ),

                "actor_speed_mps":
                    float(
                        actor_frame[
                            "speed_mps"
                        ]
                    ),

                "actor_world_x":
                    float(
                        actor_tf
                        .location
                        .x
                    ),

                "actor_world_y":
                    float(
                        actor_tf
                        .location
                        .y
                    ),

                "actor_world_z":
                    float(
                        actor_tf
                        .location
                        .z
                    ),

                "actor_world_yaw":
                    float(
                        actor_tf
                        .rotation
                        .yaw
                    ),

                "bumper_gap_m":
                    metrics[
                        "bumper_gap_m"
                    ],

                "center_distance_m":
                    metrics[
                        "center_distance_m"
                    ],

                "lateral_distance_m":
                    metrics[
                        "lateral_distance_m"
                    ],

                "closing_speed_mps":
                    metrics[
                        "closing_speed_mps"
                    ],

                "ttc_s":
                    (
                        ttc
                        if
                        math.isfinite(
                            ttc
                        )
                        else
                        ""
                    ),

                "virtual_collision":
                    int(
                        metrics[
                            "virtual_collision"
                        ]
                    ),

                "route_index":
                    route_idx,

                "route_deviation_m":
                    route_deviation,

                "destination_distance_m":
                    destination_distance,

                "command_name":
                    command_name,

                "command_value":
                    command_value,

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

                "model_steer":
                    float(
                        result[
                            "steer"
                        ]
                    ),

                "model_throttle":
                    float(
                        result[
                            "throttle"
                        ]
                    ),

                "model_brake":
                    float(
                        result[
                            "brake"
                        ]
                    ),

                "desired_speed":
                    metadata.get(
                        "desired_speed",
                        "",
                    ),

                "red_light_occ":
                    int(
                        result[
                            "red_light_occ"
                        ]
                    ),

                "angle":
                    metadata.get(
                        "angle",
                        "",
                    ),

                "angle_last":
                    metadata.get(
                        "angle_last",
                        "",
                    ),

                "angle_target":
                    metadata.get(
                        "angle_target",
                        "",
                    ),

                "angle_final":
                    metadata.get(
                        "angle_final",
                        "",
                    ),

                "bootstrap":
                    int(
                        bootstrap
                    ),
            })

            csv_file.flush()

            completed_frames += 1

            # ------------------------------------------------
            # Debug
            # ------------------------------------------------

            if (
                args.debug_every > 0
                and
                i % args.debug_every == 0
            ):

                save_debug_image(
                    (
                        debug_dir
                        /
                        f"{i:05d}.png"
                    ),

                    rgb_front,
                    rgb_left,
                    rgb_right,

                    i,
                    actor_frame,

                    speed_mps,
                    result,
                    metrics,
                    nav,
                )

            # ------------------------------------------------
            # Console
            # ------------------------------------------------

            if (
                i < 10
                or
                i % 20 == 0
            ):

                if math.isfinite(
                    ttc
                ):

                    ttc_text = (
                        f"{ttc:.2f}"
                    )

                else:

                    ttc_text = (
                        "inf"
                    )

                print(
                    f"[{i:04d}] "
                    f"t={float(actor_frame['t_s']):5.2f} "
                    f"ego={speed_mps:4.2f} "
                    f"lead={float(actor_frame['speed_mps']):4.2f} "
                    f"gap={metrics['bumper_gap_m']:6.2f} "
                    f"TTC={ttc_text:>5s} | "
                    f"NEAT "
                    f"S={result['steer']:+.3f} "
                    f"T={result['throttle']:.3f} "
                    f"B={result['brake']:.3f} "
                    f"RL={result['red_light_occ']}"
                )

            # ------------------------------------------------
            # Safety departure
            # ------------------------------------------------

            if (
                deviation_counter
                >=
                args.deviation_patience_frames
            ):

                print()
                print(
                    "[SAFETY] NEAT departed route."
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
            # End scenario
            # ------------------------------------------------

            if (
                i + 1
                >=
                len(
                    actor_frames
                )
            ):

                break

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Put adversary at NEXT scenario pose BEFORE
            # advancing synchronous CARLA.
            #
            # Therefore camera frame i+1 sees actor frame i+1.
            # ------------------------------------------------

            next_actor_tf = (
                sg_actor_to_world_transform(
                    ego0_tf,
                    actor_frames[
                        i + 1
                    ],
                    world,
                )
            )

            adversary.set_transform(
                next_actor_tf
            )

            # ------------------------------------------------
            # Advance world
            # ------------------------------------------------

            current_frame = (
                world.tick()
            )

            current_front = (
                get_sensor_frame(
                    q_front,
                    current_frame,
                    "front",
                )
            )

            current_left = (
                get_sensor_frame(
                    q_left,
                    current_frame,
                    "left",
                )
            )

            current_right = (
                get_sensor_frame(
                    q_right,
                    current_frame,
                    "right",
                )
            )

            current_gnss = (
                get_sensor_frame(
                    q_gnss,
                    current_frame,
                    "GNSS",
                )
            )

            current_imu = (
                get_sensor_frame(
                    q_imu,
                    current_frame,
                    "IMU",
                )
            )

        # ====================================================
        # Complete
        # ====================================================

        print()
        print("=" * 78)
        print(
            "NEAT / CARLA SCENARIO COMPLETE"
        )
        print("=" * 78)

        print(
            "scenario:",
            scenario[
                "scenario_id"
            ],
        )

        print(
            "frames:",
            completed_frames,
        )

        print(
            "minimum bumper gap:",
            f"{min_gap:.3f} m",
        )

        print(
            "virtual collision:",
            any_collision,
        )

        if (
            first_brake_after_event
            is None
        ):

            print(
                "first NEAT brake after event: none"
            )

        else:

            print(
                "first NEAT brake after event:",
                "frame",
                first_brake_after_event[
                    "frame_idx"
                ],
                "t=",
                f"{first_brake_after_event['t_s']:.2f}s",
                "gap=",
                f"{first_brake_after_event['gap_m']:.2f}m",
            )

        print(
            "CSV:",
            csv_path,
        )

        print(
            "debug:",
            debug_dir,
        )

        print("=" * 78)

    finally:

        print(
            "[cleanup]"
        )

        if csv_file is not None:

            csv_file.close()

        # Sensors first.

        for actor in reversed(
            actors
        ):

            try:

                if (
                    actor.type_id
                    .startswith(
                        "sensor."
                    )
                ):

                    try:
                        actor.stop()
                    except Exception:
                        pass

                actor.destroy()

            except Exception:
                pass

        try:

            world.apply_settings(
                original_settings
            )

        except Exception:
            pass


if __name__ == "__main__":

    main()