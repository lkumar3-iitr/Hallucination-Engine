from __future__ import annotations

import argparse
import csv
import queue
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import carla


# ============================================================
# Existing validated TCP integration
# ============================================================

from tcp_carla_0915_probe import (
    HE_ROOT,
    DEFAULT_TCP_ROOT,
    DEFAULT_CHECKPOINT,
    carla_image_to_rgb,
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

from tcp_carla_0915_closed_loop import (
    import_tcp_route_planner,
    make_gnss,
    make_imu,
    get_named_sensor_frame,
    build_tcp_gps_plan,
    get_tcp_navigation,
    make_bootstrap_result,
)


# ============================================================
# Shared common HE/runtime modules
# ============================================================

COMMON_DIR = (
    HE_ROOT
    / "driving_models"
    / "common"
)

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(COMMON_DIR),
    )

from carla_ego_initialization import (
    canonicalize_ego_start,
)

from he_asset_registry_v1 import (
    DEFAULT_MANIFEST,
)

from he_multi_actor_compositor_v1 import (
    HEMultiActorCompositorV1,
)

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
    / "TCP"
    / "outputs"
    / "tcp_he_multi_actor_v1"
)


# ============================================================
# Helpers
# ============================================================

def ensure_dir(
    path,
):
    path = Path(
        path
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def build_execution_origin(
    carla_map,
    ego0_tf,
):
    """
    Anchor ScenarioGenerator XY/yaw to the canonical ego start.

    Actor world_z_m is a base/road z, not ego chassis z.
    """

    waypoint = carla_map.get_waypoint(
        ego0_tf.location,
        project_to_road=True,
        lane_type=
            carla.LaneType.Driving,
    )

    if waypoint is None:
        raise RuntimeError(
            "Could not resolve driving waypoint "
            "for canonical ego start."
        )

    road_z = float(
        waypoint
        .transform
        .location
        .z
    )

    return ExecutionWorldOrigin(
        x_m=float(
            ego0_tf.location.x
        ),
        y_m=float(
            ego0_tf.location.y
        ),
        z_m=road_z,
        yaw_deg=float(
            ego0_tf.rotation.yaw
        ),
    )


def make_video_writer(
    path,
    fps,
    width,
    height,
):
    writer = cv2.VideoWriter(
        str(
            path
        ),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        float(
            fps
        ),
        (
            int(width),
            int(height),
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer: {path}"
        )

    return writer


def write_rgb_frame(
    writer,
    rgb,
):
    writer.write(
        rgb[
            :,
            :,
            ::-1,
        ]
    )


def draw_debug(
    rgb,
    scenario_frame,
    t_s,
    speed_mps,
    tcp,
    nav,
    actor_results,
    route_deviation_m,
    destination_distance_m,
):
    frame = (
        rgb[
            :,
            :,
            ::-1,
        ]
        .copy()
    )

    rendered = [
        row
        for row
        in actor_results
        if row.rendered
    ]

    nearest_depth = None

    if rendered:
        nearest_depth = min(
            float(
                row.distance_forward_m
            )
            for row
            in rendered
        )

    nearest_text = (
        "None"
        if nearest_depth is None
        else f"{nearest_depth:.2f}m"
    )

    lines = [
        (
            f"TCP + HE multi-actor  "
            f"frame={scenario_frame} "
            f"t={t_s:.2f}s"
        ),
        (
            f"Ego speed={speed_mps:.2f}m/s  "
            f"actors={len(actor_results)} "
            f"rendered={len(rendered)} "
            f"nearest={nearest_text}"
        ),
        (
            f"TCP "
            f"S={float(tcp['steer']):+.3f} "
            f"T={float(tcp['throttle']):.3f} "
            f"B={float(tcp['brake']):.3f} "
            f"status={tcp['status_used']}"
        ),
        (
            f"CMD {command_name(nav['command'])}  "
            f"route_dev={route_deviation_m:.2f}m "
            f"dest={destination_distance_m:.1f}m"
        ),
    ]

    y = 20

    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 22

    return frame


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "HE-only multi-actor TCP closed-loop smoke test. "
            "TCP is unchanged; scenario actors are virtual HE "
            "actors rendered into TCP's native camera."
        )
    )

    # --------------------------------------------------------
    # Scenario / HE assets
    # --------------------------------------------------------

    parser.add_argument(
        "--resolved",
        default=str(
            DEFAULT_RESOLVED
        ),
    )

    parser.add_argument(
        "--asset-root",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    parser.add_argument(
        "--distance-selection-mode",
        choices=[
            "linear",
            "log",
            "inverse_depth",
        ],
        default="linear",
    )

    parser.add_argument(
        "--he-bottom-y-offset-px",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
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
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    args = parser.parse_args()

    resolved_path = Path(
        args.resolved
    ).resolve()

    asset_root = Path(
        args.asset_root
    ).resolve()

    manifest_path = Path(
        args.manifest
    ).resolve()

    output_root = Path(
        args.output_root
    ).resolve()

    # ========================================================
    # TCP navigation/model imports
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

    if device.type == "cuda":
        print(
            "[GPU]",
            torch.cuda.get_device_name(
                0
            ),
        )

    net, config = load_tcp_model(
        tcp_root=tcp_root,
        checkpoint_path=Path(
            args.checkpoint
        ).resolve(),
        device=device,
    )

    print(
        "[TCP] pred_len:",
        config.pred_len,
    )

    # ========================================================
    # CARLA
    # ========================================================

    client = carla.Client(
        args.host,
        int(
            args.port
        ),
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

    input_writer = None
    debug_writer = None
    csv_fp = None

    try:
        # ====================================================
        # Synchronous mode: current scenario expects 20 Hz.
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            1.0 / 20.0
        )

        world.apply_settings(
            settings
        )

        carla_map = (
            world.get_map()
        )

        spawn_points = (
            carla_map
            .get_spawn_points()
        )

        if not spawn_points:
            raise RuntimeError(
                "CARLA map has no spawn points."
            )

        spawn_idx = (
            int(
                args.spawn_index
            )
            %
            len(
                spawn_points
            )
        )

        # ====================================================
        # Ego + route
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
            "[route] destination:",
            destination_idx,
        )

        print(
            "[route] length:",
            f"{route_length(route):.1f} m",
        )

        tcp_gps_plan = build_tcp_gps_plan(
            carla_map,
            route,
        )

        tcp_route_planner = RoutePlanner(
            4.0,
            50.0,
        )

        tcp_route_planner.set_route(
            tcp_gps_plan,
            gps=True,
        )

        # ====================================================
        # Deterministic canonical ego start
        # ====================================================

        (
            ego0_tf,
            ego0_state,
        ) = canonicalize_ego_start(
            world=world,
            ego=ego,
            nominal_spawn_tf=spawn_points[
                spawn_idx
            ],
            settle_ticks=30,
            hold_ticks=5,
        )

        print(
            "[init] EXPERIMENT START BARRIER"
        )

        # ====================================================
        # Generic multi-actor scenario runtime
        # ====================================================

        origin = build_execution_origin(
            carla_map=carla_map,
            ego0_tf=ego0_tf,
        )

        runtime = load_execution_runtime(
            resolved_json=str(
                resolved_path
            ),
            asset_root=asset_root,
            origin=origin,
            manifest_path=
                manifest_path,
        )

        summary = runtime.summary()

        fps = float(
            summary[
                "fps"
            ]
        )

        if abs(
            fps - 20.0
        ) > 1e-6:
            raise RuntimeError(
                "This TCP smoke test currently expects "
                "a 20 Hz scenario."
            )

        total_scenario_frames = (
            int(
                round(
                    float(
                        summary[
                            "duration_s"
                        ]
                    )
                    *
                    fps
                )
            )
            + 1
        )

        start_frame = max(
            0,
            int(
                args.start_frame
            ),
        )

        end_frame = (
            total_scenario_frames
            - 1
        )

        if int(
            args.max_frames
        ) > 0:
            end_frame = min(
                end_frame,
                start_frame
                +
                int(
                    args.max_frames
                )
                - 1,
            )

        if end_frame < start_frame:
            raise RuntimeError(
                "No scenario frames selected."
            )

        print(
            "[runtime]",
            summary,
        )

        print(
            "[scenario frames]",
            f"{start_frame}..{end_frame}",
        )

        # ====================================================
        # TCP native sensors
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

        camera_queue = queue.Queue()
        gnss_queue = queue.Queue()
        imu_queue = queue.Queue()

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
        # First synchronized experimental frame
        # ====================================================

        current_frame = (
            world.tick()
        )

        current_image = get_named_sensor_frame(
            camera_queue,
            current_frame,
            "RGB camera",
        )

        current_gnss = get_named_sensor_frame(
            gnss_queue,
            current_frame,
            "GNSS",
        )

        current_imu = get_named_sensor_frame(
            imu_queue,
            current_frame,
            "IMU",
        )

        # ====================================================
        # HE compositor
        # ====================================================

        compositor = HEMultiActorCompositorV1(
            distance_selection_mode=
                args.distance_selection_mode,

            bottom_y_offset_px=
                args.he_bottom_y_offset_px,
        )

        # ====================================================
        # Outputs
        # ====================================================

        scenario_id = str(
            summary[
                "scenario_id"
            ]
        )

        output_dir = ensure_dir(
            output_root
            /
            scenario_id
        )

        input_video_path = (
            output_dir
            /
            (
                scenario_id
                + "_he_tcp_input.mp4"
            )
        )

        debug_video_path = (
            output_dir
            /
            (
                scenario_id
                + "_he_debug.mp4"
            )
        )

        csv_path = (
            output_dir
            /
            (
                scenario_id
                + "_he.csv"
            )
        )

        input_writer = make_video_writer(
            input_video_path,
            fps,
            900,
            256,
        )

        debug_writer = make_video_writer(
            debug_video_path,
            fps,
            900,
            256,
        )

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

            "tcp_steer",
            "tcp_throttle",
            "tcp_brake",
            "tcp_status_used",
            "tcp_status_next",

            "active_actor_count",
            "rendered_actor_count",
            "nearest_actor_id",
            "nearest_actor_depth_m",
        ]

        csv_fp = open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            csv_fp,
            fieldnames=
                csv_fields,
        )

        writer.writeheader()

        # ====================================================
        # TCP runtime state
        # ====================================================

        fusion = TCPFusionState()

        route_idx = 0
        deviation_counter = 0
        completed_frames = 0

        min_rendered_depth = float(
            "inf"
        )

        max_brake = 0.0

        print()
        print("=" * 78)
        print(
            "TCP + HE MULTI-ACTOR CLOSED-LOOP V1"
        )
        print("=" * 78)
        print(
            "scenario :",
            scenario_id,
        )
        print(
            "TCP input: 900x256 FOV100"
        )
        print(
            "actors   : runtime-driven"
        )
        print(
            "condition: HE only"
        )
        print("=" * 78)
        print()

        # ====================================================
        # Closed-loop experiment
        # ====================================================

        for scenario_frame in range(
            start_frame,
            end_frame + 1,
        ):
            t_s = (
                float(
                    scenario_frame
                )
                /
                fps
            )

            # ------------------------------------------------
            # Exact native TCP camera background
            # ------------------------------------------------

            base_rgb = prepare_tcp_rgb_native(
                carla_image_to_rgb(
                    current_image
                )
            )

            # ------------------------------------------------
            # Generic runtime -> all active actors -> HE
            # ------------------------------------------------

            active_actors = list(
                runtime.active_actors(
                    scenario_frame
                )
            )

            composite = compositor.render(
                base_rgb=
                    base_rgb,
                camera_tf=
                    current_image.transform,
                active_actors=
                    active_actors,
                width=
                    900,
                height=
                    256,
                fov=
                    100.0,
            )

            tcp_rgb = composite.rgb

            # ------------------------------------------------
            # TCP navigation and inference unchanged
            # ------------------------------------------------

            speed_mps = (
                get_vehicle_speed_mps(
                    ego
                )
            )

            nav = get_tcp_navigation(
                tcp_route_planner,
                current_gnss,
                current_imu,
            )

            bootstrap = (
                scenario_frame
                ==
                start_frame
            )

            if bootstrap:
                tcp = (
                    make_bootstrap_result()
                )
            else:
                tcp = run_tcp_route(
                    net,
                    tcp_rgb,
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
            # TCP controls the physical ego
            # ------------------------------------------------

            ego.apply_control(
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

            # ------------------------------------------------
            # Ego / route state
            # ------------------------------------------------

            ego_tf = (
                ego.get_transform()
            )

            ego_loc = (
                ego_tf.location
            )

            route_idx = closest_route_index(
                route,
                ego_loc,
                route_idx,
            )

            route_loc = (
                route[
                    route_idx
                ][0]
                .transform
                .location
            )

            route_deviation = distance_2d(
                ego_loc,
                route_loc,
            )

            destination_distance = distance_2d(
                ego_loc,
                destination_location,
            )

            if (
                route_deviation
                >
                float(
                    args.max_route_deviation_m
                )
            ):
                deviation_counter += 1
            else:
                deviation_counter = 0

            # ------------------------------------------------
            # Multi-actor summary for logging
            # ------------------------------------------------

            rendered_rows = [
                row
                for row
                in composite.actor_results
                if row.rendered
            ]

            nearest_row = None

            if rendered_rows:
                nearest_row = min(
                    rendered_rows,
                    key=lambda row:
                        float(
                            row.distance_forward_m
                        ),
                )

                min_rendered_depth = min(
                    min_rendered_depth,
                    float(
                        nearest_row
                        .distance_forward_m
                    ),
                )

            max_brake = max(
                max_brake,
                float(
                    tcp[
                        "brake"
                    ]
                ),
            )

            # ------------------------------------------------
            # Output
            # ------------------------------------------------

            write_rgb_frame(
                input_writer,
                tcp_rgb,
            )

            debug_bgr = draw_debug(
                rgb=tcp_rgb,
                scenario_frame=
                    scenario_frame,
                t_s=t_s,
                speed_mps=
                    speed_mps,
                tcp=tcp,
                nav=nav,
                actor_results=
                    composite.actor_results,
                route_deviation_m=
                    route_deviation,
                destination_distance_m=
                    destination_distance,
            )

            debug_writer.write(
                debug_bgr
            )

            writer.writerow({
                "scenario_frame":
                    int(
                        scenario_frame
                    ),

                "carla_frame":
                    int(
                        current_frame
                    ),

                "t_s":
                    float(
                        t_s
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
                        ego_tf.rotation.yaw
                    ),

                "ego_speed_mps":
                    float(
                        speed_mps
                    ),

                "route_index":
                    int(
                        route_idx
                    ),

                "route_deviation_m":
                    float(
                        route_deviation
                    ),

                "destination_distance_m":
                    float(
                        destination_distance
                    ),

                "command_name":
                    command_name(
                        nav[
                            "command"
                        ]
                    ),

                "command_value":
                    float(
                        nav[
                            "command_value"
                        ]
                    ),

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
                    float(
                        tcp[
                            "steer"
                        ]
                    ),

                "tcp_throttle":
                    float(
                        tcp[
                            "throttle"
                        ]
                    ),

                "tcp_brake":
                    float(
                        tcp[
                            "brake"
                        ]
                    ),

                "tcp_status_used":
                    str(
                        tcp[
                            "status_used"
                        ]
                    ),

                "tcp_status_next":
                    str(
                        tcp[
                            "status_next"
                        ]
                    ),

                "active_actor_count":
                    int(
                        len(
                            active_actors
                        )
                    ),

                "rendered_actor_count":
                    int(
                        len(
                            rendered_rows
                        )
                    ),

                "nearest_actor_id":
                    (
                        None
                        if nearest_row is None
                        else
                        nearest_row.actor_id
                    ),

                "nearest_actor_depth_m":
                    (
                        None
                        if nearest_row is None
                        else
                        float(
                            nearest_row
                            .distance_forward_m
                        )
                    ),
            })

            completed_frames += 1

            if (
                scenario_frame == start_frame
                or scenario_frame == end_frame
                or scenario_frame % 20 == 0
            ):
                print(
                    (
                        f"[frame {scenario_frame:04d}] "
                        f"speed={speed_mps:.2f} "
                        f"active={len(active_actors)} "
                        f"rendered={len(rendered_rows)} "
                        f"TCP=("
                        f"S={float(tcp['steer']):+.3f},"
                        f"T={float(tcp['throttle']):.3f},"
                        f"B={float(tcp['brake']):.3f}) "
                        f"route_dev={route_deviation:.2f}"
                    )
                )

            # ------------------------------------------------
            # Safety stop on persistent route departure
            # ------------------------------------------------

            if (
                deviation_counter
                >=
                int(
                    args.deviation_patience_frames
                )
            ):
                print(
                    "[stop] persistent route deviation:",
                    f"{route_deviation:.2f} m",
                )
                break

            # ------------------------------------------------
            # Advance synchronized simulation
            # ------------------------------------------------

            if scenario_frame < end_frame:
                current_frame = (
                    world.tick()
                )

                current_image = get_named_sensor_frame(
                    camera_queue,
                    current_frame,
                    "RGB camera",
                )

                current_gnss = get_named_sensor_frame(
                    gnss_queue,
                    current_frame,
                    "GNSS",
                )

                current_imu = get_named_sensor_frame(
                    imu_queue,
                    current_frame,
                    "IMU",
                )

        # ====================================================
        # Summary
        # ====================================================

        print()
        print("=" * 78)
        print(
            "TCP + HE MULTI-ACTOR RUN COMPLETE"
        )
        print("=" * 78)

        print(
            "completed frames:",
            completed_frames,
        )

        print(
            "max TCP brake:",
            f"{max_brake:.3f}",
        )

        if np.isfinite(
            min_rendered_depth
        ):
            print(
                "minimum rendered forward depth:",
                f"{min_rendered_depth:.3f} m",
            )

        print(
            "input video:",
            input_video_path,
        )

        print(
            "debug video:",
            debug_video_path,
        )

        print(
            "CSV:",
            csv_path,
        )

        print()
        print(
            "No physical scenario adversaries were spawned."
        )

        print("=" * 78)

    finally:
        # ====================================================
        # Output cleanup
        # ====================================================

        if csv_fp is not None:
            try:
                csv_fp.close()
            except Exception:
                pass

        for writer_obj in (
            input_writer,
            debug_writer,
        ):
            if writer_obj is not None:
                try:
                    writer_obj.release()
                except Exception:
                    pass

        # ====================================================
        # CARLA cleanup
        # ====================================================

        for sensor in (
            camera,
            gnss_sensor,
            imu_sensor,
        ):
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass

                try:
                    sensor.destroy()
                except Exception:
                    pass

        if ego is not None:
            try:
                ego.destroy()
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
