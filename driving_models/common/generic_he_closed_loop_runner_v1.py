"""
generic_he_closed_loop_runner_v1.py

Generic HE-only closed-loop experiment runner.

This runner is model-independent.

Supported adapters:
    tcp
    neat

The runner owns:
    - CARLA world
    - route
    - ego vehicle
    - canonical ego initialization
    - ScenarioGenerator execution runtime
    - HE multi-actor compositor
    - model-independent CARLA stepping
    - route/deviation metrics
    - common CSV logging
    - cleanup

The selected model adapter owns:
    - model/checkpoint
    - native cameras
    - GNSS/IMU if required
    - native route/navigation preprocessing
    - model inference
    - native PID/control logic

V1 intentionally supports HE only.
After TCP and NEAT reproduce their existing smoke behavior through this
runner, V2 will add a CARLA actor backend without modifying model adapters.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import carla
import numpy as np


# ============================================================
# Repository
# ============================================================

THIS_FILE = Path(__file__).resolve()

HE_ROOT = THIS_FILE.parents[2]

COMMON_DIR = (
    HE_ROOT
    / "driving_models"
    / "common"
)

TCP_DIR = (
    HE_ROOT
    / "driving_models"
    / "TCP"
)

NEAT_DIR = (
    HE_ROOT
    / "driving_models"
    / "NEAT"
)

for path in (
    COMMON_DIR,
    TCP_DIR,
    NEAT_DIR,
):
    text = str(path)

    if text not in sys.path:
        sys.path.insert(
            0,
            text,
        )


# ============================================================
# Shared HE runtime
# ============================================================

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


# ============================================================
# Model adapters
# ============================================================

from tcp_adapter_v1 import (
    TCPAdapterV1,
)

from neat_adapter_v1 import (
    NEATAdapterV1,
)


# ============================================================
# Defaults
# ============================================================

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
    / "outputs"
    / "generic_he_closed_loop_v1"
)


# ============================================================
# Model registry
# ============================================================

MODEL_REGISTRY = {
    "tcp":
        TCPAdapterV1,

    "neat":
        NEATAdapterV1,
}


# ============================================================
# Files
# ============================================================

def ensure_dir(
    path,
):

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# CARLA RGB
# ============================================================

def carla_image_to_rgb(
    image,
):

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

    # CARLA:
    # BGRA -> RGB

    return (
        array[
            :,
            :,
            :3,
        ][:, :, ::-1]
        .copy()
    )


# ============================================================
# Generic route helpers
# ============================================================

def import_global_route_planner(
    pythonapi_hint=None,
):
    import os

    def _load():
        from agents.navigation.global_route_planner import (
            GlobalRoutePlanner,
        )
        return GlobalRoutePlanner

    # --------------------------------------------------------
    # Already importable
    # --------------------------------------------------------

    try:
        return _load()

    except ModuleNotFoundError:
        pass

    # --------------------------------------------------------
    # Candidate CARLA PythonAPI locations
    # --------------------------------------------------------

    candidates = []

    if pythonapi_hint:
        candidates.append(
            Path(
                pythonapi_hint
            )
        )

    carla_root = os.environ.get(
        "CARLA_ROOT"
    )

    if carla_root:

        root = Path(
            carla_root
        )

        candidates.extend([
            root
            / "PythonAPI"
            / "carla",

            root
            / "PythonAPI",

            root,
        ])

    # Common local installations.
    candidates.extend([
        Path(r"E:\Carla\Carla_0.9.15\PythonAPI\carla"),
        Path(r"E:\Carla\Carla_0.9.15\PythonAPI"),
    ])

    # Repository-relative possibilities.
    for base in (
        Path.cwd(),
        HE_ROOT,
        HE_ROOT.parent,
    ):

        candidates.extend([
            base
            / "PythonAPI"
            / "carla",

            base
            / "PythonAPI",

            base
            / "carla",
        ])

    # --------------------------------------------------------
    # Resolve agents package
    # --------------------------------------------------------

    checked = []

    for candidate in candidates:

        try:
            path = (
                candidate
                .expanduser()
                .resolve()
            )

        except Exception:
            continue

        checked.append(
            str(
                path
            )
        )

        if (
            path
            / "agents"
        ).is_dir():

            import_root = path

        elif (
            path
            / "carla"
            / "agents"
        ).is_dir():

            import_root = (
                path
                / "carla"
            )

        else:
            continue

        text = str(
            import_root
        )

        if text not in sys.path:

            sys.path.insert(
                0,
                text,
            )

        try:
            return _load()

        except ModuleNotFoundError:
            continue

    raise RuntimeError(
        "Could not import CARLA agents package.\n"
        "Checked:\n  "
        + "\n  ".join(
            checked
        )
        +
        "\nPass --carla-pythonapi if CARLA "
        "is installed elsewhere."
    )


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


def route_length(
    route,
):

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


def closest_route_index(
    route,
    location,
    previous_idx,
):

    start = max(
        0,
        int(previous_idx) - 5,
    )

    end = min(
        len(route),
        int(previous_idx) + 100,
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

        distance = distance_2d(
            point,
            location,
        )

        if (
            distance
            <
            best_distance
        ):

            best_distance = (
                distance
            )

            best_idx = idx

    return (
        best_idx,
        best_distance,
    )


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

    if (
        int(
            destination_idx
        )
        >= 0
    ):

        destination_idx = (
            int(
                destination_idx
            )
            %
            len(
                spawn_points
            )
        )

        if (
            destination_idx
            ==
            start_idx
        ):

            raise ValueError(
                "Destination equals start spawn."
            )

        route = (
            grp.trace_route(
                start,
                spawn_points[
                    destination_idx
                ].location,
            )
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

        straight_distance = (
            distance_2d(
                start,
                tf.location,
            )
        )

        if (
            straight_distance
            >= 35.0
        ):

            candidates.append(
                (
                    abs(
                        straight_distance
                        -
                        float(
                            desired_m
                        )
                    ),
                    idx,
                )
            )

    candidates.sort()

    for _error, idx in candidates:

        route = (
            grp.trace_route(
                start,
                spawn_points[
                    idx
                ].location,
            )
        )

        if len(route) < 2:
            continue

        length_m = (
            route_length(
                route
            )
        )

        if (
            length_m
            >= 35.0
        ):

            return (
                route,
                idx,
            )

    raise RuntimeError(
        "Could not automatically select a usable route."
    )


# ============================================================
# Scenario execution origin
# ============================================================

def build_execution_origin(
    carla_map,
    ego0_tf,
):

    waypoint = (
        carla_map.get_waypoint(
            ego0_tf.location,

            project_to_road=True,

            lane_type=(
                carla.LaneType.Driving
            ),
        )
    )

    if waypoint is None:

        raise RuntimeError(
            "Could not resolve driving waypoint "
            "for canonical ego start."
        )

    return ExecutionWorldOrigin(
        x_m=float(
            ego0_tf.location.x
        ),

        y_m=float(
            ego0_tf.location.y
        ),

        z_m=float(
            waypoint
            .transform
            .location
            .z
        ),

        yaw_deg=float(
            ego0_tf.rotation.yaw
        ),
    )


# ============================================================
# HE compositor diagnostics
# ============================================================

def rendered_rows(
    composite,
):

    return [
        row
        for row
        in composite.actor_results
        if row.rendered
    ]


def rendered_actor_ids(
    composite,
):

    return [
        row.actor_id
        for row
        in rendered_rows(
            composite
        )
    ]


def nearest_rendered_depth(
    composite,
):

    rows = rendered_rows(
        composite
    )

    if not rows:
        return None

    return min(
        float(
            row.distance_forward_m
        )
        for row
        in rows
    )


def join_actor_ids(
    ids,
):

    return "|".join(
        str(v)
        for v
        in ids
    )


# ============================================================
# Adapter
# ============================================================

def create_adapter(
    args,
):

    model_name = (
        str(
            args.model
        )
        .strip()
        .lower()
    )

    if (
        model_name
        not in MODEL_REGISTRY
    ):

        raise KeyError(
            f"Unknown model: {model_name}"
        )

    adapter_cls = (
        MODEL_REGISTRY[
            model_name
        ]
    )

    return adapter_cls(
        args
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Generic model-independent HE "
                "multi-actor closed-loop runner."
            )
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(
            MODEL_REGISTRY.keys()
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    # TCP-specific optional paths.
    parser.add_argument(
        "--tcp-root",
        default=None,
    )

    # NEAT-specific optional path.
    parser.add_argument(
        "--neat-root",
        default=None,
    )

    # Shared checkpoint override.
    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    # --------------------------------------------------------
    # Scenario / HE
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
        "--physics-settle-ticks",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--canonical-hold-ticks",
        type=int,
        default=5,
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

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    parser.add_argument(
        "--debug-every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Resolve paths
    # ========================================================

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
    # Adapter
    # ========================================================

    adapter = (
        create_adapter(
            args
        )
    )

    adapter.load()

    print()
    print("=" * 78)
    print(
        "GENERIC HE CLOSED-LOOP RUNNER V1"
    )
    print("=" * 78)
    print(
        "model     :",
        adapter.model_name,
    )
    print(
        "adapter   :",
        adapter.summary(),
    )
    print(
        "condition : HE"
    )
    print("=" * 78)
    print()

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

    world = (
        client.load_world(
            args.town
        )
    )

    original_settings = (
        world.get_settings()
    )

    actors = []

    csv_fp = None

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
            /
            float(
                adapter.required_fps
            )
        )

        world.apply_settings(
            settings
        )

        # ====================================================
        # Map
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
            int(
                args.spawn_index
            )
            %
            len(
                spawn_points
            )
        )

        # ====================================================
        # Route
        # ====================================================

        GlobalRoutePlanner = (
            import_global_route_planner(
                args.carla_pythonapi
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
            grp=
                grp,

            spawn_points=
                spawn_points,

            start_idx=
                spawn_idx,

            destination_idx=
                int(
                    args.destination_index
                ),

            desired_m=
                float(
                    args.desired_route_distance_m
                ),
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

        ego = (
            world.try_spawn_actor(
                ego_bp,
                spawn_points[
                    spawn_idx
                ],
            )
        )

        if ego is None:

            raise RuntimeError(
                "Could not spawn ego."
            )

        actors.append(
            ego
        )

        # ====================================================
        # Adapter sensors
        # ====================================================

        sensor_actors = (
            adapter.setup(
                world=
                    world,

                ego=
                    ego,

                carla_map=
                    carla_map,

                route=
                    route,
            )
        )

        actors.extend(
            list(
                sensor_actors
            )
        )

        # ====================================================
        # Canonical ego initialization
        # ====================================================

        (
            ego0_tf,
            _ego0_state,
        ) = canonicalize_ego_start(
            world=
                world,

            ego=
                ego,

            nominal_spawn_tf=
                spawn_points[
                    spawn_idx
                ],

            settle_ticks=
                int(
                    args.physics_settle_ticks
                ),

            hold_ticks=
                int(
                    args.canonical_hold_ticks
                ),
        )

        print(
            "[init] EXPERIMENT START BARRIER"
        )

        # ====================================================
        # Scenario runtime
        # ====================================================

        origin = (
            build_execution_origin(
                carla_map=
                    carla_map,

                ego0_tf=
                    ego0_tf,
            )
        )

        runtime = (
            load_execution_runtime(
                resolved_json=
                    str(
                        resolved_path
                    ),

                asset_root=
                    asset_root,

                origin=
                    origin,

                manifest_path=
                    manifest_path,
            )
        )

        summary = (
            runtime.summary()
        )

        fps = float(
            summary[
                "fps"
            ]
        )

        if (
            abs(
                fps
                -
                float(
                    adapter.required_fps
                )
            )
            >
            1e-6
        ):

            raise RuntimeError(
                f"Scenario FPS={fps} but "
                f"{adapter.model_name} requires "
                f"{adapter.required_fps} Hz."
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
            +
            1
        )

        start_frame = max(
            0,
            int(
                args.start_frame
            ),
        )

        end_frame = (
            total_scenario_frames
            -
            1
        )

        if (
            int(
                args.max_frames
            )
            >
            0
        ):

            end_frame = min(
                end_frame,

                start_frame
                +
                int(
                    args.max_frames
                )
                -
                1,
            )

        if (
            end_frame
            <
            start_frame
        ):

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
        # HE compositor
        # ====================================================

        compositor = (
            HEMultiActorCompositorV1(
                distance_selection_mode=
                    args.distance_selection_mode,

                bottom_y_offset_px=
                    float(
                        args.he_bottom_y_offset_px
                    ),
            )
        )

        # ====================================================
        # Initial synchronized observation
        # ====================================================

        current_frame = (
            world.tick()
        )

        sensor_frame = (
            adapter.read_sensor_frame(
                current_frame
            )
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
            /
            adapter.model_name
        )

        csv_path = (
            output_dir
            /
            (
                f"{scenario_id}_"
                f"{adapter.model_name}_"
                f"he.csv"
            )
        )

        print(
            "[output]",
            output_dir,
        )

        print(
            "[CSV]",
            csv_path,
        )

        # ====================================================
        # Common CSV schema
        # ====================================================

        camera_names = (
            adapter.camera_names()
        )

        csv_fields = [
            "scenario_id",
            "model",
            "condition",

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

            "steer",
            "throttle",
            "brake",
            "bootstrap",

            "command_name",
            "command_value",
            "target_x",
            "target_y",

            "active_actor_count",
        ]

        for camera_name in (
            camera_names
        ):

            csv_fields.extend([
                f"{camera_name}_rendered_count",
                f"{camera_name}_rendered_ids",
                f"{camera_name}_nearest_depth_m",
            ])

        csv_fp = open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        writer = csv.DictWriter(
            csv_fp,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        # ====================================================
        # Runtime state
        # ====================================================

        route_idx = 0

        deviation_counter = 0

        completed_frames = 0

        max_brake = 0.0

        # ====================================================
        # Scenario loop
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
            # Raw native model cameras
            # ------------------------------------------------

            base_rgb_by_camera = {}

            for camera_name in (
                camera_names
            ):

                camera_sample = (
                    sensor_frame
                    .cameras[
                        camera_name
                    ]
                )

                base_rgb_by_camera[
                    camera_name
                ] = (
                    carla_image_to_rgb(
                        camera_sample
                    )
                )

            # ------------------------------------------------
            # Scenario actors
            # ------------------------------------------------

            active_actors = list(
                runtime.active_actors(
                    scenario_frame
                )
            )

            # ------------------------------------------------
            # Render independently into every native camera
            # ------------------------------------------------

            rgb_by_camera = {}

            composite_by_camera = {}

            specs = {
                spec.name:
                    spec

                for spec in
                adapter.camera_specs()
            }

            for camera_name in (
                camera_names
            ):

                spec = (
                    specs[
                        camera_name
                    ]
                )

                camera_sample = (
                    sensor_frame
                    .cameras[
                        camera_name
                    ]
                )

                composite = (
                    compositor.render(
                        base_rgb=
                            base_rgb_by_camera[
                                camera_name
                            ],

                        camera_tf=
                            camera_sample.transform,

                        active_actors=
                            active_actors,

                        width=
                            int(
                                spec.width
                            ),

                        height=
                            int(
                                spec.height
                            ),

                        fov=
                            float(
                                spec.fov_deg
                            ),
                    )
                )

                composite_by_camera[
                    camera_name
                ] = (
                    composite
                )

                rgb_by_camera[
                    camera_name
                ] = (
                    composite.rgb
                )

            # ------------------------------------------------
            # Ego speed
            # ------------------------------------------------

            velocity = (
                ego.get_velocity()
            )

            speed_mps = math.sqrt(
                velocity.x
                * velocity.x

                +
                velocity.y
                * velocity.y

                +
                velocity.z
                * velocity.z
            )

            # ------------------------------------------------
            # Model inference
            # ------------------------------------------------

            result = (
                adapter.step(
                    rgb_by_camera=
                        rgb_by_camera,

                    sensor_frame=
                        sensor_frame,

                    speed_mps=
                        speed_mps,

                    scenario_frame=
                        scenario_frame
                        -
                        start_frame,
                )
            )

            control = (
                result.control
            )

            # ------------------------------------------------
            # Apply model control
            # ------------------------------------------------

            ego.apply_control(
                carla.VehicleControl(
                    steer=float(
                        control.steer
                    ),

                    throttle=float(
                        control.throttle
                    ),

                    brake=float(
                        control.brake
                    ),

                    hand_brake=False,

                    manual_gear_shift=False,
                )
            )

            max_brake = max(
                max_brake,
                float(
                    control.brake
                ),
            )

            # ------------------------------------------------
            # Common route metrics
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
                route=
                    route,

                location=
                    ego_loc,

                previous_idx=
                    route_idx,
            )

            destination_distance = (
                distance_2d(
                    ego_loc,
                    destination,
                )
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
            # Generic log row
            # ------------------------------------------------

            meta = (
                result.metadata
                or {}
            )

            row = {
                "scenario_id":
                    scenario_id,

                "model":
                    adapter.model_name,

                "condition":
                    "he",

                "scenario_frame":
                    int(
                        scenario_frame
                    ),

                "carla_frame":
                    int(
                        sensor_frame.carla_frame
                    ),

                "t_s":
                    float(
                        t_s
                    ),

                "ego_x":
                    float(
                        ego_tf.location.x
                    ),

                "ego_y":
                    float(
                        ego_tf.location.y
                    ),

                "ego_z":
                    float(
                        ego_tf.location.z
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

                "steer":
                    float(
                        control.steer
                    ),

                "throttle":
                    float(
                        control.throttle
                    ),

                "brake":
                    float(
                        control.brake
                    ),

                "bootstrap":
                    int(
                        bool(
                            result.bootstrap
                        )
                    ),

                "command_name":
                    meta.get(
                        "command_name",
                        "",
                    ),

                "command_value":
                    meta.get(
                        "command_value",
                        "",
                    ),

                "target_x":
                    meta.get(
                        "target_x",
                        "",
                    ),

                "target_y":
                    meta.get(
                        "target_y",
                        "",
                    ),

                "active_actor_count":
                    len(
                        active_actors
                    ),
            }

            for camera_name in (
                camera_names
            ):

                composite = (
                    composite_by_camera[
                        camera_name
                    ]
                )

                ids = rendered_actor_ids(
                    composite
                )

                depth = nearest_rendered_depth(
                    composite
                )

                row[
                    f"{camera_name}_rendered_count"
                ] = len(
                    ids
                )

                row[
                    f"{camera_name}_rendered_ids"
                ] = join_actor_ids(
                    ids
                )

                row[
                    f"{camera_name}_nearest_depth_m"
                ] = (
                    ""
                    if depth is None
                    else float(
                        depth
                    )
                )

            writer.writerow(
                row
            )

            completed_frames += 1

            # ------------------------------------------------
            # Debug
            # ------------------------------------------------

            if (
                scenario_frame
                ==
                start_frame

                or
                scenario_frame
                ==
                end_frame

                or
                (
                    int(
                        args.debug_every
                    )
                    >
                    0

                    and

                    (
                        scenario_frame
                        -
                        start_frame
                    )
                    %
                    int(
                        args.debug_every
                    )
                    ==
                    0
                )
            ):

                camera_counts = []

                for camera_name in (
                    camera_names
                ):

                    count = len(
                        rendered_actor_ids(
                            composite_by_camera[
                                camera_name
                            ]
                        )
                    )

                    camera_counts.append(
                        f"{camera_name}={count}"
                    )

                print(
                    f"[{adapter.model_name}] "
                    f"frame={scenario_frame:04d} "
                    f"active={len(active_actors)} "
                    f"speed={speed_mps:.2f} "
                    f"S={control.steer:+.3f} "
                    f"T={control.throttle:.3f} "
                    f"B={control.brake:.3f} "
                    +
                    " ".join(
                        camera_counts
                    )
                )

            # ------------------------------------------------
            # Common stopping rules
            # ------------------------------------------------

            if (
                destination_distance
                <=
                float(
                    args.destination_tolerance_m
                )
            ):

                print(
                    "[stop] destination reached"
                )

                break

            if (
                deviation_counter
                >=
                int(
                    args.deviation_patience_frames
                )
            ):

                print(
                    "[stop] route deviation limit exceeded"
                )

                break

            # ------------------------------------------------
            # Next synchronized simulation frame
            # ------------------------------------------------

            if (
                scenario_frame
                <
                end_frame
            ):

                current_frame = (
                    world.tick()
                )

                sensor_frame = (
                    adapter.read_sensor_frame(
                        current_frame
                    )
                )

        # ====================================================
        # Final
        # ====================================================

        print()
        print("=" * 78)
        print(
            "GENERIC HE RUN COMPLETE"
        )
        print("=" * 78)

        print(
            "model:",
            adapter.model_name,
        )

        print(
            "scenario:",
            scenario_id,
        )

        print(
            "frames:",
            completed_frames,
        )

        print(
            "max brake:",
            f"{max_brake:.3f}",
        )

        print(
            "csv:",
            csv_path,
        )

        print("=" * 78)

    finally:

        # ====================================================
        # Cleanup
        # ====================================================

        try:
            adapter.close()

        except Exception as exc:

            print(
                "[adapter cleanup warning]",
                exc,
            )

        if csv_fp is not None:

            csv_fp.close()

        # Stop sensors before destruction.
        for actor in reversed(
            actors
        ):

            try:

                if (
                    actor is not None
                    and
                    actor.is_alive
                    and
                    actor.type_id.startswith(
                        "sensor."
                    )
                ):

                    actor.stop()

            except Exception:

                pass

        for actor in reversed(
            actors
        ):

            try:

                if (
                    actor is not None
                    and
                    actor.is_alive
                ):

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