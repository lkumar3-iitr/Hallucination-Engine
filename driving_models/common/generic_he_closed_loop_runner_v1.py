"""
generic_he_closed_loop_runner_v1.py

Generic HE-only closed-loop experiment runner.

This runner is model-independent.

Supported adapters:
    Dynamically discovered through:
        driving_models/<MODEL>/<model>_adapter_v1.py

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
import json
import math
import queue
import sys
import importlib.util
from pathlib import Path
import cv2
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

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(COMMON_DIR),
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
from he_multi_actor_compositor_v2 import (
    HEMultiActorCompositorV2,
)

from scenario_execution_runtime_v1 import (
    ExecutionWorldOrigin,
    load_execution_runtime,
)
from route_progress_metrics_v1 import (
    RoutePoint,
    RouteProjector,
    compute_route_pair_metrics,
    route_virtual_collision,
)
from oriented_box_metrics_v1 import (
    nearest_actor_footprint_metrics,
)

from traffic_light_schedule_v1 import (
    TrafficLightScheduleExecutor,
)
from scenario_event_clock_v1 import RouteTriggeredEventClock

# ============================================================
# Model adapters
# ============================================================


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


def make_scene_depth_camera(world, ego, spec):
    bp = world.get_blueprint_library().find("sensor.camera.depth")
    bp.set_attribute("image_size_x", str(int(spec.width)))
    bp.set_attribute("image_size_y", str(int(spec.height)))
    bp.set_attribute("fov", str(float(spec.fov_deg)))
    transform = carla.Transform(
        carla.Location(x=spec.x_m, y=spec.y_m, z=spec.z_m),
        carla.Rotation(
            pitch=spec.pitch_deg,
            yaw=spec.yaw_deg,
            roll=spec.roll_deg,
        ),
    )
    return world.spawn_actor(bp, transform, attach_to=ego)


def read_exact_sensor_frame(sensor_queue, target_frame, name):
    while True:
        sample = sensor_queue.get(timeout=20.0)
        frame = int(sample.frame)
        if frame < int(target_frame):
            continue
        if frame > int(target_frame):
            raise RuntimeError(
                f"{name} skipped target frame {target_frame}; received {frame}."
            )
        return sample


def carla_depth_image_to_m(depth_image):
    bgra = np.frombuffer(depth_image.raw_data, dtype=np.uint8).reshape(
        depth_image.height, depth_image.width, 4
    )
    blue = bgra[:, :, 0].astype(np.float32)
    green = bgra[:, :, 1].astype(np.float32)
    red = bgra[:, :, 2].astype(np.float32)
    normalized = (red + 256.0 * green + 65536.0 * blue) / 16777215.0
    return normalized * 1000.0


def resolve_weather_preset(carla_module, preset_name):
    if preset_name is None or str(preset_name).strip() == "":
        return None
    preset_name = str(preset_name).strip()
    if not hasattr(carla_module.WeatherParameters, preset_name):
        raise ValueError(f"Unknown CARLA weather preset: {preset_name}")
    return getattr(carla_module.WeatherParameters, preset_name)


def weather_preset_from_environment(path):
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    weather = data.get("weather") or {}
    return weather.get("preset")


# ============================================================
# Model registry
# ============================================================


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

def make_camera_mosaic_bgr(
    rgb_by_camera,
    camera_names,
):
    """
    Build a left-to-right debug mosaic of every native model camera.

    The images passed here are exactly the HE-composited RGB images
    supplied to the driving model.

    Cameras with different heights are resized to a common height while
    preserving aspect ratio. This affects only the saved debug video,
    never the model input.
    """

    frames = []

    for camera_name in camera_names:

        frame = rgb_by_camera.get(
            camera_name
        )

        if frame is None:
            continue

        frame = np.asarray(
            frame
        )

        if (
            frame.ndim != 3
            or
            frame.shape[2] != 3
        ):
            raise RuntimeError(
                f"Invalid RGB frame for "
                f"{camera_name}: "
                f"{frame.shape}"
            )

        frames.append(
            (
                str(camera_name),
                frame,
            )
        )

    if not frames:
        return None

    target_height = max(
        int(frame.shape[0])
        for _name, frame in frames
    )

    tiles = []

    for camera_name, frame_rgb in frames:

        h = int(
            frame_rgb.shape[0]
        )

        w = int(
            frame_rgb.shape[1]
        )

        if h != target_height:

            scale = (
                float(target_height)
                /
                float(h)
            )

            target_width = max(
                1,
                int(
                    round(
                        float(w)
                        *
                        scale
                    )
                ),
            )

            interpolation = (
                cv2.INTER_AREA
                if target_height < h
                else cv2.INTER_LINEAR
            )

            frame_rgb = cv2.resize(
                frame_rgb,
                (
                    target_width,
                    target_height,
                ),
                interpolation=interpolation,
            )

        # OpenCV VideoWriter expects BGR.
        tile = (
            frame_rgb[
                :,
                :,
                ::-1
            ]
            .copy()
        )

        # Debug label only; does not affect model input.
        cv2.putText(
            tile,
            camera_name,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            tile,
            camera_name,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        tiles.append(
            tile
        )

    if len(tiles) == 1:
        return tiles[0]

    # Small separator between native camera views.
    separator_width_px = 6

    separator = np.zeros(
        (
            target_height,
            separator_width_px,
            3,
        ),
        dtype=np.uint8,
    )

    mosaic_parts = []

    for index, tile in enumerate(
        tiles
    ):

        if index > 0:
            mosaic_parts.append(
                separator
            )

        mosaic_parts.append(
            tile
        )

    return np.concatenate(
        mosaic_parts,
        axis=1,
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


def route_projector_from_carla_route(route):
    points = []
    progress_m = 0.0
    previous = None
    for route_idx, item in enumerate(route):
        location = item[0].transform.location
        if previous is not None:
            progress_m += math.hypot(
                float(location.x) - float(previous.x),
                float(location.y) - float(previous.y),
            )
        points.append(
            RoutePoint(
                route_idx=int(route_idx),
                s_m=float(progress_m),
                x=float(location.x),
                y=float(location.y),
            )
        )
        previous = location
    return RouteProjector(points)


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
def execution_actor_to_carla_transform(
    carla_map,
    state,
):
    location = carla.Location(
        x=float(state.world_x_m),
        y=float(state.world_y_m),
        z=float(state.world_z_m),
    )

    waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if waypoint is not None:
        z = float(
            waypoint.transform.location.z
        ) + float(state.ground_origin_offset_m)

        pitch = float(
            waypoint.transform.rotation.pitch
        )

        roll = float(
            waypoint.transform.rotation.roll
        )
    else:
        z = float(state.world_z_m)
        pitch = 0.0
        roll = 0.0

    return carla.Transform(
        carla.Location(
            x=float(state.world_x_m),
            y=float(state.world_y_m),
            z=z,
        ),
        carla.Rotation(
            pitch=pitch,
            yaw=float(state.world_yaw_deg),
            roll=roll,
        ),
    )


def sync_carla_scenario_actors(
    world,
    carla_map,
    actor_by_id,
    active_states,
):
    active_ids = set()

    for state in active_states:
        actor_id = str(
            state.actor_id
        )

        active_ids.add(
            actor_id
        )

        transform = (
            execution_actor_to_carla_transform(
                carla_map,
                state,
            )
        )

        actor = actor_by_id.get(
            actor_id
        )

        if actor is None:
            blueprint = (
                world
                .get_blueprint_library()
                .find(
                    state.carla_blueprint
                )
            )

            if blueprint.has_attribute(
                "role_name"
            ):
                blueprint.set_attribute(
                    "role_name",
                    "scenario_actor",
                )

            actor = world.try_spawn_actor(
                blueprint,
                transform,
            )

            if actor is None:
                raised_transform = carla.Transform(
                    carla.Location(
                        x=transform.location.x,
                        y=transform.location.y,
                        z=transform.location.z + 0.5,
                    ),
                    transform.rotation,
                )
                actor = world.try_spawn_actor(
                    blueprint,
                    raised_transform,
                )

            if actor is None:
                lane_aligned_transform = carla.Transform(
                    carla.Location(
                        x=transform.location.x,
                        y=transform.location.y,
                        z=transform.location.z + 1.0,
                    ),
                    carla.Rotation(
                        pitch=transform.rotation.pitch,
                        yaw=0.0,
                        roll=transform.rotation.roll,
                    ),
                )
                actor = world.try_spawn_actor(
                    blueprint,
                    lane_aligned_transform,
                )

            staged_spawn = False
            if actor is None:
                # Large actors such as the Fuso bus can be rejected at an
                # otherwise valid resolved pose by CARLA's spawn-overlap
                # check. Stage the kinematic actor well above the scene, then
                # place it at the scenario-owned transform below.
                staging_transform = carla.Transform(
                    carla.Location(
                        x=transform.location.x,
                        y=transform.location.y,
                        z=transform.location.z + 50.0,
                    ),
                    transform.rotation,
                )
                actor = world.try_spawn_actor(
                    blueprint,
                    staging_transform,
                )
                staged_spawn = actor is not None

            if actor is None:
                raise RuntimeError(
                    "Could not spawn scenario actor "
                    f"{actor_id}: "
                    f"{state.carla_blueprint}"
                )

            try:
                actor.set_simulate_physics(
                    False
                )
            except Exception:
                pass

            if staged_spawn:
                print(
                    "[CARLA scenario actor staged]",
                    actor_id,
                    "direct spawn rejected; using exact kinematic pose",
                )

            actor_by_id[
                actor_id
            ] = actor

            print(
                "[CARLA scenario actor]",
                actor_id,
                "->",
                state.carla_blueprint,
                "runtime_id=",
                actor.id,
            )

        actor.set_transform(
            transform
        )

    # Actors outside their active interval are hidden.
    for actor_id, actor in (
        actor_by_id.items()
    ):
        if actor_id in active_ids:
            continue

        transform = (
            actor.get_transform()
        )

        transform.location.z = -1000.0

        actor.set_transform(
            transform
        )

# ============================================================
# HE compositor diagnostics
# ============================================================

def rendered_rows(
    composite,
):
    if composite is None:
        return []
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

def load_adapter_class(
    model_name,
):
    """
    Dynamically discover a driving-model adapter.

    Convention:

        --model tcp
            driving_models/TCP/tcp_adapter_v1.py

        --model neat
            driving_models/NEAT/neat_adapter_v1.py

        --model cilpp
            driving_models/CILPP/cilpp_adapter_v1.py

        --model aim
            driving_models/AIM/aim_adapter_v1.py

    Every adapter module exposes:

        ADAPTER_CLASS = ...
    """

    model_name = (
        str(model_name)
        .strip()
        .lower()
    )

    if not model_name:
        raise ValueError(
            "Model name cannot be empty."
        )

    model_folder = (
        HE_ROOT
        / "driving_models"
        / model_name.upper()
    )

    adapter_file = (
        model_folder
        / f"{model_name}_adapter_v1.py"
    )

    if not adapter_file.exists():

        raise FileNotFoundError(
            "Could not find model adapter:\n"
            f"{adapter_file}\n\n"
            "Expected convention:\n"
            "driving_models\\MODEL\\model_adapter_v1.py"
        )

    # Make model-local imports work.
    model_folder_text = str(
        model_folder
    )

    if (
        model_folder_text
        not in sys.path
    ):

        sys.path.insert(
            0,
            model_folder_text,
        )

    module_name = (
        f"he_dynamic_adapter_"
        f"{model_name}"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            adapter_file,
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):

        raise ImportError(
            f"Could not load adapter module: "
            f"{adapter_file}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    adapter_cls = getattr(
        module,
        "ADAPTER_CLASS",
        None,
    )

    if adapter_cls is None:

        raise AttributeError(
            f"{adapter_file} does not expose "
            "ADAPTER_CLASS."
        )

    return adapter_cls


def create_adapter(
    args,
):

    adapter_cls = (
        load_adapter_class(
            args.model
        )
    )

    return adapter_cls(
        args
    )


# ============================================================
# Main
# ============================================================

def scene_occlusion_summary(composite):
    rows = [] if composite is None else composite.actor_results
    metadata = [
        row.he_metadata.get("scene_occlusion", {})
        for row in rows
    ]
    used = [row for row in metadata if row.get("enabled")]
    fractions = [float(row.get("occluded_fraction", 0.0)) for row in used]
    return {
        "depth_used": int(bool(used)),
        "occluded_actor_count": sum(value > 0.0 for value in fractions),
        "fully_occluded_actor_count": sum(value >= 0.999 for value in fractions),
        "maximum_occluded_fraction": max(fractions, default=0.0),
    }


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
        help=(
            "Driving model adapter name, e.g. "
            "tcp, neat, cilpp, aim."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--ego-blueprint",
        default=None,
        help=(
            "Optional CARLA ego blueprint override. By default, "
            "the runner uses the vehicle associated with each "
            "model's original evaluation stack."
        ),
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
        "--environment-json",
        default=None,
        help=(
            "Optional ScenarioGenerator environment sidecar "
            "for deterministic traffic-light schedules."
        ),
    )
    parser.add_argument(
        "--weather-preset",
        default=None,
        help=(
            "Optional CARLA WeatherParameters preset, e.g. "
            "ClearNoon or HardRainNoon. Overrides environment weather."
        ),
    )

    parser.add_argument(
        "--route-metrics-csv",
        default=None,
        help=(
            "Optional inspected route CSV used for "
            "turn-aware gap/TTC metrics."
        ),
    )

    parser.add_argument(
        "--metric-actor-id",
        default=None,
        help=(
            "Scenario actor used for pairwise route/safety "
            "metrics. If omitted and exactly one actor is "
            "active, that actor is used."
        ),
    )

    parser.add_argument(
        "--event-start-s",
        type=float,
        default=None,
        help=(
            "Scenario safety-event start time. Logged so "
            "response-time metrics can be calculated later."
        ),
    )
    parser.add_argument(
        "--trigger-route-progress-m",
        type=float,
        default=None,
        help="Start the authored event when ego reaches this route progress.",
    )
    parser.add_argument(
        "--event-source-start-s",
        type=float,
        default=None,
        help="Authored trajectory time corresponding to trigger-relative t=0.",
    )
    parser.add_argument(
        "--pre-trigger-source-frame",
        type=int,
        default=None,
        help="Optional authored staging frame held before the route trigger.",
    )
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
        "--condition",
        choices=["he", "carla"],
        default="he",
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
        "--he-renderer-version",
        choices=["v1", "v2"],
        default="v1",
        help="HE compositor implementation. V1 remains the control/default.",
    )

    parser.add_argument(
        "--he-bottom-y-offset-px",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--he-silhouette-scale",
        type=float,
        default=1.0,
        help="Uniform HE silhouette scale around the physical support anchor.",
    )
    parser.add_argument(
        "--he-warp-scale-mode",
        choices=[
            "independent",
            "uniform_height_preserve_aspect",
        ],
        default="independent",
        help=(
            "View-matrix sprite warp scaling mode. The default preserves "
            "existing independent X/Y scaling."
        ),
    )
    parser.add_argument(
        "--he-viewpoint-lateral-sign",
        type=float,
        choices=[
            -1.0,
            1.0,
        ],
        default=1.0,
        help=(
            "Sign applied to camera-right before querying the sprite-bank "
            "viewpoint angle. Projection and metrics are unchanged."
        ),
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
    parser.add_argument(
        "--save-video",
        action="store_true",
    )

    parser.add_argument(
        "--video-camera",
        default=None,
        help=(
            "Native camera to save. "
            "Defaults to rgb_central/front/first camera."
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
        "condition :",
        args.condition.upper(),
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

    weather_preset_name = (
        args.weather_preset
        or weather_preset_from_environment(args.environment_json)
    )

    weather = resolve_weather_preset(carla, weather_preset_name)
    if weather is not None:
        world.set_weather(weather)
        print("[weather]", weather_preset_name)

    actors = []
    scenario_carla_actors = {}
    csv_fp = None
    video_writer = None
    mosaic_writer = None
    mosaic_path = None
    traffic_light_executor = None
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

        model_ego_blueprints = {
            # The bundled TCP Leaderboard stack spawns
            # vehicle.lincoln.mkz2017. CARLA 0.9.15 exposes the
            # same vehicle under the blueprint name below.
            "tcp": "vehicle.lincoln.mkz_2017",
            "neat": "vehicle.tesla.model3",
            "cilpp": "vehicle.lincoln.mkz_2017",
            "aimmt": "vehicle.lincoln.mkz_2017",
        }

        ego_blueprint = (
            args.ego_blueprint
            or model_ego_blueprints[
                args.model
            ]
        )

        ego_bp = (
            world
            .get_blueprint_library()
            .find(
                ego_blueprint
            )
        )

        print(
            "[ego blueprint]",
            ego_blueprint,
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

        # Scene depth is an HE compositor input, not a model input.
        # Every depth camera is exactly co-located with its native RGB camera.
        scene_depth_cameras = {}
        scene_depth_queues = {}
        if args.condition == "he":
            for spec in adapter.camera_specs():
                depth_camera = make_scene_depth_camera(world, ego, spec)
                depth_queue = queue.Queue()
                depth_camera.listen(depth_queue.put)
                scene_depth_cameras[spec.name] = depth_camera
                scene_depth_queues[spec.name] = depth_queue
                actors.append(depth_camera)
            print(
                "[HE scene depth] synchronized cameras:",
                ", ".join(sorted(scene_depth_cameras)),
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

        compositor_class = (
            HEMultiActorCompositorV2
            if args.he_renderer_version == "v2"
            else HEMultiActorCompositorV1
        )
        compositor = (
            compositor_class(
                distance_selection_mode=
                    args.distance_selection_mode,

                bottom_y_offset_px=
                    float(
                        args.he_bottom_y_offset_px
                    ),
                silhouette_scale=float(args.he_silhouette_scale),
                warp_scale_mode=str(args.he_warp_scale_mode),
                viewpoint_lateral_sign=float(args.he_viewpoint_lateral_sign),
            )
        )
        # ====================================================
        # Turn-aware benchmark metrics
        # ====================================================

        route_projector = route_projector_from_carla_route(route)

        route_metric_ego_segment = None
        route_metric_actor_segment = None

        ego_bb = ego.bounding_box

        ego_length_m = (
            2.0
            *
            float(
                ego_bb.extent.x
            )
        )

        ego_width_m = (
            2.0
            *
            float(
                ego_bb.extent.y
            )
        )

        if args.route_metrics_csv is not None:

            route_projector = (
                RouteProjector.from_csv(
                    args.route_metrics_csv
                )
            )

            print(
                "[route metrics]",
                Path(
                    args.route_metrics_csv
                ).resolve(),
            )

            print(
                "[route metrics length]",
                f"{route_projector.total_length_m:.2f} m",
            )

            print(
                "[ego physical dimensions]",
                f"L={ego_length_m:.3f} "
                f"W={ego_width_m:.3f}",
            )
        else:
            print(
                "[route metrics] in-memory CARLA route",
                f"length={route_projector.total_length_m:.2f} m",
            )
            print(
                "[ego physical dimensions]",
                f"L={ego_length_m:.3f} W={ego_width_m:.3f}",
            )

        event_source_start_s = (
            args.event_start_s
            if args.event_source_start_s is None
            else args.event_source_start_s
        )
        event_source_frame = int(round(
            float(event_source_start_s or 0.0) * fps
        ))
        event_clock = RouteTriggeredEventClock(
            trigger_route_progress_m=args.trigger_route_progress_m,
            event_source_frame=event_source_frame,
            pre_trigger_source_frame=args.pre_trigger_source_frame,
        )
        actor_source_frame = event_clock.source_frame(start_frame)
        event_route_segment = None

        # ====================================================
        # Deterministic environment schedule
        # ====================================================

        if args.environment_json is not None:

            traffic_light_executor = (
                TrafficLightScheduleExecutor.from_json(
                    world=world,
                    carla_module=carla,
                    path=args.environment_json,
                )
            )

            traffic_light_executor.initialize()

            start_t_s = (
                float(start_frame)
                /
                float(fps)
            )

            traffic_light_executor.apply(
                start_t_s
            )

            print(
                "[environment]",
                Path(
                    args.environment_json
                ).resolve(),
            )

            print(
                "[traffic-light initial state]",
                traffic_light_executor
                .primary_state_name(
                    start_t_s
                ),
            )
        if args.condition == "carla":
            sync_carla_scenario_actors(
                world=world,
                carla_map=carla_map,
                actor_by_id=scenario_carla_actors,
                active_states=runtime.active_actors(
                    actor_source_frame
                ),
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
        print("[HE renderer]", args.he_renderer_version)

        scene_depth_by_camera = {}
        if args.condition == "he":
            for camera_name, depth_queue in scene_depth_queues.items():
                depth_image = read_exact_sensor_frame(
                    depth_queue, current_frame, f"HE {camera_name} depth"
                )
                scene_depth_by_camera[camera_name] = carla_depth_image_to_m(
                    depth_image
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
                f"{args.condition}.csv"
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
        # ====================================================
        # Video camera
        # ====================================================

        video_camera = None
        video_path = None

        if args.save_video:

            if args.video_camera is not None:

                video_camera = str(
                    args.video_camera
                )

            elif "rgb_central" in camera_names:

                video_camera = "rgb_central"

            elif "front" in camera_names:

                video_camera = "front"

            else:

                video_camera = camera_names[0]

            if video_camera not in camera_names:

                raise ValueError(
                    f"Unknown video camera "
                    f"{video_camera!r}. "
                    f"Available: {camera_names}"
                )

            spec_map = {
                spec.name: spec
                for spec in adapter.camera_specs()
            }

            video_spec = (
                spec_map[
                    video_camera
                ]
            )

            video_path = (
                output_dir
                /
                (
                    f"{scenario_id}_"
                    f"{adapter.model_name}_"
                    f"{args.condition}_{video_camera}.mp4"
                )
            )

            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),
                float(fps),
                (
                    int(video_spec.width),
                    int(video_spec.height),
                ),
            )

            if not video_writer.isOpened():

                raise RuntimeError(
                    "Could not open video writer: "
                    f"{video_path}"
                )

            print(
                "[VIDEO]",
                video_path,
            )
        csv_fields = [
            "scenario_id",
            "model",
            "condition",
            "weather_preset",
            "ego_blueprint",

            "scenario_frame",
            "actor_source_frame",
            "carla_frame",
            "t_s",
            "traffic_light_state",
            "event_start_s",
            "trigger_route_progress_m",
            "trigger_frame",
            "trigger_relative_frame",
            "trigger_relative_s",
            "trigger_observed_progress_m",

            "ego_x",
            "ego_y",
            "ego_z",
            "ego_yaw",
            "ego_speed_mps",
            "ego_acceleration_mps2",
            "ego_jerk_mps3",

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
            "nearest_actor_id",
            "nearest_actor_center_distance_m",
            "nearest_actor_clearance_m",
            "physical_overlap",
            "metric_actor_id",
            "metric_actor_speed_mps",

            "ego_route_progress_m",
            "actor_route_progress_m",

            "route_center_gap_m",
            "route_bumper_gap_m",

            "ego_route_lateral_m",
            "actor_route_lateral_m",
            "route_lateral_separation_m",

            "ego_route_speed_mps",
            "actor_route_speed_mps",
            "route_closing_speed_mps",

            "route_ttc_s",
            "route_virtual_collision",
        ]

        for camera_name in (
            camera_names
        ):

            csv_fields.extend([
                f"{camera_name}_rendered_count",
                f"{camera_name}_rendered_ids",
                f"{camera_name}_nearest_depth_m",
                f"{camera_name}_scene_depth_used",
                f"{camera_name}_occluded_actor_count",
                f"{camera_name}_fully_occluded_actor_count",
                f"{camera_name}_maximum_occluded_fraction",

                f"{camera_name}_selected_angle",
                f"{camera_name}_selected_distance_m",
                f"{camera_name}_selected_elevation_deg",
                f"{camera_name}_silhouette_scale",
                f"{camera_name}_warp_scale_mode",
                f"{camera_name}_viewpoint_lateral_sign",
                f"{camera_name}_sprite_width_px",
                f"{camera_name}_sprite_height_px",
                f"{camera_name}_target_box_width_px",
                f"{camera_name}_target_box_height_px",
                f"{camera_name}_subpixel_scale_x",
                f"{camera_name}_subpixel_scale_y",

                f"{camera_name}_anchor_mode",

                f"{camera_name}_source_anchor_x_px",
                f"{camera_name}_source_anchor_y_px",

                f"{camera_name}_target_anchor_x_px",
                f"{camera_name}_target_anchor_y_px",

                f"{camera_name}_legacy_source_anchor_x_px",
                f"{camera_name}_legacy_source_anchor_y_px",

                f"{camera_name}_paste_x1",
                f"{camera_name}_paste_y1",
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
        previous_speed_mps = None
        previous_acceleration_mps2 = None

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
                    actor_source_frame
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

                if args.condition == "he":

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
                            scene_depth_m=scene_depth_by_camera[
                                camera_name
                            ],
                        )
                    )

                    output_rgb = (
                        composite.rgb
                    )

                else:

                    composite = None

                    output_rgb = (
                        base_rgb_by_camera[
                            camera_name
                        ]
                    )

                composite_by_camera[
                    camera_name
                ] = (
                    composite
                )

                rgb_by_camera[
                    camera_name
                ] = (
                    output_rgb
                )

            # ------------------------------------------------
            # Save HE-composited native-camera video
            # ------------------------------------------------

            if video_writer is not None:

                video_rgb = (
                    rgb_by_camera[
                        video_camera
                    ]
                )

                video_bgr = cv2.cvtColor(
                    video_rgb,
                    cv2.COLOR_RGB2BGR,
                )

                video_writer.write(
                    video_bgr
                )
            # ------------------------------------------------
            # Multi-camera debug mosaic
            #
            # This uses the exact HE-composited images supplied
            # to the driving model.
            # ------------------------------------------------

            if (
                args.save_video
                and
                len(camera_names) > 1
            ):

                mosaic_bgr = (
                    make_camera_mosaic_bgr(
                        rgb_by_camera=
                            rgb_by_camera,

                        camera_names=
                            camera_names,
                    )
                )

                if mosaic_bgr is not None:

                    if mosaic_writer is None:

                        mosaic_path = (
                            output_dir
                            /
                            (
                                f"{scenario_id}_"
                                f"{adapter.model_name}_"
                                f"{args.condition}_all_cameras.mp4"
                            )
                        )

                        mosaic_height = int(
                            mosaic_bgr.shape[0]
                        )

                        mosaic_width = int(
                            mosaic_bgr.shape[1]
                        )

                        mosaic_writer = (
                            cv2.VideoWriter(
                                str(
                                    mosaic_path
                                ),
                                cv2.VideoWriter_fourcc(
                                    *"mp4v"
                                ),
                                float(
                                    fps
                                ),
                                (
                                    mosaic_width,
                                    mosaic_height,
                                ),
                            )
                        )

                        if not mosaic_writer.isOpened():

                            raise RuntimeError(
                                "Could not open "
                                "mosaic video writer: "
                                f"{mosaic_path}"
                            )

                        print(
                            "[MOSAIC VIDEO]",
                            mosaic_path,
                        )

                        print(
                            "[MOSAIC SIZE]",
                            f"{mosaic_width}x"
                            f"{mosaic_height}",
                        )

                    mosaic_writer.write(
                        mosaic_bgr
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
            # Deterministic environment state for this frame
            # ------------------------------------------------

            traffic_light_state = ""

            if traffic_light_executor is not None:

                traffic_light_state = (
                    traffic_light_executor
                    .primary_state_name(
                        t_s
                    )
                )

            # ------------------------------------------------
            # Select actor for safety metrics
            # ------------------------------------------------

            metric_actor = None

            footprint_metrics = nearest_actor_footprint_metrics(
                ego_x_m=float(ego_loc.x),
                ego_y_m=float(ego_loc.y),
                ego_yaw_deg=float(ego_tf.rotation.yaw),
                ego_length_m=ego_length_m,
                ego_width_m=ego_width_m,
                actors=active_actors,
            )

            if previous_speed_mps is None:
                acceleration_mps2 = 0.0
            else:
                acceleration_mps2 = (
                    float(speed_mps) - float(previous_speed_mps)
                ) * float(fps)

            if previous_acceleration_mps2 is None:
                jerk_mps3 = 0.0
            else:
                jerk_mps3 = (
                    float(acceleration_mps2)
                    - float(previous_acceleration_mps2)
                ) * float(fps)

            previous_speed_mps = float(speed_mps)
            previous_acceleration_mps2 = float(acceleration_mps2)

            if args.metric_actor_id is not None:

                requested_actor_id = str(
                    args.metric_actor_id
                )

                for candidate in active_actors:

                    if (
                        str(
                            candidate.actor_id
                        )
                        ==
                        requested_actor_id
                    ):

                        metric_actor = candidate
                        break

            elif len(active_actors) == 1:

                metric_actor = (
                    active_actors[0]
                )

            # ------------------------------------------------
            # Turn-aware route pair metrics
            # ------------------------------------------------

            pair_metrics = None
            pair_virtual_collision = False

            if (
                route_projector is not None
                and
                metric_actor is not None
                and
                metric_actor.physical_dimensions
                is not None
            ):

                actor_dims = (
                    metric_actor
                    .physical_dimensions
                )

                pair_metrics = (
                    compute_route_pair_metrics(
                        projector=
                            route_projector,

                        ego_x=
                            float(
                                ego_loc.x
                            ),

                        ego_y=
                            float(
                                ego_loc.y
                            ),

                        ego_vx=
                            float(
                                velocity.x
                            ),

                        ego_vy=
                            float(
                                velocity.y
                            ),

                        actor_x=
                            float(
                                metric_actor
                                .world_x_m
                            ),

                        actor_y=
                            float(
                                metric_actor
                                .world_y_m
                            ),

                        ego_length_m=
                            ego_length_m,

                        actor_length_m=
                            float(
                                actor_dims
                                .length_m
                            ),

                        actor_route_speed_mps=
                            float(
                                metric_actor
                                .speed_mps
                            ),

                        previous_ego_segment_idx=
                            route_metric_ego_segment,

                        previous_actor_segment_idx=
                            route_metric_actor_segment,
                    )
                )

                route_metric_ego_segment = (
                    pair_metrics
                    .ego
                    .segment_idx
                )

                route_metric_actor_segment = (
                    pair_metrics
                    .actor
                    .segment_idx
                )

                pair_virtual_collision = (
                    route_virtual_collision(
                        metrics=
                            pair_metrics,
                        ego_length_m=ego_length_m,

                        actor_length_m=float(
                            actor_dims.length_m
                        ),
                        ego_width_m=
                            ego_width_m,

                        actor_width_m=
                            float(
                                actor_dims
                                .width_m
                            ),
                    )
                )
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
                    args.condition,

                "weather_preset":
                    weather_preset_name or "",

                "ego_blueprint":
                    ego_blueprint,

                "scenario_frame":
                    int(
                        scenario_frame
                    ),

                "actor_source_frame": int(actor_source_frame),

                "carla_frame":
                    int(
                        sensor_frame.carla_frame
                    ),

                "t_s":
                    float(
                        t_s
                    ),
                "traffic_light_state":
                    traffic_light_state,

                "event_start_s":
                    (
                        ""
                        if args.event_start_s
                        is None
                        else float(
                            args.event_start_s
                        )
                    ),
                "trigger_route_progress_m": (
                    "" if args.trigger_route_progress_m is None
                    else float(args.trigger_route_progress_m)
                ),
                "trigger_frame": (
                    "" if event_clock.trigger_frame is None
                    else int(event_clock.trigger_frame)
                ),
                "trigger_relative_frame": (
                    "" if event_clock.relative_frame(scenario_frame) is None
                    else int(event_clock.relative_frame(scenario_frame))
                ),
                "trigger_relative_s": (
                    "" if event_clock.relative_frame(scenario_frame) is None
                    else float(event_clock.relative_frame(scenario_frame)) / fps
                ),
                "trigger_observed_progress_m": (
                    "" if event_clock.trigger_observed_progress_m is None
                    else float(event_clock.trigger_observed_progress_m)
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

                "ego_acceleration_mps2":
                    float(acceleration_mps2),

                "ego_jerk_mps3":
                    float(jerk_mps3),

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
                "nearest_actor_id":
                    (
                        "" if footprint_metrics is None
                        else footprint_metrics["actor_id"]
                    ),
                "nearest_actor_center_distance_m":
                    (
                        "" if footprint_metrics is None
                        else footprint_metrics["center_distance_m"]
                    ),
                "nearest_actor_clearance_m":
                    (
                        "" if footprint_metrics is None
                        else footprint_metrics["clearance_m"]
                    ),
                "physical_overlap":
                    (
                        "" if footprint_metrics is None
                        else int(footprint_metrics["overlaps"])
                    ),
                "metric_actor_id":
                    (
                        ""
                        if metric_actor is None
                        else str(
                            metric_actor.actor_id
                        )
                    ),

                "metric_actor_speed_mps":
                    (
                        ""
                        if metric_actor is None
                        else float(
                            metric_actor.speed_mps
                        )
                    ),

                "ego_route_progress_m":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .ego
                            .s_m
                        )
                    ),

                "actor_route_progress_m":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .actor
                            .s_m
                        )
                    ),

                "route_center_gap_m":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .center_gap_m
                        )
                    ),

                "route_bumper_gap_m":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .bumper_gap_m
                        )
                    ),

                "ego_route_lateral_m":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .ego
                            .lateral_m
                        )
                    ),

                "actor_route_lateral_m":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .actor
                            .lateral_m
                        )
                    ),

                "route_lateral_separation_m":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .lateral_separation_m
                        )
                    ),

                "ego_route_speed_mps":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .ego_route_speed_mps
                        )
                    ),

                "actor_route_speed_mps":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .actor_route_speed_mps
                        )
                    ),

                "route_closing_speed_mps":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .closing_speed_mps
                        )
                    ),

                "route_ttc_s":
                    (
                        ""
                        if pair_metrics is None
                        else float(
                            pair_metrics
                            .ttc_s
                        )
                    ),

                "route_virtual_collision":
                    (
                        ""
                        if pair_metrics is None
                        else int(
                            bool(
                                pair_virtual_collision
                            )
                        )
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

                occlusion = scene_occlusion_summary(composite)
                row[f"{camera_name}_scene_depth_used"] = int(
                    args.condition == "he"
                    and camera_name in scene_depth_by_camera
                )
                row[f"{camera_name}_occluded_actor_count"] = occlusion["occluded_actor_count"]
                row[f"{camera_name}_fully_occluded_actor_count"] = occlusion["fully_occluded_actor_count"]
                row[f"{camera_name}_maximum_occluded_fraction"] = occlusion["maximum_occluded_fraction"]
                rendered = rendered_rows(
                    composite
                )

                nearest_actor = None

                if rendered:
                    nearest_actor = min(
                        rendered,
                        key=lambda actor_result:
                            float(
                                actor_result.distance_forward_m
                            ),
                    )

                if nearest_actor is not None:

                    he_meta = (
                        nearest_actor.he_metadata
                        or
                        {}
                    )

                else:

                    he_meta = {}

                row[
                    f"{camera_name}_selected_angle"
                ] = he_meta.get(
                    "selected_angle",
                    "",
                )

                row[
                    f"{camera_name}_selected_distance_m"
                ] = he_meta.get(
                    "selected_distance_m",
                    "",
                )

                row[
                    f"{camera_name}_selected_elevation_deg"
                ] = he_meta.get(
                    "selected_elevation_deg",
                    "",
                )
                row[f"{camera_name}_silhouette_scale"] = he_meta.get(
                    "silhouette_scale",
                    "" if args.condition != "he" else float(args.he_silhouette_scale),
                )
                resize_info = he_meta.get(
                    "resize_info",
                    {},
                )
                if not isinstance(resize_info, dict):
                    resize_info = {}
                row[f"{camera_name}_warp_scale_mode"] = he_meta.get(
                    "warp_scale_mode",
                    resize_info.get("scale_mode", ""),
                )
                row[f"{camera_name}_viewpoint_lateral_sign"] = he_meta.get(
                    "viewpoint_lateral_sign",
                    "",
                )
                row[f"{camera_name}_sprite_width_px"] = he_meta.get(
                    "sprite_width",
                    "",
                )
                row[f"{camera_name}_sprite_height_px"] = he_meta.get(
                    "sprite_height",
                    "",
                )
                row[f"{camera_name}_target_box_width_px"] = he_meta.get(
                    "target_box_width_px",
                    "",
                )
                row[f"{camera_name}_target_box_height_px"] = he_meta.get(
                    "target_box_height_px",
                    "",
                )
                row[f"{camera_name}_subpixel_scale_x"] = resize_info.get(
                    "scale_x",
                    "",
                )
                row[f"{camera_name}_subpixel_scale_y"] = resize_info.get(
                    "scale_y",
                    "",
                )

                row[
                    f"{camera_name}_anchor_mode"
                ] = he_meta.get(
                    "anchor_mode",
                    "",
                )

                row[
                    f"{camera_name}_source_anchor_x_px"
                ] = he_meta.get(
                    "source_anchor_x_px",
                    "",
                )

                row[
                    f"{camera_name}_source_anchor_y_px"
                ] = he_meta.get(
                    "source_anchor_y_px",
                    "",
                )

                row[
                    f"{camera_name}_target_anchor_x_px"
                ] = he_meta.get(
                    "target_anchor_x_px",
                    "",
                )

                row[
                    f"{camera_name}_target_anchor_y_px"
                ] = he_meta.get(
                    "target_anchor_y_px",
                    "",
                )

                row[
                    f"{camera_name}_legacy_source_anchor_x_px"
                ] = he_meta.get(
                    "legacy_source_anchor_x_px",
                    "",
                )

                row[
                    f"{camera_name}_legacy_source_anchor_y_px"
                ] = he_meta.get(
                    "legacy_source_anchor_y_px",
                    "",
                )

                row[
                    f"{camera_name}_paste_x1"
                ] = he_meta.get(
                    "paste_x1",
                    "",
                )

                row[
                    f"{camera_name}_paste_y1"
                ] = he_meta.get(
                    "paste_y1",
                    "",
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

                ego_event_projection = route_projector.project(
                    x=float(ego_loc.x),
                    y=float(ego_loc.y),
                    previous_segment_idx=event_route_segment,
                )
                event_route_segment = ego_event_projection.segment_idx
                next_scenario_frame = scenario_frame + 1
                actor_source_frame = min(
                    end_frame,
                    event_clock.observe_for_next_frame(
                        next_execution_frame=next_scenario_frame,
                        ego_route_progress_m=ego_event_projection.s_m,
                    ),
                )

                if (
                    traffic_light_executor
                    is not None
                ):

                    next_t_s = (
                        float(
                            scenario_frame
                            + 1
                        )
                        /
                        float(
                            fps
                        )
                    )

                    traffic_light_executor.apply(
                        next_t_s
                    )
                if args.condition == "carla":

                    sync_carla_scenario_actors(
                        world=world,
                        carla_map=carla_map,
                        actor_by_id=
                            scenario_carla_actors,
                        active_states=
                            runtime.active_actors(
                                actor_source_frame
                            ),
                    )
                current_frame = (
                    world.tick()
                )

                sensor_frame = (
                    adapter.read_sensor_frame(
                        current_frame
                    )
                )

                scene_depth_by_camera = {}
                if args.condition == "he":
                    for camera_name, depth_queue in scene_depth_queues.items():
                        depth_image = read_exact_sensor_frame(
                            depth_queue,
                            current_frame,
                            f"HE {camera_name} depth",
                        )
                        scene_depth_by_camera[camera_name] = (
                            carla_depth_image_to_m(depth_image)
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
        if video_writer is not None:

            video_writer.release()
        if mosaic_writer is not None:

            mosaic_writer.release()
        for actor in (
            scenario_carla_actors.values()
        ):
            try:
                actor.destroy()
            except Exception:
                pass
        if (
            traffic_light_executor
            is not None

        ):

            try:

                traffic_light_executor.restore()

            except Exception as exc:

                print(
                    "[cleanup warning] "
                    "traffic-light restore:",
                    exc,
                )
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
