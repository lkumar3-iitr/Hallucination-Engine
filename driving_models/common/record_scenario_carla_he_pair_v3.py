#!/usr/bin/env python3
r"""
record_scenario_carla_he_pair_v3.py

Deterministic, separate-run CARLA-vs-HE rendering recorder.

This is a renderer calibration tool, NOT a closed-loop driving benchmark.

The workflow is intentionally two independent runs:

    1) --condition carla
       - physical CARLA scenario adversary is spawned
       - ego follows a deterministic scripted trajectory
       - the exact ego trajectory + execution origin are saved
       - RGB video + target instance mask + geometry metadata are recorded

    2) --condition he --reference-dir <carla-run>
       - NO physical scenario adversary is spawned
       - the exact ego trajectory and execution origin from the CARLA run
         are replayed
       - HE is rendered into the same native camera
       - RGB video + exact final HE alpha mask + HE metadata are recorded

The two conditions are NEVER mixed in the same rendered frame.

Why the scripted ego?
---------------------
The S2 re-entry scenario was authored around a 4 m/s ego:
the slow adversary moves aside, stops, lets ego pass, accelerates,
overtakes, cuts in, then hard-stops.  A deterministic 4 m/s ego
therefore reproduces the intended interaction while guaranteeing that
the CARLA and HE camera trajectories are identical.

Apples-to-apples masks
----------------------
CARLA:
    visible target pixels are isolated using synchronized instance-
    segmentation frames with the target present vs temporarily hidden.

HE:
    the exact final alpha field is reconstructed by rendering the same
    HE frame over black and white backgrounds:
        out_white - out_black = (1-alpha)*255
    This avoids RGB-background differencing and correctly handles black
    sprite pixels/windows.

The output is intended to be consumed by:
    compare_scenario_carla_he_pair_v1.py

Run from:
    D:\HallucinationEngine
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import carla


THIS_FILE = Path(__file__).resolve()
HE_ROOT = THIS_FILE.parents[2]
COMMON_DIR = HE_ROOT / "driving_models" / "common"

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from carla_ego_initialization import canonicalize_ego_start
from he_asset_registry_v1 import DEFAULT_MANIFEST
from resolved_scenario_runtime_v1 import (
    ResolvedScenarioRuntime,
)
from scenario_execution_runtime_v1 import (
    ExecutionWorldOrigin,
    load_execution_runtime,
    sg_position_to_world,
    sg_yaw_to_world,
)
from carla_actor_realizer_v1 import (
    CarlaActorRealizer,
    resolve_actor_ground_z,
    execution_state_to_carla_transform,
)
from he_multi_actor_compositor_v1 import HEMultiActorCompositorV1
from he_multi_actor_compositor_v2 import HEMultiActorCompositorV2


# ============================================================
# Arguments
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Record separate CARLA-reference and HE-replay runs "
            "for frame-accurate renderer comparison."
        )
    )

    parser.add_argument(
        "--condition",
        choices=["carla", "he"],
        required=True,
    )

    parser.add_argument(
        "--resolved",
        required=True,
        help="Resolved ScenarioGenerator v2 JSON.",
    )

    parser.add_argument(
        "--asset-root",
        required=True,
        help="Production HE asset root.",
    )

    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--reference-dir",
        default=None,
        help=(
            "Required for --condition he. Directory produced by "
            "the CARLA reference run."
        ),
    )

    parser.add_argument(
        "--actor-id",
        default="adv_reentry",
        help="Target scenario actor whose visible mask is recorded.",
    )

    # CARLA reference-run setup.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--spawn-index", type=int, default=10)
    parser.add_argument(
        "--ego-blueprint",
        default="vehicle.tesla.model3",
    )
    parser.add_argument(
        "--ego-speed-mps",
        type=float,
        default=None,
        help=(
            "Deprecated in V2 and ignored. The exact resolved "
            "ScenarioGenerator ego_frames trajectory is authoritative."
        ),
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
        "--max-carla-pose-retries",
        type=int,
        default=4,
        help=(
            "Maximum attempts for one CARLA reference frame when the "
            "tick snapshot actor pose does not equal the resolved pose."
        ),
    )
    parser.add_argument(
        "--carla-actor-position-tolerance-m",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--carla-actor-angle-tolerance-deg",
        type=float,
        default=0.01,
    )

    # Default = TCP native camera.
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument("--camera-x", type=float, default=-1.5)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=2.0)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)

    # HE runtime settings.
    parser.add_argument(
        "--distance-selection-mode",
        choices=["linear", "log", "inverse_depth"],
        default="linear",
    )
    parser.add_argument(
        "--he-renderer-version",
        choices=["v1", "v2"],
        default="v1",
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
    )
    parser.add_argument(
        "--he-warp-scale-mode",
        choices=[
            "independent",
            "uniform_height_preserve_aspect",
        ],
        default="independent",
    )
    parser.add_argument(
        "--he-viewpoint-lateral-sign",
        type=float,
        choices=[
            -1.0,
            1.0,
        ],
        default=1.0,
    )
    parser.add_argument(
        "--he-mask-alpha-threshold",
        type=int,
        default=1,
        help=(
            "Final reconstructed HE alpha threshold [0..255]. "
            "1 matches the renderer's final non-zero-alpha visibility "
            "semantics closely."
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
    )

    return parser


# ============================================================
# General helpers
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            block = fp.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                )
                + "\n"
            )


def append_jsonl(fp, row):
    fp.write(
        json.dumps(
            row,
            separators=(",", ":"),
        )
        + "\n"
    )
    fp.flush()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def transform_to_dict(tf):
    return {
        "x": float(tf.location.x),
        "y": float(tf.location.y),
        "z": float(tf.location.z),
        "pitch": float(tf.rotation.pitch),
        "yaw": float(tf.rotation.yaw),
        "roll": float(tf.rotation.roll),
    }


def dict_to_transform(data):
    return carla.Transform(
        carla.Location(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
        ),
        carla.Rotation(
            pitch=float(data.get("pitch", 0.0)),
            yaw=float(data.get("yaw", 0.0)),
            roll=float(data.get("roll", 0.0)),
        ),
    )


def normalize_angle_360(value):
    return float(value) % 360.0


def circular_abs_delta_deg(a, b):
    return abs(
        (
            (
                float(a)
                -
                float(b)
                +
                180.0
            )
            %
            360.0
        )
        -
        180.0
    )


def transform_position_error_m(a, b):
    return math.sqrt(
        (
            float(a.location.x)
            -
            float(b.location.x)
        ) ** 2
        +
        (
            float(a.location.y)
            -
            float(b.location.y)
        ) ** 2
        +
        (
            float(a.location.z)
            -
            float(b.location.z)
        ) ** 2
    )


def transform_angle_error_deg(a, b):
    return max(
        circular_abs_delta_deg(
            a.rotation.pitch,
            b.rotation.pitch,
        ),
        circular_abs_delta_deg(
            a.rotation.yaw,
            b.rotation.yaw,
        ),
        circular_abs_delta_deg(
            a.rotation.roll,
            b.rotation.roll,
        ),
    )


def geometric_view_angle_deg(actor_tf, camera_tf):
    """
    Production asset convention, computed independently from the HE selector:

        0 deg   = camera is in front of actor
        180 deg = camera is behind actor

    Compute the CAMERA POSITION expressed in ACTOR-LOCAL coordinates.
    The azimuth of that vector is exactly the geometric viewpoint around
    the physical CARLA actor.
    """
    world_to_actor = np.asarray(
        actor_tf.get_inverse_matrix(),
        dtype=np.float64,
    )

    camera_world = np.array(
        [
            float(camera_tf.location.x),
            float(camera_tf.location.y),
            float(camera_tf.location.z),
            1.0,
        ],
        dtype=np.float64,
    )

    camera_local = world_to_actor @ camera_world

    angle = math.degrees(
        math.atan2(
            float(camera_local[1]),
            float(camera_local[0]),
        )
    )

    return normalize_angle_360(angle)


def camera_relative_metrics(actor_tf, camera_tf):
    world_to_camera = np.asarray(
        camera_tf.get_inverse_matrix(),
        dtype=np.float64,
    )

    actor_world = np.array(
        [
            float(actor_tf.location.x),
            float(actor_tf.location.y),
            float(actor_tf.location.z),
            1.0,
        ],
        dtype=np.float64,
    )

    pc = world_to_camera @ actor_world

    forward = float(pc[0])
    right = float(pc[1])
    up = float(pc[2])

    return {
        "camera_forward_m": forward,
        "camera_right_m": right,
        "camera_up_m": up,
        "camera_distance_m": float(
            math.sqrt(
                forward * forward
                + right * right
                + up * up
            )
        ),
    }


# ============================================================
# CARLA helpers
# ============================================================

def clear_dynamic_actors(world):
    """
    Start from a clean deterministic scene.

    Call BEFORE spawning our ego/cameras.
    """
    actors = []

    for pattern in (
        "vehicle.*",
        "walker.*",
        "sensor.*",
    ):
        actors.extend(
            list(
                world.get_actors().filter(pattern)
            )
        )

    for actor in actors:
        try:
            if actor.type_id.startswith("sensor."):
                actor.stop()
        except Exception:
            pass

        try:
            actor.destroy()
        except Exception:
            pass


def get_sensor_frame(
    sensor_queue,
    target_frame,
    label,
    timeout_s=10.0,
):
    while True:
        try:
            sample = sensor_queue.get(
                timeout=float(timeout_s)
            )
        except queue.Empty as exc:
            raise RuntimeError(
                f"Timeout waiting for {label} frame {target_frame}"
            ) from exc

        frame = int(sample.frame)

        if frame < int(target_frame):
            continue

        if frame > int(target_frame):
            raise RuntimeError(
                f"{label}: expected CARLA frame "
                f"{target_frame}, got {frame}"
            )

        return sample


def carla_rgb_to_array(image):
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    ).reshape(
        (
            int(image.height),
            int(image.width),
            4,
        )
    )

    return (
        array[:, :, :3][:, :, ::-1]
        .copy()
    )


def raw_instance_code(image):
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    ).reshape(
        (
            int(image.height),
            int(image.width),
            4,
        )
    )

    # CARLA 0.9.15 instance segmentation carries instance identity
    # in raw B/G. We compare the pair directly rather than assuming
    # it equals actor.id.
    return array[:, :, :2].copy()


def largest_component(mask):
    binary = mask.astype(np.uint8)

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if count <= 1:
        return np.zeros_like(mask, dtype=bool)

    areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    best = 1 + int(np.argmax(areas))

    return labels == best


def target_mask_from_instance_pair(
    physical_instance,
    hidden_instance,
):
    physical = raw_instance_code(
        physical_instance
    )

    hidden = raw_instance_code(
        hidden_instance
    )

    changed = (
        (physical[:, :, 0] != hidden[:, :, 0])
        |
        (physical[:, :, 1] != hidden[:, :, 1])
    )

    return largest_component(changed)


def discover_target_instance_key(
    physical_instance,
    hidden_instance,
):
    """
    Discover the target actor's stable CARLA instance-segmentation key.

    CARLA instance segmentation stores unique instance identity in the
    raw B/G byte pair.  We use ONE calibration present/hidden pair before
    recording starts, then use that stable key directly for all frames.

    This is deliberately not done every frame: an extra hidden world.tick()
    per recorded frame would change simulation cadence and can create an
    exact one-frame camera offset between the CARLA and HE recordings.
    """
    changed_component = (
        target_mask_from_instance_pair(
            physical_instance=
                physical_instance,
            hidden_instance=
                hidden_instance,
        )
    )

    if not np.any(
        changed_component
    ):
        raise RuntimeError(
            "Could not discover target instance: "
            "present/hidden calibration mask is empty."
        )

    codes = raw_instance_code(
        physical_instance
    )[
        changed_component
    ]

    if codes.size == 0:
        raise RuntimeError(
            "Could not discover target instance key."
        )

    flat = codes.reshape(
        (-1, 2)
    )

    unique, counts = np.unique(
        flat,
        axis=0,
        return_counts=True,
    )

    # Actor instance ID should not be the background [0, 0] pair.
    nonzero = np.any(
        unique != 0,
        axis=1,
    )

    if np.any(nonzero):
        unique = unique[nonzero]
        counts = counts[nonzero]

    best_index = int(
        np.argmax(counts)
    )

    key = (
        int(unique[best_index, 0]),
        int(unique[best_index, 1]),
    )

    support_pixels = int(
        counts[best_index]
    )

    changed_pixels = int(
        np.count_nonzero(
            changed_component
        )
    )

    support_fraction = (
        float(support_pixels)
        /
        float(max(1, changed_pixels))
    )

    return (
        key,
        support_fraction,
        changed_component,
    )


def mask_from_instance_key(
    instance_image,
    instance_key,
):
    codes = raw_instance_code(
        instance_image
    )

    b_value = int(
        instance_key[0]
    )

    g_value = int(
        instance_key[1]
    )

    return (
        (codes[:, :, 0] == b_value)
        &
        (codes[:, :, 1] == g_value)
    )


def measurement_from_mask(mask):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width_px": int(x2 - x1 + 1),
        "height_px": int(y2 - y1 + 1),
        "center_x_px": float((x1 + x2) / 2.0),
        "center_y_px": float((y1 + y2) / 2.0),
        "bottom_y_px": float(y2),
        "area_px": int(np.count_nonzero(mask)),
    }


def make_camera(
    world,
    ego,
    blueprint_id,
    camera_cfg,
):
    bp = (
        world
        .get_blueprint_library()
        .find(blueprint_id)
    )

    bp.set_attribute(
        "image_size_x",
        str(int(camera_cfg["width"])),
    )

    bp.set_attribute(
        "image_size_y",
        str(int(camera_cfg["height"])),
    )

    bp.set_attribute(
        "fov",
        str(float(camera_cfg["fov_deg"])),
    )

    bp.set_attribute(
        "sensor_tick",
        "0.0",
    )

    relative_tf = carla.Transform(
        carla.Location(
            x=float(camera_cfg["x_m"]),
            y=float(camera_cfg["y_m"]),
            z=float(camera_cfg["z_m"]),
        ),
        carla.Rotation(
            pitch=float(camera_cfg["pitch_deg"]),
            yaw=float(camera_cfg["yaw_deg"]),
            roll=float(camera_cfg["roll_deg"]),
        ),
    )

    return world.spawn_actor(
        bp,
        relative_tf,
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid,
    )


def build_execution_origin(
    carla_map,
    ego0_tf,
):
    waypoint = carla_map.get_waypoint(
        ego0_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if waypoint is None:
        raise RuntimeError(
            "Could not resolve driving waypoint "
            "for canonical ego start."
        )

    return ExecutionWorldOrigin(
        x_m=float(ego0_tf.location.x),
        y_m=float(ego0_tf.location.y),
        z_m=float(
            waypoint.transform.location.z
        ),
        yaw_deg=float(ego0_tf.rotation.yaw),
    )


def generate_resolved_ego_trajectory(
    carla_map,
    ego0_tf,
    origin,
    resolved_runtime,
    frame_count,
):
    """
    Convert the exact resolved ScenarioGenerator ego_frames into CARLA
    world transforms.

    This avoids inventing a second ego trajectory inside the calibration
    recorder.  The same resolved file therefore defines BOTH:
        - adversary truth
        - scripted ego truth

    Only Z is grounded to the CARLA road surface; resolved x/y/yaw are
    preserved.
    """
    initial_road_z = float(
        origin.z_m
    )

    ego_clearance_m = (
        float(ego0_tf.location.z)
        -
        initial_road_z
    )

    rows = []

    for frame_idx in range(
        int(frame_count)
    ):
        state = (
            resolved_runtime.ego_state(
                frame_idx
            )
        )

        if state is None:
            raise RuntimeError(
                f"Resolved scenario has no ego state "
                f"for frame {frame_idx}."
            )

        (
            world_x,
            world_y,
        ) = sg_position_to_world(
            origin=origin,
            sg_x_m=float(state.x_m),
            sg_y_m=float(state.y_m),
        )

        world_yaw = sg_yaw_to_world(
            origin=origin,
            sg_yaw_deg=float(
                state.yaw_deg
            ),
        )

        seed = carla.Location(
            x=float(world_x),
            y=float(world_y),
            z=float(origin.z_m),
        )

        waypoint = carla_map.get_waypoint(
            seed,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is not None:
            road_z = float(
                waypoint.transform.location.z
            )
        else:
            road_z = float(
                origin.z_m
            )

        rows.append({
            "scenario_frame":
                int(frame_idx),
            "t_s":
                float(state.t_s),
            "speed_mps":
                float(state.speed_mps),
            "resolved_sg": {
                "x_m": float(state.x_m),
                "y_m": float(state.y_m),
                "yaw_deg": float(
                    state.yaw_deg
                ),
            },
            "transform": {
                "x": float(world_x),
                "y": float(world_y),
                "z": float(
                    road_z
                    +
                    ego_clearance_m
                ),
                "pitch": 0.0,
                "yaw": float(world_yaw),
                "roll": 0.0,
            },
        })

    return rows, ego_clearance_m


def hide_actor(actor):
    if actor is None:
        return

    tf = actor.get_transform()

    # Walkers may be constrained back toward navigation geometry when moved
    # only below the map. Park every actor far outside the calibrated camera
    # frustum, matching the proven sprite-bank capture procedure.
    tf.location.x += 500.0
    tf.location.y += 500.0
    tf.location.z += 200.0

    actor.set_transform(tf)


# ============================================================
# HE helpers
# ============================================================

def ground_execution_states(
    world,
    runtime,
    states,
):
    """
    Give HE the SAME actor-origin Z used by the physical CARLA realizer.

    This avoids an apples-vs-oranges comparison where physical CARLA
    receives road grounding / z_offset but HE uses raw ScenarioGenerator Z.
    """
    grounded = []

    for state in states:
        z_m = resolve_actor_ground_z(
            world=world,
            actor_state=state,
            execution=runtime,
        )

        grounded.append(
            replace(
                state,
                world_z_m=float(z_m),
            )
        )

    return grounded


def reconstruct_he_alpha_u8(
    compositor,
    camera_tf,
    active_actors,
    width,
    height,
    fov,
):
    black = np.zeros(
        (int(height), int(width), 3),
        dtype=np.uint8,
    )

    white = np.full(
        (int(height), int(width), 3),
        255,
        dtype=np.uint8,
    )

    black_result = compositor.render(
        base_rgb=black,
        camera_tf=camera_tf,
        active_actors=active_actors,
        width=int(width),
        height=int(height),
        fov=float(fov),
    )

    white_result = compositor.render(
        base_rgb=white,
        camera_tf=camera_tf,
        active_actors=active_actors,
        width=int(width),
        height=int(height),
        fov=float(fov),
    )

    # white - black = (1-alpha)*255.
    diff = (
        white_result.rgb.astype(np.float32)
        -
        black_result.rgb.astype(np.float32)
    )

    background_fraction = np.mean(
        diff,
        axis=2,
    ) / 255.0

    alpha = np.clip(
        1.0 - background_fraction,
        0.0,
        1.0,
    )

    alpha_u8 = np.rint(
        alpha * 255.0
    ).astype(np.uint8)

    return alpha_u8


def actor_result_for_id(
    composite,
    actor_id,
):
    for row in composite.actor_results:
        if str(row.actor_id) == str(actor_id):
            return row
    return None


# ============================================================
# Video
# ============================================================

def open_video_writer(
    path,
    fps,
    width,
    height,
):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
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


# ============================================================
# Reference setup
# ============================================================

def camera_config_from_args(args):
    return {
        "width": int(args.width),
        "height": int(args.height),
        "fov_deg": float(args.fov),
        "x_m": float(args.camera_x),
        "y_m": float(args.camera_y),
        "z_m": float(args.camera_z),
        "pitch_deg": float(args.camera_pitch),
        "yaw_deg": float(args.camera_yaw),
        "roll_deg": float(args.camera_roll),
    }


def origin_to_dict(origin):
    return {
        "x_m": float(origin.x_m),
        "y_m": float(origin.y_m),
        "z_m": float(origin.z_m),
        "yaw_deg": float(origin.yaw_deg),
    }


def dict_to_origin(data):
    return ExecutionWorldOrigin(
        x_m=float(data["x_m"]),
        y_m=float(data["y_m"]),
        z_m=float(data["z_m"]),
        yaw_deg=float(data["yaw_deg"]),
    )


# ============================================================
# Main
# ============================================================

def main():
    args = build_parser().parse_args()

    resolved_path = Path(
        args.resolved
    ).resolve()

    asset_root = Path(
        args.asset_root
    ).resolve()

    manifest_path = Path(
        args.manifest
    ).resolve()

    output_dir = ensure_dir(
        Path(args.output_dir).resolve()
    )

    mask_dir = ensure_dir(
        output_dir / "masks"
    )

    frames_path = (
        output_dir
        /
        "frames.jsonl"
    )

    setup_path = (
        output_dir
        /
        "setup.json"
    )

    trajectory_path = (
        output_dir
        /
        "ego_trajectory.jsonl"
    )

    resolved_sha256 = sha256_file(
        resolved_path
    )

    # Read scenario FPS/duration without requiring an execution origin yet.
    with open(
        resolved_path,
        "r",
        encoding="utf-8",
    ) as fp:
        raw_resolved = json.load(fp)

    resolved_runtime_source = (
        ResolvedScenarioRuntime.from_json(
            resolved_path
        )
    )

    scenario_fps = float(
        resolved_runtime_source.fps
    )

    duration_s = float(
        resolved_runtime_source.duration_s
    )

    frame_count = (
        int(
            round(
                duration_s
                *
                scenario_fps
            )
        )
        +
        1
    )

    if int(args.max_frames) > 0:
        frame_count = min(
            frame_count,
            int(args.max_frames),
        )

    if (
        args.condition == "he"
        and
        args.reference_dir is None
    ):
        raise ValueError(
            "--reference-dir is required "
            "for --condition he"
        )

    reference_setup = None
    ego_trajectory = None

    if args.condition == "he":
        reference_dir = Path(
            args.reference_dir
        ).resolve()

        with open(
            reference_dir / "setup.json",
            "r",
            encoding="utf-8",
        ) as fp:
            reference_setup = json.load(fp)

        if (
            reference_setup.get(
                "condition"
            )
            !=
            "carla"
        ):
            raise RuntimeError(
                "Reference setup is not a CARLA reference run."
            )

        if (
            str(
                reference_setup[
                    "resolved_sha256"
                ]
            )
            !=
            str(resolved_sha256)
        ):
            raise RuntimeError(
                "Resolved-scenario SHA256 differs "
                "from CARLA reference run."
            )

        ego_trajectory = load_jsonl(
            reference_dir
            /
            "ego_trajectory.jsonl"
        )

        frame_count = min(
            frame_count,
            len(ego_trajectory),
        )

        # Exact camera/world setup comes from the CARLA reference.
        town = str(
            reference_setup["town"]
        )

        spawn_index = int(
            reference_setup[
                "spawn_index"
            ]
        )

        ego_blueprint = str(
            reference_setup[
                "ego_blueprint"
            ]
        )

        camera_cfg = dict(
            reference_setup[
                "camera"
            ]
        )

        scenario_fps = float(
            reference_setup[
                "fps"
            ]
        )

        origin = dict_to_origin(
            reference_setup[
                "execution_origin"
            ]
        )

    else:
        town = str(args.town)
        spawn_index = int(
            args.spawn_index
        )
        ego_blueprint = str(
            args.ego_blueprint
        )
        camera_cfg = camera_config_from_args(
            args
        )
        origin = None

    print()
    print("=" * 96)
    print("SCENARIO CARLA <-> HE PAIR RECORDER V3")
    print("=" * 96)
    print("[condition]       ", args.condition.upper())
    print("[resolved]        ", resolved_path)
    print("[resolved sha256] ", resolved_sha256)
    print("[actor id]        ", args.actor_id)
    print("[frames]          ", frame_count)
    print("[fps]             ", scenario_fps)
    print("[town]            ", town)
    print("[spawn index]     ", spawn_index)
    print("[camera]          ", camera_cfg)
    if args.condition == "he":
        print("[reference]       ", Path(args.reference_dir).resolve())
    print("=" * 96)
    print()

    client = carla.Client(
        args.host,
        int(args.port),
    )

    client.set_timeout(
        float(args.timeout)
    )

    world = client.load_world(
        town
    )

    original_settings = (
        world.get_settings()
    )

    ego = None
    rgb_camera = None
    instance_camera = None
    realizer = None
    video_writer = None
    background_writer = None
    frames_fp = None

    try:
        clear_dynamic_actors(
            world
        )

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            1.0
            /
            float(scenario_fps)
        )

        world.apply_settings(
            settings
        )

        carla_map = (
            world.get_map()
        )

        spawn_points = (
            carla_map.get_spawn_points()
        )

        if not spawn_points:
            raise RuntimeError(
                "CARLA map has no spawn points."
            )

        spawn_idx = (
            int(spawn_index)
            %
            len(spawn_points)
        )

        nominal_spawn_tf = (
            spawn_points[
                spawn_idx
            ]
        )

        ego_bp = (
            world
            .get_blueprint_library()
            .find(
                ego_blueprint
            )
        )

        if ego_bp.has_attribute(
            "role_name"
        ):
            ego_bp.set_attribute(
                "role_name",
                "pair_reference_ego",
            )

        ego = world.try_spawn_actor(
            ego_bp,
            nominal_spawn_tf,
        )

        if ego is None:
            raise RuntimeError(
                "Could not spawn ego."
            )

        if args.condition == "carla":
            (
                ego0_tf,
                _ego0_state,
            ) = canonicalize_ego_start(
                world=world,
                ego=ego,
                nominal_spawn_tf=
                    nominal_spawn_tf,
                settle_ticks=
                    int(
                        args.physics_settle_ticks
                    ),
                hold_ticks=
                    int(
                        args.canonical_hold_ticks
                    ),
            )

            origin = build_execution_origin(
                carla_map=carla_map,
                ego0_tf=ego0_tf,
            )

            ego_trajectory, ego_clearance_m = (
                generate_resolved_ego_trajectory(
                    carla_map=carla_map,
                    ego0_tf=ego0_tf,
                    origin=origin,
                    resolved_runtime=
                        resolved_runtime_source,
                    frame_count=frame_count,
                )
            )

            write_jsonl(
                trajectory_path,
                ego_trajectory,
            )

        else:
            # No local canonicalization is used to define the comparison.
            # The CARLA reference trajectory is authoritative.
            ego0_tf = dict_to_transform(
                ego_trajectory[0][
                    "transform"
                ]
            )

            ego_clearance_m = float(
                reference_setup.get(
                    "ego_clearance_m",
                    0.0,
                )
            )

        # From this point onward ego is purely kinematic.
        try:
            ego.set_simulate_physics(
                False
            )
        except Exception:
            pass

        ego.set_transform(
            dict_to_transform(
                ego_trajectory[0][
                    "transform"
                ]
            )
        )

        # Runtime uses the SAME saved origin in both conditions.
        runtime = load_execution_runtime(
            resolved_json=str(
                resolved_path
            ),
            asset_root=asset_root,
            origin=origin,
            manifest_path=manifest_path,
        )

        runtime_summary = (
            runtime.summary()
        )

        if (
            abs(
                float(
                    runtime_summary["fps"]
                )
                -
                float(scenario_fps)
            )
            >
            1e-6
        ):
            raise RuntimeError(
                "Runtime FPS differs from pair setup."
            )

        # Cameras.
        rgb_camera = make_camera(
            world=world,
            ego=ego,
            blueprint_id="sensor.camera.rgb",
            camera_cfg=camera_cfg,
        )

        q_rgb = queue.Queue()

        rgb_camera.listen(
            q_rgb.put
        )

        q_inst = None

        if args.condition == "carla":
            instance_camera = make_camera(
                world=world,
                ego=ego,
                blueprint_id=(
                    "sensor.camera.instance_segmentation"
                ),
                camera_cfg=camera_cfg,
            )

            q_inst = queue.Queue()

            instance_camera.listen(
                q_inst.put
            )

            realizer = CarlaActorRealizer(
                world=world,
                execution=runtime,
            )

        compositor_class = (
            HEMultiActorCompositorV2
            if args.he_renderer_version == "v2"
            else HEMultiActorCompositorV1
        )
        compositor = compositor_class(
            distance_selection_mode=
                args.distance_selection_mode,
            bottom_y_offset_px=
                float(
                    args.he_bottom_y_offset_px
                ),
            silhouette_scale=
                float(
                    args.he_silhouette_scale
                ),
            warp_scale_mode=
                str(
                    args.he_warp_scale_mode
                ),
            viewpoint_lateral_sign=
                float(
                    args.he_viewpoint_lateral_sign
                ),
        )

        video_path = (
            output_dir
            /
            (
                "carla_reference.mp4"
                if args.condition == "carla"
                else "he_replay.mp4"
            )
        )

        video_writer = open_video_writer(
            video_path,
            scenario_fps,
            camera_cfg["width"],
            camera_cfg["height"],
        )
        background_video_path = None

        if args.condition == "he":
            background_video_path = (
                output_dir
                / "background_only.mp4"
            )

            background_writer = open_video_writer(
                background_video_path,
                scenario_fps,
                camera_cfg["width"],
                camera_cfg["height"],
            )
        frames_fp = open(
            frames_path,
            "w",
            encoding="utf-8",
        )

        # Setup is written before the first scenario frame.
        setup = {
            "schema":
                "carla_he_pair_recording_v3",
            "condition":
                str(args.condition),
            "resolved_path":
                str(resolved_path),
            "resolved_sha256":
                str(resolved_sha256),
            "scenario_id":
                str(
                    runtime_summary[
                        "scenario_id"
                    ]
                ),
            "fps":
                float(scenario_fps),
            "frame_count":
                int(frame_count),
            "town":
                str(town),
            "spawn_index":
                int(spawn_idx),
            "ego_blueprint":
                str(ego_blueprint),
            "ego_source":
                "resolved_scenario_ego_frames",
            "ego_initial_speed_mps":
                float(
                    ego_trajectory[0][
                        "speed_mps"
                    ]
                ),
            "ego_clearance_m":
                float(ego_clearance_m),
            "execution_origin":
                origin_to_dict(origin),
            "camera":
                camera_cfg,
            "actor_id":
                str(args.actor_id),
            "mask_semantics": (
                "CARLA target pixels from one calibrated stable "
                "instance-segmentation B/G key"
                if args.condition == "carla"
                else (
                    "HE final reconstructed alpha mask; "
                    f"alpha_u8 >= "
                    f"{int(args.he_mask_alpha_threshold)}"
                )
            ),
            "reference_dir": (
                str(
                    Path(
                        args.reference_dir
                    ).resolve()
                )
                if args.condition == "he"
                else None
            ),
            "distance_selection_mode":
                str(
                    args.distance_selection_mode
                ),
            "he_bottom_y_offset_px":
                float(
                    args.he_bottom_y_offset_px
                ),
            "he_silhouette_scale":
                float(
                    args.he_silhouette_scale
                ),
            "he_warp_scale_mode":
                str(
                    args.he_warp_scale_mode
                ),
            "he_viewpoint_lateral_sign":
                float(
                    args.he_viewpoint_lateral_sign
                ),
        }

        # ----------------------------------------------------
        # One-time CARLA target instance-ID calibration.
        #
        # IMPORTANT:
        # This is outside the recorded scenario sequence.
        # During the actual recording there is exactly ONE world.tick()
        # per scenario frame in BOTH CARLA and HE conditions.
        # ----------------------------------------------------
        target_instance_key = None
        target_instance_support_fraction = None

        if args.condition == "carla":
            ego.set_transform(
                dict_to_transform(
                    ego_trajectory[0][
                        "transform"
                    ]
                )
            )

            realizer.apply_frame(0)

            calibration_actor = (
                realizer.get_actor(
                    args.actor_id
                )
            )

            if calibration_actor is None:
                raise RuntimeError(
                    f"Target actor {args.actor_id!r} "
                    "is not active at scenario frame 0; "
                    "instance-key discovery requires an active target."
                )

            # Instance-key discovery must also work when the authored actor
            # starts behind the tested camera. This pose is used only by the
            # two calibration ticks; the common warm-up below restores frame 0.
            calibration_pose_frame = world.tick()
            get_sensor_frame(
                q_rgb,
                calibration_pose_frame,
                "RGB instance calibration camera-pose warmup",
            )
            get_sensor_frame(
                q_inst,
                calibration_pose_frame,
                "instance calibration camera-pose warmup",
            )
            calibration_camera_tf = rgb_camera.get_transform()
            calibration_forward = calibration_camera_tf.get_forward_vector()
            authored_actor_tf = calibration_actor.get_transform()
            calibration_actor.set_transform(
                carla.Transform(
                    carla.Location(
                        x=float(calibration_camera_tf.location.x)
                        + 10.0 * float(calibration_forward.x),
                        y=float(calibration_camera_tf.location.y)
                        + 10.0 * float(calibration_forward.y),
                        z=float(authored_actor_tf.location.z),
                    ),
                    authored_actor_tf.rotation,
                )
            )

            # Match the proven native-bank capture sequence. Walker render
            # state can lag a transform by more than one synchronous tick.
            for _ in range(2):
                settle_frame = world.tick()
                get_sensor_frame(
                    q_rgb,
                    settle_frame,
                    "RGB instance calibration present settle",
                )
                get_sensor_frame(
                    q_inst,
                    settle_frame,
                    "instance calibration present settle",
                )

            calibration_frame = world.tick()

            _calibration_rgb = get_sensor_frame(
                q_rgb,
                calibration_frame,
                "RGB instance calibration present",
            )

            calibration_instance_present = (
                get_sensor_frame(
                    q_inst,
                    calibration_frame,
                    "instance calibration present",
                )
            )

            hide_actor(
                calibration_actor
            )

            for _ in range(2):
                settle_frame = world.tick()
                get_sensor_frame(
                    q_rgb,
                    settle_frame,
                    "RGB instance calibration hidden settle",
                )
                get_sensor_frame(
                    q_inst,
                    settle_frame,
                    "instance calibration hidden settle",
                )

            hidden_frame = world.tick()

            _hidden_rgb = get_sensor_frame(
                q_rgb,
                hidden_frame,
                "RGB instance calibration hidden",
            )

            calibration_instance_hidden = (
                get_sensor_frame(
                    q_inst,
                    hidden_frame,
                    "instance calibration hidden",
                )
            )

            (
                target_instance_key,
                target_instance_support_fraction,
                _calibration_changed_mask,
            ) = discover_target_instance_key(
                physical_instance=
                    calibration_instance_present,
                hidden_instance=
                    calibration_instance_hidden,
            )

            print(
                "[CARLA target instance key]",
                target_instance_key,
                "support_fraction=",
                f"{target_instance_support_fraction:.4f}",
            )

            setup[
                "target_instance_key_bg"
            ] = [
                int(target_instance_key[0]),
                int(target_instance_key[1]),
            ]

            setup[
                "target_instance_key_support_fraction"
            ] = float(
                target_instance_support_fraction
            )

        # ----------------------------------------------------
        # Common warmup AFTER calibration.
        #
        # Force frame-0 ego/actor truth on every warmup tick so the
        # attached camera has the same settled parent relationship before
        # the recorded sequence starts.
        # ----------------------------------------------------
        for _ in range(2):
            ego.set_transform(
                dict_to_transform(
                    ego_trajectory[0][
                        "transform"
                    ]
                )
            )

            if args.condition == "carla":
                realizer.apply_frame(0)

            warm_frame = world.tick()

            get_sensor_frame(
                q_rgb,
                warm_frame,
                "RGB warmup",
            )

            if q_inst is not None:
                get_sensor_frame(
                    q_inst,
                    warm_frame,
                    "instance warmup",
                )

        # Setup is written only after all reference calibration metadata
        # has been resolved.
        write_json(
            setup_path,
            setup,
        )

        print("[recording]")

        for scenario_frame in range(
            frame_count
        ):
            trajectory_row = (
                ego_trajectory[
                    scenario_frame
                ]
            )

            ego_tf = dict_to_transform(
                trajectory_row[
                    "transform"
                ]
            )

            ego.set_transform(
                ego_tf
            )

            target_state = None

            raw_active_states = list(
                runtime.active_actors(
                    scenario_frame
                )
            )

            for state in raw_active_states:
                if (
                    str(state.actor_id)
                    ==
                    str(args.actor_id)
                ):
                    target_state = state
                    break

            if args.condition == "carla":
                if target_state is not None:
                    expected_actor_tf = (
                        execution_state_to_carla_transform(
                            world=world,
                            actor_state=target_state,
                            execution=runtime,
                        )
                    )
                else:
                    expected_actor_tf = None

                accepted = False
                last_position_error = float("nan")
                last_angle_error = float("nan")
                record_attempts = 0

                for attempt in range(
                    1,
                    int(args.max_carla_pose_retries) + 1,
                ):
                    record_attempts = int(attempt)

                    # Re-assert the exact same scenario truth before
                    # every attempt. A rejected attempt never advances
                    # scenario_frame.
                    ego.set_transform(
                        ego_tf
                    )

                    realizer.apply_frame(
                        scenario_frame
                    )

                    # One tick produces both the authoritative world
                    # snapshot and the sensor samples for this attempt.
                    carla_frame = world.tick()

                    physical_rgb_image = (
                        get_sensor_frame(
                            q_rgb,
                            carla_frame,
                            "RGB physical",
                        )
                    )

                    physical_inst_image = (
                        get_sensor_frame(
                            q_inst,
                            carla_frame,
                            "instance physical",
                        )
                    )

                    snapshot = (
                        world.get_snapshot()
                    )

                    if (
                        int(snapshot.frame)
                        !=
                        int(carla_frame)
                    ):
                        raise RuntimeError(
                            "CARLA snapshot/sensor frame mismatch: "
                            f"snapshot={snapshot.frame}, "
                            f"sensor={carla_frame}"
                        )

                    output_rgb = carla_rgb_to_array(
                        physical_rgb_image
                    )

                    # SensorData.transform is the camera pose for this
                    # exact image.
                    camera_tf = (
                        physical_rgb_image.transform
                    )

                    actor = realizer.get_actor(
                        args.actor_id
                    )

                    actor_tf = None

                    if actor is not None:
                        actor_snapshot = (
                            snapshot.find(
                                actor.id
                            )
                        )

                        if actor_snapshot is not None:
                            actor_tf = (
                                actor_snapshot
                                .get_transform()
                            )

                    if expected_actor_tf is None:
                        accepted = (
                            actor_tf is None
                        )

                    elif actor_tf is None:
                        accepted = False

                    else:
                        last_position_error = (
                            transform_position_error_m(
                                actor_tf,
                                expected_actor_tf,
                            )
                        )

                        last_angle_error = (
                            transform_angle_error_deg(
                                actor_tf,
                                expected_actor_tf,
                            )
                        )

                        accepted = (
                            last_position_error
                            <=
                            float(
                                args.carla_actor_position_tolerance_m
                            )
                            and
                            last_angle_error
                            <=
                            float(
                                args.carla_actor_angle_tolerance_deg
                            )
                        )

                    if accepted:
                        break

                    print(
                        f"[CARLA pose retry] "
                        f"scenario_frame={scenario_frame} "
                        f"attempt={attempt} "
                        f"pos_err={last_position_error:.6f}m "
                        f"angle_err={last_angle_error:.6f}deg"
                    )

                if not accepted:
                    raise RuntimeError(
                        f"CARLA could not realize resolved actor pose "
                        f"for scenario_frame={scenario_frame} after "
                        f"{record_attempts} attempts. "
                        f"last_position_error="
                        f"{last_position_error:.6f}m, "
                        f"last_angle_error="
                        f"{last_angle_error:.6f}deg"
                    )

                if (
                    actor_tf is not None
                    and
                    target_instance_key
                    is not None
                ):
                    target_mask = (
                        mask_from_instance_key(
                            physical_inst_image,
                            target_instance_key,
                        )
                    )
                else:
                    target_mask = np.zeros(
                        (
                            int(
                                camera_cfg[
                                    "height"
                                ]
                            ),
                            int(
                                camera_cfg[
                                    "width"
                                ]
                            ),
                        ),
                        dtype=bool,
                    )

                he_meta = None
                selected_angle = None
                requested_angle = None
                selected_distance = None
                selected_elevation = None
                he_rendered = None

            else:
                record_attempts = 1
                carla_frame = world.tick()

                base_rgb_image = (
                    get_sensor_frame(
                        q_rgb,
                        carla_frame,
                        "RGB HE background",
                    )
                )

                base_rgb = carla_rgb_to_array(
                    base_rgb_image
                )
                # Save the exact adversary-free CARLA background
                # BEFORE any HE actor is composited.
                if background_writer is not None:
                    background_writer.write(
                        cv2.cvtColor(
                            base_rgb,
                            cv2.COLOR_RGB2BGR,
                        )
                    )
                # Use the transform attached to this exact rendered
                # sensor sample, matching the CARLA-reference semantics.
                camera_tf = (
                    base_rgb_image.transform
                )

                grounded_states = (
                    ground_execution_states(
                        world=world,
                        runtime=runtime,
                        states=raw_active_states,
                    )
                )

                composite = (
                    compositor.render(
                        base_rgb=base_rgb,
                        camera_tf=camera_tf,
                        active_actors=
                            grounded_states,
                        width=
                            int(
                                camera_cfg[
                                    "width"
                                ]
                            ),
                        height=
                            int(
                                camera_cfg[
                                    "height"
                                ]
                            ),
                        fov=
                            float(
                                camera_cfg[
                                    "fov_deg"
                                ]
                            ),
                    )
                )

                output_rgb = (
                    composite.rgb
                )

                alpha_u8 = (
                    reconstruct_he_alpha_u8(
                        compositor=
                            compositor,
                        camera_tf=
                            camera_tf,
                        active_actors=
                            grounded_states,
                        width=
                            int(
                                camera_cfg[
                                    "width"
                                ]
                            ),
                        height=
                            int(
                                camera_cfg[
                                    "height"
                                ]
                            ),
                        fov=
                            float(
                                camera_cfg[
                                    "fov_deg"
                                ]
                            ),
                    )
                )

                target_mask = (
                    alpha_u8
                    >=
                    int(
                        args.he_mask_alpha_threshold
                    )
                )

                actor_tf = None

                if target_state is not None:
                    actor_tf = (
                        execution_state_to_carla_transform(
                            world=world,
                            actor_state=
                                target_state,
                            execution=runtime,
                        )
                    )

                actor_result = (
                    actor_result_for_id(
                        composite,
                        args.actor_id,
                    )
                )

                if actor_result is not None:
                    he_meta = dict(
                        actor_result.he_metadata
                    )

                    requested_angle = (
                        he_meta.get(
                            "viewpoint_angle_deg"
                        )
                    )

                    selected_angle = (
                        he_meta.get(
                            "selected_angle"
                        )
                    )

                    selected_distance = (
                        he_meta.get(
                            "selected_distance_m"
                        )
                    )

                    selected_elevation = (
                        he_meta.get(
                            "selected_elevation_deg"
                        )
                    )

                    he_rendered = bool(
                        actor_result.rendered
                    )

                else:
                    he_meta = None
                    requested_angle = None
                    selected_angle = None
                    selected_distance = None
                    selected_elevation = None
                    he_rendered = False

            mask_measurement = (
                measurement_from_mask(
                    target_mask
                )
            )

            mask_path = (
                mask_dir
                /
                (
                    f"frame_"
                    f"{scenario_frame:06d}.png"
                )
            )

            cv2.imwrite(
                str(mask_path),
                (
                    target_mask.astype(
                        np.uint8
                    )
                    *
                    255
                ),
            )

            video_writer.write(
                cv2.cvtColor(
                    output_rgb,
                    cv2.COLOR_RGB2BGR,
                )
            )

            geom_angle = None
            relative_metrics = None

            if actor_tf is not None:
                geom_angle = (
                    geometric_view_angle_deg(
                        actor_tf,
                        camera_tf,
                    )
                )

                relative_metrics = (
                    camera_relative_metrics(
                        actor_tf,
                        camera_tf,
                    )
                )

            frame_row = {
                "scenario_frame":
                    int(scenario_frame),
                "t_s":
                    float(
                        scenario_frame
                        /
                        scenario_fps
                    ),
                "carla_frame":
                    int(carla_frame),
                "record_attempts":
                    int(record_attempts),
                "ego_transform":
                    transform_to_dict(
                        ego_tf
                    ),
                "camera_transform":
                    transform_to_dict(
                        camera_tf
                    ),
                "actor_active":
                    bool(
                        actor_tf is not None
                    ),
                "actor_transform":
                    (
                        transform_to_dict(
                            actor_tf
                        )
                        if actor_tf is not None
                        else None
                    ),
                "carla_geometric_view_angle_deg":
                    (
                        float(geom_angle)
                        if geom_angle is not None
                        else None
                    ),
                "camera_relative":
                    relative_metrics,
                "mask_path":
                    str(
                        mask_path.relative_to(
                            output_dir
                        )
                    ).replace("\\", "/"),
                "mask_bbox":
                    mask_measurement,
                "visible_pixels":
                    int(
                        np.count_nonzero(
                            target_mask
                        )
                    ),
                "he_requested_viewpoint_angle_deg":
                    (
                        float(requested_angle)
                        if requested_angle is not None
                        else None
                    ),
                "he_selected_angle_deg":
                    (
                        float(selected_angle)
                        if selected_angle is not None
                        else None
                    ),
                "he_selected_distance_m":
                    (
                        float(selected_distance)
                        if selected_distance is not None
                        else None
                    ),
                "he_selected_elevation_deg":
                    (
                        float(selected_elevation)
                        if selected_elevation is not None
                        else None
                    ),
                "he_rendered":
                    he_rendered,
                "he_reason":
                    (
                        he_meta.get(
                            "reason",
                            ""
                        )
                        if he_meta is not None
                        else ""
                    ),
                "he_projection_box":
                    (
                        he_meta.get(
                            "box"
                        )
                        if he_meta is not None
                        else None
                    ),
                "he_rendered_alpha_bbox":
                    (
                        he_meta.get(
                            "rendered_alpha_bbox"
                        )
                        if he_meta is not None
                        else None
                    ),
                "he_sprite_mode":
                    (
                        he_meta.get(
                            "sprite_mode"
                        )
                        if he_meta is not None
                        else None
                    ),
                "he_selection_mode":
                    (
                        he_meta.get(
                            "selection_mode"
                        )
                        if he_meta is not None
                        else None
                    ),
                "he_distilled_neighbor_similarity":
                    (
                        he_meta.get(
                            "distilled_neighbor_similarity"
                        )
                        if he_meta is not None
                        else None
                    ),
                "he_cartesian_close":
                    (
                        he_meta.get(
                            "cartesian_close"
                        )
                        if he_meta is not None
                        else None
                    ),
                "he_anchor_mode":
                    (
                        he_meta.get("anchor_mode")
                        if he_meta is not None
                        else None
                    ),
                "he_target_physical_support":
                    (
                        he_meta.get("target_physical_support")
                        if he_meta is not None
                        else None
                    ),
                "he_camera_rotation_reprojection_meta":
                    (
                        he_meta.get("camera_rotation_reprojection_meta")
                        if he_meta is not None
                        else None
                    ),
            }

            append_jsonl(
                frames_fp,
                frame_row,
            )

            if (
                scenario_frame % 20 == 0
                or
                scenario_frame
                ==
                frame_count - 1
            ):
                distance_text = "NA"

                if relative_metrics is not None:
                    distance_text = (
                        f"{relative_metrics['camera_distance_m']:.2f}"
                    )

                print(
                    f"  frame={scenario_frame:04d} "
                    f"t={scenario_frame/scenario_fps:6.2f}s "
                    f"visible={int(np.any(target_mask))} "
                    f"distance={distance_text} "
                    f"geom_angle="
                    f"{geom_angle if geom_angle is not None else 'NA'} "
                    f"selected="
                    f"{selected_angle if selected_angle is not None else 'NA'}"
                )

        print()
        print("=" * 96)
        print("PAIR RECORDING COMPLETE")
        print("=" * 96)
        print("[condition] ", args.condition)
        print("[output]    ", output_dir)
        print("[setup]     ", setup_path)
        print("[trajectory]", trajectory_path if args.condition == "carla" else Path(args.reference_dir).resolve() / "ego_trajectory.jsonl")
        print("[frames]    ", frames_path)
        print("[masks]     ", mask_dir)
        print("[video]     ", video_path)
        if background_video_path is not None:
            print(
                "[background]",
                background_video_path,
            )
        print("=" * 96)

    finally:
        if frames_fp is not None:
            try:
                frames_fp.close()
            except Exception:
                pass

        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
        if background_writer is not None:
            try:
                background_writer.release()
            except Exception:
                pass
        if realizer is not None:
            try:
                realizer.destroy_all()
            except Exception:
                pass

        for sensor in (
            rgb_camera,
            instance_camera,
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
