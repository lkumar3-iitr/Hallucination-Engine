from __future__ import annotations

import argparse
import json
import math
import queue
from pathlib import Path

import cv2
import numpy as np
import carla

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


THIS_FILE = Path(__file__).resolve()

DEFAULT_OUTPUT_DIR = (
    THIS_FILE.parent
    / "outputs"
    / "multi_actor_carla_video_v1"
)


# ============================================================
# CARLA helpers
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
            int(image.height),
            int(image.width),
            4,
        )
    )

    # CARLA raw camera image is BGRA.
    return (
        array[
            :,
            :,
            :3,
        ][
            :,
            :,
            ::-1,
        ]
        .copy()
    )


def get_sensor_frame(
    sensor_queue,
    target_frame,
    label,
    timeout_s=10.0,
):
    while True:
        sample = sensor_queue.get(
            timeout=float(
                timeout_s
            )
        )

        frame = int(
            sample.frame
        )

        if frame < int(
            target_frame
        ):
            continue

        if frame > int(
            target_frame
        ):
            raise RuntimeError(
                f"{label} skipped target CARLA frame "
                f"{target_frame}; got {frame}."
            )

        return sample


def make_rgb_camera(
    world,
    ego,
    width,
    height,
    fov,
    x,
    y,
    z,
    pitch,
    yaw,
    roll,
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
        str(
            int(width)
        ),
    )

    bp.set_attribute(
        "image_size_y",
        str(
            int(height)
        ),
    )

    bp.set_attribute(
        "fov",
        str(
            float(fov)
        ),
    )

    relative_tf = carla.Transform(
        carla.Location(
            x=float(x),
            y=float(y),
            z=float(z),
        ),
        carla.Rotation(
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
        ),
    )

    return world.spawn_actor(
        bp,
        relative_tf,
        attach_to=ego,
        attachment_type=
            carla.AttachmentType.Rigid,
    )


def build_execution_origin(
    carla_map,
    ego0_tf,
):
    """
    ScenarioGenerator world XY/yaw are anchored to the canonical
    ego start. Actor world_z_m is a road/base z, so use the
    driving waypoint's road-surface z rather than the ego body z.
    """

    waypoint = carla_map.get_waypoint(
        ego0_tf.location,
        project_to_road=True,
        lane_type=
            carla.LaneType.Driving,
    )

    if waypoint is None:
        raise RuntimeError(
            "Could not resolve a driving waypoint "
            "for the canonical ego start."
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


def resolved_ego_transform(ego_frame, origin, ego_z_m):
    """Convert one ScenarioGenerator ego-initial pose into CARLA world space."""
    origin_yaw_rad = math.radians(float(origin.yaw_deg))
    forward_x = math.cos(origin_yaw_rad)
    forward_y = math.sin(origin_yaw_rad)
    left_x = -forward_y
    left_y = forward_x

    x_m = float(ego_frame["x_m"])
    y_m = float(ego_frame["y_m"])

    return carla.Transform(
        carla.Location(
            x=float(origin.x_m) + x_m * forward_x + y_m * left_x,
            y=float(origin.y_m) + x_m * forward_y + y_m * left_y,
            z=float(ego_z_m),
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=float(origin.yaw_deg) + float(ego_frame["yaw_deg"]),
            roll=0.0,
        ),
    )


def create_video_writer(
    path,
    fps,
    width,
    height,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            f"Could not create video writer: {path}"
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


def save_rgb(
    path,
    rgb,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok = cv2.imwrite(
        str(
            path
        ),
        rgb[
            :,
            :,
            ::-1,
        ],
    )

    if not ok:
        raise RuntimeError(
            f"Could not save image: {path}"
        )


# ============================================================
# Metadata helpers
# ============================================================

def actor_result_to_dict(
    row,
):
    return {
        "actor_id":
            row.actor_id,

        "asset_key":
            row.asset_key,

        "carla_blueprint":
            row.carla_blueprint,

        "render_order":
            int(
                row.render_order
            ),

        "rel_x_m":
            float(
                row.rel_x_m
            ),

        "rel_y_m":
            float(
                row.rel_y_m
            ),

        "rel_z_m":
            float(
                row.rel_z_m
            ),

        "rel_yaw_deg":
            float(
                row.rel_yaw_deg
            ),

        "distance_forward_m":
            float(
                row.distance_forward_m
            ),

        "distance_euclidean_m":
            float(
                row.distance_euclidean_m
            ),

        "he_view_matrix_csv":
            row.he_view_matrix_csv,

        "rendered":
            bool(
                row.rendered
            ),

        "he_metadata":
            row.he_metadata,
    }


def save_sample_frame(
    sample_dir,
    scenario_frame,
    background_rgb,
    composite_rgb,
):
    sample_dir = Path(
        sample_dir
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = (
        f"frame_"
        f"{int(scenario_frame):04d}"
    )

    save_rgb(
        sample_dir
        /
        (
            base_name
            + "_background.png"
        ),
        background_rgb,
    )

    save_rgb(
        sample_dir
        /
        (
            base_name
            + "_he.png"
        ),
        composite_rgb,
    )

    side_by_side = np.concatenate(
        [
            background_rgb,
            composite_rgb,
        ],
        axis=1,
    )

    save_rgb(
        sample_dir
        /
        (
            base_name
            + "_side_by_side.png"
        ),
        side_by_side,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a complete resolved multi-actor HE scenario "
            "over a synchronized live CARLA background."
        )
    )

    # --------------------------------------------------------
    # Scenario / asset registry
    # --------------------------------------------------------

    parser.add_argument(
        "--resolved",
        required=True,
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
        "--start-frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help=(
            "-1 derives the final frame from "
            "duration_s * scenario fps."
        ),
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=40,
        help=(
            "Save diagnostic PNGs every N scenario frames. "
            "Set <=0 to disable."
        ),
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
        "--timeout",
        type=float,
        default=20.0,
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
        "--ego-blueprint",
        default="vehicle.tesla.model3",
    )

    parser.add_argument(
        "--settle-ticks",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--hold-ticks",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--ego-motion",
        choices=["frozen", "resolved"],
        default="frozen",
        help=(
            "Keep the legacy visual-smoke ego fixed, or apply the exact "
            "ResolvedScenarioV2 ego pose on every frame."
        ),
    )

    # --------------------------------------------------------
    # Camera
    #
    # Defaults are TCP's validated native camera.
    # --------------------------------------------------------

    parser.add_argument(
        "--width",
        type=int,
        default=900,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--camera-x",
        type=float,
        default=-1.5,
    )

    parser.add_argument(
        "--camera-y",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--camera-z",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--camera-yaw",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--camera-roll",
        type=float,
        default=0.0,
    )

    # --------------------------------------------------------
    # HE
    # --------------------------------------------------------

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
        "--bottom-y-offset-px",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    args = parser.parse_args()

    resolved_path = Path(
        args.resolved
    ).resolve()

    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved_document = json.load(handle)

    resolved_ego_frames = {
        int(frame["frame_idx"]): frame
        for frame in resolved_document.get("ego_frames", [])
    }

    asset_root = Path(
        args.asset_root
    ).resolve()

    manifest_path = Path(
        args.manifest
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_dir = (
        output_dir
        /
        "samples"
    )

    print()
    print("=" * 78)
    print(
        "HE MULTI-ACTOR CARLA VIDEO V1"
    )
    print("=" * 78)

    print(
        "[resolved]",
        resolved_path,
    )

    print(
        "[asset root]",
        asset_root,
    )

    print(
        "[manifest]",
        manifest_path,
    )

    print(
        "[town]",
        args.town,
    )

    print(
        "[spawn index]",
        args.spawn_index,
    )

    print(
        "[camera]",
        (
            f"{args.width}x{args.height} "
            f"FOV={args.fov} "
            f"xyz=("
            f"{args.camera_x},"
            f"{args.camera_y},"
            f"{args.camera_z})"
        ),
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
        float(
            args.timeout
        )
    )

    world = client.load_world(
        args.town
    )

    original_settings = (
        world.get_settings()
    )

    ego = None
    camera = None

    background_writer = None
    composite_writer = None
    side_by_side_writer = None
    metadata_fp = None

    try:
        # ====================================================
        # Synchronous deterministic world
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        # Temporary 20 Hz until runtime is loaded. This matches
        # the current scenario and the validated TCP experiments.
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

        nominal_spawn_tf = (
            spawn_points[
                spawn_idx
            ]
        )

        print(
            "[CARLA] resolved spawn index:",
            spawn_idx,
        )

        # ====================================================
        # Ego only
        # ====================================================

        ego_bp = (
            world
            .get_blueprint_library()
            .find(
                args.ego_blueprint
            )
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
            nominal_spawn_tf,
        )

        if ego is None:
            raise RuntimeError(
                "Could not spawn ego vehicle."
            )

        (
            ego0_tf,
            ego0_state,
        ) = canonicalize_ego_start(
            world=world,
            ego=ego,
            nominal_spawn_tf=
                nominal_spawn_tf,
            settle_ticks=int(
                args.settle_ticks
            ),
            hold_ticks=int(
                args.hold_ticks
            ),
        )

        # This visual execution test intentionally freezes the
        # ego. Later TCP/NEAT wrappers will provide real controls.
        hold_control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
            reverse=False,
        )

        ego.apply_control(
            hold_control
        )

        origin = build_execution_origin(
            carla_map=carla_map,
            ego0_tf=ego0_tf,
        )

        print(
            "[execution origin]",
            origin,
        )

        # ====================================================
        # Scenario runtime
        # ====================================================

        runtime = load_execution_runtime(
            resolved_json=str(
                resolved_path
            ),
            asset_root=asset_root,
            origin=origin,
            manifest_path=
                manifest_path,
        )

        runtime_summary = (
            runtime.summary()
        )

        print(
            "[runtime summary]",
            runtime_summary,
        )

        scenario_fps = float(
            runtime_summary[
                "fps"
            ]
        )

        if scenario_fps <= 0.0:
            raise RuntimeError(
                "Scenario FPS must be positive."
            )

        # Now enforce CARLA tick rate == scenario tick rate.
        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            1.0
            /
            scenario_fps
        )

        world.apply_settings(
            settings
        )

        start_frame = max(
            0,
            int(
                args.start_frame
            ),
        )

        if int(
            args.end_frame
        ) >= 0:
            end_frame = int(
                args.end_frame
            )
        else:
            # Inclusive final frame. For an 8 s, 20 Hz scenario
            # this gives frames 0..160.
            end_frame = int(
                round(
                    float(
                        runtime_summary[
                            "duration_s"
                        ]
                    )
                    *
                    scenario_fps
                )
            )

        if end_frame < start_frame:
            raise RuntimeError(
                "end_frame is before start_frame."
            )

        print(
            "[scenario frame range]",
            f"{start_frame}..{end_frame}",
        )

        print(
            "[scenario fps]",
            scenario_fps,
        )

        # ====================================================
        # Camera
        # ====================================================

        camera = make_rgb_camera(
            world=world,
            ego=ego,

            width=args.width,
            height=args.height,
            fov=args.fov,

            x=args.camera_x,
            y=args.camera_y,
            z=args.camera_z,

            pitch=args.camera_pitch,
            yaw=args.camera_yaw,
            roll=args.camera_roll,
        )

        camera_queue = (
            queue.Queue()
        )

        camera.listen(
            camera_queue.put
        )

        # ====================================================
        # HE compositor
        # ====================================================

        compositor = (
            HEMultiActorCompositorV1(
                distance_selection_mode=
                    args.distance_selection_mode,

                bottom_y_offset_px=
                    args.bottom_y_offset_px,
            )
        )

        # ====================================================
        # Outputs
        # ====================================================

        background_video_path = (
            output_dir
            /
            "carla_background.mp4"
        )

        composite_video_path = (
            output_dir
            /
            "he_multi_actor.mp4"
        )

        side_by_side_video_path = (
            output_dir
            /
            "background_vs_he.mp4"
        )

        metadata_path = (
            output_dir
            /
            "frames.jsonl"
        )

        background_writer = (
            create_video_writer(
                path=
                    background_video_path,
                fps=
                    scenario_fps,
                width=
                    args.width,
                height=
                    args.height,
            )
        )

        composite_writer = (
            create_video_writer(
                path=
                    composite_video_path,
                fps=
                    scenario_fps,
                width=
                    args.width,
                height=
                    args.height,
            )
        )

        side_by_side_writer = (
            create_video_writer(
                path=
                    side_by_side_video_path,
                fps=
                    scenario_fps,
                width=
                    int(
                        args.width
                    )
                    * 2,
                height=
                    args.height,
            )
        )

        metadata_fp = open(
            metadata_path,
            "w",
            encoding="utf-8",
        )

        # ====================================================
        # Main scenario loop
        # ====================================================

        total = (
            end_frame
            -
            start_frame
            +
            1
        )

        for sequence_idx, scenario_frame in enumerate(
            range(
                start_frame,
                end_frame + 1,
            )
        ):
            if args.ego_motion == "resolved":
                ego_frame = resolved_ego_frames.get(int(scenario_frame))
                if ego_frame is None:
                    raise RuntimeError(
                        f"Resolved ego frame {scenario_frame} is missing."
                    )
                ego.set_transform(
                    resolved_ego_transform(
                        ego_frame=ego_frame,
                        origin=origin,
                        ego_z_m=ego0_tf.location.z,
                    )
                )

            ego.apply_control(
                hold_control
            )

            carla_frame = (
                world.tick()
            )

            image = get_sensor_frame(
                sensor_queue=
                    camera_queue,
                target_frame=
                    carla_frame,
                label=
                    "RGB camera",
            )

            background_rgb = (
                carla_image_to_rgb(
                    image
                )
            )

            camera_tf = (
                camera.get_transform()
            )

            active_actors = list(
                runtime.active_actors(
                    scenario_frame
                )
            )

            result = compositor.render(
                base_rgb=
                    background_rgb,
                camera_tf=
                    camera_tf,
                active_actors=
                    active_actors,
                width=
                    args.width,
                height=
                    args.height,
                fov=
                    args.fov,
            )

            write_rgb_frame(
                background_writer,
                background_rgb,
            )

            write_rgb_frame(
                composite_writer,
                result.rgb,
            )

            side_by_side = np.concatenate(
                [
                    background_rgb,
                    result.rgb,
                ],
                axis=1,
            )

            write_rgb_frame(
                side_by_side_writer,
                side_by_side,
            )

            row = {
                "sequence_idx":
                    int(
                        sequence_idx
                    ),

                "scenario_frame":
                    int(
                        scenario_frame
                    ),

                "t_s":
                    float(
                        scenario_frame
                    )
                    /
                    scenario_fps,

                "carla_frame":
                    int(
                        carla_frame
                    ),

                "camera_world": {
                    "x":
                        float(
                            camera_tf.location.x
                        ),

                    "y":
                        float(
                            camera_tf.location.y
                        ),

                    "z":
                        float(
                            camera_tf.location.z
                        ),

                    "pitch_deg":
                        float(
                            camera_tf.rotation.pitch
                        ),

                    "yaw_deg":
                        float(
                            camera_tf.rotation.yaw
                        ),

                    "roll_deg":
                        float(
                            camera_tf.rotation.roll
                        ),
                },

                "active_actor_count":
                    len(
                        active_actors
                    ),

                "actors": [
                    actor_result_to_dict(
                        actor_row
                    )
                    for actor_row
                    in result.actor_results
                ],
            }

            metadata_fp.write(
                json.dumps(
                    row,
                    default=str,
                )
                +
                "\n"
            )

            if (
                int(
                    args.sample_every
                )
                > 0
                and
                (
                    (
                        scenario_frame
                        -
                        start_frame
                    )
                    %
                    int(
                        args.sample_every
                    )
                    == 0
                    or
                    scenario_frame
                    == end_frame
                )
            ):
                save_sample_frame(
                    sample_dir=
                        sample_dir,
                    scenario_frame=
                        scenario_frame,
                    background_rgb=
                        background_rgb,
                    composite_rgb=
                        result.rgb,
                )

            if (
                sequence_idx == 0
                or
                sequence_idx == total - 1
                or
                sequence_idx % 20 == 0
            ):
                rendered_count = sum(
                    1
                    for row
                    in result.actor_results
                    if row.rendered
                )

                actor_ids = [
                    row.actor_id
                    for row
                    in result.actor_results
                ]

                print(
                    (
                        f"[frame {scenario_frame:04d}] "
                        f"CARLA={carla_frame} "
                        f"active={len(active_actors)} "
                        f"rendered={rendered_count} "
                        f"actors={actor_ids}"
                    )
                )

        # ====================================================
        # Success
        # ====================================================

        print()
        print("=" * 78)
        print(
            "MULTI-FRAME HE EXECUTION COMPLETE"
        )
        print("=" * 78)

        print(
            "[background video]",
            background_video_path,
        )

        print(
            "[HE video]",
            composite_video_path,
        )

        print(
            "[side-by-side video]",
            side_by_side_video_path,
        )

        print(
            "[metadata]",
            metadata_path,
        )

        print(
            "[sample frames]",
            sample_dir,
        )

        print()
        print(
            "No physical scenario adversaries were spawned."
        )

        print("=" * 78)

    finally:
        # ====================================================
        # Close outputs first
        # ====================================================

        if metadata_fp is not None:
            try:
                metadata_fp.close()
            except Exception:
                pass

        for writer in (
            background_writer,
            composite_writer,
            side_by_side_writer,
        ):
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

        # ====================================================
        # Destroy CARLA actors/sensors
        # ====================================================

        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

            try:
                camera.destroy()
            except Exception:
                pass

        if ego is not None:
            try:
                ego.destroy()
            except Exception:
                pass

        # ====================================================
        # Restore user's CARLA settings
        # ====================================================

        try:
            world.apply_settings(
                original_settings
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
