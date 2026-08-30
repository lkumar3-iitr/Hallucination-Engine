from __future__ import annotations

import argparse
import json
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
    print_composite_summary,
)

from scenario_execution_runtime_v1 import (
    ExecutionWorldOrigin,
    load_execution_runtime,
)


THIS_FILE = Path(__file__).resolve()

DEFAULT_OUTPUT_DIR = (
    THIS_FILE.parent
    / "outputs"
    / "multi_actor_carla_background_v1"
)


# ============================================================
# CARLA image / sensor helpers
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

    # CARLA raw image is BGRA.
    rgb = (
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

    return rgb


def get_sensor_frame(
    sensor_queue,
    target_frame,
    label,
    timeout_s=10.0,
):
    """
    Return the sensor sample matching target_frame.

    Older samples are discarded. A newer sample before the
    requested one is treated as an error because synchronous
    mode should give us a deterministic frame match.
    """

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

    camera = world.spawn_actor(
        bp,
        relative_tf,
        attach_to=ego,
        attachment_type=
            carla.AttachmentType.Rigid,
    )

    return camera


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
# Execution origin
# ============================================================

def build_execution_origin(
    carla_map,
    ego0_tf,
):
    """
    ScenarioGenerator x/y/yaw are anchored to the canonical
    ego start.

    HE actor z, however, is a ground/base z. The generic
    execution runtime deliberately does not perform CARLA
    road snapping, so for this CARLA-background smoke test we
    use the driving waypoint's road-surface z.
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


# ============================================================
# Metadata
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render a multi-actor HE scenario into a live "
            "CARLA RGB background without spawning physical "
            "scenario adversaries."
        )
    )

    # --------------------------------------------------------
    # Scenario / assets
    # --------------------------------------------------------

    parser.add_argument(
        "--resolved",
        required=True,
        help=(
            "Resolved ScenarioGenerator v2 JSON."
        ),
    )

    parser.add_argument(
        "--asset-root",
        required=True,
        help=(
            "Root containing production HE asset banks."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    parser.add_argument(
        "--scenario-frame",
        type=int,
        default=40,
        help=(
            "Scenario frame to render."
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
        "--fps",
        type=float,
        default=20.0,
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

    # --------------------------------------------------------
    # Camera
    #
    # Defaults = validated TCP native camera.
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

    print()
    print("=" * 78)
    print(
        "HE MULTI-ACTOR + CARLA BACKGROUND SMOKE TEST V1"
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
        "[scenario frame]",
        args.scenario_frame,
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
            f"{args.camera_z}) "
            f"rpy=("
            f"{args.camera_roll},"
            f"{args.camera_pitch},"
            f"{args.camera_yaw})"
        ),
    )

    # ========================================================
    # CARLA connection
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

    print()
    print(
        "[CARLA] loading world:",
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

    try:
        # ====================================================
        # Deterministic synchronous mode
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            1.0
            /
            float(
                args.fps
            )
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
        # Spawn only ego.
        #
        # IMPORTANT:
        # No physical scenario adversaries are spawned.
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

        print(
            "[CARLA] ego spawned:",
            args.ego_blueprint,
        )

        # ====================================================
        # Canonical settled ego start
        # ====================================================

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

        print()
        print(
            "[ego canonical transform]",
            (
                f"x={ego0_tf.location.x:.3f} "
                f"y={ego0_tf.location.y:.3f} "
                f"z={ego0_tf.location.z:.3f} "
                f"yaw={ego0_tf.rotation.yaw:.3f}"
            ),
        )

        # Hold ego stationary for this one-frame visual test.
        ego.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True,
                reverse=False,
            )
        )

        # ====================================================
        # Build model-independent scenario execution origin.
        # ====================================================

        origin = build_execution_origin(
            carla_map=carla_map,
            ego0_tf=ego0_tf,
        )

        print(
            "[execution origin]",
            origin,
        )

        # ====================================================
        # Runtime
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

        print(
            "[runtime summary]",
            runtime.summary(),
        )

        active_actors = list(
            runtime.active_actors(
                int(
                    args.scenario_frame
                )
            )
        )

        print(
            "[active actors]",
            len(
                active_actors
            ),
        )

        if not active_actors:
            raise RuntimeError(
                f"No active actors at scenario frame "
                f"{args.scenario_frame}."
            )

        for actor in active_actors:
            print(
                "  -",
                actor.actor_id,
                "|",
                actor.canonical_asset_key,
                "|",
                actor.carla_blueprint,
                "|",
                (
                    f"world=("
                    f"{actor.world_x_m:.3f}, "
                    f"{actor.world_y_m:.3f}, "
                    f"{actor.world_z_m:.3f})"
                ),
            )

        # ====================================================
        # RGB camera
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
        # One synchronized CARLA background frame
        # ====================================================

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

        base_rgb = carla_image_to_rgb(
            image
        )

        camera_tf = (
            camera.get_transform()
        )

        print()
        print(
            "[CARLA frame]",
            carla_frame,
        )

        print(
            "[camera world transform]",
            (
                f"x={camera_tf.location.x:.3f} "
                f"y={camera_tf.location.y:.3f} "
                f"z={camera_tf.location.z:.3f} "
                f"pitch={camera_tf.rotation.pitch:.3f} "
                f"yaw={camera_tf.rotation.yaw:.3f} "
                f"roll={camera_tf.rotation.roll:.3f}"
            ),
        )

        # ====================================================
        # HE multi-actor composition
        # ====================================================

        compositor = (
            HEMultiActorCompositorV1(
                distance_selection_mode=
                    args.distance_selection_mode,

                bottom_y_offset_px=
                    args.bottom_y_offset_px,
            )
        )

        result = compositor.render(
            base_rgb=base_rgb,
            camera_tf=camera_tf,
            active_actors=
                active_actors,
            width=args.width,
            height=args.height,
            fov=args.fov,
        )

        print()
        print_composite_summary(
            result
        )

        # ====================================================
        # Save diagnostics
        # ====================================================

        background_path = (
            output_dir
            /
            (
                f"scenario_"
                f"{args.scenario_frame:04d}"
                f"_carla_background.png"
            )
        )

        composite_path = (
            output_dir
            /
            (
                f"scenario_"
                f"{args.scenario_frame:04d}"
                f"_he_composite.png"
            )
        )

        side_by_side_path = (
            output_dir
            /
            (
                f"scenario_"
                f"{args.scenario_frame:04d}"
                f"_side_by_side.png"
            )
        )

        metadata_path = (
            output_dir
            /
            (
                f"scenario_"
                f"{args.scenario_frame:04d}"
                f"_metadata.json"
            )
        )

        save_rgb(
            background_path,
            base_rgb,
        )

        save_rgb(
            composite_path,
            result.rgb,
        )

        side_by_side = np.concatenate(
            [
                base_rgb,
                result.rgb,
            ],
            axis=1,
        )

        save_rgb(
            side_by_side_path,
            side_by_side,
        )

        metadata = {
            "scenario_frame":
                int(
                    args.scenario_frame
                ),

            "carla_frame":
                int(
                    carla_frame
                ),

            "town":
                str(
                    args.town
                ),

            "spawn_index":
                int(
                    spawn_idx
                ),

            "ego_blueprint":
                str(
                    args.ego_blueprint
                ),

            "execution_origin": {
                "x_m":
                    float(
                        origin.x_m
                    ),

                "y_m":
                    float(
                        origin.y_m
                    ),

                "z_m":
                    float(
                        origin.z_m
                    ),

                "yaw_deg":
                    float(
                        origin.yaw_deg
                    ),
            },

            "camera": {
                "width":
                    int(
                        args.width
                    ),

                "height":
                    int(
                        args.height
                    ),

                "fov_deg":
                    float(
                        args.fov
                    ),

                "world_x":
                    float(
                        camera_tf.location.x
                    ),

                "world_y":
                    float(
                        camera_tf.location.y
                    ),

                "world_z":
                    float(
                        camera_tf.location.z
                    ),

                "world_pitch_deg":
                    float(
                        camera_tf.rotation.pitch
                    ),

                "world_yaw_deg":
                    float(
                        camera_tf.rotation.yaw
                    ),

                "world_roll_deg":
                    float(
                        camera_tf.rotation.roll
                    ),
            },

            "actors": [
                actor_result_to_dict(
                    row
                )
                for row
                in result.actor_results
            ],
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(
                metadata,
                fp,
                indent=2,
                default=str,
            )

        print()
        print("=" * 78)
        print(
            "CARLA BACKGROUND SMOKE TEST COMPLETE"
        )
        print("=" * 78)

        print(
            "[background]",
            background_path,
        )

        print(
            "[composite]",
            composite_path,
        )

        print(
            "[side by side]",
            side_by_side_path,
        )

        print(
            "[metadata]",
            metadata_path,
        )

        print()
        print(
            "No physical scenario adversaries were spawned."
        )

        print("=" * 78)

    finally:
        # ====================================================
        # Cleanup
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

        try:
            world.apply_settings(
                original_settings
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
