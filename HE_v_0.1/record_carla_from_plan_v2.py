import argparse
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import cv2
import carla

import record_carla_he_pair as base


# ============================================================
# Blueprint helpers
# ============================================================

def resolve_carla_blueprint(plan_blueprint):
    """
    Resolve ScenarioGenerator logical vehicle names into concrete
    CARLA blueprint IDs.

    The v2 CARLA adapter normally already writes a concrete CARLA
    blueprint, but these aliases keep this executor compatible with
    generic logical asset names too.
    """

    aliases = {
        "vehicle.sedan.generic": "vehicle.audi.tt",
        "sedan.generic": "vehicle.audi.tt",
    }

    return aliases.get(
        plan_blueprint,
        plan_blueprint,
    )


# ============================================================
# Plan loading / validation
# ============================================================

def load_plan(path):
    """
    Load a CARLA execution plan v2.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        plan = json.load(f)

    if (
        plan.get("coordinate_frame")
        != "ego_initial"
    ):
        raise ValueError(
            "Expected coordinate_frame='ego_initial'."
        )

    if (
        str(plan.get("plan_version"))
        != "2.0"
    ):
        raise ValueError(
            "record_carla_from_plan_v2.py expects "
            "plan_version='2.0'."
        )

    if "ego" not in plan:
        raise ValueError(
            "V2 plan missing 'ego'."
        )

    if "frames" not in plan["ego"]:
        raise ValueError(
            "V2 plan missing ego.frames."
        )

    actors = plan.get(
        "actors",
        [],
    )

    actor_ids = [
        str(actor["actor_id"])
        for actor in actors
    ]

    if len(actor_ids) != len(
        set(actor_ids)
    ):
        raise ValueError(
            "CARLA plan contains duplicate "
            "actor_id values."
        )

    return plan


def index_frames(rows):
    """
    Convert a list of frame rows into:

        frame_idx -> row
    """

    return {
        int(row["frame_idx"]): row
        for row in rows
    }


def validate_contiguous_ego_frames(
    ego_frames,
):
    """
    Ego must exist throughout the entire scenario.
    """

    frame_ids = sorted(
        ego_frames
    )

    if not frame_ids:
        raise ValueError(
            "ego.frames is empty."
        )

    if frame_ids != list(
        range(
            len(frame_ids)
        )
    ):
        raise ValueError(
            "ego.frames must be contiguous "
            "from frame 0."
        )

    return frame_ids


def validate_actor_frames(
    actor_id,
    actor_frames,
    ego_frame_ids,
):
    """
    V2 actors may exist for only part of the scenario.

    However, once active, their frame sequence must be contiguous.
    """

    actor_frame_ids = sorted(
        actor_frames
    )

    if not actor_frame_ids:
        return

    expected = list(
        range(
            actor_frame_ids[0],
            actor_frame_ids[-1] + 1,
        )
    )

    if actor_frame_ids != expected:
        raise ValueError(
            f"Actor {actor_id!r} contains "
            "non-contiguous active frames."
        )

    if (
        actor_frame_ids[0]
        < ego_frame_ids[0]
        or
        actor_frame_ids[-1]
        > ego_frame_ids[-1]
    ):
        raise ValueError(
            f"Actor {actor_id!r} contains "
            "frames outside the ego timeline."
        )


# ============================================================
# Coordinate helpers
# ============================================================

def local_state(row):
    """
    Store the canonical executor-local physical state.

    local_x:
        lateral right

    local_y:
        vertical

    local_z:
        forward

    local_yaw:
        clockwise/right-positive
    """

    return {
        "x_m": float(
            row["local_x_m"]
        ),

        "y_m": float(
            row.get(
                "local_y_m",
                0.0,
            )
        ),

        "z_m": float(
            row["local_z_m"]
        ),

        "yaw_deg": float(
            row["local_yaw_deg"]
        ),
    }


def local_to_world(
    ego0_tf,
    row,
    world,
    snap_to_road,
):
    """
    Convert one ego-initial-local resolved state into CARLA world
    coordinates.

    This reuses the previously validated transformation function.
    """

    tf = (
        base.local_ego_initial_to_world_transform(
            ego0_tf=ego0_tf,

            local_x_m=float(
                row["local_x_m"]
            ),

            local_y_m=float(
                row.get(
                    "local_y_m",
                    0.0,
                )
            ),

            local_z_m=float(
                row["local_z_m"]
            ),

            local_yaw_deg=float(
                row["local_yaw_deg"]
            ),

            world=world,

            snap_to_road=
                snap_to_road,

            z_lift=0.05,
        )
    )

    # --------------------------------------------------------
    # Ego:
    # preserve settled road pitch / roll.
    # --------------------------------------------------------

    if not snap_to_road:

        tf.rotation.pitch = float(
            ego0_tf.rotation.pitch
        )

        tf.rotation.roll = float(
            ego0_tf.rotation.roll
        )

    return tf


# ============================================================
# Main executor
# ============================================================

def run(args):

    # --------------------------------------------------------
    # Load plan
    # --------------------------------------------------------

    plan = load_plan(
        args.plan
    )

    fps = float(
        plan["fps"]
    )

    # --------------------------------------------------------
    # Ego timeline
    # --------------------------------------------------------

    ego_frames = index_frames(
        plan["ego"]["frames"]
    )

    frame_ids = (
        validate_contiguous_ego_frames(
            ego_frames
        )
    )

    # --------------------------------------------------------
    # Actor timelines
    # --------------------------------------------------------

    actor_plans = {}

    actor_frames_by_id = {}

    for actor_plan in plan.get(
        "actors",
        [],
    ):

        actor_id = str(
            actor_plan["actor_id"]
        )

        actor_plans[
            actor_id
        ] = actor_plan

        indexed = index_frames(
            actor_plan.get(
                "frames",
                [],
            )
        )

        actor_frames_by_id[
            actor_id
        ] = indexed

        validate_actor_frames(
            actor_id=actor_id,
            actor_frames=indexed,
            ego_frame_ids=frame_ids,
        )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera_cfg = plan[
        "camera"
    ]

    width = int(
        camera_cfg[
            "image_width"
        ]
    )

    height = int(
        camera_cfg[
            "image_height"
        ]
    )

    fov = float(
        camera_cfg[
            "fov"
        ]
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    pair_dir = (
        Path(
            args.output_root
        )
        / args.pair_name
    )

    pair_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.mode
        == "background"
    ):

        video_path = (
            pair_dir
            / "background.mp4"
        )

        ego_pose_path = (
            pair_dir
            / "background_ego_pose.jsonl"
        )

        ground_truth_path = None

    else:

        # Keep the existing filename for compatibility even though
        # v2 may contain several adversaries.
        video_path = (
            pair_dir
            / "real_adversary.mp4"
        )

        ego_pose_path = (
            pair_dir
            / "real_ego_pose.jsonl"
        )

        ground_truth_path = (
            pair_dir
            / "real_ground_truth_v2.jsonl"
        )

    # --------------------------------------------------------
    # Connect to CARLA
    # --------------------------------------------------------

    client = carla.Client(
        args.host,
        args.port,
    )

    client.set_timeout(
        20.0
    )

    if args.town:

        world = (
            client.load_world(
                args.town
            )
        )

    else:

        world = (
            client.get_world()
        )

    original_settings = (
        world.get_settings()
    )

    # --------------------------------------------------------
    # Resource tracking
    # --------------------------------------------------------

    created_actors = []

    writer = None

    ego_file = None

    ground_truth_file = None

    try:

        # ====================================================
        # CARLA synchronous configuration
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = (
            True
        )

        settings.fixed_delta_seconds = (
            1.0 / fps
        )

        world.apply_settings(
            settings
        )

        # ====================================================
        # Spawn ego
        # ====================================================

        spawn_points = (
            world
            .get_map()
            .get_spawn_points()
        )

        if not (
            0
            <= args.spawn_index
            < len(spawn_points)
        ):

            raise ValueError(
                f"spawn-index "
                f"{args.spawn_index} "
                f"out of range "
                f"0.."
                f"{len(spawn_points)-1}"
            )

        ego_bp = (
            base.get_blueprint(
                world,
                args.ego_vehicle_filter,
            )
        )

        ego = (
            base.spawn_actor_safe(
                world,
                ego_bp,

                spawn_points[
                    args.spawn_index
                ],
            )
        )

        created_actors.append(
            ego
        )

        # ====================================================
        # Settle ego before freezing
        # ====================================================

        ego.set_simulate_physics(
            True
        )

        for _ in range(
            args.ego_settle_frames
        ):

            ego.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    steer=0.0,
                )
            )

            world.tick()

        ego0_tf = (
            ego.get_transform()
        )

        ego.set_simulate_physics(
            False
        )

        # ----------------------------------------------------
        # Put ego at exact resolved frame zero.
        # ----------------------------------------------------

        ego.set_transform(
            local_to_world(
                ego0_tf,
                ego_frames[0],
                world,
                snap_to_road=False,
            )
        )

        # ====================================================
        # Camera
        # ====================================================

        camera_args = (
            SimpleNamespace(

                width=width,

                height=height,

                fov=fov,

                camera_x=float(
                    camera_cfg.get(
                        "camera_x",
                        1.5,
                    )
                ),

                camera_y=float(
                    camera_cfg.get(
                        "camera_y",
                        0.0,
                    )
                ),

                camera_z=float(
                    camera_cfg.get(
                        "camera_z",
                        1.6,
                    )
                ),

                camera_pitch=float(
                    camera_cfg.get(
                        "pitch",
                        0.0,
                    )
                ),

                camera_yaw=float(
                    camera_cfg.get(
                        "yaw",
                        0.0,
                    )
                ),

                camera_roll=float(
                    camera_cfg.get(
                        "roll",
                        0.0,
                    )
                ),
            )
        )

        camera = (
            base.make_rgb_camera(
                world,
                ego,
                camera_args,
            )
        )

        created_actors.append(
            camera
        )

        image_queue = (
            queue.Queue()
        )

        camera.listen(
            image_queue.put
        )

        # ====================================================
        # Multi-actor runtime state
        # ====================================================

        active_adversaries = {}

        actor_blueprints = {}

        # ----------------------------------------------------
        # Resolve all CARLA blueprints once.
        # ----------------------------------------------------

        if (
            args.mode
            == "real_adversary"
        ):

            for (
                actor_id,
                actor_plan,
            ) in actor_plans.items():

                requested_blueprint = (
                    actor_plan[
                        "blueprint"
                    ]
                )

                carla_blueprint = (
                    resolve_carla_blueprint(
                        requested_blueprint
                    )
                )

                print(
                    "[INFO] Actor blueprint:",
                    actor_id,
                    requested_blueprint,
                    "->",
                    carla_blueprint,
                )

                actor_blueprints[
                    actor_id
                ] = (
                    base.get_blueprint(
                        world,
                        carla_blueprint,
                    )
                )

        # ====================================================
        # Runtime actor spawn helper
        # ====================================================

        def spawn_runtime_actor(
            actor_id,
            row,
        ):
            """
            Spawn one actor on the first frame of its lifecycle.
            """

            if (
                actor_id
                in active_adversaries
            ):
                return

            actor_tf = (
                local_to_world(
                    ego0_tf,
                    row,
                    world,
                    snap_to_road=True,
                )
            )

            carla_actor = (
                base.spawn_actor_safe(
                    world,

                    actor_blueprints[
                        actor_id
                    ],

                    actor_tf,
                )
            )

            carla_actor.set_simulate_physics(
                False
            )

            active_adversaries[
                actor_id
            ] = carla_actor

            created_actors.append(
                carla_actor
            )

            print(
                f"[INFO] SPAWN "
                f"actor={actor_id} "
                f"frame="
                f"{int(row['frame_idx'])} "
                f"t="
                f"{float(row['t_s']):.3f}"
            )

        # ====================================================
        # Runtime actor destroy helper
        # ====================================================

        def destroy_runtime_actor(
            actor_id,
            frame_idx,
            t_s,
        ):
            """
            Destroy an actor BEFORE ticking the first frame on which
            that actor should no longer exist.
            """

            carla_actor = (
                active_adversaries.pop(
                    actor_id,
                    None,
                )
            )

            if carla_actor is None:
                return

            try:
                carla_actor.destroy()

            except Exception:
                pass

            print(
                f"[INFO] DESPAWN "
                f"actor={actor_id} "
                f"before_frame="
                f"{frame_idx} "
                f"t={t_s:.3f}"
            )

        # ====================================================
        # Spawn actors active on frame zero
        # ====================================================
        #
        # They must exist while the camera warms up.
        # Their pose remains frozen at frame zero during warmup.
        # ====================================================

        if (
            args.mode
            == "real_adversary"
        ):

            for (
                actor_id,
                actor_frames,
            ) in (
                actor_frames_by_id
                .items()
            ):

                row0 = (
                    actor_frames.get(
                        0
                    )
                )

                if row0 is not None:

                    spawn_runtime_actor(
                        actor_id,
                        row0,
                    )

        # ====================================================
        # Video writer
        # ====================================================

        writer = (
            cv2.VideoWriter(

                str(
                    video_path
                ),

                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),

                fps,

                (
                    width,
                    height,
                ),
            )
        )

        if not writer.isOpened():

            raise RuntimeError(
                f"Could not open "
                f"video writer: "
                f"{video_path}"
            )

        # ====================================================
        # Metadata files
        # ====================================================

        ego_file = open(
            ego_pose_path,
            "w",
            encoding="utf-8",
        )

        if ground_truth_path:

            ground_truth_file = open(
                ground_truth_path,
                "w",
                encoding="utf-8",
            )

        # ====================================================
        # Run information
        # ====================================================

        print(
            "[INFO] Plan:",
            args.plan,
        )

        print(
            "[INFO] Scenario:",
            plan[
                "scenario_id"
            ],
        )

        print(
            "[INFO] Mode:",
            args.mode,
        )

        print(
            "[INFO] Frames:",
            len(
                frame_ids
            ),
        )

        print(
            "[INFO] FPS:",
            fps,
        )

        print(
            "[INFO] Planned actors:",
            len(
                actor_plans
            ),
        )

        print(
            "[INFO] Pair dir:",
            pair_dir,
        )

        print(
            "[INFO] Settled ego origin:",

            base.carla_transform_to_dict(
                ego0_tf
            ),
        )

        # ====================================================
        # Camera warmup
        # ====================================================
        #
        # Ego remains at frame zero.
        # Actors active at frame zero remain at frame-zero pose.
        # ====================================================

        for _ in range(
            args.warmup_frames
        ):

            carla_frame = int(
                world.tick()
            )

            try:

                base.get_image_for_carla_frame(
                    image_queue,
                    carla_frame,
                    timeout=2.0,
                )

            except TimeoutError:

                pass

        # ====================================================
        # Execute exact resolved trajectory
        # ====================================================

        for frame_idx in frame_ids:

            t_s = (
                float(
                    frame_idx
                )
                / fps
            )

            # =================================================
            # Ego
            # =================================================

            ego_row = (
                ego_frames[
                    frame_idx
                ]
            )

            ego_tf = (
                local_to_world(
                    ego0_tf,
                    ego_row,
                    world,
                    snap_to_road=False,
                )
            )

            ego.set_transform(
                ego_tf
            )

            # =================================================
            # Multi-actor lifecycle
            # =================================================

            active_rows = {}

            if (
                args.mode
                == "real_adversary"
            ):

                # ---------------------------------------------
                # Actors that should exist on THIS exact frame.
                # ---------------------------------------------

                required_actor_ids = {

                    actor_id

                    for (
                        actor_id,
                        actor_frames,
                    )
                    in (
                        actor_frames_by_id
                        .items()
                    )

                    if (
                        frame_idx
                        in actor_frames
                    )
                }

                currently_active_ids = (
                    set(
                        active_adversaries
                    )
                )

                # =============================================
                # Destroy expired actors
                # =============================================
                #
                # Example:
                #
                # last active = frame 210
                #
                # frame 210:
                #     still exists
                #
                # before frame 211 tick:
                #     destroy
                # =============================================

                for actor_id in sorted(
                    currently_active_ids
                    -
                    required_actor_ids
                ):

                    destroy_runtime_actor(
                        actor_id,
                        frame_idx,
                        t_s,
                    )

                # =============================================
                # Spawn newly active actors
                # =============================================

                missing_actor_ids = (
                    required_actor_ids
                    -
                    set(
                        active_adversaries
                    )
                )

                for actor_id in sorted(
                    missing_actor_ids
                ):

                    row = (
                        actor_frames_by_id[
                            actor_id
                        ][
                            frame_idx
                        ]
                    )

                    spawn_runtime_actor(
                        actor_id,
                        row,
                    )

                # =============================================
                # Apply exact resolved actor poses
                # =============================================

                for actor_id in sorted(
                    required_actor_ids
                ):

                    row = (
                        actor_frames_by_id[
                            actor_id
                        ][
                            frame_idx
                        ]
                    )

                    active_rows[
                        actor_id
                    ] = row

                    actor_tf = (
                        local_to_world(
                            ego0_tf,
                            row,
                            world,
                            snap_to_road=True,
                        )
                    )

                    active_adversaries[
                        actor_id
                    ].set_transform(
                        actor_tf
                    )

            # =================================================
            # Synchronized simulation tick
            # =================================================

            carla_frame = int(
                world.tick()
            )

            # -------------------------------------------------
            # Get EXACT image belonging to this CARLA frame.
            # -------------------------------------------------

            image = (
                base.get_image_for_carla_frame(
                    image_queue,
                    carla_frame,
                    timeout=5.0,
                )
            )

            # -------------------------------------------------
            # Exact camera transform associated with RGB frame.
            # -------------------------------------------------

            camera_tf = (
                image.transform
            )

            # =================================================
            # Save RGB frame
            # =================================================

            rgb = (
                base.image_to_rgb_array(
                    image
                )
            )

            bgr = (
                cv2.cvtColor(
                    rgb,
                    cv2.COLOR_RGB2BGR,
                )
            )

            writer.write(
                bgr
            )

            # =================================================
            # Ego metadata
            # =================================================

            ego_record = {

                "recorded_frame_idx":
                    int(
                        frame_idx
                    ),

                "carla_frame":
                    int(
                        carla_frame
                    ),

                "t_s":
                    float(
                        t_s
                    ),

                "ego_local_state_ego_initial":
                    local_state(
                        ego_row
                    ),

                "ego_transform":
                    base.carla_transform_to_dict(
                        ego_tf
                    ),

                "camera_transform":
                    base.carla_transform_to_dict(
                        camera_tf
                    ),

                "camera_mount": {

                    "x":
                        camera_args.camera_x,

                    "y":
                        camera_args.camera_y,

                    "z":
                        camera_args.camera_z,

                    "pitch":
                        camera_args.camera_pitch,

                    "yaw":
                        camera_args.camera_yaw,

                    "roll":
                        camera_args.camera_roll,
                },

                "camera": {

                    "width":
                        width,

                    "height":
                        height,

                    "fov":
                        fov,
                },
            }

            ego_file.write(
                json.dumps(
                    ego_record
                )
                + "\n"
            )

            # =================================================
            # Frame-centric multi-actor ground truth
            # =================================================

            if (
                ground_truth_file
                is not None
            ):

                gt_actors = []

                # ---------------------------------------------
                # One ground-truth record per active actor.
                # ---------------------------------------------

                for actor_id in sorted(
                    active_adversaries
                ):

                    carla_actor = (
                        active_adversaries[
                            actor_id
                        ]
                    )

                    row = (
                        active_rows[
                            actor_id
                        ]
                    )

                    actor_plan = (
                        actor_plans[
                            actor_id
                        ]
                    )

                    # =========================================
                    # Ground-truth 2D bbox
                    # =========================================

                    bbox = (
                        base.compute_actor_2d_bbox(

                            actor=
                                carla_actor,

                            camera_transform=
                                camera_tf,

                            width=
                                width,

                            height=
                                height,

                            fov=
                                fov,
                        )
                    )

                    # =========================================
                    # Actor GT entry
                    # =========================================

                    actor_record = {

                        "actor_id":
                            actor_id,

                        "actor_type":
                            actor_plan.get(
                                "actor_type"
                            ),

                        "role":
                            actor_plan.get(
                                "role"
                            ),

                        "asset_key":
                            actor_plan.get(
                                "asset_key"
                            ),

                        "blueprint":
                            actor_plan.get(
                                "blueprint"
                            ),

                        "planned_state":
                            local_state(
                                row
                            ),

                        "world_transform":
                            base.carla_transform_to_dict(
                                carla_actor
                                .get_transform()
                            ),

                        "bbox":
                            bbox,
                    }

                    gt_actors.append(
                        actor_record
                    )

                # ---------------------------------------------
                # One JSONL line = one exact recorded RGB frame
                # ---------------------------------------------

                ground_truth_record = {

                    "recorded_frame_idx":
                        int(
                            frame_idx
                        ),

                    "carla_frame":
                        int(
                            carla_frame
                        ),

                    "t_s":
                        float(
                            t_s
                        ),

                    "scenario":
                        plan[
                            "scenario_id"
                        ],

                    "ego_transform":
                        base.carla_transform_to_dict(
                            ego_tf
                        ),

                    "camera_transform":
                        base.carla_transform_to_dict(
                            camera_tf
                        ),

                    "actors":
                        gt_actors,
                }

                ground_truth_file.write(
                    json.dumps(
                        ground_truth_record
                    )
                    + "\n"
                )

            # =================================================
            # Progress
            # =================================================

            if (
                frame_idx
                % 30
                == 0
            ):

                print(
                    f"[INFO] frame "
                    f"{frame_idx}/"
                    f"{len(frame_ids)-1} "
                    f"active_actors="
                    f"{len(active_adversaries)}"
                )

        # ====================================================
        # Complete
        # ====================================================

        print(
            "[DONE] CARLA v2 plan "
            "recording complete."
        )

    # ========================================================
    # Cleanup
    # ========================================================

    finally:

        if writer:
            writer.release()

        if ego_file:
            ego_file.close()

        if ground_truth_file:
            ground_truth_file.close()

        # ----------------------------------------------------
        # Stop camera listener when possible.
        # ----------------------------------------------------

        try:

            if "camera" in locals():
                camera.stop()

        except Exception:
            pass

        # ----------------------------------------------------
        # Some lifecycle actors may already have been destroyed.
        # Calling destroy again is safe here because exceptions
        # are ignored.
        # ----------------------------------------------------

        for actor in reversed(
            created_actors
        ):

            try:

                actor.destroy()

            except Exception:

                pass

        # ----------------------------------------------------
        # Restore original CARLA settings.
        # ----------------------------------------------------

        try:

            world.apply_settings(
                original_settings
            )

        except Exception:

            pass


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Execute a ScenarioGenerator CARLA plan v2 "
            "with arbitrary scripted actors and "
            "actor lifecycle support."
        )
    )

    parser.add_argument(
        "--plan",
        required=True,
    )

    parser.add_argument(
        "--mode",
        required=True,

        choices=[
            "background",
            "real_adversary",
        ],
    )

    parser.add_argument(
        "--pair-name",
        required=True,
    )

    parser.add_argument(
        "--output-root",

        default=(
            "recordings/he_pairs"
        ),
    )

    parser.add_argument(
        "--host",

        default=(
            "127.0.0.1"
        ),
    )

    parser.add_argument(
        "--port",

        type=int,

        default=2000,
    )

    parser.add_argument(
        "--town",

        default=(
            "Town10HD_Opt"
        ),
    )

    parser.add_argument(
        "--spawn-index",

        type=int,

        default=10,
    )

    parser.add_argument(
        "--ego-vehicle-filter",

        default=(
            "vehicle.tesla.model3"
        ),
    )

    parser.add_argument(
        "--ego-settle-frames",

        type=int,

        default=30,
    )

    parser.add_argument(
        "--warmup-frames",

        type=int,

        default=10,
    )

    return parser.parse_args()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    run(
        parse_args()
    )