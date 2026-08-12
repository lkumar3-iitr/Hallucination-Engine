import argparse
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import cv2
import carla

import record_carla_he_pair as base

def resolve_carla_blueprint(plan_blueprint):
    """
    Resolve ScenarioGenerator logical vehicle names into concrete
    CARLA blueprint IDs.

    ScenarioGenerator should not need to know every simulator-specific
    asset name.
    """

    aliases = {
        "vehicle.sedan.generic": "vehicle.audi.tt",
    }

    return aliases.get(
        plan_blueprint,
        plan_blueprint,
    )

def load_plan(path):
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    if plan.get("coordinate_frame") != "ego_initial":
        raise ValueError("Expected coordinate_frame='ego_initial'.")

    # v1 executor: validate one adversary first.
    if len(plan.get("actors", [])) != 1:
        raise ValueError(
            "v1 plan executor currently supports exactly one adversary."
        )

    return plan


def index_frames(rows):
    return {
        int(row["frame_idx"]): row
        for row in rows
    }


def local_state(row):
    return {
        "x_m": float(row["local_x_m"]),
        "y_m": float(row.get("local_y_m", 0.0)),
        "z_m": float(row["local_z_m"]),
        "yaw_deg": float(row["local_yaw_deg"]),
    }


def local_to_world(
    ego0_tf,
    row,
    world,
    snap_to_road,
):
    tf = base.local_ego_initial_to_world_transform(
        ego0_tf=ego0_tf,
        local_x_m=float(row["local_x_m"]),
        local_y_m=float(
            row.get("local_y_m", 0.0)
        ),
        local_z_m=float(row["local_z_m"]),
        local_yaw_deg=float(
            row["local_yaw_deg"]
        ),
        world=world,
        snap_to_road=snap_to_road,
        z_lift=0.05,
    )

    # For ego motion, preserve the settled road pitch/roll.
    if not snap_to_road:
        tf.rotation.pitch = float(
            ego0_tf.rotation.pitch
        )
        tf.rotation.roll = float(
            ego0_tf.rotation.roll
        )

    return tf


def run(args):
    plan = load_plan(args.plan)

    fps = float(plan["fps"])

    ego_frames = index_frames(
        plan["ego_frames"]
    )

    actor_plan = plan["actors"][0]

    actor_frames = index_frames(
        actor_plan["frames"]
    )

    frame_ids = sorted(
        ego_frames
    )

    if frame_ids != list(
        range(len(frame_ids))
    ):
        raise ValueError(
            "ego_frames must be contiguous from frame 0."
        )

    for frame_idx in frame_ids:
        if frame_idx not in actor_frames:
            raise ValueError(
                f"Actor missing frame {frame_idx}."
            )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera_cfg = plan["camera"]

    width = int(
        camera_cfg["image_width"]
    )

    height = int(
        camera_cfg["image_height"]
    )

    fov = float(
        camera_cfg["fov"]
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    pair_dir = (
        Path(args.output_root)
        / args.pair_name
    )

    pair_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode == "background":
        video_path = (
            pair_dir / "background.mp4"
        )

        ego_pose_path = (
            pair_dir
            / "background_ego_pose.jsonl"
        )

        adv_pose_path = None
        bbox_path = None

    else:
        video_path = (
            pair_dir
            / "real_adversary.mp4"
        )

        ego_pose_path = (
            pair_dir
            / "real_ego_pose.jsonl"
        )

        adv_pose_path = (
            pair_dir
            / "real_adversary_pose.jsonl"
        )

        bbox_path = (
            pair_dir
            / "real_bbox.jsonl"
        )

    # --------------------------------------------------------
    # CARLA
    # --------------------------------------------------------

    client = carla.Client(
        args.host,
        args.port,
    )

    client.set_timeout(20.0)

    if args.town:
        world = client.load_world(
            args.town
        )
    else:
        world = client.get_world()

    original_settings = (
        world.get_settings()
    )

    actors = []

    writer = None
    ego_file = None
    adv_file = None
    bbox_file = None

    try:
        settings = world.get_settings()

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            1.0 / fps
        )

        world.apply_settings(
            settings
        )

        # ----------------------------------------------------
        # Spawn ego
        # ----------------------------------------------------

        spawn_points = (
            world.get_map()
            .get_spawn_points()
        )

        if not (
            0
            <= args.spawn_index
            < len(spawn_points)
        ):
            raise ValueError(
                f"spawn-index {args.spawn_index} "
                f"out of range "
                f"0..{len(spawn_points)-1}"
            )

        ego_bp = base.get_blueprint(
            world,
            args.ego_vehicle_filter,
        )

        ego = base.spawn_actor_safe(
            world,
            ego_bp,
            spawn_points[
                args.spawn_index
            ],
        )

        actors.append(
            ego
        )

        # ----------------------------------------------------
        # Settle ego physically before freezing it.
        # This defines our ego-initial reference transform.
        # ----------------------------------------------------

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

        # Put ego at exact resolved frame 0.
        ego.set_transform(
            local_to_world(
                ego0_tf,
                ego_frames[0],
                world,
                snap_to_road=False,
            )
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        camera_args = SimpleNamespace(
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

        camera = base.make_rgb_camera(
            world,
            ego,
            camera_args,
        )

        actors.append(
            camera
        )

        image_queue = queue.Queue()

        camera.listen(
            image_queue.put
        )

        # ----------------------------------------------------
        # Spawn adversary only for real recording
        # ----------------------------------------------------

        adversary = None

        if args.mode == "real_adversary":
            carla_blueprint = resolve_carla_blueprint(
                actor_plan["blueprint"]
            )

            print(
                "[INFO] Actor blueprint:",
                actor_plan["blueprint"],
                "->",
                carla_blueprint,
            )

            adv_bp = base.get_blueprint(
                world,
                carla_blueprint,
            )

            adv_tf0 = local_to_world(
                ego0_tf,
                actor_frames[0],
                world,
                snap_to_road=True,
            )

            adversary = (
                base.spawn_actor_safe(
                    world,
                    adv_bp,
                    adv_tf0,
                )
            )

            adversary.set_simulate_physics(
                False
            )

            actors.append(
                adversary
            )

        # ----------------------------------------------------
        # Video
        # ----------------------------------------------------

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Could not open video writer: "
                f"{video_path}"
            )

        # ----------------------------------------------------
        # Logs
        # ----------------------------------------------------

        ego_file = open(
            ego_pose_path,
            "w",
            encoding="utf-8",
        )

        if adv_pose_path:
            adv_file = open(
                adv_pose_path,
                "w",
                encoding="utf-8",
            )

        if bbox_path:
            bbox_file = open(
                bbox_path,
                "w",
                encoding="utf-8",
            )

        print(
            "[INFO] Plan:",
            args.plan,
        )

        print(
            "[INFO] Scenario:",
            plan["scenario_id"],
        )

        print(
            "[INFO] Mode:",
            args.mode,
        )

        print(
            "[INFO] Frames:",
            len(frame_ids),
        )

        print(
            "[INFO] FPS:",
            fps,
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

        # ----------------------------------------------------
        # Camera warmup.
        #
        # Keep all actors at frame-zero state.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Execute exact resolved trajectory
        # ----------------------------------------------------

        for frame_idx in frame_ids:
            t_s = (
                float(frame_idx)
                / fps
            )

            # ---------------- Ego ----------------

            ego_row = (
                ego_frames[
                    frame_idx
                ]
            )

            ego_tf = local_to_world(
                ego0_tf,
                ego_row,
                world,
                snap_to_road=False,
            )

            ego.set_transform(
                ego_tf
            )

            # -------------- Adversary ------------

            adv_row = None

            if adversary is not None:
                adv_row = (
                    actor_frames[
                        frame_idx
                    ]
                )

                adv_tf = local_to_world(
                    ego0_tf,
                    adv_row,
                    world,
                    snap_to_road=True,
                )

                adversary.set_transform(
                    adv_tf
                )

            # ------------------------------------------------
            # Tick once and fetch the RGB image belonging to
            # EXACTLY this CARLA frame.
            # ------------------------------------------------

            carla_frame = int(
                world.tick()
            )

            image = (
                base.get_image_for_carla_frame(
                    image_queue,
                    carla_frame,
                    timeout=5.0,
                )
            )

            # Exact sensor transform associated
            # with this RGB image.
            camera_tf = (
                image.transform
            )

            rgb = (
                base.image_to_rgb_array(
                    image
                )
            )

            bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )

            writer.write(
                bgr
            )

            # ------------------------------------------------
            # Ego metadata
            # ------------------------------------------------

            ego_record = {
                "recorded_frame_idx":
                    int(frame_idx),

                "carla_frame":
                    int(carla_frame),

                "t_s":
                    float(t_s),

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
                    "width": width,
                    "height": height,
                    "fov": fov,
                },
            }

            ego_file.write(
                json.dumps(
                    ego_record
                )
                + "\n"
            )

            # ------------------------------------------------
            # Adversary metadata + bbox
            # ------------------------------------------------

            if adversary is not None:
                adv_record = {
                    "recorded_frame_idx":
                        int(frame_idx),

                    "carla_frame":
                        int(carla_frame),

                    "t_s":
                        float(t_s),

                    "adversary_id":
                        actor_plan[
                            "actor_id"
                        ],

                    "scenario":
                        plan[
                            "scenario_id"
                        ],

                    "local_state_ego_initial":
                        local_state(
                            adv_row
                        ),

                    "adversary_transform":
                        base.carla_transform_to_dict(
                            adversary.get_transform()
                        ),
                }

                adv_file.write(
                    json.dumps(
                        adv_record
                    )
                    + "\n"
                )

                bbox = (
                    base.compute_actor_2d_bbox(
                        actor=adversary,

                        camera_transform=
                            camera_tf,

                        width=width,
                        height=height,
                        fov=fov,
                    )
                )

                bbox_record = {
                    "recorded_frame_idx":
                        int(frame_idx),

                    "carla_frame":
                        int(carla_frame),

                    "t_s":
                        float(t_s),

                    "adversary_id":
                        actor_plan[
                            "actor_id"
                        ],

                    "bbox":
                        bbox,
                }

                bbox_file.write(
                    json.dumps(
                        bbox_record
                    )
                    + "\n"
                )

            if frame_idx % 30 == 0:
                print(
                    f"[INFO] frame "
                    f"{frame_idx}/"
                    f"{len(frame_ids)-1}"
                )

        print(
            "[DONE] CARLA plan recording complete."
        )

    finally:
        if writer:
            writer.release()

        if ego_file:
            ego_file.close()

        if adv_file:
            adv_file.close()

        if bbox_file:
            bbox_file.close()

        for actor in reversed(
            actors
        ):
            try:
                actor.destroy()
            except Exception:
                pass

        try:
            world.apply_settings(
                original_settings
            )
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser()

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
        default="recordings/he_pairs",
    )

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
        "--ego-vehicle-filter",
        default="vehicle.tesla.model3",
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


if __name__ == "__main__":
    run(
        parse_args()
    )