import argparse
import json
from pathlib import Path


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"No rows found in: {path}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build an HE compositor scenario directly from the planned "
            "ego-initial adversary states saved by record_carla_he_pair.py."
        )
    )

    parser.add_argument(
        "--pair-name",
        required=True,
        help="Pair folder name under recordings/he_pairs/",
    )
    parser.add_argument(
        "--sprite-mode",
        choices=[
            "legacy",
            "view_matrix",
        ],
        default="legacy",
        help=(
            "Sprite selection mode. "
            "'legacy' uses the old angle-only sprite bank; "
            "'view_matrix' uses one or more view_matrix.csv files."
        ),
    )

    parser.add_argument(
        "--view-matrix-csv",
        action="append",
        default=[],
        help=(
            "Path to a view_matrix.csv file. "
            "May be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--target-height-m",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--vertical-mode",
        choices=[
            "state_y",
            "level_ground",
        ],
        default="level_ground",
    )

    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=1.6,
    )
    parser.add_argument(
        "--scenario-id",
        default=None,
        help="Optional scenario_id written into the HE JSON.",
    )

    parser.add_argument(
        "--output-json",
        default=None,
        help="Output scenario JSON path.",
    )

    parser.add_argument(
        "--sprite-root",
        default="assets/sprite_bank_rgba/vehicle_blue_sedan",
    )

    parser.add_argument(
        "--placement-lookup",
        default=(
            "assets/placement_lookup/"
            "heplacement_v2_lookup_camera_relative_x4_0_100_z05_yaw1.npz"
        ),
    )

    parser.add_argument(
        "--min-render-depth",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    pair_name = args.pair_name

    pair_dir = Path("recordings") / "he_pairs" / pair_name

    adversary_pose_path = pair_dir / "real_adversary_pose.jsonl"
    background_video_path = pair_dir / "background.mp4"
    background_ego_pose_path = pair_dir / "background_ego_pose.jsonl"

    if not adversary_pose_path.exists():
        raise FileNotFoundError(adversary_pose_path)

    if not background_video_path.exists():
        raise FileNotFoundError(background_video_path)

    if not background_ego_pose_path.exists():
        raise FileNotFoundError(background_ego_pose_path)

    rows = load_jsonl(adversary_pose_path)

    # ------------------------------------------------------------
    # Build one HE keyframe from each planned CARLA pair state.
    #
    # IMPORTANT:
    # We intentionally use local_state_ego_initial here.
    #
    # We do NOT use adversary_transform because that is CARLA
    # world-state output. Both CARLA and HE should be driven from
    # the same requested/planned trajectory.
    # ------------------------------------------------------------

    keyframes = []

    for row in rows:
        frame_idx = int(row["recorded_frame_idx"])
        t_s = float(row.get("t_s", frame_idx / 30.0))

        state = row["local_state_ego_initial"]

        keyframes.append(
            {
                "frame_idx": frame_idx,
                "t_s": t_s,
                "x_m": float(state["x_m"]),
                "y_m": float(state.get("y_m", 0.0)),
                "z_m": float(state["z_m"]),
                "yaw_deg": float(state.get("yaw_deg", 0.0)),
            }
        )

    keyframes.sort(key=lambda x: x["frame_idx"])

    first_state = keyframes[0]

    frame_count = len(keyframes)

    # Infer FPS from t_s when possible.
    if frame_count >= 2:
        dt = keyframes[1]["t_s"] - keyframes[0]["t_s"]

        if dt > 1e-9:
            fps = 1.0 / dt
        else:
            fps = 30.0
    else:
        fps = 30.0

    scenario_id = (
        args.scenario_id
        if args.scenario_id
        else f"{pair_name}_he"
    )

    # Using scenario_id keeps different render configurations
    # for the same CARLA pair isolated from one another.
    output_dir = (
        Path("he_outputs")
        / scenario_id
    )

    # ------------------------------------------------------------
    # Sprite-bank configuration
    # ------------------------------------------------------------

    if args.sprite_mode == "view_matrix":

        if not args.view_matrix_csv:
            raise RuntimeError(
                "--sprite-mode view_matrix requires at least one "
                "--view-matrix-csv"
            )

        sprite_bank_config = {
            "mode":
                "view_matrix",

            "view_matrix_csvs":
                [
                    str(Path(p))
                    for p in args.view_matrix_csv
                ],

            "target_height_m":
                float(
                    args.target_height_m
                ),

            "vertical_mode":
                str(
                    args.vertical_mode
                ),

            "camera_height_m":
                float(
                    args.camera_height_m
                ),
        }

    else:

        sprite_bank_config = {
            "root":
                args.sprite_root,

            "rgba_dir":
                "rgba",

            "angle_format":
                "angle_{angle:03d}_rgba.png",

            "angle_convention":
                {
                    "0":
                        "rear_view",

                    "90":
                        "side_view",

                    "180":
                        "front_view",

                    "270":
                        "opposite_side_view",
                },
        }

    scenario = {
        "schema_version": "he_scenario_v1",

        "scenario_id": scenario_id,

        "description": (
            "HE scenario generated from the planned ego-initial "
            f"trajectory of CARLA/HE pair {pair_name}."
        ),

        "input": {
            "video_path": str(
                background_video_path
            ).replace("\\", "/"),

            "ego_pose_path": str(
                background_ego_pose_path
            ).replace("\\", "/"),
        },

        "output": {
            "output_dir": str(output_dir).replace("\\", "/"),
            "save_frames": False,
            "save_masks": True,
            "save_metadata": True,
            "output_video_name":
                f"{scenario_id}.mp4",
        },

        "coordinate_mode": {
            "actor_state_frame": "ego_initial",
            "runtime_transform": "ego_pose_jsonl",
        },

        "visibility": {
            "min_render_depth_m": float(args.min_render_depth),
        },

        "placement_lookup_npz": args.placement_lookup,

        "placement_runtime_correction": {
            "enabled": False,
        },

        "camera": {
            "model": "heplacement_model_camera",

            "image_width": 1280,
            "image_height": 720,

            "fov": 90.0,
            "yaw_deg": 0.0,

            "intrinsics": {
                "fx": 640.0,
                "fy": 640.0,
                "cx": 640.0,
                "cy": 360.0,
            },

            "image_coordinate_system": {
                "x": "right",
                "y": "down",
                "origin": "top_left",
            },
        },

        "timeline": {
            "start_frame": 0,
            "end_frame": frame_count - 1,
            "fps": float(fps),
        },

        "sprite_bank":
            sprite_bank_config,

        "adversaries": [
            {
                "id": "adv_001",
                "type": "vehicle",
                "enabled": True,

                "size": {
                    "length_m": 4.5,
                    "width_m": 1.8,
                    "height_m": 1.5,
                },

                # Required by the general HE schema even though the
                # keyframed trajectory below is what drives the actor.
                "initial_state": {
                    "x_m": first_state["x_m"],
                    "y_m": first_state["y_m"],
                    "z_m": first_state["z_m"],
                    "yaw_deg": first_state["yaw_deg"],
                },

                "motion": {
                    "model": "keyframed_trajectory",
                    "keyframes": keyframes,
                },

                "rendering": {
                    "alpha": 1.0,

                    # The saved trajectory already contains the same
                    # yaw used to place the CARLA adversary.
                    "angle_mode": "viewpoint",
                    "angle_smoothing": False,
                    "shadow": False,
                    "refiner": False,
                },
            }
        ],
    }

    if args.output_json:
        output_json = Path(args.output_json)
    else:
        output_json = (
            Path("configs")
            / "scenarios"
            / f"{pair_name}_he.json"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            scenario,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("HE scenario generated")
    print("=" * 72)
    print("Pair        :", pair_name)
    print("Scenario ID :", scenario_id)
    print("Sprite mode :", args.sprite_mode)
    print("Frames      :", frame_count)
    print("FPS         :", fps)
    print("Output JSON :", output_json)
    print("Output dir  :", output_dir)

    if args.sprite_mode == "view_matrix":
        print(
            "View CSVs   :",
            len(args.view_matrix_csv),
        )

    print("=" * 72)


if __name__ == "__main__":
    main()