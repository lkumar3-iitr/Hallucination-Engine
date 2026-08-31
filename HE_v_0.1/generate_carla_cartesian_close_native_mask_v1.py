#!/usr/bin/env python3
"""Generate a schema-compatible close Cartesian Nissan sprite supplement."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import carla


SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_EXTERNAL_GENERATOR_DIR = Path(
    r"D:\HallucinationEngine-asset\HE_v_0.1"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--asset-class",
        choices=["vehicle", "bus", "pedestrian"],
        default="vehicle",
    )
    parser.add_argument("--actor-blueprint", default="vehicle.nissan.patrol_2021")
    parser.add_argument("--semantic-tag", default="auto")
    parser.add_argument("--color", default="0,0,255")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--generator-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the native-mask generator. Defaults to "
            "this script's directory, with the legacy local asset-tool "
            "directory retained as a compatibility fallback."
        ),
    )
    parser.add_argument("--forward-start-m", type=float, default=0.25)
    parser.add_argument("--forward-stop-m", type=float, default=7.0)
    parser.add_argument("--forward-step-m", type=float, default=0.25)
    parser.add_argument("--right-offsets-m", type=float, nargs="+", default=[3.0, 3.5, 4.0])
    parser.add_argument("--target-up-m", type=float, default=-0.9347938)
    parser.add_argument("--relative-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--all-relative-yaws",
        action="store_true",
        help="Capture the full relative-yaw circle at --yaw-step-deg.",
    )
    parser.add_argument("--yaw-step-deg", type=int, default=1)
    parser.add_argument(
        "--side-sign",
        type=int,
        choices=[-1, 1],
        default=1,
        help="+1 captures camera-right actor positions; -1 captures left.",
    )
    parser.add_argument("--image-width", type=int, default=2560)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--runtime-width", type=int, default=900)
    parser.add_argument("--runtime-fov", type=float, default=100.0)
    parser.add_argument(
        "--compact-runtime-only",
        action="store_true",
        help=(
            "Persist only runtime RGBA sprites and metadata. RGB, native "
            "instance, binary-mask, and debug PNGs are not written."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible interrupted compact or full generation.",
    )
    return parser.parse_args()


def inclusive_values(start, stop, step):
    count = int(round((stop - start) / step))
    return [round(start + index * step, 6) for index in range(count + 1)]


def update_json(path, update):
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    document.update(update)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)


def annotate_outputs(output_dir, args, forward_values, relative_yaws, capture_fov):
    csv_path = output_dir / "view_matrix.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    additions = [
        "bank_coordinate_system", "close_forward_m", "close_right_m",
        "close_target_up_m", "close_relative_yaw_deg",
        "runtime_reference_width_px", "runtime_reference_fov_deg",
    ]
    for name in additions:
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        row["bank_coordinate_system"] = "camera_cartesian_close_v1"
        row["close_forward_m"] = row["elevation_deg"]
        row["close_right_m"] = str(
            float(args.side_sign) * float(row["distance_m"])
        )
        row["close_target_up_m"] = str(float(args.target_up_m))
        row["close_relative_yaw_deg"] = row["relative_actor_yaw_deg"]
        row["runtime_reference_width_px"] = str(int(args.runtime_width))
        row["runtime_reference_fov_deg"] = str(float(args.runtime_fov))
        if args.compact_runtime_only:
            for name in (
                "rgb_path", "mask_path", "instance_path", "debug_path",
                "rgb_relpath", "mask_relpath", "instance_relpath",
                "debug_relpath",
            ):
                if name in row:
                    row[name] = ""
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    close_metadata = {
        "close_cartesian": {
            "schema_version": 1,
            "coordinate_system": "camera_cartesian_close_v1",
            "forward_values_m": forward_values,
            "right_offsets_m": [float(value) for value in args.right_offsets_m],
            "side_sign": int(args.side_sign),
            "target_up_m": float(args.target_up_m),
            "relative_yaws_deg": [float(value) for value in relative_yaws],
            "runtime_reference": {
                "width_px": int(args.runtime_width),
                "fov_deg": float(args.runtime_fov),
            },
            "capture": {
                "width_px": int(args.image_width),
                "height_px": int(args.image_height),
                "fov_deg": float(capture_fov),
                "focal_length_matches_runtime_horizontal": True,
            },
            "output_policy": {
                "compact_runtime_only": bool(args.compact_runtime_only),
                "persisted_view_artifacts": (
                    ["rgba"]
                    if args.compact_runtime_only
                    else ["rgba", "rgb", "mask", "instance", "debug"]
                ),
                "rgba_alpha_is_runtime_mask": True,
            },
        }
    }
    update_json(output_dir / "asset_metadata.json", close_metadata)
    update_json(output_dir / "generation_config.json", close_metadata)


def main():
    args = parse_args()
    forward_values = inclusive_values(
        args.forward_start_m, args.forward_stop_m, args.forward_step_m
    )
    runtime_fx = float(args.runtime_width) / (
        2.0 * math.tan(math.radians(args.runtime_fov) / 2.0)
    )
    capture_fov = math.degrees(
        2.0 * math.atan(float(args.image_width) / (2.0 * runtime_fx))
    )
    if args.all_relative_yaws:
        if args.yaw_step_deg <= 0 or 360 % args.yaw_step_deg != 0:
            raise ValueError("--yaw-step-deg must be a positive divisor of 360")
        relative_yaws = list(range(0, 360, args.yaw_step_deg))
    else:
        relative_yaws = [float(args.relative_yaw_deg) % 360.0]
    encoded_angles = sorted({
        int(round(-float(yaw))) % 360 for yaw in relative_yaws
    })

    generator_candidates = (
        [args.generator_dir]
        if args.generator_dir is not None
        else [SCRIPT_DIR, LEGACY_EXTERNAL_GENERATOR_DIR]
    )
    generator_dir = next(
        (
            candidate
            for candidate in generator_candidates
            if (
                candidate
                / "generate_carla_asset_view_matrix_native_mask_v3.py"
            ).is_file()
        ),
        None,
    )
    if generator_dir is None:
        searched = ", ".join(str(path) for path in generator_candidates)
        raise FileNotFoundError(
            "generate_carla_asset_view_matrix_native_mask_v3.py was not "
            f"found; searched: {searched}"
        )
    sys.path.insert(0, str(generator_dir))
    generator = importlib.import_module(
        "generate_carla_asset_view_matrix_native_mask_v3"
    )
    if args.compact_runtime_only:
        original_file_is_nonempty = generator.file_is_nonempty

        def compact_file_is_nonempty(path):
            path = Path(path)
            if path.parent.name in {"rgb", "mask", "instance", "debug"}:
                return True
            return original_file_is_nonempty(path)

        generator.file_is_nonempty = compact_file_is_nonempty
        generator.save_rgb = lambda *unused_args, **unused_kwargs: None
        generator.save_mask = lambda *unused_args, **unused_kwargs: None
        generator.make_debug_image = lambda *unused_args, **unused_kwargs: None
    def cartesian_camera(target_location, base_yaw_deg, distance_m, elevation_deg):
        right_m = float(args.side_sign) * float(distance_m)
        forward_m = float(elevation_deg)
        yaw_rad = math.radians(float(base_yaw_deg))
        forward_x, forward_y = math.cos(yaw_rad), math.sin(yaw_rad)
        right_x, right_y = -math.sin(yaw_rad), math.cos(yaw_rad)
        camera_location = carla.Location(
            x=float(target_location.x) - forward_m * forward_x - right_m * right_x,
            y=float(target_location.y) - forward_m * forward_y - right_m * right_y,
            z=float(target_location.z) - float(args.target_up_m),
        )
        return carla.Transform(
            camera_location,
            carla.Rotation(pitch=0.0, yaw=float(base_yaw_deg), roll=0.0),
        )

    generator.build_view_camera_transform_from_target = cartesian_camera

    sys.argv = [
        sys.argv[0],
        "--host", args.host,
        "--port", str(args.port),
        "--timeout", str(args.timeout),
        "--asset-id", args.asset_id,
        "--asset-class", args.asset_class,
        "--actor-blueprint", args.actor_blueprint,
        "--semantic-tag", args.semantic_tag,
        "--color", args.color,
        "--output-root", args.output_root,
        "--angles", *[str(value) for value in encoded_angles],
        "--distances", *[str(value) for value in args.right_offsets_m],
        "--elevations", *[str(value) for value in forward_values],
        "--target-height", "auto",
        "--image-width", str(args.image_width),
        "--image-height", str(args.image_height),
        "--fov", str(capture_fov),
        "--hide-environment-objects",
        "--skip-contact-sheets",
    ]
    if args.resume:
        sys.argv.append("--resume")
    print(
        "[CartesianClose] views:",
        len(forward_values) * len(args.right_offsets_m) * len(encoded_angles),
    )
    print("[CartesianClose] runtime-equivalent fx:", runtime_fx)
    print("[CartesianClose] capture fov:", capture_fov)
    print("[CartesianClose] asset class:", args.asset_class)
    print("[CartesianClose] compact runtime only:", args.compact_runtime_only)
    print("[CartesianClose] resume:", args.resume)
    generator.main()
    output_dir = Path(args.output_root) / args.asset_id
    annotate_outputs(
        output_dir, args, forward_values, relative_yaws, capture_fov
    )
    print("[CartesianClose] metadata annotation complete:", output_dir)


if __name__ == "__main__":
    main()
