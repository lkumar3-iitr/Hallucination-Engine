from __future__ import annotations

import argparse
import inspect

from dataclasses import fields, is_dataclass
from pathlib import Path

import cv2
import numpy as np
import carla


from scenario_execution_runtime_v1 import (
    ExecutionWorldOrigin,
    load_execution_runtime,
)

from he_asset_registry_v1 import (
    DEFAULT_MANIFEST,
)

from he_multi_actor_compositor_v1 import (
    HEMultiActorCompositorV1,
    print_composite_summary,
)


THIS_FILE = Path(__file__).resolve()

DEFAULT_OUTPUT_DIR = (
    THIS_FILE.parent
    / "outputs"
    / "multi_actor_smoke_v1"
)


# ============================================================
# Generic inspection helpers
# ============================================================
def print_dataclass_or_object(
    title,
    obj,
):
    print()
    print(title)
    print("-" * 78)

    if obj is None:
        print("None")
        return

    if is_dataclass(obj):
        for field in fields(obj):
            value = getattr(
                obj,
                field.name,
            )
            print(
                f"{field.name}: {value}"
            )
        return

    try:
        values = vars(obj)
    except TypeError:
        print(repr(obj))
        return

    for key, value in values.items():
        print(
            f"{key}: {value}"
        )
def load_runtime_robust(
    resolved_path,
    asset_root,
    origin,
    manifest_path,
):
    print()
    print(
        "[load_execution_runtime signature]",
        inspect.signature(
            load_execution_runtime
        ),
    )

    print(
        "[resolved_json]",
        resolved_path,
    )

    print(
        "[asset_root]",
        asset_root,
    )

    print(
        "[origin]",
        origin,
    )

    print(
        "[manifest]",
        manifest_path,
    )

    runtime = load_execution_runtime(
        resolved_json=str(
            resolved_path
        ),
        asset_root=Path(
            asset_root
        ),
        origin=origin,
        manifest_path=Path(
            manifest_path
        ),
    )

    return runtime


# ============================================================
# Runtime origin
# ============================================================

def find_runtime_origin(
    runtime,
):
    # --------------------------------------------------------
    # First try obvious names.
    # --------------------------------------------------------

    for name in (
        "origin",
        "world_origin",
        "execution_origin",
    ):
        value = getattr(
            runtime,
            name,
            None,
        )

        if isinstance(
            value,
            ExecutionWorldOrigin,
        ):
            return value

    # --------------------------------------------------------
    # Otherwise inspect runtime attributes by type.
    # --------------------------------------------------------

    for value in vars(
        runtime
    ).values():

        if isinstance(
            value,
            ExecutionWorldOrigin,
        ):
            return value

    return None

def get_any(
    obj,
    names,
    default=None,
):
    if obj is None:
        return default

    for name in names:
        if hasattr(obj, name):
            return getattr(
                obj,
                name,
            )

    return default
def origin_value(
    origin,
    names,
    default,
):
    if origin is None:
        return float(
            default
        )

    value = get_any(
        origin,
        names,
        default,
    )

    return float(
        value
    )


# ============================================================
# NEAT-front-like debug camera
# ============================================================

def build_test_camera(
    runtime,
):
    """
    Use a NEAT-front-like camera:

        400 x 300
        FOV 100
        x = +1.3 m
        z = +2.3 m
        yaw = 0 relative to ego origin

    This is only a compositor smoke test.
    """

    origin = find_runtime_origin(
        runtime
    )

    print_dataclass_or_object(
        "[runtime origin]",
        origin,
    ) if origin is not None else print(
        "[runtime origin] NOT FOUND - "
        "using zero world origin"
    )

    origin_x = origin_value(
        origin,
        (
            "x",
            "world_x",
            "x_m",
        ),
        0.0,
    )

    origin_y = origin_value(
        origin,
        (
            "y",
            "world_y",
            "y_m",
        ),
        0.0,
    )

    origin_z = origin_value(
        origin,
        (
            "z",
            "world_z",
            "z_m",
        ),
        0.0,
    )

    origin_yaw = origin_value(
        origin,
        (
            "yaw_deg",
            "yaw",
            "world_yaw_deg",
        ),
        0.0,
    )

    yaw_rad = np.deg2rad(
        origin_yaw
    )

    forward_x = float(
        np.cos(
            yaw_rad
        )
    )

    forward_y = float(
        np.sin(
            yaw_rad
        )
    )

    camera_forward_offset = 1.3

    camera_x = (
        origin_x
        +
        forward_x
        * camera_forward_offset
    )

    camera_y = (
        origin_y
        +
        forward_y
        * camera_forward_offset
    )

    camera_z = (
        origin_z
        +
        2.3
    )

    camera_tf = carla.Transform(
        carla.Location(
            x=float(
                camera_x
            ),
            y=float(
                camera_y
            ),
            z=float(
                camera_z
            ),
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=float(
                origin_yaw
            ),
            roll=0.0,
        ),
    )

    print()
    print(
        "[test camera]"
    )

    print(
        "  location:",
        "x=%.3f y=%.3f z=%.3f"
        %
        (
            camera_x,
            camera_y,
            camera_z,
        ),
    )

    print(
        "  yaw:",
        "%.3f"
        %
        origin_yaw,
    )

    return camera_tf


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resolved",
        required=True,
        help=(
            "Resolved ScenarioGenerator v2 JSON."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )
    parser.add_argument(
        "--asset-root",
        default=(
            r"D:\HallucinationEngine-asset"
            r"\HE_v_0.1"
            r"\assets"
            r"\sprite_bank_native_production"
        ),
    )

    parser.add_argument(
        "--origin-x",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--origin-y",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--origin-z",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--origin-yaw",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=100.0,
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
        "HE MULTI-ACTOR SMOKE TEST V1"
    )
    print("=" * 78)

    print(
        "[resolved]",
        resolved_path,
    )

    print(
        "[manifest]",
        manifest_path,
    )

    print(
        "[frame]",
        args.frame,
    )

    # ========================================================
    # Load execution runtime
    # ========================================================
    asset_root = Path(
        args.asset_root
    ).resolve()

    origin = ExecutionWorldOrigin(
        x_m=float(
            args.origin_x
        ),
        y_m=float(
            args.origin_y
        ),
        z_m=float(
            args.origin_z
        ),
        yaw_deg=float(
            args.origin_yaw
        ),
    )
    runtime = load_runtime_robust(
        resolved_path=resolved_path,
        asset_root=asset_root,
        origin=origin,
        manifest_path=manifest_path,
    )

    print()
    print(
        "[runtime type]",
        type(
            runtime
        ).__name__,
    )

    # ========================================================
    # Get active actors
    # ========================================================

    active_actors = runtime.active_actors(
        args.frame
    )

    active_actors = list(
        active_actors
    )

    print()
    print(
        "[active actors]",
        len(
            active_actors
        ),
    )

    if not active_actors:
        raise RuntimeError(
            f"No active actors at frame "
            f"{args.frame}."
        )

    # ========================================================
    # Inspect exact execution actor state
    # ========================================================

    for index, actor in enumerate(
        active_actors
    ):
        print_dataclass_or_object(
            (
                f"[actor {index}] "
                f"{getattr(actor, 'actor_id', '?')}"
            ),
            actor,
        )

    # ========================================================
    # Test camera
    # ========================================================

    camera_tf = build_test_camera(
        runtime
    )

    # ========================================================
    # Neutral background
    #
    # Gray is easier than black for visually checking alpha.
    # ========================================================

    base_rgb = np.full(
        (
            int(
                args.height
            ),
            int(
                args.width
            ),
            3,
        ),
        180,
        dtype=np.uint8,
    )

    # ========================================================
    # Multi-actor compositor
    # ========================================================

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

    # ========================================================
    # Summary
    # ========================================================

    print()
    print_composite_summary(
        result
    )

    # ========================================================
    # Save RGB -> BGR
    # ========================================================

    output_path = (
        output_dir
        /
        (
            f"frame_"
            f"{args.frame:04d}"
            f".png"
        )
    )

    ok = cv2.imwrite(
        str(
            output_path
        ),
        result.rgb[
            :,
            :,
            ::-1
        ],
    )

    if not ok:
        raise RuntimeError(
            f"Could not save "
            f"{output_path}"
        )

    print()
    print("=" * 78)
    print(
        "SMOKE TEST COMPLETE"
    )
    print("=" * 78)

    print(
        "[saved]",
        output_path,
    )

    print()
    print(
        "Expected ordering:"
    )

    print(
        "  render_order 0 = farthest actor"
    )

    print(
        "  render_order N = nearest actor"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()