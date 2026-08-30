"""
generic_he_geometry_probe_v1.py

Zero-behavior-change geometry probe for the generic HE closed-loop runner.

It wraps:
    generic_he_closed_loop_runner_v1.py

and leaves unchanged:
    - CARLA stepping
    - ScenarioGenerator runtime
    - HE renderer
    - sprite selection
    - TCP / NEAT adapters
    - model inputs
    - model controls

The only addition is a second CSV containing one row per:
    scenario frame x native camera x active actor

This is intended to diagnose close-range rendering behavior before changing
the HE projection formula.

Example (TCP)
-------------
Use the same arguments as generic_he_closed_loop_runner_v1.py, but invoke
this file instead.

python driving_models\\common\\generic_he_geometry_probe_v1.py ^
  --model tcp ^
  --resolved <resolved_v2.json> ^
  --asset-root <asset-root> ^
  --geometry-print-near-m 12

Example (NEAT)
--------------
The same probe works unchanged with:
    --model neat

Output
------
Alongside the generic runner CSV:

    <scenario_id>_<model>_he_geometry.csv

The CSV records:
    - camera-relative x/y/z/yaw
    - raw bearing
    - viewpoint query + selected sprite angle
    - actor-center / support / nearest / farthest depths
    - query + selected distance/elevation
    - projected box width/height/bottom_y
    - selected sprite path

No rendering equations are modified by this probe.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import sys
from pathlib import Path

import generic_he_closed_loop_runner_v1 as base


# ============================================================
# Probe configuration populated by main()
# ============================================================

_CAMERA_NAMES = ()
_START_FRAME = 0

_GEOMETRY_CSV_PATH = None
_GEOMETRY_PRINT_NEAR_M = 12.0
_GEOMETRY_PRINT_EVERY = 1

_GEOMETRY_FP = None
_GEOMETRY_WRITER = None


GEOMETRY_FIELDS = [
    "scenario_frame",
    "camera_name",
    "render_call_index",

    "actor_id",
    "asset_key",
    "carla_blueprint",
    "rendered",

    "rel_x_m",
    "rel_y_m",
    "rel_z_m",
    "rel_yaw_deg",

    "bearing_deg",
    "geometric_viewpoint_deg_pre_bank",
    "query_viewpoint_deg",
    "selected_angle_deg",
    "angle_error_deg",

    "distance_forward_m",
    "distance_euclidean_m",

    "center_depth_m",
    "support_depth_m",
    "nearest_depth_m",
    "farthest_depth_m",

    "support_local_x_m",
    "support_local_y_m",

    "query_distance_m",
    "selected_distance_m",
    "query_elevation_deg",
    "selected_elevation_deg",

    "cx",
    "bottom_y",
    "render_bottom_y",
    "box_width",
    "box_height",

    "sprite_width",
    "sprite_height",

    "geometry_mode",
    "sprite_path",
]


# ============================================================
# CLI helpers
# ============================================================

def read_cli_value(
    argv,
    flag,
    default=None,
):
    prefix = (
        str(flag)
        + "="
    )

    for index, token in enumerate(
        argv
    ):
        if token == flag:
            if (
                index + 1
                <
                len(argv)
            ):
                return argv[
                    index + 1
                ]

            return default

        if token.startswith(
            prefix
        ):
            return token.split(
                "=",
                1,
            )[1]

    return default


def model_camera_names(
    model_name,
):
    name = str(
        model_name
    ).strip().lower()

    if name == "tcp":
        return (
            "front",
        )

    if name == "neat":
        return (
            "front",
            "left",
            "right",
        )

    raise KeyError(
        f"Unsupported model for geometry probe: {name}"
    )


# ============================================================
# Numeric helpers
# ============================================================

def normalize_angle_360(
    angle_deg,
):
    return (
        float(
            angle_deg
        )
        % 360.0
    )


def optional_value(
    mapping,
    key,
):
    if not mapping:
        return ""

    value = mapping.get(
        key,
        None,
    )

    if value is None:
        return ""

    return value


def fmt_float(
    value,
    digits=2,
):
    if (
        value is None
        or
        value == ""
    ):
        return "NA"

    try:
        number = float(
            value
        )
    except Exception:
        return str(
            value
        )

    if not math.isfinite(
        number
    ):
        return str(
            number
        )

    return (
        f"{number:.{digits}f}"
    )


# ============================================================
# Geometry CSV
# ============================================================

def close_geometry_csv():
    global _GEOMETRY_FP

    if (
        _GEOMETRY_FP
        is not None
    ):
        try:
            _GEOMETRY_FP.flush()
            _GEOMETRY_FP.close()
        finally:
            _GEOMETRY_FP = None


atexit.register(
    close_geometry_csv
)


def geometry_writer():
    global _GEOMETRY_FP
    global _GEOMETRY_WRITER

    if (
        _GEOMETRY_WRITER
        is not None
    ):
        return _GEOMETRY_WRITER

    if (
        _GEOMETRY_CSV_PATH
        is None
    ):
        raise RuntimeError(
            "Geometry CSV path was not initialized."
        )

    path = Path(
        _GEOMETRY_CSV_PATH
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _GEOMETRY_FP = open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    )

    _GEOMETRY_WRITER = (
        csv.DictWriter(
            _GEOMETRY_FP,
            fieldnames=
                GEOMETRY_FIELDS,
        )
    )

    _GEOMETRY_WRITER.writeheader()

    print(
        "[HE geometry CSV]",
        path,
    )

    return _GEOMETRY_WRITER


# ============================================================
# Row construction
# ============================================================

def geometry_row_from_actor_result(
    actor_result,
    scenario_frame,
    camera_name,
    render_call_index,
):
    meta = (
        actor_result.he_metadata
        or {}
    )

    box = (
        meta.get(
            "box",
            {}
        )
        or {}
    )

    rel_x = float(
        actor_result.rel_x_m
    )

    rel_z = float(
        actor_result.rel_z_m
    )

    rel_yaw = float(
        actor_result.rel_yaw_deg
    )

    bearing_deg = math.degrees(
        math.atan2(
            rel_x,
            rel_z,
        )
    )

    # This is the physical camera-to-actor viewpoint before the
    # production sprite-bank convention correction (+180 deg).
    viewpoint_pre_bank = (
        normalize_angle_360(
            bearing_deg
            -
            rel_yaw
        )
    )

    return {
        "scenario_frame":
            int(
                scenario_frame
            ),

        "camera_name":
            str(
                camera_name
            ),

        "render_call_index":
            int(
                render_call_index
            ),

        "actor_id":
            actor_result.actor_id,

        "asset_key":
            actor_result.asset_key,

        "carla_blueprint":
            actor_result.carla_blueprint,

        "rendered":
            int(
                bool(
                    actor_result.rendered
                )
            ),

        "rel_x_m":
            rel_x,

        "rel_y_m":
            float(
                actor_result.rel_y_m
            ),

        "rel_z_m":
            rel_z,

        "rel_yaw_deg":
            rel_yaw,

        "bearing_deg":
            float(
                bearing_deg
            ),

        "geometric_viewpoint_deg_pre_bank":
            float(
                viewpoint_pre_bank
            ),

        "query_viewpoint_deg":
            optional_value(
                meta,
                "viewpoint_angle_deg",
            ),

        "selected_angle_deg":
            optional_value(
                meta,
                "selected_angle",
            ),

        "angle_error_deg":
            optional_value(
                meta,
                "angle_error_deg",
            ),

        "distance_forward_m":
            float(
                actor_result.distance_forward_m
            ),

        "distance_euclidean_m":
            float(
                actor_result.distance_euclidean_m
            ),

        # project_virtual_actor() currently stores actor-center
        # depth as depth_m.
        "center_depth_m":
            optional_value(
                box,
                "depth_m",
            ),

        "support_depth_m":
            optional_value(
                box,
                "support_depth_m",
            ),

        "nearest_depth_m":
            optional_value(
                box,
                "nearest_depth_m",
            ),

        "farthest_depth_m":
            optional_value(
                box,
                "farthest_depth_m",
            ),

        "support_local_x_m":
            optional_value(
                box,
                "support_local_x_m",
            ),

        "support_local_y_m":
            optional_value(
                box,
                "support_local_y_m",
            ),

        "query_distance_m":
            optional_value(
                meta,
                "query_distance_m",
            ),

        "selected_distance_m":
            optional_value(
                meta,
                "selected_distance_m",
            ),

        "query_elevation_deg":
            optional_value(
                meta,
                "query_elevation_deg",
            ),

        "selected_elevation_deg":
            optional_value(
                meta,
                "selected_elevation_deg",
            ),

        "cx":
            optional_value(
                box,
                "cx",
            ),

        "bottom_y":
            optional_value(
                box,
                "bottom_y",
            ),

        "render_bottom_y":
            optional_value(
                meta,
                "render_bottom_y",
            ),

        "box_width":
            optional_value(
                box,
                "box_width",
            ),

        "box_height":
            optional_value(
                box,
                "box_height",
            ),

        "sprite_width":
            optional_value(
                meta,
                "sprite_width",
            ),

        "sprite_height":
            optional_value(
                meta,
                "sprite_height",
            ),

        "geometry_mode":
            optional_value(
                box,
                "geometry_mode",
            ),

        "sprite_path":
            optional_value(
                meta,
                "sprite_path",
            ),
    }


def maybe_print_near_geometry(
    row,
):
    try:
        z_m = float(
            row[
                "rel_z_m"
            ]
        )
    except Exception:
        return

    if (
        z_m <= 0.0
        or
        z_m
        >
        float(
            _GEOMETRY_PRINT_NEAR_M
        )
    ):
        return

    frame_idx = int(
        row[
            "scenario_frame"
        ]
    )

    if (
        int(
            _GEOMETRY_PRINT_EVERY
        )
        >
        1
        and
        frame_idx
        %
        int(
            _GEOMETRY_PRINT_EVERY
        )
        !=
        0
    ):
        return

    print(
        "[HE-GEO]"
        f" f={frame_idx:04d}"
        f" cam={row['camera_name']}"
        f" actor={row['actor_id']}"
        f" x={fmt_float(row['rel_x_m'])}"
        f" z={fmt_float(row['rel_z_m'])}"
        f" yaw={fmt_float(row['rel_yaw_deg'], 1)}"
        f" bearing={fmt_float(row['bearing_deg'], 1)}"
        f" view={fmt_float(row['query_viewpoint_deg'], 1)}"
        f" selA={fmt_float(row['selected_angle_deg'], 1)}"
        f" centerD={fmt_float(row['center_depth_m'])}"
        f" supportD={fmt_float(row['support_depth_m'])}"
        f" nearD={fmt_float(row['nearest_depth_m'])}"
        f" elev={fmt_float(row['query_elevation_deg'], 1)}"
        f"->{fmt_float(row['selected_elevation_deg'], 1)}"
        f" box={fmt_float(row['box_width'])}"
        f"x{fmt_float(row['box_height'])}"
        f" by={fmt_float(row['bottom_y'])}"
    )


# ============================================================
# Thin compositor wrapper
# ============================================================

OriginalCompositor = (
    base.HEMultiActorCompositorV1
)


class GeometryLoggingCompositor(
    OriginalCompositor
):
    """
    Exact original compositor plus passive logging after each render call.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self._geometry_render_call_index = 0

    def render(
        self,
        *args,
        **kwargs,
    ):
        result = super().render(
            *args,
            **kwargs,
        )

        render_call_index = int(
            self._geometry_render_call_index
        )

        camera_count = max(
            1,
            len(
                _CAMERA_NAMES
            ),
        )

        camera_index = (
            render_call_index
            %
            camera_count
        )

        frame_offset = (
            render_call_index
            //
            camera_count
        )

        scenario_frame = (
            int(
                _START_FRAME
            )
            +
            frame_offset
        )

        camera_name = (
            _CAMERA_NAMES[
                camera_index
            ]
            if _CAMERA_NAMES
            else
            f"camera_{camera_index}"
        )

        writer = geometry_writer()

        for actor_result in (
            result.actor_results
        ):
            row = (
                geometry_row_from_actor_result(
                    actor_result=
                        actor_result,

                    scenario_frame=
                        scenario_frame,

                    camera_name=
                        camera_name,

                    render_call_index=
                        render_call_index,
                )
            )

            writer.writerow(
                row
            )

            maybe_print_near_geometry(
                row
            )

        self._geometry_render_call_index += 1

        return result


# ============================================================
# Main
# ============================================================

def main():
    global _CAMERA_NAMES
    global _START_FRAME
    global _GEOMETRY_CSV_PATH
    global _GEOMETRY_PRINT_NEAR_M
    global _GEOMETRY_PRINT_EVERY

    # --------------------------------------------------------
    # Consume only probe-specific arguments.
    # Everything else is passed unmodified to the real runner.
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        "--geometry-csv",
        default=None,
        help=(
            "Optional explicit per-actor geometry CSV path. "
            "By default it is written beside the generic runner CSV."
        ),
    )

    parser.add_argument(
        "--geometry-print-near-m",
        type=float,
        default=12.0,
        help=(
            "Print detailed geometry while an actor is within this "
            "positive forward-camera depth. Set <=0 to disable."
        ),
    )

    parser.add_argument(
        "--geometry-print-every",
        type=int,
        default=1,
        help=(
            "For near-range console traces, print every N scenario frames."
        ),
    )

    probe_args, remaining = (
        parser.parse_known_args(
            sys.argv[1:]
        )
    )

    model_name = read_cli_value(
        remaining,
        "--model",
        None,
    )

    if model_name is None:
        raise RuntimeError(
            "--model is required."
        )

    _CAMERA_NAMES = (
        model_camera_names(
            model_name
        )
    )

    _START_FRAME = int(
        read_cli_value(
            remaining,
            "--start-frame",
            0,
        )
    )

    _GEOMETRY_PRINT_NEAR_M = float(
        probe_args.geometry_print_near_m
    )

    _GEOMETRY_PRINT_EVERY = max(
        1,
        int(
            probe_args.geometry_print_every
        ),
    )

    # --------------------------------------------------------
    # Default geometry output beside the generic runner CSV.
    # --------------------------------------------------------

    if (
        probe_args.geometry_csv
        is not None
    ):
        _GEOMETRY_CSV_PATH = Path(
            probe_args.geometry_csv
        ).resolve()

    else:
        resolved_path = Path(
            read_cli_value(
                remaining,
                "--resolved",
                str(
                    base.DEFAULT_RESOLVED
                ),
            )
        ).resolve()

        with resolved_path.open(
            "r",
            encoding="utf-8",
        ) as fp:
            scenario_json = json.load(
                fp
            )

        scenario_id = str(
            scenario_json.get(
                "scenario_id",
                resolved_path.stem,
            )
        )

        output_root = Path(
            read_cli_value(
                remaining,
                "--output-root",
                str(
                    base.DEFAULT_OUTPUT_ROOT
                ),
            )
        ).resolve()

        _GEOMETRY_CSV_PATH = (
            output_root
            /
            scenario_id
            /
            str(
                model_name
            ).lower()
            /
            (
                f"{scenario_id}_"
                f"{str(model_name).lower()}_"
                "he_geometry.csv"
            )
        )

    print(
        "[HE geometry probe] passive logging only"
    )

    print(
        "[HE geometry probe] cameras:",
        ", ".join(
            _CAMERA_NAMES
        ),
    )

    print(
        "[HE geometry probe] start frame:",
        _START_FRAME,
    )

    print(
        "[HE geometry probe] near trace:",
        (
            "disabled"
            if _GEOMETRY_PRINT_NEAR_M <= 0.0
            else
            f"z <= {_GEOMETRY_PRINT_NEAR_M:.2f} m"
        ),
    )

    # Replace only the runner's compositor symbol.  The subclass calls the
    # original renderer first, then reads the resulting metadata.
    base.HEMultiActorCompositorV1 = (
        GeometryLoggingCompositor
    )

    # Hand the untouched generic-runner arguments back to its own parser.
    sys.argv = [
        sys.argv[0],
        *remaining,
    ]

    try:
        base.main()

    finally:
        close_geometry_csv()


if __name__ == "__main__":
    main()
