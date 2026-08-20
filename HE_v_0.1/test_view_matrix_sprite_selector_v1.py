"""
test_view_matrix_sprite_selector_v1.py

Standalone validation for the new HE sprite view-matrix selector.

Purpose
-------
Test selection using:

    azimuth
    distance
    elevation

before modifying run_he_temporal_compositor_v2.py.

The test loads multiple independently generated sprite-bank CSV files,
for example:

    elevation 0 deg:
        tesla_grabcut_view_matrix_full

    elevation 10 deg:
        tesla_grabcut_view_matrix_e10

Later we can simply add:

    elevation 20 deg:
        tesla_grabcut_view_matrix_e20

Important
---------
The 0-degree CSV may contain absolute Linux paths because it was
generated on the server.

We DO NOT trust the absolute rgba_path stored in the CSV.

Instead, we take only the PNG filename and reconstruct the local path:

    <csv parent>/rgba/<filename>

This makes copied server banks work correctly on Windows.
"""

from __future__ import print_function

import argparse
import csv
import math
import sys
from pathlib import Path


# ============================================================
# Default local banks
# ============================================================

DEFAULT_BANK_CSVS = [
    Path(
        r"D:\HE_Data\sprite_bank_grabcut"
        r"\tesla_grabcut_view_matrix_full"
        r"\view_matrix.csv"
    ),
    Path(
        r"D:\HE_Data\sprite_bank_grabcut"
        r"\tesla_grabcut_view_matrix_e10"
        r"\view_matrix.csv"
    ),
]


# ============================================================
# Angle utilities
# ============================================================

def normalize_angle_360(angle_deg):
    return float(angle_deg) % 360.0


def normalize_angle_180(angle_deg):
    return (
        float(angle_deg) + 180.0
    ) % 360.0 - 180.0


def circular_angle_error_deg(a_deg, b_deg):
    return abs(
        normalize_angle_180(
            float(a_deg) - float(b_deg)
        )
    )


# ============================================================
# View key
# ============================================================

def make_view_key(
    angle_deg,
    distance_m,
    elevation_deg,
):
    return (
        int(angle_deg) % 360,
        round(float(distance_m), 6),
        round(float(elevation_deg), 6),
    )


# ============================================================
# CSV loading
# ============================================================

def filename_from_any_path(path_text):
    """
    Extract filename from either:

        /mnt/hdd/.../file.png

    or:

        D:\\something\\file.png

    regardless of the OS running this script.
    """

    text = str(path_text).replace("\\", "/")
    return text.split("/")[-1]


def load_view_matrix(csv_paths):
    """
    Load all view-matrix CSV files into one lookup table.
    """

    records = []
    index = {}

    for csv_path in csv_paths:

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                "View-matrix CSV not found: {}".format(
                    csv_path
                )
            )

        bank_root = csv_path.parent
        rgba_dir = bank_root / "rgba"

        print()
        print(
            "[ViewMatrix] Loading:",
            csv_path
        )

        local_count = 0

        with open(
            str(csv_path),
            "r",
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                angle_deg = int(
                    float(
                        row["angle_deg"]
                    )
                ) % 360

                distance_m = float(
                    row["distance_m"]
                )

                elevation_deg = float(
                    row["elevation_deg"]
                )

                stored_rgba_path = row.get(
                    "rgba_path",
                    "",
                )

                rgba_filename = filename_from_any_path(
                    stored_rgba_path
                )

                # Rebuild using LOCAL bank directory.
                local_rgba_path = (
                    rgba_dir
                    /
                    rgba_filename
                )

                record = {
                    "angle_deg": angle_deg,
                    "distance_m": distance_m,
                    "elevation_deg": elevation_deg,

                    "sprite_width_px": int(
                        float(
                            row.get(
                                "sprite_width_px",
                                0,
                            )
                        )
                    ),

                    "sprite_height_px": int(
                        float(
                            row.get(
                                "sprite_height_px",
                                0,
                            )
                        )
                    ),

                    "alpha_pixels": int(
                        float(
                            row.get(
                                "alpha_pixels",
                                0,
                            )
                        )
                    ),

                    "anchor_x": float(
                        row.get(
                            "anchor_x",
                            0.0,
                        )
                    ),

                    "anchor_y": float(
                        row.get(
                            "anchor_y",
                            0.0,
                        )
                    ),

                    "rgba_path": str(
                        local_rgba_path
                    ),

                    "source_csv": str(
                        csv_path
                    ),
                }

                key = make_view_key(
                    angle_deg,
                    distance_m,
                    elevation_deg,
                )

                if key in index:
                    print(
                        "[ViewMatrix] WARNING: duplicate key:",
                        key
                    )

                records.append(
                    record
                )

                index[key] = record

                if local_rgba_path.exists():
                    local_count += 1

        print(
            "[ViewMatrix] Loaded records from bank:",
            local_count
        )

    if not records:
        raise RuntimeError(
            "No sprite records loaded."
        )

    return records, index


# ============================================================
# Available coordinates
# ============================================================

def discover_coordinate_values(records):

    angles = sorted(
        set(
            int(r["angle_deg"])
            for r in records
        )
    )

    distances = sorted(
        set(
            float(r["distance_m"])
            for r in records
        )
    )

    elevations = sorted(
        set(
            float(r["elevation_deg"])
            for r in records
        )
    )

    return (
        angles,
        distances,
        elevations,
    )


# ============================================================
# HE viewpoint geometry
# ============================================================

def compute_viewpoint_azimuth_deg(
    x_m,
    z_m,
    actor_relative_yaw_deg,
):
    """
    Same physical viewpoint convention currently used by HE.

    x_m:
        actor lateral position in camera frame
        negative = left
        positive = right

    z_m:
        actor forward position/depth

    actor_relative_yaw_deg:
        actor heading relative to camera heading

    Sprite convention:

        0   = rear
        90  = one side
        180 = front
        270 = opposite side
    """

    bearing_deg = math.degrees(
        math.atan2(
            float(x_m),
            float(z_m),
        )
    )

    viewpoint_angle_deg = (
        bearing_deg
        -
        float(actor_relative_yaw_deg)
    )

    return normalize_angle_360(
        viewpoint_angle_deg
    )


def compute_view_distance_and_elevation(
    x_m,
    y_m,
    z_m,
    target_height_m=0.75,
):
    """
    Convert HE camera-relative actor state into the coordinates
    used by generate_carla_sprite_view_matrix_grabcut_v1.py.

    y_m is the actor BASE height relative to the camera.

    Example for level ground:

        camera Z = 1.60 m
        actor base Z = 0.00 m

        y_m = -1.60 m

    The sprite generator targets a point target_height_m above
    the actor base.

    For target_height_m = 0.75:

        target relative Y = -1.60 + 0.75
                          = -0.85 m

    A negative target Y means the camera is above the target,
    therefore elevation is positive.
    """

    x_m = float(x_m)
    y_m = float(y_m)
    z_m = float(z_m)

    target_height_m = float(
        target_height_m
    )

    target_y_m = (
        y_m
        +
        target_height_m
    )

    horizontal_distance_m = math.sqrt(
        x_m * x_m
        +
        z_m * z_m
    )

    distance_m = math.sqrt(
        horizontal_distance_m
        *
        horizontal_distance_m
        +
        target_y_m
        *
        target_y_m
    )

    elevation_deg = math.degrees(
        math.atan2(
            -target_y_m,
            horizontal_distance_m,
        )
    )

    return (
        distance_m,
        elevation_deg,
        horizontal_distance_m,
        target_y_m,
    )


# ============================================================
# Nearest coordinates
# ============================================================

def nearest_angle(
    query_angle,
    available_angles,
):

    return min(
        available_angles,
        key=lambda value: circular_angle_error_deg(
            query_angle,
            value,
        ),
    )


def nearest_linear_value(
    query_value,
    available_values,
):

    return min(
        available_values,
        key=lambda value: abs(
            float(query_value)
            -
            float(value)
        ),
    )


# ============================================================
# Complete selector
# ============================================================

def select_view_matrix_sprite(
    records,
    index,
    x_m,
    y_m,
    z_m,
    yaw_deg,
    target_height_m=0.75,
):
    """
    Select nearest sprite using:

        viewpoint azimuth
        distance
        elevation
    """

    (
        available_angles,
        available_distances,
        available_elevations,
    ) = discover_coordinate_values(
        records
    )

    query_angle_deg = (
        compute_viewpoint_azimuth_deg(
            x_m=x_m,
            z_m=z_m,
            actor_relative_yaw_deg=yaw_deg,
        )
    )

    (
        query_distance_m,
        query_elevation_deg,
        horizontal_distance_m,
        target_y_m,
    ) = (
        compute_view_distance_and_elevation(
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            target_height_m=target_height_m,
        )
    )

    selected_angle = nearest_angle(
        query_angle_deg,
        available_angles,
    )

    selected_distance = nearest_linear_value(
        query_distance_m,
        available_distances,
    )

    selected_elevation = nearest_linear_value(
        query_elevation_deg,
        available_elevations,
    )

    key = make_view_key(
        selected_angle,
        selected_distance,
        selected_elevation,
    )

    record = index.get(
        key
    )

    if record is None:
        raise RuntimeError(
            "Selected view does not exist: {}".format(
                key
            )
        )

    return {
        "query": {
            "x_m": float(x_m),
            "y_m": float(y_m),
            "z_m": float(z_m),
            "yaw_deg": float(yaw_deg),

            "viewpoint_angle_deg":
                float(query_angle_deg),

            "distance_m":
                float(query_distance_m),

            "horizontal_distance_m":
                float(horizontal_distance_m),

            "elevation_deg":
                float(query_elevation_deg),

            "target_y_m":
                float(target_y_m),
        },

        "selected": {
            "angle_deg":
                int(selected_angle),

            "distance_m":
                float(selected_distance),

            "elevation_deg":
                float(selected_elevation),

            "angle_error_deg":
                float(
                    circular_angle_error_deg(
                        query_angle_deg,
                        selected_angle,
                    )
                ),

            "distance_error_m":
                float(
                    abs(
                        query_distance_m
                        -
                        selected_distance
                    )
                ),

            "elevation_error_deg":
                float(
                    abs(
                        query_elevation_deg
                        -
                        selected_elevation
                    )
                ),
        },

        "record": record,

        "exists": Path(
            record["rgba_path"]
        ).exists(),
    }


# ============================================================
# Printing
# ============================================================

def print_selection(
    name,
    result,
):

    q = result["query"]
    s = result["selected"]
    r = result["record"]

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    print(
        "HE state:"
    )

    print(
        "  x={:.3f}  y={:.3f}  z={:.3f}  yaw={:.3f}".format(
            q["x_m"],
            q["y_m"],
            q["z_m"],
            q["yaw_deg"],
        )
    )

    print()
    print(
        "Computed physical viewpoint:"
    )

    print(
        "  azimuth   = {:.3f} deg".format(
            q["viewpoint_angle_deg"]
        )
    )

    print(
        "  distance  = {:.3f} m".format(
            q["distance_m"]
        )
    )

    print(
        "  elevation = {:.3f} deg".format(
            q["elevation_deg"]
        )
    )

    print()
    print(
        "Selected bank coordinate:"
    )

    print(
        "  angle     = {:03d} deg".format(
            s["angle_deg"]
        )
    )

    print(
        "  distance  = {:.1f} m".format(
            s["distance_m"]
        )
    )

    print(
        "  elevation = {:.1f} deg".format(
            s["elevation_deg"]
        )
    )

    print()
    print(
        "Selection error:"
    )

    print(
        "  angle     = {:.3f} deg".format(
            s["angle_error_deg"]
        )
    )

    print(
        "  distance  = {:.3f} m".format(
            s["distance_error_m"]
        )
    )

    print(
        "  elevation = {:.3f} deg".format(
            s["elevation_error_deg"]
        )
    )

    print()
    print(
        "Sprite:"
    )

    print(
        " ",
        r["rgba_path"]
    )

    print(
        "  exists =",
        result["exists"]
    )

    print(
        "  size   = {}x{}".format(
            r["sprite_width_px"],
            r["sprite_height_px"],
        )
    )


# ============================================================
# Built-in validation
# ============================================================

def run_builtin_tests(
    records,
    index,
    camera_height_m,
    target_height_m,
):
    """
    Level-ground synthetic states.

    Actor base is assumed at ground height 0.
    Camera is camera_height_m above ground.

    Therefore:

        y_m = -camera_height_m
    """

    y_m = -float(
        camera_height_m
    )

    tests = [
        {
            "name":
                "TEST 1 - close vehicle, same direction, rear view",

            "state": {
                "x_m": 0.0,
                "y_m": y_m,
                "z_m": 5.0,
                "yaw_deg": 0.0,
            },

            "expected": (
                0,
                5.0,
                10.0,
            ),
        },

        {
            "name":
                "TEST 2 - close oncoming vehicle, front view",

            "state": {
                "x_m": 0.0,
                "y_m": y_m,
                "z_m": 5.0,
                "yaw_deg": 180.0,
            },

            "expected": (
                180,
                5.0,
                10.0,
            ),
        },

        {
            "name":
                "TEST 3 - medium vehicle, rear view",

            "state": {
                "x_m": 0.0,
                "y_m": y_m,
                "z_m": 10.0,
                "yaw_deg": 0.0,
            },

            "expected": (
                0,
                10.0,
                0.0,
            ),
        },

        {
            "name":
                "TEST 4 - far oncoming vehicle, front view",

            "state": {
                "x_m": 0.0,
                "y_m": y_m,
                "z_m": 20.0,
                "yaw_deg": 180.0,
            },

            "expected": (
                180,
                20.0,
                0.0,
            ),
        },
    ]

    passed = 0

    for test in tests:

        state = test["state"]

        result = select_view_matrix_sprite(
            records=records,
            index=index,

            x_m=state["x_m"],
            y_m=state["y_m"],
            z_m=state["z_m"],
            yaw_deg=state["yaw_deg"],

            target_height_m=target_height_m,
        )

        print_selection(
            test["name"],
            result,
        )

        selected = result["selected"]

        actual = (
            selected["angle_deg"],
            selected["distance_m"],
            selected["elevation_deg"],
        )

        expected = test["expected"]

        ok = (
            actual == expected
            and
            result["exists"]
        )

        if ok:

            passed += 1

            print()
            print(
                "[PASS]",
                actual
            )

        else:

            print()
            print(
                "[FAIL]"
            )

            print(
                "  expected:",
                expected
            )

            print(
                "  actual:  ",
                actual
            )

    print()
    print("=" * 78)

    print(
        "BUILT-IN TEST RESULT: {}/{} PASSED".format(
            passed,
            len(tests),
        )
    )

    print("=" * 78)

    return passed == len(
        tests
    )


# ============================================================
# Main
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bank-csv",
        action="append",
        default=None,
        help=(
            "View-matrix CSV. May be supplied more than once. "
            "If omitted, the local e0 and e10 Tesla banks are used."
        ),
    )

    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.60,
        help=(
            "Camera height above actor ground/base plane used "
            "for built-in level-ground tests."
        ),
    )

    parser.add_argument(
        "--target-height",
        type=float,
        default=0.75,
        help=(
            "Vehicle target height used when the sprite bank "
            "was generated."
        ),
    )

    # Optional one-off HE state query.

    parser.add_argument(
        "--x",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--y",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--z",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--yaw",
        type=float,
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.bank_csv:

        csv_paths = [
            Path(p)
            for p in args.bank_csv
        ]

    else:

        csv_paths = DEFAULT_BANK_CSVS

    (
        records,
        index,
    ) = load_view_matrix(
        csv_paths
    )

    (
        angles,
        distances,
        elevations,
    ) = discover_coordinate_values(
        records
    )

    missing_files = sum(
        1
        for record in records
        if not Path(
            record["rgba_path"]
        ).exists()
    )

    print()
    print("=" * 78)
    print("VIEW MATRIX SUMMARY")
    print("=" * 78)

    print(
        "records:",
        len(records)
    )

    print(
        "angles:",
        len(angles),
        "from",
        min(angles),
        "to",
        max(angles),
    )

    print(
        "distances:",
        distances
    )

    print(
        "elevations:",
        elevations
    )

    print(
        "missing local RGBA files:",
        missing_files
    )

    expected_cartesian_count = (
        len(angles)
        *
        len(distances)
        *
        len(elevations)
    )

    print(
        "expected Cartesian count:",
        expected_cartesian_count
    )

    if (
        len(index)
        !=
        expected_cartesian_count
    ):

        print(
            "[WARNING] View matrix is not a complete "
            "angle x distance x elevation grid."
        )

    # --------------------------------------------------------
    # Built-in tests
    # --------------------------------------------------------

    tests_ok = run_builtin_tests(
        records=records,
        index=index,
        camera_height_m=args.camera_height,
        target_height_m=args.target_height,
    )

    # --------------------------------------------------------
    # Optional manual query
    # --------------------------------------------------------

    custom_requested = any(
        value is not None
        for value in [
            args.x,
            args.y,
            args.z,
            args.yaw,
        ]
    )

    if custom_requested:

        if (
            args.x is None
            or
            args.z is None
            or
            args.yaw is None
        ):

            raise ValueError(
                "Custom query requires --x, --z and --yaw. "
                "--y is optional."
            )

        if args.y is None:

            # Level-ground default:
            # actor base = 0
            # camera = camera_height
            custom_y = -float(
                args.camera_height
            )

        else:

            custom_y = float(
                args.y
            )

        result = select_view_matrix_sprite(
            records=records,
            index=index,

            x_m=args.x,
            y_m=custom_y,
            z_m=args.z,
            yaw_deg=args.yaw,

            target_height_m=args.target_height,
        )

        print_selection(
            "CUSTOM HE STATE",
            result,
        )

    if not tests_ok:

        sys.exit(1)

    if missing_files != 0:

        print()
        print(
            "[ERROR] Some local sprite files are missing."
        )

        sys.exit(2)

    print()
    print(
        "[OK] View-matrix selector validation completed successfully."
    )


if __name__ == "__main__":
    main()