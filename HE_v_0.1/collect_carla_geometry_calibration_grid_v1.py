#!/usr/bin/env python3

"""
collect_carla_geometry_calibration_grid_v1.py

Controlled CARLA geometry-calibration dataset collector for HE.

Purpose
-------
Collect clean CARLA instance-mask geometry at controlled:

    distance x viewpoint

poses.

The collector deliberately does NOT:
    - render HE sprites,
    - run the HE compositor,
    - modify HEPlacement,
    - fit a correction model,
    - use selected sprite-bank coordinates.

It only creates CARLA geometry ground truth.

Default calibration grid
------------------------
Distances:
    10, 15, 20, 30, 40 m

Viewpoints:
    0, 15, 30, 45, 60, 90, 120, 150, 180 deg

Repeats:
    3

Total:
    5 x 9 x 3 = 135 measurements

Viewpoint convention
--------------------
Same convention as the HE view-matrix selector:

    viewpoint = bearing - relative_actor_yaw

where:
    bearing = atan2(camera_right, camera_forward)

Therefore:
    0 deg   -> rear view
    180 deg -> front view

For a centered actor, bearing is approximately 0 deg, so:

    relative_actor_yaw ~= -viewpoint

Architecture
------------
This script reuses the already validated helpers from:

    record_carla_he_pair.py

including:
    - CARLA world setup
    - ego settling
    - camera geometry
    - synchronous sensor retrieval
    - projected 3D bbox
    - robust vehicle instance-mask extraction

Output
------
geometry_calibration_v1/
    calibration_manifest.json
    measurements.jsonl
    masks/
        pose_d010_a000_r00.png
        ...
"""

import argparse
import json
import math
import queue
from pathlib import Path

import cv2
import numpy as np

try:
    import carla
except ImportError:
    raise RuntimeError(
        "Could not import CARLA. "
        "Run this script inside the CARLA Python environment."
    )


# ============================================================
# Reuse validated CARLA code.
# ============================================================

from record_carla_he_pair import (
    compute_actor_2d_bbox,
    extract_vehicle_instance_mask,
    get_blueprint,
    get_image_for_carla_frame,
    make_instance_camera,
    restore_world,
    settle_ego_on_road,
    setup_world,
    spawn_actor_safe,
    yaw_forward_right,
)


# ============================================================
# Basic helpers
# ============================================================

def normalize_angle_180(angle_deg):
    return (
        float(angle_deg) + 180.0
    ) % 360.0 - 180.0


def normalize_angle_360(angle_deg):
    """
    Normalize an angle to [0, 360), while collapsing numerical
    values extremely close to 0/360 back to exactly 0.

    This prevents floating-point noise such as -1e-14 deg from
    being reported as 359.999999999999 deg.
    """

    value = float(angle_deg) % 360.0

    if (
        math.isclose(
            value,
            0.0,
            abs_tol=1e-9,
        )
        or
        math.isclose(
            value,
            360.0,
            abs_tol=1e-9,
        )
    ):
        return 0.0

    return value


def parse_float_list(text):
    values = []

    for token in str(text).split(","):

        token = token.strip()

        if not token:
            continue

        values.append(
            float(token)
        )

    if not values:
        raise ValueError(
            "Expected at least one numeric value."
        )

    return values


def transform_point_to_camera(
    world_location,
    camera_transform,
):
    """
    Convert a CARLA world location into HE camera-relative
    coordinates.

    CARLA camera local coordinates:
        X = forward
        Y = right
        Z = up

    HE convention:
        x_m = right
        y_m = vertical
        z_m = forward
    """

    world_to_camera = np.array(
        camera_transform.get_inverse_matrix(),
        dtype=np.float64,
    )

    p_world = np.array(
        [
            float(world_location.x),
            float(world_location.y),
            float(world_location.z),
            1.0,
        ],
        dtype=np.float64,
    )

    p_camera = (
        world_to_camera
        @ p_world
    )

    return {
        "x_m":
            float(
                p_camera[1]
            ),

        "y_m":
            float(
                p_camera[2]
            ),

        "z_m":
            float(
                p_camera[0]
            ),
    }


def actor_relative_yaw_in_camera(
    actor_transform,
    camera_transform,
):
    """
    Compute actor heading relative to the camera heading.

    We transform the actor forward vector into camera coordinates
    and measure yaw in the camera horizontal plane.
    """

    actor_forward = (
        actor_transform.get_forward_vector()
    )

    world_to_camera = np.array(
        camera_transform.get_inverse_matrix(),
        dtype=np.float64,
    )

    # Direction vector -> homogeneous w = 0.
    v_world = np.array(
        [
            float(actor_forward.x),
            float(actor_forward.y),
            float(actor_forward.z),
            0.0,
        ],
        dtype=np.float64,
    )

    v_camera = (
        world_to_camera
        @ v_world
    )

    forward_component = float(
        v_camera[0]
    )

    right_component = float(
        v_camera[1]
    )

    return normalize_angle_180(
        math.degrees(
            math.atan2(
                right_component,
                forward_component,
            )
        )
    )


def compute_he_view_query(
    actor_transform,
    camera_transform,
    target_height_m,
):
    """
    Compute exactly the continuous camera-space quantities that
    geometry calibration needs.

    Viewpoint:
        bearing - relative yaw

    Distance/elevation use the target point:

        target_y = actor_origin_y + target_height_m
    """

    state = transform_point_to_camera(
        world_location=
            actor_transform.location,
        camera_transform=
            camera_transform,
    )

    relative_yaw_deg = (
        actor_relative_yaw_in_camera(
            actor_transform=
                actor_transform,
            camera_transform=
                camera_transform,
        )
    )

    x_m = float(
        state["x_m"]
    )

    y_m = float(
        state["y_m"]
    )

    z_m = float(
        state["z_m"]
    )

    bearing_deg = math.degrees(
        math.atan2(
            x_m,
            z_m,
        )
    )

    viewpoint_deg = normalize_angle_360(
        bearing_deg
        -
        relative_yaw_deg
    )

    target_y_m = (
        y_m
        +
        float(
            target_height_m
        )
    )

    horizontal_m = math.sqrt(
        x_m * x_m
        +
        z_m * z_m
    )

    distance_m = math.sqrt(
        horizontal_m * horizontal_m
        +
        target_y_m * target_y_m
    )

    elevation_deg = math.degrees(
        math.atan2(
            -target_y_m,
            max(
                horizontal_m,
                1e-9,
            ),
        )
    )

    return {
        "x_m":
            x_m,

        "y_m":
            y_m,

        "z_m":
            z_m,

        "relative_yaw_deg":
            float(
                relative_yaw_deg
            ),

        "bearing_deg":
            float(
                bearing_deg
            ),

        "viewpoint_deg":
            float(
                viewpoint_deg
            ),

        "target_y_m":
            float(
                target_y_m
            ),

        "horizontal_m":
            float(
                horizontal_m
            ),

        "distance_m":
            float(
                distance_m
            ),

        "elevation_deg":
            float(
                elevation_deg
            ),
    }


def mask_geometry(mask):
    """
    Geometry of a binary uint8 mask.
    """

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    )

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    )

    return {
        "x1":
            x1,

        "y1":
            y1,

        "x2":
            x2,

        "y2":
            y2,

        "width":
            int(
                x2 - x1 + 1
            ),

        "height":
            int(
                y2 - y1 + 1
            ),

        "cx":
            float(
                0.5
                *
                (
                    x1 + x2
                )
            ),

        "bottom_y":
            float(
                y2
            ),

        "area":
            int(
                len(xs)
            ),
    }


def transform_to_dict(transform):
    return {
        "location": {
            "x":
                float(
                    transform.location.x
                ),

            "y":
                float(
                    transform.location.y
                ),

            "z":
                float(
                    transform.location.z
                ),
        },

        "rotation": {
            "pitch":
                float(
                    transform.rotation.pitch
                ),

            "yaw":
                float(
                    transform.rotation.yaw
                ),

            "roll":
                float(
                    transform.rotation.roll
                ),
        },
    }


def to_jsonable(value):
    """
    Make debug metadata from CARLA/NumPy helpers JSON-safe.
    """

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k):
                to_jsonable(v)
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            to_jsonable(v)
            for v
            in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    return value


# ============================================================
# Controlled pose generation
# ============================================================

def road_z_at_xy(
    world,
    x,
    y,
    fallback_z,
    z_lift,
):
    """
    Find road height while preserving the requested x/y.

    Same principle as the existing pair recorder.
    """

    query_loc = carla.Location(
        x=float(x),
        y=float(y),
        z=float(fallback_z),
    )

    waypoint = (
        world.get_map().get_waypoint(
            query_loc,
            project_to_road=True,
            lane_type=
                carla.LaneType.Driving,
        )
    )

    if waypoint is None:
        return float(
            fallback_z
        )

    return float(
        waypoint.transform.location.z
        +
        float(
            z_lift
        )
    )


def build_actor_transform_for_view(
    world,
    camera_transform,
    requested_distance_m,
    requested_viewpoint_deg,
    target_height_m,
    z_lift,
    iterations=3,
):
    """
    Build a centered actor pose for the requested HE view.

    The actor is placed approximately on the optical-center ground
    ray, i.e. camera-relative lateral x ~= 0.

    We iteratively compensate horizontal distance for the vertical
    target-point offset so that:

        HE query distance ~= requested_distance_m

    Actor yaw is then selected so that:

        viewpoint = bearing - relative_yaw

    matches requested_viewpoint_deg.

    The final measured query is always stored later. We never assume
    the requested pose was realized perfectly.
    """

    requested_distance_m = float(
        requested_distance_m
    )

    requested_viewpoint_deg = float(
        requested_viewpoint_deg
    )

    if requested_distance_m <= 0.0:
        raise ValueError(
            "requested_distance_m must be > 0"
        )

    camera_yaw_deg = float(
        camera_transform.rotation.yaw
    )

    forward_xy, _ = yaw_forward_right(
        camera_yaw_deg
    )

    camera_x = float(
        camera_transform.location.x
    )

    camera_y = float(
        camera_transform.location.y
    )

    camera_z = float(
        camera_transform.location.z
    )

    # Initial horizontal-distance estimate.
    horizontal_m = float(
        requested_distance_m
    )

    actor_location = None

    for _ in range(
        max(
            1,
            int(iterations),
        )
    ):

        world_x = (
            camera_x
            +
            float(
                forward_xy[0]
            )
            *
            horizontal_m
        )

        world_y = (
            camera_y
            +
            float(
                forward_xy[1]
            )
            *
            horizontal_m
        )

        world_z = road_z_at_xy(
            world=world,
            x=world_x,
            y=world_y,
            fallback_z=
                camera_z
                -
                float(
                    target_height_m
                ),
            z_lift=z_lift,
        )

        actor_location = carla.Location(
            x=float(
                world_x
            ),
            y=float(
                world_y
            ),
            z=float(
                world_z
            ),
        )

        # Calculate actor-origin vertical position in camera
        # coordinates using a provisional yaw-independent transform.
        provisional_tf = carla.Transform(
            actor_location,
            carla.Rotation(
                pitch=0.0,
                yaw=camera_yaw_deg,
                roll=0.0,
            ),
        )

        provisional_query = (
            compute_he_view_query(
                actor_transform=
                    provisional_tf,
                camera_transform=
                    camera_transform,
                target_height_m=
                    target_height_m,
            )
        )

        target_y_m = float(
            provisional_query[
                "target_y_m"
            ]
        )

        horizontal_squared = (
            requested_distance_m
            *
            requested_distance_m
            -
            target_y_m
            *
            target_y_m
        )

        horizontal_m = math.sqrt(
            max(
                horizontal_squared,
                0.01,
            )
        )

    # ------------------------------------------------------------
    # Determine actual bearing at the final ground-snapped position.
    # ------------------------------------------------------------

    provisional_tf = carla.Transform(
        actor_location,
        carla.Rotation(
            pitch=0.0,
            yaw=camera_yaw_deg,
            roll=0.0,
        ),
    )

    provisional_query = compute_he_view_query(
        actor_transform=
            provisional_tf,
        camera_transform=
            camera_transform,
        target_height_m=
            target_height_m,
    )

    actual_bearing_deg = float(
        provisional_query[
            "bearing_deg"
        ]
    )

    # viewpoint = bearing - relative_yaw
    #
    # relative_yaw = bearing - viewpoint
    desired_relative_yaw_deg = (
        actual_bearing_deg
        -
        requested_viewpoint_deg
    )

    actor_world_yaw_deg = (
        camera_yaw_deg
        +
        desired_relative_yaw_deg
    )

    return carla.Transform(
        actor_location,
        carla.Rotation(
            pitch=0.0,
            yaw=float(
                actor_world_yaw_deg
            ),
            roll=0.0,
        ),
    )


# ============================================================
# Dataset validation
# ============================================================
def validate_repeatability(
    measurements,
    max_geometry_range_px=1.0,
):
    """
    Validate repeated measurements at each controlled pose.

    For a frozen CARLA actor/camera pose, repeated instance-mask
    bbox geometry should be identical or differ by at most a very
    small rasterization tolerance.

    Returns:
        passed, failures
    """

    grouped = {}

    for row in measurements:

        if not row.get(
            "mask_found",
            False,
        ):
            continue

        pose_id = str(
            row[
                "pose_id"
            ]
        )

        grouped.setdefault(
            pose_id,
            [],
        ).append(
            row
        )

    failures = []

    for pose_id, rows in grouped.items():

        geometries = [
            row.get(
                "mask_geometry"
            )
            for row in rows
        ]

        geometries = [
            g
            for g in geometries
            if g is not None
        ]

        if len(
            geometries
        ) <= 1:
            continue

        widths = [
            float(
                g["width"]
            )
            for g in geometries
        ]

        heights = [
            float(
                g["height"]
            )
            for g in geometries
        ]

        bottoms = [
            float(
                g["bottom_y"]
            )
            for g in geometries
        ]

        centers = [
            float(
                g["cx"]
            )
            for g in geometries
        ]

        width_range = (
            max(widths)
            -
            min(widths)
        )

        height_range = (
            max(heights)
            -
            min(heights)
        )

        bottom_range = (
            max(bottoms)
            -
            min(bottoms)
        )

        cx_range = (
            max(centers)
            -
            min(centers)
        )

        worst_range = max(
            width_range,
            height_range,
            bottom_range,
            cx_range,
        )

        if (
            worst_range
            >
            float(
                max_geometry_range_px
            )
        ):

            failures.append({
                "pose_id":
                    pose_id,

                "width_range_px":
                    float(
                        width_range
                    ),

                "height_range_px":
                    float(
                        height_range
                    ),

                "bottom_range_px":
                    float(
                        bottom_range
                    ),

                "cx_range_px":
                    float(
                        cx_range
                    ),
            })

    passed = (
        len(
            failures
        )
        ==
        0
    )

    return (
        passed,
        failures,
    )
def validate_dataset(
    measurements,
    expected_count,
    masks_dir,
):
    completed = len(
        measurements
    )

    found_count = sum(
        1
        for row in measurements
        if row.get(
            "mask_found",
            False,
        )
    )

    mask_file_count = len(
        list(
            Path(
                masks_dir
            ).glob(
                "*.png"
            )
        )
    )

    print()
    print("=" * 92)
    print(
        "GEOMETRY CALIBRATION DATASET VALIDATION"
    )
    print("=" * 92)

    print(
        "Expected measurements :",
        expected_count,
    )

    print(
        "Completed measurements:",
        completed,
    )

    print(
        "Masks found           :",
        found_count,
    )

    print(
        "Mask files            :",
        mask_file_count,
    )

    missing = (
        expected_count
        -
        found_count
    )

    print(
        "Missing valid masks    :",
        missing,
    )

    passed = (
        completed
        ==
        expected_count
        and
        found_count
        ==
        expected_count
        and
        mask_file_count
        ==
        expected_count
    )

    if passed:

        print()
        print(
            "DATASET VALIDATION PASSED"
        )

    else:

        print()
        print(
            "DATASET VALIDATION FAILED"
        )

    print("=" * 92)

    return passed


# ============================================================
# Collection
# ============================================================

def run(args):

    distances = parse_float_list(
        args.distances
    )

    viewpoints = parse_float_list(
        args.viewpoints
    )

    repeats = int(
        args.repeats
    )

    if repeats <= 0:
        raise ValueError(
            "--repeats must be >= 1"
        )

    expected_count = (
        len(
            distances
        )
        *
        len(
            viewpoints
        )
        *
        repeats
    )

    output_dir = Path(
        args.output_dir
    )

    masks_dir = (
        output_dir
        /
        "masks"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    masks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    measurements_path = (
        output_dir
        /
        "measurements.jsonl"
    )

    manifest_path = (
        output_dir
        /
        "calibration_manifest.json"
    )

    client = carla.Client(
        args.host,
        args.port,
    )

    client.set_timeout(
        20.0
    )

    # Existing pair-recorder convention.
    world_before = client.get_world()
    original_settings = (
        world_before.get_settings()
    )

    world = None

    actors = []

    instance_queue = queue.Queue()

    measurement_file = None

    measurements = []

    try:

        # ========================================================
        # World
        # ========================================================

        world = setup_world(
            client=client,
            town=args.town,
            fps=args.fps,
        )

        spawn_points = (
            world.get_map()
            .get_spawn_points()
        )

        if not spawn_points:

            raise RuntimeError(
                "CARLA map has no spawn points."
            )

        if (
            args.spawn_index < 0
            or
            args.spawn_index
            >=
            len(
                spawn_points
            )
        ):

            raise RuntimeError(
                f"spawn_index {args.spawn_index} "
                f"out of range 0.."
                f"{len(spawn_points)-1}"
            )

        # ========================================================
        # Ego
        # ========================================================

        ego_bp = get_blueprint(
            world,
            args.ego_vehicle_filter,
        )

        ego = spawn_actor_safe(
            world,
            ego_bp,
            spawn_points[
                args.spawn_index
            ],
        )

        actors.append(
            ego
        )

        settled_ego_tf = (
            settle_ego_on_road(
                world=world,
                ego=ego,
                settle_frames=
                    args.ego_settle_frames,
            )
        )

        # ========================================================
        # Instance camera
        # ========================================================

        instance_camera = (
            make_instance_camera(
                world=world,
                ego=ego,
                args=args,
            )
        )

        actors.append(
            instance_camera
        )

        instance_camera.listen(
            instance_queue.put
        )

        # Allow attachment transform to become available.
        first_frame = world.tick()

        first_image = (
            get_image_for_carla_frame(
                instance_queue,
                target_frame=
                    first_frame,
                timeout=5.0,
            )
        )

        camera_transform = (
            first_image.transform
        )

        # ========================================================
        # Adversary
        # ========================================================

        adversary_bp = get_blueprint(
            world,
            args.adversary_vehicle_filter,
        )

        initial_transform = (
            build_actor_transform_for_view(
                world=world,
                camera_transform=
                    camera_transform,
                requested_distance_m=
                    distances[0],
                requested_viewpoint_deg=
                    viewpoints[0],
                target_height_m=
                    args.target_height_m,
                z_lift=
                    args.z_lift,
            )
        )

        adversary = spawn_actor_safe(
            world,
            adversary_bp,
            initial_transform,
        )

        adversary.set_autopilot(
            False
        )

        adversary.set_simulate_physics(
            False
        )

        actors.append(
            adversary
        )
        # ========================================================
        # Initial adversary render warm-up
        #
        # A newly spawned CARLA vehicle may have a valid transform
        # before its complete instance-segmentation representation
        # is available to the sensor.
        #
        # We therefore consume several synchronized sensor frames
        # after spawning the adversary and before recording the
        # first calibration pose.
        #
        # This is an initialization-only warm-up. It is different
        # from transition_ticks, which is used after changing poses.
        # ========================================================

        print(
            "[INFO] Warming newly spawned adversary for",
            args.actor_warmup_ticks,
            "ticks..."
        )

        for _ in range(
            int(
                args.actor_warmup_ticks
            )
        ):

            warmup_frame = world.tick()

            warmup_image = (
                get_image_for_carla_frame(
                    instance_queue,
                    target_frame=
                        warmup_frame,
                    timeout=5.0,
                )
            )

            # Keep the exact latest sensor transform.
            camera_transform = (
                warmup_image.transform
            )
        semantic_tags = list(
            adversary.semantic_tags
        )

        print()
        print("=" * 92)
        print(
            "HE CARLA GEOMETRY CALIBRATION GRID V1"
        )
        print("=" * 92)

        print(
            "Town:",
            world.get_map().name,
        )

        print(
            "Distances:",
            distances,
        )

        print(
            "Viewpoints:",
            viewpoints,
        )

        print(
            "Repeats:",
            repeats,
        )

        print(
            "Total measurements:",
            expected_count,
        )

        print(
            "Actor semantic tags:",
            semantic_tags,
        )

        print(
            "Output:",
            output_dir,
        )

        print("=" * 92)

        # ========================================================
        # Manifest
        # ========================================================

        manifest = {
            "dataset":
                "he_geometry_calibration_grid_v1",

            "town":
                str(
                    world.get_map().name
                ),

            "spawn_index":
                int(
                    args.spawn_index
                ),

            "distances_m":
                [
                    float(v)
                    for v
                    in distances
                ],

            "viewpoints_deg":
                [
                    float(v)
                    for v
                    in viewpoints
                ],

            "repeats_per_pose":
                repeats,

            "unique_pose_count":
                int(
                    len(distances)
                    *
                    len(viewpoints)
                ),

            "expected_measurement_count":
                int(
                    expected_count
                ),

            "vehicle": {
                "ego_filter":
                    str(
                        args.ego_vehicle_filter
                    ),

                "adversary_filter":
                    str(
                        args.adversary_vehicle_filter
                    ),

                "adversary_semantic_tags":
                    [
                        int(v)
                        for v
                        in semantic_tags
                    ],
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

                "mount": {
                    "x":
                        float(
                            args.camera_x
                        ),

                    "y":
                        float(
                            args.camera_y
                        ),

                    "z":
                        float(
                            args.camera_z
                        ),

                    "pitch":
                        float(
                            args.camera_pitch
                        ),

                    "yaw":
                        float(
                            args.camera_yaw
                        ),

                    "roll":
                        float(
                            args.camera_roll
                        ),
                },
            },

            "he_geometry": {
                "target_height_m":
                    float(
                        args.target_height_m
                    ),

                "viewpoint_definition":
                    (
                        "viewpoint = "
                        "bearing - relative_yaw"
                    ),

                "viewpoint_zero":
                    "rear",

                "viewpoint_180":
                    "front",
            },

            "collection": {
                "fps":
                    int(
                        args.fps
                    ),

                "transition_ticks":
                    int(
                        args.transition_ticks
                    ),

                "z_lift":
                    float(
                        args.z_lift
                    ),
                "actor_warmup_ticks":
                    int(
                        args.actor_warmup_ticks
                    ),
            },

            "settled_ego_transform":
                transform_to_dict(
                    settled_ego_tf
                ),

            "source_mask_implementation":
                (
                    "record_carla_he_pair."
                    "extract_vehicle_instance_mask"
                ),
        }

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                manifest,
                f,
                indent=2,
            )

        measurement_file = (
            measurements_path.open(
                "w",
                encoding="utf-8",
            )
        )

        # ========================================================
        # Grid
        # ========================================================

        measurement_index = 0

        for distance_m in distances:

            for viewpoint_deg in viewpoints:

                pose_name = (
                    f"d{int(round(distance_m)):03d}"
                    f"_a{int(round(viewpoint_deg)):03d}"
                )

                desired_transform = (
                    build_actor_transform_for_view(
                        world=world,
                        camera_transform=
                            camera_transform,
                        requested_distance_m=
                            distance_m,
                        requested_viewpoint_deg=
                            viewpoint_deg,
                        target_height_m=
                            args.target_height_m,
                        z_lift=
                            args.z_lift,
                    )
                )

                adversary.set_transform(
                    desired_transform
                )

                # ------------------------------------------------
                # Transition ticks.
                #
                # Discard these measurements so every saved repeat
                # belongs to a fully established pose.
                # ------------------------------------------------

                for _ in range(
                    int(
                        args.transition_ticks
                    )
                ):

                    transition_frame = (
                        world.tick()
                    )

                    transition_image = (
                        get_image_for_carla_frame(
                            instance_queue,
                            target_frame=
                                transition_frame,
                            timeout=5.0,
                        )
                    )

                    # Keep latest exact sensor pose.
                    camera_transform = (
                        transition_image.transform
                    )

                # ------------------------------------------------
                # Repeated measurements
                # ------------------------------------------------

                for repeat_idx in range(
                    repeats
                ):

                    carla_frame = (
                        world.tick()
                    )

                    instance_image = (
                        get_image_for_carla_frame(
                            instance_queue,
                            target_frame=
                                carla_frame,
                            timeout=5.0,
                        )
                    )

                    camera_transform = (
                        instance_image.transform
                    )

                    actor_transform = (
                        adversary.get_transform()
                    )

                    projected_bbox = (
                        compute_actor_2d_bbox(
                            actor=adversary,
                            camera_transform=
                                camera_transform,
                            width=args.width,
                            height=args.height,
                            fov=args.fov,
                        )
                    )

                    instance_mask, mask_info = (
                        extract_vehicle_instance_mask(
                            instance_image=
                                instance_image,
                            projected_bbox=
                                projected_bbox,
                            semantic_tags=
                                semantic_tags,
                        )
                    )

                    geometry = mask_geometry(
                        instance_mask
                    )

                    view_query = (
                        compute_he_view_query(
                            actor_transform=
                                actor_transform,
                            camera_transform=
                                camera_transform,
                            target_height_m=
                                args.target_height_m,
                        )
                    )

                    filename = (
                        f"pose_{pose_name}"
                        f"_r{repeat_idx:02d}.png"
                    )

                    mask_path = (
                        masks_dir
                        /
                        filename
                    )

                    mask_found = (
                        geometry
                        is not None
                        and
                        bool(
                            mask_info.get(
                                "found",
                                False,
                            )
                        )
                    )

                    if mask_found:

                        ok = cv2.imwrite(
                            str(
                                mask_path
                            ),
                            instance_mask,
                        )

                        if not ok:

                            raise RuntimeError(
                                "Could not write mask: "
                                f"{mask_path}"
                            )

                    record = {
                        "measurement_index":
                            int(
                                measurement_index
                            ),

                        "pose_id":
                            pose_name,

                        "repeat_id":
                            int(
                                repeat_idx
                            ),

                        "carla_frame":
                            int(
                                instance_image.frame
                            ),

                        # ----------------------------------------
                        # Requested controlled pose
                        # ----------------------------------------

                        "requested": {
                            "distance_m":
                                float(
                                    distance_m
                                ),

                            "viewpoint_deg":
                                float(
                                    viewpoint_deg
                                ),
                        },

                        # ----------------------------------------
                        # Actual continuous HE query coordinates
                        # ----------------------------------------

                        "actual_he_query":
                            view_query,

                        # ----------------------------------------
                        # Exact transforms
                        # ----------------------------------------

                        "camera_transform":
                            transform_to_dict(
                                camera_transform
                            ),

                        "actor_transform":
                            transform_to_dict(
                                actor_transform
                            ),

                        # ----------------------------------------
                        # CARLA projected 3-D box
                        # diagnostic only
                        # ----------------------------------------

                        "projected_bbox":
                            to_jsonable(
                                projected_bbox
                            ),

                        # ----------------------------------------
                        # Instance mask
                        # ----------------------------------------

                        "mask_found":
                            bool(
                                mask_found
                            ),

                        "mask_geometry":
                            to_jsonable(
                                geometry
                            ),

                        "mask_info":
                            to_jsonable(
                                mask_info
                            ),

                        "mask_path":
                            (
                                str(
                                    mask_path
                                )
                                if mask_found
                                else None
                            ),
                    }

                    measurement_file.write(
                        json.dumps(
                            record
                        )
                        +
                        "\n"
                    )

                    measurement_file.flush()

                    measurements.append(
                        record
                    )

                    status = (
                        "OK"
                        if mask_found
                        else
                        "FAIL"
                    )

                    print(
                        f"[{measurement_index + 1:03d}/"
                        f"{expected_count:03d}] "
                        f"d={distance_m:5.1f}m "
                        f"view={viewpoint_deg:6.1f}deg "
                        f"r={repeat_idx} "
                        f"actual_d="
                        f"{view_query['distance_m']:6.3f} "
                        f"actual_view="
                        f"{view_query['viewpoint_deg']:7.3f} "
                        f"elev="
                        f"{view_query['elevation_deg']:6.3f} "
                        f"{status}"
                    )

                    measurement_index += 1

        # ========================================================
        # Validation
        # ========================================================

        passed = validate_dataset(
            measurements=
                measurements,
            expected_count=
                expected_count,
            masks_dir=
                masks_dir,
        )

        if not passed:

            raise RuntimeError(
                "Geometry calibration dataset "
                "validation failed."
            )

        print()
        print(
            "[DONE] Geometry calibration "
            "grid collection complete."
        )

        print(
            "[DONE] Manifest:",
            manifest_path,
        )

        print(
            "[DONE] Measurements:",
            measurements_path,
        )

        print(
            "[DONE] Masks:",
            masks_dir,
        )

    finally:

        if measurement_file is not None:

            measurement_file.close()

        for actor in reversed(
            actors
        ):

            try:

                actor.destroy()

            except Exception:

                pass

        if (
            world is not None
            and
            original_settings is not None
        ):

            try:

                restore_world(
                    world,
                    original_settings,
                )

            except Exception:

                pass


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

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
        "--output-dir",
        default=(
            "recordings/"
            "geometry_calibration_v1"
        ),
    )

    # ------------------------------------------------------------
    # Controlled grid
    # ------------------------------------------------------------

    parser.add_argument(
        "--distances",
        default=
            "10,15,20,30,40",
    )

    parser.add_argument(
        "--viewpoints",
        default=
            "0,15,30,45,60,90,120,150,180",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--transition-ticks",
        type=int,
        default=1,
        help=(
            "Discarded CARLA ticks after "
            "moving to each new pose."
        ),
    
    )
    parser.add_argument(
        "--actor-warmup-ticks",
        type=int,
        default=5,
        help=(
            "Synchronized sensor ticks discarded after "
            "initial adversary spawn before calibration "
            "measurements begin."
        ),
    )

    # ------------------------------------------------------------
    # CARLA timing
    # ------------------------------------------------------------

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--spawn-index",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--ego-settle-frames",
        type=int,
        default=30,
    )

    # ------------------------------------------------------------
    # Vehicle
    # ------------------------------------------------------------

    parser.add_argument(
        "--ego-vehicle-filter",
        default=
            "vehicle.tesla.model3",
    )

    parser.add_argument(
        "--adversary-vehicle-filter",
        default=
            "vehicle.tesla.model3",
    )

    # ------------------------------------------------------------
    # Fixed HE camera convention
    # ------------------------------------------------------------

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--camera-x",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--camera-y",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--camera-z",
        type=float,
        default=1.6,
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

    # ------------------------------------------------------------
    # HE/view-matrix convention
    # ------------------------------------------------------------

    parser.add_argument(
        "--target-height-m",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--z-lift",
        type=float,
        default=0.05,
        help=(
            "Actor origin lift above CARLA "
            "road waypoint height."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(
        parse_args()
    )