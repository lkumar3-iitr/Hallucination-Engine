"""
build_signalized_lead_follow_v1.py

Build the first long-horizon HE benchmark scenario from the inspected
Town10HD_Opt route without changing ScenarioSchema v2 or the v2 resolver.

Inputs
------
route.csv
traffic_lights.csv

Outputs
-------
1. ScenarioSpecV2 JSON
2. ResolvedScenarioV2 JSON
3. Environment-event sidecar (traffic-light schedule)
4. Actor trajectory CSV for QA

Coordinate convention
---------------------
The route CSV contains CARLA world coordinates.

This builder converts them to the existing ScenarioGenerator frame:

    +x   = ego-initial forward
    +y   = ego-initial left
    +yaw = counter-clockwise / left relative to ego initial heading

The downstream TCP/NEAT runners already map this frame back into CARLA.

Scenario
--------
A Tesla starts ahead of the ego and follows the same CARLA route.
Traffic light 709 is red during the approach.

The lead vehicle:
    cruises
    smoothly decelerates to a stop before the signal
    waits
    accelerates when the signal turns green
    follows the right-turn portion of the route
    continues to near the route end

The traffic-light schedule is written as a sidecar. The driving-model
runner will apply that schedule identically in CARLA and HE conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scenario_generator.schema.scenario_schema_v2 import (
    ScenarioSpecV2,
)

from scenario_generator.trajectory.trajectory_resolver_v2 import (
    V2TrajectoryResolver,
)


# ============================================================
# Helpers
# ============================================================

def normalize_angle_180(angle_deg: float) -> float:
    return (
        float(angle_deg)
        + 180.0
    ) % 360.0 - 180.0


def shortest_angle_delta_deg(
    from_deg: float,
    to_deg: float,
) -> float:
    return normalize_angle_180(
        float(to_deg)
        - float(from_deg)
    )


def interpolate_angle_deg(
    yaw0: float,
    yaw1: float,
    u: float,
) -> float:
    return normalize_angle_180(
        float(yaw0)
        + float(u)
        * shortest_angle_delta_deg(
            yaw0,
            yaw1,
        )
    )


def load_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as fp:
        return list(
            csv.DictReader(fp)
        )


def write_csv(
    path: Path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            data,
            fp,
            indent=2,
        )
        fp.write("\n")


# ============================================================
# Route
# ============================================================

def parse_route_rows(rows):
    route = []

    for row in rows:
        route.append(
            {
                "route_idx":
                    int(
                        row["route_idx"]
                    ),

                "s_m":
                    float(
                        row[
                            "route_progress_m"
                        ]
                    ),

                "x":
                    float(row["x"]),

                "y":
                    float(row["y"]),

                "z":
                    float(row["z"]),

                "yaw_deg":
                    float(
                        row["yaw_deg"]
                    ),

                "road_option":
                    row[
                        "road_option"
                    ],

                "road_id":
                    int(
                        row["road_id"]
                    ),

                "lane_id":
                    int(
                        row["lane_id"]
                    ),

                "is_junction":
                    str(
                        row["is_junction"]
                    ).strip().lower()
                    in (
                        "1",
                        "true",
                        "yes",
                    ),
            }
        )

    route.sort(
        key=lambda row:
            row["s_m"]
    )

    if len(route) < 2:
        raise RuntimeError(
            "Route must contain at least "
            "two samples."
        )

    return route


def route_state_at_s(
    route,
    s_m: float,
):
    s_m = max(
        route[0]["s_m"],
        min(
            float(s_m),
            route[-1]["s_m"],
        ),
    )

    if s_m <= route[0]["s_m"]:
        return dict(route[0])

    if s_m >= route[-1]["s_m"]:
        return dict(route[-1])

    lo = 0
    hi = len(route) - 1

    while lo + 1 < hi:
        mid = (
            lo + hi
        ) // 2

        if route[mid]["s_m"] <= s_m:
            lo = mid
        else:
            hi = mid

    a = route[lo]
    b = route[lo + 1]

    ds = (
        b["s_m"]
        - a["s_m"]
    )

    if abs(ds) <= 1e-9:
        u = 0.0
    else:
        u = (
            s_m
            - a["s_m"]
        ) / ds

    return {
        "route_idx":
            a["route_idx"],

        "s_m":
            s_m,

        "x":
            a["x"]
            + u
            * (
                b["x"]
                - a["x"]
            ),

        "y":
            a["y"]
            + u
            * (
                b["y"]
                - a["y"]
            ),

        "z":
            a["z"]
            + u
            * (
                b["z"]
                - a["z"]
            ),

        "yaw_deg":
            interpolate_angle_deg(
                a["yaw_deg"],
                b["yaw_deg"],
                u,
            ),

        "road_option":
            (
                b["road_option"]
                if u >= 0.5
                else a["road_option"]
            ),

        "road_id":
            (
                b["road_id"]
                if u >= 0.5
                else a["road_id"]
            ),

        "lane_id":
            (
                b["lane_id"]
                if u >= 0.5
                else a["lane_id"]
            ),

        "is_junction":
            (
                b["is_junction"]
                if u >= 0.5
                else a["is_junction"]
            ),
    }


# ============================================================
# CARLA world -> ScenarioGenerator ego-initial frame
# ============================================================

def world_to_sg(
    world_x,
    world_y,
    world_yaw_deg,
    origin_x,
    origin_y,
    origin_yaw_deg,
):
    yaw = math.radians(
        float(origin_yaw_deg)
    )

    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)

    right_x = -math.sin(yaw)
    right_y = math.cos(yaw)

    dx = (
        float(world_x)
        - float(origin_x)
    )

    dy = (
        float(world_y)
        - float(origin_y)
    )

    sg_x = (
        dx * forward_x
        + dy * forward_y
    )

    # Existing runners use:
    #
    # world = origin
    #       + forward * sg_x
    #       - right   * sg_y
    #
    # therefore:
    sg_y = -(
        dx * right_x
        + dy * right_y
    )

    # Existing runners use:
    #
    # world_yaw =
    #     initial_ego_yaw - sg_yaw
    sg_yaw = normalize_angle_180(
        float(origin_yaw_deg)
        - float(world_yaw_deg)
    )

    return (
        float(sg_x),
        float(sg_y),
        float(sg_yaw),
    )


# ============================================================
# Lead-vehicle longitudinal schedule
# ============================================================

def make_schedule(
    initial_s_m,
    stop_s_m,
    cruise_speed_mps,
    brake_duration_s,
    green_time_s,
    accel_duration_s,
):
    """
    Constant-speed cruise -> constant deceleration -> stop/hold ->
    constant acceleration -> constant-speed cruise.

    Braking starts at exactly the route progress required to stop
    at stop_s_m after brake_duration_s.
    """

    v = float(
        cruise_speed_mps
    )

    brake_duration_s = float(
        brake_duration_s
    )

    accel_duration_s = float(
        accel_duration_s
    )

    if v <= 0.0:
        raise ValueError(
            "cruise_speed_mps must be > 0"
        )

    if brake_duration_s <= 0.0:
        raise ValueError(
            "brake_duration_s must be > 0"
        )

    if accel_duration_s <= 0.0:
        raise ValueError(
            "accel_duration_s must be > 0"
        )

    braking_distance = (
        0.5
        * v
        * brake_duration_s
    )

    brake_start_s_m = (
        float(stop_s_m)
        - braking_distance
    )

    if (
        brake_start_s_m
        <= float(initial_s_m)
    ):
        raise ValueError(
            "Not enough distance for the "
            "requested cruise/braking profile."
        )

    brake_start_time_s = (
        (
            brake_start_s_m
            - float(initial_s_m)
        )
        / v
    )

    stop_time_s = (
        brake_start_time_s
        + brake_duration_s
    )

    if (
        float(green_time_s)
        < stop_time_s
    ):
        raise ValueError(
            "green_time_s occurs before "
            "the lead vehicle finishes braking. "
            f"stop_time_s={stop_time_s:.3f}"
        )

    braking_accel = (
        -v
        / brake_duration_s
    )

    acceleration = (
        v
        / accel_duration_s
    )

    accel_end_time_s = (
        float(green_time_s)
        + accel_duration_s
    )

    accel_distance = (
        0.5
        * v
        * accel_duration_s
    )

    def state(t_s: float):
        t_s = float(t_s)

        # ----------------------------------------------------
        # Cruise
        # ----------------------------------------------------
        if (
            t_s
            <= brake_start_time_s
        ):
            return {
                "phase":
                    "cruise",

                "s_m":
                    float(initial_s_m)
                    + v * t_s,

                "speed_mps":
                    v,
            }

        # ----------------------------------------------------
        # Braking
        # ----------------------------------------------------
        if t_s <= stop_time_s:
            dt = (
                t_s
                - brake_start_time_s
            )

            speed = max(
                0.0,
                v
                + braking_accel
                * dt,
            )

            s_m = (
                brake_start_s_m
                + v * dt
                + 0.5
                * braking_accel
                * dt * dt
            )

            return {
                "phase":
                    "braking",

                "s_m":
                    min(
                        float(stop_s_m),
                        s_m,
                    ),

                "speed_mps":
                    speed,
            }

        # ----------------------------------------------------
        # Stopped at red
        # ----------------------------------------------------
        if t_s <= float(
            green_time_s
        ):
            return {
                "phase":
                    "stopped_red",

                "s_m":
                    float(stop_s_m),

                "speed_mps":
                    0.0,
            }

        # ----------------------------------------------------
        # Acceleration after green
        # ----------------------------------------------------
        if t_s <= accel_end_time_s:
            dt = (
                t_s
                - float(
                    green_time_s
                )
            )

            speed = min(
                v,
                acceleration
                * dt,
            )

            s_m = (
                float(stop_s_m)
                + 0.5
                * acceleration
                * dt * dt
            )

            return {
                "phase":
                    "accelerating_green",

                "s_m":
                    s_m,

                "speed_mps":
                    speed,
            }

        # ----------------------------------------------------
        # Cruise after acceleration
        # ----------------------------------------------------
        dt = (
            t_s
            - accel_end_time_s
        )

        return {
            "phase":
                "post_green_cruise",

            "s_m":
                float(stop_s_m)
                + accel_distance
                + v * dt,

            "speed_mps":
                v,
        }

    return {
        "brake_start_s_m":
            brake_start_s_m,

        "braking_distance_m":
            braking_distance,

        "brake_start_time_s":
            brake_start_time_s,

        "stop_time_s":
            stop_time_s,

        "green_time_s":
            float(
                green_time_s
            ),

        "accel_end_time_s":
            accel_end_time_s,

        "state":
            state,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--route-csv",
        default=(
            "driving_models/common/outputs/"
            "town10_spawn10_to45_route_v1/"
            "route.csv"
        ),
    )

    parser.add_argument(
        "--traffic-lights-csv",
        default=(
            "driving_models/common/outputs/"
            "town10_spawn10_to45_route_v1/"
            "traffic_lights.csv"
        ),
    )

    parser.add_argument(
        "--scenario-id",
        default=(
            "signalized_lead_follow_001"
        ),
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
        "--destination-index",
        type=int,
        default=45,
    )

    parser.add_argument(
        "--traffic-light-id",
        type=int,
        default=709,
    )

    parser.add_argument(
        "--duration-s",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--keyframe-step-s",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--lead-initial-route-progress-m",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--lead-cruise-speed-mps",
        type=float,
        default=5.5,
    )

    parser.add_argument(
        "--stop-before-light-m",
        type=float,
        default=3.5,
    )

    parser.add_argument(
        "--brake-duration-s",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--green-time-s",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--accel-duration-s",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--output-spec",
        default=(
            "ScenarioGenerator/outputs/"
            "v2_specs/"
            "signalized_lead_follow_001."
            "scenario_v2.json"
        ),
    )

    parser.add_argument(
        "--output-resolved",
        default=(
            "ScenarioGenerator/outputs/"
            "v2_resolved/"
            "signalized_lead_follow_001."
            "resolved_v2.json"
        ),
    )

    parser.add_argument(
        "--output-environment",
        default=(
            "ScenarioGenerator/outputs/"
            "v2_environment/"
            "signalized_lead_follow_001."
            "environment_v1.json"
        ),
    )

    parser.add_argument(
        "--output-trajectory-csv",
        default=(
            "ScenarioGenerator/outputs/"
            "v2_debug/"
            "signalized_lead_follow_001."
            "actor_trajectory.csv"
        ),
    )

    args = parser.parse_args()

    route_csv = Path(
        args.route_csv
    )

    traffic_lights_csv = Path(
        args.traffic_lights_csv
    )

    route = parse_route_rows(
        load_csv(
            route_csv
        )
    )

    traffic_lights = load_csv(
        traffic_lights_csv
    )

    target_light = None

    for row in traffic_lights:
        if (
            int(
                row[
                    "traffic_light_id"
                ]
            )
            == int(
                args.traffic_light_id
            )
        ):
            target_light = row
            break

    if target_light is None:
        raise RuntimeError(
            f"Traffic light "
            f"{args.traffic_light_id} "
            f"not found in "
            f"{traffic_lights_csv}"
        )

    light_s_m = float(
        target_light[
            "route_progress_m"
        ]
    )

    stop_s_m = (
        light_s_m
        - float(
            args.stop_before_light_m
        )
    )

    if stop_s_m <= 0.0:
        raise ValueError(
            "Computed stop progress "
            "is invalid."
        )

    route_origin = route[0]

    origin_x = float(
        route_origin["x"]
    )

    origin_y = float(
        route_origin["y"]
    )

    origin_yaw = float(
        route_origin[
            "yaw_deg"
        ]
    )

    schedule = make_schedule(
        initial_s_m=(
            args
            .lead_initial_route_progress_m
        ),

        stop_s_m=
            stop_s_m,

        cruise_speed_mps=(
            args
            .lead_cruise_speed_mps
        ),

        brake_duration_s=(
            args
            .brake_duration_s
        ),

        green_time_s=(
            args
            .green_time_s
        ),

        accel_duration_s=(
            args
            .accel_duration_s
        ),
    )

    # ========================================================
    # Dense timed route keyframes
    # ========================================================

    keyframes = []
    trajectory_rows = []

    step_s = float(
        args.keyframe_step_s
    )

    if step_s <= 0.0:
        raise ValueError(
            "keyframe-step-s must be > 0"
        )

    count = int(
        round(
            float(
                args.duration_s
            )
            / step_s
        )
    )

    times = [
        i * step_s
        for i in range(
            count + 1
        )
    ]

    # Ensure exact duration is present.
    if abs(
        times[-1]
        - float(
            args.duration_s
        )
    ) > 1e-9:
        times.append(
            float(
                args.duration_s
            )
        )

    for t_s in times:
        longitudinal = (
            schedule["state"](
                t_s
            )
        )

        route_state = (
            route_state_at_s(
                route,
                longitudinal[
                    "s_m"
                ],
            )
        )

        (
            sg_x,
            sg_y,
            sg_yaw,
        ) = world_to_sg(
            world_x=
                route_state["x"],

            world_y=
                route_state["y"],

            world_yaw_deg=
                route_state[
                    "yaw_deg"
                ],

            origin_x=
                origin_x,

            origin_y=
                origin_y,

            origin_yaw_deg=
                origin_yaw,
        )

        keyframes.append(
            {
                "t_s":
                    round(
                        float(t_s),
                        6,
                    ),

                "x_m":
                    sg_x,

                "y_m":
                    sg_y,

                "yaw_deg":
                    sg_yaw,
            }
        )

        trajectory_rows.append(
            {
                "t_s":
                    round(
                        float(t_s),
                        6,
                    ),

                "phase":
                    longitudinal[
                        "phase"
                    ],

                "route_progress_m":
                    longitudinal[
                        "s_m"
                    ],

                "target_speed_mps":
                    longitudinal[
                        "speed_mps"
                    ],

                "world_x":
                    route_state["x"],

                "world_y":
                    route_state["y"],

                "world_yaw_deg":
                    route_state[
                        "yaw_deg"
                    ],

                "sg_x_m":
                    sg_x,

                "sg_y_m":
                    sg_y,

                "sg_yaw_deg":
                    sg_yaw,

                "road_option":
                    route_state[
                        "road_option"
                    ],

                "road_id":
                    route_state[
                        "road_id"
                    ],

                "lane_id":
                    route_state[
                        "lane_id"
                    ],

                "is_junction":
                    route_state[
                        "is_junction"
                    ],
            }
        )

    # ========================================================
    # Spawn = exact first keyframe pose
    # ========================================================

    first = keyframes[0]

    scenario_dict = {
        "schema_version":
            "2.0",

        "scenario_id":
            args.scenario_id,

        "description":
            (
                "Long-horizon signalized lead-follow scenario. "
                "A Tesla follows the Town10HD_Opt spawn-10 to "
                "destination-45 route, stops for traffic light 709, "
                "resumes on green, performs the route's right turn, "
                "and continues. Ego is externally controlled by the "
                "driving model during execution; the static ego "
                "motion here is a resolver placeholder."
            ),

        "duration_s":
            float(
                args.duration_s
            ),

        "fps":
            int(
                args.fps
            ),

        "seed":
            0,

        "ego": {
            "actor_id":
                "ego",

            "spawn": {
                "x_m": 0.0,
                "y_m": 0.0,
                "yaw_deg": 0.0,
            },

            "motion": {
                "mode":
                    "maneuver",

                "maneuver":
                    "static",

                "speed_mps":
                    0.0,

                "start_time_s":
                    0.0,
            },
        },

        "actors": [
            {
                "actor_id":
                    "adv_lead",

                "actor_type":
                    "vehicle",

                "role":
                    "traffic",

                "asset_key":
                    "vehicle.tesla.model3",

                "dimensions_m": {
                    "length_m":
                        4.792,

                    "width_m":
                        2.163,

                    "height_m":
                        1.488,
                },

                "spawn": {
                    "x_m":
                        first["x_m"],

                    "y_m":
                        first["y_m"],

                    "yaw_deg":
                        first[
                            "yaw_deg"
                        ],
                },

                "lifecycle": {
                    "spawn_time_s":
                        0.0,

                    "despawn_time_s":
                        None,
                },

                "motion": {
                    "mode":
                        "keyframes",

                    "keyframes":
                        keyframes,

                    # Dense 0.10 s samples already follow the
                    # route accurately. Linear interpolation
                    # avoids adding extra smoothstep timing.
                    "interpolation":
                        "linear",
                },
            }
        ],

        # Legacy camera metadata is intentionally retained
        # for ScenarioSchema v2 compatibility.
        # TCP/NEAT ignore this and use native cameras.
        "camera": {
            "image_width":
                1280,

            "image_height":
                720,

            "fov_deg":
                90.0,

            "x_m":
                1.5,

            "y_m":
                0.0,

            "z_m":
                1.6,

            "pitch_deg":
                0.0,

            "yaw_deg":
                0.0,

            "roll_deg":
                0.0,
        },
    }

    scenario = (
        ScenarioSpecV2
        .model_validate(
            scenario_dict
        )
    )

    resolver = (
        V2TrajectoryResolver()
    )

    resolved = resolver.resolve(
        scenario
    )

    # ========================================================
    # Environment sidecar
    # ========================================================

    environment = {
        "schema_version":
            "1.0",

        "scenario_id":
            args.scenario_id,

        "town":
            args.town,

        "route": {
            "spawn_index":
                int(
                    args.spawn_index
                ),

            "destination_index":
                int(
                    args.destination_index
                ),

            "route_csv":
                str(
                    route_csv.resolve()
                ),
        },

        "traffic_lights": [
            {
                "traffic_light_id":
                    int(
                        args.traffic_light_id
                    ),

                "route_progress_m":
                    light_s_m,

                "stop_route_progress_m":
                    stop_s_m,

                "phases": [
                    {
                        "state":
                            "Red",

                        "start_s":
                            0.0,

                        "end_s":
                            float(
                                args
                                .green_time_s
                            ),
                    },
                    {
                        "state":
                            "Green",

                        "start_s":
                            float(
                                args
                                .green_time_s
                            ),

                        "end_s":
                            float(
                                args
                                .duration_s
                            ),
                    },
                ],
            }
        ],

        "weather": {
            "preset":
                "ClearNoon",
        },

        "lead_motion": {
            "actor_id":
                "adv_lead",

            "initial_route_progress_m":
                float(
                    args
                    .lead_initial_route_progress_m
                ),

            "cruise_speed_mps":
                float(
                    args
                    .lead_cruise_speed_mps
                ),

            "brake_start_route_progress_m":
                schedule[
                    "brake_start_s_m"
                ],

            "brake_start_time_s":
                schedule[
                    "brake_start_time_s"
                ],

            "stop_route_progress_m":
                stop_s_m,

            "stop_time_s":
                schedule[
                    "stop_time_s"
                ],

            "green_time_s":
                schedule[
                    "green_time_s"
                ],

            "accel_end_time_s":
                schedule[
                    "accel_end_time_s"
                ],
        },
    }

    # ========================================================
    # Save
    # ========================================================

    output_spec = Path(
        args.output_spec
    )

    output_resolved = Path(
        args.output_resolved
    )

    output_environment = Path(
        args.output_environment
    )

    output_trajectory = Path(
        args.output_trajectory_csv
    )

    save_json(
        output_spec,
        scenario.model_dump(
            mode="json"
        ),
    )

    save_json(
        output_resolved,
        resolved.model_dump(
            mode="json"
        ),
    )

    save_json(
        output_environment,
        environment,
    )

    write_csv(
        output_trajectory,
        trajectory_rows,
    )

    # ========================================================
    # Console
    # ========================================================

    final_state = (
        schedule["state"](
            args.duration_s
        )
    )

    print()
    print("=" * 92)
    print(
        "SIGNALIZED LEAD-FOLLOW SCENARIO BUILT"
    )
    print("=" * 92)

    print(
        f"scenario             : "
        f"{args.scenario_id}"
    )

    print(
        f"duration             : "
        f"{args.duration_s:.2f} s "
        f"@ {args.fps} FPS"
    )

    print(
        f"route                : "
        f"{args.town} "
        f"{args.spawn_index}"
        f" -> "
        f"{args.destination_index}"
    )

    print(
        f"traffic light        : "
        f"{args.traffic_light_id} "
        f"at s={light_s_m:.2f} m"
    )

    print(
        f"lead starts          : "
        f"s="
        f"{args.lead_initial_route_progress_m:.2f} m"
    )

    print(
        f"lead cruise speed    : "
        f"{args.lead_cruise_speed_mps:.2f} m/s"
    )

    print(
        f"braking starts       : "
        f"t="
        f"{schedule['brake_start_time_s']:.2f} s "
        f"s="
        f"{schedule['brake_start_s_m']:.2f} m"
    )

    print(
        f"lead stops           : "
        f"t="
        f"{schedule['stop_time_s']:.2f} s "
        f"s="
        f"{stop_s_m:.2f} m"
    )

    print(
        f"signal green         : "
        f"t="
        f"{schedule['green_time_s']:.2f} s"
    )

    print(
        f"acceleration ends    : "
        f"t="
        f"{schedule['accel_end_time_s']:.2f} s"
    )

    print(
        f"lead final progress  : "
        f"s="
        f"{final_state['s_m']:.2f} m"
    )

    print(
        f"keyframes            : "
        f"{len(keyframes)}"
    )

    print(
        f"resolved actor frames: "
        f"{len(resolved.actor_frames)}"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        " ",
        output_spec,
    )

    print(
        " ",
        output_resolved,
    )

    print(
        " ",
        output_environment,
    )

    print(
        " ",
        output_trajectory,
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
