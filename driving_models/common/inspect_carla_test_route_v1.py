"""
inspect_carla_test_route_v1.py

Inspect a CARLA route for HE benchmark scenario design.

Exports:
    route.csv
    route_events.csv
    traffic_lights.csv
    route_summary.json
    route_topdown.png

The script does NOT run a driving model and does NOT spawn an ego vehicle.
It only loads the CARLA map and inspects its route topology.

Primary HE benchmark route:
    Town10HD_Opt
    spawn 10
    destination 45
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import carla


# ============================================================
# CARLA agents
# ============================================================

def import_global_route_planner(pythonapi_hint=None):

    def _load():
        from agents.navigation.global_route_planner import (
            GlobalRoutePlanner
        )
        return GlobalRoutePlanner

    try:
        return _load()

    except ModuleNotFoundError:
        pass

    candidates = []

    if pythonapi_hint:
        candidates.append(
            Path(pythonapi_hint).resolve()
        )

    for path in candidates:

        possible = [
            path,
            path / "carla",
        ]

        for candidate in possible:

            if not (
                candidate / "agents"
            ).is_dir():
                continue

            if str(candidate) not in sys.path:
                sys.path.insert(
                    0,
                    str(candidate),
                )

            try:
                return _load()

            except ModuleNotFoundError:
                pass

    raise RuntimeError(
        "Could not import CARLA agents.\n"
        "Pass:\n"
        "--carla-pythonapi "
        "E:\\Carla\\Carla_0.9.15\\PythonAPI\\carla"
    )


# ============================================================
# Basic geometry
# ============================================================

def distance_2d(a, b):

    return math.hypot(
        float(a.x) - float(b.x),
        float(a.y) - float(b.y),
    )


def command_name(command):

    if command is None:
        return "UNKNOWN"

    name = getattr(
        command,
        "name",
        None,
    )

    if name is not None:
        return str(name)

    text = str(command)

    if "." in text:
        text = text.split(".")[-1]

    return text


def route_cumulative_distance(route):

    cumulative = [0.0]

    for i in range(
        1,
        len(route),
    ):

        a = (
            route[i - 1][0]
            .transform.location
        )

        b = (
            route[i][0]
            .transform.location
        )

        cumulative.append(
            cumulative[-1]
            + distance_2d(a, b)
        )

    return cumulative


def nearest_route_point(
    route,
    location,
):

    best_idx = None
    best_distance = float("inf")

    for idx, (
        waypoint,
        _,
    ) in enumerate(route):

        d = distance_2d(
            waypoint.transform.location,
            location,
        )

        if d < best_distance:

            best_distance = d
            best_idx = idx

    return (
        best_idx,
        best_distance,
    )


# ============================================================
# Traffic lights
# ============================================================

def get_stop_waypoints(light):

    """
    CARLA 0.9.15 normally provides get_stop_waypoints().

    Keep this defensive so the inspector still works if a
    particular CARLA build exposes a slightly different API.
    """

    if not hasattr(
        light,
        "get_stop_waypoints",
    ):
        return []

    try:

        waypoints = (
            light.get_stop_waypoints()
        )

        if waypoints is None:
            return []

        return list(waypoints)

    except RuntimeError:
        return []

    except Exception as exc:

        print(
            f"[warning] traffic light "
            f"{light.id}: "
            f"get_stop_waypoints failed: "
            f"{exc}"
        )

        return []


def traffic_light_rows(
    world,
    route,
    cumulative,
    max_route_distance_m,
):

    rows = []

    lights = (
        world.get_actors()
        .filter(
            "traffic.traffic_light*"
        )
    )

    for light in lights:

        light_tf = (
            light.get_transform()
        )

        stop_waypoints = (
            get_stop_waypoints(
                light
            )
        )

        candidates = []

        # Prefer the actual stop waypoint.
        for stop_wp in stop_waypoints:

            loc = (
                stop_wp
                .transform
                .location
            )

            route_idx, route_distance = (
                nearest_route_point(
                    route,
                    loc,
                )
            )

            candidates.append(
                (
                    route_distance,
                    route_idx,
                    stop_wp,
                )
            )

        # Fallback to traffic-light actor location.
        if not candidates:

            route_idx, route_distance = (
                nearest_route_point(
                    route,
                    light_tf.location,
                )
            )

            candidates.append(
                (
                    route_distance,
                    route_idx,
                    None,
                )
            )

        (
            nearest_distance,
            nearest_idx,
            nearest_stop_wp,
        ) = min(
            candidates,
            key=lambda row: row[0],
        )

        relevant = (
            nearest_distance
            <= float(
                max_route_distance_m
            )
        )

        if nearest_stop_wp is not None:

            stop_tf = (
                nearest_stop_wp
                .transform
            )

            stop_x = float(
                stop_tf.location.x
            )

            stop_y = float(
                stop_tf.location.y
            )

            stop_z = float(
                stop_tf.location.z
            )

            stop_yaw = float(
                stop_tf.rotation.yaw
            )

            stop_road_id = int(
                nearest_stop_wp.road_id
            )

            stop_lane_id = int(
                nearest_stop_wp.lane_id
            )

        else:

            stop_x = float(
                light_tf.location.x
            )

            stop_y = float(
                light_tf.location.y
            )

            stop_z = float(
                light_tf.location.z
            )

            stop_yaw = float(
                light_tf.rotation.yaw
            )

            stop_road_id = None
            stop_lane_id = None

        state = str(
            light.get_state()
        )

        if "." in state:
            state = (
                state
                .split(".")[-1]
            )

        rows.append(
            {
                "traffic_light_id":
                    int(light.id),

                "initial_state":
                    state,

                "relevant_to_route":
                    bool(relevant),

                "route_idx":
                    int(nearest_idx),

                "route_progress_m":
                    float(
                        cumulative[
                            nearest_idx
                        ]
                    ),

                "distance_to_route_m":
                    float(
                        nearest_distance
                    ),

                "light_x":
                    float(
                        light_tf.location.x
                    ),

                "light_y":
                    float(
                        light_tf.location.y
                    ),

                "light_z":
                    float(
                        light_tf.location.z
                    ),

                "stop_x":
                    stop_x,

                "stop_y":
                    stop_y,

                "stop_z":
                    stop_z,

                "stop_yaw_deg":
                    stop_yaw,

                "stop_road_id":
                    stop_road_id,

                "stop_lane_id":
                    stop_lane_id,

                "num_stop_waypoints":
                    len(stop_waypoints),
            }
        )

    rows.sort(
        key=lambda row:
            (
                not row[
                    "relevant_to_route"
                ],
                row[
                    "route_progress_m"
                ],
            )
    )

    return rows


# ============================================================
# Route events
# ============================================================

def get_junction_id(waypoint):

    if not waypoint.is_junction:
        return None

    try:

        junction = (
            waypoint.get_junction()
        )

        if junction is None:
            return None

        return int(junction.id)

    except Exception:
        return None


def build_route_events(
    route,
    cumulative,
    traffic_lights,
):

    events = []

    # --------------------------------------------------------
    # Junction entry / exit
    # --------------------------------------------------------

    previous_junction = False

    for idx, (
        waypoint,
        road_option,
    ) in enumerate(route):

        current_junction = bool(
            waypoint.is_junction
        )

        if (
            current_junction
            and
            not previous_junction
        ):

            tf = waypoint.transform

            events.append(
                {
                    "event_type":
                        "junction_entry",

                    "route_idx":
                        idx,

                    "route_progress_m":
                        cumulative[idx],

                    "x":
                        tf.location.x,

                    "y":
                        tf.location.y,

                    "yaw_deg":
                        tf.rotation.yaw,

                    "road_option":
                        command_name(
                            road_option
                        ),

                    "details":
                        f"junction_id="
                        f"{get_junction_id(waypoint)}",
                }
            )

        if (
            not current_junction
            and
            previous_junction
            and
            idx > 0
        ):

            previous_wp = (
                route[idx - 1][0]
            )

            tf = (
                previous_wp.transform
            )

            events.append(
                {
                    "event_type":
                        "junction_exit",

                    "route_idx":
                        idx - 1,

                    "route_progress_m":
                        cumulative[
                            idx - 1
                        ],

                    "x":
                        tf.location.x,

                    "y":
                        tf.location.y,

                    "yaw_deg":
                        tf.rotation.yaw,

                    "road_option":
                        command_name(
                            route[
                                idx - 1
                            ][1]
                        ),

                    "details":
                        f"junction_id="
                        f"{get_junction_id(previous_wp)}",
                }
            )

        previous_junction = (
            current_junction
        )

    # --------------------------------------------------------
    # Road-option transitions
    # --------------------------------------------------------

    previous_command = None

    for idx, (
        waypoint,
        road_option,
    ) in enumerate(route):

        current_command = (
            command_name(
                road_option
            )
        )

        if (
            current_command
            != previous_command
        ):

            tf = waypoint.transform

            events.append(
                {
                    "event_type":
                        "road_option_change",

                    "route_idx":
                        idx,

                    "route_progress_m":
                        cumulative[idx],

                    "x":
                        tf.location.x,

                    "y":
                        tf.location.y,

                    "yaw_deg":
                        tf.rotation.yaw,

                    "road_option":
                        current_command,

                    "details":
                        (
                            f"{previous_command}"
                            f" -> "
                            f"{current_command}"
                        ),
                }
            )

        previous_command = (
            current_command
        )

    # --------------------------------------------------------
    # Traffic-light stop positions
    # --------------------------------------------------------

    for light in traffic_lights:

        if not light[
            "relevant_to_route"
        ]:
            continue

        events.append(
            {
                "event_type":
                    "traffic_light",

                "route_idx":
                    light["route_idx"],

                "route_progress_m":
                    light[
                        "route_progress_m"
                    ],

                "x":
                    light["stop_x"],

                "y":
                    light["stop_y"],

                "yaw_deg":
                    light[
                        "stop_yaw_deg"
                    ],

                "road_option":
                    "",

                "details":
                    (
                        f"traffic_light_id="
                        f"{light['traffic_light_id']}"
                    ),
            }
        )

    events.sort(
        key=lambda row:
            (
                float(
                    row[
                        "route_progress_m"
                    ]
                ),
                row[
                    "event_type"
                ],
            )
    )

    return events


# ============================================================
# CSV
# ============================================================

def write_csv(
    path,
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
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Top-down map
# ============================================================

def save_topdown(
    path,
    route,
    cumulative,
    events,
):

    try:
        from PIL import (
            Image,
            ImageDraw,
        )

    except ImportError:

        print(
            "[warning] Pillow not installed; "
            "skipping route_topdown.png"
        )

        return

    points = [
        (
            float(
                wp.transform.location.x
            ),
            float(
                wp.transform.location.y
            ),
        )
        for wp, _ in route
    ]

    xs = [
        point[0]
        for point in points
    ]

    ys = [
        point[1]
        for point in points
    ]

    margin_world = 15.0

    min_x = min(xs) - margin_world
    max_x = max(xs) + margin_world

    min_y = min(ys) - margin_world
    max_y = max(ys) + margin_world

    width = 1400
    height = 1000
    margin_px = 60

    usable_w = (
        width
        - 2 * margin_px
    )

    usable_h = (
        height
        - 2 * margin_px
    )

    scale_x = (
        usable_w
        / max(
            max_x - min_x,
            1e-6,
        )
    )

    scale_y = (
        usable_h
        / max(
            max_y - min_y,
            1e-6,
        )
    )

    scale = min(
        scale_x,
        scale_y,
    )

    def to_image(
        x,
        y,
    ):

        px = (
            margin_px
            + (
                float(x)
                - min_x
            ) * scale
        )

        # Flip world Y for image display.
        py = (
            height
            - margin_px
            - (
                float(y)
                - min_y
            ) * scale
        )

        return (
            int(round(px)),
            int(round(py)),
        )

    image = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        image
    )

    route_pixels = [
        to_image(
            x,
            y,
        )
        for x, y in points
    ]

    if len(route_pixels) >= 2:

        draw.line(
            route_pixels,
            fill="black",
            width=5,
        )

    # Start.
    sx, sy = (
        route_pixels[0]
    )

    draw.ellipse(
        (
            sx - 9,
            sy - 9,
            sx + 9,
            sy + 9,
        ),
        fill="green",
    )

    draw.text(
        (
            sx + 12,
            sy - 8,
        ),
        "START",
        fill="green",
    )

    # Destination.
    ex, ey = (
        route_pixels[-1]
    )

    draw.ellipse(
        (
            ex - 9,
            ey - 9,
            ex + 9,
            ey + 9,
        ),
        fill="blue",
    )

    draw.text(
        (
            ex + 12,
            ey - 8,
        ),
        "DEST",
        fill="blue",
    )

    # Events.
    for event in events:

        px, py = to_image(
            event["x"],
            event["y"],
        )

        event_type = (
            event["event_type"]
        )

        if event_type == "traffic_light":

            fill = "red"
            label = (
                f"TL @ "
                f"{event['route_progress_m']:.1f}m"
            )

        elif (
            event_type
            == "junction_entry"
        ):

            fill = "orange"
            label = (
                f"J-IN "
                f"{event['route_progress_m']:.1f}m"
            )

        elif (
            event_type
            == "junction_exit"
        ):

            fill = "purple"
            label = (
                f"J-OUT "
                f"{event['route_progress_m']:.1f}m"
            )

        elif (
            event_type
            == "road_option_change"
            and
            event["road_option"]
            in (
                "RIGHT",
                "LEFT",
            )
        ):

            fill = "darkred"

            label = (
                f"{event['road_option']} "
                f"{event['route_progress_m']:.1f}m"
            )

        else:
            continue

        draw.rectangle(
            (
                px - 6,
                py - 6,
                px + 6,
                py + 6,
            ),
            fill=fill,
        )

        draw.text(
            (
                px + 9,
                py - 7,
            ),
            label,
            fill=fill,
        )

    draw.text(
        (
            20,
            20,
        ),
        (
            f"Route length: "
            f"{cumulative[-1]:.1f} m"
        ),
        fill="black",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        path
    )


# ============================================================
# Main
# ============================================================

def main():

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
        "--route-sampling-resolution",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--traffic-light-route-distance-m",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--carla-pythonapi",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "driving_models/common/outputs/"
            "town10_spawn10_to45_route_v1"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    GlobalRoutePlanner = (
        import_global_route_planner(
            args.carla_pythonapi
        )
    )

    client = carla.Client(
        args.host,
        args.port,
    )

    client.set_timeout(
        20.0
    )

    print(
        "[CARLA] loading:",
        args.town,
    )

    world = client.load_world(
        args.town
    )

    carla_map = (
        world.get_map()
    )

    spawn_points = (
        carla_map
        .get_spawn_points()
    )

    start_idx = (
        int(args.spawn_index)
        % len(spawn_points)
    )

    destination_idx = (
        int(
            args.destination_index
        )
        % len(spawn_points)
    )

    if (
        start_idx
        == destination_idx
    ):
        raise ValueError(
            "Start and destination "
            "spawn indices are equal."
        )

    grp = GlobalRoutePlanner(
        carla_map,
        float(
            args.route_sampling_resolution
        ),
    )

    route = grp.trace_route(
        spawn_points[
            start_idx
        ].location,
        spawn_points[
            destination_idx
        ].location,
    )

    if len(route) < 2:
        raise RuntimeError(
            "No usable route."
        )

    cumulative = (
        route_cumulative_distance(
            route
        )
    )

    # ========================================================
    # Route rows
    # ========================================================

    route_rows = []

    for idx, (
        waypoint,
        road_option,
    ) in enumerate(route):

        tf = waypoint.transform

        route_rows.append(
            {
                "route_idx":
                    idx,

                "route_progress_m":
                    cumulative[idx],

                "x":
                    float(
                        tf.location.x
                    ),

                "y":
                    float(
                        tf.location.y
                    ),

                "z":
                    float(
                        tf.location.z
                    ),

                "yaw_deg":
                    float(
                        tf.rotation.yaw
                    ),

                "road_option":
                    command_name(
                        road_option
                    ),

                "road_id":
                    int(
                        waypoint.road_id
                    ),

                "section_id":
                    int(
                        waypoint.section_id
                    ),

                "lane_id":
                    int(
                        waypoint.lane_id
                    ),

                "waypoint_s":
                    float(
                        waypoint.s
                    ),

                "lane_width_m":
                    float(
                        waypoint.lane_width
                    ),

                "is_junction":
                    bool(
                        waypoint.is_junction
                    ),

                "junction_id":
                    get_junction_id(
                        waypoint
                    ),
            }
        )

    # ========================================================
    # Traffic lights
    # ========================================================

    light_rows = (
        traffic_light_rows(
            world=world,

            route=route,

            cumulative=cumulative,

            max_route_distance_m=(
                args
                .traffic_light_route_distance_m
            ),
        )
    )

    relevant_lights = [
        row
        for row in light_rows
        if row[
            "relevant_to_route"
        ]
    ]

    # ========================================================
    # Events
    # ========================================================

    events = build_route_events(
        route=route,
        cumulative=cumulative,
        traffic_lights=light_rows,
    )

    # ========================================================
    # Save
    # ========================================================

    write_csv(
        output_dir
        / "route.csv",
        route_rows,
    )

    write_csv(
        output_dir
        / "route_events.csv",
        events,
    )

    write_csv(
        output_dir
        / "traffic_lights.csv",
        light_rows,
    )

    save_topdown(
        output_dir
        / "route_topdown.png",

        route=route,
        cumulative=cumulative,
        events=events,
    )

    summary = {
        "town":
            args.town,

        "spawn_index":
            start_idx,

        "destination_index":
            destination_idx,

        "route_sampling_resolution_m":
            float(
                args.route_sampling_resolution
            ),

        "route_points":
            len(route),

        "route_length_m":
            float(
                cumulative[-1]
            ),

        "relevant_traffic_lights":
            relevant_lights,

        "events":
            events,
    }

    with (
        output_dir
        / "route_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            summary,
            fp,
            indent=2,
        )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 88)
    print(
        "HE BENCHMARK ROUTE INSPECTION"
    )
    print("=" * 88)

    print(
        f"town        : "
        f"{args.town}"
    )

    print(
        f"spawn       : "
        f"{start_idx}"
    )

    print(
        f"destination : "
        f"{destination_idx}"
    )

    print(
        f"route points: "
        f"{len(route)}"
    )

    print(
        f"route length: "
        f"{cumulative[-1]:.2f} m"
    )

    print()

    print(
        "Important route events"
    )

    print("-" * 88)

    for event in events:

        important = (
            event["event_type"]
            in (
                "junction_entry",
                "junction_exit",
                "traffic_light",
            )
            or
            (
                event["event_type"]
                == "road_option_change"
                and
                event["road_option"]
                in (
                    "RIGHT",
                    "LEFT",
                )
            )
        )

        if not important:
            continue

        print(
            f"s={event['route_progress_m']:7.2f} m | "
            f"idx={event['route_idx']:4d} | "
            f"{event['event_type']:18s} | "
            f"{event['road_option']:12s} | "
            f"{event['details']}"
        )

    print()

    print(
        "Traffic lights near route"
    )

    print("-" * 88)

    if not relevant_lights:

        print(
            "No traffic light stop waypoint "
            "matched this route."
        )

    for row in relevant_lights:

        print(
            f"id={row['traffic_light_id']:4d} | "
            f"s={row['route_progress_m']:7.2f} m | "
            f"route_distance="
            f"{row['distance_to_route_m']:.2f} m | "
            f"road={row['stop_road_id']} | "
            f"lane={row['stop_lane_id']} | "
            f"state={row['initial_state']}"
        )

    print()

    print(
        "Saved:"
    )

    for name in (
        "route.csv",
        "route_events.csv",
        "traffic_lights.csv",
        "route_summary.json",
        "route_topdown.png",
    ):

        print(
            " ",
            output_dir / name,
        )

    print("=" * 88)


if __name__ == "__main__":
    main()