"""
discover_route_spawn_seeds_v1.py

Batch-inspect CARLA spawn/destination route pairs for paper scenario design.

This is the multi-route companion to inspect_carla_test_route_v1.py. It does
not spawn ego or run a driving model; it only loads the CARLA map, traces route
topology, detects route-relevant traffic lights/events, and writes a ranked
overview so we can choose more ego spawn seeds before freezing critical
scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import carla

THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[2]
COMMON_DIR = REPO_ROOT / "driving_models" / "common"

for path in (SCRIPT_DIR, COMMON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inspect_carla_test_route_v1 import (
    build_route_events,
    command_name,
    distance_2d,
    route_cumulative_distance,
    save_topdown,
    traffic_light_rows,
)


def import_global_route_planner_with_candidates(pythonapi_hint=None):
    def _load():
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        return GlobalRoutePlanner

    try:
        return _load()
    except ModuleNotFoundError:
        pass

    candidates = []
    if pythonapi_hint:
        candidates.append(Path(pythonapi_hint))

    carla_root = os.environ.get("CARLA_ROOT")
    if carla_root:
        root = Path(carla_root)
        candidates.extend([
            root / "PythonAPI" / "carla",
            root / "PythonAPI",
            root,
        ])

    for base in (Path.cwd(), REPO_ROOT, REPO_ROOT.parent):
        candidates.extend([
            base / "PythonAPI" / "carla",
            base / "PythonAPI",
            base / "carla",
        ])

    candidates.extend([
        Path("E:/Carla/Carla_0.9.15/PythonAPI/carla"),
        Path("D:/CARLA_0.9.15/PythonAPI/carla"),
        Path("C:/CARLA_0.9.15/PythonAPI/carla"),
    ])

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        possible = [candidate, candidate / "carla"]
        for path in possible:
            if not (path / "agents").is_dir():
                continue
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            try:
                return _load()
            except ModuleNotFoundError:
                pass

    raise RuntimeError(
        "Could not import CARLA agents. Pass --carla-pythonapi pointing to "
        "the CARLA PythonAPI/carla directory, or set CARLA_ROOT."
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_indices(text: str | None, total: int) -> list[int]:
    if text is None or str(text).strip() == "":
        return list(range(total))
    out = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = item.split("-", 1)
            out.extend(range(int(first), int(last) + 1))
        else:
            out.append(int(item))
    return sorted({idx % total for idx in out})


def route_length_from_cumulative(route, spawn_tf) -> tuple[list[float], float]:
    first_route_tf = route[0][0].transform
    spawn_to_first_route_m = distance_2d(
        spawn_tf.location,
        first_route_tf.location,
    )
    trace_cumulative = route_cumulative_distance(route)
    cumulative = [
        float(spawn_to_first_route_m) + float(s_m)
        for s_m in trace_cumulative
    ]
    return cumulative, spawn_to_first_route_m


def event_counts(events: list[dict]) -> dict[str, int]:
    counts = {
        "traffic_lights": 0,
        "junction_entries": 0,
        "junction_exits": 0,
        "left_turns": 0,
        "right_turns": 0,
    }
    for event in events:
        event_type = event.get("event_type")
        road_option = event.get("road_option")
        if event_type == "traffic_light":
            counts["traffic_lights"] += 1
        elif event_type == "junction_entry":
            counts["junction_entries"] += 1
        elif event_type == "junction_exit":
            counts["junction_exits"] += 1
        elif event_type == "road_option_change" and road_option == "LEFT":
            counts["left_turns"] += 1
        elif event_type == "road_option_change" and road_option == "RIGHT":
            counts["right_turns"] += 1
    return counts


def score_route(length_m: float, counts: dict[str, int]) -> float:
    score = 0.0
    if 90.0 <= length_m <= 260.0:
        score += 2.0
    if counts["traffic_lights"]:
        score += 5.0 + min(counts["traffic_lights"], 3)
    if counts["junction_entries"]:
        score += 1.5
    if counts["left_turns"] or counts["right_turns"]:
        score += 1.5
    return score


def route_rows(route, cumulative, spawn_tf):
    rows = [
        {
            "route_idx": -1,
            "route_progress_m": 0.0,
            "x": float(spawn_tf.location.x),
            "y": float(spawn_tf.location.y),
            "z": float(spawn_tf.location.z),
            "yaw_deg": float(spawn_tf.rotation.yaw),
            "road_option": command_name(route[0][1]),
            "road_id": int(route[0][0].road_id),
            "section_id": int(route[0][0].section_id),
            "lane_id": int(route[0][0].lane_id),
            "waypoint_s": float(route[0][0].s),
            "lane_width_m": float(route[0][0].lane_width),
            "is_junction": bool(route[0][0].is_junction),
            "junction_id": None,
        }
    ]
    for idx, (waypoint, road_option) in enumerate(route):
        tf = waypoint.transform
        rows.append(
            {
                "route_idx": idx,
                "route_progress_m": cumulative[idx],
                "x": float(tf.location.x),
                "y": float(tf.location.y),
                "z": float(tf.location.z),
                "yaw_deg": float(tf.rotation.yaw),
                "road_option": command_name(road_option),
                "road_id": int(waypoint.road_id),
                "section_id": int(waypoint.section_id),
                "lane_id": int(waypoint.lane_id),
                "waypoint_s": float(waypoint.s),
                "lane_width_m": float(waypoint.lane_width),
                "is_junction": bool(waypoint.is_junction),
                "junction_id": None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--spawn-indices", default=None)
    parser.add_argument("--destination-indices", default=None)
    parser.add_argument("--route-sampling-resolution", type=float, default=2.0)
    parser.add_argument("--min-route-length-m", type=float, default=90.0)
    parser.add_argument("--max-route-length-m", type=float, default=280.0)
    parser.add_argument("--max-pairs-per-spawn", type=int, default=10)
    parser.add_argument("--traffic-light-route-distance-m", type=float, default=8.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--export-top-k", type=int, default=8)
    parser.add_argument("--carla-pythonapi", default=None)
    parser.add_argument(
        "--output-dir",
        default="ScenarioGenerator/outputs/route_spawn_seed_discovery_v1",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    GlobalRoutePlanner = import_global_route_planner_with_candidates(
        args.carla_pythonapi
    )
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    print("[CARLA] loading:", args.town)
    world = client.load_world(args.town)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available.")

    spawns = parse_indices(args.spawn_indices, len(spawn_points))
    destinations = parse_indices(args.destination_indices, len(spawn_points))
    grp = GlobalRoutePlanner(carla_map, float(args.route_sampling_resolution))

    candidates = []
    failures = []
    for start_idx in spawns:
        start_tf = spawn_points[start_idx]
        start_ranked = sorted(
            (
                (
                    abs(
                        distance_2d(start_tf.location, dest_tf.location)
                        - 160.0
                    ),
                    dest_idx,
                )
                for dest_idx, dest_tf in enumerate(spawn_points)
                if dest_idx in destinations and dest_idx != start_idx
            )
        )
        for _, dest_idx in start_ranked[: max(1, int(args.max_pairs_per_spawn))]:
            try:
                route = grp.trace_route(
                    start_tf.location,
                    spawn_points[dest_idx].location,
                )
                if len(route) < 2:
                    raise RuntimeError("GlobalRoutePlanner returned no route.")
                cumulative, spawn_to_first_route_m = route_length_from_cumulative(
                    route,
                    start_tf,
                )
                length_m = float(cumulative[-1])
                if (
                    length_m < float(args.min_route_length_m)
                    or length_m > float(args.max_route_length_m)
                ):
                    continue
                lights = traffic_light_rows(
                    world=world,
                    route=route,
                    cumulative=cumulative,
                    max_route_distance_m=args.traffic_light_route_distance_m,
                )
                events = build_route_events(route, cumulative, lights)
                counts = event_counts(events)
                relevant_lights = [
                    row for row in lights if row["relevant_to_route"]
                ]
                candidates.append(
                    {
                        "score": score_route(length_m, counts),
                        "town": args.town,
                        "spawn_index": start_idx,
                        "destination_index": dest_idx,
                        "route_length_m": length_m,
                        "route_points": len(route) + 1,
                        "spawn_to_first_route_m": spawn_to_first_route_m,
                        **counts,
                        "first_traffic_light_s_m": (
                            relevant_lights[0]["route_progress_m"]
                            if relevant_lights
                            else ""
                        ),
                        "first_turn": next(
                            (
                                event["road_option"]
                                for event in events
                                if event.get("event_type")
                                == "road_option_change"
                                and event.get("road_option") in ("LEFT", "RIGHT")
                            ),
                            "",
                        ),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "spawn_index": start_idx,
                        "destination_index": dest_idx,
                        "error": str(exc),
                    }
                )

    candidates.sort(
        key=lambda row: (
            -float(row["score"]),
            -int(row["traffic_lights"]),
            abs(float(row["route_length_m"]) - 160.0),
            int(row["spawn_index"]),
            int(row["destination_index"]),
        )
    )

    write_csv(output_dir / "spawn_seed_candidates.csv", candidates)
    write_csv(output_dir / "failures.csv", failures)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "town": args.town,
                "total_spawn_points": len(spawn_points),
                "scanned_spawns": spawns,
                "candidate_count": len(candidates),
                "failure_count": len(failures),
                "top_k": candidates[: int(args.top_k)],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for row in candidates[: max(0, int(args.export_top_k))]:
        stem = (
            f"town10_spawn{int(row['spawn_index']):03d}"
            f"_to{int(row['destination_index']):03d}_route_v1"
        )
        route_dir = output_dir / "routes" / stem
        route = grp.trace_route(
            spawn_points[int(row["spawn_index"])].location,
            spawn_points[int(row["destination_index"])].location,
        )
        cumulative, _ = route_length_from_cumulative(
            route,
            spawn_points[int(row["spawn_index"])],
        )
        lights = traffic_light_rows(
            world=world,
            route=route,
            cumulative=cumulative,
            max_route_distance_m=args.traffic_light_route_distance_m,
        )
        events = build_route_events(route, cumulative, lights)
        write_csv(route_dir / "route.csv", route_rows(route, cumulative, spawn_points[int(row["spawn_index"])]))
        write_csv(route_dir / "route_events.csv", events)
        write_csv(route_dir / "traffic_lights.csv", lights)
        save_topdown(route_dir / "route_topdown.png", route, cumulative, events)
        (route_dir / "route_summary.json").write_text(
            json.dumps({**row, "events": events}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print()
    print("=" * 88)
    print("ROUTE SPAWN SEED DISCOVERY")
    print("=" * 88)
    print("town              :", args.town)
    print("spawn points      :", len(spawn_points))
    print("candidate routes  :", len(candidates))
    print("failures          :", len(failures))
    print("overview          :", output_dir / "spawn_seed_candidates.csv")
    print("top route exports :", output_dir / "routes")
    print()
    for row in candidates[: int(args.top_k)]:
        print(
            f"score={row['score']:4.1f} | "
            f"spawn={row['spawn_index']:3d} -> dest={row['destination_index']:3d} | "
            f"len={row['route_length_m']:6.1f}m | "
            f"TL={row['traffic_lights']} | "
            f"J={row['junction_entries']} | "
            f"L/R={row['left_turns']}/{row['right_turns']} | "
            f"first_TL={row['first_traffic_light_s_m']}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
