"""Benchmark ASTRA native-bank selection on poses from a recorded replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import carla

try:
    from .passby_renderer import aimed_camera
    from .selector import Selector, transform
except ImportError:
    from passby_renderer import aimed_camera
    from selector import Selector, transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).parent / "artifacts/cache")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.frames.read_text().splitlines()]
    poses = [row for row in rows if row.get("he_sprite_mode") == "view_matrix" and row.get("he_rendered")]
    selector = Selector(args.bank, args.artifacts)
    selector.build_hull()

    selected = []
    started = time.perf_counter()
    for _ in range(args.repeat):
        selected.clear()
        for row in poses:
            actor = transform(row["actor_transform"])
            camera = transform(row["camera_transform"])
            virtual_camera, _ = aimed_camera(
                actor,
                camera,
                selector.center,
                selector.extents,
            )
            result = selector.select(
                actor,
                virtual_camera,
                args.width,
                args.height,
                args.fov,
            )
            selected.append(result["key"])
    elapsed = time.perf_counter() - started
    calls = len(poses) * args.repeat
    expected = [[
        float(row["he_selected_angle_deg"]),
        float(row["he_selected_distance_m"]),
        float(row["he_selected_elevation_deg"]),
    ] for row in poses]
    mismatches = [
        {"pose": index, "expected": expected_key, "actual": actual_key}
        for index, (expected_key, actual_key) in enumerate(zip(expected, selected))
        if actual_key != expected_key
    ]
    print(json.dumps({
        "poses": len(poses),
        "repeat": args.repeat,
        "calls": calls,
        "elapsed_s": elapsed,
        "ms_per_select": 1000.0 * elapsed / max(calls, 1),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "selected_keys": selected,
    }))


if __name__ == "__main__":
    main()
