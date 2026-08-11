#!/usr/bin/env python3
"""
test_camera_projection.py

Step 4 of HE temporal pipeline:
  - load scenario JSON
  - update adversary state frame by frame
  - project 3D adversary state into image coordinates
  - print projected boxes

This does NOT render sprites yet.
"""

import argparse
import json
from pathlib import Path


# ============================================================
# Basic scenario utilities
# ============================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_frame_time(frame_idx, start_frame, fps):
    return (frame_idx - start_frame) / float(fps)


def update_constant_velocity_state(adversary, t_sec):
    init = adversary["initial_state"]
    vel = adversary["motion"]["velocity"]

    return {
        "id": adversary["id"],
        "type": adversary["type"],

        "x_m": float(init["x_m"]) + float(vel["x_mps"]) * t_sec,
        "y_m": float(init["y_m"]) + float(vel["y_mps"]) * t_sec,
        "z_m": float(init["z_m"]) + float(vel["z_mps"]) * t_sec,

        "yaw_deg": float(init["yaw_deg"])
    }


# ============================================================
# Camera projection
# ============================================================

def project_point_pinhole(x_m, y_m, z_m, intrinsics):
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    if z_m <= 0.1:
        return None

    u = fx * (x_m / z_m) + cx
    v = fy * (y_m / z_m) + cy

    return u, v


def project_vehicle_box(state, adversary, camera):
    """
    Project one adversary state to an approximate 2D box.

    Coordinate convention:
      x_m: horizontal offset, right positive
      y_m: vertical image/camera offset, down positive
      z_m: forward distance

    For first version:
      - horizontal center comes from x_m/z_m
      - bottom y uses camera.placement.ground_y_m
      - width/height use physical vehicle size / z_m
    """

    image_width = int(camera["image_width"])
    image_height = int(camera["image_height"])
    intr = camera["intrinsics"]

    fx = float(intr["fx"])
    fy = float(intr["fy"])

    x = float(state["x_m"])
    z = float(state["z_m"])

    if z <= 0.1:
        return {
            "visible": False,
            "reason": "behind_camera_or_too_close"
        }

    size = adversary["size"]
    vehicle_width_m = float(size["width_m"])
    vehicle_height_m = float(size["height_m"])

    placement = camera.get("placement", {})
    ground_y_m = float(placement.get("ground_y_m", 1.35))

    # Horizontal center.
    center = project_point_pinhole(
        x_m=x,
        y_m=ground_y_m,
        z_m=z,
        intrinsics=intr
    )

    if center is None:
        return {
            "visible": False,
            "reason": "projection_failed"
        }

    u_center, v_bottom = center

    box_w = fx * vehicle_width_m / z
    box_h = fy * vehicle_height_m / z

    x1 = u_center - box_w / 2.0
    x2 = u_center + box_w / 2.0
    y2 = v_bottom
    y1 = v_bottom - box_h

    # Visibility test with loose margin.
    margin = 100

    visible = not (
        x2 < -margin or
        x1 > image_width + margin or
        y2 < -margin or
        y1 > image_height + margin
    )

    return {
        "visible": bool(visible),

        "cx": float(u_center),
        "bottom_y": float(v_bottom),

        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),

        "box_width": float(box_w),
        "box_height": float(box_h),

        "z_m": float(z)
    }


# ============================================================
# Simulation
# ============================================================

def simulate_projection_samples(scenario, sample_every=30):
    timeline = scenario["timeline"]

    start_frame = int(timeline["start_frame"])
    end_frame = int(timeline["end_frame"])
    fps = float(timeline["fps"])

    camera = scenario["camera"]
    adversaries = scenario["adversaries"]

    rows = []

    for frame_idx in range(start_frame, end_frame + 1):
        if (frame_idx - start_frame) % sample_every != 0 and frame_idx != end_frame:
            continue

        t_sec = get_frame_time(frame_idx, start_frame, fps)

        frame_items = []

        for adv in adversaries:
            if not adv.get("enabled", True):
                continue

            if adv["motion"]["model"] != "constant_velocity":
                raise ValueError(f"Unsupported motion model: {adv['motion']['model']}")

            state = update_constant_velocity_state(adv, t_sec)
            box = project_vehicle_box(state, adv, camera)

            frame_items.append({
                "state": state,
                "box": box
            })

        rows.append({
            "frame_idx": frame_idx,
            "t_sec": t_sec,
            "items": frame_items
        })

    return rows


def print_projection_samples(rows):
    print("\n========== Projection Samples ==========")

    for row in rows:
        print(f"\nframe={row['frame_idx']}  t={row['t_sec']:.3f}s")

        for item in row["items"]:
            state = item["state"]
            box = item["box"]

            print(
                f"  {state['id']}: "
                f"x={state['x_m']:.2f}m, "
                f"z={state['z_m']:.2f}m, "
                f"yaw={state['yaw_deg']:.1f}deg"
            )

            if not box["visible"]:
                print(f"    not visible: {box.get('reason', 'out_of_frame')}")
            else:
                print(
                    f"    box: "
                    f"x1={box['x1']:.1f}, "
                    f"y1={box['y1']:.1f}, "
                    f"x2={box['x2']:.1f}, "
                    f"y2={box['y2']:.1f}, "
                    f"w={box['box_width']:.1f}, "
                    f"h={box['box_height']:.1f}, "
                    f"bottom_y={box['bottom_y']:.1f}"
                )


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        default="configs/scenarios/oncoming_vehicle_001.json"
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=30
    )

    return parser.parse_args()


def main():
    args = parse_args()

    scenario = load_json(args.scenario)

    rows = simulate_projection_samples(
        scenario=scenario,
        sample_every=args.sample_every
    )

    print_projection_samples(rows)

    print("\n[OK] Camera projection v1 works.")


if __name__ == "__main__":
    main()