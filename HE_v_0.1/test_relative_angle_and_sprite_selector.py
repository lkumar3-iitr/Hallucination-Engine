#!/usr/bin/env python3
"""
test_relative_angle_and_sprite_selector.py

Step 5 and Step 6 of HE temporal pipeline:
  - load scenario JSON
  - update adversary state
  - compute relative sprite angle
  - select nearest RGBA sprite
  - verify sprite file exists

This does NOT render/composite sprites yet.
"""

import argparse
import json
import math
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
# Angle utilities
# ============================================================

def normalize_angle_360(angle_deg):
    """
    Convert any angle to [0, 360).
    """
    return float(angle_deg) % 360.0


def normalize_angle_180(angle_deg):
    """
    Convert angle to [-180, 180).
    """
    a = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return a


def round_to_nearest_int_angle(angle_deg):
    """
    Round angle to nearest integer degree in [0, 359].
    """
    return int(round(normalize_angle_360(angle_deg))) % 360


def nearest_available_angle(angle_deg, available_angles):
    """
    Select nearest available sprite angle using circular distance.
    """
    if not available_angles:
        raise ValueError("available_angles is empty")

    target = normalize_angle_360(angle_deg)

    best_angle = available_angles[0]
    best_dist = 999999.0

    for a in available_angles:
        d = abs(normalize_angle_180(target - a))
        if d < best_dist:
            best_dist = d
            best_angle = a

    return int(best_angle), float(best_dist)


# ============================================================
# Relative angle computation
# ============================================================

def compute_relative_sprite_angle(state, camera_yaw_deg=0.0):
    """
    Compute which sprite angle should be shown.

    For v1, our HE camera coordinate system assumes:
      - ego camera looks along +z
      - camera yaw = 0
      - adversary yaw_deg describes vehicle facing direction relative to camera

    Sprite convention:
      0   = rear view of adversary
      180 = front view of adversary

    Therefore, for the current v1 convention:
      sprite_angle = adversary_yaw_deg - camera_yaw_deg

    Examples:
      yaw=180 -> front-facing/oncoming sprite
      yaw=0   -> rear-facing/receding sprite
      yaw=90  -> side sprite
      yaw=270 -> opposite side sprite
    """

    yaw = float(state["yaw_deg"])
    rel = yaw - float(camera_yaw_deg)
    return normalize_angle_360(rel)


# ============================================================
# Sprite bank selector
# ============================================================

def discover_available_sprite_angles(sprite_bank):
    root = Path(sprite_bank["root"])
    rgba_dir = sprite_bank.get("rgba_dir", "rgba")
    angle_format = sprite_bank.get("angle_format", "angle_{angle:03d}_rgba.png")

    sprite_dir = root / rgba_dir

    if not sprite_dir.exists():
        raise FileNotFoundError(f"Sprite RGBA directory not found: {sprite_dir}")

    available = []

    # Usually we have angle_000_rgba.png ... angle_359_rgba.png.
    # But this also supports 30-degree test banks.
    for angle in range(360):
        fname = angle_format.format(angle=angle)
        path = sprite_dir / fname

        if path.exists():
            available.append(angle)

    if not available:
        raise FileNotFoundError(f"No sprite files found in: {sprite_dir}")

    return available


def select_sprite(sprite_bank, relative_angle_deg, available_angles):
    root = Path(sprite_bank["root"])
    rgba_dir = sprite_bank.get("rgba_dir", "rgba")
    angle_format = sprite_bank.get("angle_format", "angle_{angle:03d}_rgba.png")

    selected_angle, angle_error = nearest_available_angle(
        relative_angle_deg,
        available_angles
    )

    sprite_path = root / rgba_dir / angle_format.format(angle=selected_angle)

    return {
        "relative_angle_deg": float(relative_angle_deg),
        "selected_angle": int(selected_angle),
        "angle_error_deg": float(angle_error),
        "sprite_path": str(sprite_path),
        "exists": sprite_path.exists()
    }


# ============================================================
# Simulation
# ============================================================

def simulate_angle_sprite_samples(scenario, sample_every=30):
    timeline = scenario["timeline"]

    start_frame = int(timeline["start_frame"])
    end_frame = int(timeline["end_frame"])
    fps = float(timeline["fps"])

    camera = scenario["camera"]
    sprite_bank = scenario["sprite_bank"]
    adversaries = scenario["adversaries"]

    camera_yaw_deg = float(camera.get("yaw_deg", 0.0))

    available_angles = discover_available_sprite_angles(sprite_bank)

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

            relative_angle = compute_relative_sprite_angle(
                state=state,
                camera_yaw_deg=camera_yaw_deg
            )

            sprite = select_sprite(
                sprite_bank=sprite_bank,
                relative_angle_deg=relative_angle,
                available_angles=available_angles
            )

            frame_items.append({
                "state": state,
                "relative_angle": relative_angle,
                "sprite": sprite
            })

        rows.append({
            "frame_idx": frame_idx,
            "t_sec": t_sec,
            "items": frame_items
        })

    return rows, available_angles


def print_angle_sprite_samples(rows, available_angles):
    print("\n========== Sprite Bank ==========")
    print("available sprite count:", len(available_angles))
    print("first few angles      :", available_angles[:10])
    print("last few angles       :", available_angles[-10:])

    print("\n========== Relative Angle + Sprite Selection Samples ==========")

    for row in rows:
        print(f"\nframe={row['frame_idx']}  t={row['t_sec']:.3f}s")

        for item in row["items"]:
            state = item["state"]
            sprite = item["sprite"]

            print(
                f"  {state['id']}: "
                f"x={state['x_m']:.2f}m, "
                f"z={state['z_m']:.2f}m, "
                f"yaw={state['yaw_deg']:.1f}deg"
            )

            print(
                f"    relative_angle={sprite['relative_angle_deg']:.1f} deg, "
                f"selected_angle={sprite['selected_angle']:03d}, "
                f"error={sprite['angle_error_deg']:.2f} deg, "
                f"exists={sprite['exists']}"
            )

            print(f"    sprite_path={sprite['sprite_path']}")


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

    rows, available_angles = simulate_angle_sprite_samples(
        scenario=scenario,
        sample_every=args.sample_every
    )

    print_angle_sprite_samples(rows, available_angles)

    print("\n[OK] Relative angle computation and sprite selector work.")


if __name__ == "__main__":
    main()