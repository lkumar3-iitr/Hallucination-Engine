#!/usr/bin/env python3
"""
test_scenario_json_loader.py

Step 2 of HE temporal pipeline:
  - load scenario JSON
  - validate important fields
  - simulate adversary state over time
  - print frame-wise state samples

This does NOT render sprites yet.
"""

import argparse
import json
from pathlib import Path


# ============================================================
# Scenario loading
# ============================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def require_key(obj, key, context):
    if key not in obj:
        raise KeyError(f"Missing key '{key}' in {context}")
    return obj[key]


def validate_scenario(scenario):
    schema_version = require_key(scenario, "schema_version", "scenario")
    scenario_id = require_key(scenario, "scenario_id", "scenario")

    if schema_version != "he_scenario_v1":
        raise ValueError(f"Unsupported schema_version: {schema_version}")

    require_key(scenario, "input", "scenario")
    require_key(scenario, "output", "scenario")
    require_key(scenario, "camera", "scenario")
    require_key(scenario, "timeline", "scenario")
    require_key(scenario, "sprite_bank", "scenario")
    require_key(scenario, "adversaries", "scenario")

    adversaries = scenario["adversaries"]

    if not isinstance(adversaries, list) or len(adversaries) == 0:
        raise ValueError("scenario.adversaries must be a non-empty list")

    for idx, adv in enumerate(adversaries):
        context = f"adversaries[{idx}]"

        require_key(adv, "id", context)
        require_key(adv, "type", context)
        require_key(adv, "enabled", context)
        require_key(adv, "size", context)
        require_key(adv, "initial_state", context)
        require_key(adv, "motion", context)
        require_key(adv, "rendering", context)

        init = adv["initial_state"]
        require_key(init, "x_m", context + ".initial_state")
        require_key(init, "y_m", context + ".initial_state")
        require_key(init, "z_m", context + ".initial_state")
        require_key(init, "yaw_deg", context + ".initial_state")

        motion = adv["motion"]
        require_key(motion, "model", context + ".motion")

        if motion["model"] != "constant_velocity":
            raise ValueError(f"Only constant_velocity is supported now. Got: {motion['model']}")

        velocity = require_key(motion, "velocity", context + ".motion")
        require_key(velocity, "x_mps", context + ".motion.velocity")
        require_key(velocity, "y_mps", context + ".motion.velocity")
        require_key(velocity, "z_mps", context + ".motion.velocity")

    return True


# ============================================================
# Adversary state engine v1
# ============================================================

def get_frame_time(frame_idx, start_frame, fps):
    return (frame_idx - start_frame) / float(fps)


def update_constant_velocity_state(adversary, t_sec):
    init = adversary["initial_state"]
    vel = adversary["motion"]["velocity"]

    state = {
        "id": adversary["id"],
        "type": adversary["type"],

        "x_m": float(init["x_m"]) + float(vel["x_mps"]) * t_sec,
        "y_m": float(init["y_m"]) + float(vel["y_mps"]) * t_sec,
        "z_m": float(init["z_m"]) + float(vel["z_mps"]) * t_sec,

        "yaw_deg": float(init["yaw_deg"])
    }

    return state


def simulate_scenario_states(scenario, sample_every=30):
    timeline = scenario["timeline"]

    start_frame = int(timeline["start_frame"])
    end_frame = int(timeline["end_frame"])
    fps = float(timeline["fps"])

    adversaries = scenario["adversaries"]

    rows = []

    for frame_idx in range(start_frame, end_frame + 1):
        if (frame_idx - start_frame) % sample_every != 0 and frame_idx != end_frame:
            continue

        t_sec = get_frame_time(frame_idx, start_frame, fps)

        frame_states = []

        for adv in adversaries:
            if not adv.get("enabled", True):
                continue

            if adv["motion"]["model"] == "constant_velocity":
                state = update_constant_velocity_state(adv, t_sec)
            else:
                raise ValueError(f"Unsupported motion model: {adv['motion']['model']}")

            frame_states.append(state)

        rows.append({
            "frame_idx": frame_idx,
            "t_sec": t_sec,
            "states": frame_states
        })

    return rows


# ============================================================
# Pretty printing
# ============================================================

def print_scenario_summary(scenario):
    print("\n========== HE Scenario Summary ==========")
    print("schema_version :", scenario["schema_version"])
    print("scenario_id    :", scenario["scenario_id"])
    print("description    :", scenario.get("description", ""))

    print("\n[input]")
    print("video_path     :", scenario["input"]["video_path"])

    print("\n[output]")
    print("output_dir     :", scenario["output"]["output_dir"])
    print("output_video   :", scenario["output"]["output_video_name"])

    cam = scenario["camera"]
    intr = cam["intrinsics"]

    print("\n[camera]")
    print("model          :", cam["model"])
    print("image size     :", cam["image_width"], "x", cam["image_height"])
    print("fx fy          :", intr["fx"], intr["fy"])
    print("cx cy          :", intr["cx"], intr["cy"])

    timeline = scenario["timeline"]

    print("\n[timeline]")
    print("start_frame    :", timeline["start_frame"])
    print("end_frame      :", timeline["end_frame"])
    print("fps            :", timeline["fps"])

    sprite = scenario["sprite_bank"]

    print("\n[sprite_bank]")
    print("root           :", sprite["root"])
    print("rgba_dir       :", sprite["rgba_dir"])
    print("angle_format   :", sprite["angle_format"])

    print("\n[adversaries]")
    for adv in scenario["adversaries"]:
        print("id             :", adv["id"])
        print("type           :", adv["type"])
        print("enabled        :", adv["enabled"])
        print("initial_state  :", adv["initial_state"])
        print("motion         :", adv["motion"])
        print("rendering      :", adv["rendering"])


def print_state_samples(rows):
    print("\n========== State Samples ==========")

    for row in rows:
        print(f"\nframe={row['frame_idx']}  t={row['t_sec']:.3f}s")

        for state in row["states"]:
            print(
                f"  {state['id']}: "
                f"x={state['x_m']:.2f} m, "
                f"y={state['y_m']:.2f} m, "
                f"z={state['z_m']:.2f} m, "
                f"yaw={state['yaw_deg']:.1f} deg"
            )


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        default="configs/scenarios/oncoming_vehicle_001.json",
        help="Path to HE scenario JSON."
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=30,
        help="Print state every N frames."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    scenario = load_json(args.scenario)

    validate_scenario(scenario)

    print_scenario_summary(scenario)

    rows = simulate_scenario_states(
        scenario=scenario,
        sample_every=args.sample_every
    )

    print_state_samples(rows)

    print("\n[OK] Scenario JSON loaded and adversary state engine v1 works.")


if __name__ == "__main__":
    main()