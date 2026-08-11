import argparse
import json
import math
from pathlib import Path


def normalize_angle_180(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def load_pose_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

    records.sort(key=lambda r: int(r["recorded_frame_idx"]))
    return records


def yaw_of(rec):
    return float(rec["ego_transform"]["yaw"])


def speed_of(rec):
    return float(rec.get("ego_speed_mps", 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ego-pose-jsonl", required=True)
    parser.add_argument("--window", type=int, default=90, help="Window size in frames.")
    parser.add_argument("--step", type=int, default=15, help="Step size in frames.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-yaw-change", type=float, default=10.0)
    args = parser.parse_args()

    path = Path(args.ego_pose_jsonl)
    records = load_pose_jsonl(path)

    if len(records) < args.window + 1:
        raise RuntimeError("Not enough frames in ego pose file.")

    windows = []

    for i in range(0, len(records) - args.window, args.step):
        a = records[i]
        b = records[i + args.window]

        f0 = int(a["recorded_frame_idx"])
        f1 = int(b["recorded_frame_idx"])

        yaw0 = yaw_of(a)
        yaw1 = yaw_of(b)
        dyaw = normalize_angle_180(yaw1 - yaw0)

        sp0 = speed_of(a)
        sp1 = speed_of(b)

        windows.append({
            "start_frame": f0,
            "end_frame": f1,
            "yaw0": yaw0,
            "yaw1": yaw1,
            "yaw_change_deg": dyaw,
            "abs_yaw_change_deg": abs(dyaw),
            "speed0_mps": sp0,
            "speed1_mps": sp1,
        })

    windows_sorted = sorted(
        windows,
        key=lambda w: w["abs_yaw_change_deg"],
        reverse=True,
    )

    print("\nTop turning windows:")
    print("-" * 90)

    for w in windows_sorted[:args.top_k]:
        direction = "right" if w["yaw_change_deg"] > 0 else "left"

        print(
            f"frames {w['start_frame']:4d} -> {w['end_frame']:4d} | "
            f"yaw_change={w['yaw_change_deg']:8.2f} deg ({direction}) | "
            f"speed={w['speed0_mps']:.2f}->{w['speed1_mps']:.2f} m/s"
        )

    print("\nCandidate start frames where yaw change is significant:")
    print("-" * 90)

    candidates = [
        w for w in windows_sorted
        if w["abs_yaw_change_deg"] >= args.min_yaw_change
    ]

    for w in candidates[:args.top_k]:
        print(
            f"--start-frame {w['start_frame']}  "
            f"# yaw_change={w['yaw_change_deg']:.2f} deg over {args.window} frames"
        )


if __name__ == "__main__":
    main()