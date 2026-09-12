"""Build a full-length closed-loop comparison; do not claim pose alignment."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def optional_float(value):
    return float(value) if value not in (None, "") else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-dir", type=Path, required=True)
    parser.add_argument("--he-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=801)
    args = parser.parse_args()
    if Path(__file__).resolve().parent not in args.output.resolve().parents:
        parser.error("Output must remain inside astra")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"frames").mkdir()
    captures, rows = [], []
    for directory, condition in ((args.carla_dir, "carla"), (args.he_dir, "he")):
        files = list(directory.glob(f"*_{condition}_all_cameras.mp4"))
        csv_files = list(directory.glob(f"*_{condition}.csv"))
        if len(files) != 1 or len(csv_files) != 1:
            raise ValueError(f"Ambiguous/missing evidence in {directory}")
        captures.append(cv2.VideoCapture(str(files[0])))
        with csv_files[0].open(newline="") as handle:
            rows.append(list(csv.DictReader(handle)))
        if len(rows[-1]) != args.expected_frames:
            raise ValueError(f"Incomplete CSV in {directory}: {len(rows[-1])}")
    width, height = (int(captures[0].get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.output/"closed_loop_comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            20, (width, height*2+80))
    if not writer.isOpened():
        raise IOError("Could not open video writer")
    try:
        for i in range(args.expected_frames):
            frames = [capture.read() for capture in captures]
            if not all(ok for ok, _ in frames):
                raise IOError(f"Video incomplete at frame {i}")
            if any(frame.shape[:2] != (height, width) for _, frame in frames):
                raise ValueError("Camera layouts differ")
            panel = np.zeros((height*2+80, width, 3), np.uint8)
            panel[40:40+height] = frames[0][1]
            panel[height+80:] = frames[1][1]
            for j, title in enumerate(("CARLA physical", "ASTRA calibrated-hull candidate")):
                row = rows[j][i]
                y = 27 if j == 0 else height+67
                progress = optional_float(row["ego_route_progress_m"])
                progress_text = "unlogged" if progress is None else f"{progress:.2f}m"
                cv2.putText(panel, f"{title}  t={i/20:.2f}s  progress={progress_text}  independent closed loop",
                            (8, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
            writer.write(panel)
            if i % 10 == 0 and 120 <= i <= 340:
                cv2.imwrite(str(args.output/"frames"/f"frame_{i:06d}.jpg"), panel)
    finally:
        writer.release()
        for capture in captures:
            capture.release()
    summary = {"frames": args.expected_frames, "fps": 20,
               "comparison": "independent closed loops at equal elapsed time; not pixel/pose aligned"}
    for condition, condition_rows in zip(("carla", "he"), rows):
        clearances = [optional_float(r["nearest_actor_clearance_m"]) for r in condition_rows]
        summary[condition] = {"final_progress_m": optional_float(condition_rows[-1]["ego_route_progress_m"]),
                              "final_speed_mps": float(condition_rows[-1]["ego_speed_mps"]),
                              "physical_overlap_frames": sum(int(r["physical_overlap"]) for r in condition_rows),
                              "minimum_clearance_m": min(v for v in clearances if v is not None),
                              "trigger_frame": condition_rows[-1]["trigger_frame"]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
