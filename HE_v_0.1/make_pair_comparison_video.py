import os
import cv2
import glob
import argparse
import numpy as np


def read_frame(path, target_size=None):
    """
    Read image and optionally resize.
    """

    frame = cv2.imread(path)

    if frame is None:
        return None

    if target_size is not None:
        frame = cv2.resize(frame, target_size)

    return frame


def put_label(frame, label):
    """
    Add label text at top-left.
    """

    output = frame.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 42),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        label,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


def make_comparison_video(run_dir, output_name="comparison.mp4", fps=20):
    """
    Creates side-by-side comparison video:

        clean | real_object | he_oracle_box | he_camera_only
    """

    clean_dir = os.path.join(run_dir, "clean")
    real_dir = os.path.join(run_dir, "real_object")
    oracle_dir = os.path.join(run_dir, "he_oracle_box")
    camera_only_dir = os.path.join(run_dir, "he_camera_only")
    oracle_refined_dir = os.path.join(run_dir, "he_oracle_refined")

    required_dirs = {
        "clean": clean_dir,
        "real_object": real_dir,
        "he_oracle_box": oracle_dir,
    }

    # he_oracle_refined is optional because it exists only after apply_local_refiner.py
    if os.path.isdir(oracle_refined_dir):
        print("[Video] Found he_oracle_refined folder.")
    else:
        print("[Video] he_oracle_refined not found. It will use black placeholders.")
        oracle_refined_dir = None

    for name, folder in required_dirs.items():
        if not os.path.isdir(folder):
            raise RuntimeError(f"Missing folder '{name}': {folder}")

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

    if len(clean_paths) == 0:
        raise RuntimeError(f"No clean frames found in: {clean_dir}")

    first_frame = cv2.imread(clean_paths[0])

    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {clean_paths[0]}")

    h, w = first_frame.shape[:2]

    # Resize each panel to half size to avoid extremely wide output.
    panel_w = w // 2
    panel_h = h // 2
    panel_size = (panel_w, panel_h)

    output_width = panel_w * 4
    output_height = panel_h

    output_path = os.path.join(run_dir, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, output_height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    frame_count = 0

    for clean_path in clean_paths:
        frame_name = os.path.basename(clean_path)

        real_path = os.path.join(real_dir, frame_name)
        oracle_path = os.path.join(oracle_dir, frame_name)

        oracle_refined_path = (
            os.path.join(oracle_refined_dir, frame_name)
            if oracle_refined_dir is not None
            else None
        )

        clean = read_frame(clean_path, panel_size)
        real = read_frame(real_path, panel_size)
        oracle = read_frame(oracle_path, panel_size)
        oracle_refined = read_frame(oracle_refined_path, panel_size)

        if clean is None:
            print("Skipping missing clean frame:", clean_path)
            continue

        # If any stream is missing a frame, use black placeholder.
        if real is None:
            real = np.zeros_like(clean)

        if oracle is None:
            oracle = np.zeros_like(clean)

        if oracle_refined is None:
            oracle_refined = np.zeros_like(clean)

        clean = put_label(clean, "Clean")
        real = put_label(real, "Real CARLA Object")
        oracle = put_label(oracle, "HE Oracle Box")
        oracle_refined = put_label(oracle_refined, "HE Oracle Refined")

        comparison = np.hstack([
            clean,
            real,
            oracle,
            oracle_refined
        ])

        writer.write(comparison)

        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Written {frame_count} frames...")

    writer.release()

    print("Saved comparison video:", output_path)
    print("Frames written:", frame_count)


def find_latest_pair_run(base_dir="paired_data"):
    """
    Find latest pair_run folder.
    """

    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Base directory not found: {base_dir}")

    runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("pair_run_")
        and os.path.isdir(os.path.join(base_dir, d))
    ]

    if len(runs) == 0:
        raise RuntimeError(f"No pair_run folders found in: {base_dir}")

    runs = sorted(runs, key=os.path.getmtime, reverse=True)

    return runs[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to pair_run folder. If omitted, latest run in paired_data is used."
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Output video FPS."
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="comparison.mp4",
        help="Output video filename."
    )

    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = find_latest_pair_run("paired_data")
    else:
        run_dir = args.run_dir

    print("Using run_dir:", run_dir)

    make_comparison_video(
        run_dir=run_dir,
        output_name=args.output_name,
        fps=args.fps
    )


if __name__ == "__main__":
    main()