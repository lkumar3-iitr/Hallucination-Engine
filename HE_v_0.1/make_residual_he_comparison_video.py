import os
import cv2
import glob
import argparse
import numpy as np


# ============================================================
# Settings
# ============================================================

PAIRED_BASE_DIR = "paired_data"

DEFAULT_OUTPUT_NAME = "comparison_residual_learned_he_2x2.mp4"
DEFAULT_FPS = 20

# For 1280x720 input:
# PANEL_SCALE = 0.5 gives each panel 640x360
# Final video becomes 1280x720
PANEL_SCALE = 0.5


# ============================================================
# Utility
# ============================================================

def find_latest_pair_run(base_dir=PAIRED_BASE_DIR):
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Paired-data base folder not found: {base_dir}")

    runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("pair_run_")
        and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not runs:
        raise RuntimeError(f"No pair_run folders found in: {base_dir}")

    runs = sorted(runs, key=os.path.getmtime, reverse=True)
    return runs[0]


def read_frame(path, target_size=None):
    if path is None:
        return None

    frame = cv2.imread(path)

    if frame is None:
        return None

    if target_size is not None:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    return frame


def put_label(frame, label):
    output = frame.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 42),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        label,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def make_placeholder_like(reference_frame, label):
    placeholder = np.zeros_like(reference_frame)
    placeholder = put_label(placeholder, label)
    return placeholder


# ============================================================
# Main video builder
# ============================================================

def make_residual_comparison_video(
    run_dir,
    output_name=DEFAULT_OUTPUT_NAME,
    fps=DEFAULT_FPS,
):
    """
    Creates 2x2 final learned-HE comparison video:

        Clean                  | Real CARLA Object
        HE Box Predicted       | HE Residual Refined
    """

    clean_dir = os.path.join(run_dir, "clean")
    real_dir = os.path.join(run_dir, "real_object")
    box_pred_dir = os.path.join(run_dir, "he_box_predicted")
    residual_refined_dir = os.path.join(run_dir, "he_box_predicted_residual_refined")

    required_dirs = {
        "clean": clean_dir,
        "real_object": real_dir,
        "he_box_predicted": box_pred_dir,
        "he_box_predicted_residual_refined": residual_refined_dir,
    }

    for name, folder in required_dirs.items():
        if not os.path.isdir(folder):
            raise RuntimeError(
                f"Missing folder '{name}': {folder}\n"
                f"Make sure you already ran apply_box_predictor.py and "
                f"apply_residual_refiner_box_predicted.py."
            )

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

    if not clean_paths:
        raise RuntimeError(f"No clean frames found in: {clean_dir}")

    first_frame = cv2.imread(clean_paths[0])

    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {clean_paths[0]}")

    h, w = first_frame.shape[:2]

    panel_w = int(w * PANEL_SCALE)
    panel_h = int(h * PANEL_SCALE)

    panel_size = (panel_w, panel_h)

    # 2 panels wide, 2 panels high
    output_width = panel_w * 2
    output_height = panel_h * 2

    output_path = os.path.join(run_dir, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, output_height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    frame_count = 0
    skipped = 0

    print("[ResidualVideo2x2] Run dir:", run_dir)
    print("[ResidualVideo2x2] Output:", output_path)
    print("[ResidualVideo2x2] Panel size:", panel_size)
    print("[ResidualVideo2x2] Video size:", (output_width, output_height))
    print("[ResidualVideo2x2] Total clean frames:", len(clean_paths))

    for clean_path in clean_paths:
        frame_name = os.path.basename(clean_path)

        real_path = os.path.join(real_dir, frame_name)
        box_pred_path = os.path.join(box_pred_dir, frame_name)
        residual_refined_path = os.path.join(residual_refined_dir, frame_name)

        clean = read_frame(clean_path, panel_size)

        if clean is None:
            skipped += 1
            continue

        real = read_frame(real_path, panel_size)
        box_pred = read_frame(box_pred_path, panel_size)
        residual_refined = read_frame(residual_refined_path, panel_size)

        if real is None:
            real = make_placeholder_like(clean, "Missing Real")

        if box_pred is None:
            box_pred = make_placeholder_like(clean, "Missing Box Pred")

        if residual_refined is None:
            residual_refined = make_placeholder_like(clean, "Missing Residual Refined")

        clean = put_label(clean, "Clean")
        real = put_label(real, "Real CARLA Object")
        box_pred = put_label(box_pred, "HE Box Predicted")
        residual_refined = put_label(residual_refined, "HE Residual Refined")

        top_row = np.hstack([
            clean,
            real,
        ])

        bottom_row = np.hstack([
            box_pred,
            residual_refined,
        ])

        comparison = np.vstack([
            top_row,
            bottom_row,
        ])

        writer.write(comparison)

        frame_count += 1

        if frame_count % 20 == 0:
            print(f"[ResidualVideo2x2] Written {frame_count} frames...")

    writer.release()

    print("[ResidualVideo2x2] Done.")
    print("[ResidualVideo2x2] Saved:", output_path)
    print("[ResidualVideo2x2] Frames written:", frame_count)
    print("[ResidualVideo2x2] Skipped:", skipped)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to pair_run folder. If omitted, latest pair_run is used.",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Output MP4 filename.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Output video FPS.",
    )

    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = find_latest_pair_run(PAIRED_BASE_DIR)
    else:
        run_dir = args.run_dir

    make_residual_comparison_video(
        run_dir=run_dir,
        output_name=args.output_name,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()