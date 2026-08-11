import os
import csv
import cv2
import json
import glob
import shutil
import argparse
from datetime import datetime


# ============================================================
# Settings
# ============================================================

PAIRED_BASE_DIR = "paired_data"
OUTPUT_BASE_DIR = "box_dataset"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

MIN_BOX_WIDTH = 8
MIN_BOX_HEIGHT = 8

MAX_BOX_WIDTH = 1200
MAX_BOX_HEIGHT = 700

# If True, copy clean images into box_dataset/images.
# If False, labels.csv will reference original clean frame paths.
COPY_IMAGES = True


# ============================================================
# Utilities
# ============================================================

def find_pair_runs(base_dir=PAIRED_BASE_DIR, limit=None):
    """
    Find pair_run folders inside paired_data.
    """

    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Paired-data base directory not found: {base_dir}")

    runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("pair_run_")
        and os.path.isdir(os.path.join(base_dir, d))
    ]

    runs = sorted(runs, key=os.path.getmtime)

    if limit is not None:
        runs = runs[-int(limit):]

    if not runs:
        raise RuntimeError(f"No pair_run folders found in: {base_dir}")

    return runs


def create_output_dirs(output_base_dir=OUTPUT_BASE_DIR):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    out_dir = os.path.abspath(
        os.path.join(output_base_dir, f"box_run_{timestamp}")
    )

    dirs = {
        "root": out_dir,
        "images": os.path.join(out_dir, "images"),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"[BoxDataV2] Created {name}: {path}")

    return dirs


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_box(box):
    """
    Validate 2D box.
    """

    if box is None:
        return False

    if not isinstance(box, list):
        return False

    if len(box) != 4:
        return False

    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return False

    bw = x2 - x1
    bh = y2 - y1

    if bw < MIN_BOX_WIDTH or bh < MIN_BOX_HEIGHT:
        return False

    if bw > MAX_BOX_WIDTH or bh > MAX_BOX_HEIGHT:
        return False

    if x2 <= x1 or y2 <= y1:
        return False

    return True


def normalize_box(box, image_width, image_height):
    """
    Convert [x1,y1,x2,y2] to normalized [cx,cy,w,h].
    """

    x1, y1, x2, y2 = [float(v) for v in box]

    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    bw = (x2 - x1) / image_width
    bh = (y2 - y1) / image_height

    return cx, cy, bw, bh


def clamp_box(box, image_width, image_height):
    """
    Clamp box to image limits.
    """

    x1, y1, x2, y2 = [float(v) for v in box]

    x1 = max(0.0, min(image_width - 1.0, x1))
    y1 = max(0.0, min(image_height - 1.0, y1))
    x2 = max(0.0, min(image_width - 1.0, x2))
    y2 = max(0.0, min(image_height - 1.0, y2))

    return [x1, y1, x2, y2]


def get_frame_id_from_name(path):
    """
    frame_000123.png -> 123
    """

    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]

    if not stem.startswith("frame_"):
        return None

    return int(stem.replace("frame_", ""))


def find_real_meta_by_frame(real_sequence, frame_id):
    for item in real_sequence:
        if int(item.get("frame", -1)) == int(frame_id):
            return item

    return None


def safe_float(value, default_value=0.0):
    try:
        if value is None:
            return default_value
        return float(value)
    except Exception:
        return default_value


def safe_int(value, default_value=0):
    try:
        if value is None:
            return default_value
        return int(float(value))
    except Exception:
        return default_value


def safe_get_scenario_value(scenario_config, key, default_value=0.0):
    value = scenario_config.get(key, default_value)
    return safe_float(value, default_value)


def safe_get_int_scenario_value(scenario_config, key, default_value=0):
    value = scenario_config.get(key, default_value)
    return safe_int(value, default_value)


def verify_image_readable(path):
    img = cv2.imread(path)

    if img is None:
        return False

    return True


def estimate_distance_progress(
    distance_to_adversary_m,
    initial_distance_m,
    fallback_frame_progress,
):
    """
    Estimate distance progress:
        0.0 means initial distance
        1.0 means ego reached adversary

    If numeric distance is unavailable, fallback to frame_progress.
    """

    if initial_distance_m is None or initial_distance_m <= 1e-6:
        return fallback_frame_progress

    if distance_to_adversary_m is None:
        return fallback_frame_progress

    progress = 1.0 - (float(distance_to_adversary_m) / float(initial_distance_m))

    # Let it go slightly beyond 1 if ego gets very close/passes,
    # but keep reasonable.
    progress = max(0.0, min(1.5, progress))

    return progress


# ============================================================
# Dataset building
# ============================================================

def process_pair_run(pair_run_dir, output_dirs, start_sample_id):
    """
    Process one pair_run folder.

    Returns:
        rows, next_sample_id, stats
    """

    metadata_path = os.path.join(
        pair_run_dir,
        "metadata",
        "sequence_metadata.json"
    )

    if not os.path.exists(metadata_path):
        print("[BoxDataV2] Missing metadata, skipping:", pair_run_dir)
        return [], start_sample_id, {
            "source_pair_run": pair_run_dir,
            "saved": 0,
            "skipped": 0,
            "reason": "missing_metadata",
        }

    sequence_metadata = load_json(metadata_path)

    real_sequence = sequence_metadata.get("real_sequence", [])
    scenario_config = sequence_metadata.get("scenario_config", {})
    weather = sequence_metadata.get("weather", "")

    clean_dir = os.path.join(pair_run_dir, "clean")

    if not os.path.isdir(clean_dir):
        print("[BoxDataV2] Missing clean dir, skipping:", clean_dir)
        return [], start_sample_id, {
            "source_pair_run": pair_run_dir,
            "saved": 0,
            "skipped": 0,
            "reason": "missing_clean_dir",
        }

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

    rows = []
    saved = 0
    skipped = 0

    num_frames = len(clean_paths)

    if num_frames == 0:
        return [], start_sample_id, {
            "source_pair_run": pair_run_dir,
            "saved": 0,
            "skipped": 0,
            "reason": "no_clean_frames",
        }

    adversary_forward_m = safe_get_scenario_value(
        scenario_config,
        "adversary_forward_m",
        0.0
    )

    adversary_lateral_m = safe_get_scenario_value(
        scenario_config,
        "adversary_lateral_m",
        0.0
    )

    ego_throttle = safe_get_scenario_value(
        scenario_config,
        "ego_throttle",
        0.0
    )

    ego_spawn_index = safe_get_int_scenario_value(
        scenario_config,
        "ego_spawn_index",
        0
    )

    run_id = safe_get_int_scenario_value(
        scenario_config,
        "run_id",
        -1
    )

    # Prefer measured initial distance from metadata if available.
    adversary_meta = sequence_metadata.get("adversary", {})
    initial_distance_to_adversary_m = safe_float(
        adversary_meta.get("initial_distance_to_adversary_m", None),
        adversary_forward_m,
    )

    fixed_delta_seconds = safe_float(
        sequence_metadata.get("fixed_delta_seconds", 0.05),
        0.05,
    )

    sequence_fps = safe_float(
        sequence_metadata.get("sequence_fps", 20.0),
        20.0,
    )

    for clean_path in clean_paths:
        frame_id = get_frame_id_from_name(clean_path)

        if frame_id is None:
            skipped += 1
            continue

        real_meta = find_real_meta_by_frame(
            real_sequence=real_sequence,
            frame_id=frame_id
        )

        if real_meta is None:
            skipped += 1
            continue

        box = real_meta.get("adversary_box_2d", None)

        if not is_valid_box(box):
            skipped += 1
            continue

        box = clamp_box(box, IMAGE_WIDTH, IMAGE_HEIGHT)

        if not is_valid_box(box):
            skipped += 1
            continue

        if not verify_image_readable(clean_path):
            skipped += 1
            continue

        cx, cy, bw, bh = normalize_box(
            box=box,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT
        )

        frame_progress = 0.0

        if num_frames > 1:
            frame_progress = float(frame_id) / float(num_frames - 1)

        # --------------------------------------------------------
        # New distance-aware features from real_sequence metadata
        # --------------------------------------------------------

        ego_speed_mps = safe_float(
            real_meta.get("ego_speed_mps", 0.0),
            0.0,
        )

        ego_forward_displacement_m = safe_float(
            real_meta.get("ego_forward_displacement_m", 0.0),
            0.0,
        )

        distance_to_adversary_m = safe_float(
            real_meta.get("distance_to_adversary_m", None),
            None,
        )

        # Some frames may have explicit initial_adversary_distance_m.
        per_frame_initial_distance_m = safe_float(
            real_meta.get("initial_adversary_distance_m", None),
            initial_distance_to_adversary_m,
        )

        distance_progress = safe_float(
            real_meta.get("distance_progress", None),
            None,
        )

        if distance_progress is None:
            distance_progress = estimate_distance_progress(
                distance_to_adversary_m=distance_to_adversary_m,
                initial_distance_m=per_frame_initial_distance_m,
                fallback_frame_progress=frame_progress,
            )

        # Useful derived values.
        time_seconds = float(frame_id) * fixed_delta_seconds

        estimated_distance_from_displacement_m = (
            per_frame_initial_distance_m - ego_forward_displacement_m
        )

        if estimated_distance_from_displacement_m < 0.0:
            estimated_distance_from_displacement_m = 0.0

        sample_id = start_sample_id

        if COPY_IMAGES:
            sample_name = f"sample_{sample_id:06d}.png"
            output_image_path = os.path.join(
                output_dirs["images"],
                sample_name
            )

            shutil.copy2(clean_path, output_image_path)
            image_path_for_csv = output_image_path
        else:
            image_path_for_csv = clean_path

        x1, y1, x2, y2 = box

        row = {
            "sample_id": sample_id,
            "image_path": image_path_for_csv,
            "source_pair_run": pair_run_dir,
            "source_run_name": os.path.basename(pair_run_dir),
            "frame_id": frame_id,

            # Target normalized box
            "cx": cx,
            "cy": cy,
            "w": bw,
            "h": bh,

            # Target pixel box
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,

            # Original scalar features
            "frame_progress": frame_progress,
            "adversary_forward_m": adversary_forward_m,
            "adversary_lateral_m": adversary_lateral_m,
            "ego_throttle": ego_throttle,
            "ego_spawn_index": ego_spawn_index,

            # New temporal/distance-aware features
            "time_seconds": time_seconds,
            "fixed_delta_seconds": fixed_delta_seconds,
            "sequence_fps": sequence_fps,
            "initial_distance_to_adversary_m": initial_distance_to_adversary_m,
            "initial_adversary_distance_m": per_frame_initial_distance_m,
            "distance_to_adversary_m": distance_to_adversary_m,
            "estimated_distance_from_displacement_m": estimated_distance_from_displacement_m,
            "distance_progress": distance_progress,
            "ego_speed_mps": ego_speed_mps,
            "ego_forward_displacement_m": ego_forward_displacement_m,

            "run_id": run_id,
            "weather": weather,
        }

        rows.append(row)

        start_sample_id += 1
        saved += 1

    stats = {
        "source_pair_run": pair_run_dir,
        "saved": saved,
        "skipped": skipped,
        "num_clean_frames": num_frames,
        "scenario_config": scenario_config,
        "weather": weather,
        "initial_distance_to_adversary_m": initial_distance_to_adversary_m,
    }

    print(
        "[BoxDataV2] Processed:",
        os.path.basename(pair_run_dir),
        "| saved:",
        saved,
        "| skipped:",
        skipped
    )

    return rows, start_sample_id, stats


def write_labels_csv(rows, csv_path):
    fieldnames = [
        "sample_id",
        "image_path",
        "source_pair_run",
        "source_run_name",
        "frame_id",

        "cx",
        "cy",
        "w",
        "h",

        "x1",
        "y1",
        "x2",
        "y2",

        # Original scalar features
        "frame_progress",
        "adversary_forward_m",
        "adversary_lateral_m",
        "ego_throttle",
        "ego_spawn_index",

        # New v2 scalar features
        "time_seconds",
        "fixed_delta_seconds",
        "sequence_fps",
        "initial_distance_to_adversary_m",
        "initial_adversary_distance_m",
        "distance_to_adversary_m",
        "estimated_distance_from_displacement_m",
        "distance_progress",
        "ego_speed_mps",
        "ego_forward_displacement_m",

        "run_id",
        "weather",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def build_box_dataset(pair_runs, output_dirs):
    all_rows = []
    all_stats = []

    sample_id = 0

    for pair_run_dir in pair_runs:
        rows, sample_id, stats = process_pair_run(
            pair_run_dir=pair_run_dir,
            output_dirs=output_dirs,
            start_sample_id=sample_id,
        )

        all_rows.extend(rows)
        all_stats.append(stats)

    labels_csv_path = os.path.join(
        output_dirs["root"],
        "labels.csv"
    )

    write_labels_csv(all_rows, labels_csv_path)

    metadata = {
        "dataset_version": "box_dataset_v2_distance_aware",
        "num_samples": len(all_rows),
        "num_pair_runs": len(pair_runs),
        "copy_images": COPY_IMAGES,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "label_format": "normalized_cx_cy_w_h",
        "inputs": {
            "image": "clean RGB frame",
            "scalar_features_v1": [
                "frame_progress",
                "adversary_forward_m",
                "adversary_lateral_m",
                "ego_throttle",
                "ego_spawn_index",
            ],
            "scalar_features_v2_distance_aware": [
                "frame_progress",
                "adversary_forward_m",
                "adversary_lateral_m",
                "ego_throttle",
                "ego_spawn_index",
                "time_seconds",
                "initial_distance_to_adversary_m",
                "distance_to_adversary_m",
                "estimated_distance_from_displacement_m",
                "distance_progress",
                "ego_speed_mps",
                "ego_forward_displacement_m",
            ],
        },
        "target": {
            "cx": "normalized box center x",
            "cy": "normalized box center y",
            "w": "normalized box width",
            "h": "normalized box height",
        },
        "pair_runs": pair_runs,
        "run_stats": all_stats,
    }

    metadata_path = os.path.join(
        output_dirs["root"],
        "metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("[BoxDataV2] Dataset complete.")
    print("[BoxDataV2] Samples:", len(all_rows))
    print("[BoxDataV2] Labels:", labels_csv_path)
    print("[BoxDataV2] Metadata:", metadata_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paired_base_dir",
        type=str,
        default=PAIRED_BASE_DIR,
        help="Directory containing pair_run folders.",
    )

    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=OUTPUT_BASE_DIR,
        help="Output base directory for box dataset.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the latest N pair runs.",
    )

    args = parser.parse_args()

    pair_runs = find_pair_runs(
        base_dir=args.paired_base_dir,
        limit=args.limit,
    )

    print("[BoxDataV2] Found pair runs:", len(pair_runs))

    for run in pair_runs:
        print("   ", run)

    output_dirs = create_output_dirs(
        output_base_dir=args.output_base_dir
    )

    build_box_dataset(
        pair_runs=pair_runs,
        output_dirs=output_dirs,
    )


if __name__ == "__main__":
    main()