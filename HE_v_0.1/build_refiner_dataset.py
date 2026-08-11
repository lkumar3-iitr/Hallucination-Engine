import os
import cv2
import json
import glob
import argparse
import numpy as np
from datetime import datetime


# ============================================================
# Settings
# ============================================================

PAIRED_BASE_DIR = "paired_data"
OUTPUT_BASE_DIR = "refiner_dataset"

CROP_SIZE = 256

# Expand crop around object bbox.
# 1.0 = exact box, 2.0 = double size around center.
CROP_EXPAND_FACTOR = 2.0

MIN_BOX_WIDTH = 12
MIN_BOX_HEIGHT = 12

MAX_BOX_WIDTH = 900
MAX_BOX_HEIGHT = 700

# If rough-clean difference is weak, fallback to bbox mask.
DIFF_THRESHOLD = 12


# ============================================================
# Utility
# ============================================================

def find_latest_pair_run(base_dir=PAIRED_BASE_DIR):
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Base directory not found: {base_dir}")

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


def create_output_dirs(output_base_dir=OUTPUT_BASE_DIR):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    out_dir = os.path.abspath(
        os.path.join(output_base_dir, f"refiner_run_{timestamp}")
    )

    dirs = {
        "root": out_dir,
        "clean_crop": os.path.join(out_dir, "clean_crop"),
        "rough_crop": os.path.join(out_dir, "rough_crop"),
        "target_crop": os.path.join(out_dir, "target_crop"),
        "mask_crop": os.path.join(out_dir, "mask_crop"),
        "preview": os.path.join(out_dir, "preview"),
        "metadata": os.path.join(out_dir, "metadata"),
    }

    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"[RefinerData] Created {name}: {path}")

    return dirs


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_bgr(path):
    img = cv2.imread(path)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    return img


def save_bgr(path, img):
    cv2.imwrite(path, img)


def is_valid_box(box):
    if box is None:
        return False

    if len(box) != 4:
        return False

    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
        return False

    if w > MAX_BOX_WIDTH or h > MAX_BOX_HEIGHT:
        return False

    return True


def square_expanded_crop_box(box, image_w, image_h, expand_factor):
    """
    Build square crop around bbox center.

    Returns:
        [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = [float(v) for v in box]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw = x2 - x1
    bh = y2 - y1

    side = max(bw, bh) * expand_factor

    # Avoid tiny crops
    side = max(side, 64.0)

    crop_x1 = int(round(cx - side / 2.0))
    crop_y1 = int(round(cy - side / 2.0))
    crop_x2 = int(round(cx + side / 2.0))
    crop_y2 = int(round(cy + side / 2.0))

    # Shift crop into image while preserving size as much as possible.
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0

    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0

    if crop_x2 > image_w:
        shift = crop_x2 - image_w
        crop_x1 -= shift
        crop_x2 = image_w

    if crop_y2 > image_h:
        shift = crop_y2 - image_h
        crop_y1 -= shift
        crop_y2 = image_h

    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(image_w, crop_x2)
    crop_y2 = min(image_h, crop_y2)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    return [crop_x1, crop_y1, crop_x2, crop_y2]


def crop_and_resize(img, crop_box, size=CROP_SIZE):
    x1, y1, x2, y2 = crop_box

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    return crop


def make_mask_from_rough_difference(clean_bgr, rough_bgr, box, crop_box):
    """
    Create mask from difference between rough HE image and clean image,
    constrained around bbox.

    This is better than full bbox mask because it follows sprite shape
    when the sprite has transparent background.
    """

    diff = cv2.absdiff(clean_bgr, rough_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    mask = (diff_gray > DIFF_THRESHOLD).astype(np.uint8) * 255

    # Constrain to bbox plus margin.
    h, w = mask.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in box]

    bw = x2 - x1
    bh = y2 - y1

    margin_x = int(0.20 * bw)
    margin_y = int(0.20 * bh)

    x1m = max(0, x1 - margin_x)
    y1m = max(0, y1 - margin_y)
    x2m = min(w - 1, x2 + margin_x)
    y2m = min(h - 1, y2 + margin_y)

    constraint = np.zeros_like(mask)
    constraint[y1m:y2m + 1, x1m:x2m + 1] = 255

    mask = cv2.bitwise_and(mask, constraint)

    # Clean mask.
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fallback if difference mask is too small.
    if np.sum(mask > 0) < 50:
        mask = np.zeros_like(mask)
        mask[y1:y2 + 1, x1:x2 + 1] = 255

    mask_crop = crop_and_resize(mask, crop_box, CROP_SIZE)

    if mask_crop is None:
        return None

    if len(mask_crop.shape) == 3:
        mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_BGR2GRAY)

    # Make binary-ish after resize.
    mask_crop = (mask_crop > 30).astype(np.uint8) * 255

    return mask_crop


def make_preview(clean_crop, rough_crop, target_crop, mask_crop):
    """
    Create side-by-side preview:
        clean | rough | target | mask
    """

    if len(mask_crop.shape) == 2:
        mask_vis = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
    else:
        mask_vis = mask_crop

    # Add labels
    panels = [
        ("clean", clean_crop),
        ("rough", rough_crop),
        ("target", target_crop),
        ("mask", mask_vis),
    ]

    labeled = []

    for label, img in panels:
        panel = img.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labeled.append(panel)

    return np.hstack(labeled)


def find_metadata_for_frame(real_sequence_metadata, frame_id):
    for item in real_sequence_metadata:
        if int(item.get("frame", -1)) == int(frame_id):
            return item

    return None


# ============================================================
# Main dataset builder
# ============================================================

def build_dataset(pair_run_dir, output_dirs):
    metadata_path = os.path.join(pair_run_dir, "metadata", "sequence_metadata.json")

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Missing metadata file: {metadata_path}")

    sequence_metadata = load_json(metadata_path)

    real_sequence = sequence_metadata.get("real_sequence", [])

    clean_dir = os.path.join(pair_run_dir, "clean")
    rough_dir = os.path.join(pair_run_dir, "he_oracle_box")
    target_dir = os.path.join(pair_run_dir, "real_object")

    if not os.path.isdir(clean_dir):
        raise RuntimeError(f"Missing clean dir: {clean_dir}")

    if not os.path.isdir(rough_dir):
        raise RuntimeError(f"Missing he_oracle_box dir: {rough_dir}")

    if not os.path.isdir(target_dir):
        raise RuntimeError(f"Missing real_object dir: {target_dir}")

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

    samples = []
    skipped = 0
    saved = 0

    print("[RefinerData] Source pair run:", pair_run_dir)
    print("[RefinerData] Found clean frames:", len(clean_paths))

    for clean_path in clean_paths:
        frame_name = os.path.basename(clean_path)
        frame_id_str = os.path.splitext(frame_name)[0].replace("frame_", "")
        frame_id = int(frame_id_str)

        frame_meta = find_metadata_for_frame(real_sequence, frame_id)

        if frame_meta is None:
            skipped += 1
            continue

        box = frame_meta.get("adversary_box_2d", None)

        if not is_valid_box(box):
            skipped += 1
            continue

        rough_path = os.path.join(rough_dir, frame_name)
        target_path = os.path.join(target_dir, frame_name)

        if not os.path.exists(rough_path) or not os.path.exists(target_path):
            skipped += 1
            continue

        clean_bgr = read_bgr(clean_path)
        rough_bgr = read_bgr(rough_path)
        target_bgr = read_bgr(target_path)

        h, w = clean_bgr.shape[:2]

        crop_box = square_expanded_crop_box(
            box=box,
            image_w=w,
            image_h=h,
            expand_factor=CROP_EXPAND_FACTOR,
        )

        if crop_box is None:
            skipped += 1
            continue

        clean_crop = crop_and_resize(clean_bgr, crop_box, CROP_SIZE)
        rough_crop = crop_and_resize(rough_bgr, crop_box, CROP_SIZE)
        target_crop = crop_and_resize(target_bgr, crop_box, CROP_SIZE)

        if clean_crop is None or rough_crop is None or target_crop is None:
            skipped += 1
            continue

        mask_crop = make_mask_from_rough_difference(
            clean_bgr=clean_bgr,
            rough_bgr=rough_bgr,
            box=box,
            crop_box=crop_box,
        )

        if mask_crop is None:
            skipped += 1
            continue

        sample_name = f"sample_{saved:06d}.png"

        clean_out = os.path.join(output_dirs["clean_crop"], sample_name)
        rough_out = os.path.join(output_dirs["rough_crop"], sample_name)
        target_out = os.path.join(output_dirs["target_crop"], sample_name)
        mask_out = os.path.join(output_dirs["mask_crop"], sample_name)

        save_bgr(clean_out, clean_crop)
        save_bgr(rough_out, rough_crop)
        save_bgr(target_out, target_crop)
        cv2.imwrite(mask_out, mask_crop)

        preview = make_preview(clean_crop, rough_crop, target_crop, mask_crop)
        preview_out = os.path.join(output_dirs["preview"], sample_name)
        save_bgr(preview_out, preview)

        samples.append({
            "sample_id": saved,
            "frame_id": frame_id,
            "source_pair_run": pair_run_dir,
            "box_2d": box,
            "crop_box": crop_box,
            "paths": {
                "clean_crop": clean_out,
                "rough_crop": rough_out,
                "target_crop": target_out,
                "mask_crop": mask_out,
                "preview": preview_out,
            }
        })

        saved += 1

        if saved % 25 == 0:
            print(f"[RefinerData] Saved {saved} samples...")

    dataset_meta = {
        "source_pair_run": pair_run_dir,
        "crop_size": CROP_SIZE,
        "crop_expand_factor": CROP_EXPAND_FACTOR,
        "num_samples": saved,
        "num_skipped": skipped,
        "samples": samples,
    }

    meta_out = os.path.join(output_dirs["metadata"], "dataset_metadata.json")

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=4)

    print("[RefinerData] Done.")
    print("[RefinerData] Saved samples:", saved)
    print("[RefinerData] Skipped frames:", skipped)
    print("[RefinerData] Metadata:", meta_out)

    return dataset_meta


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pair_run_dir",
        type=str,
        default=None,
        help="Path to pair_run folder. If omitted, latest run in paired_data is used."
    )

    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=OUTPUT_BASE_DIR,
        help="Output base directory for refiner dataset."
    )

    args = parser.parse_args()

    if args.pair_run_dir is None:
        pair_run_dir = find_latest_pair_run(PAIRED_BASE_DIR)
    else:
        pair_run_dir = args.pair_run_dir

    output_dirs = create_output_dirs(args.output_base_dir)

    build_dataset(
        pair_run_dir=pair_run_dir,
        output_dirs=output_dirs,
    )


if __name__ == "__main__":
    main()