import os
import csv
import cv2
import argparse
import random
from datetime import datetime


# ============================================================
# Settings
# ============================================================

BOX_DATASET_BASE = "box_dataset"

OUTPUT_PREVIEW_DIR_NAME = "box_visualizations"

NUM_PREVIEWS = 100

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


# ============================================================
# Utility
# ============================================================

def find_latest_box_run(base_dir=BOX_DATASET_BASE):
    """
    Find latest box_run folder.
    """

    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Box dataset base not found: {base_dir}")

    runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("box_run_")
        and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not runs:
        raise RuntimeError(f"No box_run folders found in: {base_dir}")

    runs = sorted(runs, key=os.path.getmtime, reverse=True)

    return runs[0]


def read_labels_csv(csv_path):
    """
    Read labels.csv into list of dictionaries.
    """

    rows = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def denormalize_box(row, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
    """
    Convert normalized cx, cy, w, h to pixel x1, y1, x2, y2.

    We also support x1,y1,x2,y2 directly from CSV.
    """

    if all(k in row for k in ["x1", "y1", "x2", "y2"]):
        x1 = safe_float(row["x1"])
        y1 = safe_float(row["y1"])
        x2 = safe_float(row["x2"])
        y2 = safe_float(row["y2"])

        return [int(x1), int(y1), int(x2), int(y2)]

    cx = safe_float(row["cx"])
    cy = safe_float(row["cy"])
    bw = safe_float(row["w"])
    bh = safe_float(row["h"])

    cx_px = cx * image_width
    cy_px = cy * image_height
    bw_px = bw * image_width
    bh_px = bh * image_height

    x1 = int(cx_px - bw_px / 2.0)
    y1 = int(cy_px - bh_px / 2.0)
    x2 = int(cx_px + bw_px / 2.0)
    y2 = int(cy_px + bh_px / 2.0)

    return [x1, y1, x2, y2]


def draw_label_box(img, row):
    """
    Draw target box and text information.
    """

    output = img.copy()

    h, w = output.shape[:2]

    x1, y1, x2, y2 = denormalize_box(
        row,
        image_width=w,
        image_height=h
    )

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    # Draw target box.
    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    sample_id = row.get("sample_id", "NA")
    frame_id = row.get("frame_id", "NA")
    source_run = row.get("source_run_name", "NA")

    cx = safe_float(row.get("cx", 0.0))
    cy = safe_float(row.get("cy", 0.0))
    bw = safe_float(row.get("w", 0.0))
    bh = safe_float(row.get("h", 0.0))

    frame_progress = safe_float(row.get("frame_progress", 0.0))
    adv_dist = safe_float(row.get("adversary_forward_m", 0.0))
    lat = safe_float(row.get("adversary_lateral_m", 0.0))
    throttle = safe_float(row.get("ego_throttle", 0.0))
    spawn_index = row.get("ego_spawn_index", "NA")

    lines = [
        f"sample={sample_id} frame={frame_id}",
        f"run={source_run}",
        f"box cx={cx:.3f} cy={cy:.3f} w={bw:.3f} h={bh:.3f}",
        f"progress={frame_progress:.3f} dist={adv_dist:.1f} lat={lat:.2f}",
        f"throttle={throttle:.2f} spawn={spawn_index}",
    ]

    # Background panel.
    panel_h = 28 * len(lines) + 8

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], panel_h),
        (0, 0, 0),
        -1
    )

    for i, text in enumerate(lines):
        y = 25 + i * 28

        cv2.putText(
            output,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


def make_contact_sheet(images, cols=2):
    """
    Make contact sheet from list of BGR images.
    """

    if not images:
        return None

    # Resize all to same smaller size.
    resized = []

    target_w = 640
    target_h = 360

    for img in images:
        resized.append(cv2.resize(img, (target_w, target_h)))

    rows = []

    for i in range(0, len(resized), cols):
        row_imgs = resized[i:i + cols]

        while len(row_imgs) < cols:
            row_imgs.append(255 * (row_imgs[0] * 0 + 1))

        row = cv2.hconcat(row_imgs)
        rows.append(row)

    sheet = cv2.vconcat(rows)

    return sheet


# ============================================================
# Visualization
# ============================================================

def visualize_box_dataset(box_run_dir, num_previews=NUM_PREVIEWS, random_sample=True):
    labels_path = os.path.join(box_run_dir, "labels.csv")

    if not os.path.exists(labels_path):
        raise RuntimeError(f"labels.csv not found: {labels_path}")

    rows = read_labels_csv(labels_path)

    if not rows:
        raise RuntimeError("No rows found in labels.csv")

    output_dir = os.path.join(box_run_dir, OUTPUT_PREVIEW_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    print("[BoxViz] Box run:", box_run_dir)
    print("[BoxViz] Labels:", labels_path)
    print("[BoxViz] Rows:", len(rows))
    print("[BoxViz] Output:", output_dir)

    if random_sample:
        selected = random.sample(rows, min(num_previews, len(rows)))
    else:
        selected = rows[:num_previews]

    preview_images = []

    saved = 0
    skipped = 0

    for i, row in enumerate(selected):
        image_path = row.get("image_path", "")

        if not os.path.exists(image_path):
            print("[BoxViz] Missing image:", image_path)
            skipped += 1
            continue

        img = cv2.imread(image_path)

        if img is None:
            print("[BoxViz] Could not read:", image_path)
            skipped += 1
            continue

        vis = draw_label_box(img, row)

        sample_id = safe_int(row.get("sample_id", i))
        out_name = f"box_preview_{sample_id:06d}.png"

        out_path = os.path.join(output_dir, out_name)

        cv2.imwrite(out_path, vis)

        # Store first few for contact sheet.
        if len(preview_images) < 12:
            preview_images.append(vis)

        saved += 1

        if saved % 25 == 0:
            print(f"[BoxViz] Saved {saved} previews...")

    if preview_images:
        sheet = make_contact_sheet(preview_images, cols=2)

        if sheet is not None:
            sheet_path = os.path.join(output_dir, "contact_sheet.png")
            cv2.imwrite(sheet_path, sheet)
            print("[BoxViz] Contact sheet:", sheet_path)

    print("[BoxViz] Done.")
    print("[BoxViz] Saved:", saved)
    print("[BoxViz] Skipped:", skipped)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--box_run_dir",
        type=str,
        default=None,
        help="Path to box_run folder. If omitted, latest box_run is used."
    )

    parser.add_argument(
        "--num_previews",
        type=int,
        default=NUM_PREVIEWS,
        help="Number of preview images to generate."
    )

    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Use first N samples instead of random samples."
    )

    args = parser.parse_args()

    if args.box_run_dir is None:
        box_run_dir = find_latest_box_run(BOX_DATASET_BASE)
    else:
        box_run_dir = args.box_run_dir

    visualize_box_dataset(
        box_run_dir=box_run_dir,
        num_previews=args.num_previews,
        random_sample=not args.ordered,
    )


if __name__ == "__main__":
    main()