import os
import csv
import json
import math
import argparse
from statistics import mean, median


# ============================================================
# Settings
# ============================================================

PAIRED_BASE_DIR = "paired_data"


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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def is_valid_box(box):
    if box is None:
        return False

    if not isinstance(box, list):
        return False

    if len(box) != 4:
        return False

    x1, y1, x2, y2 = [safe_float(v) for v in box]

    if x2 <= x1 or y2 <= y1:
        return False

    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return False

    return True


def box_center(box):
    x1, y1, x2, y2 = [safe_float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def box_size(box):
    x1, y1, x2, y2 = [safe_float(v) for v in box]
    w = x2 - x1
    h = y2 - y1
    return w, h


def box_area(box):
    w, h = box_size(box)
    return max(0.0, w) * max(0.0, h)


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [safe_float(v) for v in box_a]
    bx1, by1, bx2, by2 = [safe_float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0

    return inter_area / union


def center_error_px(box_a, box_b):
    acx, acy = box_center(box_a)
    bcx, bcy = box_center(box_b)

    dx = acx - bcx
    dy = acy - bcy

    dist = math.sqrt(dx * dx + dy * dy)
    return dist, dx, dy


def find_real_meta_by_frame(real_sequence, frame_id):
    for item in real_sequence:
        if int(item.get("frame", -1)) == int(frame_id):
            return item
    return None


def summarize(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


# ============================================================
# Main evaluation
# ============================================================

def evaluate_pair_run(run_dir):
    sequence_metadata_path = os.path.join(
        run_dir,
        "metadata",
        "sequence_metadata.json",
    )

    pred_metadata_path = os.path.join(
        run_dir,
        "metadata",
        "he_box_predicted_metadata.json",
    )

    if not os.path.exists(sequence_metadata_path):
        raise RuntimeError(f"Missing sequence metadata: {sequence_metadata_path}")

    if not os.path.exists(pred_metadata_path):
        raise RuntimeError(
            f"Missing predicted-box metadata: {pred_metadata_path}\n"
            f"Run apply_box_predictor.py first."
        )

    sequence_metadata = load_json(sequence_metadata_path)
    pred_metadata = load_json(pred_metadata_path)

    real_sequence = sequence_metadata.get("real_sequence", [])
    pred_frames = pred_metadata.get("frames", [])

    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    per_frame_csv_path = os.path.join(
        metrics_dir,
        "box_predictor_eval_per_frame.csv",
    )

    summary_json_path = os.path.join(
        metrics_dir,
        "box_predictor_eval_summary.json",
    )

    per_frame_rows = []

    ious = []
    center_errors = []
    abs_dx_errors = []
    abs_dy_errors = []
    width_errors = []
    height_errors = []
    width_abs_pct_errors = []
    height_abs_pct_errors = []
    area_ratios = []

    total_frames = 0
    valid_frames = 0
    invalid_pred_frames = 0
    invalid_real_frames = 0

    for item in pred_frames:
        frame_id = int(item.get("frame", -1))
        pred_box = item.get("pred_box_2d", None)

        real_meta = find_real_meta_by_frame(real_sequence, frame_id)
        real_box = None if real_meta is None else real_meta.get("adversary_box_2d", None)

        total_frames += 1

        pred_valid = is_valid_box(pred_box)
        real_valid = is_valid_box(real_box)

        if not pred_valid:
            invalid_pred_frames += 1

        if not real_valid:
            invalid_real_frames += 1

        row = {
            "frame": frame_id,
            "pred_valid": pred_valid,
            "real_valid": real_valid,
            "pred_x1": None,
            "pred_y1": None,
            "pred_x2": None,
            "pred_y2": None,
            "real_x1": None,
            "real_y1": None,
            "real_x2": None,
            "real_y2": None,
            "iou": None,
            "center_error_px": None,
            "dx_px": None,
            "dy_px": None,
            "pred_w": None,
            "pred_h": None,
            "real_w": None,
            "real_h": None,
            "width_error_px": None,
            "height_error_px": None,
            "width_abs_pct_error": None,
            "height_abs_pct_error": None,
            "area_ratio_pred_over_real": None,
        }

        if pred_valid:
            row["pred_x1"], row["pred_y1"], row["pred_x2"], row["pred_y2"] = pred_box
            pred_w, pred_h = box_size(pred_box)
            row["pred_w"] = pred_w
            row["pred_h"] = pred_h

        if real_valid:
            row["real_x1"], row["real_y1"], row["real_x2"], row["real_y2"] = real_box
            real_w, real_h = box_size(real_box)
            row["real_w"] = real_w
            row["real_h"] = real_h

        if pred_valid and real_valid:
            valid_frames += 1

            iou = compute_iou(pred_box, real_box)
            c_err, dx, dy = center_error_px(pred_box, real_box)

            pred_w, pred_h = box_size(pred_box)
            real_w, real_h = box_size(real_box)

            width_error = pred_w - real_w
            height_error = pred_h - real_h

            width_abs_pct_error = abs(width_error) / max(1e-6, real_w)
            height_abs_pct_error = abs(height_error) / max(1e-6, real_h)

            pred_area = box_area(pred_box)
            real_area = box_area(real_box)
            area_ratio = pred_area / max(1e-6, real_area)

            row["iou"] = iou
            row["center_error_px"] = c_err
            row["dx_px"] = dx
            row["dy_px"] = dy
            row["width_error_px"] = width_error
            row["height_error_px"] = height_error
            row["width_abs_pct_error"] = width_abs_pct_error
            row["height_abs_pct_error"] = height_abs_pct_error
            row["area_ratio_pred_over_real"] = area_ratio

            ious.append(iou)
            center_errors.append(c_err)
            abs_dx_errors.append(abs(dx))
            abs_dy_errors.append(abs(dy))
            width_errors.append(abs(width_error))
            height_errors.append(abs(height_error))
            width_abs_pct_errors.append(width_abs_pct_error)
            height_abs_pct_errors.append(height_abs_pct_error)
            area_ratios.append(area_ratio)

        per_frame_rows.append(row)

    # Write CSV
    fieldnames = [
        "frame",
        "pred_valid",
        "real_valid",
        "pred_x1", "pred_y1", "pred_x2", "pred_y2",
        "real_x1", "real_y1", "real_x2", "real_y2",
        "iou",
        "center_error_px",
        "dx_px",
        "dy_px",
        "pred_w",
        "pred_h",
        "real_w",
        "real_h",
        "width_error_px",
        "height_error_px",
        "width_abs_pct_error",
        "height_abs_pct_error",
        "area_ratio_pred_over_real",
    ]

    with open(per_frame_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_frame_rows:
            writer.writerow(row)

    summary = {
        "run_dir": run_dir,
        "total_predicted_frames": total_frames,
        "valid_eval_frames": valid_frames,
        "invalid_pred_frames": invalid_pred_frames,
        "invalid_real_frames": invalid_real_frames,
        "metrics": {
            "iou": summarize(ious),
            "center_error_px": summarize(center_errors),
            "abs_dx_error_px": summarize(abs_dx_errors),
            "abs_dy_error_px": summarize(abs_dy_errors),
            "abs_width_error_px": summarize(width_errors),
            "abs_height_error_px": summarize(height_errors),
            "width_abs_pct_error": summarize(width_abs_pct_errors),
            "height_abs_pct_error": summarize(height_abs_pct_errors),
            "area_ratio_pred_over_real": summarize(area_ratios),
        },
        "interpretation": {
            "better_iou": "higher is better",
            "better_center_error_px": "lower is better",
            "better_width_height_error_px": "lower is better",
            "better_area_ratio": "closer to 1.0 is better",
        },
        "files": {
            "per_frame_csv": per_frame_csv_path,
            "summary_json": summary_json_path,
        },
    }

    save_json(summary_json_path, summary)

    print("[EvalBox] Done.")
    print("[EvalBox] Run:", run_dir)
    print("[EvalBox] Total predicted frames:", total_frames)
    print("[EvalBox] Valid eval frames:", valid_frames)
    print("[EvalBox] IoU mean:", summary["metrics"]["iou"]["mean"])
    print("[EvalBox] IoU median:", summary["metrics"]["iou"]["median"])
    print("[EvalBox] Center error mean (px):", summary["metrics"]["center_error_px"]["mean"])
    print("[EvalBox] Width error mean abs (px):", summary["metrics"]["abs_width_error_px"]["mean"])
    print("[EvalBox] Height error mean abs (px):", summary["metrics"]["abs_height_error_px"]["mean"])
    print("[EvalBox] Area ratio mean:", summary["metrics"]["area_ratio_pred_over_real"]["mean"])
    print("[EvalBox] CSV:", per_frame_csv_path)
    print("[EvalBox] Summary:", summary_json_path)


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

    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = find_latest_pair_run(PAIRED_BASE_DIR)
    else:
        run_dir = args.run_dir

    evaluate_pair_run(run_dir)


if __name__ == "__main__":
    main()