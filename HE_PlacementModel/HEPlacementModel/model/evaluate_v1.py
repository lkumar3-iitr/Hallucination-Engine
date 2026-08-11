import argparse
import json
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class PlacementMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

        self.visible_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.backbone(x)
        bbox = self.bbox_head(h)
        visible_logit = self.visible_head(h)
        return bbox, visible_logit


def load_rows(labels_path):
    rows = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {labels_path}")
    return rows


def build_input_vector(row):
    rel = row["relative_state"]
    cam = row["camera"]

    width = float(cam["width"])
    height = float(cam["height"])

    x = np.array(
        [
            float(rel["rel_x"]),
            float(rel["rel_z"]),
            float(rel["rel_y"]),
            float(rel["rel_yaw"]),
            float(cam["fx"]),
            float(cam["fy"]),
            float(cam["cx"]),
            float(cam["cy"]),
            width,
            height,
            float(cam["fov"]),
            float(cam["mount_x"]),
            float(cam["mount_y"]),
            float(cam["mount_z"]),
            float(cam["pitch"]),
            float(cam["yaw"]),
            float(cam["roll"]),
        ],
        dtype=np.float32,
    )

    return x


def target_to_pixels(row):
    target = row["target"]

    return {
        "visible": int(target["visible"]),
        "center_x": float(target["center_x"]),
        "bottom_y": float(target["bottom_y"]),
        "box_width": float(target["box_width"]),
        "box_height": float(target["box_height"]),
    }


def pred_to_pixels(pred_bbox_norm, pred_visible_prob, row):
    cam = row["camera"]
    width = float(cam["width"])
    height = float(cam["height"])

    center_x = float(pred_bbox_norm[0]) * width
    bottom_y = float(pred_bbox_norm[1]) * height
    box_width = float(pred_bbox_norm[2]) * width
    box_height = float(pred_bbox_norm[3]) * height

    return {
        "visible_prob": float(pred_visible_prob),
        "visible": int(pred_visible_prob >= 0.5),
        "center_x": center_x,
        "bottom_y": bottom_y,
        "box_width": box_width,
        "box_height": box_height,
    }


def box_to_xyxy(box):
    cx = box["center_x"]
    by = box["bottom_y"]
    w = box["box_width"]
    h = box["box_height"]

    x1 = cx - w / 2.0
    x2 = cx + w / 2.0
    y2 = by
    y1 = by - h

    return x1, y1, x2, y2


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_to_xyxy(box_a)
    bx1, by1, bx2, by2 = box_to_xyxy(box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def normalize_input(x, normalizer, device):
    mean = torch.tensor(normalizer["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(normalizer["std"], dtype=torch.float32, device=device)

    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    return (x_t - mean) / std


def distance_bucket(rel_z):
    rel_z = float(rel_z)

    if rel_z < 10:
        return "00_10m"
    if rel_z < 20:
        return "10_20m"
    if rel_z < 40:
        return "20_40m"
    if rel_z < 60:
        return "40_60m"
    return "60m_plus"


def yaw_bucket(rel_yaw):
    a = abs(float(rel_yaw))

    if a < 30:
        return "yaw_000_030"
    if a < 60:
        return "yaw_030_060"
    if a < 90:
        return "yaw_060_090"
    if a < 120:
        return "yaw_090_120"
    if a < 150:
        return "yaw_120_150"
    return "yaw_150_180"


def add_bucket_stats(bucket_dict, bucket_name, err):
    if bucket_name not in bucket_dict:
        bucket_dict[bucket_name] = []
    bucket_dict[bucket_name].append(err)


def summarize_errors(errors):
    if not errors:
        return {}

    arr = np.array(errors, dtype=np.float32)

    return {
        "count": int(arr.shape[0]),
        "mae_center_x_px": float(np.mean(np.abs(arr[:, 0]))),
        "mae_bottom_y_px": float(np.mean(np.abs(arr[:, 1]))),
        "mae_box_width_px": float(np.mean(np.abs(arr[:, 2]))),
        "mae_box_height_px": float(np.mean(np.abs(arr[:, 3]))),
        "mean_iou": float(np.mean(arr[:, 4])),
    }


def save_scatter_plot(xs, ys, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=8, alpha=0.35)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_error_hist(values, xlabel, title, out_path):
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_box(ax, box, color, label):
    x1, y1, x2, y2 = box_to_xyxy(box)

    rect_x = x1
    rect_y = y1
    rect_w = x2 - x1
    rect_h = y2 - y1

    ax.add_patch(
        plt.Rectangle(
            (rect_x, rect_y),
            rect_w,
            rect_h,
            fill=False,
            edgecolor=color,
            linewidth=2,
            label=label,
        )
    )

    ax.scatter([box["center_x"]], [box["bottom_y"]], c=color, s=25)


def save_box_visualization(row, gt, pred, out_path):
    cam = row["camera"]
    width = int(cam["width"])
    height = int(cam["height"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    draw_box(ax, gt, "green", "GT")
    draw_box(ax, pred, "red", "Pred")

    rel = row["relative_state"]
    iou = compute_iou(gt, pred)

    title = (
        f"id={row.get('global_sample_id', row.get('sample_id'))} | "
        f"rel_z={float(rel['rel_z']):.2f} m | "
        f"rel_x={float(rel['rel_x']):.2f} m | "
        f"rel_yaw={float(rel['rel_yaw']):.1f} deg | "
        f"IoU={iou:.3f}"
    )

    ax.set_title(title)
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        default="dataset/v1_straight_combined/labels.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/v1_mlp_combined/heplacement_v1_mlp_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/v1_eval",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for faster evaluation.",
    )
    parser.add_argument(
        "--num-visuals",
        type=int,
        default=100,
        help="Number of GT-vs-pred box visualizations to save.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    visual_dir = output_dir / "box_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    print(f"[INFO] Loading labels: {args.labels}")
    rows = load_rows(args.labels)

    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    print(f"[INFO] Loaded rows: {len(rows)}")

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = PlacementMLP(input_dim=int(checkpoint["input_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer = checkpoint["normalizer"]

    all_errors = []
    distance_stats = {}
    yaw_stats = {}

    visibility_total = 0
    visibility_correct = 0

    rel_z_values = []
    rel_yaw_values = []

    err_center_x_values = []
    err_bottom_y_values = []
    err_width_values = []
    err_height_values = []
    iou_values = []

    visible_rows = []

    with torch.no_grad():
        for row in rows:
            x = build_input_vector(row)
            x_norm = normalize_input(x, normalizer, device)

            pred_bbox_norm, pred_visible_logit = model(x_norm)

            pred_bbox_norm = pred_bbox_norm.squeeze(0).cpu().numpy()
            pred_visible_prob = torch.sigmoid(pred_visible_logit).item()

            pred = pred_to_pixels(pred_bbox_norm, pred_visible_prob, row)
            gt = target_to_pixels(row)

            gt_visible = int(gt["visible"])
            pred_visible = int(pred["visible"])

            visibility_total += 1
            if gt_visible == pred_visible:
                visibility_correct += 1

            if gt_visible != 1:
                continue

            err_center_x = pred["center_x"] - gt["center_x"]
            err_bottom_y = pred["bottom_y"] - gt["bottom_y"]
            err_width = pred["box_width"] - gt["box_width"]
            err_height = pred["box_height"] - gt["box_height"]
            iou = compute_iou(gt, pred)

            err_row = [
                err_center_x,
                err_bottom_y,
                err_width,
                err_height,
                iou,
            ]

            all_errors.append(err_row)

            rel = row["relative_state"]
            rel_z = float(rel["rel_z"])
            rel_yaw = float(rel["rel_yaw"])

            add_bucket_stats(distance_stats, distance_bucket(rel_z), err_row)
            add_bucket_stats(yaw_stats, yaw_bucket(rel_yaw), err_row)

            rel_z_values.append(rel_z)
            rel_yaw_values.append(rel_yaw)

            err_center_x_values.append(abs(err_center_x))
            err_bottom_y_values.append(abs(err_bottom_y))
            err_width_values.append(abs(err_width))
            err_height_values.append(abs(err_height))
            iou_values.append(iou)

            visible_rows.append(
                {
                    "row": row,
                    "gt": gt,
                    "pred": pred,
                    "iou": iou,
                    "abs_error_sum": (
                        abs(err_center_x)
                        + abs(err_bottom_y)
                        + abs(err_width)
                        + abs(err_height)
                    ),
                }
            )

    overall = summarize_errors(all_errors)
    overall["visibility_accuracy"] = float(
        visibility_correct / max(visibility_total, 1)
    )

    distance_summary = {
        k: summarize_errors(v)
        for k, v in sorted(distance_stats.items())
    }

    yaw_summary = {
        k: summarize_errors(v)
        for k, v in sorted(yaw_stats.items())
    }

    summary = {
        "labels": args.labels,
        "checkpoint": args.checkpoint,
        "num_rows": len(rows),
        "overall": overall,
        "distance_buckets": distance_summary,
        "yaw_buckets": yaw_summary,
    }

    summary_path = output_dir / "evaluation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("[RESULT] Overall")
    print(json.dumps(overall, indent=2))
    print()
    print(f"[DONE] Summary saved to: {summary_path}")

    # Plots
    if len(all_errors) > 0:
        save_scatter_plot(
            rel_z_values,
            err_center_x_values,
            "Distance rel_z (m)",
            "|center_x error| (px)",
            "Center-x error vs distance",
            output_dir / "center_x_error_vs_distance.png",
        )

        save_scatter_plot(
            rel_z_values,
            err_bottom_y_values,
            "Distance rel_z (m)",
            "|bottom_y error| (px)",
            "Bottom-y error vs distance",
            output_dir / "bottom_y_error_vs_distance.png",
        )

        save_scatter_plot(
            rel_z_values,
            err_width_values,
            "Distance rel_z (m)",
            "|box_width error| (px)",
            "Box-width error vs distance",
            output_dir / "box_width_error_vs_distance.png",
        )

        save_scatter_plot(
            rel_z_values,
            err_height_values,
            "Distance rel_z (m)",
            "|box_height error| (px)",
            "Box-height error vs distance",
            output_dir / "box_height_error_vs_distance.png",
        )

        save_scatter_plot(
            rel_yaw_values,
            err_width_values,
            "Relative yaw (deg)",
            "|box_width error| (px)",
            "Box-width error vs relative yaw",
            output_dir / "box_width_error_vs_yaw.png",
        )

        save_error_hist(
            iou_values,
            "IoU",
            "Predicted box IoU distribution",
            output_dir / "iou_histogram.png",
        )

    # Save best/mid/worst visualizations.
    visible_rows = sorted(visible_rows, key=lambda x: x["abs_error_sum"])

    selected = []

    n_vis = min(args.num_visuals, len(visible_rows))

    if n_vis > 0:
        # Some best examples.
        selected.extend(visible_rows[: max(1, n_vis // 3)])

        # Some middle examples.
        mid_start = max(0, len(visible_rows) // 2 - n_vis // 6)
        selected.extend(visible_rows[mid_start: mid_start + max(1, n_vis // 3)])

        # Some worst examples.
        selected.extend(visible_rows[-max(1, n_vis // 3):])

        selected = selected[:n_vis]

        for i, item in enumerate(selected):
            out_path = visual_dir / f"eval_box_{i:04d}_iou_{item['iou']:.3f}.png"
            save_box_visualization(
                item["row"],
                item["gt"],
                item["pred"],
                out_path,
            )

    print(f"[DONE] Plots saved to: {output_dir}")
    print(f"[DONE] Box visualizations saved to: {visual_dir}")


if __name__ == "__main__":
    main()