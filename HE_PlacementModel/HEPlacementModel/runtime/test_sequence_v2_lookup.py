import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from he_lookup_placement_v2 import HELookupPlacementV2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--labels", type=str, default="dataset/v2_lookup_full/labels.jsonl")
    parser.add_argument("--metadata", type=str, default="dataset/v2_lookup_full/metadata.json")
    parser.add_argument("--output-dir", type=str, default="outputs/v2_lookup_sequence_test")

    parser.add_argument("--rel-x", type=float, default=0.0)
    parser.add_argument("--rel-yaw", type=float, default=180.0)

    parser.add_argument("--z-start", type=float, default=60.0)
    parser.add_argument("--z-end", type=float, default=10.0)
    parser.add_argument("--num-frames", type=int, default=100)

    parser.add_argument("--mode", type=str, default="clamp", choices=["clamp", "edge", "invisible"])
    parser.add_argument("--method", type=str, default="interpolated", choices=["nearest", "interpolated", "trilinear"])

    return parser.parse_args()


def save_line_plot(xs, ys, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, marker="o", markersize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_overlay_plot(rows, width, height, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    step = max(1, len(rows) // 20)

    for row in rows[::step]:
        x = row["x"]
        y = row["y"]
        w = row["w"]
        h = row["h"]
        rel_z = row["rel_z"]

        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        ax.text(
            x,
            y - 4,
            f"{rel_z:.0f}m",
            fontsize=8,
        )

    ax.set_title("V2 lookup predicted placement sequence")
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading V2 lookup model...")
    model = HELookupPlacementV2(
        labels_path=args.labels,
        metadata_path=args.metadata,
        safe_distance_margin_m=5.0,
        default_mode=args.mode,
    )
    print("[INFO] Loaded.")
    print(f"[INFO] safe_min_rel_z = {model.safe_min_rel_z}")

    rel_z_values = np.linspace(
        float(args.z_start),
        float(args.z_end),
        int(args.num_frames),
    )

    rows = []

    for frame_id, rel_z in enumerate(rel_z_values):
        pred = model.predict(
            rel_x=args.rel_x,
            rel_z=float(rel_z),
            rel_yaw=args.rel_yaw,
            mode=args.mode,
            method=args.method,
        )

        row = {
            "frame_id": frame_id,
            "rel_x": float(args.rel_x),
            "rel_z": float(rel_z),
            "rel_yaw": float(args.rel_yaw),
            "center_x": pred["center_x"],
            "bottom_y": pred["bottom_y"],
            "box_width": pred["box_width"],
            "box_height": pred["box_height"],
            "visible": pred["visible"],
            "x": pred["x"],
            "y": pred["y"],
            "w": pred["w"],
            "h": pred["h"],
            "rel_z_used": pred["query"]["rel_z_used"],
            "close_range": pred["query"]["close_range"],
            "method": pred["lookup"]["method"],
        }

        rows.append(row)

    csv_path = output_dir / "sequence_predictions.csv"
    json_path = output_dir / "sequence_predictions.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    rel_z = [r["rel_z"] for r in rows]
    bottom_y = [r["bottom_y"] for r in rows]
    box_width = [r["box_width"] for r in rows]
    box_height = [r["box_height"] for r in rows]
    center_x = [r["center_x"] for r in rows]

    save_line_plot(
        rel_z,
        bottom_y,
        "Distance rel_z (m)",
        "Predicted bottom_y (px)",
        "V2 bottom_y vs distance",
        output_dir / "bottom_y_vs_distance.png",
    )

    save_line_plot(
        rel_z,
        box_width,
        "Distance rel_z (m)",
        "Predicted box width (px)",
        "V2 box width vs distance",
        output_dir / "box_width_vs_distance.png",
    )

    save_line_plot(
        rel_z,
        box_height,
        "Distance rel_z (m)",
        "Predicted box height (px)",
        "V2 box height vs distance",
        output_dir / "box_height_vs_distance.png",
    )

    save_line_plot(
        rel_z,
        center_x,
        "Distance rel_z (m)",
        "Predicted center_x (px)",
        "V2 center_x vs distance",
        output_dir / "center_x_vs_distance.png",
    )

    save_overlay_plot(
        rows,
        width=model.image_width,
        height=model.image_height,
        out_path=output_dir / "image_plane_overlay.png",
    )

    print("[DONE] V2 sequence test complete.")
    print(f"[DONE] CSV  : {csv_path}")
    print(f"[DONE] JSON : {json_path}")
    print(f"[DONE] Plots: {output_dir}")

    print()
    print("[FIRST FRAME]")
    print(json.dumps(rows[0], indent=2))

    print()
    print("[LAST FRAME]")
    print(json.dumps(rows[-1], indent=2))


if __name__ == "__main__":
    main()