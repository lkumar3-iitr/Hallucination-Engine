import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from he_placement_model import HEPlacementModel, build_camera_from_fov


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/v1_mlp_combined/heplacement_v1_mlp_best.pt",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/v1_sequence_test",
    )

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov", type=float, default=90.0)

    parser.add_argument("--rel-x", type=float, default=0.0)
    parser.add_argument("--rel-yaw", type=float, default=180.0)

    parser.add_argument("--z-start", type=float, default=60.0)
    parser.add_argument("--z-end", type=float, default=10.0)
    parser.add_argument("--num-frames", type=int, default=100)

    return parser.parse_args()


def placement_to_sprite_rect(placement):
    center_x = placement["center_x"]
    bottom_y = placement["bottom_y"]
    box_width = placement["box_width"]
    box_height = placement["box_height"]

    return {
        "sprite_x": center_x - box_width / 2.0,
        "sprite_y": bottom_y - box_height,
        "sprite_w": box_width,
        "sprite_h": box_height,
    }


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
    """
    Draw predicted boxes over a blank image plane.
    This helps verify that the box grows and moves downward as distance decreases.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Draw every few boxes to avoid clutter.
    step = max(1, len(rows) // 20)

    for i, row in enumerate(rows[::step]):
        x = row["sprite_x"]
        y = row["sprite_y"]
        w = row["sprite_w"]
        h = row["sprite_h"]
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

    ax.set_title("Predicted placement sequence on image plane")
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HEPlacementModel(args.checkpoint, device="cpu")

    camera = build_camera_from_fov(
        width=args.width,
        height=args.height,
        fov=args.fov,
    )

    rel_z_values = np.linspace(
        float(args.z_start),
        float(args.z_end),
        int(args.num_frames),
    )

    rows = []

    for frame_id, rel_z in enumerate(rel_z_values):
        relative_state = {
            "rel_x": float(args.rel_x),
            "rel_z": float(rel_z),
            "rel_yaw": float(args.rel_yaw),
        }

        placement = model.predict(
            relative_state=relative_state,
            camera=camera,
        )

        rect = placement_to_sprite_rect(placement)

        row = {
            "frame_id": frame_id,
            "rel_x": float(args.rel_x),
            "rel_z": float(rel_z),
            "rel_yaw": float(args.rel_yaw),
            "center_x": placement["center_x"],
            "bottom_y": placement["bottom_y"],
            "box_width": placement["box_width"],
            "box_height": placement["box_height"],
            "visible": placement["visible"],
            "visible_prob": placement["visible_prob"],
            "sprite_x": rect["sprite_x"],
            "sprite_y": rect["sprite_y"],
            "sprite_w": rect["sprite_w"],
            "sprite_h": rect["sprite_h"],
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
        "bottom_y vs distance",
        output_dir / "bottom_y_vs_distance.png",
    )

    save_line_plot(
        rel_z,
        box_width,
        "Distance rel_z (m)",
        "Predicted box width (px)",
        "box width vs distance",
        output_dir / "box_width_vs_distance.png",
    )

    save_line_plot(
        rel_z,
        box_height,
        "Distance rel_z (m)",
        "Predicted box height (px)",
        "box height vs distance",
        output_dir / "box_height_vs_distance.png",
    )

    save_line_plot(
        rel_z,
        center_x,
        "Distance rel_z (m)",
        "Predicted center_x (px)",
        "center_x vs distance",
        output_dir / "center_x_vs_distance.png",
    )

    save_overlay_plot(
        rows,
        width=args.width,
        height=args.height,
        out_path=output_dir / "image_plane_overlay.png",
    )

    print("[DONE] Sequence test complete.")
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