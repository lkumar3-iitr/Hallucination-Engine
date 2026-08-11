import argparse
import json
import math
from pathlib import Path

import numpy as np


def safe_float(x, default=math.nan):
    try:
        return float(x)
    except Exception:
        return default


def load_rows(labels_path):
    rows = []

    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            # Use the requested grid state for table indexing.
            # The measured camera-relative state has tiny CARLA/floating-point deviations
            # and should be kept only for debugging, not for defining grid axes.
            rs_measured = rec.get("relative_state", {})
            rs_requested = rec.get(
                "requested_camera_relative_state",
                rec.get("requested_ego_relative_state", rs_measured)
            )

            tgt = rec["target"]

            rel_x = round(safe_float(rs_requested["rel_x"]), 6)
            rel_z = round(safe_float(rs_requested["rel_z"]), 6)
            rel_yaw = safe_float(rs_requested["rel_yaw"])

            # Store yaw in 0..359 convention for the NPZ adapter.
            rel_yaw = rel_yaw % 360.0
            rel_yaw = round(rel_yaw, 6)

            rows.append({
                "rel_x": rel_x,
                "rel_z": rel_z,
                "rel_yaw": rel_yaw,
                "visible": int(tgt["visible"]),
                "center_x": safe_float(tgt["center_x"]),
                "bottom_y": safe_float(tgt["bottom_y"]),
                "box_width": safe_float(tgt["box_width"]),
                "box_height": safe_float(tgt["box_height"]),
                "x_min": safe_float(tgt["x_min"]),
                "y_min": safe_float(tgt["y_min"]),
                "x_max": safe_float(tgt["x_max"]),
                "y_max": safe_float(tgt["y_max"]),
                "unclipped_x_min": safe_float(tgt.get("unclipped_x_min")),
                "unclipped_y_min": safe_float(tgt.get("unclipped_y_min")),
                "unclipped_x_max": safe_float(tgt.get("unclipped_x_max")),
                "unclipped_y_max": safe_float(tgt.get("unclipped_y_max")),
            })

    return rows


def nearest_index_map(values):
    return {round(float(v), 6): i for i, v in enumerate(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="heplacement_v2_lookup_full_0_100_z05_yaw1")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    metadata_path = Path(args.metadata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading labels:", labels_path)
    rows = load_rows(labels_path)
    print("[INFO] Rows loaded:", len(rows))

    with open(metadata_path, "r", encoding="utf-8") as f:
        source_metadata = json.load(f)

    rel_x_values = sorted({r["rel_x"] for r in rows})
    rel_z_values = sorted({r["rel_z"] for r in rows})
    yaw_values = sorted({r["rel_yaw"] for r in rows})

    nx = len(rel_x_values)
    nz = len(rel_z_values)
    nyaw = len(yaw_values)

    expected = nx * nz * nyaw

    print("[INFO] Grid:")
    print("  rel_x:", nx, rel_x_values[0], "to", rel_x_values[-1])
    print("  rel_z:", nz, rel_z_values[0], "to", rel_z_values[-1])
    print("  yaw:  ", nyaw, yaw_values[0], "to", yaw_values[-1])
    print("  expected:", expected)

    if expected != len(rows):
        print("[WARN] Row count does not match full grid.")
        print("[WARN] expected:", expected, "actual:", len(rows))

    x_to_i = nearest_index_map(rel_x_values)
    z_to_i = nearest_index_map(rel_z_values)
    yaw_to_i = nearest_index_map(yaw_values)

    # Layout: [z_index, x_index, yaw_index]
    visible = np.zeros((nz, nx, nyaw), dtype=np.uint8)

    center_x = np.full((nz, nx, nyaw), np.nan, dtype=np.float32)
    bottom_y = np.full((nz, nx, nyaw), np.nan, dtype=np.float32)
    box_width = np.full((nz, nx, nyaw), np.nan, dtype=np.float32)
    box_height = np.full((nz, nx, nyaw), np.nan, dtype=np.float32)

    x_min = np.full((nz, nx, nyaw), np.nan, dtype=np.float32)
    y_min = np.full((nz, nx, nyaw), np.nan, dtype=np.float32)
    x_max = np.full((nz, nx, nyaw), np.nan, dtype=np.float32)
    y_max = np.full((nz, nx, nyaw), np.nan, dtype=np.float32)

    written = 0

    for r in rows:
        iz = z_to_i[r["rel_z"]]
        ix = x_to_i[r["rel_x"]]
        iy = yaw_to_i[r["rel_yaw"]]

        visible[iz, ix, iy] = int(r["visible"])

        center_x[iz, ix, iy] = float(r["center_x"])
        bottom_y[iz, ix, iy] = float(r["bottom_y"])
        box_width[iz, ix, iy] = float(r["box_width"])
        box_height[iz, ix, iy] = float(r["box_height"])

        x_min[iz, ix, iy] = float(r["x_min"])
        y_min[iz, ix, iy] = float(r["y_min"])
        x_max[iz, ix, iy] = float(r["x_max"])
        y_max[iz, ix, iy] = float(r["y_max"])

        written += 1

    npz_path = out_dir / f"{args.name}.npz"
    meta_out_path = out_dir / f"{args.name}_metadata.json"

    print("[INFO] Saving:", npz_path)

    np.savez_compressed(
        npz_path,
        rel_x_values=np.array(rel_x_values, dtype=np.float32),
        rel_z_values=np.array(rel_z_values, dtype=np.float32),
        yaw_values=np.array(yaw_values, dtype=np.float32),
        visible=visible,
        center_x=center_x,
        bottom_y=bottom_y,
        box_width=box_width,
        box_height=box_height,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )

    visible_count = int(visible.sum())
    total_count = int(visible.size)

    meta_out = {
        "name": args.name,
        "format": "npz_grid_v1",
        "source_labels": str(labels_path),
        "source_metadata": str(metadata_path),
        "npz_path": str(npz_path),
        "grid_layout": "[rel_z_index, rel_x_index, yaw_index]",
        "fields": [
            "visible",
            "center_x",
            "bottom_y",
            "box_width",
            "box_height",
            "x_min",
            "y_min",
            "x_max",
            "y_max"
        ],
        "rel_x_values": rel_x_values,
        "rel_z_values": rel_z_values,
        "yaw_values": yaw_values,
        "num_rel_x": nx,
        "num_rel_z": nz,
        "num_yaw": nyaw,
        "total_grid_points": total_count,
        "visible_count": visible_count,
        "visible_ratio": visible_count / total_count if total_count else None,
        "camera": source_metadata.get("camera", {}),
        "ego_locked_transform": source_metadata.get("ego_locked_transform", {}),
        "source_grid": source_metadata.get("grid", {}),
        "near_field_notes": {
            "0_1_m": "mostly invalid or collision/too-close rendering",
            "1_2_m": "partial visibility; handle separately",
            "2_5_m": "potential near-field lookup range",
            "5_100_m": "normal lookup range"
        }
    }

    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print("[DONE] Runtime lookup saved.")
    print("[DONE] NPZ:", npz_path)
    print("[DONE] metadata:", meta_out_path)
    print("[DONE] visible:", visible_count, "/", total_count)


if __name__ == "__main__":
    main()