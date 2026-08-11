import argparse
import json
import math
from pathlib import Path

import pandas as pd


def safe_float(x, default=math.nan):
    try:
        return float(x)
    except Exception:
        return default


def load_labels(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            rs = rec.get("relative_state", {})
            tgt = rec.get("target", {})

            rows.append({
                "sample_id": int(rec.get("sample_id", -1)),
                "rel_x": safe_float(rs.get("rel_x")),
                "rel_z": safe_float(rs.get("rel_z")),
                "rel_yaw": safe_float(rs.get("rel_yaw")),
                "visible": int(tgt.get("visible", 0)),
                "center_x": safe_float(tgt.get("center_x")),
                "bottom_y": safe_float(tgt.get("bottom_y")),
                "box_width": safe_float(tgt.get("box_width")),
                "box_height": safe_float(tgt.get("box_height")),
                "x_min": safe_float(tgt.get("x_min")),
                "y_min": safe_float(tgt.get("y_min")),
                "x_max": safe_float(tgt.get("x_max")),
                "y_max": safe_float(tgt.get("y_max")),
                "unclipped_x_min": safe_float(tgt.get("unclipped_x_min")),
                "unclipped_y_min": safe_float(tgt.get("unclipped_y_min")),
                "unclipped_x_max": safe_float(tgt.get("unclipped_x_max")),
                "unclipped_y_max": safe_float(tgt.get("unclipped_y_max")),
            })

    return pd.DataFrame(rows)


def add_bins(df):
    bins = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 30),
        (30, 50),
        (50, 80),
        (80, 101),
    ]

    labels = []
    for z in df["rel_z"].values:
        label = "unknown"
        for a, b in bins:
            if a <= z < b:
                label = f"{a}-{b}"
                break
        labels.append(label)

    df["z_bin"] = labels
    return df


def summarize(df):
    total = len(df)
    visible = int(df["visible"].sum())
    invisible = total - visible

    print("\n================ v2 lookup dataset validation ================")
    print(f"Total rows:      {total}")
    print(f"Visible rows:    {visible}")
    print(f"Invisible rows:  {invisible}")
    print(f"Visible ratio:   {visible / total:.4f}" if total else "Visible ratio: nan")

    print("\nGrid:")
    print(f"rel_x count:     {df['rel_x'].nunique()}  range: {df['rel_x'].min()} to {df['rel_x'].max()}")
    print(f"rel_z count:     {df['rel_z'].nunique()}  range: {df['rel_z'].min()} to {df['rel_z'].max()}")
    print(f"rel_yaw count:   {df['rel_yaw'].nunique()}  range: {df['rel_yaw'].min()} to {df['rel_yaw'].max()}")

    expected = df["rel_x"].nunique() * df["rel_z"].nunique() * df["rel_yaw"].nunique()
    print(f"Expected grid:   {expected}")
    print(f"Missing rows:    {expected - total}")

    print("\nVisible counts by distance bin:")
    bin_stats = (
        df.groupby("z_bin")
        .agg(
            rows=("visible", "count"),
            visible=("visible", "sum"),
        )
        .reset_index()
    )
    bin_stats["visible_ratio"] = bin_stats["visible"] / bin_stats["rows"]

    for _, r in bin_stats.iterrows():
        print(
            f"{str(r['z_bin']).rjust(8)} m | "
            f"rows={int(r['rows']):7d} | "
            f"visible={int(r['visible']):7d} | "
            f"ratio={float(r['visible_ratio']):.4f}"
        )

    v = df[df["visible"] == 1].copy()

    print("\nVisible bbox numeric ranges:")
    for col in ["center_x", "bottom_y", "box_width", "box_height"]:
        print(
            f"{col.rjust(12)} | "
            f"min={v[col].min():9.3f} "
            f"mean={v[col].mean():9.3f} "
            f"max={v[col].max():9.3f}"
        )

    suspicious = v[
        (v["box_width"] <= 0)
        | (v["box_height"] <= 0)
        | (v["bottom_y"] < 0)
        | (v["center_x"] < -5000)
        | (v["center_x"] > 6000)
        | (v["box_width"] > 5000)
        | (v["box_height"] > 5000)
    ]

    print(f"\nSuspicious visible rows: {len(suspicious)}")

    if len(suspicious) > 0:
        print(suspicious.head(20).to_string(index=False))

    print("===============================================================\n")


def save_reports(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "validation_all_rows.csv", index=False)

    visible = df[df["visible"] == 1].copy()
    visible.to_csv(out_dir / "validation_visible_rows.csv", index=False)

    bin_stats = (
        df.groupby("z_bin")
        .agg(
            rows=("visible", "count"),
            visible=("visible", "sum"),
            center_x_mean=("center_x", "mean"),
            bottom_y_mean=("bottom_y", "mean"),
            box_width_mean=("box_width", "mean"),
            box_height_mean=("box_height", "mean"),
        )
        .reset_index()
    )
    bin_stats["visible_ratio"] = bin_stats["visible"] / bin_stats["rows"]
    bin_stats.to_csv(out_dir / "validation_distance_bins.csv", index=False)

    # Useful curve for the central lane and frontal yaw.
    central = visible[
        (visible["rel_x"].abs() < 1e-6)
        & (visible["rel_yaw"].abs() < 1e-6)
    ].sort_values("rel_z")

    central.to_csv(out_dir / "curve_centerlane_yaw0.csv", index=False)

    print("[SAVED]", out_dir / "validation_all_rows.csv")
    print("[SAVED]", out_dir / "validation_visible_rows.csv")
    print("[SAVED]", out_dir / "validation_distance_bins.csv")
    print("[SAVED]", out_dir / "curve_centerlane_yaw0.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    df = load_labels(args.labels)
    df = add_bins(df)

    summarize(df)
    save_reports(df, args.out_dir)


if __name__ == "__main__":
    main()