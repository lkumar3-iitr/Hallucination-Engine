import argparse
import json
import math
from pathlib import Path


def load_real_bbox_jsonl(path):
    rows = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            frame_idx = int(rec["recorded_frame_idx"])
            bbox = rec.get("bbox", {})

            rows[frame_idx] = {
                "frame_idx": frame_idx,
                "visible": bool(bbox.get("visible", False)),
                "cx": float(bbox.get("cx", math.nan)),
                "bottom_y": float(bbox.get("bottom_y", math.nan)),
                "box_width": float(bbox.get("box_width", math.nan)),
                "box_height": float(bbox.get("box_height", math.nan)),
                "depth_m": float(bbox.get("depth_m", math.nan)),
                "raw": rec,
            }

    return rows


def load_he_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rows = {}

    for frame in meta.get("frames", []):
        frame_idx = int(frame["frame_idx"])
        adversaries = frame.get("adversaries", [])

        if not adversaries:
            rows[frame_idx] = {
                "frame_idx": frame_idx,
                "visible": False,
                "rendered": False,
                "cx": math.nan,
                "bottom_y": math.nan,
                "box_width": math.nan,
                "box_height": math.nan,
                "z_m": math.nan,
                "raw": frame,
            }
            continue

        adv = adversaries[0]
        box = adv.get("box", {})

        rows[frame_idx] = {
            "frame_idx": frame_idx,
            "visible": bool(box.get("visible", False)),
            "rendered": bool(adv.get("rendered", False)),
            "cx": float(box.get("cx", math.nan)),
            "bottom_y": float(box.get("bottom_y", math.nan)),
            "box_width": float(box.get("box_width", math.nan)),
            "box_height": float(box.get("box_height", math.nan)),
            "z_m": float(box.get("z_m", math.nan)),
            "raw": adv,
        }

    return rows


def is_finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def mean(values):
    values = [float(v) for v in values if is_finite(v)]
    if not values:
        return math.nan
    return sum(values) / len(values)


def mae(values):
    values = [abs(float(v)) for v in values if is_finite(v)]
    if not values:
        return math.nan
    return sum(values) / len(values)


def fmt(x):
    if not is_finite(x):
        return "nan"
    return f"{float(x):.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-bbox", required=True)
    parser.add_argument("--he-metadata", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--min-real-depth", type=float, default=0.0)
    args = parser.parse_args()

    real = load_real_bbox_jsonl(args.real_bbox)
    he = load_he_metadata(args.he_metadata)

    common_frames = sorted(set(real.keys()) & set(he.keys()))

    rows = []

    for frame_idx in common_frames:
        r = real[frame_idx]
        h = he[frame_idx]

        real_visible = bool(r["visible"])
        he_visible = bool(h["visible"]) and bool(h.get("rendered", False))

        row = {
            "frame_idx": frame_idx,
            "real_visible": real_visible,
            "he_visible": he_visible,
            "visibility_match": real_visible == he_visible,

            "real_cx": r["cx"],
            "he_cx": h["cx"],
            "err_cx": h["cx"] - r["cx"] if real_visible and he_visible else math.nan,

            "real_bottom_y": r["bottom_y"],
            "he_bottom_y": h["bottom_y"],
            "err_bottom_y": h["bottom_y"] - r["bottom_y"] if real_visible and he_visible else math.nan,

            "real_box_width": r["box_width"],
            "he_box_width": h["box_width"],
            "err_box_width": h["box_width"] - r["box_width"] if real_visible and he_visible else math.nan,

            "real_box_height": r["box_height"],
            "he_box_height": h["box_height"],
            "err_box_height": h["box_height"] - r["box_height"] if real_visible and he_visible else math.nan,

            "real_depth_m": r["depth_m"],
            "he_z_m": h["z_m"],
        }

        rows.append(row)

    both_visible = [
        r for r in rows
        if r["real_visible"]
        and r["he_visible"]
        and math.isfinite(float(r["real_depth_m"]))
        and float(r["real_depth_m"]) >= float(args.min_real_depth)
    ]
    visibility_matches = [r["visibility_match"] for r in rows]

    summary = {
        "num_common_frames": len(common_frames),
        "num_both_visible": len(both_visible),
        "visibility_agreement": (
            sum(1 for v in visibility_matches if v) / len(visibility_matches)
            if visibility_matches else math.nan
        ),

        "mae_cx_px": mae([r["err_cx"] for r in both_visible]),
        "mae_bottom_y_px": mae([r["err_bottom_y"] for r in both_visible]),
        "mae_box_width_px": mae([r["err_box_width"] for r in both_visible]),
        "mae_box_height_px": mae([r["err_box_height"] for r in both_visible]),

        "mean_signed_cx_px": mean([r["err_cx"] for r in both_visible]),
        "mean_signed_bottom_y_px": mean([r["err_bottom_y"] for r in both_visible]),
        "mean_signed_box_width_px": mean([r["err_box_width"] for r in both_visible]),
        "mean_signed_box_height_px": mean([r["err_box_height"] for r in both_visible]),
    }

    print("\n================ Real CARLA vs HE bbox comparison ================")
    print("Frames compared:       ", summary["num_common_frames"])
    print("Both visible frames:   ", summary["num_both_visible"])
    print("Visibility agreement:  ", fmt(summary["visibility_agreement"]))
    print()
    print("MAE cx px:             ", fmt(summary["mae_cx_px"]))
    print("MAE bottom_y px:       ", fmt(summary["mae_bottom_y_px"]))
    print("MAE width px:          ", fmt(summary["mae_box_width_px"]))
    print("MAE height px:         ", fmt(summary["mae_box_height_px"]))
    print()
    print("Signed mean cx px:     ", fmt(summary["mean_signed_cx_px"]))
    print("Signed mean bottom_y:  ", fmt(summary["mean_signed_bottom_y_px"]))
    print("Signed mean width:     ", fmt(summary["mean_signed_box_width_px"]))
    print("Signed mean height:    ", fmt(summary["mean_signed_box_height_px"]))
    print("===================================================================\n")

    print("Sample rows:")
    print("frame | real_vis he_vis | real_cx he_cx err_cx | real_by he_by err_by | real_w he_w err_w | real_h he_h err_h")
    print("-" * 140)

    sample_source = both_visible if both_visible else rows

    for r in sample_source[::max(1, len(sample_source) // 12)]:
        print(
            f"{r['frame_idx']:5d} | "
            f"{int(r['real_visible'])}        {int(r['he_visible'])}     | "
            f"{fmt(r['real_cx']):>7} {fmt(r['he_cx']):>7} {fmt(r['err_cx']):>7} | "
            f"{fmt(r['real_bottom_y']):>7} {fmt(r['he_bottom_y']):>7} {fmt(r['err_bottom_y']):>7} | "
            f"{fmt(r['real_box_width']):>7} {fmt(r['he_box_width']):>7} {fmt(r['err_box_width']):>7} | "
            f"{fmt(r['real_box_height']):>7} {fmt(r['he_box_height']):>7} {fmt(r['err_box_height']):>7}"
        )

    if args.out_json:
        out = {
            "summary": summary,
            "rows": rows,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("[SAVED]", args.out_json)

    if args.out_csv:
        import csv

        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("[SAVED]", args.out_csv)


if __name__ == "__main__":
    main()