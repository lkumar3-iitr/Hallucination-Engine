"""
compare_real_vs_he_domain_v2.py

Placement-operating-domain diagnostic for the multi-actor
CARLA-vs-HE comparison.

This script DOES NOT modify the HE renderer.

It uses:
    compare_real_vs_he_bbox_v2.py

and adds classification based on the HE current-camera-relative state:

    rel_x = adv["state"]["x_m"]
    rel_z = adv["state"]["z_m"]

Default frozen HE placement domain:

    -4 <= rel_x <= +4 m
     0 <= rel_z <= 100 m

Yaw is not restricted because the frozen lookup contains the full
360-degree yaw range.

Metrics are reported separately for:
    - in-domain placement states
    - out-of-domain placement states
"""

import argparse
import math
from collections import Counter

from compare_real_vs_he_bbox_v2 import (
    build_comparison_rows,
    fmt,
    is_finite,
    load_he_metadata_v2,
    load_real_ground_truth_v2,
    mae,
    mean,
    safe_float,
)


# ============================================================
# Domain classification
# ============================================================

def classify_domain(
    x_m,
    z_m,
    x_min,
    x_max,
    z_min,
    z_max,
    epsilon=1e-5,
):
    """
    Return:

        (in_domain, reason)

    A small numerical tolerance is used at lookup boundaries because
    resolved/interpolated trajectories may contain values such as
    4.000000000000001 for a mathematically exact 4.0 m state.
    """

    if (
        not is_finite(x_m)
        or not is_finite(z_m)
    ):
        return None, "unknown_state"

    reasons = []

    if x_m < (x_min - epsilon):
        reasons.append("x_below_min")

    if x_m > (x_max + epsilon):
        reasons.append("x_above_max")

    if z_m < (z_min - epsilon):
        reasons.append("z_below_min")

    if z_m > (z_max + epsilon):
        reasons.append("z_above_max")

    if reasons:
        return False, "+".join(reasons)

    return True, "in_domain"

def annotate_domain(
    rows,
    he_frames,
    x_min,
    x_max,
    z_min,
    z_max,
):
    """
    Add current-camera-relative HE state and placement-domain
    classification to every comparison row.
    """

    for row in rows:

        frame_idx = row["frame_idx"]
        actor_id = row["actor_id"]

        he_frame = he_frames.get(
            frame_idx,
            {},
        )

        he_actor = (
            he_frame
            .get("actors", {})
            .get(actor_id)
        )

        if he_actor is None:

            row["he_state_x_m"] = math.nan
            row["he_state_z_m"] = math.nan
            row["he_state_yaw_deg"] = math.nan

            row["he_in_domain"] = None
            row["he_domain_reason"] = "actor_not_present"

            continue

        raw_adv = (
            he_actor.get("raw")
            or {}
        )

        state = (
            raw_adv.get("state")
            or {}
        )

        x_m = safe_float(
            state.get("x_m")
        )

        z_m = safe_float(
            state.get("z_m")
        )

        yaw_deg = safe_float(
            state.get("yaw_deg")
        )

        in_domain, reason = (
            classify_domain(
                x_m=x_m,
                z_m=z_m,

                x_min=x_min,
                x_max=x_max,

                z_min=z_min,
                z_max=z_max,
            )
        )

        row["he_state_x_m"] = x_m
        row["he_state_z_m"] = z_m
        row["he_state_yaw_deg"] = yaw_deg

        row["he_in_domain"] = in_domain
        row["he_domain_reason"] = reason


# ============================================================
# Metric filtering
# ============================================================

def passes_depth_filter(
    row,
    min_real_depth,
):
    if not row["both_visible"]:
        return False

    if min_real_depth <= 0.0:
        return True

    depth = row[
        "real_filter_depth_m"
    ]

    if not is_finite(depth):
        return False

    return (
        float(depth)
        >=
        float(min_real_depth)
    )


def geometry_summary(
    rows,
):
    """
    Calculate the same bbox metrics as the v2 comparator.
    """

    return {
        "count":
            len(rows),

        "mae_cx_px":
            mae([
                r["err_cx"]
                for r in rows
            ]),

        "mae_bottom_y_px":
            mae([
                r["err_bottom_y"]
                for r in rows
            ]),

        "mae_box_width_px":
            mae([
                r["err_box_width"]
                for r in rows
            ]),

        "mae_box_height_px":
            mae([
                r["err_box_height"]
                for r in rows
            ]),

        "mean_signed_cx_px":
            mean([
                r["err_cx"]
                for r in rows
            ]),

        "mean_signed_bottom_y_px":
            mean([
                r["err_bottom_y"]
                for r in rows
            ]),

        "mean_signed_box_width_px":
            mean([
                r["err_box_width"]
                for r in rows
            ]),

        "mean_signed_box_height_px":
            mean([
                r["err_box_height"]
                for r in rows
            ]),
    }


# ============================================================
# Reporting
# ============================================================

def print_geometry(
    name,
    summary,
):
    print()
    print(name)
    print("-" * 64)

    print(
        "BBox frames:             ",
        summary["count"],
    )

    print(
        "MAE cx px:              ",
        fmt(summary["mae_cx_px"]),
    )

    print(
        "MAE bottom_y px:        ",
        fmt(summary["mae_bottom_y_px"]),
    )

    print(
        "MAE width px:           ",
        fmt(summary["mae_box_width_px"]),
    )

    print(
        "MAE height px:          ",
        fmt(summary["mae_box_height_px"]),
    )

    print()

    print(
        "Signed mean cx px:      ",
        fmt(summary["mean_signed_cx_px"]),
    )

    print(
        "Signed mean bottom_y:   ",
        fmt(summary["mean_signed_bottom_y_px"]),
    )

    print(
        "Signed mean width:      ",
        fmt(summary["mean_signed_box_width_px"]),
    )

    print(
        "Signed mean height:     ",
        fmt(summary["mean_signed_box_height_px"]),
    )


def report_actor(
    actor_id,
    rows,
    min_real_depth,
):
    actor_rows = [
        r
        for r in rows
        if r["actor_id"] == actor_id
    ]

    lifecycle_rows = [
        r
        for r in actor_rows
        if r["both_present"]
    ]

    in_domain_lifecycle = [
        r
        for r in lifecycle_rows
        if r["he_in_domain"] is True
    ]

    ood_lifecycle = [
        r
        for r in lifecycle_rows
        if r["he_in_domain"] is False
    ]

    unknown_lifecycle = [
        r
        for r in lifecycle_rows
        if r["he_in_domain"] is None
    ]

    eligible = [
        r
        for r in lifecycle_rows
        if passes_depth_filter(
            r,
            min_real_depth,
        )
    ]

    in_domain_bbox = [
        r
        for r in eligible
        if r["he_in_domain"] is True
    ]

    ood_bbox = [
        r
        for r in eligible
        if r["he_in_domain"] is False
    ]

    reason_counts = Counter(
        r["he_domain_reason"]
        for r in ood_lifecycle
    )

    print()
    print("=" * 76)
    print(
        f"HE PLACEMENT DOMAIN [{actor_id}]"
    )
    print("=" * 76)

    print(
        "Lifecycle frames:          ",
        len(lifecycle_rows),
    )

    print(
        "In-domain frames:          ",
        len(in_domain_lifecycle),
    )

    print(
        "Out-of-domain frames:      ",
        len(ood_lifecycle),
    )

    print(
        "Unknown-domain frames:     ",
        len(unknown_lifecycle),
    )

    if lifecycle_rows:

        print(
            "In-domain fraction:       ",
            fmt(
                len(in_domain_lifecycle)
                /
                len(lifecycle_rows)
            ),
        )

    print()

    print(
        "Min real depth filter:     ",
        fmt(min_real_depth),
        "m",
    )

    if reason_counts:

        print()
        print("OOD reasons:")

        for reason, count in sorted(
            reason_counts.items()
        ):
            print(
                f"  {reason:24s} {count}"
            )

    in_summary = geometry_summary(
        in_domain_bbox
    )

    out_summary = geometry_summary(
        ood_bbox
    )

    print_geometry(
        "IN-DOMAIN GEOMETRY",
        in_summary,
    )

    print_geometry(
        "OUT-OF-DOMAIN GEOMETRY",
        out_summary,
    )

    # --------------------------------------------------------
    # Show domain transitions.
    # --------------------------------------------------------

    print()
    print("Domain transition samples:")
    print(
        "frame | x_m      z_m      domain | "
        "real_cx   he_cx    err_cx"
    )
    print("-" * 76)

    previous = object()
    transition_rows = []

    for row in lifecycle_rows:

        current = row[
            "he_in_domain"
        ]

        if current != previous:

            transition_rows.append(
                row
            )

            previous = current

    # Also include final lifecycle frame.
    if lifecycle_rows:

        if (
            lifecycle_rows[-1]
            not in transition_rows
        ):
            transition_rows.append(
                lifecycle_rows[-1]
            )

    for row in transition_rows:

        domain_text = {
            True: "IN ",
            False: "OOD",
            None: "UNK",
        }[
            row["he_in_domain"]
        ]

        print(
            f"{row['frame_idx']:5d} | "
            f"{fmt(row['he_state_x_m']):>7} "
            f"{fmt(row['he_state_z_m']):>7} "
            f"{domain_text:>6} | "
            f"{fmt(row['real_cx']):>8} "
            f"{fmt(row['he_cx']):>8} "
            f"{fmt(row['err_cx']):>8}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CARLA-vs-HE geometry separately inside "
            "and outside the frozen HE placement lookup domain."
        )
    )

    parser.add_argument(
        "--real-ground-truth",
        required=True,
    )

    parser.add_argument(
        "--he-metadata",
        required=True,
    )

    parser.add_argument(
        "--actor-id",
        default=None,
    )

    parser.add_argument(
        "--min-real-depth",
        type=float,
        default=0.0,
    )

    # Frozen placement lookup bounds.
    parser.add_argument(
        "--domain-x-min",
        type=float,
        default=-4.0,
    )

    parser.add_argument(
        "--domain-x-max",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--domain-z-min",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--domain-z-max",
        type=float,
        default=100.0,
    )

    args = parser.parse_args()

    real_frames = (
        load_real_ground_truth_v2(
            args.real_ground_truth
        )
    )

    he_frames = (
        load_he_metadata_v2(
            args.he_metadata
        )
    )

    rows, common_frames, actor_ids = (
        build_comparison_rows(
            real_frames,
            he_frames,
        )
    )

    annotate_domain(
        rows=rows,
        he_frames=he_frames,

        x_min=args.domain_x_min,
        x_max=args.domain_x_max,

        z_min=args.domain_z_min,
        z_max=args.domain_z_max,
    )

    if args.actor_id:

        if args.actor_id not in actor_ids:

            raise ValueError(
                f"Unknown actor: {args.actor_id}. "
                f"Available: {actor_ids}"
            )

        actor_ids = [
            args.actor_id
        ]

    print()
    print("=" * 76)
    print("HE PLACEMENT OPERATING DOMAIN")
    print("=" * 76)

    print(
        "Common frames: ",
        len(common_frames),
    )

    print(
        "Actors:        ",
        ", ".join(actor_ids),
    )

    print()

    print(
        "rel_x domain:  "
        f"[{args.domain_x_min}, "
        f"{args.domain_x_max}] m"
    )

    print(
        "rel_z domain:  "
        f"[{args.domain_z_min}, "
        f"{args.domain_z_max}] m"
    )

    print(
        "yaw domain:    full 360 degrees"
    )

    for actor_id in actor_ids:

        report_actor(
            actor_id=actor_id,
            rows=rows,
            min_real_depth=args.min_real_depth,
        )


if __name__ == "__main__":
    main()