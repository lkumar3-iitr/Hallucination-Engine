from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# This script is used in unattended validation runs where no desktop display
# or Tcl/Tk installation is guaranteed to be available.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedScenarioV2,
)
from validate_resolved_clearance_v1 import (
    rectangle_clearance,
    rectangle_corners,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a ResolvedScenarioV2 BEV trajectory and physical "
            "ego/actor rectangle-clearance trace."
        )
    )
    parser.add_argument("resolved_json")
    parser.add_argument("--clearance-report", required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.resolved_json).open("r", encoding="utf-8") as handle:
        scenario = ResolvedScenarioV2.model_validate(json.load(handle))
    with Path(args.clearance_report).open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    ego_dimensions = report["ego_dimensions"]
    actor_dimensions = report["actor_dimensions"]
    ego = {
        frame.frame_idx: frame
        for frame in scenario.ego_frames
    }
    actor = {
        frame.frame_idx: frame
        for frame in scenario.actor_frames
        if frame.actor_id == args.actor_id
    }
    common = sorted(set(ego) & set(actor))

    clearances = []
    for frame_idx in common:
        ego_frame = ego[frame_idx]
        actor_frame = actor[frame_idx]
        ego_rectangle = rectangle_corners(
            ego_frame.x_m,
            ego_frame.y_m,
            ego_frame.yaw_deg,
            ego_dimensions["length_m"],
            ego_dimensions["width_m"],
        )
        actor_rectangle = rectangle_corners(
            actor_frame.x_m,
            actor_frame.y_m,
            actor_frame.yaw_deg,
            actor_dimensions["length_m"],
            actor_dimensions["width_m"],
        )
        clearance_m, overlaps = rectangle_clearance(
            ego_rectangle,
            actor_rectangle,
        )
        clearances.append((
            ego_frame.t_s,
            clearance_m,
            overlaps,
        ))

    last = common[-1]
    review_frames = sorted(set([
        common[0],
        int(round(3.5 * scenario.fps)),
        int(round(6.5 * scenario.fps)),
        int(round(12.5 * scenario.fps)),
        int(round(15.45 * scenario.fps)),
        last,
    ]))
    review_frames = [
        frame_idx
        for frame_idx in review_frames
        if frame_idx in ego and frame_idx in actor
    ]

    figure, (bev_axis, clearance_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )

    bev_axis.plot(
        [ego[index].x_m for index in common],
        [ego[index].y_m for index in common],
        color="#2563eb",
        linewidth=2.0,
        label="canonical ego",
    )
    bev_axis.plot(
        [actor[index].x_m for index in common],
        [actor[index].y_m for index in common],
        color="#dc2626",
        linewidth=2.0,
        label=args.actor_id,
    )
    bev_axis.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
    bev_axis.axhline(-3.5, color="#9ca3af", linestyle="--", linewidth=1.0)

    for frame_idx in review_frames:
        for frame, dimensions, color in (
            (ego[frame_idx], ego_dimensions, "#2563eb"),
            (actor[frame_idx], actor_dimensions, "#dc2626"),
        ):
            corners = rectangle_corners(
                frame.x_m,
                frame.y_m,
                frame.yaw_deg,
                dimensions["length_m"],
                dimensions["width_m"],
            )
            bev_axis.add_patch(Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                alpha=0.65,
            ))
        label_offset = (0, 8)
        if frame_idx == int(round(15.45 * scenario.fps)):
            label_offset = (-24, 8)
        elif frame_idx == last:
            label_offset = (24, 8)
        bev_axis.annotate(
            f"f{frame_idx}\n{actor[frame_idx].t_s:.2f}s",
            xy=(actor[frame_idx].x_m, actor[frame_idx].y_m + 1.1),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=7,
            ha="center",
        )

    bev_axis.set_title(
        "Resolved pre-contact scenario: BEV trajectories and physical footprints"
    )
    bev_axis.set_xlabel("Scenario +x forward (m)")
    bev_axis.set_ylabel("Scenario +y left (m)")
    bev_axis.set_ylim(-5.2, 2.2)
    bev_axis.grid(True, alpha=0.25)
    bev_axis.legend(loc="upper left")

    clearance_axis.plot(
        [sample[0] for sample in clearances],
        [sample[1] for sample in clearances],
        color="#059669",
        linewidth=2.0,
        label="oriented-rectangle clearance",
    )
    threshold = float(report["minimum_clearance_required_m"])
    clearance_axis.axhline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"required minimum = {threshold:.2f} m",
    )
    terminal = report["terminal_clearance"]
    clearance_axis.scatter(
        [terminal["t_s"]],
        [terminal["clearance_m"]],
        color="#7c3aed",
        zorder=3,
        label=(
            f"terminal = {terminal['clearance_m']:.3f} m "
            f"at {terminal['t_s']:.2f} s"
        ),
    )
    clearance_axis.set_title(
        "Physical clearance across the full resolved trajectory"
    )
    clearance_axis.set_xlabel("Time (s)")
    clearance_axis.set_ylabel("Rectangle clearance (m)")
    clearance_axis.set_ylim(bottom=0.0)
    clearance_axis.grid(True, alpha=0.25)
    clearance_axis.legend(loc="upper right")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print("saved:", output_path)


if __name__ == "__main__":
    main()
