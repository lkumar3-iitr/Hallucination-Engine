from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from scenario_generator.schema import ResolvedActorFrame, ResolvedScenario


def _group_frames_by_actor(frames: list[ResolvedActorFrame]) -> dict[str, list[ResolvedActorFrame]]:
    grouped: dict[str, list[ResolvedActorFrame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.actor_id].append(frame)

    for actor_id in grouped:
        grouped[actor_id] = sorted(grouped[actor_id], key=lambda f: f.frame_idx)

    return grouped


def plot_resolved_bev(
    scenario: ResolvedScenario,
    output_path: str | Path,
    every_n_frames: int = 5,
    show_frame_labels: bool = False,
) -> None:
    """Create a simple BEV debug plot for a resolved scenario.

    Coordinate convention:
    - x-axis: forward distance in meters.
    - y-axis: lateral distance in meters.
    - ego starts at (0, 0), yaw=0.
    - positive y means left of ego lane center.
    - opposite lane in current v1 examples is y=-lane_width.

    The plot is intentionally simple. It is for debugging trajectory logic,
    not for publication-quality visualization.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    road = scenario.road
    lane_width = road.lane_width_m

    grouped = _group_frames_by_actor(scenario.frames)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw road/lane center lines.
    x_min = -10.0
    x_max = max(road.road_length_m, 80.0)

    # Ego lane center.
    ax.plot([x_min, x_max], [0.0, 0.0], linestyle="-", linewidth=1.5, label="ego lane center")

    # Same-direction lane centers.
    for i in range(1, road.num_lanes_same_direction):
        y = i * lane_width
        ax.plot([x_min, x_max], [y, y], linestyle="--", linewidth=1.0, label=f"same lane {i}")

    # Opposite-direction lane centers.
    for i in range(road.num_lanes_opposite_direction):
        y = -(i + 1) * lane_width
        ax.plot([x_min, x_max], [y, y], linestyle="--", linewidth=1.0, label=f"opposite lane {i}")

    # Approximate lane boundaries.
    min_lane_y = -road.num_lanes_opposite_direction * lane_width - lane_width / 2.0
    max_lane_y = max((road.num_lanes_same_direction - 1) * lane_width + lane_width / 2.0, lane_width / 2.0)

    y = min_lane_y
    while y <= max_lane_y + 1e-6:
        ax.plot([x_min, x_max], [y, y], linestyle=":", linewidth=0.8)
        y += lane_width

    # Draw ego marker.
    ax.scatter([scenario.ego.initial_x_m], [scenario.ego.initial_y_m], marker="*", s=180, label="ego start")
    ax.arrow(
        scenario.ego.initial_x_m,
        scenario.ego.initial_y_m,
        5.0,
        0.0,
        head_width=0.35,
        head_length=1.0,
        length_includes_head=True,
    )

    # Draw actor trajectories.
    for actor_id, frames in grouped.items():
        xs = [f.x_m for f in frames]
        ys = [f.y_m for f in frames]

        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.5, label=f"{actor_id} trajectory")

        # Start/end markers.
        # If the actor is static, start and end overlap. Show only one label.
        dx_total = xs[-1] - xs[0]
        dy_total = ys[-1] - ys[0]
        is_static = abs(dx_total) + abs(dy_total) < 1e-4

        if is_static:
            ax.scatter([xs[0]], [ys[0]], marker="s", s=80)
            ax.text(xs[0], ys[0] + 0.35, f"{actor_id} static", fontsize=8)
        else:
            ax.scatter([xs[0]], [ys[0]], marker="s", s=70)
            ax.text(xs[0], ys[0] + 0.35, f"{actor_id} start", fontsize=8)

            ax.scatter([xs[-1]], [ys[-1]], marker="x", s=80)
            ax.text(xs[-1], ys[-1] + 0.35, f"{actor_id} end", fontsize=8)

        # Direction arrows every N frames.
        step = max(1, every_n_frames)
        for idx in range(0, len(frames) - 1, step):
            f0 = frames[idx]
            f1 = frames[min(idx + 1, len(frames) - 1)]
            dx = f1.x_m - f0.x_m
            dy = f1.y_m - f0.y_m

            # Avoid drawing zero-length arrows.
            if abs(dx) + abs(dy) < 1e-6:
                continue

            ax.arrow(
                f0.x_m,
                f0.y_m,
                dx,
                dy,
                head_width=0.20,
                head_length=0.50,
                length_includes_head=True,
                alpha=0.8,
            )

            if show_frame_labels:
                ax.text(f0.x_m, f0.y_m - 0.25, str(f0.frame_idx), fontsize=7)

    ax.set_title(f"BEV Debug Plot: {scenario.scenario_id}")
    ax.set_xlabel("x forward distance (m)")
    ax.set_ylabel("y lateral distance (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4)
    ax.legend(loc="best", fontsize=8)

    # Keep a useful view around the generated actors.
    all_x = [f.x_m for f in scenario.frames] + [scenario.ego.initial_x_m]
    all_y = [f.y_m for f in scenario.frames] + [scenario.ego.initial_y_m]

    if all_x:
        ax.set_xlim(min(min(all_x) - 10.0, -10.0), max(max(all_x) + 10.0, 60.0))
    if all_y:
        ax.set_ylim(min(min(all_y) - 3.0, min_lane_y - 1.0), max(max(all_y) + 3.0, max_lane_y + 1.0))

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
