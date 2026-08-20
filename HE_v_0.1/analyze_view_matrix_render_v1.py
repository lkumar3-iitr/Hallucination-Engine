"""
analyze_view_matrix_render_v1.py

Analyze HE temporal compositor metadata for view-matrix sprite selection.

Reports:
    - rendered frame/adversary counts
    - selected distance/elevation distributions
    - angle/distance/elevation quantization errors
    - number of sprite-selection switches
    - exact frames where distance/elevation selection changes
    - per-frame selector timeline

Works with metadata produced by:
    run_he_temporal_compositor_v2.py
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Helpers
# ============================================================

def mean(values):
    if not values:
        return 0.0

    return sum(values) / float(len(values))


def circular_angle_delta_deg(a, b):
    """
    Smallest absolute difference between two angles.
    """

    return abs(
        (
            float(a)
            -
            float(b)
            +
            180.0
        )
        %
        360.0
        -
        180.0
    )


def fmt_counter(counter):
    if not counter:
        return "none"

    parts = []

    for key in sorted(
        counter.keys()
    ):
        parts.append(
            "{}:{}".format(
                key,
                counter[key],
            )
        )

    return ", ".join(
        parts
    )


def get_sprite_row(
    frame_idx,
    adversary,
):
    sprite = adversary.get(
        "sprite",
        None,
    )

    if not sprite:
        return None

    if (
        sprite.get(
            "mode",
            ""
        )
        !=
        "view_matrix"
    ):
        return None

    state = adversary.get(
        "state",
        {}
    )

    box = adversary.get(
        "box",
        {}
    )

    paste = adversary.get(
        "paste",
        {}
    )

    resize = adversary.get(
        "view_matrix_resize",
        {}
    ) or {}

    return {
        "frame_idx":
            int(frame_idx),

        "actor_id":
            adversary.get(
                "id",
                ""
            ),

        "rendered":
            bool(
                adversary.get(
                    "rendered",
                    False
                )
            ),

        # ----------------------------------------------------
        # Camera-relative actor state
        # ----------------------------------------------------

        "x_m":
            float(
                state.get(
                    "x_m",
                    0.0
                )
            ),

        "y_m":
            float(
                state.get(
                    "y_m",
                    0.0
                )
            ),

        "z_m":
            float(
                state.get(
                    "z_m",
                    0.0
                )
            ),

        "yaw_deg":
            float(
                state.get(
                    "yaw_deg",
                    0.0
                )
            ),

        # ----------------------------------------------------
        # Angle selection
        # ----------------------------------------------------

        "query_angle_deg":
            float(
                sprite.get(
                    "relative_angle_deg",
                    0.0
                )
            ),

        "selected_angle_deg":
            float(
                sprite.get(
                    "selected_angle",
                    0.0
                )
            ),

        "angle_error_deg":
            float(
                sprite.get(
                    "angle_error_deg",
                    0.0
                )
            ),

        # ----------------------------------------------------
        # Distance selection
        # ----------------------------------------------------

        "query_distance_m":
            float(
                sprite.get(
                    "query_distance_m",
                    0.0
                )
            ),

        "selected_distance_m":
            float(
                sprite.get(
                    "selected_distance_m",
                    0.0
                )
            ),

        "distance_error_m":
            float(
                sprite.get(
                    "distance_error_m",
                    0.0
                )
            ),

        # ----------------------------------------------------
        # Elevation selection
        # ----------------------------------------------------

        "query_elevation_deg":
            float(
                sprite.get(
                    "query_elevation_deg",
                    0.0
                )
            ),

        "selected_elevation_deg":
            float(
                sprite.get(
                    "selected_elevation_deg",
                    0.0
                )
            ),

        "elevation_error_deg":
            float(
                sprite.get(
                    "elevation_error_deg",
                    0.0
                )
            ),

        # ----------------------------------------------------
        # Placement
        # ----------------------------------------------------

        "box_cx":
            float(
                box.get(
                    "cx",
                    0.0
                )
            ),

        "box_bottom_y":
            float(
                box.get(
                    "bottom_y",
                    0.0
                )
            ),

        "box_width":
            float(
                box.get(
                    "box_width",
                    0.0
                )
            ),

        "box_height":
            float(
                box.get(
                    "box_height",
                    0.0
                )
            ),

        "paste_x1":
            int(
                paste.get(
                    "x1",
                    0
                )
            ),

        "paste_y1":
            int(
                paste.get(
                    "y1",
                    0
                )
            ),

        "sprite_width":
            int(
                paste.get(
                    "sprite_width",
                    0
                )
            ),

        "sprite_height":
            int(
                paste.get(
                    "sprite_height",
                    0
                )
            ),

        "resize_scale":
            float(
                resize.get(
                    "scale",
                    0.0
                )
            ),

        "sprite_path":
            str(
                sprite.get(
                    "sprite_path",
                    ""
                )
            ),
    }


# ============================================================
# Main analysis
# ============================================================

def analyze(
    metadata_path,
    output_dir,
):
    metadata_path = Path(
        metadata_path
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(
            f
        )

    if (
        metadata.get(
            "sprite_bank_mode"
        )
        !=
        "view_matrix"
    ):
        raise RuntimeError(
            "Metadata is not from a view-matrix render."
        )

    output_dir = Path(
        output_dir
        if output_dir
        else metadata_path.parent
        /
        "view_matrix_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = metadata.get(
        "frames",
        []
    )

    timeline_rows = []

    active_frame_count = 0

    for frame in frames:

        frame_idx = int(
            frame.get(
                "frame_idx",
                -1
            )
        )

        if frame.get(
            "active",
            False
        ):
            active_frame_count += 1

        for adversary in frame.get(
            "adversaries",
            []
        ):

            row = get_sprite_row(
                frame_idx=frame_idx,
                adversary=adversary,
            )

            if row is None:
                continue

            timeline_rows.append(
                row
            )

    rendered_rows = [
        row
        for row in timeline_rows
        if row[
            "rendered"
        ]
    ]

    if not rendered_rows:

        raise RuntimeError(
            "No rendered view-matrix adversary rows found."
        )

    # ========================================================
    # Distributions
    # ========================================================

    distance_counter = Counter(
        row[
            "selected_distance_m"
        ]
        for row
        in rendered_rows
    )

    elevation_counter = Counter(
        row[
            "selected_elevation_deg"
        ]
        for row
        in rendered_rows
    )

    angle_counter = Counter(
        int(
            round(
                row[
                    "selected_angle_deg"
                ]
            )
        )
        %
        360
        for row
        in rendered_rows
    )

    # ========================================================
    # Errors
    # ========================================================

    angle_errors = [
        abs(
            row[
                "angle_error_deg"
            ]
        )
        for row
        in rendered_rows
    ]

    distance_errors = [
        abs(
            row[
                "distance_error_m"
            ]
        )
        for row
        in rendered_rows
    ]

    elevation_errors = [
        abs(
            row[
                "elevation_error_deg"
            ]
        )
        for row
        in rendered_rows
    ]

    # ========================================================
    # Transitions per actor
    # ========================================================

    by_actor = defaultdict(
        list
    )

    for row in rendered_rows:

        by_actor[
            row[
                "actor_id"
            ]
        ].append(
            row
        )

    transition_rows = []

    angle_switches = 0
    distance_switches = 0
    elevation_switches = 0
    any_sprite_switches = 0

    for actor_id, rows in (
        by_actor.items()
    ):

        rows.sort(
            key=lambda r:
                r[
                    "frame_idx"
                ]
        )

        previous = None

        for current in rows:

            if previous is None:

                previous = current
                continue

            # Only compare consecutive rendered frames.
            frame_gap = (
                current[
                    "frame_idx"
                ]
                -
                previous[
                    "frame_idx"
                ]
            )

            angle_changed = (
                int(
                    round(
                        current[
                            "selected_angle_deg"
                        ]
                    )
                )
                %
                360
                !=
                int(
                    round(
                        previous[
                            "selected_angle_deg"
                        ]
                    )
                )
                %
                360
            )

            distance_changed = (
                current[
                    "selected_distance_m"
                ]
                !=
                previous[
                    "selected_distance_m"
                ]
            )

            elevation_changed = (
                current[
                    "selected_elevation_deg"
                ]
                !=
                previous[
                    "selected_elevation_deg"
                ]
            )

            if angle_changed:
                angle_switches += 1

            if distance_changed:
                distance_switches += 1

            if elevation_changed:
                elevation_switches += 1

            if (
                angle_changed
                or
                distance_changed
                or
                elevation_changed
            ):

                any_sprite_switches += 1

                transition_rows.append({
                    "actor_id":
                        actor_id,

                    "frame_idx":
                        current[
                            "frame_idx"
                        ],

                    "previous_frame_idx":
                        previous[
                            "frame_idx"
                        ],

                    "frame_gap":
                        frame_gap,

                    "query_distance_m":
                        current[
                            "query_distance_m"
                        ],

                    "previous_distance_m":
                        previous[
                            "selected_distance_m"
                        ],

                    "selected_distance_m":
                        current[
                            "selected_distance_m"
                        ],

                    "query_elevation_deg":
                        current[
                            "query_elevation_deg"
                        ],

                    "previous_elevation_deg":
                        previous[
                            "selected_elevation_deg"
                        ],

                    "selected_elevation_deg":
                        current[
                            "selected_elevation_deg"
                        ],

                    "query_angle_deg":
                        current[
                            "query_angle_deg"
                        ],

                    "previous_angle_deg":
                        previous[
                            "selected_angle_deg"
                        ],

                    "selected_angle_deg":
                        current[
                            "selected_angle_deg"
                        ],

                    "selected_angle_delta_deg":
                        circular_angle_delta_deg(
                            current[
                                "selected_angle_deg"
                            ],
                            previous[
                                "selected_angle_deg"
                            ],
                        ),

                    "angle_changed":
                        angle_changed,

                    "distance_changed":
                        distance_changed,

                    "elevation_changed":
                        elevation_changed,
                })

            previous = current

    # ========================================================
    # Save timeline CSV
    # ========================================================

    timeline_path = (
        output_dir
        /
        "view_selection_timeline.csv"
    )

    timeline_fields = list(
        timeline_rows[
            0
        ].keys()
    )

    with open(
        timeline_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=timeline_fields,
        )

        writer.writeheader()

        writer.writerows(
            timeline_rows
        )

    # ========================================================
    # Save transitions
    # ========================================================

    transitions_path = (
        output_dir
        /
        "view_selection_transitions.csv"
    )

    transition_fields = [
        "actor_id",
        "frame_idx",
        "previous_frame_idx",
        "frame_gap",

        "query_distance_m",
        "previous_distance_m",
        "selected_distance_m",

        "query_elevation_deg",
        "previous_elevation_deg",
        "selected_elevation_deg",

        "query_angle_deg",
        "previous_angle_deg",
        "selected_angle_deg",
        "selected_angle_delta_deg",

        "angle_changed",
        "distance_changed",
        "elevation_changed",
    ]

    with open(
        transitions_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=transition_fields,
        )

        writer.writeheader()

        writer.writerows(
            transition_rows
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "scenario_id":
            metadata.get(
                "scenario_id"
            ),

        "sprite_bank_mode":
            metadata.get(
                "sprite_bank_mode"
            ),

        "available_sprite_count":
            metadata.get(
                "available_sprite_count"
            ),

        "video_frame_count":
            len(
                frames
            ),

        "active_frame_count":
            active_frame_count,

        "view_matrix_rows":
            len(
                timeline_rows
            ),

        "rendered_rows":
            len(
                rendered_rows
            ),

        "actor_count":
            len(
                by_actor
            ),

        "selected_distance_distribution":
            {
                str(k): v
                for k, v
                in sorted(
                    distance_counter.items()
                )
            },

        "selected_elevation_distribution":
            {
                str(k): v
                for k, v
                in sorted(
                    elevation_counter.items()
                )
            },

        "unique_selected_angles":
            len(
                angle_counter
            ),

        "angle_error_deg": {
            "mean":
                mean(
                    angle_errors
                ),

            "max":
                max(
                    angle_errors
                ),
        },

        "distance_error_m": {
            "mean":
                mean(
                    distance_errors
                ),

            "max":
                max(
                    distance_errors
                ),
        },

        "elevation_error_deg": {
            "mean":
                mean(
                    elevation_errors
                ),

            "max":
                max(
                    elevation_errors
                ),
        },

        "switches": {
            "any":
                any_sprite_switches,

            "angle":
                angle_switches,

            "distance":
                distance_switches,

            "elevation":
                elevation_switches,
        },
    }

    summary_path = (
        output_dir
        /
        "view_selection_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Console report
    # ========================================================

    print()
    print("=" * 76)
    print(
        "HE VIEW-MATRIX RENDER ANALYSIS V1"
    )
    print("=" * 76)

    print(
        "Scenario:",
        summary[
            "scenario_id"
        ]
    )

    print(
        "Available sprites:",
        summary[
            "available_sprite_count"
        ]
    )

    print(
        "Video frames:",
        summary[
            "video_frame_count"
        ]
    )

    print(
        "Active frames:",
        summary[
            "active_frame_count"
        ]
    )

    print(
        "Rendered rows:",
        summary[
            "rendered_rows"
        ]
    )

    print(
        "Actors:",
        summary[
            "actor_count"
        ]
    )

    print()
    print(
        "Selected distances:"
    )

    print(
        " ",
        fmt_counter(
            distance_counter
        )
    )

    print(
        "Selected elevations:"
    )

    print(
        " ",
        fmt_counter(
            elevation_counter
        )
    )

    print(
        "Unique selected angles:",
        len(
            angle_counter
        )
    )

    print()
    print(
        "Quantization errors"
    )

    print(
        "  angle     mean={:.4f} deg  max={:.4f} deg".format(
            mean(
                angle_errors
            ),
            max(
                angle_errors
            ),
        )
    )

    print(
        "  distance  mean={:.4f} m    max={:.4f} m".format(
            mean(
                distance_errors
            ),
            max(
                distance_errors
            ),
        )
    )

    print(
        "  elevation mean={:.4f} deg  max={:.4f} deg".format(
            mean(
                elevation_errors
            ),
            max(
                elevation_errors
            ),
        )
    )

    print()
    print(
        "Selection switches"
    )

    print(
        "  any      :",
        any_sprite_switches
    )

    print(
        "  angle    :",
        angle_switches
    )

    print(
        "  distance :",
        distance_switches
    )

    print(
        "  elevation:",
        elevation_switches
    )

    print()
    print(
        "Distance/elevation transitions:"
    )

    important = [
        row
        for row
        in transition_rows
        if (
            row[
                "distance_changed"
            ]
            or
            row[
                "elevation_changed"
            ]
        )
    ]

    for row in important:

        print(
            "  frame={:3d} "
            "qD={:6.2f} "
            "{} -> {}   "
            "qE={:6.2f} "
            "{} -> {}".format(
                int(
                    row[
                        "frame_idx"
                    ]
                ),

                float(
                    row[
                        "query_distance_m"
                    ]
                ),

                row[
                    "previous_distance_m"
                ],

                row[
                    "selected_distance_m"
                ],

                float(
                    row[
                        "query_elevation_deg"
                    ]
                ),

                row[
                    "previous_elevation_deg"
                ],

                row[
                    "selected_elevation_deg"
                ],
            )
        )

    print()
    print(
        "Saved:",
        timeline_path
    )

    print(
        "Saved:",
        transitions_path
    )

    print(
        "Saved:",
        summary_path
    )

    print()
    print(
        "[ViewMatrixAnalysis] DONE"
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    args = parser.parse_args()

    analyze(
        metadata_path=(
            args.metadata
        ),
        output_dir=(
            args.output_dir
        ),
    )


if __name__ == "__main__":
    main()