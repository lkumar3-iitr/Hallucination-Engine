import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


# ============================================================
# Numeric helpers
# ============================================================

def is_finite(x):
    return (
        isinstance(x, (int, float))
        and math.isfinite(float(x))
    )


def mean(values):
    values = [
        float(v)
        for v in values
        if is_finite(v)
    ]

    if not values:
        return math.nan

    return sum(values) / len(values)


def mae(values):
    values = [
        abs(float(v))
        for v in values
        if is_finite(v)
    ]

    if not values:
        return math.nan

    return sum(values) / len(values)


def fmt(x):
    if not is_finite(x):
        return "nan"

    return f"{float(x):.3f}"


def safe_float(
    value,
    default=math.nan,
):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

        return default

    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# CARLA v2 ground truth loader
# ============================================================

def load_real_ground_truth_v2(path):
    """
    Load:

        real_ground_truth_v2.jsonl

    Output:

        frames[frame_idx][actor_id] = actor record

    Actor absence is preserved:
    if an actor is not in a frame's actors[] list, it did not exist
    on that frame according to the CARLA executor.
    """

    frames = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rec = json.loads(
                line
            )

            frame_idx = int(
                rec[
                    "recorded_frame_idx"
                ]
            )

            actor_rows = {}

            for actor in rec.get(
                "actors",
                [],
            ):

                actor_id = str(
                    actor["actor_id"]
                )

                bbox = actor.get(
                    "bbox",
                    {},
                )

                planned_state = (
                    actor.get(
                        "planned_state",
                        {},
                    )
                )

                # ------------------------------------------------
                # Primary real depth:
                # projected CARLA bbox depth when available.
                #
                # Fallback:
                # planned ego-initial forward distance.
                #
                # The fallback is useful for filtering, but is not
                # identical to current-camera depth when ego moves.
                # ------------------------------------------------

                bbox_depth = safe_float(
                    bbox.get(
                        "depth_m"
                    )
                )

                planned_z = safe_float(
                    planned_state.get(
                        "z_m"
                    )
                )

                if is_finite(
                    bbox_depth
                ):
                    filter_depth = (
                        bbox_depth
                    )

                    depth_source = (
                        "bbox_depth_m"
                    )

                else:
                    filter_depth = (
                        planned_z
                    )

                    depth_source = (
                        "planned_z_m"
                    )

                actor_rows[
                    actor_id
                ] = {

                    "frame_idx":
                        frame_idx,

                    "actor_id":
                        actor_id,

                    "present":
                        True,

                    "visible":
                        bool(
                            bbox.get(
                                "visible",
                                False,
                            )
                        ),

                    "cx":
                        safe_float(
                            bbox.get(
                                "cx"
                            )
                        ),

                    "bottom_y":
                        safe_float(
                            bbox.get(
                                "bottom_y"
                            )
                        ),

                    "box_width":
                        safe_float(
                            bbox.get(
                                "box_width"
                            )
                        ),

                    "box_height":
                        safe_float(
                            bbox.get(
                                "box_height"
                            )
                        ),

                    "depth_m":
                        bbox_depth,

                    "planned_z_m":
                        planned_z,

                    "filter_depth_m":
                        filter_depth,

                    "depth_source":
                        depth_source,

                    "bbox":
                        bbox,

                    "raw":
                        actor,
                }

            frames[
                frame_idx
            ] = {

                "frame_idx":
                    frame_idx,

                "t_s":
                    safe_float(
                        rec.get(
                            "t_s"
                        )
                    ),

                "actors":
                    actor_rows,

                "raw":
                    rec,
            }

    return frames


# ============================================================
# HE metadata loader
# ============================================================

def load_he_metadata_v2(path):
    """
    Load HE metadata.json.

    Output:

        frames[frame_idx][actor_id] = adversary record

    Because compositor v2 now respects lifecycle, actor absence in
    frame["adversaries"] means the actor did not exist on that frame.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        meta = json.load(
            f
        )

    frames = {}

    for frame in meta.get(
        "frames",
        [],
    ):

        frame_idx = int(
            frame[
                "frame_idx"
            ]
        )

        actor_rows = {}

        for adv in frame.get(
            "adversaries",
            [],
        ):

            actor_id = str(
                adv["id"]
            )

            box = adv.get(
                "box",
                {},
            )

            actor_rows[
                actor_id
            ] = {

                "frame_idx":
                    frame_idx,

                "actor_id":
                    actor_id,

                "present":
                    True,

                "box_visible":
                    bool(
                        box.get(
                            "visible",
                            False,
                        )
                    ),

                "rendered":
                    bool(
                        adv.get(
                            "rendered",
                            False,
                        )
                    ),

                # For geometric comparison, retain the convention
                # used by the original comparator:
                #
                # HE visible = placement says visible AND sprite was
                # actually rendered.
                "visible":
                    (
                        bool(
                            box.get(
                                "visible",
                                False,
                            )
                        )
                        and
                        bool(
                            adv.get(
                                "rendered",
                                False,
                            )
                        )
                    ),

                "cx":
                    safe_float(
                        box.get(
                            "cx"
                        )
                    ),

                "bottom_y":
                    safe_float(
                        box.get(
                            "bottom_y"
                        )
                    ),

                "box_width":
                    safe_float(
                        box.get(
                            "box_width"
                        )
                    ),

                "box_height":
                    safe_float(
                        box.get(
                            "box_height"
                        )
                    ),

                "z_m":
                    safe_float(
                        box.get(
                            "z_m"
                        )
                    ),

                "box":
                    box,

                "raw":
                    adv,
            }

        frames[
            frame_idx
        ] = {

            "frame_idx":
                frame_idx,

            "actors":
                actor_rows,

            "raw":
                frame,
        }

    return frames


# ============================================================
# Empty actor records
# ============================================================

def empty_real_actor(
    frame_idx,
    actor_id,
):
    return {
        "frame_idx":
            frame_idx,

        "actor_id":
            actor_id,

        "present":
            False,

        "visible":
            False,

        "cx":
            math.nan,

        "bottom_y":
            math.nan,

        "box_width":
            math.nan,

        "box_height":
            math.nan,

        "depth_m":
            math.nan,

        "planned_z_m":
            math.nan,

        "filter_depth_m":
            math.nan,

        "depth_source":
            None,

        "bbox":
            {},

        "raw":
            None,
    }


def empty_he_actor(
    frame_idx,
    actor_id,
):
    return {
        "frame_idx":
            frame_idx,

        "actor_id":
            actor_id,

        "present":
            False,

        "box_visible":
            False,

        "rendered":
            False,

        "visible":
            False,

        "cx":
            math.nan,

        "bottom_y":
            math.nan,

        "box_width":
            math.nan,

        "box_height":
            math.nan,

        "z_m":
            math.nan,

        "box":
            {},

        "raw":
            None,
    }


# ============================================================
# Actor / frame discovery
# ============================================================

def collect_actor_ids(
    real_frames,
    he_frames,
):
    actor_ids = set()

    for frame in real_frames.values():
        actor_ids.update(
            frame[
                "actors"
            ].keys()
        )

    for frame in he_frames.values():
        actor_ids.update(
            frame[
                "actors"
            ].keys()
        )

    return sorted(
        actor_ids
    )


# ============================================================
# Comparison row generation
# ============================================================

def build_comparison_rows(
    real_frames,
    he_frames,
):
    """
    One row for every:

        (frame_idx, actor_id)

    over the union of actor identities and common video frames.
    """

    common_frames = sorted(
        set(
            real_frames.keys()
        )
        &
        set(
            he_frames.keys()
        )
    )

    actor_ids = (
        collect_actor_ids(
            real_frames,
            he_frames,
        )
    )

    rows = []

    for frame_idx in common_frames:

        real_frame = (
            real_frames[
                frame_idx
            ]
        )

        he_frame = (
            he_frames[
                frame_idx
            ]
        )

        for actor_id in actor_ids:

            r = (
                real_frame[
                    "actors"
                ].get(
                    actor_id,
                    empty_real_actor(
                        frame_idx,
                        actor_id,
                    ),
                )
            )

            h = (
                he_frame[
                    "actors"
                ].get(
                    actor_id,
                    empty_he_actor(
                        frame_idx,
                        actor_id,
                    ),
                )
            )

            real_present = bool(
                r["present"]
            )

            he_present = bool(
                h["present"]
            )

            both_present = (
                real_present
                and
                he_present
            )

            real_visible = (
                bool(
                    r["visible"]
                )
                if real_present
                else False
            )

            he_visible = (
                bool(
                    h["visible"]
                )
                if he_present
                else False
            )

            both_visible = (
                both_present
                and
                real_visible
                and
                he_visible
            )

            row = {

                "frame_idx":
                    frame_idx,

                "actor_id":
                    actor_id,

                # ---------------------------------------------
                # Lifecycle / actor existence
                # ---------------------------------------------

                "real_present":
                    real_present,

                "he_present":
                    he_present,

                "presence_match":
                    (
                        real_present
                        ==
                        he_present
                    ),

                "both_present":
                    both_present,

                # ---------------------------------------------
                # Visibility
                # ---------------------------------------------

                "real_visible":
                    real_visible,

                "he_visible":
                    he_visible,

                "he_rendered":
                    bool(
                        h.get(
                            "rendered",
                            False,
                        )
                    ),

                "visibility_match":
                    (
                        real_visible
                        ==
                        he_visible
                    )
                    if both_present
                    else None,

                "both_visible":
                    both_visible,

                # ---------------------------------------------
                # Horizontal center
                # ---------------------------------------------

                "real_cx":
                    r["cx"],

                "he_cx":
                    h["cx"],

                "err_cx":
                    (
                        h["cx"]
                        -
                        r["cx"]
                    )
                    if both_visible
                    else math.nan,

                # ---------------------------------------------
                # Bottom y
                # ---------------------------------------------

                "real_bottom_y":
                    r[
                        "bottom_y"
                    ],

                "he_bottom_y":
                    h[
                        "bottom_y"
                    ],

                "err_bottom_y":
                    (
                        h[
                            "bottom_y"
                        ]
                        -
                        r[
                            "bottom_y"
                        ]
                    )
                    if both_visible
                    else math.nan,

                # ---------------------------------------------
                # Width
                # ---------------------------------------------

                "real_box_width":
                    r[
                        "box_width"
                    ],

                "he_box_width":
                    h[
                        "box_width"
                    ],

                "err_box_width":
                    (
                        h[
                            "box_width"
                        ]
                        -
                        r[
                            "box_width"
                        ]
                    )
                    if both_visible
                    else math.nan,

                # ---------------------------------------------
                # Height
                # ---------------------------------------------

                "real_box_height":
                    r[
                        "box_height"
                    ],

                "he_box_height":
                    h[
                        "box_height"
                    ],

                "err_box_height":
                    (
                        h[
                            "box_height"
                        ]
                        -
                        r[
                            "box_height"
                        ]
                    )
                    if both_visible
                    else math.nan,

                # ---------------------------------------------
                # Depth information
                # ---------------------------------------------

                "real_depth_m":
                    r[
                        "depth_m"
                    ],

                "real_planned_z_m":
                    r[
                        "planned_z_m"
                    ],

                "real_filter_depth_m":
                    r[
                        "filter_depth_m"
                    ],

                "real_depth_source":
                    r[
                        "depth_source"
                    ],

                "he_z_m":
                    h[
                        "z_m"
                    ],
            }

            rows.append(
                row
            )

    return (
        rows,
        common_frames,
        actor_ids,
    )


# ============================================================
# Summary calculations
# ============================================================

def summarize_rows(
    rows,
    min_real_depth,
):
    """
    Summarize one actor's rows or all actors together.
    """

    total_rows = len(
        rows
    )

    presence_matches = [
        row
        for row in rows
        if row[
            "presence_match"
        ]
    ]

    both_present = [
        row
        for row in rows
        if row[
            "both_present"
        ]
    ]

    visibility_comparable = [
        row
        for row in rows
        if (
            row[
                "both_present"
            ]
            and
            row[
                "visibility_match"
            ]
            is not None
        )
    ]

    visibility_matches = [
        row
        for row
        in visibility_comparable
        if row[
            "visibility_match"
        ]
    ]

    # --------------------------------------------------------
    # Both-visible rows used for geometric error.
    #
    # Depth filtering is applied only here, matching the original
    # comparator's behavior.
    # --------------------------------------------------------

    both_visible = []

    for row in rows:

        if not row[
            "both_visible"
        ]:
            continue

        depth = row[
            "real_filter_depth_m"
        ]

        if (
            float(
                min_real_depth
            )
            > 0.0
        ):

            if not is_finite(
                depth
            ):
                continue

            if (
                float(depth)
                <
                float(
                    min_real_depth
                )
            ):
                continue

        both_visible.append(
            row
        )

    real_only_present = [
        row
        for row in rows
        if (
            row[
                "real_present"
            ]
            and
            not row[
                "he_present"
            ]
        )
    ]

    he_only_present = [
        row
        for row in rows
        if (
            row[
                "he_present"
            ]
            and
            not row[
                "real_present"
            ]
        )
    ]

    real_visible_he_not = [
        row
        for row in both_present
        if (
            row[
                "real_visible"
            ]
            and
            not row[
                "he_visible"
            ]
        )
    ]

    he_visible_real_not = [
        row
        for row in both_present
        if (
            row[
                "he_visible"
            ]
            and
            not row[
                "real_visible"
            ]
        )
    ]

    summary = {

        # ----------------------------------------------------
        # Lifecycle
        # ----------------------------------------------------

        "num_rows":
            total_rows,

        "num_both_present":
            len(
                both_present
            ),

        "num_real_only_present":
            len(
                real_only_present
            ),

        "num_he_only_present":
            len(
                he_only_present
            ),

        "presence_agreement":
            (
                len(
                    presence_matches
                )
                /
                total_rows
            )
            if total_rows
            else math.nan,

        # ----------------------------------------------------
        # Visibility
        # ----------------------------------------------------

        "num_visibility_comparable":
            len(
                visibility_comparable
            ),

        "visibility_agreement":
            (
                len(
                    visibility_matches
                )
                /
                len(
                    visibility_comparable
                )
            )
            if visibility_comparable
            else math.nan,

        "num_real_visible_he_not":
            len(
                real_visible_he_not
            ),

        "num_he_visible_real_not":
            len(
                he_visible_real_not
            ),

        # ----------------------------------------------------
        # Geometric comparison
        # ----------------------------------------------------

        "num_both_visible":
            len(
                both_visible
            ),

        "min_real_depth_m":
            float(
                min_real_depth
            ),

        "mae_cx_px":
            mae(
                [
                    row[
                        "err_cx"
                    ]
                    for row
                    in both_visible
                ]
            ),

        "mae_bottom_y_px":
            mae(
                [
                    row[
                        "err_bottom_y"
                    ]
                    for row
                    in both_visible
                ]
            ),

        "mae_box_width_px":
            mae(
                [
                    row[
                        "err_box_width"
                    ]
                    for row
                    in both_visible
                ]
            ),

        "mae_box_height_px":
            mae(
                [
                    row[
                        "err_box_height"
                    ]
                    for row
                    in both_visible
                ]
            ),

        # ----------------------------------------------------
        # Signed bias
        # ----------------------------------------------------

        "mean_signed_cx_px":
            mean(
                [
                    row[
                        "err_cx"
                    ]
                    for row
                    in both_visible
                ]
            ),

        "mean_signed_bottom_y_px":
            mean(
                [
                    row[
                        "err_bottom_y"
                    ]
                    for row
                    in both_visible
                ]
            ),

        "mean_signed_box_width_px":
            mean(
                [
                    row[
                        "err_box_width"
                    ]
                    for row
                    in both_visible
                ]
            ),

        "mean_signed_box_height_px":
            mean(
                [
                    row[
                        "err_box_height"
                    ]
                    for row
                    in both_visible
                ]
            ),
    }

    return (
        summary,
        both_visible,
    )


# ============================================================
# Printing
# ============================================================

def print_summary(
    title,
    summary,
):
    print()

    print(
        "=" * 76
    )

    print(
        title
    )

    print(
        "=" * 76
    )

    print(
        "Rows evaluated:             ",
        summary[
            "num_rows"
        ],
    )

    print(
        "Both present:               ",
        summary[
            "num_both_present"
        ],
    )

    print(
        "Real-only present:          ",
        summary[
            "num_real_only_present"
        ],
    )

    print(
        "HE-only present:            ",
        summary[
            "num_he_only_present"
        ],
    )

    print(
        "Presence agreement:         ",
        fmt(
            summary[
                "presence_agreement"
            ]
        ),
    )

    print()

    print(
        "Visibility comparable:      ",
        summary[
            "num_visibility_comparable"
        ],
    )

    print(
        "Visibility agreement:       ",
        fmt(
            summary[
                "visibility_agreement"
            ]
        ),
    )

    print(
        "Real visible / HE not:      ",
        summary[
            "num_real_visible_he_not"
        ],
    )

    print(
        "HE visible / Real not:      ",
        summary[
            "num_he_visible_real_not"
        ],
    )

    print()

    print(
        "Both visible for bbox:      ",
        summary[
            "num_both_visible"
        ],
    )

    print(
        "Min real depth filter:      ",
        fmt(
            summary[
                "min_real_depth_m"
            ]
        ),
        "m",
    )

    print()

    print(
        "MAE cx px:                  ",
        fmt(
            summary[
                "mae_cx_px"
            ]
        ),
    )

    print(
        "MAE bottom_y px:            ",
        fmt(
            summary[
                "mae_bottom_y_px"
            ]
        ),
    )

    print(
        "MAE width px:               ",
        fmt(
            summary[
                "mae_box_width_px"
            ]
        ),
    )

    print(
        "MAE height px:              ",
        fmt(
            summary[
                "mae_box_height_px"
            ]
        ),
    )

    print()

    print(
        "Signed mean cx px:          ",
        fmt(
            summary[
                "mean_signed_cx_px"
            ]
        ),
    )

    print(
        "Signed mean bottom_y px:    ",
        fmt(
            summary[
                "mean_signed_bottom_y_px"
            ]
        ),
    )

    print(
        "Signed mean width px:       ",
        fmt(
            summary[
                "mean_signed_box_width_px"
            ]
        ),
    )

    print(
        "Signed mean height px:      ",
        fmt(
            summary[
                "mean_signed_box_height_px"
            ]
        ),
    )

    print(
        "=" * 76
    )


def print_sample_rows(
    actor_id,
    rows,
):
    if not rows:
        print()
        print(
            f"No both-visible rows "
            f"for actor {actor_id}."
        )
        return

    print()
    print(
        f"Sample rows: {actor_id}"
    )

    print(
        "frame | real he | "
        "real_cx he_cx err | "
        "real_by he_by err | "
        "real_w he_w err | "
        "real_h he_h err"
    )

    print(
        "-" * 142
    )

    step = max(
        1,
        len(rows)
        // 10,
    )

    for row in rows[
        ::step
    ]:

        print(
            f"{row['frame_idx']:5d} | "

            f"{int(row['real_visible'])}    "
            f"{int(row['he_visible'])}  | "

            f"{fmt(row['real_cx']):>7} "
            f"{fmt(row['he_cx']):>7} "
            f"{fmt(row['err_cx']):>7} | "

            f"{fmt(row['real_bottom_y']):>7} "
            f"{fmt(row['he_bottom_y']):>7} "
            f"{fmt(row['err_bottom_y']):>7} | "

            f"{fmt(row['real_box_width']):>7} "
            f"{fmt(row['he_box_width']):>7} "
            f"{fmt(row['err_box_width']):>7} | "

            f"{fmt(row['real_box_height']):>7} "
            f"{fmt(row['he_box_height']):>7} "
            f"{fmt(row['err_box_height']):>7}"
        )


# ============================================================
# JSON-safe conversion
# ============================================================

def json_safe(
    obj,
):
    """
    Convert NaN / inf values to None so result JSON is standards-safe.
    """

    if isinstance(
        obj,
        dict,
    ):
        return {
            key:
                json_safe(
                    value
                )
            for (
                key,
                value,
            )
            in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):
        return [
            json_safe(
                value
            )
            for value
            in obj
        ]

    if isinstance(
        obj,
        float,
    ):

        if not math.isfinite(
            obj
        ):
            return None

    return obj


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Actor-ID-aware CARLA vs HE "
                "multi-actor geometric comparison."
            )
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
        help=(
            "Optionally evaluate only one actor."
        ),
    )

    parser.add_argument(
        "--min-real-depth",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--out-json",
        default=None,
    )

    parser.add_argument(
        "--out-csv",
        default=None,
    )

    args = parser.parse_args()

    # ========================================================
    # Load
    # ========================================================

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

    if not common_frames:
        raise RuntimeError(
            "CARLA ground truth and HE metadata "
            "have no common frames."
        )

    if not actor_ids:
        raise RuntimeError(
            "No actor IDs found in either source."
        )

    # ========================================================
    # Optional actor selection
    # ========================================================

    if args.actor_id:

        if (
            args.actor_id
            not in actor_ids
        ):

            raise ValueError(
                f"Actor {args.actor_id!r} "
                f"not found. Available actors: "
                f"{actor_ids}"
            )

        selected_actor_ids = [
            args.actor_id
        ]

        selected_rows = [
            row
            for row in rows
            if (
                row[
                    "actor_id"
                ]
                ==
                args.actor_id
            )
        ]

    else:

        selected_actor_ids = (
            actor_ids
        )

        selected_rows = (
            rows
        )

    # ========================================================
    # Header
    # ========================================================

    print()

    print(
        "CARLA frames:",
        len(
            real_frames
        ),
    )

    print(
        "HE frames:",
        len(
            he_frames
        ),
    )

    print(
        "Common frames:",
        len(
            common_frames
        ),
    )

    print(
        "Actors:",
        ", ".join(
            selected_actor_ids
        ),
    )

    # ========================================================
    # Per-actor summaries
    # ========================================================

    per_actor_summary = {}

    per_actor_both_visible = {}

    for actor_id in selected_actor_ids:

        actor_rows = [
            row
            for row
            in selected_rows
            if (
                row[
                    "actor_id"
                ]
                ==
                actor_id
            )
        ]

        (
            summary,
            both_visible,
        ) = summarize_rows(
            actor_rows,
            args.min_real_depth,
        )

        per_actor_summary[
            actor_id
        ] = summary

        per_actor_both_visible[
            actor_id
        ] = both_visible

        print_summary(
            (
                "CARLA vs HE "
                f"[{actor_id}]"
            ),
            summary,
        )

        print_sample_rows(
            actor_id,
            both_visible,
        )

    # ========================================================
    # Overall aggregate
    # ========================================================

    (
        overall_summary,
        overall_both_visible,
    ) = summarize_rows(
        selected_rows,
        args.min_real_depth,
    )

    if (
        len(
            selected_actor_ids
        )
        > 1
    ):

        print_summary(
            "CARLA vs HE [ALL ACTORS]",
            overall_summary,
        )

    # ========================================================
    # Save JSON
    # ========================================================

    if args.out_json:

        output = {

            "real_ground_truth":
                args.real_ground_truth,

            "he_metadata":
                args.he_metadata,

            "common_frames":
                len(
                    common_frames
                ),

            "actor_ids":
                selected_actor_ids,

            "min_real_depth_m":
                args.min_real_depth,

            "per_actor":
                per_actor_summary,

            "overall":
                overall_summary,

            "rows":
                selected_rows,
        }

        output = json_safe(
            output
        )

        output_path = Path(
            args.out_json
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
                allow_nan=False,
            )

        print(
            "[SAVED]",
            output_path,
        )

    # ========================================================
    # Save CSV
    # ========================================================

    if args.out_csv:

        output_path = Path(
            args.out_csv
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if selected_rows:

            fieldnames = list(
                selected_rows[
                    0
                ].keys()
            )

            with output_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = (
                    csv.DictWriter(
                        f,
                        fieldnames=
                            fieldnames,
                    )
                )

                writer.writeheader()

                for row in selected_rows:

                    safe_row = {}

                    for (
                        key,
                        value,
                    ) in row.items():

                        if (
                            isinstance(
                                value,
                                float,
                            )
                            and
                            not math.isfinite(
                                value
                            )
                        ):
                            value = ""

                        safe_row[
                            key
                        ] = value

                    writer.writerow(
                        safe_row
                    )

        print(
            "[SAVED]",
            output_path,
        )


if __name__ == "__main__":
    main()