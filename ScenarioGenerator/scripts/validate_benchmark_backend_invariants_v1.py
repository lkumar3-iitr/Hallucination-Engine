"""
validate_benchmark_backend_invariants_v1.py

Milestone 4C2.

Verify that the CARLA Plan V2 and HE Scenario V2 generated from the
same ResolvedScenarioV2 contain identical actor trajectories after
their shared ScenarioGenerator -> backend coordinate conversion.

This performs NO simulation and NO rendering.

It checks:

    scenario identity
    FPS / timeline
    actor IDs
    actor frame counts
    lifecycle frame ranges
    frame indices
    time
    lateral position
    forward position
    vertical position
    yaw

Expected backend invariant:

    CARLA local_x_m      == HE x_m
    CARLA local_y_m      == HE y_m
    CARLA local_z_m      == HE z_m
    CARLA local_yaw_deg  == HE yaw_deg

A failure here means the benchmark must NOT be executed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from pathlib import Path


# ============================================================
# Basic IO
# ============================================================

def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            allow_nan=False,
        )

        f.write(
            "\n"
        )


# ============================================================
# Numeric helpers
# ============================================================

def angle_difference_deg(
    a: float,
    b: float,
) -> float:
    """
    Signed shortest angular difference a-b in [-180, 180).
    """

    return (
        (
            float(a)
            -
            float(b)
            +
            180.0
        )
        % 360.0
    ) - 180.0


def abs_difference(
    a,
    b,
):
    return abs(
        float(a)
        -
        float(b)
    )


# ============================================================
# CARLA plan helpers
# ============================================================

def carla_actor_map(
    carla_plan: dict,
):
    actors = (
        carla_plan.get(
            "actors",
            []
        )
    )

    output = {}

    for actor in actors:

        actor_id = str(
            actor[
                "actor_id"
            ]
        )

        if actor_id in output:

            raise ValueError(
                f"Duplicate CARLA actor_id: "
                f"{actor_id}"
            )

        output[
            actor_id
        ] = actor

    return output


def carla_frame_map(
    actor: dict,
):
    output = {}

    for frame in actor.get(
        "frames",
        [],
    ):

        frame_idx = int(
            frame[
                "frame_idx"
            ]
        )

        if frame_idx in output:

            raise ValueError(
                "Duplicate CARLA actor "
                f"frame_idx={frame_idx}"
            )

        output[
            frame_idx
        ] = frame

    return output


# ============================================================
# HE helpers
# ============================================================

def he_actor_map(
    he_scenario: dict,
):
    adversaries = (
        he_scenario.get(
            "adversaries",
            []
        )
    )

    output = {}

    for actor in adversaries:

        actor_id = str(
            actor[
                "id"
            ]
        )

        if actor_id in output:

            raise ValueError(
                f"Duplicate HE actor id: "
                f"{actor_id}"
            )

        output[
            actor_id
        ] = actor

    return output


def he_frame_map(
    actor: dict,
):
    motion = actor.get(
        "motion",
        {}
    )

    keyframes = motion.get(
        "keyframes",
        []
    )

    output = {}

    for frame in keyframes:

        frame_idx = int(
            frame[
                "frame_idx"
            ]
        )

        if frame_idx in output:

            raise ValueError(
                "Duplicate HE actor "
                f"frame_idx={frame_idx}"
            )

        output[
            frame_idx
        ] = frame

    return output


# ============================================================
# Case comparison
# ============================================================

def compare_case(
    *,
    case_id: str,
    carla_plan: dict,
    he_scenario: dict,
    tolerance: float,
):
    errors = []

    actor_reports = {}

    max_errors = {
        "t_s": 0.0,
        "x_m": 0.0,
        "y_m": 0.0,
        "z_m": 0.0,
        "yaw_deg": 0.0,
    }

    # ========================================================
    # Scenario identity
    # ========================================================

    carla_scenario_id = str(
        carla_plan.get(
            "scenario_id",
            ""
        )
    )

    he_scenario_id = str(
        he_scenario.get(
            "scenario_id",
            ""
        )
    )

    if (
        carla_scenario_id
        != case_id
    ):
        errors.append(
            "CARLA scenario_id mismatch: "
            f"{carla_scenario_id!r} "
            f"!= {case_id!r}"
        )

    if (
        he_scenario_id
        != case_id
    ):
        errors.append(
            "HE scenario_id mismatch: "
            f"{he_scenario_id!r} "
            f"!= {case_id!r}"
        )

    # ========================================================
    # FPS
    # ========================================================

    carla_fps = float(
        carla_plan[
            "fps"
        ]
    )

    he_timeline = (
        he_scenario.get(
            "timeline",
            {}
        )
    )

    he_fps = float(
        he_timeline[
            "fps"
        ]
    )

    if (
        abs(
            carla_fps
            -
            he_fps
        )
        >
        tolerance
    ):
        errors.append(
            "FPS mismatch: "
            f"CARLA={carla_fps}, "
            f"HE={he_fps}"
        )

    # ========================================================
    # Timeline
    # ========================================================

    ego_frames = (
        carla_plan.get(
            "ego_frames",
            []
        )
    )

    if ego_frames:

        carla_first_frame = min(
            int(
                frame[
                    "frame_idx"
                ]
            )
            for frame
            in ego_frames
        )

        carla_last_frame = max(
            int(
                frame[
                    "frame_idx"
                ]
            )
            for frame
            in ego_frames
        )

        he_start_frame = int(
            he_timeline.get(
                "start_frame",
                -1
            )
        )

        he_end_frame = int(
            he_timeline.get(
                "end_frame",
                -1
            )
        )

        if (
            carla_first_frame
            != he_start_frame
        ):
            errors.append(
                "Timeline start mismatch: "
                f"CARLA={carla_first_frame}, "
                f"HE={he_start_frame}"
            )

        if (
            carla_last_frame
            != he_end_frame
        ):
            errors.append(
                "Timeline end mismatch: "
                f"CARLA={carla_last_frame}, "
                f"HE={he_end_frame}"
            )

    # ========================================================
    # Actor identity sets
    # ========================================================

    carla_actors = (
        carla_actor_map(
            carla_plan
        )
    )

    he_actors = (
        he_actor_map(
            he_scenario
        )
    )

    carla_ids = set(
        carla_actors.keys()
    )

    he_ids = set(
        he_actors.keys()
    )

    if carla_ids != he_ids:

        errors.append(
            "Actor ID set mismatch: "
            f"CARLA={sorted(carla_ids)}, "
            f"HE={sorted(he_ids)}"
        )

    common_actor_ids = sorted(
        carla_ids
        &
        he_ids
    )

    # ========================================================
    # Actor-by-actor comparison
    # ========================================================

    for actor_id in common_actor_ids:

        carla_actor = (
            carla_actors[
                actor_id
            ]
        )

        he_actor = (
            he_actors[
                actor_id
            ]
        )

        carla_frames = (
            carla_frame_map(
                carla_actor
            )
        )

        he_frames = (
            he_frame_map(
                he_actor
            )
        )

        carla_frame_ids = set(
            carla_frames.keys()
        )

        he_frame_ids = set(
            he_frames.keys()
        )

        actor_errors = []

        # ----------------------------------------------------
        # Frame set must match exactly.
        # ----------------------------------------------------

        if (
            carla_frame_ids
            != he_frame_ids
        ):

            missing_in_he = sorted(
                carla_frame_ids
                -
                he_frame_ids
            )

            missing_in_carla = sorted(
                he_frame_ids
                -
                carla_frame_ids
            )

            actor_errors.append(
                "frame set mismatch: "
                f"missing_in_HE="
                f"{missing_in_he[:10]}, "
                f"missing_in_CARLA="
                f"{missing_in_carla[:10]}"
            )

        common_frames = sorted(
            carla_frame_ids
            &
            he_frame_ids
        )

        # ----------------------------------------------------
        # Lifecycle range
        # ----------------------------------------------------

        if common_frames:

            expected_start = min(
                common_frames
            )

            expected_end = max(
                common_frames
            )

            if (
                "active_start_frame"
                in he_actor
            ):

                he_active_start = int(
                    he_actor[
                        "active_start_frame"
                    ]
                )

                if (
                    he_active_start
                    != expected_start
                ):

                    actor_errors.append(
                        "HE active_start_frame "
                        f"{he_active_start} "
                        f"!= {expected_start}"
                    )

            if (
                "active_end_frame"
                in he_actor
            ):

                he_active_end = int(
                    he_actor[
                        "active_end_frame"
                    ]
                )

                if (
                    he_active_end
                    != expected_end
                ):

                    actor_errors.append(
                        "HE active_end_frame "
                        f"{he_active_end} "
                        f"!= {expected_end}"
                    )

        # ----------------------------------------------------
        # Numeric comparison
        # ----------------------------------------------------

        actor_max = {
            "t_s": 0.0,
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_deg": 0.0,
        }

        first_numeric_mismatch = None

        for frame_idx in common_frames:

            c = carla_frames[
                frame_idx
            ]

            h = he_frames[
                frame_idx
            ]

            # -----------------------------------------------
            # HE has explicit t_s.
            #
            # CARLA execution uses exact frame_idx/fps timing,
            # so derive expected time from CARLA frame index.
            # -----------------------------------------------

            expected_t_s = (
                frame_idx
                /
                carla_fps
            )

            t_error = (
                abs_difference(
                    expected_t_s,
                    h[
                        "t_s"
                    ],
                )
            )

            x_error = (
                abs_difference(
                    c[
                        "local_x_m"
                    ],
                    h[
                        "x_m"
                    ],
                )
            )

            y_error = (
                abs_difference(
                    c.get(
                        "local_y_m",
                        0.0,
                    ),
                    h.get(
                        "y_m",
                        0.0,
                    ),
                )
            )

            z_error = (
                abs_difference(
                    c[
                        "local_z_m"
                    ],
                    h[
                        "z_m"
                    ],
                )
            )

            yaw_error = abs(
                angle_difference_deg(
                    c[
                        "local_yaw_deg"
                    ],
                    h[
                        "yaw_deg"
                    ],
                )
            )

            current_errors = {
                "t_s": t_error,
                "x_m": x_error,
                "y_m": y_error,
                "z_m": z_error,
                "yaw_deg": yaw_error,
            }

            for (
                name,
                value,
            ) in current_errors.items():

                actor_max[
                    name
                ] = max(
                    actor_max[
                        name
                    ],
                    value,
                )

                max_errors[
                    name
                ] = max(
                    max_errors[
                        name
                    ],
                    value,
                )

            if (
                first_numeric_mismatch
                is None
                and
                any(
                    value
                    >
                    tolerance
                    for value
                    in current_errors.values()
                )
            ):

                first_numeric_mismatch = {

                    "frame_idx":
                        frame_idx,

                    "carla": {
                        "x_m":
                            c[
                                "local_x_m"
                            ],

                        "y_m":
                            c.get(
                                "local_y_m",
                                0.0,
                            ),

                        "z_m":
                            c[
                                "local_z_m"
                            ],

                        "yaw_deg":
                            c[
                                "local_yaw_deg"
                            ],
                    },

                    "he": {
                        "x_m":
                            h[
                                "x_m"
                            ],

                        "y_m":
                            h.get(
                                "y_m",
                                0.0,
                            ),

                        "z_m":
                            h[
                                "z_m"
                            ],

                        "yaw_deg":
                            h[
                                "yaw_deg"
                            ],
                    },

                    "errors":
                        current_errors,
                }

        if (
            first_numeric_mismatch
            is not None
        ):

            actor_errors.append(
                "numeric trajectory mismatch "
                f"starting at frame "
                f"{first_numeric_mismatch['frame_idx']}"
            )

        actor_pass = (
            len(
                actor_errors
            )
            == 0
            and
            all(
                value
                <= tolerance
                for value
                in actor_max.values()
            )
        )

        actor_reports[
            actor_id
        ] = {

            "status":
                (
                    "PASS"
                    if actor_pass
                    else "FAIL"
                ),

            "carla_frames":
                len(
                    carla_frames
                ),

            "he_frames":
                len(
                    he_frames
                ),

            "common_frames":
                len(
                    common_frames
                ),

            "first_frame":
                (
                    min(
                        common_frames
                    )
                    if common_frames
                    else None
                ),

            "last_frame":
                (
                    max(
                        common_frames
                    )
                    if common_frames
                    else None
                ),

            "max_abs_error":
                actor_max,

            "first_numeric_mismatch":
                first_numeric_mismatch,

            "errors":
                actor_errors,
        }

        for error in actor_errors:

            errors.append(
                f"{actor_id}: "
                f"{error}"
            )

    # ========================================================
    # Final case decision
    # ========================================================

    passed = (
        len(
            errors
        )
        == 0
    )

    return {

        "case_id":
            case_id,

        "status":
            (
                "PASS"
                if passed
                else "FAIL"
            ),

        "tolerance":
            tolerance,

        "carla_actor_ids":
            sorted(
                carla_ids
            ),

        "he_actor_ids":
            sorted(
                he_ids
            ),

        "max_abs_error":
            max_errors,

        "actors":
            actor_reports,

        "errors":
            errors,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Validate CARLA Plan V2 vs HE Scenario V2 "
            "trajectory invariants for an M4C benchmark export."
        )
    )

    parser.add_argument(
        "--m4c-summary",
        required=True,
    )

    parser.add_argument(
        "--case-id",
        default=None,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
    )

    args = parser.parse_args()

    summary_path = Path(
        args.m4c_summary
    )

    summary = load_json(
        summary_path
    )

    m4c_root = (
        summary_path.parent
    )

    benchmark_id = str(
        summary[
            "benchmark_id"
        ]
    )

    cases = (
        summary.get(
            "cases",
            []
        )
    )

    if args.case_id is not None:

        cases = [

            case

            for case in cases

            if (
                case.get(
                    "case_id"
                )
                ==
                args.case_id
            )
        ]

        if not cases:

            raise ValueError(
                "Requested case not found "
                "in M4C summary: "
                f"{args.case_id}"
            )

    print()
    print(
        "=" * 100
    )

    print(
        "M4C2 CARLA <-> HE BACKEND TRAJECTORY INVARIANTS"
    )

    print(
        "=" * 100
    )

    print(
        "benchmark:",
        benchmark_id,
    )

    print(
        "cases:",
        len(
            cases
        ),
    )

    print(
        "tolerance:",
        args.tolerance,
    )

    print()

    reports = []

    passed_count = 0
    failed_count = 0

    for case in cases:

        case_id = str(
            case[
                "case_id"
            ]
        )

        case_dir = (
            m4c_root
            / case_id
        )

        carla_path = (
            case_dir
            / "carla_plan_v2.json"
        )

        he_path = (
            case_dir
            / "he_scenario_v2.json"
        )

        if not carla_path.exists():

            raise FileNotFoundError(
                f"Missing CARLA plan: "
                f"{carla_path}"
            )

        if not he_path.exists():

            raise FileNotFoundError(
                f"Missing HE scenario: "
                f"{he_path}"
            )

        report = compare_case(

            case_id=
                case_id,

            carla_plan=
                load_json(
                    carla_path
                ),

            he_scenario=
                load_json(
                    he_path
                ),

            tolerance=
                float(
                    args.tolerance
                ),
        )

        reports.append(
            report
        )

        if (
            report[
                "status"
            ]
            ==
            "PASS"
        ):

            passed_count += 1

        else:

            failed_count += 1

        max_error = (
            report[
                "max_abs_error"
            ]
        )

        print(
            f"[{report['status']}] "
            f"{case_id}"
        )

        print(
            "       max errors: "
            f"x={max_error['x_m']:.3e}, "
            f"y={max_error['y_m']:.3e}, "
            f"z={max_error['z_m']:.3e}, "
            f"yaw={max_error['yaw_deg']:.3e}, "
            f"t={max_error['t_s']:.3e}"
        )

        for (
            actor_id,
            actor_report,
        ) in report[
            "actors"
        ].items():

            print(
                f"       {actor_id}: "
                f"{actor_report['common_frames']} frames, "
                f"{actor_report['status']}"
            )

        if report[
            "errors"
        ]:

            for error in report[
                "errors"
            ]:

                print(
                    f"       ERROR: "
                    f"{error}"
                )

    # ========================================================
    # Save aggregate report
    # ========================================================

    aggregate = {

        "benchmark_id":
            benchmark_id,

        "tolerance":
            float(
                args.tolerance
            ),

        "cases_checked":
            len(
                cases
            ),

        "passed":
            passed_count,

        "failed":
            failed_count,

        "all_passed":
            (
                failed_count
                == 0
            ),

        "cases":
            reports,
    }

    output_path = (
        m4c_root
        / "m4c_invariant_summary.json"
    )

    save_json(
        output_path,
        aggregate,
    )

    print()
    print(
        "-" * 100
    )

    print(
        "Cases checked:",
        len(
            cases
        ),
    )

    print(
        "Passed:       ",
        passed_count,
    )

    print(
        "Failed:       ",
        failed_count,
    )

    print(
        "Summary:      ",
        output_path,
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Hard gate for future automation.
    #
    # Any mismatch returns a non-zero process status so a future
    # benchmark runner cannot silently continue.
    # --------------------------------------------------------

    if failed_count > 0:

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()