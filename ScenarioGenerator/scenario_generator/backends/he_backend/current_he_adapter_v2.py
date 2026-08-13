"""
current_he_adapter_v2.py

ScenarioGenerator ResolvedScenarioV2 -> current HE scenario JSON.

This adapter is intentionally backend-only.

It does NOT:
    - generate trajectories
    - reason about traffic
    - calculate image-space placement
    - select CARLA assets
    - calculate HE visibility

Its job is only:

    ResolvedScenarioV2
            ↓
    coordinate conversion
            ↓
    exact HE keyframes
            ↓
    actor lifecycle metadata
            ↓
    current HE scenario JSON


Coordinate conventions
----------------------

ScenarioGenerator:

    +x = forward
    +y = left
    +yaw = left / counter-clockwise

Current HE:

    +x = right
    +y = vertical
    +z = forward
    +yaw = right / clockwise

Therefore:

    HE x   = -SG y
    HE y   = 0
    HE z   =  SG x
    HE yaw = -SG yaw


Important
---------

The generated adversaries contain:

    active_start_frame
    active_end_frame

These fields require the lifecycle-aware
run_he_temporal_compositor_v2.py that we will create next.

The old v1 compositor does not enforce actor-specific lifecycle.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedScenarioV2,
)


# ============================================================
# Basic JSON helpers
# ============================================================

def load_he_template(
    template_path: str | Path,
) -> dict[str, Any]:
    """
    Load an existing known-good HE scenario JSON.

    We preserve renderer-specific configuration from this template,
    such as:
        - sprite bank
        - placement lookup
        - rendering options
        - output options
        - calibrated camera intrinsics
    """

    template_path = Path(
        template_path
    )

    with template_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def save_he_json(
    output_path: str | Path,
    data: dict[str, Any],
) -> None:
    """
    Save HE scenario JSON.
    """

    output_path = Path(
        output_path
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
            data,
            f,
            indent=2,
        )

        f.write(
            "\n"
        )


# ============================================================
# Coordinate conversion
# ============================================================

def sg_state_to_he_state(
    x_m: float,
    y_m: float,
    yaw_deg: float,
) -> dict[str, float]:
    """
    Convert ScenarioGenerator physical coordinates into the
    convention used by the current HE compositor.

    ScenarioGenerator:

        x = forward
        y = left
        +yaw = left turn

    HE:

        x = right
        y = vertical
        z = forward
        +yaw = right turn

    Therefore:

        HE x   = -SG y
        HE z   =  SG x
        HE yaw = -SG yaw
    """

    return {
        "x_m":
            -float(
                y_m
            ),

        "y_m":
            0.0,

        "z_m":
            float(
                x_m
            ),

        "yaw_deg":
            -float(
                yaw_deg
            ),
    }


# ============================================================
# ResolvedScenarioV2 helpers
# ============================================================

def _frames_for_actor(
    resolved: ResolvedScenarioV2,
    actor_id: str,
):
    """
    Return the exact physical frames belonging to one actor.
    """

    frames = [

        frame

        for frame
        in resolved.actor_frames

        if (
            frame.actor_id
            == actor_id
        )
    ]

    frames.sort(
        key=lambda frame:
            frame.frame_idx
    )

    return frames


def _actor_info_by_id(
    resolved: ResolvedScenarioV2,
):
    """
    actor_id -> ResolvedActorInfoV2
    """

    return {

        actor.actor_id:
            actor

        for actor
        in resolved.actors
    }


def _validate_actor_frames(
    actor_id: str,
    frames,
):
    """
    An actor may exist for only part of a scenario.

    During that lifecycle its resolved frame sequence must be
    contiguous.
    """

    if not frames:

        raise ValueError(
            f"No resolved frames found "
            f"for actor_id={actor_id!r}"
        )

    frame_ids = [

        int(
            frame.frame_idx
        )

        for frame
        in frames
    ]

    expected = list(
        range(
            frame_ids[0],
            frame_ids[-1] + 1,
        )
    )

    if (
        frame_ids
        != expected
    ):

        raise ValueError(
            f"Actor {actor_id!r} "
            "contains non-contiguous "
            "resolved frames."
        )


# ============================================================
# HE adversary generation
# ============================================================

def _make_keyframed_adversary_from_resolved_v2(
    resolved: ResolvedScenarioV2,
    actor_id: str,
    actor_info,
    template_adversary: dict[str, Any],
) -> dict[str, Any]:
    """
    Export exactly one actor from ResolvedScenarioV2.

    No trajectory approximation occurs here.

    Every resolved actor frame becomes one HE keyframe.
    """

    frames = _frames_for_actor(
        resolved,
        actor_id,
    )

    _validate_actor_frames(
        actor_id,
        frames,
    )

    info = actor_info[
        actor_id
    ]

    adversary = copy.deepcopy(
        template_adversary
    )

    # ========================================================
    # Identity
    # ========================================================

    adversary[
        "id"
    ] = actor_id

    adversary[
        "type"
    ] = (
        info.actor_type.value
    )

    adversary[
        "enabled"
    ] = True

    # ========================================================
    # Actor lifecycle
    # ========================================================
    #
    # These are derived from ACTUAL resolved frames rather than
    # recomputing frame numbers from time.
    #
    # Therefore:
    #
    #     first resolved frame = first rendered-eligible frame
    #     last resolved frame  = final rendered-eligible frame
    #
    # Example crossing actor:
    #
    #     active_start_frame = 60
    #     active_end_frame   = 210
    #
    # ========================================================

    first_frame_idx = int(
        frames[0].frame_idx
    )

    last_frame_idx = int(
        frames[-1].frame_idx
    )

    adversary[
        "active_start_frame"
    ] = first_frame_idx

    adversary[
        "active_end_frame"
    ] = last_frame_idx

    adversary[
        "lifecycle"
    ] = {

        "active_start_frame":
            first_frame_idx,

        "active_end_frame":
            last_frame_idx,

        "spawn_time_s":
            float(
                info.spawn_time_s
            ),

        "despawn_time_s": (
            float(
                info.despawn_time_s
            )
            if (
                info.despawn_time_s
                is not None
            )
            else None
        ),
    }

    # ========================================================
    # Traceability
    # ========================================================

    adversary[
        "source"
    ] = {

        "backend":
            "ScenarioGenerator",

        "schema":
            "ResolvedScenarioV2",

        "actor_id":
            actor_id,

        "actor_type":
            info.actor_type.value,

        "role":
            info.role.value,

        "asset_key":
            info.asset_key,

        "coordinate_frame":
            resolved.coordinate_frame,
    }

    # ========================================================
    # Physical dimensions
    # ========================================================

    if (
        info.dimensions_m
        is not None
    ):

        adversary[
            "size"
        ] = {

            "length_m":
                float(
                    info
                    .dimensions_m
                    .length_m
                ),

            "width_m":
                float(
                    info
                    .dimensions_m
                    .width_m
                ),

            "height_m":
                float(
                    info
                    .dimensions_m
                    .height_m
                ),
        }

    else:

        # Preserve the known-good template values when ScenarioSpec
        # did not provide explicit dimensions.

        adversary.setdefault(
            "size",
            {
                "length_m": 4.5,
                "width_m": 1.8,
                "height_m": 1.6,
            },
        )

    # ========================================================
    # Initial HE state
    # ========================================================

    first = frames[
        0
    ]

    first_he = (
        sg_state_to_he_state(
            x_m=
                first.x_m,

            y_m=
                first.y_m,

            yaw_deg=
                first.yaw_deg,
        )
    )

    adversary[
        "initial_state"
    ] = {

        "x_m":
            first_he[
                "x_m"
            ],

        "y_m":
            first_he[
                "y_m"
            ],

        "z_m":
            first_he[
                "z_m"
            ],

        "yaw_deg":
            first_he[
                "yaw_deg"
            ],
    }

    # ========================================================
    # Exact resolved trajectory
    # ========================================================

    keyframes = []

    for frame in frames:

        he_state = (
            sg_state_to_he_state(
                x_m=
                    frame.x_m,

                y_m=
                    frame.y_m,

                yaw_deg=
                    frame.yaw_deg,
            )
        )

        keyframes.append(
            {
                "frame_idx":
                    int(
                        frame.frame_idx
                    ),

                "t_s":
                    float(
                        frame.t_s
                    ),

                "x_m":
                    he_state[
                        "x_m"
                    ],

                "y_m":
                    he_state[
                        "y_m"
                    ],

                "z_m":
                    he_state[
                        "z_m"
                    ],

                "yaw_deg":
                    he_state[
                        "yaw_deg"
                    ],

                # ------------------------------------------------
                # These are not currently required for sprite
                # placement, but retain physical truth.
                # ------------------------------------------------

                "speed_mps":
                    float(
                        frame.speed_mps
                    ),

                "vx_sg_mps":
                    float(
                        frame.vx_mps
                    ),

                "vy_sg_mps":
                    float(
                        frame.vy_mps
                    ),
            }
        )

    adversary[
        "motion"
    ] = {

        "model":
            "keyframed_trajectory",

        "coordinate_system":
            {
                "x_m":
                    "lateral_right",

                "y_m":
                    "vertical",

                "z_m":
                    "forward",

                "yaw_positive":
                    "right",

                "state_frame":
                    "ego_initial",
            },

        "keyframes":
            keyframes,
    }

    # ========================================================
    # Rendering
    # ========================================================

    rendering = (
        adversary.setdefault(
            "rendering",
            {},
        )
    )

    rendering[
        "alpha"
    ] = float(
        rendering.get(
            "alpha",
            1.0,
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # sprite viewpoint is determined from:
    #
    #     camera->actor bearing
    #              -
    #     actor relative yaw
    #
    # and not directly from trajectory yaw.
    # --------------------------------------------------------

    rendering[
        "angle_mode"
    ] = "viewpoint"

    rendering[
        "angle_smoothing"
    ] = False

    rendering.setdefault(
        "shadow",
        False,
    )

    rendering.setdefault(
        "refiner",
        False,
    )

    # --------------------------------------------------------
    # Remove obsolete angle-selection settings inherited from
    # older templates.
    # --------------------------------------------------------

    rendering.pop(
        "sprite_yaw_sign",
        None,
    )

    rendering.pop(
        "angle_lateral_sign",
        None,
    )

    rendering.pop(
        "angle_lateral_deadzone_ratio",
        None,
    )

    return adversary


# ============================================================
# Main adapter
# ============================================================

def resolved_v2_to_current_he_json(
    resolved: ResolvedScenarioV2,
    template_json: dict[str, Any],
    output_dir: str | None = None,
    output_video_name: str | None = None,
    use_keyframes: bool = True,
    background_video_path: str | None = None,
    ego_pose_path: str | None = None,
) -> dict[str, Any]:
    """
    Convert ResolvedScenarioV2 into a JSON scenario for the current
    HE pipeline.

    Intended architecture:

        ScenarioSpecV2
                ↓
        V2TrajectoryResolver
                ↓
        ResolvedScenarioV2
                ↓
        this adapter
                ↓
        exact HE keyframes
                ↓
        lifecycle-aware HE compositor

    Actor trajectories remain in ego-initial coordinates until the
    HE compositor transforms them using the recorded ego/camera pose
    sequence.
    """

    if not use_keyframes:

        raise ValueError(
            "ResolvedScenarioV2 must be "
            "exported using exact "
            "keyframed trajectories."
        )

    he = copy.deepcopy(
        template_json
    )

    # ========================================================
    # Identity
    # ========================================================

    he[
        "scenario_id"
    ] = resolved.scenario_id

    he[
        "description"
    ] = (
        resolved.source_description
    )

    # ========================================================
    # Input
    # ========================================================

    he.setdefault(
        "input",
        {},
    )

    if (
        background_video_path
        is not None
    ):

        he[
            "input"
        ][
            "video_path"
        ] = background_video_path

    if (
        ego_pose_path
        is not None
    ):

        he[
            "input"
        ][
            "ego_pose_path"
        ] = ego_pose_path

    # ========================================================
    # Output
    # ========================================================

    he.setdefault(
        "output",
        {},
    )

    if (
        output_dir
        is not None
    ):

        he[
            "output"
        ][
            "output_dir"
        ] = output_dir

    if (
        output_video_name
        is not None
    ):

        he[
            "output"
        ][
            "output_video_name"
        ] = output_video_name

    # ========================================================
    # Timeline
    # ========================================================

    ego_frame_indices = [

        int(
            frame.frame_idx
        )

        for frame
        in resolved.ego_frames
    ]

    if not ego_frame_indices:

        raise ValueError(
            "ResolvedScenarioV2 "
            "contains no ego frames."
        )

    ego_frame_indices.sort()

    expected_ego_frames = list(
        range(
            ego_frame_indices[0],
            ego_frame_indices[-1] + 1,
        )
    )

    if (
        ego_frame_indices
        != expected_ego_frames
    ):

        raise ValueError(
            "ResolvedScenarioV2 ego "
            "frames are not contiguous."
        )

    if (
        ego_frame_indices[0]
        != 0
    ):

        raise ValueError(
            "HE v2 export currently expects "
            "the scenario timeline to begin "
            "at frame 0."
        )

    end_frame = (
        ego_frame_indices[
            -1
        ]
    )

    he[
        "timeline"
    ] = {

        "start_frame":
            0,

        "end_frame":
            int(
                end_frame
            ),

        "fps":
            float(
                resolved.fps
            ),
    }

    # ========================================================
    # Coordinate semantics
    # ========================================================

    he[
        "coordinate_mode"
    ] = {

        "actor_state_frame":
            "ego_initial",

        "runtime_transform":
            "ego_pose_jsonl",
    }

    # --------------------------------------------------------
    # Remove legacy synthetic ego motion.
    #
    # Ego movement now comes from the recorded background CARLA
    # camera/ego trajectory.
    # --------------------------------------------------------

    he.pop(
        "ego_motion",
        None,
    )

    # ========================================================
    # Camera
    # ========================================================

    he.setdefault(
        "camera",
        {},
    )

    he[
        "camera"
    ][
        "image_width"
    ] = int(
        resolved
        .camera
        .image_width
    )

    he[
        "camera"
    ][
        "image_height"
    ] = int(
        resolved
        .camera
        .image_height
    )

    he[
        "camera"
    ][
        "fov"
    ] = float(
        resolved
        .camera
        .fov_deg
    )

    # --------------------------------------------------------
    # Current compositor also checks yaw_deg for camera-relative
    # sprite calculations.
    # --------------------------------------------------------

    he[
        "camera"
    ][
        "yaw_deg"
    ] = float(
        resolved
        .camera
        .yaw_deg
    )

    # --------------------------------------------------------
    # Preserve known-good calibrated intrinsics from template.
    #
    # Only create a fallback when none are present.
    # --------------------------------------------------------

    if (
        "intrinsics"
        not in he[
            "camera"
        ]
    ):

        cx = (
            resolved
            .camera
            .image_width
            / 2.0
        )

        cy = (
            resolved
            .camera
            .image_height
            / 2.0
        )

        # Canonical 1280px / 90deg camera:
        # fx = 640px.
        fx = float(
            resolved
            .camera
            .image_width
            / 2.0
        )

        fy = fx

        he[
            "camera"
        ][
            "intrinsics"
        ] = {

            "fx":
                fx,

            "fy":
                fy,

            "cx":
                cx,

            "cy":
                cy,
        }

    # ========================================================
    # Visibility
    # ========================================================

    he.setdefault(
        "visibility",
        {},
    )

    # Maintain the behavior chosen for our corrected paired
    # validation pipeline.
    he[
        "visibility"
    ][
        "min_render_depth_m"
    ] = 0.0

    # ========================================================
    # Adversaries
    # ========================================================

    actor_info = (
        _actor_info_by_id(
            resolved
        )
    )

    template_adversaries = (
        he.get(
            "adversaries",
            [],
        )
    )

    if template_adversaries:

        template_adversary = (
            template_adversaries[
                0
            ]
        )

    else:

        template_adversary = {

            "id":
                "adv_001",

            "type":
                "vehicle",

            "enabled":
                True,

            "size":
                {
                    "length_m": 4.5,
                    "width_m": 1.8,
                    "height_m": 1.6,
                },

            "initial_state":
                {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_m": 30.0,
                    "yaw_deg": 0.0,
                },

            "rendering":
                {
                    "alpha": 1.0,
                    "angle_mode": "viewpoint",
                    "angle_smoothing": False,
                    "shadow": False,
                    "refiner": False,
                },
        }

    adversaries = []

    for actor in resolved.actors:

        adversaries.append(
            _make_keyframed_adversary_from_resolved_v2(

                resolved=
                    resolved,

                actor_id=
                    actor.actor_id,

                actor_info=
                    actor_info,

                template_adversary=
                    template_adversary,
            )
        )

    he[
        "adversaries"
    ] = adversaries

    # ========================================================
    # Traceability
    # ========================================================

    he[
        "scenario_generator_bridge"
    ] = {

        "version":
            "v2",

        "source_schema":
            "ResolvedScenarioV2",

        "source_schema_version":
            resolved.source_schema_version,

        "source_coordinate_frame":
            resolved.coordinate_frame,

        "trajectory_export":
            "exact_keyframed",

        "actor_lifecycle":
            "explicit_frame_range",

        "coordinate_conversion":
            {
                "he_x":
                    "-scenario_y",

                "he_z":
                    "scenario_x",

                "he_yaw":
                    "-scenario_yaw",
            },

        "sprite_selection":
            "viewpoint",

        "required_compositor_feature":
            "per_actor_lifecycle",
    }

    return he


# ============================================================
# Optional compatibility alias
# ============================================================

def resolved_to_current_he_json(
    resolved: ResolvedScenarioV2,
    template_json: dict[str, Any],
    output_dir: str | None = None,
    output_video_name: str | None = None,
    use_keyframes: bool = True,
    background_video_path: str | None = None,
    ego_pose_path: str | None = None,
) -> dict[str, Any]:
    """
    Compatibility alias so callers can retain the previous function
    name while importing from current_he_adapter_v2.py.
    """

    return (
        resolved_v2_to_current_he_json(

            resolved=
                resolved,

            template_json=
                template_json,

            output_dir=
                output_dir,

            output_video_name=
                output_video_name,

            use_keyframes=
                use_keyframes,

            background_video_path=
                background_video_path,

            ego_pose_path=
                ego_pose_path,
        )
    )