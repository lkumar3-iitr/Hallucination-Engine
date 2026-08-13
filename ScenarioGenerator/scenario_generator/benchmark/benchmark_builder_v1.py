"""
benchmark_builder_v1.py

Milestone 4B.

Responsibilities:

    BenchmarkCaseV1
          ↓
    ScenarioSpecV2
          ↓
    V2TrajectoryResolver
          ↓
    ResolvedScenarioV2
          ↓
    HE placement capability analysis

This file DOES NOT execute CARLA or HE.

It also does not modify the V2 trajectory resolver.
"""

from __future__ import annotations

import math
from collections import Counter

from scenario_generator.benchmark.benchmark_spec_v1 import (
    BenchmarkCaseV1,
)

from scenario_generator.schema.scenario_schema_v2 import (
    ActorLifecycleV2,
    ActorSpecV2,
    CameraSpecV2,
    DimensionsV2,
    EgoSpecV2,
    ManeuverMotionV2,
    ManeuverTypeV2,
    PathMotionV2,
    PathWaypointV2,
    Pose2DV2,
    ScenarioSpecV2,
)

from scenario_generator.trajectory.trajectory_resolver_v2 import (
    V2TrajectoryResolver,
)


EPS = 1e-9

DOMAIN_EPS = 1e-5


# ============================================================
# Frozen Paper-1 HE configuration
# ============================================================

HE_DOMAIN_X_MIN = -4.0
HE_DOMAIN_X_MAX = 4.0

HE_DOMAIN_Z_MIN = 0.0
HE_DOMAIN_Z_MAX = 100.0
HE_MIN_RENDER_DEPTH_M = 0.0

SUPPORTED_ASSET_KEYS = {
    "sedan.generic",
}


CANONICAL_CAMERA = {
    "image_width": 1280,
    "image_height": 720,
    "fov_deg": 90.0,

    "x_m": 1.5,
    "y_m": 0.0,
    "z_m": 1.6,

    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
}


# Current validated generic sedan dimensions.
GENERIC_SEDAN_DIMENSIONS = (
    DimensionsV2(
        length_m=4.2,
        width_m=1.8,
        height_m=1.4,
    )
)


# ============================================================
# Small helpers
# ============================================================

def smoothstep(
    u: float,
) -> float:
    """
    Cubic smoothstep.

        3u^2 - 2u^3

    Used only to define the lateral SHAPE of the benchmark path.

    Timing is still determined by geometric arc length and
    PathMotionV2.target_speed_mps.
    """

    u = max(
        0.0,
        min(
            1.0,
            float(u),
        ),
    )

    return (
        3.0 * u * u
        -
        2.0 * u * u * u
    )


def normalize_angle_deg(
    angle_deg: float,
) -> float:

    return (
        (float(angle_deg) + 180.0)
        % 360.0
    ) - 180.0


# ============================================================
# Semantic lateral position
# ============================================================

def resolve_start_lane_y(
    case: BenchmarkCaseV1,
) -> float:
    """
    Resolve the semantic starting side into the ScenarioGenerator
    +y-left coordinate convention.

    Explicit lane_y_m wins.

    If lane_y_m == 0 and a side exists:
        left  -> +lane_width
        right -> -lane_width
    """

    p = case.parameters

    lane_y = float(
        p.lane_y_m
    )

    if (
        abs(lane_y) <= EPS
        and
        p.side is not None
    ):

        if p.side == "left":

            lane_y = float(
                p.lane_width_m
            )

        elif p.side == "right":

            lane_y = -float(
                p.lane_width_m
            )

    # --------------------------------------------------------
    # Detect contradictory semantic requests.
    # --------------------------------------------------------

    if p.side == "left":

        if lane_y < -EPS:

            raise ValueError(
                f"{case.case_id}: "
                "side='left' conflicts with "
                f"lane_y_m={lane_y}. "
                "ScenarioGenerator uses +y=left."
            )

    if p.side == "right":

        if lane_y > EPS:

            raise ValueError(
                f"{case.case_id}: "
                "side='right' conflicts with "
                f"lane_y_m={lane_y}. "
                "ScenarioGenerator uses -y=right."
            )

    return lane_y


# ============================================================
# Exact-duration cut-in path construction
# ============================================================

def build_cutin_waypoints(
    start_x_m: float,
    start_y_m: float,
    target_y_m: float,

    speed_mps: float,

    cut_start_s: float,
    cut_duration_s: float,

    scenario_duration_s: float,

    cut_subdivisions: int = 20,
) -> list[PathWaypointV2]:
    """
    Construct a cut-in as a dense geometric path.

    Important property
    ------------------

    Every lane-change subsegment has path length:

        speed * segment_duration

    Therefore the total lane-change arc length is exactly:

        speed * cut_duration

    and the actor reaches the end of the cut at approximately the
    requested cut_end time under the constant-speed path resolver.

    Lateral displacement follows smoothstep.

    The path itself uses linear interpolation because we already
    construct enough small segments to approximate a smooth curve.
    This avoids changing the underlying resolver and preserves exact
    arc-length timing.
    """

    speed_mps = float(
        speed_mps
    )

    if speed_mps <= 0.0:

        raise ValueError(
            "cut_in requires "
            "actor_speed_mps > 0."
        )

    if cut_subdivisions < 2:

        raise ValueError(
            "cut_subdivisions must be >= 2."
        )

    cut_end_s = (
        float(cut_start_s)
        +
        float(cut_duration_s)
    )

    if (
        cut_end_s
        >
        float(scenario_duration_s)
        + EPS
    ):

        raise ValueError(
            "Cut-in finishes after "
            "scenario duration."
        )

    waypoints = []

    current_x = float(
        start_x_m
    )

    current_y = float(
        start_y_m
    )

    target_y = float(
        target_y_m
    )

    # ========================================================
    # Phase 1:
    # straight motion before lane change
    # ========================================================

    if cut_start_s > EPS:

        current_x += (
            speed_mps
            *
            float(cut_start_s)
        )

        waypoints.append(
            PathWaypointV2(
                x_m=current_x,
                y_m=current_y,
            )
        )

    # ========================================================
    # Phase 2:
    # lane change
    #
    # We prescribe y using smoothstep and solve dx so each
    # subsegment has exactly the desired path length.
    # ========================================================

    segment_duration = (
        float(cut_duration_s)
        /
        float(cut_subdivisions)
    )

    segment_distance = (
        speed_mps
        *
        segment_duration
    )

    lateral_delta = (
        target_y
        -
        float(start_y_m)
    )

    previous_y = current_y

    for i in range(
        1,
        cut_subdivisions + 1,
    ):

        u = (
            i
            /
            float(cut_subdivisions)
        )

        next_y = (
            float(start_y_m)
            +
            lateral_delta
            *
            smoothstep(u)
        )

        dy = (
            next_y
            -
            previous_y
        )

        # ----------------------------------------------------
        # A segment cannot move farther laterally than its
        # entire distance budget.
        # ----------------------------------------------------

        if (
            abs(dy)
            >
            segment_distance
            + EPS
        ):

            raise ValueError(
                "Requested cut-in is physically "
                "incompatible with the requested "
                "speed/duration. "
                f"segment_distance="
                f"{segment_distance:.4f} m, "
                f"required lateral displacement="
                f"{abs(dy):.4f} m. "
                "Increase actor_speed_mps or "
                "cut_duration_s."
            )

        dx_squared = (
            segment_distance
            * segment_distance
            -
            dy * dy
        )

        dx = math.sqrt(
            max(
                0.0,
                dx_squared,
            )
        )

        current_x += dx
        current_y = next_y

        waypoints.append(
            PathWaypointV2(
                x_m=current_x,
                y_m=current_y,
            )
        )

        previous_y = next_y

    # ========================================================
    # Phase 3:
    # continue straight after lane change
    # ========================================================

    remaining_s = (
        float(scenario_duration_s)
        -
        cut_end_s
    )

    if remaining_s > EPS:

        current_x += (
            speed_mps
            *
            remaining_s
        )

        waypoints.append(
            PathWaypointV2(
                x_m=current_x,
                y_m=target_y,
            )
        )

    return waypoints


# ============================================================
# BenchmarkCaseV1 -> ScenarioSpecV2
# ============================================================

def benchmark_case_to_scenario_v2(
    case: BenchmarkCaseV1,
) -> ScenarioSpecV2:

    p = case.parameters

    # ========================================================
    # Common ego
    # ========================================================

    ego = EgoSpecV2(

        actor_id="ego",

        spawn=Pose2DV2(
            x_m=0.0,
            y_m=0.0,
            yaw_deg=0.0,
        ),

        motion=ManeuverMotionV2(
            maneuver=(
                ManeuverTypeV2.STRAIGHT
            ),

            speed_mps=float(
                p.ego_speed_mps
            ),

            start_time_s=0.0,
        ),
    )

    # ========================================================
    # CUT-IN
    # ========================================================

    if case.scenario_type == "cut_in":

        start_y_m = (
            resolve_start_lane_y(
                case
            )
        )

        target_y_m = float(
            p.target_lane_y_m
        )

        waypoints = (
            build_cutin_waypoints(
                start_x_m=float(
                    p.start_distance_m
                ),

                start_y_m=start_y_m,

                target_y_m=target_y_m,

                speed_mps=float(
                    p.actor_speed_mps
                ),

                cut_start_s=float(
                    p.cut_start_s
                ),

                cut_duration_s=float(
                    p.cut_duration_s
                ),

                scenario_duration_s=float(
                    p.duration_s
                ),
            )
        )

        adversary_motion = PathMotionV2(

            target_speed_mps=float(
                p.actor_speed_mps
            ),

            interpolation="linear",

            waypoints=waypoints,
        )

        adversary_spawn = Pose2DV2(
            x_m=float(
                p.start_distance_m
            ),

            y_m=start_y_m,

            yaw_deg=0.0,
        )

    # ========================================================
    # ONCOMING
    # ========================================================

    elif case.scenario_type == "oncoming":

        start_y_m = float(
            p.lane_y_m
        )

        adversary_spawn = Pose2DV2(
            x_m=float(
                p.start_distance_m
            ),

            y_m=start_y_m,

            # Facing toward ego.
            yaw_deg=180.0,
        )

        adversary_motion = ManeuverMotionV2(

            maneuver=(
                ManeuverTypeV2.ONCOMING
            ),

            speed_mps=float(
                p.actor_speed_mps
            ),

            start_time_s=0.0,
        )

    else:

        raise NotImplementedError(
            "M4B currently supports "
            "'cut_in' and 'oncoming'. "
            f"Received {case.scenario_type!r}."
        )

    # ========================================================
    # Common adversary
    # ========================================================

    adversary = ActorSpecV2(

        actor_id="adv_001",

        actor_type="vehicle",

        role="adversary",

        asset_key="sedan.generic",

        dimensions_m=(
            GENERIC_SEDAN_DIMENSIONS
        ),

        spawn=adversary_spawn,

        lifecycle=ActorLifecycleV2(
            spawn_time_s=0.0,
            despawn_time_s=None,
        ),

        motion=adversary_motion,
    )

    description = (
        f"Benchmark case {case.case_id} "
        f"from {case.benchmark_id}. "
        f"Swept values: "
        f"{case.swept_values}"
    )

    return ScenarioSpecV2(

        schema_version="2.0",

        scenario_id=case.case_id,

        description=description,

        duration_s=float(
            p.duration_s
        ),

        fps=int(
            p.fps
        ),

        seed=int(
            case.case_index
        ),

        ego=ego,

        actors=[
            adversary
        ],

        camera=CameraSpecV2(),
    )


# ============================================================
# Current-camera-relative geometry
# ============================================================

def actor_to_current_camera_state(
    actor_frame,
    ego_frame,
    camera,
):
    """
    Transform ScenarioGenerator ego-initial coordinates into the
    CURRENT camera coordinate frame.

    ScenarioGenerator:
        +x = forward
        +y = left
        +yaw = left / CCW

    HE placement state:
        x = lateral RIGHT
        z = forward
        yaw positive RIGHT
    """

    ego_yaw_rad = math.radians(
        float(
            ego_frame.yaw_deg
        )
    )

    # --------------------------------------------------------
    # Camera position in ScenarioGenerator physical coordinates.
    #
    # Camera mount translation rotates with ego.
    # --------------------------------------------------------

    camera_world_x = (
        float(
            ego_frame.x_m
        )
        +
        float(
            camera.x_m
        )
        * math.cos(
            ego_yaw_rad
        )
        -
        float(
            camera.y_m
        )
        * math.sin(
            ego_yaw_rad
        )
    )

    camera_world_y = (
        float(
            ego_frame.y_m
        )
        +
        float(
            camera.x_m
        )
        * math.sin(
            ego_yaw_rad
        )
        +
        float(
            camera.y_m
        )
        * math.cos(
            ego_yaw_rad
        )
    )

    camera_yaw_deg = (
        float(
            ego_frame.yaw_deg
        )
        +
        float(
            camera.yaw_deg
        )
    )

    camera_yaw_rad = math.radians(
        camera_yaw_deg
    )

    dx_world = (
        float(
            actor_frame.x_m
        )
        -
        camera_world_x
    )

    dy_world = (
        float(
            actor_frame.y_m
        )
        -
        camera_world_y
    )

    # Camera forward axis.
    rel_forward = (
        dx_world
        * math.cos(
            camera_yaw_rad
        )
        +
        dy_world
        * math.sin(
            camera_yaw_rad
        )
    )

    # Camera LEFT axis in SG convention.
    rel_left = (
        -dx_world
        * math.sin(
            camera_yaw_rad
        )
        +
        dy_world
        * math.cos(
            camera_yaw_rad
        )
    )

    # HE uses lateral RIGHT.
    he_x_m = -rel_left

    he_z_m = rel_forward

    relative_yaw_sg = (
        normalize_angle_deg(
            float(
                actor_frame.yaw_deg
            )
            -
            camera_yaw_deg
        )
    )

    # HE yaw-positive convention is opposite SG.
    he_yaw_deg = (
        normalize_angle_deg(
            -relative_yaw_sg
        )
    )

    return {
        "x_m": he_x_m,
        "z_m": he_z_m,
        "yaw_deg": he_yaw_deg,
    }


# ============================================================
# Placement-domain check
# ============================================================

def classify_he_frame(
    x_m: float,
    z_m: float,

    x_min: float = HE_DOMAIN_X_MIN,
    x_max: float = HE_DOMAIN_X_MAX,

    z_max: float = HE_DOMAIN_Z_MAX,

    min_render_depth_m: float = HE_MIN_RENDER_DEPTH_M,

    epsilon: float = DOMAIN_EPS,
):
    """
    Classify one current-camera-relative actor state.

    Returns:
        state, reason

    States
    ------
    CULLED
        Actor physically exists, but HE does not need placement
        because it is too close to or behind the camera.

    SUPPORTED
        Actor should be rendered and lies inside the validated
        HE placement domain.

    OOD
        Actor should be rendered, but its placement state lies
        outside the validated lookup domain.
    """

    # --------------------------------------------------------
    # Visibility / camera relevance comes FIRST.
    #
    # This is not an HE placement failure.
    # --------------------------------------------------------

    if (
        z_m
        <=
        min_render_depth_m + epsilon
    ):
        return (
            "CULLED",
            "behind_or_too_close_to_camera",
        )

    # --------------------------------------------------------
    # Placement is required from here onward.
    # --------------------------------------------------------

    reasons = []

    if (
        x_m
        <
        x_min - epsilon
    ):
        reasons.append(
            "x_below_min"
        )

    if (
        x_m
        >
        x_max + epsilon
    ):
        reasons.append(
            "x_above_max"
        )

    if (
        z_m
        >
        z_max + epsilon
    ):
        reasons.append(
            "z_above_max"
        )

    if reasons:
        return (
            "OOD",
            "+".join(
                reasons
            ),
        )

    return (
        "SUPPORTED",
        "in_domain",
    )

# ============================================================
# Camera compatibility
# ============================================================

def analyze_camera_support(
    camera,
):
    reasons = []

    exact_integer_fields = [
        "image_width",
        "image_height",
    ]

    float_fields = [
        "fov_deg",

        "x_m",
        "y_m",
        "z_m",

        "pitch_deg",
        "yaw_deg",
        "roll_deg",
    ]

    for name in exact_integer_fields:

        actual = int(
            getattr(
                camera,
                name,
            )
        )

        expected = int(
            CANONICAL_CAMERA[
                name
            ]
        )

        if actual != expected:

            reasons.append(
                f"{name}:"
                f"{actual}!="
                f"{expected}"
            )

    for name in float_fields:

        actual = float(
            getattr(
                camera,
                name,
            )
        )

        expected = float(
            CANONICAL_CAMERA[
                name
            ]
        )

        if (
            abs(
                actual - expected
            )
            >
            1e-6
        ):

            reasons.append(
                f"{name}:"
                f"{actual}!="
                f"{expected}"
            )

    return {
        "supported":
            len(
                reasons
            ) == 0,

        "reasons":
            reasons,
    }


# ============================================================
# Full HE capability analysis
# ============================================================

def analyze_he_capability(
    resolved,
):
    """
    Analyze the resolved physical scenario against the current
    HE rendering/placement capability.

    Important distinction:

        physical actor existence
                !=
        camera visibility
                !=
        placement capability

    A frame can therefore be:

        SUPPORTED
            placement required and inside lookup domain

        CULLED
            actor exists physically, but is too close to or
            behind the camera; no placement lookup is required

        OOD
            placement is required but the state is outside the
            validated HE placement domain
    """

    ego_by_frame = {
        int(frame.frame_idx):
            frame

        for frame
        in resolved.ego_frames
    }

    camera_support = (
        analyze_camera_support(
            resolved.camera
        )
    )

    actor_reports = {}

    total_actor_frames = 0

    total_placement_required = 0
    total_supported_placement = 0

    total_culled = 0
    total_ood = 0

    asset_reasons = []

    # ========================================================
    # Static asset support
    # ========================================================

    for actor_info in resolved.actors:

        asset_key = (
            actor_info.asset_key
        )

        if (
            asset_key
            not in SUPPORTED_ASSET_KEYS
        ):

            asset_reasons.append(
                f"{actor_info.actor_id}:"
                f"unsupported_asset:"
                f"{asset_key}"
            )

    # ========================================================
    # Group actor frames
    # ========================================================

    grouped = {}

    for frame in resolved.actor_frames:

        grouped.setdefault(
            frame.actor_id,
            [],
        ).append(
            frame
        )

    # ========================================================
    # Analyze actor by actor
    # ========================================================

    for actor_id, frames in (
        grouped.items()
    ):

        frames = sorted(
            frames,
            key=lambda x:
                x.frame_idx,
        )

        supported_count = 0
        culled_count = 0
        ood_count = 0

        reason_counts = Counter()

        rel_x_values = []
        rel_z_values = []

        first_culled_frame = None
        first_ood_frame = None

        for actor_frame in frames:

            frame_idx = int(
                actor_frame.frame_idx
            )

            ego_frame = (
                ego_by_frame.get(
                    frame_idx
                )
            )

            if ego_frame is None:

                raise ValueError(
                    f"Missing ego frame "
                    f"{frame_idx} while "
                    f"analyzing actor "
                    f"{actor_id}."
                )

            relative = (
                actor_to_current_camera_state(
                    actor_frame=
                        actor_frame,

                    ego_frame=
                        ego_frame,

                    camera=
                        resolved.camera,
                )
            )

            rel_x = float(
                relative["x_m"]
            )

            rel_z = float(
                relative["z_m"]
            )

            rel_yaw = float(
                relative["yaw_deg"]
            )

            rel_x_values.append(
                rel_x
            )

            rel_z_values.append(
                rel_z
            )

            (
                frame_state,
                reason,
            ) = classify_he_frame(
                x_m=rel_x,
                z_m=rel_z,
            )

            # ------------------------------------------------
            # Supported placement
            # ------------------------------------------------

            if (
                frame_state
                == "SUPPORTED"
            ):

                supported_count += 1

            # ------------------------------------------------
            # Visibility cull
            # ------------------------------------------------

            elif (
                frame_state
                == "CULLED"
            ):

                culled_count += 1

                reason_counts[
                    reason
                ] += 1

                if (
                    first_culled_frame
                    is None
                ):

                    first_culled_frame = {
                        "frame_idx":
                            frame_idx,

                        "t_s":
                            float(
                                actor_frame.t_s
                            ),

                        "x_m":
                            rel_x,

                        "z_m":
                            rel_z,

                        "yaw_deg":
                            rel_yaw,

                        "reason":
                            reason,
                    }

            # ------------------------------------------------
            # True placement OOD
            # ------------------------------------------------

            elif (
                frame_state
                == "OOD"
            ):

                ood_count += 1

                reason_counts[
                    reason
                ] += 1

                if (
                    first_ood_frame
                    is None
                ):

                    first_ood_frame = {
                        "frame_idx":
                            frame_idx,

                        "t_s":
                            float(
                                actor_frame.t_s
                            ),

                        "x_m":
                            rel_x,

                        "z_m":
                            rel_z,

                        "yaw_deg":
                            rel_yaw,

                        "reason":
                            reason,
                    }

            else:

                raise ValueError(
                    "Unknown HE capability "
                    f"state: {frame_state}"
                )

        active_frames = len(
            frames
        )

        placement_required = (
            supported_count
            +
            ood_count
        )

        total_actor_frames += (
            active_frames
        )

        total_placement_required += (
            placement_required
        )

        total_supported_placement += (
            supported_count
        )

        total_culled += (
            culled_count
        )

        total_ood += (
            ood_count
        )

        actor_reports[
            actor_id
        ] = {

            # -----------------------------------------------
            # Physical existence
            # -----------------------------------------------

            "active_frames":
                active_frames,

            # -----------------------------------------------
            # Camera / placement semantics
            # -----------------------------------------------

            "placement_required_frames":
                placement_required,

            "supported_placement_frames":
                supported_count,

            "culled_frames":
                culled_count,

            "out_of_domain_frames":
                ood_count,

            # -----------------------------------------------
            # Fractions
            # -----------------------------------------------

            "supported_placement_fraction":
                (
                    supported_count
                    /
                    placement_required
                    if placement_required
                    else 1.0
                ),

            "culled_fraction":
                (
                    culled_count
                    /
                    active_frames
                    if active_frames
                    else 0.0
                ),

            # -----------------------------------------------
            # Camera-relative extent
            # -----------------------------------------------

            "rel_x_min_m":
                min(
                    rel_x_values
                )
                if rel_x_values
                else None,

            "rel_x_max_m":
                max(
                    rel_x_values
                )
                if rel_x_values
                else None,

            "rel_z_min_m":
                min(
                    rel_z_values
                )
                if rel_z_values
                else None,

            "rel_z_max_m":
                max(
                    rel_z_values
                )
                if rel_z_values
                else None,

            # -----------------------------------------------
            # Diagnostics
            # -----------------------------------------------

            "reason_counts":
                dict(
                    reason_counts
                ),

            "first_culled_frame":
                first_culled_frame,

            "first_ood_frame":
                first_ood_frame,
        }

    # ========================================================
    # Final scenario support
    # ========================================================

    camera_ok = bool(
        camera_support[
            "supported"
        ]
    )

    assets_ok = (
        len(
            asset_reasons
        )
        == 0
    )

    placement_ok = (
        total_ood
        == 0
    )

    supported = (
        camera_ok
        and
        assets_ok
        and
        placement_ok
    )

    status = (
        "HE_SUPPORTED"
        if supported
        else "HE_OOD"
    )

    return {

        "scenario_id":
            resolved.scenario_id,

        "status":
            status,

        "supported":
            supported,

        "camera_support":
            camera_support,

        "asset_support": {

            "supported":
                assets_ok,

            "reasons":
                asset_reasons,
        },

        "visibility_policy": {

            "min_render_depth_m":
                HE_MIN_RENDER_DEPTH_M,

            "culled_when":
                "z_m <= min_render_depth_m",
        },

        "placement_domain": {

            "x_min_m":
                HE_DOMAIN_X_MIN,

            "x_max_m":
                HE_DOMAIN_X_MAX,

            "z_max_m":
                HE_DOMAIN_Z_MAX,

            "epsilon":
                DOMAIN_EPS,

            "yaw":
                "full_360",
        },

        "total_actor_frames":
            total_actor_frames,

        "placement_required_frames":
            total_placement_required,

        "supported_placement_frames":
            total_supported_placement,

        "culled_frames":
            total_culled,

        "out_of_domain_frames":
            total_ood,

        "actors":
            actor_reports,
    }
# ============================================================
# Complete M4B operation for one case
# ============================================================

def resolve_benchmark_case(
    case: BenchmarkCaseV1,
):
    """
    Convenience function:

        case
          ↓
        ScenarioSpecV2
          ↓
        ResolvedScenarioV2
          ↓
        HE capability report
    """

    scenario = (
        benchmark_case_to_scenario_v2(
            case
        )
    )

    resolver = (
        V2TrajectoryResolver()
    )

    resolved = (
        resolver.resolve(
            scenario
        )
    )

    capability = (
        analyze_he_capability(
            resolved
        )
    )

    return (
        scenario,
        resolved,
        capability,
    )