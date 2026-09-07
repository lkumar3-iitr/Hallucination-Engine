"""
trajectory_resolver_v2.py

Resolve ScenarioSpecV2 into exact per-frame physical states.

Supported in Milestone 2:

    maneuver:
        static
        straight
        following
        oncoming

    path:
        linear
        smooth

    keyframes:
        linear
        smooth

Actor lifecycle:
    before spawn  -> no actor frame
    during life   -> resolved actor frame
    after despawn -> no actor frame

No CARLA or HE knowledge belongs here.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from scenario_generator.schema.resolved_schema_v2 import (
    ResolvedActorFrameV2,
    ResolvedActorInfoV2,
    ResolvedEgoFrameV2,
    ResolvedScenarioV2,
)

from scenario_generator.schema.scenario_schema_v2 import (
    ActorSpecV2,
    EgoSpecV2,
    KeyframeMotionV2,
    ManeuverMotionV2,
    ManeuverTypeV2,
    PathMotionV2,
    Pose2DV2,
    ScenarioSpecV2,
    AccelerateStepV2,
    BrakeStepV2,
    CruiseStepV2,
    HoldStepV2,
    LaneChangeStepV2,
    SequenceMotionV2,
)


EPS = 1e-9


# ============================================================
# Internal state
# ============================================================

@dataclass
class MotionState:
    x_m: float
    y_m: float
    yaw_deg: float

    speed_mps: float
    vx_mps: float
    vy_mps: float


# ============================================================
# Angle helpers
# ============================================================

def normalize_yaw_deg(
    angle_deg: float,
) -> float:
    """
    Normalize yaw to [-180, 180).
    """

    return (
        (angle_deg + 180.0)
        % 360.0
    ) - 180.0


def shortest_angle_delta_deg(
    from_deg: float,
    to_deg: float,
) -> float:
    return normalize_yaw_deg(
        to_deg - from_deg
    )


def interpolate_yaw_deg(
    yaw0: float,
    yaw1: float,
    u: float,
) -> float:
    delta = shortest_angle_delta_deg(
        yaw0,
        yaw1,
    )

    return normalize_yaw_deg(
        yaw0 + u * delta
    )


# ============================================================
# Interpolation helpers
# ============================================================

def smoothstep(u: float) -> float:
    """
    Cubic smoothstep:
        3u^2 - 2u^3
    """

    u = max(
        0.0,
        min(
            1.0,
            u,
        ),
    )

    return (
        3.0 * u * u
        -
        2.0 * u * u * u
    )


def smoothstep_derivative(
    u: float,
) -> float:
    """
    d/du smoothstep(u)
        = 6u(1-u)
    """

    u = max(
        0.0,
        min(
            1.0,
            u,
        ),
    )

    return (
        6.0
        * u
        * (1.0 - u)
    )


# ============================================================
# Shape-preserving smooth geometric path
# ============================================================

def remove_duplicate_points(
    points,
):
    """
    Remove consecutive duplicate physical points.
    """

    if not points:
        return []

    output = [
        points[0]
    ]

    for point in points[1:]:
        previous = output[-1]

        distance = math.hypot(
            point[0] - previous[0],
            point[1] - previous[1],
        )

        if distance > EPS:
            output.append(
                point
            )

    return output


def pchip_slopes(
    parameter,
    values,
):
    """
    Compute shape-preserving cubic Hermite slopes.

    This is the Fritsch-Carlson / PCHIP-style slope rule.

    Important property for our path resolver:

        if the supplied values vary monotonically between
        control points, the interpolated curve does not create
        artificial overshoot between those points.

    We apply this independently to x(s) and y(s), where s is
    cumulative chord length along the supplied path.
    """

    n = len(
        parameter
    )

    if n != len(values):
        raise ValueError(
            "parameter/value length mismatch"
        )

    if n < 2:
        raise ValueError(
            "At least two points are required."
        )

    # --------------------------------------------------------
    # Two points -> simple linear slope
    # --------------------------------------------------------

    if n == 2:
        h = (
            parameter[1]
            - parameter[0]
        )

        if h <= EPS:
            raise ValueError(
                "Invalid zero-length path interval."
            )

        delta = (
            values[1]
            - values[0]
        ) / h

        return [
            delta,
            delta,
        ]

    # --------------------------------------------------------
    # Interval widths and secant slopes
    # --------------------------------------------------------

    h = []

    delta = []

    for i in range(
        n - 1
    ):
        hi = (
            parameter[i + 1]
            - parameter[i]
        )

        if hi <= EPS:
            raise ValueError(
                "Path parameter must be "
                "strictly increasing."
            )

        h.append(
            hi
        )

        delta.append(
            (
                values[i + 1]
                - values[i]
            )
            / hi
        )

    slopes = [
        0.0
    ] * n

    # --------------------------------------------------------
    # Interior slopes
    # --------------------------------------------------------

    for i in range(
        1,
        n - 1
    ):

        previous_delta = (
            delta[i - 1]
        )

        next_delta = (
            delta[i]
        )

        # Local extremum or flat interval:
        # force zero derivative to prevent overshoot.
        if (
            abs(previous_delta) <= EPS
            or abs(next_delta) <= EPS
            or (
                previous_delta
                * next_delta
            ) <= 0.0
        ):
            slopes[i] = 0.0
            continue

        w1 = (
            2.0 * h[i]
            + h[i - 1]
        )

        w2 = (
            h[i]
            + 2.0 * h[i - 1]
        )

        slopes[i] = (
            (w1 + w2)
            /
            (
                w1 / previous_delta
                +
                w2 / next_delta
            )
        )

    # --------------------------------------------------------
    # First endpoint
    # --------------------------------------------------------

    first_slope = (
        (
            2.0 * h[0]
            + h[1]
        )
        * delta[0]
        -
        h[0]
        * delta[1]
    ) / (
        h[0]
        + h[1]
    )

    if (
        first_slope
        * delta[0]
    ) <= 0.0:

        first_slope = 0.0

    elif (
        delta[0]
        * delta[1]
        < 0.0
        and abs(first_slope)
        > 3.0 * abs(delta[0])
    ):
        first_slope = (
            3.0
            * delta[0]
        )

    slopes[0] = (
        first_slope
    )

    # --------------------------------------------------------
    # Last endpoint
    # --------------------------------------------------------

    last_slope = (
        (
            2.0 * h[-1]
            + h[-2]
        )
        * delta[-1]
        -
        h[-1]
        * delta[-2]
    ) / (
        h[-1]
        + h[-2]
    )

    if (
        last_slope
        * delta[-1]
    ) <= 0.0:

        last_slope = 0.0

    elif (
        delta[-1]
        * delta[-2]
        < 0.0
        and abs(last_slope)
        > 3.0 * abs(delta[-1])
    ):
        last_slope = (
            3.0
            * delta[-1]
        )

    slopes[-1] = (
        last_slope
    )

    return slopes


def cubic_hermite_value(
    value0: float,
    value1: float,
    slope0: float,
    slope1: float,
    interval_length: float,
    u: float,
) -> float:
    """
    Evaluate one cubic Hermite segment.

    u:
        normalized segment parameter in [0, 1]

    slopes:
        derivatives with respect to the global path parameter.
    """

    u = max(
        0.0,
        min(
            1.0,
            u,
        ),
    )

    u2 = u * u
    u3 = u2 * u

    h00 = (
        2.0 * u3
        - 3.0 * u2
        + 1.0
    )

    h10 = (
        u3
        - 2.0 * u2
        + u
    )

    h01 = (
        -2.0 * u3
        + 3.0 * u2
    )

    h11 = (
        u3
        - u2
    )

    return (
        h00 * value0
        +
        h10
        * interval_length
        * slope0
        +
        h01 * value1
        +
        h11
        * interval_length
        * slope1
    )


def build_shape_preserving_smooth_path(
    control_points,
    subdivisions: int = 40,
):
    """
    Build a smooth parametric path through all control points.

    Parameterization:
        cumulative chord length.

    Interpolation:
        shape-preserving cubic Hermite interpolation
        independently for x(s) and y(s).

    Advantages over the previous uniform Catmull-Rom curve:

        - passes through every supplied waypoint
        - smooth position curve
        - avoids artificial coordinate overshoot
        - preserves horizontal sections correctly
        - works without SciPy
    """

    control_points = (
        remove_duplicate_points(
            control_points
        )
    )

    if len(control_points) < 2:
        raise ValueError(
            "Path must contain at least two "
            "distinct physical points."
        )

    # --------------------------------------------------------
    # Cumulative chord-length parameter
    # --------------------------------------------------------

    parameter = [
        0.0
    ]

    for i in range(
        1,
        len(control_points),
    ):
        p0 = control_points[
            i - 1
        ]

        p1 = control_points[
            i
        ]

        distance = math.hypot(
            p1[0] - p0[0],
            p1[1] - p0[1],
        )

        parameter.append(
            parameter[-1]
            + distance
        )

    x_values = [
        point[0]
        for point in control_points
    ]

    y_values = [
        point[1]
        for point in control_points
    ]

    x_slopes = pchip_slopes(
        parameter,
        x_values,
    )

    y_slopes = pchip_slopes(
        parameter,
        y_values,
    )

    # --------------------------------------------------------
    # Sample each cubic segment densely.
    # Arc-length timing is still assigned later.
    # --------------------------------------------------------

    dense = []

    for i in range(
        len(control_points) - 1
    ):

        interval_length = (
            parameter[i + 1]
            - parameter[i]
        )

        for j in range(
            subdivisions
        ):
            u = (
                j
                / subdivisions
            )

            x = cubic_hermite_value(
                x_values[i],
                x_values[i + 1],
                x_slopes[i],
                x_slopes[i + 1],
                interval_length,
                u,
            )

            y = cubic_hermite_value(
                y_values[i],
                y_values[i + 1],
                y_slopes[i],
                y_slopes[i + 1],
                interval_length,
                u,
            )

            dense.append(
                (
                    x,
                    y,
                )
            )

    # Preserve final waypoint exactly.
    dense.append(
        control_points[-1]
    )

    return remove_duplicate_points(
        dense
    )


def build_dense_path(
    spawn: Pose2DV2,
    motion: PathMotionV2,
):
    """
    Convert spawn + supplied waypoints into a dense geometric path.

    linear:
        exact piecewise-linear path.

    smooth:
        shape-preserving cubic path through all waypoints.
    """

    control_points = [
        (
            float(spawn.x_m),
            float(spawn.y_m),
        )
    ]

    control_points.extend(
        (
            float(point.x_m),
            float(point.y_m),
        )
        for point in motion.waypoints
    )

    control_points = (
        remove_duplicate_points(
            control_points
        )
    )

    if len(control_points) < 2:
        raise ValueError(
            "Path must contain at least two "
            "distinct physical points."
        )

    if (
        motion.interpolation
        == "linear"
    ):
        return control_points

    return (
        build_shape_preserving_smooth_path(
            control_points
        )
    )
# ============================================================
# Arc-length path representation
# ============================================================

@dataclass
class ArcLengthPath:
    points: list
    cumulative_m: list
    total_length_m: float


def build_arc_length_path(
    spawn: Pose2DV2,
    motion: PathMotionV2,
) -> ArcLengthPath:

    points = build_dense_path(
        spawn,
        motion,
    )

    cumulative = [
        0.0
    ]

    total = 0.0

    for i in range(
        1,
        len(points),
    ):
        p0 = points[i - 1]
        p1 = points[i]

        total += math.hypot(
            p1[0] - p0[0],
            p1[1] - p0[1],
        )

        cumulative.append(
            total
        )

    if total <= EPS:
        raise ValueError(
            "Path length is zero."
        )

    return ArcLengthPath(
        points=points,
        cumulative_m=cumulative,
        total_length_m=total,
    )


def state_on_arc_path(
    path: ArcLengthPath,
    distance_m: float,
    target_speed_mps: float,
) -> MotionState:

    # --------------------------------------------------------
    # End of path: hold final position
    # --------------------------------------------------------

    if (
        distance_m
        >= path.total_length_m - EPS
    ):
        p1 = path.points[-1]
        p0 = path.points[-2]

        yaw = math.degrees(
            math.atan2(
                p1[1] - p0[1],
                p1[0] - p0[0],
            )
        )

        return MotionState(
            x_m=p1[0],
            y_m=p1[1],
            yaw_deg=normalize_yaw_deg(
                yaw
            ),
            speed_mps=0.0,
            vx_mps=0.0,
            vy_mps=0.0,
        )

    distance_m = max(
        0.0,
        distance_m,
    )

    segment = (
        bisect.bisect_right(
            path.cumulative_m,
            distance_m,
        )
        - 1
    )

    segment = max(
        0,
        min(
            segment,
            len(path.points) - 2,
        ),
    )

    d0 = path.cumulative_m[
        segment
    ]

    d1 = path.cumulative_m[
        segment + 1
    ]

    p0 = path.points[
        segment
    ]

    p1 = path.points[
        segment + 1
    ]

    segment_length = (
        d1 - d0
    )

    if segment_length <= EPS:
        u = 0.0
    else:
        u = (
            distance_m - d0
        ) / segment_length

    x = (
        p0[0]
        +
        u * (
            p1[0] - p0[0]
        )
    )

    y = (
        p0[1]
        +
        u * (
            p1[1] - p0[1]
        )
    )

    dx = (
        p1[0] - p0[0]
    )

    dy = (
        p1[1] - p0[1]
    )

    yaw_rad = math.atan2(
        dy,
        dx,
    )

    yaw_deg = math.degrees(
        yaw_rad
    )

    vx = (
        target_speed_mps
        * math.cos(
            yaw_rad
        )
    )

    vy = (
        target_speed_mps
        * math.sin(
            yaw_rad
        )
    )

    return MotionState(
        x_m=x,
        y_m=y,
        yaw_deg=normalize_yaw_deg(
            yaw_deg
        ),
        speed_mps=(
            target_speed_mps
        ),
        vx_mps=vx,
        vy_mps=vy,
    )


# ============================================================
# Maneuver resolver
# ============================================================

def resolve_maneuver_state(
    spawn: Pose2DV2,
    motion: ManeuverMotionV2,
    t_s: float,
) -> MotionState:

    if (
        motion.maneuver
        == ManeuverTypeV2.STATIC
    ):
        return MotionState(
            x_m=spawn.x_m,
            y_m=spawn.y_m,
            yaw_deg=spawn.yaw_deg,
            speed_mps=0.0,
            vx_mps=0.0,
            vy_mps=0.0,
        )

    supported_straight = {
        ManeuverTypeV2.STRAIGHT,
        ManeuverTypeV2.FOLLOWING,
        ManeuverTypeV2.ONCOMING,
    }

    if (
        motion.maneuver
        not in supported_straight
    ):
        raise NotImplementedError(
            "ScenarioSchema v2 maneuver "
            f"'{motion.maneuver.value}' "
            "is not resolved directly yet. "
            "Use path or keyframes for "
            "complex v2 motion."
        )

    # --------------------------------------------------------
    # Wait until maneuver start
    # --------------------------------------------------------

    if (
        t_s
        <= motion.start_time_s
    ):
        elapsed = 0.0
        moving = False
    else:
        elapsed = (
            t_s
            - motion.start_time_s
        )

        moving = True

    # --------------------------------------------------------
    # Optional maneuver duration
    # --------------------------------------------------------

    if (
        motion.duration_s
        is not None
    ):
        if (
            elapsed
            >= motion.duration_s
        ):
            elapsed = (
                motion.duration_s
            )

            moving = False

    yaw_rad = math.radians(
        spawn.yaw_deg
    )

    distance = (
        motion.speed_mps
        * elapsed
    )

    x = (
        spawn.x_m
        +
        distance
        * math.cos(
            yaw_rad
        )
    )

    y = (
        spawn.y_m
        +
        distance
        * math.sin(
            yaw_rad
        )
    )

    speed = (
        motion.speed_mps
        if moving
        else 0.0
    )

    vx = (
        speed
        * math.cos(
            yaw_rad
        )
    )

    vy = (
        speed
        * math.sin(
            yaw_rad
        )
    )

    return MotionState(
        x_m=x,
        y_m=y,
        yaw_deg=normalize_yaw_deg(
            spawn.yaw_deg
        ),
        speed_mps=speed,
        vx_mps=vx,
        vy_mps=vy,
    )

# ============================================================
# Sequence motion resolver
# ============================================================

def _state_with_speed(
    x_m: float,
    y_m: float,
    yaw_deg: float,
    speed_mps: float,
) -> MotionState:
    yaw_rad = math.radians(yaw_deg)

    return MotionState(
        x_m=x_m,
        y_m=y_m,
        yaw_deg=normalize_yaw_deg(yaw_deg),
        speed_mps=speed_mps,
        vx_mps=speed_mps * math.cos(yaw_rad),
        vy_mps=speed_mps * math.sin(yaw_rad),
    )


def _sequence_step_duration(
    step,
    start_speed_mps: float,
) -> float:

    if isinstance(step, CruiseStepV2):
        return step.duration_s

    if isinstance(step, LaneChangeStepV2):
        return step.duration_s

    if isinstance(step, HoldStepV2):
        return step.duration_s

    if isinstance(step, AccelerateStepV2):
        if (
            step.target_speed_mps
            < start_speed_mps - EPS
        ):
            raise ValueError(
                "AccelerateStepV2 target speed is below "
                "the current speed. Use BrakeStepV2."
            )

        return (
            step.target_speed_mps
            - start_speed_mps
        ) / step.acceleration_mps2

    if isinstance(step, BrakeStepV2):
        if (
            step.target_speed_mps
            > start_speed_mps + EPS
        ):
            raise ValueError(
                "BrakeStepV2 target speed is above "
                "the current speed. Use AccelerateStepV2."
            )

        return (
            start_speed_mps
            - step.target_speed_mps
        ) / step.deceleration_mps2

    raise TypeError(
        f"Unknown sequence step: {type(step)}"
    )


def _resolve_sequence_step(
    start: MotionState,
    step,
    elapsed_s: float,
) -> MotionState:

    yaw0_deg = start.yaw_deg
    yaw0_rad = math.radians(yaw0_deg)

    # --------------------------------------------------------
    # Cruise
    # --------------------------------------------------------

    if isinstance(step, CruiseStepV2):

        speed = (
            start.speed_mps
            if step.speed_mps is None
            else step.speed_mps
        )

        distance = speed * elapsed_s

        return _state_with_speed(
            x_m=(
                start.x_m
                + distance * math.cos(yaw0_rad)
            ),
            y_m=(
                start.y_m
                + distance * math.sin(yaw0_rad)
            ),
            yaw_deg=yaw0_deg,
            speed_mps=speed,
        )

    # --------------------------------------------------------
    # Hold
    # --------------------------------------------------------

    if isinstance(step, HoldStepV2):

        return _state_with_speed(
            x_m=start.x_m,
            y_m=start.y_m,
            yaw_deg=yaw0_deg,
            speed_mps=0.0,
        )

    # --------------------------------------------------------
    # Accelerate
    # --------------------------------------------------------

    if isinstance(step, AccelerateStepV2):

        v0 = start.speed_mps
        a = step.acceleration_mps2

        speed = min(
            step.target_speed_mps,
            v0 + a * elapsed_s,
        )

        distance = (
            v0 * elapsed_s
            + 0.5 * a * elapsed_s * elapsed_s
        )

        return _state_with_speed(
            x_m=(
                start.x_m
                + distance * math.cos(yaw0_rad)
            ),
            y_m=(
                start.y_m
                + distance * math.sin(yaw0_rad)
            ),
            yaw_deg=yaw0_deg,
            speed_mps=speed,
        )

    # --------------------------------------------------------
    # Brake
    # --------------------------------------------------------

    if isinstance(step, BrakeStepV2):

        v0 = start.speed_mps
        a = step.deceleration_mps2

        speed = max(
            step.target_speed_mps,
            v0 - a * elapsed_s,
        )

        distance = (
            v0 * elapsed_s
            - 0.5 * a * elapsed_s * elapsed_s
        )

        distance = max(
            0.0,
            distance,
        )

        return _state_with_speed(
            x_m=(
                start.x_m
                + distance * math.cos(yaw0_rad)
            ),
            y_m=(
                start.y_m
                + distance * math.sin(yaw0_rad)
            ),
            yaw_deg=yaw0_deg,
            speed_mps=speed,
        )

    # --------------------------------------------------------
    # Lane change
    # --------------------------------------------------------

    if isinstance(step, LaneChangeStepV2):

        if (
            abs(step.lateral_delta_m) > EPS
            and start.speed_mps <= EPS
        ):
            raise ValueError(
                "Lane change requires non-zero "
                "longitudinal speed."
            )

        duration = step.duration_s

        u = max(
            0.0,
            min(
                1.0,
                elapsed_s / duration,
            ),
        )

        lateral = (
            step.lateral_delta_m
            * smoothstep(u)
        )

        lateral_speed = (
            step.lateral_delta_m
            * smoothstep_derivative(u)
            / duration
        )

        forward = (
            start.speed_mps
            * elapsed_s
        )

        # Local vehicle frame:
        #
        # forward = ( cos(yaw), sin(yaw) )
        # left    = (-sin(yaw), cos(yaw) )
        #
        # Therefore positive lateral_delta_m means LEFT.

        x = (
            start.x_m
            + forward * math.cos(yaw0_rad)
            - lateral * math.sin(yaw0_rad)
        )

        y = (
            start.y_m
            + forward * math.sin(yaw0_rad)
            + lateral * math.cos(yaw0_rad)
        )

        vx = (
            start.speed_mps * math.cos(yaw0_rad)
            - lateral_speed * math.sin(yaw0_rad)
        )

        vy = (
            start.speed_mps * math.sin(yaw0_rad)
            + lateral_speed * math.cos(yaw0_rad)
        )

        speed = math.hypot(
            vx,
            vy,
        )

        if speed > EPS:
            yaw = math.degrees(
                math.atan2(
                    vy,
                    vx,
                )
            )
        else:
            yaw = yaw0_deg

        return MotionState(
            x_m=x,
            y_m=y,
            yaw_deg=normalize_yaw_deg(yaw),
            speed_mps=speed,
            vx_mps=vx,
            vy_mps=vy,
        )

    raise TypeError(
        f"Unknown sequence step: {type(step)}"
    )


def resolve_sequence_state(
    spawn: Pose2DV2,
    motion: SequenceMotionV2,
    t_s: float,
    active_start_s: float,
) -> MotionState:
    """
    Resolve an ordered physical behavior sequence.

    Every state is reconstructed deterministically from the sequence
    start so results do not depend on frame rate or previous calls.
    """

    state = _state_with_speed(
        x_m=spawn.x_m,
        y_m=spawn.y_m,
        yaw_deg=spawn.yaw_deg,
        speed_mps=motion.initial_speed_mps,
    )

    remaining = max(
        0.0,
        t_s - active_start_s,
    )

    for step in motion.steps:

        duration = _sequence_step_duration(
            step,
            state.speed_mps,
        )

        # Zero-duration speed transition.
        if duration <= EPS:
            state = _resolve_sequence_step(
                state,
                step,
                0.0,
            )
            continue

        # Current time lies inside this step.
        if remaining < duration - EPS:
            return _resolve_sequence_step(
                state,
                step,
                remaining,
            )

        # Resolve exact end of completed step.
        state = _resolve_sequence_step(
            state,
            step,
            duration,
        )

        remaining -= duration

        if remaining < EPS:
            return state

    # --------------------------------------------------------
    # After sequence
    # --------------------------------------------------------

    if (
        motion.end_behavior == "continue"
        and remaining > EPS
    ):
        distance = (
            state.speed_mps
            * remaining
        )

        yaw_rad = math.radians(
            state.yaw_deg
        )

        return _state_with_speed(
            x_m=(
                state.x_m
                + distance * math.cos(yaw_rad)
            ),
            y_m=(
                state.y_m
                + distance * math.sin(yaw_rad)
            ),
            yaw_deg=state.yaw_deg,
            speed_mps=state.speed_mps,
        )

    if remaining > EPS:
        return _state_with_speed(
            x_m=state.x_m,
            y_m=state.y_m,
            yaw_deg=state.yaw_deg,
            speed_mps=0.0,
        )

    return state
# ============================================================
# Keyframe resolver
# ============================================================

@dataclass
class PreparedKeyframe:
    t_s: float
    x_m: float
    y_m: float
    yaw_deg: float


def prepare_keyframes(
    spawn: Pose2DV2,
    motion: KeyframeMotionV2,
    active_start_s: float,
):
    frames = [
        PreparedKeyframe(
            t_s=k.t_s,
            x_m=k.x_m,
            y_m=k.y_m,
            yaw_deg=k.yaw_deg,
        )
        for k in motion.keyframes
    ]

    if not frames:
        raise ValueError(
            "Keyframe motion has no keyframes."
        )

    first = frames[0]

    # --------------------------------------------------------
    # Spawn pose is an implicit initial keyframe.
    # --------------------------------------------------------

    if (
        first.t_s
        > active_start_s + EPS
    ):
        frames.insert(
            0,
            PreparedKeyframe(
                t_s=active_start_s,
                x_m=spawn.x_m,
                y_m=spawn.y_m,
                yaw_deg=spawn.yaw_deg,
            )
        )

    elif (
        abs(
            first.t_s
            - active_start_s
        )
        <= EPS
    ):
        pose_difference = max(
            abs(
                first.x_m
                - spawn.x_m
            ),
            abs(
                first.y_m
                - spawn.y_m
            ),
            abs(
                shortest_angle_delta_deg(
                    spawn.yaw_deg,
                    first.yaw_deg,
                )
            ),
        )

        if pose_difference > 1e-6:
            raise ValueError(
                "First keyframe occurs at spawn_time_s "
                "but does not match actor spawn pose."
            )

    elif (
        first.t_s
        < active_start_s - EPS
    ):
        raise ValueError(
            "Keyframe occurs before actor "
            "becomes active."
        )

    return frames


def resolve_keyframe_state(
    prepared,
    interpolation: str,
    t_s: float,
) -> MotionState:

    # --------------------------------------------------------
    # Before first / after last: hold
    # --------------------------------------------------------

    if (
        t_s
        <= prepared[0].t_s
    ):
        k = prepared[0]

        return MotionState(
            x_m=k.x_m,
            y_m=k.y_m,
            yaw_deg=normalize_yaw_deg(
                k.yaw_deg
            ),
            speed_mps=0.0,
            vx_mps=0.0,
            vy_mps=0.0,
        )

    if (
        t_s
        >= prepared[-1].t_s
    ):
        k = prepared[-1]

        return MotionState(
            x_m=k.x_m,
            y_m=k.y_m,
            yaw_deg=normalize_yaw_deg(
                k.yaw_deg
            ),
            speed_mps=0.0,
            vx_mps=0.0,
            vy_mps=0.0,
        )

    times = [
        k.t_s
        for k in prepared
    ]

    i = (
        bisect.bisect_right(
            times,
            t_s,
        )
        - 1
    )

    k0 = prepared[i]
    k1 = prepared[i + 1]

    dt = (
        k1.t_s
        - k0.t_s
    )

    if dt <= EPS:
        raise ValueError(
            "Invalid keyframe interval."
        )

    u = (
        t_s
        - k0.t_s
    ) / dt

    if interpolation == "smooth":
        position_u = smoothstep(
            u
        )

        velocity_scale = (
            smoothstep_derivative(
                u
            )
            / dt
        )

    else:
        position_u = u

        velocity_scale = (
            1.0 / dt
        )

    x = (
        k0.x_m
        +
        position_u
        * (
            k1.x_m
            - k0.x_m
        )
    )

    y = (
        k0.y_m
        +
        position_u
        * (
            k1.y_m
            - k0.y_m
        )
    )

    yaw_u = (
        position_u
        if interpolation == "smooth"
        else u
    )

    yaw = interpolate_yaw_deg(
        k0.yaw_deg,
        k1.yaw_deg,
        yaw_u,
    )

    vx = (
        k1.x_m
        - k0.x_m
    ) * velocity_scale

    vy = (
        k1.y_m
        - k0.y_m
    ) * velocity_scale

    speed = math.hypot(
        vx,
        vy,
    )

    return MotionState(
        x_m=x,
        y_m=y,
        yaw_deg=yaw,
        speed_mps=speed,
        vx_mps=vx,
        vy_mps=vy,
    )


# ============================================================
# Generic motion resolver
# ============================================================

class PreparedMotion:
    def __init__(
        self,
        spawn,
        motion,
        active_start_s,
    ):
        self.spawn = spawn
        self.motion = motion

        self.path = None
        self.keyframes = None

        if isinstance(
            motion,
            PathMotionV2,
        ):
            self.path = (
                build_arc_length_path(
                    spawn,
                    motion,
                )
            )

        elif isinstance(
            motion,
            KeyframeMotionV2,
        ):
            self.keyframes = (
                prepare_keyframes(
                    spawn,
                    motion,
                    active_start_s,
                )
            )

    def resolve(
        self,
        t_s: float,
        active_start_s: float,
    ) -> MotionState:

        motion = self.motion

        if isinstance(
            motion,
            ManeuverMotionV2,
        ):
            return resolve_maneuver_state(
                self.spawn,
                motion,
                t_s,
            )
        if isinstance(
            motion,
            SequenceMotionV2,
        ):
            return resolve_sequence_state(
                spawn=self.spawn,
                motion=motion,
                t_s=t_s,
                active_start_s=active_start_s,
            )

        if isinstance(
            motion,
            PathMotionV2,
        ):
            elapsed = max(
                0.0,
                t_s
                - active_start_s,
            )

            distance = (
                motion.target_speed_mps
                * elapsed
            )

            return state_on_arc_path(
                self.path,
                distance,
                motion.target_speed_mps,
            )

        if isinstance(
            motion,
            KeyframeMotionV2,
        ):
            return resolve_keyframe_state(
                self.keyframes,
                motion.interpolation,
                t_s,
            )

        raise TypeError(
            f"Unknown motion type: "
            f"{type(motion)}"
        )

# ============================================================
# Final resolved velocity reconstruction
# ============================================================

def recompute_velocities_from_positions(
    frames,
):
    """
    Reconstruct vx, vy and speed from the FINAL resolved positions.

    Velocity at frame i represents the motion over:

        frame i -> frame i+1

    Therefore:
        - the first active keyframe can immediately have non-zero speed
        - a frame at which motion finishes can immediately become zero
        - all motion sources share identical velocity semantics

    For the final frame, backward difference is used.
    """

    if not frames:
        return []

    if len(frames) == 1:
        frame = frames[0]

        return [
            frame.model_copy(
                update={
                    "vx_mps": 0.0,
                    "vy_mps": 0.0,
                    "speed_mps": 0.0,
                }
            )
        ]

    output = []

    for i, frame in enumerate(
        frames
    ):

        # ----------------------------------------------------
        # Normal case:
        # use current -> next
        # ----------------------------------------------------

        if i < len(frames) - 1:
            first = frame
            second = frames[i + 1]

        # ----------------------------------------------------
        # Final frame:
        # use previous -> current
        # ----------------------------------------------------

        else:
            first = frames[i - 1]
            second = frame

        dt = (
            second.t_s
            - first.t_s
        )

        if dt <= EPS:
            vx = 0.0
            vy = 0.0

        else:
            vx = (
                second.x_m
                - first.x_m
            ) / dt

            vy = (
                second.y_m
                - first.y_m
            ) / dt

        speed = math.hypot(
            vx,
            vy,
        )

        output.append(
            frame.model_copy(
                update={
                    "vx_mps": vx,
                    "vy_mps": vy,
                    "speed_mps": speed,
                }
            )
        )

    return output


def recompute_actor_velocities(
    actor_frames,
):
    """
    Recompute velocities independently for every actor.

    Actors may have different lifecycles, therefore each actor's
    active frame sequence must be differentiated independently.
    """

    grouped = {}

    for frame in actor_frames:
        grouped.setdefault(
            frame.actor_id,
            [],
        ).append(
            frame
        )

    updated = {}

    for actor_id, frames in grouped.items():

        frames = sorted(
            frames,
            key=lambda x: x.frame_idx,
        )

        resolved = (
            recompute_velocities_from_positions(
                frames
            )
        )

        for frame in resolved:
            updated[
                (
                    actor_id,
                    frame.frame_idx,
                )
            ] = frame

    # Preserve the original frame-major ordering.
    return [
        updated[
            (
                frame.actor_id,
                frame.frame_idx,
            )
        ]
        for frame in actor_frames
    ]

# ============================================================
# Scenario resolver
# ============================================================

class V2TrajectoryResolver:

    def resolve(
        self,
        scenario: ScenarioSpecV2,
    ) -> ResolvedScenarioV2:

        total_frames = (
            round(
                scenario.duration_s
                * scenario.fps
            )
            + 1
        )

        # ----------------------------------------------------
        # Prepare ego
        # ----------------------------------------------------

        ego_prepared = PreparedMotion(
            spawn=scenario.ego.spawn,
            motion=scenario.ego.motion,
            active_start_s=0.0,
        )

        # ----------------------------------------------------
        # Prepare actors once
        # ----------------------------------------------------

        prepared_actors = {}

        for actor in scenario.actors:
            prepared_actors[
                actor.actor_id
            ] = PreparedMotion(
                spawn=actor.spawn,
                motion=actor.motion,
                active_start_s=(
                    actor.lifecycle
                    .spawn_time_s
                ),
            )

        # ----------------------------------------------------
        # Static actor metadata
        # ----------------------------------------------------

        actor_info = [
            ResolvedActorInfoV2(
                actor_id=actor.actor_id,
                actor_type=actor.actor_type,
                role=actor.role,
                asset_key=actor.asset_key,
                dimensions_m=(
                    actor.dimensions_m
                ),
                spawn_time_s=(
                    actor.lifecycle
                    .spawn_time_s
                ),
                despawn_time_s=(
                    actor.lifecycle
                    .despawn_time_s
                ),
            )
            for actor in scenario.actors
        ]

        ego_frames = []
        actor_frames = []

        # ----------------------------------------------------
        # Resolve every scenario frame
        # ----------------------------------------------------

        for frame_idx in range(
            total_frames
        ):
            t_s = (
                frame_idx
                / scenario.fps
            )

            # ------------------------------------------------
            # Ego always exists
            # ------------------------------------------------

            ego_state = (
                ego_prepared.resolve(
                    t_s=t_s,
                    active_start_s=0.0,
                )
            )

            ego_frames.append(
                ResolvedEgoFrameV2(
                    frame_idx=frame_idx,
                    t_s=t_s,
                    x_m=ego_state.x_m,
                    y_m=ego_state.y_m,
                    yaw_deg=(
                        ego_state.yaw_deg
                    ),
                    speed_mps=(
                        ego_state.speed_mps
                    ),
                    vx_mps=(
                        ego_state.vx_mps
                    ),
                    vy_mps=(
                        ego_state.vy_mps
                    ),
                )
            )

            # ------------------------------------------------
            # Independent actors
            # ------------------------------------------------

            for actor in scenario.actors:

                start_s = (
                    actor.lifecycle
                    .spawn_time_s
                )

                end_s = (
                    actor.lifecycle
                    .despawn_time_s
                    if (
                        actor.lifecycle
                        .despawn_time_s
                        is not None
                    )
                    else scenario.duration_s
                )

                # Actor does not exist outside its lifecycle.
                if (
                    t_s
                    < start_s - EPS
                    or t_s
                    > end_s + EPS
                ):
                    continue

                prepared = (
                    prepared_actors[
                        actor.actor_id
                    ]
                )

                state = prepared.resolve(
                    t_s=t_s,
                    active_start_s=start_s,
                )

                actor_frames.append(
                    ResolvedActorFrameV2(
                        frame_idx=frame_idx,
                        t_s=t_s,
                        actor_id=(
                            actor.actor_id
                        ),
                        x_m=state.x_m,
                        y_m=state.y_m,
                        yaw_deg=(
                            state.yaw_deg
                        ),
                        speed_mps=(
                            state.speed_mps
                        ),
                        vx_mps=(
                            state.vx_mps
                        ),
                        vy_mps=(
                            state.vy_mps
                        ),
                    )
                )
        # ----------------------------------------------------
        # Final physical velocity reconstruction
        # ----------------------------------------------------

        ego_frames = (
            recompute_velocities_from_positions(
                ego_frames
            )
        )

        actor_frames = (
            recompute_actor_velocities(
                actor_frames
            )
        )
        return ResolvedScenarioV2(
            scenario_id=(
                scenario.scenario_id
            ),
            source_description=(
                scenario.description
            ),
            duration_s=(
                scenario.duration_s
            ),
            fps=scenario.fps,
            seed=scenario.seed,
            camera=scenario.camera,
            actors=actor_info,
            ego_frames=ego_frames,
            actor_frames=actor_frames,
        )
