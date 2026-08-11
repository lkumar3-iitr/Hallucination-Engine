from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CameraVisibilityResult:
    visible: bool
    reason: str
    bearing_deg: float
    distance_m: float


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize angle to [-180, 180)."""
    while angle_deg >= 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def check_front_camera_visibility(
    x_m: float,
    y_m: float,
    fov_deg: float = 90.0,
    min_distance_m: float = 0.5,
    max_distance_m: float = 120.0,
) -> CameraVisibilityResult:
    """Check whether an actor is visible in the current front-facing HE camera.

    Coordinate convention:
    - x_m is forward relative to ego.
    - y_m is lateral relative to ego.
    - front camera looks along +x.
    - bearing 0 deg means directly ahead.
    - positive bearing means actor is left of ego.

    Important:
    This function should only decide whether to render in a given camera.
    It should NOT remove the actor from the resolved scenario or BEV memory.
    """

    distance_m = math.sqrt(x_m * x_m + y_m * y_m)

    if distance_m < min_distance_m:
        return CameraVisibilityResult(
            visible=False,
            reason="too_close",
            bearing_deg=0.0,
            distance_m=distance_m,
        )

    if distance_m > max_distance_m:
        return CameraVisibilityResult(
            visible=False,
            reason="too_far",
            bearing_deg=0.0,
            distance_m=distance_m,
        )

    # Behind ego for a front camera.
    if x_m <= 0.0:
        bearing_deg = math.degrees(math.atan2(y_m, x_m))
        return CameraVisibilityResult(
            visible=False,
            reason="behind_front_camera",
            bearing_deg=bearing_deg,
            distance_m=distance_m,
        )

    bearing_deg = math.degrees(math.atan2(y_m, x_m))
    half_fov = fov_deg / 2.0

    if abs(bearing_deg) > half_fov:
        return CameraVisibilityResult(
            visible=False,
            reason="outside_front_camera_fov",
            bearing_deg=bearing_deg,
            distance_m=distance_m,
        )

    return CameraVisibilityResult(
        visible=True,
        reason="visible",
        bearing_deg=bearing_deg,
        distance_m=distance_m,
    )


def check_yaw_camera_visibility(
    x_m: float,
    y_m: float,
    camera_yaw_deg: float,
    fov_deg: float = 90.0,
    min_distance_m: float = 0.5,
    max_distance_m: float = 120.0,
) -> CameraVisibilityResult:
    """Generic visibility check for a camera looking at a given yaw.

    camera_yaw_deg convention:
    - 0 deg: front
    - 90 deg: left
    - -90 deg: right
    - 180 deg: rear

    This is useful for future 360 / multi-camera rendering.
    """

    distance_m = math.sqrt(x_m * x_m + y_m * y_m)

    if distance_m < min_distance_m:
        return CameraVisibilityResult(
            visible=False,
            reason="too_close",
            bearing_deg=0.0,
            distance_m=distance_m,
        )

    if distance_m > max_distance_m:
        return CameraVisibilityResult(
            visible=False,
            reason="too_far",
            bearing_deg=0.0,
            distance_m=distance_m,
        )

    actor_bearing_deg = math.degrees(math.atan2(y_m, x_m))
    relative_bearing_deg = normalize_angle_deg(actor_bearing_deg - camera_yaw_deg)

    if abs(relative_bearing_deg) > fov_deg / 2.0:
        return CameraVisibilityResult(
            visible=False,
            reason="outside_camera_fov",
            bearing_deg=relative_bearing_deg,
            distance_m=distance_m,
        )

    return CameraVisibilityResult(
        visible=True,
        reason="visible",
        bearing_deg=relative_bearing_deg,
        distance_m=distance_m,
    )