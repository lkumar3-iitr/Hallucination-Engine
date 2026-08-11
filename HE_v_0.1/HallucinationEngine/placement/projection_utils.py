import math
import numpy as np


def build_camera_intrinsics(image_width, image_height, fov_degrees):
    """
    Build camera intrinsic matrix from image size and horizontal FOV.

    CARLA camera usually provides horizontal FOV.
    """

    fov_rad = math.radians(fov_degrees)

    fx = image_width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx

    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K


def get_matrix_from_transform(transform):
    """
    Convert a CARLA-like transform into a 4x4 matrix.

    This function does not import carla.
    It assumes transform has:
        transform.location.x/y/z
        transform.rotation.pitch/yaw/roll
    """

    location = transform.location
    rotation = transform.rotation

    pitch = math.radians(rotation.pitch)
    yaw = math.radians(rotation.yaw)
    roll = math.radians(rotation.roll)

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    cp = math.cos(pitch)
    sp = math.sin(pitch)

    cr = math.cos(roll)
    sr = math.sin(roll)

    # CARLA-like rotation matrix
    matrix = np.identity(4, dtype=np.float32)

    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z

    matrix[0, 0] = cp * cy
    matrix[0, 1] = cy * sp * sr - sy * cr
    matrix[0, 2] = -cy * sp * cr - sy * sr

    matrix[1, 0] = cp * sy
    matrix[1, 1] = sy * sp * sr + cy * cr
    matrix[1, 2] = -sy * sp * cr + cy * sr

    matrix[2, 0] = sp
    matrix[2, 1] = -cp * sr
    matrix[2, 2] = cp * cr

    return matrix


def world_to_camera(world_location, camera_transform):
    """
    Convert world location to CARLA camera coordinate.

    Args:
        world_location: object with x, y, z OR list/tuple [x, y, z]
        camera_transform: CARLA-like camera transform

    Returns:
        point in camera coordinates as numpy array [x, y, z]
    """

    if hasattr(world_location, "x"):
        point = np.array(
            [world_location.x, world_location.y, world_location.z, 1.0],
            dtype=np.float32
        )
    else:
        point = np.array(
            [world_location[0], world_location[1], world_location[2], 1.0],
            dtype=np.float32
        )

    camera_matrix = get_matrix_from_transform(camera_transform)
    world_to_camera_matrix = np.linalg.inv(camera_matrix)

    point_camera = world_to_camera_matrix @ point

    return point_camera[:3]


def camera_to_image(point_camera, K):
    """
    Convert CARLA camera coordinates to image pixel coordinates.

    Important:
    CARLA camera coordinate after inverse transform is:
        x = forward
        y = right
        z = up

    Computer vision convention usually expects:
        x = right
        y = down
        z = forward

    So we convert:
        CV x = CARLA y
        CV y = -CARLA z
        CV z = CARLA x
    """

    x_carla, y_carla, z_carla = point_camera

    # Point behind camera
    if x_carla <= 0.1:
        return None

    point_cv = np.array([
        y_carla,
        -z_carla,
        x_carla
    ], dtype=np.float32)

    projected = K @ point_cv

    u = projected[0] / projected[2]
    v = projected[1] / projected[2]

    return float(u), float(v), float(projected[2])


def project_world_to_image(world_location, camera_transform, K):
    """
    Full world → camera → image projection.
    """

    point_camera = world_to_camera(world_location, camera_transform)
    return camera_to_image(point_camera, K)


def make_2d_box_from_projected_center(
    center_u,
    center_v,
    depth,
    vehicle_width_m,
    vehicle_height_m,
    focal_length_px,
    image_width,
    image_height
):
    """
    Create approximate 2D box from projected center and object size.

    This is not full 3D box projection yet.
    It is a good intermediate step.
    """

    if depth <= 0.1:
        return None

    box_width_px = focal_length_px * vehicle_width_m / depth
    box_height_px = focal_length_px * vehicle_height_m / depth

    x1 = center_u - box_width_px / 2.0
    y1 = center_v - box_height_px / 2.0
    x2 = center_u + box_width_px / 2.0
    y2 = center_v + box_height_px / 2.0

    # Clip to image boundaries
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [int(x1), int(y1), int(x2), int(y2)]

def rotate_point_2d(x, y, yaw_degrees):
    """
    Rotate a 2D point around origin using yaw angle in degrees.
    """

    yaw = math.radians(yaw_degrees)

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    rx = x * cos_yaw - y * sin_yaw
    ry = x * sin_yaw + y * cos_yaw

    return rx, ry


def build_3d_vehicle_corners(
    center_xyz,
    yaw_degrees,
    length_m,
    width_m,
    height_m
):
    """
    Build 8 corners of a vehicle-like 3D bounding box.

    CARLA convention:
        x/y are ground-plane world coordinates
        z is vertical

    center_xyz is treated as the center of the vehicle body.
    """

    cx, cy, cz = center_xyz

    half_l = length_m / 2.0
    half_w = width_m / 2.0
    half_h = height_m / 2.0

    # Local cuboid corners before rotation.
    # x = forward/backward, y = left/right, z = up/down
    local_corners = [
        [ half_l,  half_w,  half_h],
        [ half_l, -half_w,  half_h],
        [-half_l, -half_w,  half_h],
        [-half_l,  half_w,  half_h],

        [ half_l,  half_w, -half_h],
        [ half_l, -half_w, -half_h],
        [-half_l, -half_w, -half_h],
        [-half_l,  half_w, -half_h],
    ]

    world_corners = []

    for lx, ly, lz in local_corners:
        rx, ry = rotate_point_2d(lx, ly, yaw_degrees)

        wx = cx + rx
        wy = cy + ry
        wz = cz + lz

        world_corners.append([wx, wy, wz])

    return world_corners


def project_3d_box_to_2d(
    center_xyz,
    yaw_degrees,
    camera_transform,
    camera_intrinsics,
    length_m,
    width_m,
    height_m,
    image_width,
    image_height
):
    """
    Project a 3D cuboid vehicle box into the camera image.

    Returns:
        2D bounding box [x1, y1, x2, y2]
        or None if projection fails.
    """

    corners_3d = build_3d_vehicle_corners(
        center_xyz=center_xyz,
        yaw_degrees=yaw_degrees,
        length_m=length_m,
        width_m=width_m,
        height_m=height_m
    )

    projected_points = []

    for corner in corners_3d:
        projected = project_world_to_image(
            world_location=corner,
            camera_transform=camera_transform,
            K=camera_intrinsics
        )

        if projected is None:
            continue

        u, v, depth = projected

        # Keep only points in front of camera.
        if depth <= 0.1:
            continue

        projected_points.append([u, v])

    # Need enough visible corners for a meaningful box
    if len(projected_points) < 2:
        return None

    xs = [p[0] for p in projected_points]
    ys = [p[1] for p in projected_points]

    x1 = max(0, min(image_width - 1, min(xs)))
    y1 = max(0, min(image_height - 1, min(ys)))
    x2 = max(0, min(image_width - 1, max(xs)))
    y2 = max(0, min(image_height - 1, max(ys)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [int(x1), int(y1), int(x2), int(y2)]