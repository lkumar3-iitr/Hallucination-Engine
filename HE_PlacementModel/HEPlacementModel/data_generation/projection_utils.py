import math
import numpy as np
import carla


def build_camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """
    Build pinhole camera intrinsic matrix from CARLA camera width/height/FOV.

    CARLA RGB camera FOV is horizontal FOV.
    """
    fov_rad = math.radians(fov_deg)
    fx = width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def carla_transform_to_matrix(transform: carla.Transform) -> np.ndarray:
    """
    Convert CARLA Transform to 4x4 homogeneous matrix.
    """
    return np.array(transform.get_matrix(), dtype=np.float32)


def world_to_camera_matrix(camera_transform: carla.Transform) -> np.ndarray:
    """
    Return inverse camera transform matrix.
    """
    return np.array(camera_transform.get_inverse_matrix(), dtype=np.float32)


def get_bbox_world_corners(vehicle: carla.Actor) -> np.ndarray:
    """
    Get 8 corners of a CARLA actor bounding box in world coordinates.

    Returns:
        np.ndarray of shape [8, 3]
    """
    bbox = vehicle.bounding_box
    extent = bbox.extent

    corners_local = np.array(
        [
            [ extent.x,  extent.y,  extent.z, 1.0],
            [ extent.x, -extent.y,  extent.z, 1.0],
            [-extent.x, -extent.y,  extent.z, 1.0],
            [-extent.x,  extent.y,  extent.z, 1.0],
            [ extent.x,  extent.y, -extent.z, 1.0],
            [ extent.x, -extent.y, -extent.z, 1.0],
            [-extent.x, -extent.y, -extent.z, 1.0],
            [-extent.x,  extent.y, -extent.z, 1.0],
        ],
        dtype=np.float32,
    )

    # Actor transform moves from actor local to world.
    actor_matrix = carla_transform_to_matrix(vehicle.get_transform())

    # Bounding box has its own local offset relative to actor origin.
    bbox_transform = carla.Transform(bbox.location)
    bbox_matrix = carla_transform_to_matrix(bbox_transform)

    world_matrix = actor_matrix @ bbox_matrix
    corners_world = (world_matrix @ corners_local.T).T[:, :3]

    return corners_world


def project_world_points_to_image(
    points_world: np.ndarray,
    camera_transform: carla.Transform,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project world points into CARLA camera image coordinates.

    Important:
    CARLA world/camera coordinates differ from standard CV coordinates.
    After world-to-camera:
      CARLA camera coords: x forward, y right, z up
      CV camera coords:    x right, y down, z forward

    Conversion:
      x_cv = y_carla
      y_cv = -z_carla
      z_cv = x_carla
    """
    assert points_world.shape[1] == 3

    points_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    w2c = world_to_camera_matrix(camera_transform)
    points_camera = (w2c @ points_h.T).T[:, :3]

    # CARLA camera -> standard computer vision camera
    points_cv = np.stack(
        [
            points_camera[:, 1],
            -points_camera[:, 2],
            points_camera[:, 0],
        ],
        axis=1,
    )

    z = points_cv[:, 2]
    valid_depth = z > 1e-4

    projected = np.zeros((points_cv.shape[0], 2), dtype=np.float32)
    projected[:, 0] = K[0, 0] * points_cv[:, 0] / np.maximum(z, 1e-4) + K[0, 2]
    projected[:, 1] = K[1, 1] * points_cv[:, 1] / np.maximum(z, 1e-4) + K[1, 2]

    return projected, valid_depth


def compute_2d_bbox_from_projected_points(
    points_2d: np.ndarray,
    valid_depth: np.ndarray,
    image_width: int,
    image_height: int,
    min_bbox_width_px: float = 2.0,
    min_bbox_height_px: float = 2.0,
) -> dict:
    """
    Compute clipped 2D bbox from projected 3D bbox corners.

    Returns:
        {
          visible,
          center_x,
          bottom_y,
          box_width,
          box_height,
          x_min,
          y_min,
          x_max,
          y_max,
          unclipped_x_min,
          unclipped_y_min,
          unclipped_x_max,
          unclipped_y_max
        }
    """
    if not np.any(valid_depth):
        return invisible_bbox()

    pts = points_2d[valid_depth]

    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))

    unclipped_x_min = x_min
    unclipped_y_min = y_min
    unclipped_x_max = x_max
    unclipped_y_max = y_max

    # Check whether bbox intersects the image.
    intersects = not (
        x_max < 0
        or x_min >= image_width
        or y_max < 0
        or y_min >= image_height
    )

    if not intersects:
        return invisible_bbox(
            unclipped_x_min,
            unclipped_y_min,
            unclipped_x_max,
            unclipped_y_max,
        )

    # Clip bbox to image boundaries.
    x_min = max(0.0, min(float(image_width - 1), x_min))
    x_max = max(0.0, min(float(image_width - 1), x_max))
    y_min = max(0.0, min(float(image_height - 1), y_min))
    y_max = max(0.0, min(float(image_height - 1), y_max))

    box_width = x_max - x_min
    box_height = y_max - y_min

    visible = (
        box_width >= min_bbox_width_px
        and box_height >= min_bbox_height_px
    )

    if not visible:
        return invisible_bbox(
            unclipped_x_min,
            unclipped_y_min,
            unclipped_x_max,
            unclipped_y_max,
        )

    center_x = 0.5 * (x_min + x_max)
    bottom_y = y_max

    return {
        "visible": 1,
        "center_x": float(center_x),
        "bottom_y": float(bottom_y),
        "box_width": float(box_width),
        "box_height": float(box_height),
        "x_min": float(x_min),
        "y_min": float(y_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "unclipped_x_min": float(unclipped_x_min),
        "unclipped_y_min": float(unclipped_y_min),
        "unclipped_x_max": float(unclipped_x_max),
        "unclipped_y_max": float(unclipped_y_max),
    }


def invisible_bbox(
    unclipped_x_min=None,
    unclipped_y_min=None,
    unclipped_x_max=None,
    unclipped_y_max=None,
) -> dict:
    return {
        "visible": 0,
        "center_x": 0.0,
        "bottom_y": 0.0,
        "box_width": 0.0,
        "box_height": 0.0,
        "x_min": 0.0,
        "y_min": 0.0,
        "x_max": 0.0,
        "y_max": 0.0,
        "unclipped_x_min": unclipped_x_min,
        "unclipped_y_min": unclipped_y_min,
        "unclipped_x_max": unclipped_x_max,
        "unclipped_y_max": unclipped_y_max,
    }


def compute_relative_state(
    ego_transform: carla.Transform,
    adv_transform: carla.Transform,
) -> dict:
    """
    Compute adversary state in ego-local coordinates.

    For HEPlacementModel v1:
      rel_x: lateral position, positive right
      rel_z: forward position, positive in front of ego
      rel_y: vertical difference
      rel_yaw: adversary yaw relative to ego yaw
    """
    ego_loc = ego_transform.location
    adv_loc = adv_transform.location

    dx = adv_loc.x - ego_loc.x
    dy = adv_loc.y - ego_loc.y
    dz = adv_loc.z - ego_loc.z

    yaw = math.radians(ego_transform.rotation.yaw)

    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)

    right_x = math.cos(yaw + math.pi / 2.0)
    right_y = math.sin(yaw + math.pi / 2.0)

    rel_forward = dx * forward_x + dy * forward_y
    rel_right = dx * right_x + dy * right_y
    rel_up = dz

    rel_yaw = normalize_angle_deg(
        adv_transform.rotation.yaw - ego_transform.rotation.yaw
    )

    return {
        "rel_x": float(rel_right),
        "rel_z": float(rel_forward),
        "rel_y": float(rel_up),
        "rel_yaw": float(rel_yaw),
    }


def normalize_angle_deg(angle: float) -> float:
    """
    Normalize angle to [-180, 180].
    """
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle