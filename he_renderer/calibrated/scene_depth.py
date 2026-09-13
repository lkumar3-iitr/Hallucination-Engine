"""Aligned camera-forward metric depth occlusion, isolated to HE."""

import numpy as np

from .calibrated_close import camera_rays


def validate_depth(depth, shape):
    depth = np.asarray(depth)
    if depth.shape != tuple(shape) or depth.dtype.kind not in 'fiu':
        raise ValueError('Expected aligned HxW numeric camera-forward depth in metres')
    return depth


def surface_depth(hull, actor, camera, width, height, fov):
    target = np.asarray(actor.get_inverse_matrix()) @ np.asarray(camera.get_matrix())
    directions = camera_rays(width, height, fov) @ target[:3, :3].T
    points, hit = hull.intersect(target[:3, 3], directions)
    camera_points = (points-target[:3, 3]) @ target[:3, :3]
    return np.where(hit & (camera_points[:, :, 0] > 0), camera_points[:, :, 0], np.nan)


def occlude(image, background, alpha, actor_depth, scene_depth, tolerance_m=.05):
    """Restore background at occluded pixels, preserving all other RGB exactly."""
    depth = validate_depth(scene_depth, alpha.shape)
    if actor_depth.shape != alpha.shape or image.shape != background.shape:
        raise ValueError('Depth/composite dimensions differ')
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('Depth tolerance must be finite and nonnegative')
    known = np.isfinite(actor_depth) & (actor_depth > 0)
    valid_scene = np.isfinite(depth) & (depth > 0)
    hidden = known & valid_scene & (depth+tolerance_m < actor_depth)
    visible = alpha > 10/255
    count = int(visible.sum())
    removed = int((hidden & visible).sum())
    rgb, result_alpha = image.copy(), alpha.copy()
    rgb[hidden] = background[hidden]
    result_alpha[hidden] = 0
    return rgb, result_alpha, dict(enabled=True, depth_convention='camera_forward_metres',
        depth_source='bank_visual_hull', tolerance_m=tolerance_m,
        occluded_pixels=removed, input_visible_pixels=count,
        occluded_fraction=removed/count if count else 0.,
        unknown_surface_pixels=int((visible & ~known).sum()),
        invalid_scene_pixels=int((visible & ~valid_scene).sum()))
