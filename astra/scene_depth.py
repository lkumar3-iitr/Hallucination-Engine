"""Aligned camera-forward metric depth occlusion, isolated to ASTRA."""
import ast
import inspect
import textwrap

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


def enable_runner_depth(runner):
    """Remove only the legacy version gate in the process-local runner function."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(runner.main)))

    class EnableDepth(ast.NodeTransformer):
        count = 0
        log_count = 0

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                target = node.targets[0]
                if (isinstance(target.value, ast.Name) and target.value.id == 'row'
                        and isinstance(target.slice, ast.JoinedStr)
                        and any(isinstance(v, ast.Constant) and v.value == '_scene_depth_used'
                                for v in target.slice.values)):
                    node.value = ast.parse("int(occlusion['depth_used'])", mode='eval').body
                    self.log_count += 1
            return self.generic_visit(node)

        def visit_keyword(self, node):
            if (node.arg == 'scene_depth_m' and isinstance(node.value, ast.IfExp)
                    and isinstance(node.value.body, ast.Constant)
                    and node.value.body.value is None
                    and isinstance(node.value.orelse, ast.Subscript)
                    and isinstance(node.value.orelse.value, ast.Name)
                    and node.value.orelse.value.id == 'scene_depth_by_camera'):
                node.value = node.value.orelse
                self.count += 1
            return self.generic_visit(node)

    transform = EnableDepth()
    tree = transform.visit(tree)
    if transform.count != 1 or transform.log_count != 1:
        raise RuntimeError('Runner depth gate changed; inspect before enabling depth')
    scope = {}
    exec(compile(ast.fix_missing_locations(tree), '<astra_depth_runner>', 'exec'),
         runner.main.__globals__, scope)
    runner.main = scope['main']
