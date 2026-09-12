"""Side-pass renderer: select/size in a centered camera, reproject to native rays.

The two cameras share a position, so image reprojection is a pure rotation.
No runtime target images/masks are inputs. Production geometry helpers are
read-only dependencies; all experimental integration lives here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import carla
import cv2
import numpy as np

if __package__:
    from .selector import Selector
else:
    from selector import Selector

COMMON = Path(__file__).resolve().parents[1] / "driving_models/common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
import he_camera_renderer as geometry


def aimed_camera(actor_tf, camera_tf, center, extents=None):
    actor_matrix = np.asarray(actor_tf.get_matrix())
    camera_world = np.array(
        [camera_tf.location.x, camera_tf.location.y, camera_tf.location.z, 1.0]
    )
    target_local = np.asarray(center, dtype=float)
    if extents is not None:
        camera_local = (np.asarray(actor_tf.get_inverse_matrix()) @ camera_world)[:3]
        lo = np.asarray(center, dtype=float) - np.asarray(extents, dtype=float)
        hi = np.asarray(center, dtype=float) + np.asarray(extents, dtype=float)
        target_local = np.clip(camera_local, lo, hi)
        if np.linalg.norm(target_local - camera_local) < 1e-5:
            raise ValueError("Camera intersects physical bbox")
    target_world = actor_matrix @ np.r_[target_local, 1.0]
    ray = target_world[:3]-camera_world[:3]
    distance = np.linalg.norm(ray)
    if distance < .05:
        raise ValueError("Camera is at the actor center")
    return carla.Transform(camera_tf.location, carla.Rotation(
        yaw=math.degrees(math.atan2(ray[1], ray[0])),
        pitch=math.degrees(math.atan2(ray[2], math.hypot(ray[0], ray[1]))))), distance


def camera_rotation_map(source_camera, target_camera, width, height, fov):
    """Map target image rays into source pixels; same intrinsics and camera origin."""
    source_xyz = np.array([source_camera.location.x, source_camera.location.y, source_camera.location.z])
    target_xyz = np.array([target_camera.location.x, target_camera.location.y, target_camera.location.z])
    if not np.allclose(source_xyz, target_xyz, atol=1e-6, rtol=0):
        raise ValueError("Pure rotation mapping requires identical camera positions")
    fx = width/(2*math.tan(math.radians(fov)/2))
    k = np.array([[fx, 0, width/2], [0, fx, height/2], [0, 0, 1]])
    axes = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
    rs = np.asarray(source_camera.get_matrix())[:3, :3]
    rt = np.asarray(target_camera.get_matrix())[:3, :3]
    return k @ axes @ rs.T @ rt @ axes.T @ np.linalg.inv(k)


def reproject_sprite(background, sprite, box, target_to_source):
    """Inverse sample visible rays, including objects crossing the target camera plane."""
    ys, xs = np.where(sprite[:, :, 3] > 10)
    if not len(xs):
        raise ValueError("Empty source sprite")
    crop = sprite[ys.min():ys.max()+1, xs.min():xs.max()+1]
    x1, y1, x2, y2 = map(float, box)
    if not np.all(np.isfinite(box)) or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid full source-camera box")
    h, w = background.shape[:2]
    y, x = np.ogrid[:h, :w]
    matrix = target_to_source
    depth = matrix[2, 0]*x + matrix[2, 1]*y + matrix[2, 2]
    valid = depth > 1e-6
    safe = np.where(valid, depth, 1)
    u = (matrix[0, 0]*x+matrix[0, 1]*y+matrix[0, 2])/safe
    v = (matrix[1, 0]*x+matrix[1, 1]*y+matrix[1, 2])/safe
    valid &= (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
    map_x = np.where(valid, (u-x1+.5)*crop.shape[1]/(x2-x1)-.5, -1).astype(np.float32)
    map_y = np.where(valid, (v-y1+.5)*crop.shape[0]/(y2-y1)-.5, -1).astype(np.float32)
    alpha = crop[:, :, 3:4].astype(np.float32)/255
    source = np.concatenate((crop[:, :, :3].astype(np.float32)*alpha, alpha), axis=2)
    sample = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = sample[:, :, 3]
    result = np.rint(np.clip(sample[:, :, :3] + background*(1-a[:, :, None]), 0, 255)).astype(np.uint8)
    return result, a


class PassbyRenderer:
    def __init__(self, bank, artifacts):
        self.selector = Selector(bank, artifacts)
        self.selector.build_hull()
        self.bank_config = {"mode": "view_matrix", "view_matrix_csvs": [str(Path(bank)/"view_matrix.csv")],
                            "target_height_m": float(self.selector.center[2]),
                            "vertical_mode": "state_y", "camera_height_m": 1.6,
                            "distance_selection_mode": "linear"}
        self.view_matrix = geometry.load_view_matrix_sprite_bank(self.bank_config)
        self.centered_view_matrix = dict(self.view_matrix)
        self.centered_view_matrix["index"] = {}
        for raw in self.selector.rows:
            key = (int(float(raw["angle_deg"])), float(raw["distance_m"]), float(raw["elevation_deg"]))
            record = dict(self.view_matrix["index"][key])
            # The existing helper expects an alpha-box center here, not a wheel/contact anchor.
            record["anchor_x"] = (float(raw["visible_x1_full_px"])+float(raw["visible_x2_full_px"]))/2-float(raw["crop_x1_px"])
            record["anchor_y"] = float(raw["visible_y2_full_px"])-float(raw["crop_y1_px"])
            self.centered_view_matrix["index"][key] = record

    def render(self, background_bgr, actor_tf, camera_tf, fov, box_mode="hull", box_provider=None):
        height, width = background_bgr.shape[:2]
        virtual, _ = aimed_camera(
            actor_tf, camera_tf, self.selector.center, self.selector.extents
        )
        center_world = np.asarray(actor_tf.get_matrix()) @ np.r_[self.selector.center, 1.0]
        distance = np.linalg.norm(center_world[:3] - np.array([
            camera_tf.location.x, camera_tf.location.y, camera_tf.location.z
        ]))
        choice = self.selector.select(actor_tf, virtual, width, height, fov)
        if box_provider is not None:
            box = box_provider(actor_tf, virtual, width, height, fov, choice)
            box_mode = "external"
        elif box_mode in {"existing", "anchor_fixed"}:
            physical = geometry.project_asset_physical_bbox_image_box(
                actor_tf, virtual, self.view_matrix["physical_bbox"], width, height, fov)
            if physical is None:
                raise ValueError("Physical bbox intersects centered camera plane")
            correction = geometry.interpolate_silhouette_correction_inverse_depth(
                self.centered_view_matrix if box_mode == "anchor_fixed" else self.view_matrix,
                choice["key"][0], choice["key"][2], distance)
            if correction is None:
                raise ValueError("Bank silhouette correction is missing")
            bw, bh = physical["width_px"], physical["height_px"]
            cx = physical["center_x"] + correction["center_x_offset_ratio"]*bw
            bottom = physical["bottom_y"] + correction["bottom_y_offset_ratio"]*bh
            sw, sh = bw*correction["width_ratio"], bh*correction["height_ratio"]
            box = [cx-sw/2, bottom-sh, cx+sw/2, bottom]
        elif box_mode == "hull":
            box = choice["predicted_box"]
        else:
            raise ValueError("box_mode must be existing, anchor_fixed, or hull")
        sprite = cv2.imread(choice["sprite_path"], cv2.IMREAD_UNCHANGED)
        image, alpha = reproject_sprite(background_bgr, sprite, box,
                                        camera_rotation_map(virtual, camera_tf, width, height, fov))
        meta = {k: v for k, v in choice.items() if k != "predicted_mask"}
        meta.update(box_mode=box_mode, virtual_box=box, center_distance_m=float(distance),
                    rendered=bool(np.any(alpha > .04)), visible_pixels=int(np.count_nonzero(alpha > .04)))
        return image, alpha, meta
