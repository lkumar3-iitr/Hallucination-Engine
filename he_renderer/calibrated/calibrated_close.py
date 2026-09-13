"""Experimental calibrated, actor-local close bank; no target masks or box fitting.

At a capture node reprojection is an exact camera rotation. Between nodes a
side-plane homography approximates parallax before premultiplied interpolation.
The approximation is deliberately explicit, not a claim of recovered 3-D shape.
"""
from __future__ import annotations

import csv
import itertools
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=16)
def camera_rays(width, height, fov):
    fx = width / (2 * math.tan(math.radians(fov)/2))
    y, x = np.mgrid[:height, :width]
    rays = np.stack((np.ones_like(x), (x-width/2)/fx, -(y-height/2)/fx), axis=-1)
    rays.setflags(write=False)
    return rays


def bracket(values, value, tolerance=1e-4):
    values = np.asarray(values, dtype=float)
    if value < values[0] - tolerance or value > values[-1] + tolerance:
        return None
    value = float(np.clip(value, values[0], values[-1]))
    upper = int(np.searchsorted(values, value))
    if upper == 0:
        return [(float(values[0]), 1.0)]
    lower = upper - 1
    weight = (value - values[lower]) / (values[upper] - values[lower])
    return [(float(values[lower]), float(1-weight)), (float(values[upper]), float(weight))]


def pose(row, prefix):
    import carla
    return np.asarray(carla.Transform(
        carla.Location(**{a: float(row[f"{prefix}_location_{a}_m"]) for a in "xyz"}),
        carla.Rotation(**{a: float(row[f"{prefix}_{a}_deg"]) for a in ("yaw", "pitch", "roll")}),
    ).get_matrix())


class CalibratedCloseBank:
    def __init__(self, csv_path, center, side_half_width, hull=None):
        self.path = Path(csv_path)
        self.center = np.asarray(center, dtype=float)
        self.side_half_width = float(side_half_width)
        self.hull = hull
        with self.path.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.axes = [sorted({float(row[key]) for row in self.rows}) for key in
                     ("close_forward_m", "close_right_m", "close_target_up_m")]
        self.index = {}
        self.matrices = []
        for i, row in enumerate(self.rows):
            key = tuple(round(float(row[k]), 6) for k in
                        ("close_forward_m", "close_right_m", "close_target_up_m"))
            if key in self.index:
                raise ValueError(f"Duplicate capture node: {key}")
            self.index[key] = i
            self.matrices.append(np.linalg.inv(pose(row, "camera")) @ pose(row, "actor"))

    @lru_cache(maxsize=24)
    def source(self, index):
        row = self.rows[index]
        image = cv2.imread(str(self.path.parent / row["rgba_relpath"]), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"Invalid source {row['rgba_relpath']}")
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        return np.concatenate((image[:, :, :3].astype(np.float32)*alpha, alpha), axis=2)

    def render(self, actor_tf, camera_tf, width, height, fov):
        target = np.asarray(actor_tf.get_inverse_matrix()) @ np.asarray(camera_tf.get_matrix())
        origin = target[:3, 3]
        query = self.center - origin
        # Never interpolate across the actor through opposite-side captures.
        right_values = [v for v in self.axes[1] if v * query[1] > 0]
        if not right_values:
            return None
        brackets = [bracket(values, q) for values, q in
                    zip((self.axes[0], right_values, self.axes[2]), query)]
        if any(value is None for value in brackets):
            return None
        rays = camera_rays(width, height, fov)
        directions = rays @ target[:3, :3].T
        plane_y = self.center[1] + np.sign(origin[1]-self.center[1])*self.side_half_width
        if self.hull is not None:
            points, valid = self.hull.intersect(origin, directions)
        else:
            dy = directions[:, :, 1]
            distance = (plane_y-origin[1]) / np.where(np.abs(dy) > 1e-8, dy, 1)
            valid = (distance > 0) & (np.abs(dy) > 1e-8)
            points = origin + directions*distance[:, :, None]
        output = np.zeros((height, width, 4), np.float32)
        ys, xs = np.nonzero(valid)
        roi = (slice(int(ys.min()), int(ys.max())+1),
               slice(int(xs.min()), int(xs.max())+1)) if len(xs) else None
        if roi is not None:
            points, valid = points[roi], valid[roi]
        selections = []
        for node in itertools.product(*brackets):
            weight = math.prod(pair[1] for pair in node)
            if weight < 1e-8:
                continue
            key = tuple(round(pair[0], 6) for pair in node)
            if key not in self.index:
                return None
            index = self.index[key]
            selections.append({"node": key, "weight": weight, "index": index})
            if roi is None:
                continue
            row, matrix = self.rows[index], self.matrices[index]
            projected = points @ matrix[:3, :3].T + matrix[:3, 3]
            depth = projected[:, :, 0]
            good = valid & (depth > 1e-6)
            safe = np.where(good, depth, 1)
            u = float(row["camera_cx_px"]) + float(row["camera_fx_px"])*projected[:, :, 1]/safe
            v = float(row["camera_cy_px"]) - float(row["camera_fy_px"])*projected[:, :, 2]/safe
            mx = np.where(good, u-float(row["crop_x1_px"]), -1).astype(np.float32)
            my = np.where(good, v-float(row["crop_y1_px"]), -1).astype(np.float32)
            output[roi] += weight * cv2.remap(self.source(index), mx, my, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
        return output, {"backend": "he_calibrated_close_experimental_v1",
                        "query_actor_local": query.tolist(), "sources": selections,
                        "parallax_proxy": "bank_visual_hull" if self.hull is not None else "actor_side_plane",
                        "side_plane_y": float(plane_y)}


def composite(background, premultiplied):
    alpha = np.clip(premultiplied[:, :, 3:4], 0, 1)
    return np.rint(np.clip(premultiplied[:, :, :3] + background*(1-alpha), 0, 255)).astype(np.uint8)


def coverage_weight(bank, query):
    """Smooth every finite coverage boundary, not only the longitudinal exit."""
    side = [v for v in bank.axes[1] if v*query[1] > 0]
    if not side:
        return 0.0
    weight = 1.0
    for values, value, margin in zip((bank.axes[0], side, bank.axes[2]), query, (2., .1, .1)):
        edge = min(value-values[0], values[-1]-value)
        t = float(np.clip(edge/margin, 0, 1))
        weight *= t*t*(3-2*t)
    return weight
