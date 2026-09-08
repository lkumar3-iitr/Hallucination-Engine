"""Bank-only visual-hull reconstruction and mask-free runtime sprite selection."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import carla
import cv2
import numpy as np


POPCOUNT_U8 = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None],
    axis=1,
).sum(axis=1).astype(np.uint8)


def transform(data):
    return carla.Transform(
        carla.Location(**{k: float(data[k]) for k in ("x", "y", "z")}),
        carla.Rotation(**{k: float(data.get(k, 0)) for k in ("pitch", "yaw", "roll")}),
    )


def capture_transform(row, prefix):
    return transform({
        **{k: row[f"{prefix}_location_{k}_m"] for k in ("x", "y", "z")},
        **{k: row[f"{prefix}_{k}_deg"] for k in ("pitch", "yaw", "roll")},
    })


def relative_matrix(actor, camera):
    return np.asarray(camera.get_inverse_matrix()) @ np.asarray(actor.get_matrix())


def project(points, matrix, fx, fy, cx, cy):
    camera = points @ matrix[:3, :3].T + matrix[:3, 3]
    depth = camera[:, 0]
    safe = np.where(depth > 0.01, depth, np.nan)
    return np.column_stack((cx + fx * camera[:, 1] / safe,
                            cy - fy * camera[:, 2] / safe)), depth


def crop_mask(mask):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("Empty silhouette")
    return mask[y.min():y.max()+1, x.min():x.max()+1]


def canonical(mask, size):
    return cv2.resize(crop_mask(mask).astype(np.uint8), (size, size),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def iou_scores(masks, target):
    intersections = np.count_nonzero(masks & target, axis=(1, 2))
    unions = np.count_nonzero(masks | target, axis=(1, 2))
    return intersections / np.maximum(unions, 1)


def packed_iou_scores(packed_masks, target):
    packed_target = np.packbits(target, axis=None)
    intersections = POPCOUNT_U8[np.bitwise_and(packed_masks, packed_target)].sum(axis=1)
    unions = POPCOUNT_U8[np.bitwise_or(packed_masks, packed_target)].sum(axis=1)
    return intersections / np.maximum(unions, 1)


def composite_to_box(background_bgr, sprite_bgra, box):
    """External boxFinder contract: full, unclipped (x1,y1,x2,y2), exclusive end.

    The whole alpha crop is scaled once, then clipped by the image viewport.
    Return actual compositing alpha, never an RGB-difference approximation.
    """
    if len(box) != 4 or not np.all(np.isfinite(box)):
        raise ValueError("Box must contain four finite coordinates")
    x1, y1, x2, y2 = np.rint(box).astype(int)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box must have positive width and height")
    ys, xs = np.where(sprite_bgra[:, :, 3] > 10)
    if not len(xs):
        raise ValueError("Empty sprite alpha")
    crop = sprite_bgra[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = background_bgr.shape[:2]
    output = background_bgr.copy()
    alpha_canvas = np.zeros((h, w), np.float32)
    left, top, right, bottom = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if right <= left or bottom <= top:
        return output, alpha_canvas
    # Sample only visible destination pixels, avoiding giant offscreen allocations.
    scale_x, scale_y = crop.shape[1]/(x2-x1), crop.shape[0]/(y2-y1)
    map_x, map_y = np.meshgrid((np.arange(left, right)-x1+.5)*scale_x-.5,
                              (np.arange(top, bottom)-y1+.5)*scale_y-.5)
    alpha = crop[:, :, 3:4].astype(np.float32)/255
    premultiplied = np.concatenate((crop[:, :, :3].astype(np.float32)*alpha, alpha), axis=2)
    sampled = cv2.remap(premultiplied, map_x.astype(np.float32), map_y.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    visible_alpha = sampled[:, :, 3:4]
    output[top:bottom, left:right] = np.rint(np.clip(
        sampled[:, :, :3] + output[top:bottom, left:right]*(1-visible_alpha), 0, 255)).astype(np.uint8)
    alpha_canvas[top:bottom, left:right] = visible_alpha[:, :, 0]
    return output, alpha_canvas


class Selector:
    def __init__(self, bank, artifacts, size=128):
        self.bank, self.artifacts = Path(bank), Path(artifacts)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.size = size
        with (self.bank / "view_matrix.csv").open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.metadata = json.loads((self.bank / "asset_metadata.json").read_text())
        self.keys = np.array([[float(r[k]) for k in
                              ("angle_deg", "distance_m", "elevation_deg")]
                             for r in self.rows])
        # Include PNG identity as well as calibration in every artifact cache key.
        digest = hashlib.sha256((self.bank / "view_matrix.csv").read_bytes())
        digest.update((self.bank / "asset_metadata.json").read_bytes())
        for row in self.rows:
            stat = self.path(row).stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        self.fingerprint = digest.hexdigest()
        cache = self.artifacts / f"masks_{size}_{self.fingerprint[:12]}.npz"
        if cache.exists():
            self.masks = np.load(cache)["masks"]
        else:
            self.masks = np.stack([canonical(self.alpha(i), size)
                                   for i in range(len(self.rows))])
            np.savez_compressed(cache, masks=self.masks)
        self.packed_masks = np.packbits(self.masks.reshape((len(self.masks), -1)), axis=1)
        bbox = self.metadata["physical_bbox"]
        self.center = np.array([bbox[f"local_center_{a}_m"] for a in "xyz"])
        self.extents = np.array([bbox[f"extent_{a}_m"] for a in "xyz"])

    def path(self, row):
        return self.bank / row["rgba_relpath"]

    def alpha(self, index):
        image = cv2.imread(str(self.path(self.rows[index])), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[2] != 4:
            raise ValueError(f"Invalid RGBA sprite: {self.path(self.rows[index])}")
        return image[:, :, 3] > 10

    def build_hull(self, spacing=0.035, angle_step=10, tolerance_px=1):
        capture_distance = float(np.min(self.keys[:, 1]))
        config = {"version": 2, "bank": self.fingerprint, "spacing": spacing,
                  "angle_step": angle_step, "tolerance_px": tolerance_px,
                  "capture_distance_m": capture_distance}
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        path = self.artifacts / f"hull_{key}.npz"
        if path.exists():
            data = np.load(path)
            self.faces = data["faces"]
            self.hull_config = json.loads(str(data["config"]))
            return
        axes = [np.arange(c-e, c+e+spacing/2, spacing)
                for c, e in zip(self.center, self.extents)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        points = grid.reshape(-1, 3)
        occupied = np.ones(len(points), dtype=bool)
        capture_count = 0
        for index, row in enumerate(self.rows):
            if (not np.isclose(float(row["distance_m"]), capture_distance)
                    or int(float(row["angle_deg"])) % angle_step):
                continue
            # Clipped source silhouettes cannot constrain the entire volume.
            if row.get("projected_bbox_fully_inside_capture", "True").lower() != "true":
                continue
            alpha = self.alpha(index).astype(np.uint8)
            if tolerance_px:
                alpha = cv2.dilate(alpha, np.ones((2*tolerance_px+1,)*2, np.uint8))
            active = np.flatnonzero(occupied)
            matrix = relative_matrix(capture_transform(row, "actor"),
                                     capture_transform(row, "camera"))
            uv, depth = project(points[active], matrix, *[float(row[k]) for k in
                                ("camera_fx_px", "camera_fy_px", "camera_cx_px", "camera_cy_px")])
            uv -= [float(row["crop_x1_px"]), float(row["crop_y1_px"])]
            pixels = np.rint(np.nan_to_num(uv, nan=-1)).astype(int)
            valid = ((depth > 0.01) & (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0)
                     & (pixels[:, 0] < alpha.shape[1]) & (pixels[:, 1] < alpha.shape[0]))
            inside = np.zeros(len(active), dtype=bool)
            inside[valid] = alpha[pixels[valid, 1], pixels[valid, 0]] > 0
            occupied[active[~inside]] = False
            capture_count += 1
            if capture_count % 24 == 0:
                print(f"hull: {capture_count} captures, {occupied.sum()} voxels", flush=True)
        volume = occupied.reshape(grid.shape[:3])
        if not volume.any():
            raise RuntimeError("Hull empty; check capture calibration")
        # Surface voxel faces provide a watertight rasterizable proxy without a mesh dependency.
        faces = []
        padded = np.pad(volume, 1)
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            for sign in (-1, 1):
                slices = [slice(1, -1)] * 3
                slices[axis] = slice(0, -2) if sign == -1 else slice(2, None)
                exposed = volume & ~padded[tuple(slices)]
                centers = grid[exposed]
                offsets = np.zeros((4, 3))
                offsets[:, axis] = sign * spacing / 2
                offsets[:, other] = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * spacing / 2
                faces.append(centers[:, None, :] + offsets)
        self.faces = np.concatenate(faces).astype(np.float32)
        config.update(captures=capture_count, voxels=int(volume.sum()), faces=len(self.faces))
        self.hull_config = config
        np.savez_compressed(path, faces=self.faces, config=json.dumps(config))
        print(f"hull complete: {config}", flush=True)

    def predicted_mask(self, actor, camera, width, height, fov):
        fx = width / (2*np.tan(np.deg2rad(fov)/2))
        uv, depth = project(self.faces.reshape(-1, 3), relative_matrix(actor, camera),
                            fx, fx, width/2, height/2)
        if np.any(depth <= 0.01):
            raise ValueError("Hull intersects near plane; requires external clipping contract")
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        if np.any(hi-lo < 1e-6):
            raise ValueError("Degenerate projected hull")
        normalized = (uv-lo) / (hi-lo) * (self.size-1)
        polygons = np.rint(normalized).astype(np.int32).reshape(-1, 4, 2)
        mask = np.zeros((self.size, self.size), np.uint8)
        for polygon in polygons:
            cv2.fillConvexPoly(mask, polygon, 1)
        return mask.astype(bool), [*lo.tolist(), *hi.tolist()]

    def query(self, actor, camera):
        camera_local = (np.asarray(actor.get_inverse_matrix()) @
                        np.array([camera.location.x, camera.location.y, camera.location.z, 1]))[:3]
        ray = camera_local-self.center
        return np.array([np.rad2deg(np.arctan2(ray[1], ray[0])) % 360,
                         np.linalg.norm(ray),
                         np.rad2deg(np.arctan2(ray[2], np.hypot(ray[0], ray[1])))])

    def select(self, actor, camera, width, height, fov):
        """Inputs are poses and intrinsics only. No target mask or target box."""
        predicted, box = self.predicted_mask(actor, camera, width, height, fov)
        query = self.query(actor, camera)
        angle_delta = np.abs((self.keys[:, 0]-query[0]+180) % 360-180)
        # Silhouettes alone confuse opposite sides; enforce physically compatible appearance.
        eligible_indices = np.flatnonzero(angle_delta <= 35)
        scores = packed_iou_scores(self.packed_masks[eligible_indices], predicted)
        ranked = scores - 0.002*(angle_delta[eligible_indices]/35)**2
        local_selected = int(np.argmax(ranked))
        selected = int(eligible_indices[local_selected])
        selected_score = float(scores[local_selected])
        baseline = int(np.argmin(angle_delta**2 + (self.keys[:, 2]-query[2])**2
                                + (self.keys[:, 1]-query[1])**2))
        return {"index": selected, "sprite_path": str(self.path(self.rows[selected])),
                "key": self.keys[selected].tolist(), "query": query.tolist(),
                "predicted_iou": selected_score, "baseline_index": baseline,
                "predicted_box": box, "predicted_mask": predicted}
