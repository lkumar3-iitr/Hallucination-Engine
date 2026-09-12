"""Ray intersections with the cached bank-only visual hull, on CPU or CUDA."""
from __future__ import annotations

import numpy as np
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as functional
from scipy.ndimage import binary_fill_holes


class HullRays:
    @classmethod
    def from_bounds(cls, center, radius, spacing=.01, device=None):
        """Conservative box prior for carving, including limbs beyond a capsule box."""
        if not np.isfinite(radius) or not np.isfinite(spacing) or radius <= 0 or spacing <= 0:
            raise ValueError("Hull radius and spacing must be finite and positive")
        result = cls.__new__(cls)
        result.spacing = float(spacing)
        result.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        result.low = np.asarray(center, dtype=float)-radius
        shape = np.ceil(2*radius/spacing).astype(int)+1
        result.high = result.low+(shape-1)*spacing
        result.volume = torch.ones((1, 1, int(shape), int(shape), int(shape)),
                                   dtype=torch.float32, device=result.device)
        result.bounds_low = torch.as_tensor(result.low, device=result.device, dtype=torch.float32)
        result.bounds_high = torch.as_tensor(result.high, device=result.device, dtype=torch.float32)
        return result

    def __init__(self, faces, spacing=.035, device=None):
        self.spacing = float(spacing)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.low = np.min(faces, axis=(0, 1)).astype(np.float64)-spacing*2
        self.high = np.max(faces, axis=(0, 1)).astype(np.float64)+spacing*2
        shape = np.ceil((self.high-self.low)/spacing).astype(int)+1
        shell = np.zeros(tuple(shape), dtype=bool)
        # Populate both sides of each surface face; this bounds reconstruction
        # uncertainty to one voxel instead of silently dropping a face orientation.
        centers = faces.mean(axis=1)
        for dx in (-.3, .3):
            for dy in (-.3, .3):
                for dz in (-.3, .3):
                    coordinates = np.rint((centers-self.low)/spacing+[dx, dy, dz]).astype(int)
                    shell[tuple(coordinates.T)] = True
        occupied = binary_fill_holes(shell)
        self.volume = torch.as_tensor(occupied.transpose(2, 1, 0).copy(), dtype=torch.float32,
                                      device=self.device)[None, None]
        self.high = self.low+(shape-1)*spacing
        self.bounds_low = torch.as_tensor(self.low, device=self.device, dtype=torch.float32)
        self.bounds_high = torch.as_tensor(self.high, device=self.device, dtype=torch.float32)

    @torch.inference_mode()
    def refine(self, bank, cache):
        """Carve the conservative proxy with the complete calibrated close masks."""
        self._intersection_cache = None
        digest = hashlib.sha256(bank.path.read_bytes())
        digest.update(np.asarray([self.low, self.high], dtype=np.float64).tobytes())
        digest.update(str(self.spacing).encode())
        digest.update(self.volume.cpu().numpy().tobytes())
        for row in bank.rows:
            stat = (bank.path.parent/row["rgba_relpath"]).stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache/f"refined_hull_v1_{digest.hexdigest()[:16]}.npz"
        if path.exists():
            occupied = np.load(path)["occupied"]
            self.volume = torch.as_tensor(occupied, device=self.device, dtype=torch.float32)[None, None]
            return
        indices = torch.nonzero(self.volume[0, 0] > .5, as_tuple=False)
        keep = torch.ones(len(indices), dtype=torch.bool, device=self.device)
        points = indices.flip(dims=(1,)).to(torch.float32)*self.spacing+self.bounds_low
        for i, row in enumerate(bank.rows):
            active = torch.nonzero(keep, as_tuple=False).flatten()
            if not len(active):
                raise ValueError("Close-bank refinement emptied the hull")
            matrix = torch.as_tensor(bank.matrices[i], device=self.device, dtype=torch.float32)
            p = points[active] @ matrix[:3, :3].T+matrix[:3, 3]
            depth = p[:, 0]
            uv = torch.stack((float(row["camera_cx_px"])+float(row["camera_fx_px"])*p[:, 1]/depth-float(row["crop_x1_px"]),
                              float(row["camera_cy_px"])-float(row["camera_fy_px"])*p[:, 2]/depth-float(row["crop_y1_px"])), dim=1)
            mask = bank.source(i)[:, :, 3].copy()
            h, w = mask.shape
            grid = 2*uv/torch.tensor([w-1, h-1], device=self.device)-1
            alpha = functional.grid_sample(torch.as_tensor(mask, device=self.device)[None, None],
                                           grid[None, None], align_corners=True)[0, 0, 0]
            keep[active] = (alpha > .15) & (depth > 0)
            if i % 25 == 0:
                print(f"close-hull refinement {i+1}/{len(bank.rows)} voxels={int(keep.sum())}", flush=True)
        volume = torch.zeros_like(self.volume[0, 0])
        selected = indices[keep]
        volume[selected[:, 0], selected[:, 1], selected[:, 2]] = 1
        np.savez_compressed(path, occupied=volume.cpu().numpy().astype(bool))
        self.volume = volume[None, None]

    def intersect(self, origin, directions):
        """Reuse exact rays only; retain at most one camera's intersection."""
        origin, directions = np.asarray(origin), np.asarray(directions)
        # Inference tensors have no mutation counter; refine explicitly clears cache.
        version = None if self.volume.is_inference() else self.volume._version
        signature = (id(self.volume), version, self.spacing,
                     tuple(self.low), tuple(self.high))
        cached = getattr(self, '_intersection_cache', None)
        if (cached is not None and cached[0] == signature
                and np.array_equal(origin, cached[1])
                and np.array_equal(directions, cached[2])):
            return cached[3]
        result = self._intersect_uncached(origin, directions)
        for array in result:
            array.setflags(write=False)
        self._intersection_cache = (signature, origin.copy(), directions.copy(), result)
        return result

    @torch.inference_mode()
    def _intersect_uncached(self, origin, directions):
        shape = directions.shape[:2]
        ray = torch.as_tensor(directions.reshape(-1, 3), device=self.device, dtype=torch.float32)
        start = torch.as_tensor(origin, device=self.device, dtype=torch.float32)
        safe = torch.where(ray.abs() > 1e-9, ray, torch.full_like(ray, 1e-9))
        t0, t1 = (self.bounds_low-start)/safe, (self.bounds_high-start)/safe
        near = torch.maximum(torch.minimum(t0, t1).amax(dim=1), torch.zeros(len(ray), device=self.device))
        far = torch.maximum(t0, t1).amin(dim=1)
        active = torch.nonzero(far > near, as_tuple=False).flatten()
        points = torch.zeros_like(ray)
        hit_mask = torch.zeros(len(ray), dtype=torch.bool, device=self.device)
        for ids in active.split(2048):
            if not len(ids):
                continue
            d = ray[ids]
            length = torch.linalg.vector_norm(d, dim=1)
            step = self.spacing*.5/length
            count = int(torch.ceil(((far[ids]-near[ids])/step).max()).item())+1
            times = near[ids, None] + torch.arange(count, device=self.device)[None, :]*step[:, None]
            sample_points = start+d[:, None, :]*times[:, :, None]
            grid = 2*(sample_points-self.bounds_low)/(self.bounds_high-self.bounds_low)-1
            values = functional.grid_sample(self.volume, grid[None, None], mode="bilinear",
                                             padding_mode="zeros", align_corners=True)[0, 0, 0]
            inside = (values >= .5) & (times <= far[ids, None])
            hit = inside.any(dim=1)
            first = inside.to(torch.int32).argmax(dim=1)
            previous = torch.clamp(first-1, min=0)
            batch = torch.arange(len(ids), device=self.device)
            before, after = values[batch, previous], values[batch, first]
            fraction = torch.clamp((.5-before)/torch.clamp(after-before, min=1e-6), 0, 1)
            depth = times[batch, previous] + fraction*step
            points[ids] = start+d*depth[:, None]
            hit_mask[ids] = hit
        return points.cpu().numpy().reshape(*shape, 3), hit_mask.cpu().numpy().reshape(shape)
